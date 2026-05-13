import json
import logging
import pickle
import time
from collections import deque

from cspy.cspympi.threshold_tasks import (
    PerturbedStructure,
    MC_MinimizedStructure,
)
from cspy.cspympi.ml_threshold_tasks import MLThresholdWorker, ThresholdTaskTag
from cspy.db import CspDataStore
from cspy.distributed.threshold_manager import Threshold, TrialState
from cspy.util.logging_config import DATEFMT, FORMATS
from cspy.util.path import Path
from cspy.util.time_estimate import strfdelta, timedelta
from mpi4py import MPI
from numpy.random import (
    PCG64DXSM,
    Generator,
    SeedSequence,
    default_rng,
)
from cspy import Crystal
from cspy.ml.descriptors import PowderPattern
from cspy.util.constants import (
    EV2KJ_PER_MOL,
    HA2KJ_PER_MOL,
)

LOG = logging.getLogger(__name__)

def get_min_id(en_id, final_opt_step):
    """
    convert structure id to a given opt stage
    """
    contents = en_id.split('-')
    if len(contents) == 8:
        name, task, sg, tn, tmp, ms, opt, _ = en_id.split('-')
        return '-'.join([name, task, sg, tn, tmp, ms, opt, str(final_opt_step)])
    elif len(contents) == 6:
        name, task, sg, tn, _, ms = en_id.split('-')
        return '-'.join([name, task, sg, tn, str(final_opt_step), ms])
    else:
        raise NotImplementedError("Incompalible structure id")

energy_to_kjmol = {
    'eV': lambda en: en * EV2KJ_PER_MOL,
    'Ha': lambda en: en * HA2KJ_PER_MOL,
    'kjmol': lambda en: en, # assuming cspy kjmol per molecule
}

class TrialState:
    """
    Maintains and updates the state of a monte carlo trajectory. Determines whether to
    accept or reject monte carlo steps and controls the lid energy.
    """
    def __init__(
        self, 
        id, 
        res_old, 
        initial_energy, 
        interval_para, 
        increase_para, 
        iterations=0, 
        lid_state=1,
        state="R",
        restart_para=None,
        trajectory=None,
        rng=None,
        **kwargs,
    ):
        self.id = id
        self._res_old = res_old
        self._initial_energy = initial_energy 
        self._lid_state = lid_state
        self.rng = rng
        self.mlp_en_converged = False # is the initial energy converged (for MLP)

        increase_energy = self._get_increase(increase_para, restart_para)
        if restart_para is not None:
            increase_energy = None

        self.interval_tool = self._get_interval(
            interval_para, initial_energy, increase_energy, restart_para, **kwargs
        )
        self._iterations = iterations
        self._state = state
        if trajectory is None:
            datas = self.interval_tool.data
            datas.insert(0, 0)
            self._trajectory = [datas]
        else:
            self._trajectory = trajectory
    
    def _get_interval(
        self, interval_para, initial_energy, increase_energy, restart_para, **kwargs
    ):
        """
        setups tool for controlling number of steps at each threshold energy
        """
        from cspy.sample.mc.thre_interval import Interval_Fixed
        interval_type = interval_para[0]

        if interval_type == "fixed":
            return Interval_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        else:
            raise Exception("No threshold interval type ", interval_type)
   
    def _get_increase(self, increase_para, restart_para, **kwargs):
        """
        setups tool for controlling threshold/lid energy
        """
        from cspy.sample.mc.thre_increase import Increase_Fixed, Increase_Specified
        increase_type = increase_para[0]

        if increase_type == "fixed":
            self.increase_tool = Increase_Fixed(increase_para, restart_para)
        elif increase_type == "specified":
            raise NotImplementedError("specified increase needs fixing, see generate_restart()")
            self.increase_tool = Increase_Specified(increase_para, restart_para)
        else:
            raise Exception("No threshold increase type ", increase_type)

        return self.increase_tool.increase_energy

    def _get_id(self, name, sg, trial_number):
        return "-".join([name, str(sg), str(trial_number)])

    def update_step(self, res_new, energy, rng):
        """
        Accept or reject step based on current threshold energy and update trajectory 
        iteration.
        """
        self._iterations += 1
        self.rng = rng
        
        valid, lift_lid = self.interval_tool.update_step(energy)

        if valid:
            self._res_old = res_new

        if lift_lid:
            self._lid_state += 1
            increase_energy = self.increase_tool.increase_energy
            datas = self.interval_tool.update_lid_energy(increase_energy)
            datas.insert(0, self._iterations)
            LOG.info("Increase lid energy %s (increase = %s; steps = %s)", 
                self._lid_state, increase_energy, self.interval_tool.interval
            )
            self._trajectory.append(datas)

        return valid

    @property
    def iterations(self):
        return self._iterations

    @property
    def res_old(self):
        return self._res_old

    @property
    def trial_state(self):
        return self._state

    @property
    def lid_state(self):
        return self._lid_state

    def finish_trial(self):
        self._state = "F"

    def check_duplicate(self, iterations):
        return iterations != (self._iterations + 1)or self._state == "F"

    def generate_restart(self):
        """
        data required to restart trajectory status 
        """
        meta = {
            'res_old':self._res_old,
            'initial_energy':self._initial_energy,
            'trajectory':self._trajectory,
            'lid_state':self._lid_state,
            'increase': self.increase_tool.increase_energy, # TODO: how do this if increase_energy is not fixed?
            'rng_state': pickle.dumps(self.rng).decode('latin1'),
            'restart_para':self.interval_tool.restart,
        }
        return self.id, self._iterations, self._state, meta

class ML_Threshold(Threshold):
    """
    Manager class for running a threshold trajectory serially while generating
    data for training a machine learned potential  
    """
    def __init__(
        self,
        name,
        spacegroups,
        res_files,
        worker_data,
        dbs_dir,
        status_file="status.txt",
        random_seed=None,
        **kwargs,
    ):
        mc_para = worker_data["mc"]
        self.start_time = time.time()
        self.name = name
        LOG.info('Started Threshold algorithm for target: "%s"', self.name)

        self.initial_crystals = [
            (path[:-4], Path(path).read_text()) for path in res_files
        ]
        self.num_trials = len(self.initial_crystals)

        self.interval_para = mc_para["interval_para"]
        self.increase_para = mc_para["increase_para"]
        self.minimize_s = mc_para["minimize_s"]
        self.min_energy = mc_para["min_energy"]
        self.trial_step = mc_para["trial_step"]
        self.dump_accept = mc_para["dump_accept"]
        self.move_scale = mc_para["move_scale"]
        self.distributed = mc_para["distributed"]
        self.singlepoint_method = mc_para["singlepoint_method"]
        self.reference_method = worker_data["reference_method"]
        self.conformational_energy = worker_data["conformational_energy"]
        self.energy_units = worker_data["nnp_energy_units"]

        # self.sleep_time = 1.0 if self.minimize_s else 0.1
        self.sleep_time = 0.2

        rng = default_rng(random_seed)
        ss = SeedSequence(int.from_bytes(rng.bytes(8), "big"))
        # ss = SeedSequence(getrandbits(128))
        self.trial_seed_states = {i: x for i, x in enumerate(ss.spawn(self.num_trials))}

        self.current_job = None
        self.target = self.num_trials * self.trial_step
        #self.unique_structures = {}
        self.unique_index = 0
        self.trial_states = {}
        self.finish_s = {}
        self.status_file = status_file
        self.initialize_spacegroups(spacegroups)
        self.successful_minimizations = {sg: deque() for sg in self.structures}
        self.ref_data = {sg: deque() for sg in self.structures}
        for sg in self.structures:
            self.trial_states[sg] = {}
            #self.unique_structures[sg] = {}  # UniqueStructures()
            self.finish_s[sg] = False

        self.status_file = status_file
        self.load_status_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())

        self.db_thread = None
        self.db_writer = None
        self.create_databases()
        
        self.ref_db_thread = None
        self.ref_db_writer = None
        self.create_ref_databases(dbs_dir)

        self.worker = MLThresholdWorker(worker_data)

    def create_databases(self):
        """
        initialize cspy database for storing trajectory data
        """
        from cspy.db.datastore_writer import Threshold_DatastoreWriter
        import os
        import threading

        LOG.info("Setting up trajectory database")
        for sg in self.structures:
            filename = f"{self.name}-{sg}.db"
            if not os.path.exists(filename):
                ds = CspDataStore.create_and_connect(filename)
                ds.disconnect()
            else:
                self.restart_from_database(sg, filename)
        self.db_writer = Threshold_DatastoreWriter(
            self.successful_minimizations,
            self.trial_states,
            filename_format=f"{self.name}-{{sg}}.db",
            interval=0.5,
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f"{self.name}-dbworker"
        )
        self.db_thread.start()
        
    def create_ref_databases(self, dir_path):
        """
        initialize cspy database for storing reference calculation data
        """
        from cspy.db.datastore_writer import Threshold_DatastoreWriter
        import os
        import threading
        from glob import glob
        filename_stem = f"{os.path.join(dir_path, self.name)}-ref"
        count = 1
        while glob(f"{filename_stem}*"): 
            # this means if restart will create new db, which is maybe undesirable
            # but will require somehow communicating the ref db names between the trajectories
            # or putting the name in the cspy.toml 
            # (could use res name but if have multiple trajs per res won't work)
            filename_stem = f"{os.path.join(dir_path, self.name)}-ref" + f"_{count}"
            count += 1

        filename_format = f"{filename_stem}-{{sg}}.db"

        LOG.info("Setting up ref database")
        for sg in self.structures:
            filename = filename_format.format(sg=sg)

            ds = CspDataStore.create_and_connect(filename)
            ds.disconnect()
            # don't need to restart as it just sets up the trial state and num structures
        self.ref_db_writer = Threshold_DatastoreWriter(
            self.ref_data,
            self.trial_states,
            filename_format=filename_format,
            interval=0.3,
        )
        self.ref_db_thread = threading.Thread(
            target=self.ref_db_writer.run, name=f"{self.name}-ref-dbworker"
        )
        self.ref_db_thread.start()

    def initial_singlepoint(self):
        """
        Run initial single point of starting structure for MC trajectory
        """
        for sg in self.structures:
            for i, structure in enumerate(self.initial_crystals):
                structure = PerturbedStructure(
                    name=structure[0],
                    trial_number=i,
                    spacegroup=sg,
                    mc_step=0,
                    file_content=structure[1],
                    molecule_id=None,
                    Zp="calculate"
                )
        if self.singlepoint_method == "dmacrys":
            LOG.info("running initial single point with DMACRYS")
            result = self.worker.energy_evaluation(structure)
        elif self.singlepoint_method == "vasp":
            LOG.info("running initial single point with vasp")
            result = self.worker.energy_evaluation_vasp(structure)
        elif self.singlepoint_method == "mlp":
            result = self.worker.energy_evaluation_MLP(structure)
        else:
            raise NotImplementedError("initial calculation type not supported")
        
        return (ThresholdTaskTag.INI, result, None)
     
    def initial_singlepoint_referenced(self):
        """
        run initial single point with both the standard single point method and 
        the reference method as defined in the toml file (ml_threshold.reference_method)
        """
        energy_methods = self.worker.energy_methods
        
        structure = self.initial_crystals[0]
        self.initial_res = self.initial_crystals[0]
        sg = list(self.structures.keys())[0]
        input_structure = PerturbedStructure(
            name=structure[0],
            trial_number=0,
            spacegroup=sg,
            mc_step=0,
            file_content=structure[1],
            molecule_id=None,
            Zp="calculate"
        )
        try:
            result = energy_methods[self.singlepoint_method](input_structure)
            ref_result = energy_methods[self.reference_method](
                input_structure, save_forces=True, reference=True
            )
        except KeyError as e:
            raise NotImplementedError(f"calculation type not supported: {e}")
            
        self.process_initial_singlepoint_referenced(result, ref_result)
        
    def process_initial_singlepoint_referenced(self, result, ref_result):
        """
        Process results of initial single points with standard method and reference 
        method and initiate trajectory
        """
        crystal = result
        sg = crystal.spacegroup
        trial_number = crystal.trial_number
        mc_step = crystal.mc_step
        name = crystal.name

        if crystal.energy is not None and ref_result.energy is not None:
            self.unique_index += 1
            crystal = crystal._replace(unique_index=self.unique_index)
            self.structures[sg]["ttime"] += crystal.time
            energy_kjmol = crystal.energy
            ref_energy_kjmol = ref_result.energy

            cspy_crystal = Crystal.from_shelx_string(crystal.file_content)
            nmols = len(cspy_crystal.unit_cell_molecules())
            pp = PowderPattern.from_cif_string(
                cspy_crystal.to_cif_string()
            )
            if pp is not None:
                xrd = pp.pattern

            

            # save energy in ref db as original
            ref_energy = ref_result.metadata.get('ref_energy', ref_result.energy)
            # subtract conformational energy if given
            if self.conformational_energy:
                ref_energy -= (self.conformational_energy * nmols)
                ref_energy_kjmol = energy_to_kjmol[self.energy_units](ref_energy) / nmols
            ref_result = ref_result._replace(
                energy=ref_energy
            )
            
            # if self.conformational_energy and crystal.metadata.get('raw_energy', False):
            #     energy = crystal.metadata['raw_energy'] #- (self.conformational_energy * nmols)
            LOG.info(
                "Processing referenced initial singlepoint sg %s trial_number %s mc_step %s energy %s ref energy %s",
                sg,
                trial_number,
                mc_step,
                energy_kjmol,
                ref_energy_kjmol
            )

            # set up trajectory/trial details
            trial_id = "-".join([name, str(sg), str(trial_number)])
            res_old = crystal.file_content
            if self.min_energy is None:
                initial_energy = ref_energy_kjmol
            else:
                initial_energy = self.min_energy

            seed_state = self.trial_seed_states.pop(trial_number)
            rng = Generator(PCG64DXSM(seed_state))

            # initialize trajectory data using reference energy
            trial_state = TrialState(
                trial_id,
                res_old,
                initial_energy,
                self.interval_para,
                self.increase_para,
                rng=rng,
            )
            self.trial_states[sg][trial_number] = trial_state
            self.structures[sg]["active_trial"] += 1

            # if using mlp check if energy is converged
            if (
                self.singlepoint_method == "mlp" and
                abs(crystal.energy - ref_result.energy) < 1.0 # kJ/mol
            ):
                # set as converged
                LOG.info('MLP initial energy is converged')
                self.trial_states[sg][trial_number].mlp_en_converged = True
            elif self.singlepoint_method == "cNNP":
                uncertainty = result.metadata["energy_std_dev"]
                threshold =  self.worker.data["cNNP_uncertainty_threshold"]
                LOG.info(f"cNNP uncertainty = {uncertainty} (threshold = {threshold})")
                check = uncertainty < threshold
                LOG.info(f"cNNP initial energy converged: {check}")
                self.trial_states[sg][trial_number].mlp_en_converged = check
            
            # Save structures to dbs
            if not self.trial_states[sg][trial_number].mlp_en_converged:
                self.ref_data[sg].append(ref_result)
                
            crystal = crystal._replace(energy=ref_energy_kjmol)
            self.successful_minimizations[sg].append(crystal)

            # add second crystal for minimized initial (assumes starting from minimized crystal)
            self.unique_index += 1
            final_step_id = get_min_id(crystal.id, 1)
            final_step_crystal = MC_MinimizedStructure(
                name=crystal.name,
                id=final_step_id,
                spacegroup=crystal.spacegroup,
                trial_number=crystal.trial_number,
                minimization_step=1, # assuming single stage NNP opt
                mc_step=crystal.mc_step,
                unique_index=self.unique_index,
                energy=crystal.energy,
                density=crystal.density,
                file_content=crystal.file_content,
                initial_res=crystal.initial_res,
                xrd=xrd,
                time=crystal.time,
                accept=False,
            )
            self.successful_minimizations[sg].append(final_step_crystal)


            self.current_job = (sg, trial_number, 1, name, res_old, 1.0, rng)
        else:
            LOG.info("initial singlepoint failed")
            self.current_job = None

    def process_initial_singlepoint(self, crystal):
        """
        Process initial single point of starting structure and initiate trajectory
        """
        # Get minimized structures
        # crystal, rng = result
        sg = crystal.spacegroup
        trial_number = crystal.trial_number
        mc_step = crystal.mc_step
        name = crystal.name

        if crystal.energy is not None:
            self.unique_index += 1
            crystal = crystal._replace(unique_index=self.unique_index)
            #self.structures[sg]["valid"] += 1
            self.structures[sg]["ttime"] += crystal.time
            energy = crystal.energy

            LOG.info(
                "Processing initial singlepoint sg %s trial_number %s mc_step %s energy %s",
                sg,
                trial_number,
                mc_step,
                energy,
            )

            trial_id = "-".join([name, str(sg), str(trial_number)])
            res_old = crystal.file_content
            if self.min_energy is None:
                initial_energy = crystal.energy
            else:
                initial_energy = self.min_energy

            seed_state = self.trial_seed_states.pop(trial_number)
            rng = Generator(PCG64DXSM(seed_state))

            #self.unique_structures[sg][trial_number] = UniqueStructures()
            # self.trial_states[sg][trial_number] = TrialState(
            trial_state = TrialState(
                trial_id,
                res_old,
                initial_energy,
                self.interval_para,
                self.increase_para,
                rng=rng,  # pickle.dumps(rng).decode('latin1'),
                #unique_structures=self.unique_structures[sg][trial_number],
            )
            self.trial_states[sg][trial_number] = trial_state
            self.structures[sg]["active_trial"] += 1

            cspy_crystal = Crystal.from_shelx_string(crystal.file_content)
            nmols = len(cspy_crystal.unit_cell_molecules())
            pp = PowderPattern.from_cif_string(
                cspy_crystal.to_cif_string()
            )
            if pp is not None:
                xrd = pp.pattern

            self.successful_minimizations[sg].append(crystal)
            
            #for step in range(1, self.final_opt_step):
            self.unique_index += 1
            final_step_id = get_min_id(crystal.id, self.final_opt_step)
            final_step_crystal = MC_MinimizedStructure(
                name=crystal.name,
                id=final_step_id,
                spacegroup=crystal.spacegroup,
                trial_number=crystal.trial_number,
                minimization_step=self.final_opt_step,
                mc_step=crystal.mc_step,
                unique_index=self.unique_index,
                energy=crystal.energy,
                density=crystal.density,
                file_content=crystal.file_content,
                initial_res=crystal.initial_res,
                xrd=xrd,
                time=crystal.time,
                accept=False,
            )
            self.successful_minimizations[sg].append(final_step_crystal)
            

            return (sg, trial_number, 1, name, res_old, 1.0, rng)

        LOG.info("initial singlepoint failed")
        return None
    
    def process_mc_step(self, result, ttime):
        """
        process mc step structure, determine whether to accept or reject step, and 
        submit next step
        """
        crystal, rng = result
        sg = crystal.spacegroup
        trial_number = crystal.trial_number
        mc_step = crystal.mc_step
        name = crystal.name

        if crystal.energy is not None:
            self.unique_index += 1
            crystal = crystal._replace(unique_index=self.unique_index)
            self.structures[sg]["perturb"] += 1
            self.structures[sg]["ttime"] += ttime
            energy = crystal.energy
            energy_kjmol = crystal.energy
            cspy_crystal = Crystal.from_shelx_string(crystal.file_content)
            nmols = len(cspy_crystal.unit_cell_molecules())

            if trial_number in self.trial_states[sg]:
                if self.trial_states[sg][trial_number].check_duplicate(mc_step):
                    LOG.warning(
                        "Duplicate result! sg %s, seed %s, mc_step %s",
                        sg,
                        trial_number,
                        mc_step,
                    )
                    return

            # TODO: make the initial energy check a separate function
            # check if using mlp and the initial energy is unconverged (check if running on the fly?)
            # maybe only do every n steps, i.e. not every step since weights won't update that fast
            if self.reference_method and not self.trial_states[sg][trial_number].mlp_en_converged:
                initial_structure = self.initial_res
                input_structure = PerturbedStructure(
                    name=initial_structure[0],
                    trial_number=trial_number,
                    spacegroup=sg,
                    mc_step=0,
                    file_content=initial_structure[1],
                    molecule_id=None,
                    Zp="calculate"
                )
                # recalculate initial energy of initial structure using MLP
                
                result = self.worker.energy_methods[self.singlepoint_method](
                    input_structure
                )
                new_energy = result.energy
                
                # if uncertainty is below cutoff set as converged
                # TODO: if not converged always do reference method
                if self.singlepoint_method == "cNNP":
                    uncertainty = result.metadata["energy_std_dev"]
                    threshold =  self.worker.data["cNNP_uncertainty_threshold"]
                    LOG.info(f"cNNP uncertainty = {uncertainty} (threshold = {threshold})")
                    check = uncertainty < threshold
                else:
                    check = abs(
                        new_energy - self.trial_states[sg][trial_number]._initial_energy
                    ) < 1.0
                
                LOG.info(f"MLP initial energy converged: {check}")
                self.trial_states[sg][trial_number].mlp_en_converged = check

            if self.trial_states[sg][trial_number].update_step(
                crystal.file_content,
                energy,
                rng,
            ):
                self.structures[sg]["accept"] += 1
                crystal = crystal._replace(accept=True)
                self.successful_minimizations[sg].append(crystal)
            else:
                self.structures[sg]["reject"] += 1
                if not self.dump_accept:
                    self.successful_minimizations[sg].append(crystal)

            LOG.info(
                "Processed per_en sg %s trial_number %s mc_step %s energy %s accept %s",
                sg,
                trial_number,
                mc_step,
                energy_kjmol,
                crystal.accept,
            )
            
            mc_step += 1
            if mc_step > self.trial_step:
                self.trial_states[sg][trial_number].finish_trial()
                self.structures[sg]["active_trial"] -= 1
                self.structures[sg]["finish_trial"] += 1
                return None

        res_in = self.trial_states[sg][trial_number].res_old
        lid_state = self.trial_states[sg][trial_number].lid_state
        scale = self.move_scale**lid_state
        LOG.debug(
            "Submit new perturb sg %s trial_number %s mc_step %s scale %s",
            sg,
            trial_number,
            mc_step,
            scale,
        )
        return (sg, trial_number, mc_step, name, res_in, scale, rng)
    
    def process_reference_data_mc_step(self, result, ttime):
        """
        process mc step structure using energy from reference method, 
        determine whether to accept or reject step, and submit next step. 
        Save reference data regardless of accept/reject.
        """
        crystal, ref_result, rng = result
        sg = crystal.spacegroup
        trial_number = crystal.trial_number
        mc_step = crystal.mc_step
        name = crystal.name

        if crystal.energy is not None and ref_result.energy is not None:
            
            self.unique_index += 1
            crystal = crystal._replace(unique_index=self.unique_index)
            ref_result = ref_result._replace(unique_index=self.unique_index)
            self.structures[sg]["perturb"] += 1
            self.structures[sg]["ttime"] += ttime
            energy_kjmol = crystal.energy
            ref_energy_kjmol = ref_result.energy
            
            cspy_crystal = Crystal.from_shelx_string(crystal.file_content)
            nmols = len(cspy_crystal.unit_cell_molecules())

            if trial_number in self.trial_states[sg]:
                if self.trial_states[sg][trial_number].check_duplicate(mc_step):
                    LOG.warning(
                        "Duplicate result! sg %s, seed %s, mc_step %s",
                        sg,
                        trial_number,
                        mc_step,
                    )
                    return

            ref_energy = ref_result.metadata.get('ref_energy', ref_result.energy)
            # subtract conformational energy if given
            if self.conformational_energy:
                ref_energy -= (self.conformational_energy * nmols)
                ref_energy_kjmol = energy_to_kjmol[self.energy_units](ref_energy) / nmols

            # add crystal to trajectory db with energy = ref energy
            crystal = crystal._replace(energy=ref_energy_kjmol)
            if self.trial_states[sg][trial_number].update_step(
                res_new=crystal.file_content,
                energy=ref_energy_kjmol,
                rng=rng,
            ):
                self.structures[sg]["accept"] += 1
                crystal = crystal._replace(accept=True)
                self.successful_minimizations[sg].append(crystal)
            else:
                self.structures[sg]["reject"] += 1
                if not self.dump_accept:
                    self.successful_minimizations[sg].append(crystal)
            
            # add ref result to ref db with (adjusted) raw energy
            ref_result = ref_result._replace(
                energy=ref_energy
            )
            self.ref_data[sg].append(ref_result)
            
            LOG.info(
                "Processed ref_result sg %s trial_number %s mc_step %s energy %s ref_energy %s accept %s",
                sg,
                trial_number,
                mc_step,
                energy_kjmol,
                ref_energy_kjmol,
                crystal.accept,
            )
            
            mc_step += 1
            if mc_step > self.trial_step:
                self.trial_states[sg][trial_number].finish_trial()
                self.structures[sg]["active_trial"] -= 1
                self.structures[sg]["finish_trial"] += 1
                return None

        res_in = self.trial_states[sg][trial_number].res_old
        lid_state = self.trial_states[sg][trial_number].lid_state
        scale = self.move_scale**lid_state
        LOG.debug(
            "Submit new perturb sg %s trial_number %s mc_step %s scale %s",
            sg,
            trial_number,
            mc_step,
            scale,
        )
        return (sg, trial_number, mc_step, name, res_in, scale, rng)

    def process_result(self, result):
        """
        Process results from worker according to task type
        """
        task_type, result, ttime = result
        
        if task_type == ThresholdTaskTag.INI:
            return self.process_initial_singlepoint(result)
        elif task_type == ThresholdTaskTag.PER_EN:
            return self.process_mc_step(result, ttime)
        elif task_type == ThresholdTaskTag.REF_DATA:
            return self.process_reference_data_mc_step(result, ttime)

    def estimated_time_remaining(self, spacegroups=None):
        eta = None
        if spacegroups is None:
            spacegroups = self.structures.keys()

        time_remaining = {}
        ncompleted = 0
        ntarget = 0
        total_time = 0.0
        nworkers = 1
        for sg in spacegroups:
            i = self.structures[sg]
            if i["perturb"] < i["target"]:
                ncompleted += i["perturb"]
                ntarget += i["target"]
                if i["perturb"] > 0:
                    time_remaining[sg] = (i["target"] / i["perturb"] - 1.0) * i["ttime"]
                    total_time += i["ttime"]

        if ncompleted == ntarget:
            eta = "complete"
        elif ncompleted > 0:
            total = int(((ntarget / ncompleted - 1.0) * total_time) / nworkers)
            time_fmt = "{d}d {h}h {m}m {s}s"
            eta = strfdelta(timedelta(seconds=total), time_fmt)
        heading = "ETA: "
        return f"{nworkers} MPI workers. {heading}{eta}"

    def run(self):
        """
        Run threshold trajectory serially after evaluating initial structure energy
        """
        if self.current_job is None and self.initial_crystals:
            self.initial_singlepoint_referenced()
            time.sleep(self.sleep_time)

        while self.current_job is not None:
            loop_start = time.time()
            result = self.worker.perturb_and_evaluate_for_training_NNP(self.current_job)
            self.current_job = self.process_result(result)
            LOG.debug("iteration took {}".format(time.time() - loop_start))
            time.sleep(self.sleep_time)
            self.update_status()
            
        self.shutdown()
        
        return

    def shutdown(self):
        """
        Finish writing to databases and close connections. Write summary.
        """
        LOG.info("Waiting to finish writing database...")
        if self.db_thread is not None:
            #time.sleep(2)
            self.db_writer.complete = True
            self.db_thread.join()
        if self.ref_db_thread is not None:
            self.ref_db_writer.complete=True
            self.ref_db_thread.join()
        LOG.info("Done writing databases")
        LOG.info("%d QR valid minimizations", self.valid_structure_count())
        LOG.info("%d QR invalid minimizations", self.invalid_structure_count())
        LOG.info("%d finished BH trials", self.finish_trial_count())
        LOG.info("%d unfinished BH trials", self.active_trial_count())

        LOG.info("Final structure table\n%s", self.structure_table_string())
        endtime = time.time()
        time_fmt = "{d}d {h}h {m}m {s}s"
        LOG.info("Threshold complete!")
        LOG.info(
            "Walltime: %s",
            strfdelta(timedelta(seconds=endtime - self.start_time), time_fmt),
        )
        ttime = self.total_cpu_time()
        mtime = self.total_minimization_time()
        LOG.info("Total CPU time: %s", strfdelta(timedelta(seconds=ttime), time_fmt))
        LOG.info(
            "Minimization CPU time: %s", strfdelta(timedelta(seconds=mtime), time_fmt)
        )

    def restart_from_database(self, sg, filename):
        """
        Restarts unfinished trajectory using restart data in cspy database trial table
        """
        ds = CspDataStore(filename)
        finish_trial = 0
        active_trial = 0
        total_steps = 0
        ntrial = ds.query("select max(trial_number) from trial_structure").fetchone()[0]
        max_min_step = ds.query(
            "select max(minimization_step) from trial_structure"
        ).fetchone()[0]
        if ntrial is not None:
            for trial_id, trial_number, iterations, state, metadata in ds.query(
                "select trial_id, trial_number, iterations, state, metadata from trial"
            ).fetchall():
                LOG.debug(
                    "SG(%d): getting trial %s %s state %s ",
                    sg,
                    trial_number,
                    iterations,
                    state,
                )

                if state == "F" and iterations < self.trial_step:
                    state = "R"

                meta = json.loads(metadata)
                lid_state = meta["lid_state"]
                res_old = meta["res_old"]
                rng = pickle.loads(
                    meta["rng_state"].encode("latin1")
                )  # required for pickling rng
                trial_state = TrialState(
                    id=trial_id,
                    res_old=res_old,
                    initial_energy=meta["initial_energy"],
                    interval_para=self.interval_para,
                    increase_para=self.increase_para,
                    iterations=iterations,
                    lid_state=lid_state,
                    state=state,
                    restart_para=meta["restart_para"],
                    trajectory=meta["trajectory"],
                    rng=rng,
                )
                self.trial_states[sg][trial_number] = trial_state

                total_steps += iterations

                if state == "R":
                    LOG.info(
                        "SG(%d): restarting unfinished trial %s %s",
                        sg,
                        trial_number,
                        iterations,
                    )
                    active_trial += 1
                    scale = self.move_scale**lid_state

                    name = trial_id.split('-')[0]

                    self.current_job = (
                        sg,
                        trial_number,
                        iterations + 1,
                        name,
                        res_old,
                        scale,
                        rng,
                    )
                elif state == "F":
                    finish_trial += 1

            naccept = ds.query(
                "select count(*) from trial_structure where minimization_step=0 and valid=True"
            ).fetchone()[0]
            if max_min_step > 0:
                nvalid = ds.query(
                    f"select count(*) from trial_structure where minimization_step={max_min_step}"
                ).fetchone()[0]
            else:
                nvalid = 0


            self.structures[sg]["perturb"] = total_steps
            self.structures[sg]["accept"] = naccept
            self.structures[sg]["reject"] = total_steps - naccept
            self.structures[sg]["valid"] = nvalid
            self.structures[sg]["active_trial"] = active_trial
            self.structures[sg]["finish_trial"] = finish_trial

        self.initial_res = self.initial_crystals[0]
        self.initial_crystals = []

        ds.disconnect()
