import logging
import time
import numpy as np
import os
from collections import namedtuple
from enum import IntEnum
from cspy.chem.multipole import DistributedMultipoles
from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from cspy.minimize import single_point_evaluation
from cspy.sample.mc.mc_change import MC
from cspy.cspympi.threshold_tasks import (
    PerturbedStructure,
    ThresholdTaskTag,
    ThresholdWorker,
)

LOG = logging.getLogger(__name__)

ML_ThresholdStructure = namedtuple(
    "ML_ThresholdStructure",
    "name id spacegroup trial_number minimization_step mc_step unique_index "
    "energy density file_content initial_res xrd time accept metadata",
)


def read_precalculated_vasp(crystal, location='completed_calcs/'):
    from cspy.formats.vasp_output import VaspOutput

    if not isinstance(crystal, Crystal):
        LOG.info(f'{crystal} is not a Crystal object.')
        return None

    name = crystal.titl
    try:
        with open(f'{location}/{name}/OUTCAR', 'r') as f:
            output_contents = f.read()
    except:
        LOG.info(f'{location}/{name}/OUTCAR cannot be read.')
        return None

    vasp_output = VaspOutput(output_contents)
    if not vasp_output._valid("single_point"):
        LOG.debug(f"Invalid calculation in {location}/{name}")
        LOG.debug("Output contents:\n%s", output_contents)
        return None

    else:
        LOG.info(f'Skipping energy evaluation by VASP and reading the outputs from {location}/{name}')

        vasp_output_properties = vasp_output._parse_file()
        new_crystal = Crystal.from_CONTCAR(f'{location}/{name}/CONTCAR')
        new_crystal.properties["energy_eV"] = float(
            vasp_output_properties['initial_energy']
        )
        new_crystal.properties["energy"] = (float(
            vasp_output_properties['initial_energy']) * 96.48530749925793) / len(
            crystal.unit_cell_molecules()
        )
        new_crystal.properties["atom_forces"] = vasp_output._parse_forces()

    return new_crystal


class ThresholdTaskTag(IntEnum):
    """
    Task tags to identify what job to run and how to process the results
    """
    INI = 1
    PER_EN = 2
    OPT = 3
    REF_DATA = 4

class MLThresholdWorker(ThresholdWorker):
    """
    Worker class for the ML_threshold_manager. Performs the monte carlo step and 
    evaluates the energy of the resulting structures using the single point method and 
    if required the reference method. 
    """
    def __init__(self, data):
        self.data = data
        self.task_name = "Thre"
        
        self.energy_methods = {
            'dmacrys': self.energy_evaluation_dmacrys,
            'mlp': self.energy_evaluation_MLP,
            'vasp': self.energy_evaluation_vasp,
            'cNNP': self.energy_evaluation_committee_NNP
        }
        
        # self.ref_method_units = {
        #     'vasp': 'energy_eV'
        # }
        
    def energy_evaluation_dmacrys(self, structure, **kwargs):
        """
        evaluate single point energy by dmacrys
        """
        #return super().energy_evaluation(structure)
        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]
    
        name = structure.name
        sg = structure.spacegroup
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        bondlength_cutoffs = self.data["bondlength_cutoffs"]
        u_index = float("nan")
    
        electrostatics = DistributedMultipoles.from_dma_string(self.data["electrostatics"])
    
        time1 = time.time()
        structure = Crystal.from_shelx_string(structure.file_content)
        energy, density = single_point_evaluation(
            structure,
            electrostatics,
            self.data["axis"],
            bondlength_cutoffs=bondlength_cutoffs,
            potential=self.data["minimization"]["neighcrys"]["potential"],
            name=name,
            **self.data["minimization"]["dmacrys"],
        )
        structure_id = CspDatabaseId.from_components(
            name, self.task_name, sg, trial_number, 0, mc_step
        )
    
        mtime = time.time() - time1
        res = structure.to_shelx_string(titl=f"{structure_id} {energy} {density}")
        xrd = None
    
        return ML_ThresholdStructure(
            name=name,
            id=structure_id,
            spacegroup=sg,
            trial_number=trial_number,
            minimization_step=0,
            mc_step=mc_step,
            unique_index=u_index,
            energy=energy,
            density=density,
            file_content=res,
            initial_res=ini_res,
            xrd=xrd,
            time=mtime,
            accept=False,
            metadata={},
        )
    
    def energy_evaluation_dftb(self, structure, **kwargs):
        """
        evaluate single point energy by DFTB
        """
        raise NotImplementedError('Evaluating energy by DFTB+ not currently possible')

    def energy_evaluation_MLP(self, structure, **kwargs):
        """
        evaluate single point energy by machine learned potential
        """
        from cspy.ml.nnp.executable.nnp_singlepoint import n2p2_predict
        
        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]
    
        name = structure.name
        sg = structure.spacegroup
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        u_index = float("nan")
        
        structure = Crystal.from_shelx_string(structure.file_content)
        
        try:
            new_structure = n2p2_predict(structure, **kwargs)
            energy = new_structure.properties['energy']
            structure.properties.update(new_structure.properties)
        except:
            LOG.info(f'NNP energy evaluation failed for step {mc_step}')
            energy = None
        
        density = structure.density
        structure_id = CspDatabaseId.from_components(
            name, "thre", sg, trial_number, 0, mc_step
        )
    
        mtime = time.time() - t1
        res = structure.to_shelx_string(titl=f"{structure_id} {energy} {density}") # if changed atom ordering, maybe not needed here
        xrd = None
        metadata = {
            'raw_energy': structure.properties.get('raw_energy', float("nan"))
        }
        if kwargs.get('save_forces', False):
            metadata['forces'] = structure.properties.get("atom_forces", [])
        if kwargs.get('reference', False):
            metadata['ref_energy'] = metadata['raw_energy']
    
        return ML_ThresholdStructure(
            name=name,
            id=structure_id,
            spacegroup=sg,
            trial_number=trial_number,
            minimization_step=0,
            mc_step=mc_step,
            unique_index=u_index,
            energy=energy,
            density=density,
            file_content=res,
            initial_res=ini_res,
            xrd=xrd,
            time=mtime,
            accept=False,
            metadata=metadata,
        )

    def energy_evaluation_committee_NNP(self, structure, **kwargs):
        """
        evaluate single point energy by committee machine learned potential model
        """
        from cspy.ml.nnp.executable.nnp_singlepoint import n2p2_committee_predict
        
        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]
    
        name = structure.name
        sg = structure.spacegroup
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        u_index = float("nan")
        
        structure = Crystal.from_shelx_string(structure.file_content)
        
        try:
            new_structure = n2p2_committee_predict(structure, silent=True, **kwargs)
            energy = new_structure.properties['energy']
            structure.properties.update(new_structure.properties)
        except Exception as e:
            LOG.info(f'cNNP energy evaluation failed for step {mc_step}: {e}')
            energy = None
        
        density = structure.density
        structure_id = CspDatabaseId.from_components(
            name, "thre", sg, trial_number, 0, mc_step
        )
    
        mtime = time.time() - t1
        res = structure.to_shelx_string(titl=f"{structure_id} {energy} {density}") # if changed atom ordering, maybe not needed here
        xrd = None
        metadata = {
            'raw_energy': structure.properties.get('raw_energy', None),
            'energy_std_dev': structure.properties.get('energy_std_dev', None),
            'raw_energy_std_dev': structure.properties.get('raw_energy_std_dev', None)
        }
        if kwargs.get('save_forces', False):
            metadata['forces'] = structure.properties.get("atom_forces", [])
            metadata['forces_std_dev'] = structure.properties.get(
                "forces_std_dev", []
            )
        if kwargs.get('reference', False):
            metadata['ref_energy'] = metadata['raw_energy']
    
        return ML_ThresholdStructure(
            name=name,
            id=structure_id,
            spacegroup=sg,
            trial_number=trial_number,
            minimization_step=0,
            mc_step=mc_step,
            unique_index=u_index,
            energy=energy,
            density=density,
            file_content=res,
            initial_res=ini_res,
            xrd=xrd,
            time=mtime,
            accept=False,
            metadata=metadata,
        )

    def energy_evaluation_vasp(self, structure, **kwargs):
        """
        evaluate single point energy by VASP
        """
        from cspy.minimize.vasp_minimizer import VaspMinimizer
        from cspy.configuration import CspyConfiguration
        config = CspyConfiguration()

        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]
    
        name = structure.name
        sg = structure.spacegroup
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        u_index = float("nan")

        structure = Crystal.from_shelx_string(structure.file_content)
        structure = structure.standardized()
        # The line below added to take care of those cases where n_atoms in 'structure' is different before and
        # After VaspMinimizer call.
        structure = structure.as_P1()  # This won't change anything, as VaspMinimizer uses c.as_P1() anyway.
        structure.titl = name
        new_structure_ordering = []
        new_structure = None

        completed_calcs_folder = config.get("ml_threshold.completed_calcs_folder", 'completed_calcs')
        additional_completed_calcs_folders = config.get("ml_threshold.additional_completed_calcs_folders",
                                                       None)
        # Check to see if calculation already exists in completed calcs folder
        if os.path.exists(f'{completed_calcs_folder}/{name}'):
            LOG.info(f'Calculations were found in {completed_calcs_folder}/{name}.')
            new_structure = read_precalculated_vasp(structure, location=completed_calcs_folder)

        # The following lines are for passing other completed calcs folders that outputs can be read from.
        elif additional_completed_calcs_folders is not None:
            if not isinstance(additional_completed_calcs_folders, list):
                additional_completed_calcs_folders = [additional_completed_calcs_folders]
            for additional_completed_calcs_folder in additional_completed_calcs_folders:
                if os.path.exists(f'{additional_completed_calcs_folder}/{name}'):
                    LOG.info(f'Calculations were found in {additional_completed_calcs_folder}/{name}.')
                    new_structure = read_precalculated_vasp(structure, location=additional_completed_calcs_folder)
                if new_structure is not None:
                    break

        if not new_structure:
            vasp_keywords = config.get("vasp")
            vasp_keywords["incar_settings"]['NSW'] = "0"  # single point
            vasp_keywords["incar_settings"]['EDIFF'] = len(structure.unit_cell_atoms()['element']) * 1e-7
    
            LOG.debug("vasp_keywords:\n%s", vasp_keywords)

            try:
                minimizer = VaspMinimizer(**vasp_keywords, **kwargs)
                new_structure = minimizer.minimize(structure, completed_calcs_folder=completed_calcs_folder)
            except Exception as e:
                LOG.info('DFT single point failed: {}'.format(e))
                energy = None
                density = None
        structure_id = CspDatabaseId.from_components(
            name, self.task_name, sg, trial_number, 0, mc_step
        )
        if new_structure is None:
            res = structure.to_shelx_string(titl=f"{structure_id} None None")
            return ML_ThresholdStructure(
                name=name,
                id=structure_id,
                spacegroup=sg,
                trial_number=trial_number,
                minimization_step=0,
                mc_step=mc_step,
                unique_index=u_index,
                energy=None,
                density=None,
                file_content=res,
                initial_res=ini_res,
                xrd=None,
                time=0,
                accept=False,
                metadata={},
            )
        energy = new_structure.properties["energy"]
        density = new_structure.density
        structure.properties.update(new_structure.properties)
        new_structure_ordering = [ np.where(new_structure.site_labels == x)[0][0] for x in structure.site_labels ]
        res = structure.to_shelx_string(titl=f"{structure_id} {energy} {density}")
    
        metadata = {
            'energy_eV': structure.properties.get("energy_eV", float("nan"))
        }
        if kwargs.get('save_forces', False):
            if len(structure.properties.get("atom_forces", [])) == len(new_structure_ordering):
                metadata['forces'] = [ structure.properties.get("atom_forces", [])[i] for i in new_structure_ordering ]
            else:
                LOG.warning(f'Inconsistency between size of positions and forces of {structure_id}:')
                LOG.warning(f'len(forces): {len(structure.properties.get("atom_forces", []))} != len(positions): {len(new_structure_ordering)}.')
                LOG.warning(f"metadata['forces'] won't be set.")
                
        if kwargs.get('reference', False):
            metadata['ref_energy'] = metadata['energy_eV']
        
        mtime = time.time() - t1
        xrd = None
    
        return ML_ThresholdStructure(
            name=name,
            id=structure_id,
            spacegroup=sg,
            trial_number=trial_number,
            minimization_step=0,
            mc_step=mc_step,
            unique_index=u_index,
            energy=energy,
            density=density,
            file_content=res,
            initial_res=ini_res,
            xrd=xrd,
            time=mtime,
            accept=False,
            metadata=metadata,
        )
    
    def perturb_and_evaluate(self, args):
        """
        perturbs structure and evaluates the single point energy
        """

        t1 = time.time()
        _, _, _, _, _, _, rng = args
        structure, move_data = self.perturb_structure(args, tol=1e-2)
        crystal = self.energy_methods[
            self.data["mc"]["singlepoint_method"]
        ](structure)
        crystal.metadata['move'] = move_data[0]
        crystal.metadata['move_rand'] = move_data[1]
        return ThresholdTaskTag.PER_EN, (crystal, rng), time.time() - t1

    def perturb_and_evaluate_for_training_NNP(self, args):
        """
        Generates training data for neural network potential by first evaluating 
        with NNP and then if errors large evaluating energy and forces with DFT
        """
        import random

        t1 = time.time()
        _, _, _, _, _, _, rng = args # is this line necessary
        structure, move_data = self.perturb_structure(args, tol=1e-2)
        crystal = self.energy_methods[self.data["mc"]["singlepoint_method"]](structure)
        crystal.metadata['move'] = move_data[0]
        crystal.metadata['move_rand'] = move_data[1]
        
        if crystal.energy is None:
            run_reference = True
        else:
            if self.data["mc"]["singlepoint_method"] == "cNNP": # could do dict like calcs
                # TODO: if initial energy is unconverged, always run_reference
                threshold = self.data["cNNP_uncertainty_threshold"]
                energy_std_dev = crystal.metadata['energy_std_dev']
                run_reference = True if energy_std_dev > threshold else False
                LOG.info(f"cNNP uncertainty = {energy_std_dev} (threshold = {threshold})")
            else:
                error = random.random()
                crystal.metadata['error'] = error
                run_reference = True if error > 0.5 else False
        
        if run_reference:
            # run reference method, store energy and forces
            ref_result = self.energy_methods[
                self.data["reference_method"]
            ](structure, save_forces=True, reference=True)
            
            return ThresholdTaskTag.REF_DATA, (crystal, ref_result, rng), time.time() - t1
        else:
            return ThresholdTaskTag.PER_EN, (crystal, rng), time.time() - t1

    def perturb_structure(self, args, tol=1e-3):
        sg, seed, mc_step, name, res_in, scale, rng = args
        mc_control = MC(
            self.data["mc"]["move"],
            self.data["mc"]["auto_prob"],
            self.data["mc"]["auto_cutoff"],
            all_s=self.data["mc"]["move_all"],
            sat_s=self.data["mc"]["sat_expand"],
            cut_off_scale=scale,
            rng=rng,
        )
        mc_control.update_sg(res_in)
        crys_tmp = Crystal.from_shelx_string(res_in) 
        old_nmols = len(crys_tmp.asym_mols())

        
        while True:
            res_new, move_data = mc_control.mc_move(res_in)
            crys_tmp = Crystal.from_shelx_string(res_new)
            molecules = crys_tmp.symmetry_unique_molecules()
            if sum(len(x) for x in molecules) != len(crys_tmp.asymmetric_unit):
                LOG.warning("number of atoms in unique molecules does not equal atoms in asymmetric unit. Trying new move")
                continue
            if old_nmols is not None and old_nmols != len(crys_tmp.asym_mols()):
                LOG.warning("number of molecules changed (%s -> %s). Trying new move",
                            old_nmols, len(crys_tmp.asym_mols())
                        )
                continue
                
            valid = True
            # can't check bonds if want to reorder to vasp ordering. (otherwise reorder at start/before)
            if "atoms" not in [ x[0] for x in self.data["mc"]["move"]]:
                for mol in molecules:
                    for bond in self.data["mc"]["bonds"]:
                        try:
                            if np.allclose(mol.bonds.todense(), bond, atol=tol):
                                break
                        except ValueError:
                            continue
                    else:
                        LOG.warning("bond check failed")
                        valid = False
            if valid:
                return PerturbedStructure(
                    name=name,
                    trial_number=seed,
                    spacegroup=sg,
                    mc_step=mc_step,
                    file_content=res_new,
                    molecule_id=None,
                    Zp="calculate"
                ), move_data