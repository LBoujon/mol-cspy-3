import logging
import time
from matplotlib import tri
import numpy as np
from collections import defaultdict, namedtuple, deque
from enum import IntEnum
from cspy.chem.multipole import DistributedMultipoles
from cspy.crystal import Crystal
from cspy.crystal.space_group import n_symops_from_SG
from cspy.db.key import CspDatabaseId
from cspy.minimize import CompositeMinimizer
from cspy.minimize import single_point_evaluation
from cspy.ml.descriptors import calculate_crystal_powder_pattern
from cspy.sample.mc.mc_change import MC
from .worker import Worker, WorkerCommTag
from cspy.crystal.util import find_formula_unit


LOG = logging.getLogger(__name__)


PerturbedStructure = namedtuple(
    "PerturbedStructure", "name spacegroup trial_number mc_step file_content molecule_id Zp"
)
MC_MinimizedStructure = namedtuple(
    "MC_MinimizedStructure",
    "name id spacegroup trial_number minimization_step mc_step unique_index "
    "energy density file_content initial_res xrd time accept Zp",
)

#prime_numbers_trucated = [3, 5, 7, 11, 13]
#prime_numbers_trucated = [13, 17, 19, 23,]
prime_numbers_trucated = [19, 23, 29, 31, 37, 41, 43, 47]


class ThresholdTaskTag(IntEnum):
    PER_EN = 1
    OPT = 2
    RUN_MC = 3
    MC_RESULTS = 9999

def parse_moves_list(move_list):
    move_dict = {}
    for i_move in move_list:
        move_type = str(i_move[0])
        move_prob = str(i_move[1])
        move_cutoff = str(i_move[2])
        if len(i_move) > 3:
            extra_params = list(i_move[3:])
        else:
            extra_params = []
        
        move_dict[move_type] = list(i_move[1:])
        
    return move_dict

def to_move_list(move_dict):
    move_list = []

    for move_type, move_params in move_dict.items():
        move_list.append(
            [move_type] + move_params
        )
    
    return move_list

def update_move_cutoffs(move_dict, move_type, energies, target=1.0, method='percent_diff'):

    old_cutoff = float(move_dict[move_type][1])

    average_energy = sum(energies[-200:]) / len(energies[-200:]) #len(energies)

    diff = (average_energy - target) / target

    if method == 'simple':
        #stepsize = 0.1 if len(energies) < 10 else 0.1 * (1/ (0.1* len(energies)))
        #stepsize = 0.05

        stepsize = 0.05 * old_cutoff
        
        if abs(diff) < 0.1:
            new_cutoff = old_cutoff
        elif diff < -0.1:
            new_cutoff = old_cutoff + stepsize
        else:
            new_cutoff = old_cutoff - stepsize

        # if abs(diff) < 0.01:
        #     new_cutoff = old_cutoff
        # else:
        #     new_cutoff = old_cutoff * (1 + stepsize) if diff < 0 \
        #             else old_cutoff * (1 - stepsize)
    elif method =='percent_diff':
        if diff < 0.0:
            diff = max(diff, -0.10)
        else:
            diff = min(diff, 0.10)
        
        stepsize = diff * old_cutoff
        new_cutoff = old_cutoff - stepsize
    else:
        raise NotImplementedError('method does not exist')
    
    LOG.info(f'{move_type} cutoff change from {old_cutoff} to {new_cutoff} ave abs energy change {average_energy} ')
    
    move_dict[move_type][1] = new_cutoff
    

class ThresholdWorker(Worker):
    def __init__(self, data):
        """The CSP worker class.

        Parameters
        ----------
        data: dict
            Dictionary containing all the data required to run the
            MC step and minimisation tasks.
        """
        super().__init__()
        self.data = data
        self.task_name = "Thre"

    def calculate(self, data):
        """Override of the calculate method for CSPy tasks.

        Parameters
        ----------
        data: tuple
            Tuple containing the task tag to be done and the args for
            that task.

        Returns
        -------
        tuple
            Tuple of the results which contains the task tag, results
            and time taken.
        """
        task, args = data
        results = None

        if task == ThresholdTaskTag.PER_EN:
            results = self.perturb_and_evaluate(args)
        elif task == ThresholdTaskTag.OPT:
            results = self.minimize_structure(args)
        elif task == ThresholdTaskTag.RUN_MC:
            results = self.run_trajectory(args)

        return results

    def minimize_structure(self, structure):
        """
        Energy Minimize structure using DMACRYs. For threshold MC simulations the 
        structures minimized are either initial structures (before the trajectory 
        starts) or accepted MC structures.

        Parameters
        ----------
        structure : PerturbedStructure:
            The input structure to energy minimize

        Returns
        -------
        int: ThresholdTaskTag.OPT
            Job type for processing
        tuple (bool, list[<MC_MinimizedStructure>])
            contains bool indicating if minimization is valid and a list of the final 
            crystal structures from each stage of the minimization protocol as 
            MCMinimizedStructure objects
        float: time
            time to complete minimization
        """
        from cspy.configuration import configure

        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]

        name = structure.name
        sg = structure.spacegroup
        Zp = structure.Zp
        if Zp == "calculate":
            c = Crystal.from_shelx_string(structure.file_content)
            mols = c.symmetry_unique_molecules()
            _, _, Zp = find_formula_unit(mols)
        crystal_info = {"spacegroup" : sg, "Zp" : Zp}
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        u_index = float("nan")

        charges = DistributedMultipoles.from_dma_string(self.data["charges"])
        multipoles = DistributedMultipoles.from_dma_string(self.data["multipoles"])
        axis = self.data["axis"]
        bondlength_cutoffs = self.data["bondlength_cutoffs"]
        minimization_settings = self.data["minimization"]
        check_spe = self.data["check_single_point_energy"]
        configure(minimization_settings)

        minimizer = CompositeMinimizer.from_defaults(
            charges=charges, axis=axis, multipoles=multipoles,
            bondlength_cutoffs=bondlength_cutoffs, crystal_info = crystal_info, Zp=Zp
        )

        structure = Crystal.from_shelx_string(structure.file_content)
        structure_id = CspDatabaseId.from_components(
            name, self.task_name, sg, trial_number, 0, mc_step
        )
        res = structure.to_shelx_string(titl=structure_id)
        crystals = [
            MC_MinimizedStructure(
                name=name,
                id=structure_id,
                spacegroup=sg,
                trial_number=trial_number,
                minimization_step=0,
                mc_step=mc_step,
                unique_index=u_index,
                energy=float("nan"),
                density=float("nan"),
                file_content=res,
                initial_res=ini_res,
                xrd=None,
                time=float("nan"),
                accept=False,
                Zp=Zp # this might need updating from initial value after minimization if not rigid
            )
        ]
        minimized_crystals = minimizer(structure)
        valid = False

        for minimization_step, crystal in enumerate(minimized_crystals, start=1):
            if isinstance(crystal, str):
                # the minimisation failed and the string is the error?
                LOG.debug(f"Failed minimisation (step {minimization_step}) with error: {crystal}")
                valid = crystal
                continue

            structure_id = CspDatabaseId.from_components(
                name, self.task_name, sg, trial_number, minimization_step, mc_step
            )
            energy = crystal.properties["lattice_energy"]
            density = crystal.properties["density"]
            mtime = crystal.properties["minimization_time"]
            res = crystal.to_shelx_string(titl=f"{structure_id} {energy} {density}")
            xrd = None

            if minimization_step == minimizer.step_count:
                check_through = True
                if check_spe:
                    energy_check, _ = single_point_evaluation(
                        crystal,
                        multipoles,
                        axis,
                        bondlength_cutoffs=bondlength_cutoffs,
                        potential=self.data["minimization"]["neighcrys"]["potential"],
                        name=name,
                        **self.data["minimization"]["dmacrys"],
                    )
                    if abs(energy - energy_check) > 10.0:
                        check_through = False
                        LOG.info(
                            "Different energy from single point evaluation id %s, "
                            "energy %s, energy_check %s",
                            structure_id, energy, energy_check
                        )
                if check_through:
                        valid = True
                        xrd = calculate_crystal_powder_pattern(
                            crystal, self.data["descriptors"].get("pxrd")
                        )

            crystals.append(
                MC_MinimizedStructure(
                    name=name,
                    id=structure_id,
                    spacegroup=sg,
                    trial_number=trial_number,
                    minimization_step=minimization_step,
                    mc_step=mc_step,
                    unique_index=u_index,
                    energy=energy,
                    density=density,
                    file_content=res,
                    initial_res=ini_res,
                    xrd=xrd,
                    time=mtime,
                    accept=False,
                    Zp=Zp # this might need updating from initial value after minimization if not rigid
                )
            )
        return ThresholdTaskTag.OPT, (valid, crystals), time.time() - t1
    

    def perturb_and_evaluate(self, args):
        """
        Perturb given structure using a randomly selected MC move and evaluate the 
        energy of the resulting structure

        Parameters
        ----------
        args : tuple
            the input for the MC step containing: 
                space group number
                trial/trajectory number
                MC step number, 
                name of simulation, 
                crystal structure as SHELX file contents, 
                scalar to multiple by of MC move magnitude

        Returns
        -------
        int: ThresholdTaskTag.PER_EN
            Job type for processing
        MC_MinimizedStructure
            Perturbed crystal structure as MC_MinimizedStructure object
        float: time
            time to complete perturbation and energy evaluation
        """
        *_, rng = args
        t1 = time.time()
        structure, _move_data = self.perturb_structure(args, tol=1e-2)
        crystal = self.energy_evaluation(structure)
        return ThresholdTaskTag.PER_EN, (crystal, rng), time.time() - t1


    def run_trajectory(self, args):
        sg, trial_number, initial_mc_step, target_mc_steps, name, trial_state, scale = args

        mc_step = initial_mc_step
        params = sg, trial_number
        rng = trial_state.rng
        
        energy_changes = defaultdict(list)
        move_amounts = defaultdict(list)
        accepted = 0
        last_accepted_energy = trial_state._initial_energy
        move_dict = parse_moves_list(self.data['mc']['move'])
        for move_type in move_dict:
            with open(f'{move_type}_energy_change_{trial_number}.csv', 'w') as f:
                f.write('rand,move_amount,energy_change,accepted\n')
        if self.data['mc']['on_the_fly']:
            with open(f'cutoffs_{trial_number}.csv', 'w') as f:
                f.write('mc_step,' + ','.join([x for x in move_dict]) + '\n')

        send_request = None
        results = []
        ttime = 0.0
        LOG.info(f'starting trajectory {trial_number} in SG {sg}')

        send_interval = 40
        #send_interval = prime_numbers_trucated[trial_number % len(prime_numbers_trucated)]
        LOG.info('trajectory {} sends results every {} steps'.format(trial_number, send_interval))

        for mc_step in range(initial_mc_step, target_mc_steps + 1):
            LOG.info('calculating  mc step {} for trial {}'.format(mc_step, trial_number))
            energy = None
            valid = 0
            while energy is None:
                res = trial_state.res_old
                perturb_args = sg, trial_number, mc_step, name, res, scale, rng
                t1 = time.time()
                structure, move_data = self.perturb_structure(perturb_args, tol=1e-2)
                crystal = self.energy_evaluation(structure)
                t2 = time.time() - t1

                if crystal.energy is not None:
                    energy = crystal.energy

                    move_type = move_data[0]
                    energy_change = energy - last_accepted_energy
                    if move_type == "tra":
                        move_amount = float(move_data[1]) * float(move_dict[move_type][1])
                    else:
                        move_amount = (2*float(move_data[1]) - 1) * float(move_dict[move_type][1])
                    energy_changes[move_type].append(energy_change)
                    move_amounts[move_type].append(move_amount)

            # if trial_state.check_duplicate(mc_step): ## as just checks mc step not needed for serial
            ttime += t2
            if trial_state.update_step(crystal.file_content, energy, rng):
                crystal = crystal._replace(accept=True)
                accepted += 1
                valid = 1
                last_accepted_energy = energy

            with open(f'{move_type}_energy_change_{trial_number}.csv', 'a') as f:
                f.write(f'{move_data[1]},{move_amount},{abs(energy_change)},{valid}\n')
            
            LOG.info(f'trial num {trial_number} mc_step {mc_step}: move {move_type} energy change {energy_changes[move_type][-1]} accepted {crystal.accept}')

            #if send_request is not None:
            #    finished = send_request.test()[0]
            #    if finished:
            #        send_request.wait()
            #        send_request = None

            results.append(crystal)

            if len(results) >= send_interval and send_request is None:
                #LOG.info(f"worker (rank {self.rank}) sending results async")
                #send_request = self.comm.isend( 
                self.comm.send( 
                    (ThresholdTaskTag.RUN_MC, (results, params, trial_state), ttime),
                    dest=0,
                    tag=ThresholdTaskTag.MC_RESULTS,
                )
                results.clear()
                ttime = 0.0

            if self.data['mc']['on_the_fly'] and len(energy_changes[move_type]) % 40 == 0:
                abs_energy_changes = [ abs(x) for x in energy_changes[move_type] ] #if abs(x) < 10 ]
                #abs_energy_changes = []
                #for en, amount in zip(energy_changes[move_type], move_amounts[move_type]):
                    #LOG.info(f'{en=} {amount=}')
                    #cutoff = move_dict[move_type][1]
                    #abs_energy_changes.append(min(abs(en), 10.0))
                    #abs_energy_changes.append(min(abs(en)/(amount), 5.0))
                update_move_cutoffs(move_dict, move_type, abs_energy_changes, target=1.0)
                self.data['mc']['move'] = to_move_list(move_dict)

                with open('cutoffs_{}.csv'.format(trial_number), 'a') as f:
                    f.write(str(mc_step) + ',' + ','.join([str(move_dict[x][1]) for x in move_dict]) + '\n')

        
        trial_state.finish_trial()

        #if send_request is not None:
        #    send_request.wait()
        
        LOG.info(f'trail {trial_number} finished')
        LOG.info(f'trial number {trial_number} cutoffs: {move_dict}')
        abs_energy = {
            x : sum([ abs(y) for y in energy_changes[x] ]) / len(energy_changes[x]) for x in energy_changes
        }
        ave_energy = {x: sum(energy_changes[x])/len(energy_changes[x]) for x in energy_changes}
        LOG.info(f'ave abs energy changes: {abs_energy}')
        return ThresholdTaskTag.RUN_MC, (results, params, trial_state), ttime


    def perturb_structure(self, args, tol=1e-3):
        """
        Perturb structure using randomly selected MC move and random magnitude 
        (within move cutoff/limit)

        Parameters
        ----------
        args : tuple
            the input for the MC step containing: 
                space group number
                trial/trajectory number
                MC step number, 
                name of simulation, 
                crystal structure as SHELX file contents, 
                scalar to multiple by of MC move magnitude

        Returns
        -------
        PerturbedStructure
            Perturbed crystal structure object
        """
        sg, seed, mc_step, name, res_in, scale, rng = args
        
        # set up Monte Carlo move controller and initial structure
        mc_control = MC(
            self.data["mc"]["move"],
            self.data["mc"]["auto_prob"],
            self.data["mc"]["auto_cutoff"],
            all_s=self.data["mc"]["move_all"],
            cut_off_scale=scale,
            rng=rng,
        )
        mc_control.update_sg(res_in)
        
        # Apply random MC move and check bonding is unchanged, repeat until valid.
        while True:
            res_new, move_data = mc_control.mc_move(res_in)
            LOG.debug("trial {} step {} move selected: {}".format(seed, mc_step, move_data))
            crys_tmp = Crystal.from_shelx_string(res_new)
            molecules = crys_tmp.symmetry_unique_molecules()
            if sum(len(x) for x in molecules) != len(crys_tmp.asymmetric_unit):
                continue
            valid = True
            for mol in molecules:
                for bond in self.data["mc"]["bonds"]:
                    try:
                        if np.allclose(mol.bonds.todense(), bond, atol=tol):
                            break
                    except ValueError:
                        continue
                else:
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
    
    
    def energy_evaluation(self, structure):
        """
        Single point Evaluation of perturbed structure

        Parameters
        ----------
        structure : PerturbedStructure
            Valid perturbed structure from MC
              
        Returns
        -------
        MC_MinimizedStructure
            Structure object containing single point energy
        """
        if isinstance(structure, list):
            structure = structure[0]
    
        name = structure.name
        sg = structure.spacegroup
        trial_number = structure.trial_number
        mc_step = structure.mc_step
        ini_res = structure.file_content
        Zp = structure.Zp
        bondlength_cutoffs = self.data["bondlength_cutoffs"]
        u_index = float("nan")
    
        electrostatics = DistributedMultipoles.from_dma_string(self.data["electrostatics"])
    
        time1 = time.time()
        structure = Crystal.from_shelx_string(structure.file_content)

        # Calculate Zp
        if Zp == "calculate":
            mols = structure.symmetry_unique_molecules()
            _, _, Zp = find_formula_unit(mols)
        
        energy, density = single_point_evaluation(
            structure,
            electrostatics,
            self.data["axis"],
            bondlength_cutoffs=bondlength_cutoffs,
            potential=self.data["minimization"]["neighcrys"]["potential"],
            name=name,
            **self.data["minimization"]["dmacrys"],
            Zp=Zp
        )
        structure_id = CspDatabaseId.from_components(
            name, self.task_name, sg, trial_number, 0, mc_step
        )
    
        mtime = time.time() - time1
        res = structure.to_shelx_string(titl=f"{structure_id} {energy} {density}")
        xrd = None
    
        return MC_MinimizedStructure(
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
            Zp=Zp
        )
