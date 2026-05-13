import logging
from collections import deque, defaultdict
from cspy.db import CspDataStore
import json
import threading
import time
import os
import pickle
from numpy.random import SeedSequence, PCG64DXSM, Generator, default_rng
from cspy.util.path import Path
from cspy.util.time_estimate import strfdelta, timedelta
from cspy.cspympi import WorkQueue
from cspy.distributed.qrbh_manager import QRBHCSP
from cspy.cspympi.threshold_tasks import PerturbedStructure, ThresholdTaskTag
from cspy.similarity.unique_structures import Structure
from cspy.sample.mc.thre_increase import Increase_Fixed
from cspy.sample.mc.thre_interval import (
    Interval_Fixed, 
    Interval_Max_Ratio, 
    Interval_Exponential_Fixed, 
    Interval_Linear_Fixed,
    Interval_Exp_Norm_Fixed,
    Interval_Empirical_Exp_Fixed,
    Interval_On_The_Fly_Clustering
)

LOG = logging.getLogger(__name__)

class TrialState:
    """
    Threshold MC Trial/Trajectory
    
    Maintains and updates the state of a single threshold MC trajectory, including 
    determining the current energy lid, whether a move will be accepted, and when to 
    update the energy lid/threshold.
    
    Attributes
    ----------
    id : str
        trajectory id composed of name, space group, trajectory number
    interval_tool : obj
        Determines the MC steps for each energy lid and when to move to next lid
    increase_tool : obj
        sets the energy threshold for each energy lid
    _iterations : int
        number of completed MC steps
    _lid_state : int
        energy lid/threshold iteration
    _state : str
        trajectory state as single character (R = running, F = finished, etc)
    _initial_energy: float
        minimised energy of the structure from which trajectory was initiated
    _res_old : str
        res file contents of last accepted MC structure 
    _trajectory : `list` of tuple (int, int)
        records the energy lid iteration and the corresponding MC step when the lid 
        started
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
        **kwargs
    ):
        self.id = id
        self._res_old = res_old
        self._initial_energy = initial_energy 
        self._lid_state = lid_state
        self.rng = rng

        increase_energy = self._get_increase(increase_para)
        if restart_para is not None:
            increase_energy = None

        self._get_interval(interval_para, initial_energy, increase_energy, restart_para, **kwargs)
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
        Set up the interval tool for determining the number of steps per energy lid.

        Parameters
        ----------
        interval_para: list[<str>]
            List containing interval tool type and required options. Parsed from 
            cspy.toml threshold.interval_para
        initial_energy : float
            The energy of the starting structure, which the energy lids are 
            relative to.
        increase_energy : float
            The current energy threshold
        restart_para : dict
            parameters required to restart threshold MC simulation. See generate_restart 
            below
        """
        interval_type = interval_para[0]

        if interval_type == "fixed":
            self.interval_tool = Interval_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        elif interval_type == "max_ratio":
            self.interval_tool = Interval_Max_Ratio(
                interval_para, initial_energy, increase_energy, restart_para
            )
        elif interval_type == "cluster_s":
            unique_structures = kwargs.get('unique_structures')
            self.interval_tool = Interval_On_The_Fly_Clustering(
                interval_para, unique_structures, initial_energy, increase_energy, restart_para,
            )
        elif interval_type == "fixed_exp":
            self.interval_tool = Interval_Exponential_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        elif interval_type == "fixed_linear":
            self.interval_tool = Interval_Linear_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        elif interval_type == "fixed_exp_norm":
            self.interval_tool = Interval_Exp_Norm_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        elif interval_type == "fixed_empirical_exp":
            self.interval_tool = Interval_Empirical_Exp_Fixed(
                interval_para, initial_energy, increase_energy, restart_para
            )
        else:
            raise Exception("No threshold interval type ", interval_type)
   
    def _get_increase(self, increase_para):
        """
        Set up the increase tool for determining the energy thresholds, i.e the 
        energy schedule

        Parameters
        ----------
        increase_para: list[<str>]
            List containing increase tool type and required options. Parsed from 
            cspy.toml threshold.increase_para
        
        Returns
        -------
        increase_energy : float
            current energy threshold
        """
        increase_type = increase_para[0]

        if increase_type == "fixed":
            self.increase_tool = Increase_Fixed(increase_para)
        else:
            raise Exception("No threshold increase type ", increase_type)

        return self.increase_tool.increase_energy

    def _get_id(self, name, sg, trial_number):
        return "-".join([name, str(sg), str(trial_number)])

    def update_step(self, res_new, energy, rng):
        """
        Determines whether an MC move is accepted of rejected and updates the MC 
        trajectory data. Also, checks if increase the energy threshold.

        Parameters
        ----------
        res_new: 
            The structure from the current MC move as SHELX file content
        energy: 
            The single point energy of the current MC move structure
        
        Returns
        -------
        valid : bool
            Whether the move is accepted or rejected based on the current energy lid
        """
        self._iterations += 1
        valid, lift_lid = self.interval_tool.update_step(energy)
        self.rng = rng
        if valid:
            self._res_old = res_new

        if lift_lid:
            self._lid_state += 1
            LOG.info("Increase lid energy %s", self._lid_state)
            increase_energy = self.increase_tool.increase_energy
            datas = self.interval_tool.update_lid_energy(increase_energy)
            datas.insert(0, self._iterations)
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
        meta = {
            'res_old':self._res_old,
            'initial_energy':self._initial_energy,
            'trajectory':self._trajectory,
            'lid_state':self._lid_state,
            'increase': self.increase_tool.increase_energy,
            'rng_state': pickle.dumps(self.rng).decode('latin1'), # 'rng_state': self.rng.bit_generator.state,
            'restart_para':self.interval_tool.restart
        }
        return self.id, self._iterations, self._state, meta


class Threshold(QRBHCSP):
    """
    Threshold MC Simulation Manager
    
    Communicates MC step and optimisation jobs to workers for all MC trajectories, 
    processes results, and writes to the database.
    
    Attributes
    ----------
    initial_crystals : `list` of str
        res file content of the starting structures to initiate MC trajectories from
    successful_minimizations : `dict` of {int: `deque`}
        Stores the accepted and minimized MC structures for writing to databases. Keys 
        are space groups and deque contains structures as MCMinimizedStructure objects. 
        If dump_accept, rejected structures will also be added.
    structures : `dict` of {int: `dict`}
        Records the MC trajectory statistics per space group. Keys are space groups and 
        the value dict contains keys for number of pertubed, accepted, rejected, valid, 
        active_trial, finish_trial etc (see initialize_spacegroups) Used to update 
        status table.
    trial_states : `dict` of {int: `dict`}
        Stores the TrialState objects for each trajectory organized by space group. 
        The dict corresponding to the space group is of the form {int: TrialState} 
        where the key is the trial number.
    trial_step : int
        Target number of MC steps per MC trajectory
    target : int
        Target number of MC steps over all MC trajectories
    """
    def __init__(
        self,
        workers,
        name,
        spacegroups,
        res_files,
        mc_para,
        status_file="status.txt",
        random_seed = None,
    ):
        self.start_time = time.time()
        self.n_workers = len(workers)
        self.work_queue = WorkQueue(workers)
        self.comm = self.work_queue.master.comm
        self.name = name
        self.random_seed = random_seed
        LOG.info('Started Threshold algorithm for target: "%s"', self.name)

        self.initial_crystals = [(x.rstrip('.res'), Path(x).read_text()) for x in res_files]
        self.num_trials = len(self.initial_crystals)

        self.interval_para = mc_para["interval_para"]
        self.increase_para = mc_para["increase_para"]
        self.minimize_s = mc_para["minimize_s"]
        self.min_energy = mc_para["min_energy"]

        #self.cluster_s = mc_para["cluster_s"]
        self.trial_step = mc_para['trial_step']
        self.dump_accept = mc_para["dump_accept"]
        self.move_scale = mc_para["move_scale"]
        self.distributed = mc_para['distributed']

        if mc_para["interval_para"][0] == "cluster_s":
            assert self.minimize_s is True, "toml thresold.minimize must be set true for on-the-fly clustering"
            LOG.warning("!! restarting on-the-fly clustering may not work as expected !!")
            self.cluster_s = True

        LOG.info('running trajectories serially: {}'.format(self.distributed))

        #self.sleep_time = 1.0 if self.minimize_s else 0.1
        self.sleep_time = 0.1

        rng = default_rng(random_seed)
        ss = SeedSequence(int.from_bytes(rng.bytes(8), 'big'))
        #ss = SeedSequence(getrandbits(128))
        self.trial_seed_states = { 
            i: x for i, x in enumerate(ss.spawn(self.num_trials)) 
        }

        self.target = self.num_trials * self.trial_step
        self.unique_structures = {}
        self.unique_index = 0
        self.trial_states = {}
        self.initialize_spacegroups(spacegroups)
        self.successful_minimizations = {sg: deque() for sg in self.structures}
        for sg in self.structures:
            self.trial_states[sg] = {}
            self.unique_structures[sg] = {}

        self.status_file = status_file
        self.load_status_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())

        self.db_thread = None
        self.db_writer = None
        self.create_databases()


    def initialize_spacegroups(self, spacegroups):
        """
        Set up structures dict for recording threshold statistics organised by 
        spacegroup
        
        Parameters
        ----------
        spacegroups : str
            Space separated list of space groups
        """
        try:
            spacegroups = [int(x) for x in spacegroups.split()]
        except ValueError as e:
            LOG.error("Error interpreting requested spacegroups: %s", e)
            raise ValueError("Invalid spacegroup") from e
        self.structures = {
            x: {
                "target": self.target,
                "perturb": 0,
                "accept": 0,
                "reject": 0,
                "valid": 0,
                "invalid": 0,
                "target_trial": self.num_trials,
                "active_trial": 0,
                "finish_trial": 0,
                "ttime": 0.0,
                "mtime": 0.0,
            }
            for x in spacegroups
        }
        self.minimized_trials = defaultdict(set)

    def minimize_initial(self):
        """
        Energy minimize input crystal structures to determine starting configuration 
        and energy for threshold MC trajectories
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
                self.work_queue.prepend_job((ThresholdTaskTag.OPT, structure))

    def create_databases(self):
        from cspy.db.datastore_writer import Threshold_DatastoreWriter

        LOG.info("Setting up database files")
        for sg in self.structures:
            filename = f"{self.name}-{sg}.db"
            if not os.path.exists(filename): # i.e. not restart
                ds = CspDataStore.create_and_connect(filename)
                ds.disconnect()
                
                #set up trajectory rng seeds
                sq = SeedSequence(self.random_seed)
                self.trial_seed_states = {
                    i: x for i, x in enumerate(sq.spawn(self.num_trials)) 
                }
                LOG.info("random seed/entropy to repeat simulation: %d", sq.entropy)
            else:
                LOG.info("restarting databases")
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

    def restart_from_database(self, sg, filename):
        ds = CspDataStore(filename)
        finish_trial = 0
        active_trial = 0
        total_steps = 0
        ntrial = ds.query(
            "select max(trial_number) from trial_structure"
        ).fetchone()[0]
        max_min_step = ds.query(
            "select max(minimization_step) from trial_structure"
        ).fetchone()[0]
        if ntrial is not None:
            for trial_id, trial_number, iterations, state, metadata in ds.query(
                "select trial_id, trial_number, iterations, state, metadata from trial"
            ).fetchall():
                LOG.debug(
                    "SG(%d): getting trial %s %s state %s ",
                    sg, trial_number, iterations, state
                )

                meta = json.loads(metadata)
                lid_state=meta['lid_state']
                res_old = meta['res_old']

                rng = pickle.loads(meta['rng_state'].encode('latin1')) #rng.bit_generator.state = meta["rng_state"]
                trial_state = TrialState(
                    id=trial_id,
                    res_old=res_old,
                    initial_energy=meta['initial_energy'],
                    interval_para=self.interval_para,
                    increase_para=self.increase_para,
                    iterations=iterations,
                    lid_state=lid_state,
                    state=state,
                    restart_para=meta['restart_para'],
                    trajectory=meta['trajectory'],
                    rng=rng,
                    unique_structures=self.unique_structures[sg][trial_number],
                )
                self.trial_states[sg][trial_number] = trial_state

                total_steps += iterations

                if state == "R":
                    LOG.info(
                        "SG(%d): restarting unfinished trial %s mc step %s",
                        sg, trial_number, iterations
                    )
                    active_trial += 1
                    scale = self.move_scale ** lid_state

                    if self.distributed is not True:
                        self.work_queue.append_job((
                            ThresholdTaskTag.RUN_MC,
                            (sg, trial_number, iterations + 1, self.trial_step, self.name, trial_state, scale)
                        ))
                    else:
                        self.work_queue.append_job((
                            ThresholdTaskTag.PER_EN,
                            (sg, trial_number, iterations + 1, self.name, res_old, scale, rng)
                        ))
                elif state == "F":
                    finish_trial += 1

            naccept = ds.query(
                "select count(*) from trial_structure where minimization_step=0 and valid=True"
            ).fetchone()[0]
            nvalid = ds.query(
                f"select count(*) from trial_structure where minimization_step={max_min_step}"
            ).fetchone()[0]

            self.structures[sg]["perturb"] = total_steps
            self.structures[sg]["accept"] = naccept
            self.structures[sg]["reject"] = total_steps - naccept
            self.structures[sg]["valid"] = nvalid
            self.structures[sg]["active_trial"] = active_trial
            self.structures[sg]["finish_trial"] = finish_trial

        #Start new trials that didn't initiate
        for i, structure in enumerate(self.initial_crystals[int(ntrial)+1:]):
            structure = PerturbedStructure(
                name=structure[0], 
                trial_number=i, 
                spacegroup=sg, 
                mc_step=0, 
                file_content=structure[1],
                molecule_id=None,
                Zp="calculate"
            )
            self.work_queue.prepend_job((ThresholdTaskTag.OPT, structure))
            
        self.initial_crystals = []
        ds.disconnect()
        
    def fly_cluster(self, structure, sg, trial_number):
        """
        cluster optimised structure with unique structures from trajectory
        
        Return
        ------
        bool
            True if duplicate, false if unique (or unable to cluster)
        """
        if structure.xrd is None:
            return False
        cluster_structure = Structure(
            id = int(structure.unique_index),
            coords = (structure.energy, structure.density),
            xrd = structure.xrd,
            res = structure.file_content,
            errs = (0.5, 0.025),
            trial = structure.trial_number,
        )
        equivalent = self.unique_structures[sg][trial_number].insert_without_duplicates(
            cluster_structure,
            method = 'xrd',
        )
        return len(equivalent) > 0

    def process_result(self, result):
        """
        process results from workers. Either MC steps (PER_EN) or energy minimizations 
        (OPT). Updates threshold trajectories and submits next job to work queue.
        """
        task_type, result, ttime = result
        if task_type == ThresholdTaskTag.PER_EN:
            #Get perturbed structures
            crystal, rng = result
            sg = crystal.spacegroup
            trial_number = crystal.trial_number
            mc_step = crystal.mc_step
            name = crystal.name

            """
            if crystal.valid:
                self.unique_index += 1
                crystal = crystal._replace(unique_index=self.unique_index)
            """

            if crystal.energy is not None:
                self.unique_index += 1
                crystal = crystal._replace(unique_index=self.unique_index)
                self.structures[sg]["perturb"] += 1
                self.structures[sg]["ttime"] += ttime
                energy = crystal.energy

                if trial_number in self.trial_states[sg]:
                    if self.trial_states[sg][trial_number].check_duplicate(mc_step):
                        LOG.warning(
                            "Duplicate result! sg %s, seed %s, mc_step %s",
                            sg, trial_number, mc_step,
                        )
                        return

                LOG.info(
                    "Processing per_en sg %s trial_number %s mc_step %s energy %s",
                    sg, trial_number, mc_step, energy
                )

                if self.trial_states[sg][trial_number].update_step(
                    crystal.file_content, energy, rng
                ):
                    self.structures[sg]["accept"] += 1
                    crystal = crystal._replace(accept=True)
                    self.successful_minimizations[sg].append(crystal)
                    if self.minimize_s:
                        res_in = crystal.file_content
                        LOG.debug(
                            "Submit new min sg %s trial_number %s mc_step %s",
                            sg, trial_number, mc_step
                        )
                        structure = PerturbedStructure(
                            name=name, 
                            spacegroup=sg, 
                            trial_number=trial_number, 
                            mc_step=mc_step,
                            file_content=res_in,
                            molecule_id=None,
                            Zp=crystal.Zp
                        )
                        self.work_queue.append_job((ThresholdTaskTag.OPT, structure))
                else:
                    self.structures[sg]["reject"] += 1
                    if not self.dump_accept:
                        self.successful_minimizations[sg].append(crystal)

                mc_step += 1
                if mc_step > self.trial_step:
                    self.trial_states[sg][trial_number].finish_trial()
                    self.structures[sg]["active_trial"] -= 1
                    self.structures[sg]["finish_trial"] += 1
                    return

            res_in = self.trial_states[sg][trial_number].res_old
            lid_state = self.trial_states[sg][trial_number].lid_state
            scale = self.move_scale ** lid_state
            LOG.debug(
                "Submit new perturb sg %s trial_number %s mc_step %s scale %s",
                sg, trial_number, mc_step, scale
            )
            self.work_queue.append_job((
                ThresholdTaskTag.PER_EN,
                (sg, trial_number, mc_step, self.name, res_in, scale, rng)
            ))
        elif task_type == ThresholdTaskTag.OPT:
            #Get minimized structures
            valid, crystals = result
            sg = crystals[0].spacegroup
            trial_number = crystals[0].trial_number
            mc_step = crystals[0].mc_step
            name = crystals[0].name
            self.structures[sg]["ttime"] += ttime

            LOG.info(
                "Processing min sg %s trial_number %s mc_step %s",
                sg, trial_number, mc_step,
            )

            if valid == True:
                self.structures[sg]["valid"] += 1
                crystals = self.set_u_index(crystals)

                if mc_step == 0:
                    trial_id = "-".join([name, str(sg), str(trial_number)])
                    res_old = crystals[-1].file_content
                    if self.min_energy is None: 
                        initial_energy = crystals[-1].energy
                    else:
                        initial_energy = self.min_energy

                    seed_state = self.trial_seed_states.pop(trial_number)
                    rng = Generator(PCG64DXSM(seed_state))
                    
                    trial_state = TrialState(
                        trial_id,
                        res_old, 
                        initial_energy, 
                        self.interval_para, 
                        self.increase_para,
                        rng=rng, #pickle.dumps(rng).decode('latin1'),
                    )
                    self.trial_states[sg][trial_number] = trial_state
                    new_s = True
                    self.structures[sg]["active_trial"] += 1

                    for c in crystals:
                        if c.minimization_step < 0:
                            continue
                        self.successful_minimizations[c.spacegroup].append(c)
                        if c.time == c.time:
                            self.structures[c.spacegroup]["mtime"] += c.time

                    if self.distributed is not True:
                        self.work_queue.append_job((
                            ThresholdTaskTag.RUN_MC,
                            (sg, trial_number, 1, self.trial_step, name, trial_state, 1.0)
                        ))
                    else:
                        self.work_queue.append_job((
                            ThresholdTaskTag.PER_EN,
                            (sg, trial_number, 1, self.name, res_old, 1.0, rng)
                        ))
                else:
                    for c in crystals:
                        if c.minimization_step <= 0:
                            continue
                        self.successful_minimizations[c.spacegroup].append(c)
                        if c.time == c.time:
                            self.structures[c.spacegroup]["mtime"] += c.time

                # if self.cluster_s:
                #     duplicate_s = self.fly_cluster(crystals[-1])
                #     self.structures[sg]["unique_min"] = self.unique_structures[
                #         sg
                #     ].number_structures

            else:
                self.structures[sg]["invalid"] += 1
                if mc_step == 0:
                    LOG.error(
                        "Failed initialization sg %s, trial_number %s",
                        sg,
                        trial_number,
                    )
                    return
        elif task_type == ThresholdTaskTag.RUN_MC:
            crystals, params, trial_state = result
            sg, trial_number = params

            for crystal in crystals:
                mc_step = crystal.mc_step
                name = crystal.name

                self.unique_index += 1
                crystal = crystal._replace(unique_index=self.unique_index)
                self.structures[sg]["perturb"] += 1
                self.structures[sg]["ttime"] += ttime
                energy = crystal.energy                

                LOG.info(
                    "Processing MC step: sg %s trial_number %s mc_step %s energy %s",
                    sg, trial_number, mc_step, energy
                )

                if crystal.accept:
                    self.structures[sg]["accept"] += 1
                    #crystal = crystal._replace(accept=True)
                    self.successful_minimizations[sg].append(crystal)
                    if self.minimize_s:
                        res_in = crystal.file_content
                        LOG.debug(
                            "Submit new min sg %s trial_number %s mc_step %s",
                            sg, trial_number, mc_step
                        )
                        structure = PerturbedStructure(
                            name=name, 
                            spacegroup=sg, 
                            trial_number=trial_number, 
                            mc_step=mc_step,
                            file_content=res_in,
                            molecule_id=None,
                            Zp="calculate"
                        )
                        self.work_queue.append_job((ThresholdTaskTag.OPT, structure))
                else:
                    self.structures[sg]["reject"] += 1
                    if not self.dump_accept:
                        self.successful_minimizations[sg].append(crystal)
            
            self.trial_states[sg][trial_number] = trial_state

            if trial_state.trial_state == "F":
                self.structures[sg]["active_trial"] -= 1
                self.structures[sg]["finish_trial"] += 1


    def estimated_time_remaining(self, spacegroups=None):
        eta = None
        if spacegroups is None:
            spacegroups = self.structures.keys()

        time_remaining = {}
        ncompleted = 0
        ntarget = 0
        total_time = 0.0
        nworkers = min(5 * len(self.trial_states), self.n_workers) if self.minimize_s else min(self.n_workers, len(self.trial_states))
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
        return f"{self.n_workers} MPI workers. {heading}{eta}"


    def run(self):
        self.minimize_initial()

        # keeping running if there are jobs to do or if there are
        # completed results to gather
        while not self.work_queue.done:

            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(self.sleep_time)

            for result in self.work_queue.results:
                self.process_result(result)

            has_message = self.comm.Iprobe(tag=ThresholdTaskTag.MC_RESULTS)
            iteration=0
            while has_message and iteration < self.n_workers:
            #if has_message:
                result = self.comm.recv(tag=ThresholdTaskTag.MC_RESULTS) # tag = ThresholdTaskTag.MC_RESULTS
                self.process_result(result)

                has_message = self.comm.Iprobe(tag=ThresholdTaskTag.MC_RESULTS)
                iteration+=1

            self.update_status()

        self.shutdown()

    def shutdown(self):
        LOG.info("Waiting to finish writing database...")
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
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
        self.work_queue.terminate_workers()
