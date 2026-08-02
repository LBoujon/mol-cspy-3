import warnings
import logging
import os
import sys
import subprocess
import threading
import time
import math
import statistics
from functools import reduce
import operator
import json
from collections import deque, namedtuple
import pandas as pd
import numpy as np
import pandas as pd
from cspy.db import CspDataStore
from cspy.util.path import Path
from cspy.util.time_estimate import strfdelta, timedelta
from cspy.cspympi import ReoptWorker
from cspy.cspympi import ReoptTaskTag
from cspy.cspympi import WorkQueue
from contextlib import nullcontext
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_minimisation_arguments

LOG = logging.getLogger(__name__)


MinimizationStructure = namedtuple(
    "MinimizationStructure", "id spacegroup energy trial_number filename file_content"
)

MC_MinimizedStructure = namedtuple(
    "MC_MinimizedStructure",
    "name id spacegroup trial_number minimization_step mc_step unique_index "
    "energy density file_content initial_res xrd time accept",
)


def get_unminimized_key(min_id, step=0):
    contents = min_id.split('-')
    contents[4] = step
    return "-".join(contents)


def strip_minimization_step(id):
    contents = id.split('-')
    contents[-2] = "{}"
    return '-'.join(contents)


class ReoptimizationManager:

    def __init__(
        self,
        database_files,
        workers,
        name,
        restart_minimize=False,
        minimize_unique=False,
        errors_file="errors.txt",
        status_file="reopt_status.txt",
        original_minimization_step = 0,
        **kwargs
    ):
        self.name = name
        self.restart_minimize = restart_minimize
        self.minimize_unique = minimize_unique
        self.database_files = database_files
        self.status_file = status_file
        self.original_minimization_step = original_minimization_step
        self.energy_window = kwargs.get('energy_window', 1e50)
        self.energy_cap = kwargs.get('energy_cap', 1e7)

        self.start_time = time.time()
        self.work_queue = WorkQueue(workers)
        self.n_workers=len(workers)
        LOG.info('Starting ReoptimizationManager for target: "%s"', self.name)
        self.db_thread = None
        self.db_writer = None
        self.smart_reopt_restart = None
        self.errors_file = errors_file

        self.successful_minimizations = {}
        self.structures = {}
        self.errors = {
                'summary' : {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
            }
        self.load_errors_file()
        for filename in self.database_files:
            structures = deque()
            db = CspDataStore(filename)
            min_energy = db.query('select min(energy) from crystal').fetchone()[0]
            name = Path(filename).stem
            if self.restart_minimize:
                contents = db.final_minimizations(with_trial_data=True).fetchall()

                # sort contens by energy
                contents = sorted(contents, key=lambda x: x[3])
                for id, sg, _, en, _, _, _, trial_number, _ in contents:
                    if en > min_energy + self.energy_window or en > self.energy_cap:
                        continue
                    unmin_id = get_unminimized_key(id, self.original_minimization_step)
                    file_content = db.query(
                        "select file_content from crystal where id='{}'".format(
                            unmin_id)
                    ).fetchone()[0]
                    structures.append(
                        MinimizationStructure(
                            id=unmin_id,
                            filename=name,
                            energy=energy,
                            file_content=file_content,
                            spacegroup=sg,
                            trial_number=trial_number,
                        )
                    )
            else:
                if self.minimize_unique:
                    contents = db.unique_structures(with_file_content=True,
                        with_trial_data=True, fallback=True
                    ).fetchall()
                else:
                    contents = db.final_minimizations(with_trial_data=True).fetchall()
                    # sort contens by energy
                    contents = sorted(contents, key=lambda x: x[3])
                for row in contents:
                    id, sg, _density, energy, _molecule_id, file_content,  _min_step, trial_number, _min_time = row
                    if energy > min_energy + self.energy_window or energy > self.energy_cap:
                        continue
                    structures.append(
                        MinimizationStructure(
                            id=id,
                            filename=name,
                            energy=energy,
                            file_content=file_content,
                            spacegroup=sg,
                            trial_number=trial_number,
                        )
                    )
            db.close()
            self.successful_minimizations[name] = deque()
            self.structures[name] = structures
        self.progress = {
            x: {
                "target": len(self.structures[x]),
                "completed": 0,
                "valid": 0,
                "failed": 0,
                "ttime": 0.0,
                "mtime": 0.0,
            } for x in self.structures
        }
        self.create_databases(**kwargs)
        self.minimizations_remaining = sum(
            self.progress[x]["target"] - self.progress[x]["completed"] for x in self.structures)
        LOG.info("%d structures to re-optimize", self.minimizations_remaining)
        self.load_status_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())

    def restart_from_database(self, filename):
        ds = CspDataStore(filename)
        name_stem = Path(filename).stem
        if name_stem[-4:] == ".opt":
            name = name_stem[:-4]
        else:
            name = name_stem
        if name not in self.structures.keys():
            return
        LOG.info("Pruning structures found in database: %s", filename)
        complete_ids = {"-".join(x.id.split("-")[:4]): i for i, x in enumerate(self.structures[name])}
        to_remove = set()
        if self.restart_minimize:
            chosen_step = self.original_minimization_step
        else:
            chosen_step = ds.final_minimization_step()
        for crystal_id, trial_number in ds.query(
            "select id, trial_number from trial_structure where minimization_step={}"
            .format(chosen_step)
        ):
            crystal_id = "-".join(crystal_id.split("-")[:4])
            if crystal_id in complete_ids:
                to_remove.add(complete_ids[crystal_id])
        for idx in sorted(to_remove, reverse=True):
            del self.structures[name][idx]
        completed_minimizations = len(to_remove)
        LOG.info("pruned %d structures", completed_minimizations)
        self.progress[name]["completed"] = completed_minimizations
        ds.close()
        self.smart_reopt_restart = True

    def create_databases(self, **kwargs):
        from cspy.db.datastore_writer import DatastoreWriter

        LOG.info("Setting up database files")
        for filename in self.structures:
            filename = f"{filename}.opt.db"
            if not os.path.exists(filename):
                ds = CspDataStore.create_and_connect(filename)
                ds.disconnect()
            else:
                LOG.info(f'Database {filename} exits. Restarting...')
                if os.path.getsize(filename) == 0:
                    LOG.info("Database %s is empty. Nothing to restart from.", filename)
                else:
                    self.restart_from_database(filename)
        self.db_writer = DatastoreWriter(
            self.successful_minimizations,
            filename_format=f"{{db_id}}.opt.db",
            interval=kwargs.get('write_interval', 0.5),
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f"{self.name}-dbworker"
        )
        self.db_thread.start()

    def structure_table_string(self):
        df = pd.DataFrame.from_dict(self.progress, orient="index")
        df.mtime = df.mtime.round(2)
        df.ttime = df.ttime.round(2)
        df = df.map(str)
        return df.to_string()

    def update_status(self):
        stat_string = self.structure_table_string()
        LOG.debug("Current status:\n%s", stat_string)
        with open(self.status_file, "w") as f:
            f.write(stat_string + "\n")
            f.write(f"\n{self.estimated_time_remaining()}")

    def load_status_file(self):
        if os.path.exists(self.status_file):
            LOG.info("Loading structure table from %s", self.status_file)
            LOG.info("qr_fail and invalid counts may not be accurate when restarting!")
            table_data = pd.read_table(
                self.status_file, skipfooter=1, sep="\s+", engine="python"
            )
            for r in table_data.itertuples():
                name = r.Index
                if name not in self.structures:
                    continue
                self.progress[name]["ttime"] = r.ttime
                self.progress[name]["mtime"] = r.mtime
                self.progress[name]["valid"] = r.valid

    def estimated_time_remaining(self, dbs=None):
        eta = None
        if dbs is None:
            dbs = self.progress.keys()

        time_remaining = {}
        ncompleted = 0
        ntarget = 0
        total_time = 0.0
        for db in dbs:
            i = self.progress[db]
            if i["completed"] < i["target"]:
                ncompleted += i["completed"]
                ntarget += i["target"]
                if i["completed"] > 0:
                    time_remaining[db] = (i["target"] / i["completed"] - 1.0) * i["ttime"]
                    total_time += i["ttime"]

        if ncompleted  == ntarget:
            eta = "complete"
        elif ncompleted > 0:
            total = int((ntarget / ncompleted - 1.0) * total_time / self.n_workers)
            time_fmt = "{d}d {h}h {m}m {s}s"
            eta = strfdelta(timedelta(seconds=total), time_fmt)
        heading = "ETA: "
        return f"{self.n_workers} MPI workers. {heading}{eta}"

    def run(self, reopt_task_tag=ReoptTaskTag.OPT, **kwargs):
        if self.minimizations_remaining == 0:
            LOG.info("Nothing to do!")
            self.shutdown()
            return

        for k, v in self.structures.items():
            for structure in v:
                self.work_queue.append_job((reopt_task_tag, structure))

        self.process_remaining_work()
        LOG.info("All minimizations complete!")
        self.shutdown()

    def process_remaining_work(self):
        while not self.work_queue.done:
            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(1)

            # if a known exception occurs, valid will be a string, otherwise bool
            for task, (valid, crystals), ttime in self.work_queue.results:
                if not isinstance(crystals, list):
                    crystals = [crystals]
                LOG.debug("Successful %s: %s s %d", task, ttime, len(crystals))

                self.progress[crystals[0].filename]["completed"] += 1
                self.progress[crystals[0].filename]["ttime"] += ttime
                if valid == True:
                    self.progress[crystals[0].filename]["valid"] += 1
                    for c in crystals:
                        self.successful_minimizations[c.filename].append(c)
                        if c.time == c.time:
                            self.progress[c.filename]["mtime"] += c.time
                else:
                    c = crystals[0]
                    self.successful_minimizations[c.filename].append(c)

                    self.errors["summary"]["total"] += 1
                    if valid == "Timeout":
                        self.errors["summary"]["timeout"] += 1
                    elif valid == 'MaxIts':
                        self.errors["summary"]["max_its"] += 1
                    elif valid == 'BuckCat':
                        self.errors["summary"]["buck_cat"] += 1
                    elif valid == 'MolClash':
                        self.errors["summary"]["mol_clash"] += 1
                    else:
                        self.errors["summary"]["unknown"] += 1
                    self.progress[crystals[0].filename]["failed"] += 1
            self.update_errors()
            self.update_status()

    def update_errors(self):
        errors_string = self.errors_table_string()
        LOG.debug("Current status:\n%s", errors_string)
        with open(self.errors_file, "w") as f:
            f.write(errors_string + "\n")

    def errors_table_string(self):
        df = pd.DataFrame.from_dict(self.errors, orient="index")

        # If minor version is greater than 11, filter out: DataFrame.applymap has been deprecated
        context = warnings.catch_warnings(action="ignore", category=FutureWarning) if sys.version_info[1] >= 11 else nullcontext()
        with context:
            df = df.applymap(str)
        return df.to_string()
    
    def load_errors_file(self):
        if os.path.exists(self.errors_file):
            LOG.info("Loading structure table from %s", self.errors_file)
            err_table_data = pd.read_table(
                self.errors_file, skipfooter=1, sep="\s+", engine="python"
            )
            for r in err_table_data.itertuples():
                sg = r.Index
                try:
                    sg = int(sg)
                except:
                    pass
                if sg not in self.errors:
                    continue
                self.errors[sg]["total"] = r.total
                self.errors[sg]["timeout"] = r.timeout
                self.errors[sg]["max_its"] = r.max_its
                self.errors[sg]["buck_cat"] = r.buck_cat
                self.errors[sg]["unknown"] = r.unknown

    def detect_outliers(self,
                        energy_dict, 
                        method='median',
                        const=100):
        LOG.info('Running outlier detection')
        outlier_ids = []

        if method == 'median':
            # https://en.wikipedia.org/wiki/Median_absolute_deviation
            median = np.median([float(x) for x in energy_dict.values()])
            mad = np.median([np.absolute(float(x) - median) for x in energy_dict.values()])
            if mad == 0:
                return outlier_ids
            upper_bound = median + ((const * 1.4286) * mad) # 1.4286 scale factor is the approximate relation between std_dev and mad (assuming normal dist)
            lower_bound = median - ((const * 1.4286) * mad)
            LOG.info(f'median energy: {median}, median absolute deviation: {mad}, upper bound: {upper_bound}, lower bound {lower_bound}')
            for structure, energy in energy_dict.items():
                if not lower_bound <= energy <= upper_bound:
                    LOG.info(f'outlier detected: {structure} with energy {energy}')
                    try:
                        outlier_ids.append(structure)
                    except KeyError:
                        pass

        elif method == 'z_score':
            mean = np.mean([float(x) for x in energy_dict.values()])
            std = np.std([float(x) for x in energy_dict.values()])
            for structure, energy in energy_dict.items():
                zscore = np.absolute((energy - mean))/std
                if zscore >= 3:
                    LOG.info(f'outlier detected: {structure} with energy {energy}')
                    try:
                        outlier_ids.append(structure)
                    except KeyError:
                        pass

        else:
            raise NotImplementedError("Outlier detection method not recognised. Continuing without outlier detection")

        return outlier_ids

    def calculate_stats(self, e_window):
        time.sleep(1) # make sure db is written to

        # find initial energies - read from initial db
        for filename in self.database_files:
            db = CspDataStore(filename)
            contents = db.final_minimizations(with_trial_data=True).fetchall()
            energies= {id:energy for id, _sg, _density, energy, _mol_id, _file_content, _min_step, trial_number, _min_time in contents}

        # find final energies
        for filename in self.database_files:
            db = CspDataStore(f'{filename.split(".")[0]}.opt.db')
            contents = db.final_minimizations(with_trial_data=True).fetchall()
            final_energies = {id.split('-OPT')[0]:energy for id, _sg, _density, energy, _mol_id, _file_content, _min_step, trial_number, _min_time in contents}
        LOG.debug(f'There are {len(final_energies.keys())} structures that have been optimised')

        outlier_ids = self.detect_outliers(energy_dict=final_energies, method='median', const=100)
        final_energies = {k:v for k,v in final_energies.items() if k not in outlier_ids}
        LOG.info(f'Finished outlier detection: {len(outlier_ids)} outliers found.')

        # create new dictionary with structure id as key and a list containing the initial and final energies
        all_energies = {k: [energies[k], final_energies[k]] for k in final_energies.keys()}
        if len(all_energies.keys()) == 1:
            LOG.debug(f'only one structure - exiting stats calculation')
            return 0

        LOG.debug('Collected all energies required for stats calculation')       
        diff = [energy_list[-1] - energy_list[0] for energy_list in all_energies.values()]
        e_limit = min(final_energies.values()) + e_window
        mu = statistics.mean(diff)
        sigma = statistics.stdev(diff)

        LOG.debug('Gathering remaining initial energies')
        remaining_initial_energies = [energy for id, energy in energies.items() if id not in outlier_ids and id not in final_energies.keys()]   
        LOG.debug(f'There are {len(remaining_initial_energies)} initial energies remaining.')
        LOG.debug(f'There are {len(all_energies.keys())} structures for which a calculation has succesfully completed.')
        LOG.debug(f'There are {len(outlier_ids)} outlier structures.')
        LOG.debug(f'Total num structures: {len(energies.keys())}, sum of previous three lines: {len(remaining_initial_energies) + len(all_energies.keys()) + len(outlier_ids)}.') 
        LOG.debug('Performing probability calculation')
        p = [0.5 + 0.5 * (math.erf((x + mu - e_limit) / (math.sqrt(2) * sigma)))
             for x in remaining_initial_energies]
        probability = reduce(operator.mul, p, 1) # for python 3.8+ this can be changed to math.prod(p)

        LOG.info(f'Number of structures optimised: {len(final_energies)}, '
                  f'mean: {mu}, '
                  f'sigma: {sigma}, '
                  f'e_limit: {e_limit}, '
                  f'probability: {probability}'
                  )

        return probability

    def smart_optimization_run(self, e_window, num_workers):
        if self.num_minimizations == 0:
            LOG.info("Nothing to do!")
            self.shutdown()
            return

        energies = [structure.energy for structure_list in self.structures.values()
                    for structure in structure_list]
        e_cutoff = min(energies) + e_window
        LOG.debug(f'global min E: {min(energies)}, energy window = {e_window}, cutoff = {e_cutoff}')
        remaining_structures = []
        count = 0
        all_structures = sorted([structure for list_of_structures in self.structures.values() for structure in list_of_structures], key=lambda x:x.energy)
        if not self.smart_reopt_restart:
            for structure in all_structures:
                if structure.energy <= e_cutoff or count < 10:
                    self.work_queue.append_job((ReoptTaskTag.OPT, structure))
                    count += 1
                else:
                    remaining_structures.append(structure)
        else:
            LOG.debug('Job restarted from previous run - ignoring e_window parameter')
            for structure in all_structures:
                if count < num_workers:
                    self.work_queue.append_job((ReoptTaskTag.OPT, structure))
                    count += 1
                else:
                    remaining_structures.append(structure)

        LOG.debug(f'There are {len(remaining_structures)} remaining structures. Starting initial calculations on {num_workers} structures.')
        remaining_structures = sorted(remaining_structures, key=lambda x:x.energy) # might be obsolete now that we sort all_structures on line 291
        self.process_remaining_work()
        probability = self.calculate_stats(e_window=e_window)
        while probability < 0.99:  
            LOG.info(f'probability of {probability} < 0.99. Starting jobs using {self.work_queue.num_ready} workers')
            LOG.debug(f'There are {len(remaining_structures)} structures remaining to be queued')
            if len(remaining_structures) == 0:
                 probability = 1
                 break
            for _ in range(self.work_queue.num_ready):
                 LOG.debug(f'starting next calculation on {remaining_structures[0]}')
                 self.work_queue.append_job((ReoptTaskTag.OPT, remaining_structures.pop(0)))
            self.process_remaining_work()
            probability = self.calculate_stats(e_window=e_window)
        LOG.info(f'probability of {probability} is > 0.99')
        LOG.info("Job complete!")
        self.shutdown()
    def total_cpu_time(self):
        return sum(x["ttime"] for x in self.progress.values())

    def total_minimization_time(self):
        return sum(x["mtime"] for x in self.progress.values())

    def shutdown(self):
        LOG.info("Shutting down...")
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
        LOG.info("Done writing databases")
        LOG.info("Final structure table\n%s", self.structure_table_string())
        endtime = time.time()
        time_fmt = "{d}d {h}h {m}m {s}s"
        LOG.info("Reopt complete!")
        LOG.info(
            "Walltime (this run): %s",
            strfdelta(timedelta(seconds=endtime - self.start_time), time_fmt),
        )
        ttime = self.total_cpu_time()
        mtime = self.total_minimization_time()
        LOG.info("Total CPU time: %s", strfdelta(timedelta(seconds=ttime), time_fmt))
        LOG.info(
            "Valid structures CPU time: %s",
            strfdelta(timedelta(seconds=mtime), time_fmt),
        )
        self.work_queue.terminate_workers()
        LOG.info("Shutdown complete")


class ThresholdOptimizationManager(ReoptimizationManager):
    def __init__(
            self,
            database_files,
            workers,
            name,
            minimize_interval=None,
            status_file="opt_status.txt",
    ):
        self.name = name
        self.minimize_interval = minimize_interval
        self.database_files = database_files
        self.status_file = status_file

        self.start_time = time.time()
        self.n_workers = len(workers)
        self.work_queue = WorkQueue(workers)
        LOG.info('Starting ReoptimizationManager for target: "%s"', self.name)
        self.db_thread = None
        self.db_writer = None

        self.progress = {}
        self.successful_minimizations = {}
        self.structures = {}

        for filename in self.database_files:
            structures = deque()
            db = CspDataStore(filename)
            name = Path(filename).stem
            if name[-4:] == '.opt':
                name = name.rstrip('.opt')
            

            self.progress[name] = {
                "target": 0,
                "completed": 0,
                "valid": 0,
                "failed": 0,
                "ttime": 0.0,
                "mtime": 0.0,
            }

            LOG.info('checking for last minimized structure in each trial')
            final_min_step = db.final_minimization_step()
            trial_nums = sorted([x[0] for x in db.query("select trial_number from trial")])
            last_optimized_in_trial = {}

            for trial_num in trial_nums:
                query_text = (
                    "select id, metadata from trial_structure "
                    "where minimization_step == {} and trial_number == {} "
                    #"order by length(id) desc, id desc "
                    "order by id desc "
                    "limit 1 "
                ).format(final_min_step, trial_num)

                last_row = db.query(query_text).fetchone()
                #mc_step = int(json.loads(last_row[1])['mc_step'])
                #last_optimized_in_trial[trial_num] = mc_step
                #last_optimized_in_trial[trial_num] = get_unminimized_key(last_row[0])
                last_optimized_in_trial = get_unminimized_key(last_row[0])

                LOG.info('Importing structures from trial {}'.format(trial_num))
                query_text = (
                    "select C.id, C.spacegroup, C.file_content, T.* from crystal C join "
                    "trial_structure T "
                    "on T.id == C.id "
                    "where T.valid == 1 and T.minimization_step == 0 "
                    " and T.trial_number == {} and T.id > '{}' "
                    #"order by length(C.id) asc, C.id asc "
                    "order by C.id asc "
                ).format(trial_num, last_optimized_in_trial)
                contents = db.query(query_text)

                for index, row in enumerate(contents):
                    structure_id, sg, res, _, minimization_step, trial_number, valid, minimization_time, metadata = row
                    meta = json.loads(metadata)
                    mc_step = meta['mc_step']
                    u_index = meta['unique_index']

                    LOG.info(f'is {structure_id} > {last_optimized_in_trial}')

                    if index % self.minimize_interval == 0: 
                        LOG.info('added structure %s', structure_id)
                        structures.append(
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
                                initial_res=None,
                                xrd=None,
                                time=float("nan"),
                                accept=valid,
                            )
                        )
            
            LOG.info('structure list length: {}'.format(len(structures)))

            self.progress[name]['completed'] = db.query(
                "select count(*) from trial_structure where minimization_step == 3 and valid == true"
            ).fetchone()[0]

            db.close()
            self.successful_minimizations[name] = deque()
            self.structures[name] = structures
            
            self.progress[name]['target'] = len(structures) + self.progress[name]['completed']

        self.create_databases()
        self.minimizations_remaining = sum(
            self.progress[x]["target"] - self.progress[x]["completed"] for x in self.structures
        )
        LOG.info("%d structures to re-optimize", self.minimizations_remaining)
        self.load_status_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())

    def restart_from_database(self, filename):
        """Restart optimization of threshold database

        continues a previously created <db>.opt.db, removing any already minimized
        structures, i.e. minimization step = 3

        Args:
            filename: Str, name of <db>.opt.db

        """
        ds = CspDataStore(filename)
        name = Path(filename).stem.split(".")[0]
        if name not in self.structures:
            return
        LOG.info("Pruning structures found in database: %s", filename)
        complete_ids_index = {x.id: i for i, x in enumerate(self.structures[name])}

        to_remove = set()

        chosen_step = ds.final_minimization_step()
        for crystal_id, trial_number in ds.query(
                "select id, trial_number from trial_structure where minimization_step={}"
                .format(chosen_step)
        ):
            zero_minstep_id = strip_minimization_step(crystal_id).format("0")
            if zero_minstep_id in complete_ids_index:
                LOG.debug("%s already optimized", zero_minstep_id)
                to_remove.add(complete_ids_index[zero_minstep_id])
        for idx in sorted(to_remove, reverse=True):
            del self.structures[name][idx]
        completed_minimizations = len(to_remove)
        LOG.info("pruned %d structures", completed_minimizations)
        self.progress[name]["completed"] = completed_minimizations
        ds.close()

    def create_databases(self):
        from cspy.db.datastore_writer import Threshold_Reopt_DatastoreWriter

        LOG.info("Setting up database files")
        for filename in self.structures:
            opt_db_filename = f"{filename}.opt.db" if filename[-4:] != '.opt' else f"{filename}.db"
            if not os.path.exists(opt_db_filename):
                subprocess.run(['cp', f'{filename}.db', f'{opt_db_filename}'])
            else:
                #self.restart_from_database(opt_db_filename)
                LOG.info('opt db already exists, restarting...')
        self.db_writer = Threshold_Reopt_DatastoreWriter(
            self.successful_minimizations,
            filename_format=f"{{db_id}}.opt.db",
            interval=0.5,
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f"{self.name}-dbworker"
        )
        self.db_thread.start()

    def run(self):
        LOG.info("manager has started run()")
        if self.minimizations_remaining == 0:
            LOG.info("Nothing to do!")
            self.shutdown()
            return

        for k, v in self.structures.items():
            for structure in v:
                self.work_queue.append_job((ReoptTaskTag.TA_OPT, structure))

        while not self.work_queue.done:

            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(0.1)

            for task, (valid, crystals), ttime in self.work_queue.results:
                LOG.debug("Successful %s: %s s %d", task, ttime, len(crystals))
                self.progress[crystals[0].name]["completed"] += 1
                self.progress[crystals[0].name]["ttime"] += ttime
                if valid:
                    self.progress[crystals[0].name]["valid"] += 1
                    for c in crystals[1:]:
                        self.successful_minimizations[c.name].append(c)
                        if c.time == c.time:
                            self.progress[c.name]["mtime"] += c.time
                else:
                    LOG.info('invalid minimization: %s', crystals[0].id)
                    self.progress[crystals[0].name]["failed"] += 1
                self.minimizations_remaining -= 1
                LOG.info("minimizations remaining: %d", self.minimizations_remaining)
                self.update_status()

        LOG.info("All minimizations complete!")
        self.shutdown()


def main():
    from mpi4py import MPI

    import argparse
    from cspy.configuration import CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database_files",
        type=str,
        nargs="+",
        help="Database files containing crystals for optimization",
    )
    parser.add_argument(
        "-x",
        "--xyz-files",
        default=[],
        type=str,
        nargs="+",
        help="Xyz files containing molecules for generation",
    )
    parser = add_DMACRYS_arguments(parser)
    parser = add_minimisation_arguments(parser)
    parser.add_argument(
        "--energy_window", 
        type=float, 
        help="Energy window from minimum energy of each database to consider for optimization.", 
        default=1e50
    )
    parser.add_argument(
        "--energy_cap",
        type=float,
        help="Max energy of the structures to consider for optimization",
        default=1e7
    )
    parser.add_argument(
        "-r",
        "--restart-minimize",
        action="store_true",
        default=False,
        help="Reminimize the structure from generated structure",
    )
    parser.add_argument(
        "-mq",
        "--minimize-unique",
        action="store_true",
        default=False,
        help="Only minimize the unique structures",
    )
    parser.add_argument(
        "--smart-optimization",
        action="store_true",
        default=False,
        help='''This option optimises all structure within a specified energy window. Then it calculates the distribution
         of energy shifts between the initial and optimised structures and calculates the probability that all remaining
         crystal structures will have new lattice energies that fall within the specified energy window. The remaining
         crystal structures are optimised sequentially untill a target probability is reach (i.e. We are confident there
         are no more high energy structures that will re-optimise into the specified energy window''',
    )
    parser.add_argument(
        "-ta",
        "--threshold-algo",
        action="store_true",
        default=False,
        help="minimize an unminimized threshold algorithm database",
    )
    parser.add_argument(
        "-mi",
        "--minimize-interval",
        type=int,
        default=1,
        help="minimize structures only at given interval",
    )
    parser.add_argument(
        "--initial-min-step",
        type=int,
        default=0,
        help="initial minimization step",
    )
    parser.add_argument(
        "--write-interval",
        type=float,
        default=0.5,
        help="Database writing interval.",
    )
    parser = add_common_arguments(parser)

    args = parser.parse_args()
 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        LOG.error("This program requires at least 2 MPI processes. " \
                "It's because one process is the manager. " \
                "Exiting...")
        sys.exit(1)

    app_name = sys.argv[0].split('/')[-1]
    app = CspyApp(app_name=app_name, args=args, rank=rank, size=size)
    app.configure_app(clg=False, minimisation=True, skip_header=args.skip_header)
    worker_data = app.worker_data

    if rank == 0:
        comm.Split(0)
        if args.threshold_algo:
            csp_manager = ThresholdOptimizationManager(
                database_files=app.database_files,
                workers=range(1, app.num_workers+1),
                name=app.default_name,
                minimize_interval=args.minimize_interval,
            )
            csp_manager.run()

        else:
            csp_manager = ReoptimizationManager(
                database_files=app.database_files,
                workers=range(1, app.num_workers+1),
                name=app.default_name,
                restart_minimize=args.restart_minimize,
                minimize_unique=args.minimize_unique,
                original_minimization_step=args.initial_min_step,
                write_interval=args.write_interval,
                energy_window=app.energy_window,
                energy_cap=app.energy_cap
            )
            if args.smart_optimization:
                csp_manager.smart_optimization_run(e_window=CONFIG.get("reoptimization.smart_optimization_energy_window"), 
                                               num_workers=MPI.COMM_WORLD.Get_size()-1)
            else:
                csp_manager.run(reopt_task_tag=ReoptTaskTag.OPT)

    elif not app.non_worker_core:
        LOG.debug("Core %s is a worker core. Running reoptimisation here.", rank)
        comm.Split(1)
        ReoptWorker(worker_data).run()

    else:
        comm.Split(1)
        LOG.debug("Core %s is a non-worker core. Not running reoptimisation here.", rank)


if __name__ == "__main__":
    main()
