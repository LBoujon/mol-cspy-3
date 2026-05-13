import warnings
from collections import defaultdict, deque
import logging
import pandas as pd
import numpy as np
import math
import os
import sys
import time
import toml
import sqlite3
from cspy.db import CspDataStore
from cspy.configuration import COMMON_SAMPLING_SETTINGS, CONFIG
from cspy.util.time_estimate import strfdelta, timedelta
from cspy.cspympi import WorkQueue
from cspy.cspympi import CSPyTaskTag
import threading
from contextlib import nullcontext
    

LOG = logging.getLogger(__name__)


class QuasiRandomCSP:

    def __init__(
        self,
        workers,
        name,
        spacegroups,
        number_structures=None,
        status_file="status.txt",
        errors_file="errors.txt",
    ):
        self.start_time = time.time()
        self.last_backup_time = self.start_time
        self.n_workers = len(workers)
        self.work_queue = WorkQueue(workers)
        self.name = name

        #JG - when minimization steps < 3 (i.e. standard) valid structures 
        #     won't be loaded from previous database. Therefore, added 
        #     self.max_steps which is then incorporated into restart_from_database.

        self.max_steps = len(CONFIG.get("csp_minimization_step"))
        self.overrides = dict()
        if CONFIG.get("dmacrys.timeout_spg"):
            self.overrides['dmacrys.timeout_spg'] = CONFIG.get("dmacrys.timeout_spg")
        LOG.info('Started quasi-random CSP for target: "%s"', self.name)

        if CONFIG.get("dmacrys.timeout_auto"):
            self.overrides['dmacrys.pilot_timeout_spg'] = dict()
            self.timeout_automate = True
            LOG.info(
                'Setting up for automatic timeout selection. First "%d" successful '
                'GOs for each spacegroup will run with timeout of "%d" seconds.'
                , CONFIG.get("dmacrys.timeout_auto_sample"), CONFIG.get("dmacrys.timeout_auto_cap")
                )
            
            self.overrides['dmacrys.timeout'] = CONFIG.get("dmacrys.timeout_auto_cap")
            self.timeout_bin = CONFIG.get("dmacrys.timeout_auto_bin")
            self.timeout_sample = int(CONFIG.get("dmacrys.timeout_auto_sample"))

            if CONFIG.get("dmacrys.timeout_spg"):
                LOG.info(
                    'Timeouts provided for spacegroup(s): ' + 
                    ', '.join(CONFIG.get("dmacrys.timeout_spg").keys()) + '. Locking these.'
                )
            else:
                self.overrides['dmacrys.timeout_spg'] = dict()

        else:
            self.timeout_automate = False

        self.initialize_spacegroups(spacegroups, number_structures=number_structures)
        self.successful_minimizations = {sg: deque() for sg in self.structures}
        self.status_file = status_file
        self.errors_file = errors_file
        self.load_status_file()
        self.load_errors_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())


        self.db_thread = None
        self.db_writer = None
        self.create_databases()
        if self.timeout_automate:
            self.old_valid_structures = {sg: self.structures[sg]["valid"] for sg in self.structures.keys()}

    def initialize_spacegroups(self, spacegroups, number_structures=None):
        if CONFIG.get('sampling_settings', None):
            LOG.debug('Getting sampling settings from TOML file')
            sampling_settings = CONFIG.get('sampling_settings')
            self.structures = {
                int(x): {
                    "target": int(num),
                    "valid": 0,
                    "qr_fail": 0,
                    "invalid": 0,
                    "seed": 1,
                    "running": 0,
                    "ttime": 0.0,
                    "mtime": 0.0
                }
                for x, num in sampling_settings.items()
            }
            self.errors = {
                int(x) : {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
                for x in sampling_settings.keys()
            }
            self.errors['summary'] = {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
            }
        elif spacegroups in COMMON_SAMPLING_SETTINGS:
            sg = COMMON_SAMPLING_SETTINGS[spacegroups]
            self.structures = {
                x: {
                    "target": sg["number_structures"][x],
                    "valid": 0,
                    "qr_fail": 0,
                    "invalid": 0,
                    "seed": 1,
                    "running": 0,
                    "ttime": 0.0,
                    "mtime": 0.0
                }
                for x in sg["space_group"]
            }
            self.errors = {
                x : {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
            for x in sg["space_group"]}
            self.errors['summary'] = {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
            
            if number_structures is not None:
                for k, v in self.structures.items():
                    v["target"] = number_structures
        else:
            if number_structures is None:
                raise ValueError(
                    "Must set number of structures for custom spacegroup set"
                )
            try:
                spacegroups = [int(x) for x in spacegroups.split()]
            except ValueError as e:
                LOG.error("Error interpreting requested spacegroups: %s", e)
                raise ValueError("Invalid spacegroup") from e
            self.structures = {
                x: {
                    "target": number_structures,
                    "valid": 0,
                    "qr_fail": 0,
                    "invalid": 0,
                    "seed": 1,
                    "running": 0,
                    "ttime": 0.0,
                    "mtime": 0.0
                }
                for x in spacegroups
            }
            self.errors = {
                x : {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
            for x in spacegroups}
            self.errors['summary'] = {
                    "timeout": 0,
                    "max_its": 0,
                    "buck_cat": 0,
                    "mol_clash": 0,
                    "unknown": 0,
                    "total": 0,
                }
        if self.timeout_automate:
            # + 1 because last bin is reserved for logging num timeouts
            # self.GOtimes = {
            #     x : np.zeros(math.ceil((CONFIG.get("dmacrys.timeout_auto_cap") / self.timeout_bin)) + 1)
            #     for x in spacegroups
            #     }
            self.GOtimes = {
                x : []
                for x in self.structures.keys()
                }
        self.minimized_trials = defaultdict(set)

    def generate_structures(self):
        if self.timeout_automate:
            # short batch size means we do fewer GOs with a long timeout 
            # after optimising the timeout
            sample_size = self.timeout_sample
            batch_size = int(sample_size / 5)
            # force a minimum batch size of 1
            if batch_size == 0:
                batch_size = 1
        else:
            batch_size = 500
        for spacegroup, info in self.structures.items():
            valid = info["valid"]
            running = info["running"]
            if self.timeout_automate and not str(spacegroup) in self.overrides['dmacrys.timeout_spg'].keys():
                if not str(spacegroup) in self.overrides['dmacrys.pilot_timeout_spg'].keys():
                    # queue a reduced quantity until we can find a sensible pilot timeout
                    target = sample_size + self.old_valid_structures[spacegroup]
                else:
                    target = info["target"]
            else:
                target = info["target"]

            to_submit = target - (valid + running)

            start_seed, end_seed = info["seed"], info["seed"] + to_submit
            for min_seed in range(start_seed, end_seed, batch_size):
                max_seed = min(end_seed, min_seed + batch_size)
                LOG.debug(
                    "sg: %s, seeds(%d, %d), n = %d",
                    spacegroup,
                    min_seed,
                    max_seed,
                    max_seed - min_seed,
                )
                self.work_queue.append_job((
                    CSPyTaskTag.QR,
                    (spacegroup, (min_seed, max_seed), self.name), dict()
                ))
                info["running"] += max_seed - min_seed
                info["seed"] = max_seed

            if to_submit > 0:
                LOG.debug(
                    "SG(%d): submitting %d more structures (%d, %d, %d)",
                    spacegroup,
                    to_submit,
                    target,
                    valid,
                    running,
                )

    def active_seed_count(self):
        return sum(v["running"] for v in self.structures.values())

    def valid_structure_count(self):
        return sum(v["valid"] for v in self.structures.values())

    def invalid_structure_count(self):
        return sum(v["invalid"] for v in self.structures.values())

    def qr_fail_structure_count(self):
        return sum(v["qr_fail"] for v in self.structures.values())

    def create_databases(self):
        from cspy.db.datastore_writer import DatastoreWriter

        LOG.info("Setting up database files")
        for sg in self.structures:
            filename = f"{self.name}-{sg}.db"
            if not os.path.exists(filename):
                ds = CspDataStore.create_and_connect(filename)
                ds.disconnect()
            else:
                self.restart_from_database(sg, filename)
        self.db_writer = DatastoreWriter(
            self.successful_minimizations,
            filename_format=f"{self.name}-{{db_id}}.db",
            interval=0.5,
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f"{self.name}-dbworker"
        )
        self.db_thread.start()

    def backup_database(self):
        now = time.time()
        time_since_backup = now - self.last_backup_time

        if time_since_backup > CONFIG.get("csp.backup_interval"):
            for sg in self.structures:
                filename = f"{self.name}-{sg}.db"
                src = sqlite3.connect(filename)
                dst = sqlite3.connect(filename + "_bkp")
                src.backup(dst)
                LOG.info("Backed up database %s.", filename)
                src.close()
                dst.close()
            self.last_backup_time = time.time()

    def restart_from_database(self, sg, filename):
        ds = CspDataStore(filename)
        (x,) = ds.query(
            "select max(trial_number) from trial_structure"
        ).fetchall()[0]
        if x is not None:
            self.structures[sg]["seed"] = int(x) + 1
            (nvalid,) = ds.query(
                "select count(*) from trial_structure "
                "where minimization_step={}".format(self.max_steps)
            ).fetchall()[0]
            self.structures[sg]["valid"] = int(nvalid)
        ds.disconnect()

    def structure_table_string(self):
        df = pd.DataFrame.from_dict(self.structures, orient="index")
        df.mtime = df.mtime.round(2)
        df.ttime = df.ttime.round(2)
        # If minor version is greater than 11, filter out: DataFrame.applymap has been deprecated
        context = warnings.catch_warnings(action="ignore", category=FutureWarning) if sys.version_info[1] >= 11 else nullcontext()
        with context:
            df = df.applymap(str)
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
            stat_table_data = pd.read_table(
                self.status_file, skipfooter=1, sep="\s+", engine="python"
            )
            for r in stat_table_data.itertuples():
                sg = r.Index
                if sg not in self.structures:
                    continue
                self.structures[sg]["ttime"] = r.ttime
                self.structures[sg]["mtime"] = r.mtime
                # Unless we start storing which structures failed,
                # we cannot do the following as it will skip some possible
                # valid seeds.
                # self.structures[sg]["valid"] = r.valid
                # self.structures[sg]["qr_fail"] = r.qr_fail
                # self.structures[sg]["invalid"] = r.invalid
                # self.structures[sg]["seed"] = r.seed - r.running

    def update_errors(self):
        errors_string = self.errors_table_string()
        LOG.debug("Current status:\n%s", errors_string)
        with open(self.errors_file, "w") as f:
            f.write(errors_string + "\n\n")

    def errors_table_string(self):
        dfe = pd.DataFrame.from_dict(self.errors, orient="index")

        # If minor version is greater than 11, filter out: DataFrame.applymap has been deprecated
        context = warnings.catch_warnings(action="ignore", category=FutureWarning) if sys.version_info[1] >= 11 else nullcontext()
        with context:
            dfe = dfe.applymap(str)
        return dfe.to_string()
    
    def load_errors_file(self):
        if os.path.exists(self.errors_file) and os.path.getsize(self.errors_file) > 0:
            try:
                LOG.info("Loading structure table from %s", self.errors_file)
                err_table_data = pd.read_table(
                    self.errors_file, skipfooter=1, sep="\s+", engine="python"
                )
                for r in err_table_data.itertuples():
                    sg = r.Index
                    try:
                        sg = int(sg)
                    except:
                        pass   # This needs to handled better
                    if sg not in self.errors:
                        continue
                    self.errors[sg]["total"] = r.total
                    self.errors[sg]["timeout"] = r.timeout
                    self.errors[sg]["max_its"] = r.max_its
                    self.errors[sg]["buck_cat"] = r.buck_cat
                    self.errors[sg]["mol_clash"] = r.mol_clash
                    self.errors[sg]["unknown"] = r.unknown
            except Exception as e:
                LOG.warning("errors.txt file exists but it can't be loaded")
                LOG.warning("Error: %s", e)

    def process_result(self, result):
        """When overriding this method make sure it is FAST"""
        task_type, result, ttime = result
        if task_type == CSPyTaskTag.QR:
            if len(result) == 2:
                sg, crystals = result
            else:
                sg, crystals, au_ids = result
            sginfo = self.structures[sg]
            nsuccess = 0
            nfail = 0
            overrides = dict()
            if 'dmacrys.timeout' in self.overrides.keys():
                overrides['dmacrys'] = {'timeout' : self.overrides['dmacrys.timeout']}
            if 'dmacrys.timeout_spg' in self.overrides.keys():
                if str(sg) in self.overrides['dmacrys.timeout_spg'].keys():
                    overrides['dmacrys'] = {'timeout' : self.overrides['dmacrys.timeout_spg'][str(sg)]}
                elif str(sg) in self.overrides['dmacrys.pilot_timeout_spg'].keys():
                    overrides['dmacrys'] = {'timeout' : self.overrides['dmacrys.pilot_timeout_spg'][str(sg)]}
                    overrides['dmacrys']['pilot_timeout'] = True
                else:
                    overrides['dmacrys']['pilot_timeout'] = True
            for c in crystals:
                if c is None:
                    nfail += 1
                else:
                    nsuccess += 1
                    self.work_queue.prepend_job((CSPyTaskTag.OPT, c, overrides))
                    LOG.debug(
                        "Added minimization: sg %s seed %s", sg, c.trial_number
                    )
            sginfo["running"] -= nfail
            sginfo["invalid"] += nfail
            sginfo["qr_fail"] += nfail
            sginfo["ttime"] += ttime
        elif task_type == CSPyTaskTag.OPT:
            valid, crystals = result
            # valid = len(crystals) == self.expected_crystals
            sg = crystals[0].spacegroup
            sginfo = self.structures[sg]
            trial_number = crystals[0].trial_number
            LOG.debug("Processing result: sg %s seed %s", sg, trial_number)
            if trial_number not in self.minimized_trials[sg]:
                self.minimized_trials[sg].add(trial_number)
                sginfo["running"] -= 1
                sginfo["valid" if valid == True else "invalid"] += 1
                sginfo["ttime"] += ttime
                if valid == True:
                    GO_times = []
                    for c in crystals:
                        if c.minimization_step < 0:
                            continue
                        self.successful_minimizations[c.spacegroup].append(c)
                        if c.time == c.time:
                            self.structures[c.spacegroup]["mtime"] += c.time
                            GO_times.append(c.time)
                    if self.timeout_automate and not sg in self.overrides['dmacrys.timeout_spg'].keys():
                        max_time = np.max(np.array(GO_times))
                        self.GOtimes[sg].append(max_time)
                else:
                    self.errors[sg]["total"] += 1
                    self.errors["summary"]["total"] += 1
                    if valid == "Timeout":
                        self.errors[sg]["timeout"] += 1 
                        self.errors["summary"]["timeout"] += 1
                        if self.timeout_automate:
                            self.GOtimes[sg].append(self.overrides['dmacrys.timeout'])
                    elif valid == 'MaxIts':
                        self.errors[sg]["max_its"] += 1
                        self.errors["summary"]["max_its"] += 1
                    elif valid == 'BuckCat':
                        self.errors[sg]["buck_cat"] += 1
                        self.errors["summary"]["buck_cat"] += 1
                    elif valid == 'MolClash':
                        self.errors[sg]["mol_clash"] += 1
                        self.errors["summary"]["mol_clash"] += 1
                    else:
                        self.errors[sg]["unknown"] += 1
                        self.errors["summary"]["unknown"] += 1

    def estimated_time_remaining(self, spacegroups=None):
        eta = None
        if spacegroups is None:
            spacegroups = self.structures.keys()

        time_remaining = {}
        nvalid = 0
        ntarget = 0
        total_time = 0.0
        for sg in spacegroups:
            i = self.structures[sg]
            if i["valid"] < i["target"]:
                nvalid += i["valid"]
                ntarget += i["target"]
                if i["valid"] > 0:
                    time_remaining[sg] = (i["target"] / i["valid"] - 1.0) * i["ttime"]
                    total_time += i["ttime"]

        if nvalid == ntarget:
            eta = "complete"
        elif nvalid > 0:
            total = int((ntarget / nvalid - 1.0) * total_time / self.n_workers)
            time_fmt = "{d}d {h}h {m}m {s}s"
            eta = strfdelta(timedelta(seconds=total), time_fmt)
        heading = "ETA: "
        return f"{self.n_workers} MPI workers. {heading}{eta}"
    
    def update_toml(self):
        """ Update toml file so automated timeouts can be read in by a future run """
        import shutil
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/cspy.toml'):
            CONFIG.save(cwd + '/cspy.toml')
            
        shutil.copyfile(cwd + '/cspy.toml', cwd + '/cspy.toml_bkp')

        with open('cspy.toml', 'r') as f:
            toml_contents = toml.load(f)
        
        if not 'timeout_spg' in toml_contents['dmacrys']:
            toml_contents['dmacrys']['timeout_spg'] = dict()

        for sg in self.overrides['dmacrys.timeout_spg'].keys():
            if not sg in toml_contents['dmacrys']['timeout_spg'].keys():
                toml_contents['dmacrys']['timeout_spg'][sg] = float(self.overrides['dmacrys.timeout_spg'][sg])

        with open('cspy.toml', 'w') as f:
            toml.dump(toml_contents, f)

    def optimise_timeout(self, unlocked_sgs):
        """ Predicts optimal timeout by calculating what the success rate would have been for 
        timeouts in steps of self.timeout_bin. Then picks maximum value as optimal timeout. """

        self.timeout_automate = False
        updated_timeout = False
        for sg in unlocked_sgs:
            timeout_type = None
            if self.structures[sg]['valid'] >= (self.timeout_sample + self.old_valid_structures[sg]):
                timeout_type = "Final"
            # take first 25% of sample size as pilot for timeout (protects against overly large caps)
            elif self.structures[sg]['valid'] >= ((self.timeout_sample / 4)+ self.old_valid_structures[sg]):
                if not str(sg) in self.overrides['dmacrys.pilot_timeout_spg'].keys():
                    timeout_type = "Pilot"
            if timeout_type:
                GO_counts = np.asarray(self.GOtimes[sg])
                
                # don't include the timedout crystals in the std calc
                if timeout_type == "Final":
                    current_timeout = self.overrides['dmacrys.pilot_timeout_spg'][str(sg)]
                else:
                    current_timeout = self.overrides['dmacrys.timeout']
                bin_size = round(np.std(GO_counts[GO_counts < current_timeout]) / 2)
                if bin_size == 0:
                    bin_size = 1
                num_GO_bins = int(current_timeout / bin_size)
                GO_rates = np.zeros(num_GO_bins)

                for ind, _ in enumerate(GO_rates):
                    timeout = (ind+1) * bin_size
                    num_invalid_GOs = len(GO_counts[GO_counts >= timeout])
                    invalid_GO_time = num_invalid_GOs * timeout
                    valid_GOs = GO_counts[GO_counts < timeout]
                    num_valid_structures = len(valid_GOs)
                    valid_GO_time = np.sum(valid_GOs)

                    GO_rates[ind] = num_valid_structures / (valid_GO_time + invalid_GO_time)

                max_rate_bin = np.argmax(GO_rates)
                
                # opt_timeout = (max_rate_bin + 1) * bin_size
                #Add a standard deviation to timeout, for safety
                opt_timeout = ((max_rate_bin + 1) * bin_size) + (2 * bin_size)
                if not timeout_type == "Pilot":
                    LOG.info('DMACRYS timeout for spacegroup %s automatically set to %s s.', sg, opt_timeout)
                    self.overrides['dmacrys.timeout_spg'][str(sg)] = opt_timeout
                    updated_timeout = True
                else:
                    opt_timeout = opt_timeout * 2
                    LOG.info('DMACRYS pilot timeout for spacegroup %s automatically set to %s s.', sg, opt_timeout)
                    self.overrides['dmacrys.pilot_timeout_spg'][str(sg)] = opt_timeout

            if not timeout_type == "Final":
                self.timeout_automate = True

        if updated_timeout:
            self.update_toml()

    def run(self):

        self.generate_structures()
        # keeping running if there are jobs to do or if there are
        # completed results to gather
        while not self.work_queue.done:
            if CONFIG.get("csp.backup_interval") >= 0:
                self.backup_database()

            if self.timeout_automate:
                unlocked_sgs = [sg for sg in self.structures.keys() if 
                                str(sg) not in self.overrides['dmacrys.timeout_spg'].keys()]
                if len(unlocked_sgs) == 0:
                    self.timeout_automate = False
                else:
                    self.optimise_timeout(unlocked_sgs)

            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            #LOG.info('Num idle cores: %s', len(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(1)

            for result in self.work_queue.results:
                self.process_result(result)
            self.generate_structures()

            self.update_status()
            self.update_errors()

        self.shutdown()

    def total_cpu_time(self):
        return sum(x["ttime"] for x in self.structures.values())

    def total_minimization_time(self):
        return sum(x["mtime"] for x in self.structures.values())

    def shutdown(self):
        LOG.info("Waiting to finish writing database...")
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
        LOG.info("Done writing databases")
        LOG.info("%d valid minimizations", self.valid_structure_count())
        LOG.info("%d failed quasi-random seeds", self.qr_fail_structure_count())
        LOG.info(
            "%d invalid minimizations",
            self.invalid_structure_count() - self.qr_fail_structure_count(),
        )
        LOG.info("Final structure table\n%s", self.structure_table_string())
        endtime = time.time()
        time_fmt = "{d}d {h}h {m}m {s}s"
        LOG.info("CSP complete!")
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
