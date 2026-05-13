import logging
from collections import deque
from time import sleep
import json
import threading
import numpy as np
import time
import os
import random
import sys
from cspy.util.time_estimate import strfdelta, timedelta
from cspy.db import CspDataStore
from cspy.configuration import COMMON_SAMPLING_SETTINGS, CONFIG
from cspy.cspympi import WorkQueue
from cspy.distributed.csp_manager import QuasiRandomCSP
from cspy.similarity.unique_structures import Structure, UniqueStructures
from cspy.sample.mc.mc_change import calc_beta
from cspy.cspympi import CSPyTaskTag

LOG = logging.getLogger(__name__)


class TrialState:
    def __init__(self, id, res_old, en_old, mc_step=0, state="R"):
        self.id = id
        self.res_old = res_old
        self.en_old = en_old
        self.iterations = mc_step
        self.state = "R"

    def _get_id(self, name, sg, trial_number):
        return "-".join([name, str(sg), str(trial_number)])

    def update_accept(self, res_old, en_old):
        self.res_old = res_old
        self.en_old = en_old

    def update_step(self):
        self.iterations += 1
        return self.iterations

    def truncate_trial(self):
        self.state = "T"

    def finish_trial(self):
        self.state = "F"

    def check_duplicate(self, mc_step):
        return mc_step != self.iterations or self.state == "T" or self.state == "F"


class QRBHCSP(QuasiRandomCSP):
    def __init__(
        self,
        workers,
        name,
        spacegroups,
        mc_para,
        number_structures=None,
        status_file="status.txt",
        errors_file="errors.txt"
    ):
        self.start_time = time.time()
        self.last_backup_time = self.start_time
        self.n_workers = len(workers)
        self.work_queue = WorkQueue(workers)
        self.name = name
        self.overrides = dict()
        LOG.info('Started QRBH CSP for target: "%s"', self.name)

        if CONFIG.get("dmacrys.timeout_spg"):
            self.overrides['dmacrys.timeout_spg'] = CONFIG.get("dmacrys.timeout_spg")
        LOG.info('Started quasi-random CSP for target: "%s"', self.name)
        if CONFIG.get("dmacrys.timeout_auto"):
            self.timeout_automate = True
            LOG.info(
                'Setting up for automatic timeout selection. First "%d" successful '
                'GOs for each spacegroup will run with timeout of "%d" seconds.'
                , CONFIG.get("dmacrys.timeout_auto_sample"), CONFIG.get("dmacrys.timeout_auto_cap")
                )
            self.overrides['dmacrys.timeout'] = CONFIG.get("dmacrys.timeout_auto_cap")
            self.timeout_bin = CONFIG.get("dmacrys.timeout_auto_bin")
            if CONFIG.get("dmacrys.timeout_spg"):
                LOG.info(
                    'Timeouts provided for spacegroup(s): ' + 
                    ', '.join(CONFIG.get("dmacrys.timeout_spg").keys()) + '. Locking these.'
                )
            else:
                self.overrides['dmacrys.timeout_spg'] = dict()
        else:
            self.timeout_automate = False

        if mc_para["raw_s"]:
            self.res_to_perturb = self.raw_res_to_perturb
        else:
            self.res_to_perturb = self.significant_res_to_perturb

        self.trial_step = mc_para["trial_step"]
        self.num_trials = mc_para["num_trials"]
        self.cluster_s = mc_para["cluster_s"]
        self.beta = calc_beta(mc_para["temper"])
        self.dump_accept = mc_para["dump_accept"]
        self.niggli_s = mc_para["niggli_s"]
        self.continue_running = mc_para["continue_running"]
        if self.cluster_s:
            self.continue_running = False

        self.initialize_spacegroups(spacegroups, number_structures=number_structures)

        self.unique_structures = {}
        self.unique_index = 0
        self.trial_states = {}
        self.finish_s = {}
        self.trials_to_dump = {}
        self.successful_minimizations = {sg: deque() for sg in self.structures}

        for sg in self.structures:
            self.structures[sg]["valid_min"] = 0
            self.structures[sg]["invalid_min"] = 0
            self.structures[sg]["unique_min"] = 0
            self.structures[sg]["target_trial"] = self.num_trials
            self.structures[sg]["active_trial"] = 0
            self.structures[sg]["truncate_trial"] = 0
            self.structures[sg]["finish_trial"] = 0

            self.trial_states[sg] = {}
            self.trials_to_dump[sg] = deque()
            self.unique_structures[sg] = UniqueStructures()
            self.finish_s[sg] = False

        self.status_file = status_file
        self.errors_file = errors_file
        self.load_status_file()
        self.load_errors_file()
        LOG.info("Initial structure table:\n%s", self.structure_table_string())

        self.db_thread = None
        self.db_writer = None
        self.create_databases()

    def raw_res_to_perturb(self, crystal):
        return crystal.initial_res

    def significant_res_to_perturb(self, crystal):
        return crystal.file_content

    def generate_structures(self):
        batch_size = 50
        for spacegroup, info in self.structures.items():
            if self.finish_s[spacegroup]:
                continue
            target = info["target_trial"]
            valid = info["valid"]
            invalid = info["invalid"]
            running = info["running"]
            active_trial = info["active_trial"]
            finish_trial = info["finish_trial"]
            if self.cluster_s:
                to_submit = target - (running + active_trial)
            else:
                to_submit = target - (running + active_trial + finish_trial)

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
                    (spacegroup, (min_seed, max_seed), self.name)
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

    def finish_trial_count(self):
        return sum(v["finish_trial"] for v in self.structures.values())

    def truncate_trial_count(self):
        return sum(v["truncate_trial"] for v in self.structures.values())

    def active_trial_count(self):
        return sum(v["active_trial"] for v in self.structures.values())

    def create_databases(self):
        from cspy.db.datastore_writer import QRBH_DatastoreWriter

        LOG.info("Setting up database files")
        for sg in self.structures:
            filename = f"{self.name}-{sg}.db"
            if not os.path.exists(filename):
                ds = CspDataStore.create_and_connect(filename)
                ds.disconnect()
            else:
                self.restart_from_database(sg, filename)
        self.db_writer = QRBH_DatastoreWriter(
            self.successful_minimizations,
            self.trials_to_dump,
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
        truncate_trial = 0
        ntrial = ds.query(
            "select max(trial_number) from trial_structure"
        ).fetchone()[0]
        if ntrial is not None:
            nvalid = ds.query(
                "select count(distinct(trial_number)) from trial_structure"
            ).fetchone()[0]
            nvalid_min = nvalid
            for trial_id, trial_number, mc_step, state, metadata in ds.query(
                "select trial_id, trial_number, iterations, state, metadata from trial"
            ).fetchall():
                LOG.debug(
                    "SG(%d): getting trial %s %s state %s ",
                    sg, trial_number, mc_step, state
                )
                
                meta = json.loads(metadata)
                res_old = meta['res_old']

                if self.continue_running:
                    state = "R"

                nvalid_min += mc_step

                self.trial_states[sg][trial_number] = TrialState(
                    id=trial_id,
                    res_old=res_old, 
                    en_old=meta['en_old'], 
                    mc_step=mc_step, 
                    state=state, 
                )

                if state == "R":
                    active_trial += 1
                    LOG.info(
                        "SG(%d): restarting unfinished trial %s %s",
                        sg, trial_number, mc_step
                    )
                    self.work_queue.append_job((
                        CSPyTaskTag.PER_OPT,
                        (sg, trial_number, mc_step, self.name, res_old)
                    ))
                elif state == "F":
                    finish_trial += 1
                elif state == "T":
                    truncate_trial += 1

                
            self.structures[sg]["seed"] = int(ntrial) + 1
            self.structures[sg]["valid"] = nvalid
            self.structures[sg]["valid_min"] = nvalid_min
            self.structures[sg]["active_trial"] = active_trial
            self.structures[sg]["finish_trial"] = finish_trial
            self.structures[sg]["truncate_trial"] = truncate_trial
        ds.disconnect()

    def set_u_index(self, crystals):
        self.unique_index += 1
        return [c._replace(unique_index=self.unique_index) for c in crystals]

    def set_accept(self, crystals):
        return [c._replace(accept=True) for c in crystals]

    def metropolis_accept(self, en_new, en_old):
        delta_E_kj = en_new - en_old
        delta_E = delta_E_kj * 1000
        if delta_E > 0:
            mc_frac = np.exp(-self.beta * delta_E)
            ran_num = np.random.random()
            if ran_num < mc_frac:
                return True
        else:
            return True
        return False

    def fly_cluster(self, struc_tuple):
        if struc_tuple.xrd is None:
            return False
        sg = struc_tuple.spacegroup
        struc_tmp = Structure(
            int(struc_tuple.unique_index),
            (struc_tuple.energy, struc_tuple.density),
            struc_tuple.xrd,
            struc_tuple.file_content,
            errs=(0.5, 0.025),
            trial=struc_tuple.trial_number,
        )
        dup, self_cover = self.unique_structures[sg].insert_check_trial(
            struc_tmp, method="xrd"
        )
        return (len(dup) != 0) and not self_cover

    def update_structure(self, sg):
        self.structures[sg]["valid_min"] += 1
        if self.structures[sg]["valid_min"] > self.structures[sg]["target"]:
            LOG.debug(
                "Finished %d minimisations for space group %d",
                self.structures[sg]["target"],
                sg,
            )
            self.finish_s[sg] = True

    def process_result(self, result):
        """When overriding this method make sure it is FAST"""
        task_type, result, ttime = result
        if task_type == CSPyTaskTag.QR:
            #Get quasi-random generated structures
            sg, crystals = result
            nsuccess = 0
            nfail = 0
            for c in crystals:
                if c is None:
                    nfail += 1
                    continue
                nsuccess += 1
                self.work_queue.prepend_job((CSPyTaskTag.OPT, c))
                LOG.debug("Added minimization: sg %s seed %s", sg, c.trial_number)
            self.structures[sg]["running"] -= nfail
            self.structures[sg]["invalid"] += nfail
            self.structures[sg]["qr_fail"] += nfail
            self.structures[sg]["ttime"] += ttime
        elif task_type == CSPyTaskTag.OPT:
            #Get minimized structures
            valid, crystals = result
            sg = crystals[0].spacegroup
            trial_number = crystals[0].trial_number
            mc_step = crystals[0].mc_step
            self.structures[sg]["ttime"] += ttime

            if trial_number in self.trial_states[sg]:
                if self.trial_states[sg][trial_number].check_duplicate(mc_step):
                    return

            LOG.info(
                "Processing result sg %s trial_number %s mc_step %s",
                sg, trial_number, mc_step,
            )
            if valid == True:
                #With valid minimization, calculate the acceptance
                #probability and decide whether to accept the perturbation
                #with a random number
                # print("\n\n\n\n\n\nPrinting crystals")
                # if crystals[-1].molecule_id:
                #     print("crystal has molecule_id")
                # else:
                #     print("no molecule_id")

                crystals = self.set_u_index(crystals)
                self.update_structure(sg)
                new_s = False
                if mc_step == 0:
                    self.structures[sg]["running"] -= 1
                    self.structures[sg]["valid"] += 1
                    self.structures[sg]["active_trial"] += 1
                    crystals = self.set_accept(crystals)

                    trial_id = "-".join([self.name, str(sg), str(trial_number)])
                    res_old = self.res_to_perturb(crystals[-1])
                    self.trial_states[sg][trial_number] = TrialState(
                        trial_id, res_old, crystals[-1].energy
                    )
                    new_s = True
                else:
                    if self.metropolis_accept(
                        crystals[-1].energy,
                        self.trial_states[sg][trial_number].en_old,
                    ):
                        crystals = self.set_accept(crystals)
                        res_old = self.res_to_perturb(crystals[-1])
                        self.trial_states[sg][trial_number].update_accept(
                            res_old, crystals[-1].energy
                        )
                        new_s = True

                for c in crystals:
                    if c.minimization_step < 0:
                        continue
                    if c.time == c.time:
                        self.structures[sg]["mtime"] += c.time
                    if new_s or not self.dump_accept:
                        self.successful_minimizations[sg].append(c)

                if self.cluster_s:
                    #On-the-fly clustering
                    duplicate_s = self.fly_cluster(crystals[-1])
                    LOG.debug("On the fly clustering %s", duplicate_s)
                    self.structures[sg]["unique_min"] = self.unique_structures[
                        sg
                    ].number_structures
                    if duplicate_s:
                        self.trial_states[sg][trial_number].truncate_trial()
                        self.trials_to_dump[sg].append(
                            (trial_number, self.trial_states[sg][trial_number])
                        )
                        self.structures[sg]["active_trial"] -= 1
                        self.structures[sg]["truncate_trial"] += 1
                        return

                mc_step = self.trial_states[sg][trial_number].update_step()
                if mc_step > self.trial_step and not self.continue_running:
                    #If finished, stop the trial
                    self.trial_states[sg][trial_number].finish_trial()
                    self.trials_to_dump[sg].append(
                        (trial_number, self.trial_states[sg][trial_number])
                    )
                    self.structures[sg]["active_trial"] -= 1
                    self.structures[sg]["finish_trial"] += 1
                    return

                self.trials_to_dump[sg].append(
                    (trial_number, self.trial_states[sg][trial_number])
                )

            else:
                self.errors[sg]["total"] += 1
                self.errors["summary"]["total"] += 1
                if valid == "Timeout":
                    self.errors[sg]["timeout"] += 1 
                    self.errors["summary"]["timeout"] += 1
                    if self.timeout_automate:
                        self.GOtimes[sg][-1] += 1
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
                self.structures[sg]["invalid_min"] += 1
                if mc_step == 0:
                    self.structures[sg]["running"] -= 1
                    self.structures[sg]["invalid"] += 1
                    return

            if not self.finish_s[sg]:
                #If not finished, submitting the next perturbation taks
                res_in = self.trial_states[sg][trial_number].res_old
                self.work_queue.append_job((
                    CSPyTaskTag.PER_OPT,
                    (sg, trial_number, mc_step, self.name, res_in)
                ))
                LOG.debug(
                    "Added per_min: sg %s trial_number %s mc_step %s",
                    sg,
                    trial_number,
                    mc_step,
                )

    def shutdown(self):
        LOG.info("Waiting to finish writing database...")
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
        LOG.info("Done writing databases")
        LOG.info("%d valid minimizations", self.valid_structure_count())
        LOG.info("%d QR valid minimizations", self.valid_structure_count())
        LOG.info("%d failed quasi-random seeds", self.qr_fail_structure_count())
        LOG.info(
            "%d QR invalid minimizations",
            self.invalid_structure_count() - self.qr_fail_structure_count(),
        )
        LOG.info("%d finished BH trials", self.finish_trial_count())
        LOG.info("%d truncated BH trials", self.truncate_trial_count())
        LOG.info("%d unfinished BH trials", self.active_trial_count())

        LOG.info("Final structure table\n%s", self.structure_table_string())
        endtime = time.time()
        time_fmt = "{d}d {h}h {m}m {s}s"
        LOG.info("QRBH complete!")
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
