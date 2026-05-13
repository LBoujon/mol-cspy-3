import warnings
from collections import deque
import logging
import pandas as pd
import numpy as np
import os
import sys
import time
from cspy.db import CspDataStoreAUs
from cspy.configuration import CONFIG
from cspy.cspympi import CSPyTaskTag
from cspy.distributed.csp_manager import QuasiRandomCSP
from cspy.chem.energy import boltzmann_weighting, boltzmann_weight_to_temperature
import threading
from contextlib import nullcontext

LOG = logging.getLogger(__name__)

class AUTCSP(QuasiRandomCSP):
    def __init__(
        self,
        workers,
        name,
        spacegroups,
        asymmetric_unit,
        number_structures=None,
        status_file="status.txt",
        errors_file="errors.txt",
        utilisation_file="utilisation.txt"
    ):
        # inherit as much as we can from QuasiRandomCSP
        super().__init__(
            workers,
            name,
            spacegroups,
            number_structures,
            status_file,
            errors_file
        )

        self.utilisation_file = utilisation_file
        self.energy_temperature = None

        if CONFIG.get("aut.median_energy_weight"):
            self.med_en_weight = CONFIG.get("aut.median_energy_weight")
        else:
            self.med_en_weight = 0.5
        LOG.info('Setting weight of median energy asymmetric unit to "%s"', self.med_en_weight)

        if CONFIG.get("aut.num_asym"):
            self.asymu_per_spg = CONFIG.get("aut.num_asym")
        else:
            self.asymu_per_spg = 1000
        LOG.info('Using "%s" asymmetric units per spacegroup', self.asymu_per_spg)

        # if self.timeout_automate:
        #     LOG.info('Increasing automatic timeout sample size (%s) by the asymmetric unit sample size (%s) to %s', self.timeout_sample, self.asymu_per_spg, self.timeout_sample+self.asymu_per_spg)
        #     self.timeout_sample += self.asymu_per_spg

        self.target_num_asymus = {sg: self.asymu_per_spg for sg in self.structures}
        self.valid_num_asymus = {sg: 0 for sg in self.structures}
        # this dict is just used for writing to the database
        self.valid_asymus_storage_queue = {sg: deque() for sg in self.structures}
        self.valid_asymus = {sg: deque() for sg in self.structures}
        self.asymu_energy_weights = {sg: [] for sg in self.structures}
        self.asymu_weights = {sg: [] for sg in self.structures}
        self.asymu_utilisation = {sg: [0 for _ in range(self.asymu_per_spg)] for sg in self.structures}

        self.asymu_dbs = {sg: None for sg in self.structures}
        if CONFIG.get("aut.dbs"):
            for sg in CONFIG.get("aut.dbs").keys():
                self.asymu_dbs[int(sg)] = CONFIG.get("aut.dbs")[sg]

        self.create_aut_databases()
        self.load_utilisation_file()
        self.update_weights()
        self.original_asymmetric_unit = asymmetric_unit


    def create_aut_databases(self) -> None:
        from cspy.db.datastore_writer_aus import DatastoreWriterAUs

        LOG.info("Setting up database files")
        for sg in self.structures:
            if not self.asymu_dbs[sg]:
                filename = f"{self.name}-{sg}-AU.db"
                self.asymu_dbs[sg] = filename
                if not os.path.exists(filename):
                    ds = CspDataStoreAUs.create_and_connect(filename)
                    # TODO: need to write molecule information to db from here
                    ds.disconnect()
                else:
                    self.restart_from_aut_database(sg, filename)
            else:
                self.restart_from_aut_database(sg, self.asymu_dbs[sg])
                # this should make sure that we don't try to make any new asymus
                # for this spacegroup
                LOG.info('Sourcing "%s" asymmetric units for spacegroup "%s" from %d.', len(self.valid_asymus[sg]), sg, self.asymu_dbs[sg])
                self.target_num_asymus[sg] = len(self.valid_asymus[sg])
                self.asymu_utilisation[sg] = [0 for _ in range(len(self.valid_asymus[sg]))]

        self.aut_db_writer = DatastoreWriterAUs(
            self.valid_asymus_storage_queue,
            filename_format=f"{self.name}-{{db_id}}-AU.db",
            interval=0.5,
        )
        self.aut_db_thread = threading.Thread(
            target=self.aut_db_writer.run, name=f"{self.name}-dbworker"
        )
        self.aut_db_thread.start()


    def restart_from_aut_database(self, sg : int, filename : str) -> None:
        """
        If a database already existed, we should load
        its contents, update our dictionaries and carry
        on where it left off.

        """
        ds = CspDataStoreAUs(filename)
        (nvalid,) = ds.query(
            "select count(*) from asym_units "
        ).fetchall()[0]
        self.valid_num_asymus[sg] = int(nvalid)
        for row in ds.select("asym_units",
                                ['id','energy',
                                'xyz_coordinates','molecule_ids']):
            self.valid_asymus[sg].append({"energy" : row[1], "xyz_coordinates" : row[2], "molecule_ids" : row[3]})
        ds.disconnect()


    def asymu_utilisation_table_string(self) -> str:
        """Convert utilisation data to string so we can write to a file"""
        df = pd.DataFrame.from_dict(self.asymu_utilisation, orient="index")
        context = warnings.catch_warnings(action="ignore", category=FutureWarning) if sys.version_info[1] >= 11 else nullcontext()
        with context:
            df = df.applymap(str)
        return df.to_string()


    def update_utilisation(self) -> None:
        """Write utilisations data to file"""
        util_string = self.asymu_utilisation_table_string()
        LOG.debug("Current asymmetric unit utilisation:\n%s", util_string)
        with open(self.utilisation_file, "w") as f:
            f.write(util_string)


    def load_utilisation_file(self) -> None:
        """
        If a utilisation file already existed, we should load
        its contents, update our dictionaries so we know
        which asymmetric units we've already used.

        """
        if os.path.exists(self.utilisation_file):
            LOG.info("Loading utilisation table from %s", self.utilisation_file)
            LOG.info("Total utilisation may exceed structures in database!")
            util_table_data = pd.read_table(
                self.utilisation_file, skipfooter=0, sep="\s+", engine="python"
            )

            for r in util_table_data.itertuples():
                sg = r.Index
                if sg not in self.asymu_utilisation:
                    continue
                for ind, asym_id in enumerate(r._fields):
                    if not ind == 0:
                        # this key corresponds to the index, so skip it
                        self.asymu_utilisation[sg][ind-1] = getattr(r, asym_id)


    def run(self) -> None:

        # check to see if we already have the automatic timeouts
        if self.timeout_automate:
            unlocked_sgs = [sg for sg in self.structures.keys() if 
                            str(sg) not in self.overrides['dmacrys.timeout_spg'].keys()]
            if len(unlocked_sgs) == 0:
                self.timeout_automate = False

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
            time.sleep(1)

            for result in self.work_queue.results:
                self.process_result(result)
                self.extract_asymUs(result)
                self.tally_utilisation(result)

            self.update_weights()

            self.generate_structures()

            self.update_status()
            self.update_errors()
            self.update_utilisation()

        self.shutdown()

        if self.aut_db_thread is not None:
            self.aut_db_writer.complete = True
            self.aut_db_thread.join()


    def schedule_clg(self, spacegroup : int, info : dict) -> tuple:
        """ Figure out how many crystals we should schedule to be generated, 
         which CLG to use, and how big the batch should be.
         This number is influenced by whether we're figuring out automatic timeouts, 
         whether we have enough AUTs, and whether there are idle workers """
        
        sg_timeout_automate = False
        if self.timeout_automate:
            if str(spacegroup) not in self.overrides['dmacrys.timeout_spg'].keys():
                sg_timeout_automate = True
        
        if sg_timeout_automate:
            # short batch size means we do fewer GOs with a long timeout 
            # after optimising the timeout
            sample_size = self.timeout_sample
            batch_size = int(sample_size / 5)
            # force a minimum batch size of 1
            if batch_size == 0:
                batch_size = 1

            if not str(spacegroup) in self.overrides['dmacrys.pilot_timeout_spg'].keys():
                # queue a reduced quantity until we can find a sensible pilot timeout
                target = sample_size + self.old_valid_structures[spacegroup]
            else:
                target = info["target"]

        else:
            batch_size = 500
            sample_size = None
            target = info["target"]

        valid = info["valid"]
        running = info["running"]
        idle_workers = len(self.work_queue.idle)
        LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
        LOG.debug('MPI running: %s', sorted(self.work_queue.running))
        LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)

        # if we don't have enough asymmetric units yet, we'll set a smaller target
        # which we use with CLG 2.0
        if len(self.asymu_energy_weights[spacegroup]) < self.target_num_asymus[spacegroup]:
            CLG = 'clg2.0'
            target = self.target_num_asymus[spacegroup]
            valid = self.valid_num_asymus[spacegroup]
            to_submit = target - (valid + running)
            LOG.debug("QR - To Submit: %d Target: %d Valid: %d Running %d", to_submit, target, valid, running)

            # if we're waiting for asymmetric units but we have idle cores, we should queue some AUT
            # this might lead to generating more structures than we wanted
            # we enforce a maximum excess equal to the number of idle workers at any given time
            # without this highly parallelised systems can lead to some extremely excessive excesses
            expected_excess = (valid + running) - target
            if idle_workers > 0 and idle_workers > expected_excess:
                batch_size = 1
                to_submit = idle_workers - expected_excess
                LOG.debug("QR - Detected idle workers while waiting for asymmetric units. Queueing %d more CLG2 crystals", to_submit)

        else:
            CLG = 'AUT'
            to_submit = target - (valid + running)
            LOG.debug("AUT - To Submit: %d Target: %d Valid: %d Running %d", to_submit, target, valid, running)

        start_seed, end_seed = info["seed"], info["seed"] + to_submit

        if to_submit > 0:
            LOG.debug(
                "SG(%d): submitting %d more structures (%d, %d, %d)",
                spacegroup,
                to_submit,
                target,
                valid,
                running,
            )

        return start_seed, end_seed, batch_size, CLG


    def generate_structures(self) -> None:
        for spacegroup, info in self.structures.items():
            start_seed, end_seed, batch_size, CLG = self.schedule_clg(spacegroup, info)

            for min_seed in range(start_seed, end_seed, batch_size):
                overrides = dict()
                # First few crystals should use the standard CLG. 
                # We use these crystals to get our asymmetric units and then use AUT from here.
                if CLG == 'clg2.0':
                    overrides['clg'] = 'clg2.0'
                    # the asymmetric unit would have been overwritten if the CPU has already been used for AUT
                    # we restore it here
                    overrides['asymmetric_unit'] = self.original_asymmetric_unit
                else:
                    overrides['clg'] = 'aut'
                    overrides['au_weights'] = self.asymu_weights[spacegroup]
                    overrides['au_properties'] = self.valid_asymus[spacegroup]
                    overrides['aus'] = np.arange(len(self.valid_asymus[spacegroup]))

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
                    (spacegroup, (min_seed, max_seed), self.name), overrides
                ))
                info["running"] += max_seed - min_seed
                info["seed"] = max_seed


    def extract_asymUs(self, result : tuple) -> None:
        """
        When overriding this method make sure it is FAST
        Collect completed tasks from MPI worker.
        If task had the OPT tag, we queue up an EAU task to
        extract the asymmetric unit.
        If the task had the EAU tag, the asymmetric unit
        has been extracted and now we must store it and 
        update the number of valid asymmetric units.

        """
        task_type, result, ttime = result
        if task_type == CSPyTaskTag.OPT:
            if len(result) == 2:
                valid, crystals = result
                sg = crystals[0].spacegroup
                sginfo = self.structures[sg]
                trial_number = crystals[0].trial_number
                if valid == True:
                    LOG.debug("Preparing to extract asymmetric unit: sg %s seed %s", sg, trial_number)
                    c = crystals[-1]
                    if self.valid_num_asymus[sg] < self.target_num_asymus[sg]:
                        extract_args = {"structure" : c,
                                        "spacegroup" : sg}
                        self.work_queue.prepend_job((CSPyTaskTag.EAU, extract_args, dict()))
            else:
                LOG.error("Result returned to extract_asymUs does not contain two variables, %d", result)

        elif task_type == CSPyTaskTag.EAU:
            valid, aus_data, sg = result
            if valid == True:
                if self.valid_num_asymus[sg] < self.target_num_asymus[sg]:
                    aus_data["id"] = str(sg) + "-" + str(self.valid_num_asymus[sg])
                    self.valid_num_asymus[sg] += 1
                    self.valid_asymus_storage_queue[sg].append(aus_data)
                    self.valid_asymus[sg].append(aus_data)
                else:
                    LOG.debug("Not extracting the asymmetric unit as we already have enough.")
            else:
                LOG.debug("Invalid asymmetric unit returned. Skipping...")


    def tally_utilisation(self, result : tuple) -> None:
        """
        When overriding this method make sure it is FAST
        Collect completed tasks from MPI worker.
        If task had the QR tag, update the utilisation
        dictionaries so we know how many times we used
        each asymmetric unit.
        """
        task_type, result, ttime = result
        if task_type == CSPyTaskTag.QR:
            if len(result) == 3:
                sg, crystals, au_ids = result
                for ind, c in enumerate(crystals):
                    if c:
                        sg = c.spacegroup
                        au_id = au_ids[ind]
                        self.asymu_utilisation[sg][au_id] += 1
            else:
                # This crystal probably used the CLG 2.0 So it has no source asymu
                pass

    
    def update_weights(self) -> None:
        """
        Calculate the weight (used for random selection) of
        each asymmetric unit.
        The weight is the product of an energy weight and
        a utilisations weight.
        The energy weight is determined by the 'energy_weight'
        (in Kelvin), the energy of the asymmetric unit (in kJ/mol),
        and the energy of the lowest energy asymmetric unit in
        the database (in kJ/mol).
        """
        for sg in self.structures:
            LOG.debug("Valid num asyms: %s", self.valid_num_asymus[sg])
            LOG.debug("AsymU per spg: %s", self.asymu_per_spg)
            if self.valid_num_asymus[sg] >= self.target_num_asymus[sg]: 
                # these wont change, so lets only do it once and store it
                if len(self.asymu_energy_weights[sg]) == 0:
                    energies = []
                    for au in self.valid_asymus[sg]:
                        energies.append(au["energy"])
                    if len(energies) >= self.target_num_asymus[sg]:
                        if not self.energy_temperature:
                            self.energy_temperature = boltzmann_weight_to_temperature([np.min(energies), np.median(energies)],self.med_en_weight)
                        self.asymu_energy_weights[sg] = boltzmann_weighting(energies, temperature=self.energy_temperature)

                # 5000 K makes an AU with 0 uses ~10x more highly weighted than something with 100 uses
                # 17000 K makes an AU with 0 uses ~2x more highly weighted than something with 100 uses
                if len(self.asymu_energy_weights[sg]) == self.target_num_asymus[sg]:
                    utilization_weights = boltzmann_weighting(self.asymu_utilisation[sg], temperature=17000)

                    weights = self.asymu_energy_weights[sg] * utilization_weights
                    self.asymu_weights[sg] = weights