"""
Module that provides the `ThresholdMinimizationManager` and `ThresholdMinimizationWorker` 
used to minimize structures from a threshold database.

Classes
-------

- `ThresholdMinimizationManager`: Manager of threshold structure minimizations
- `ThresholdMinimizationWorker`: Worker that does the minimizations

Named Tuples
------------

- `MinimizationStructure`: Used to store the information of the crystals to be
minimized

Constants
---------

- `QUERY_TEXT_INI`: SQL query to pull the data of all the initial structures in 
all trials
- `QUERY_TEXT`: SQL query to pull the data of all structures that are valid MC
steps from all trials
"""
import warnings
from typing import List, Dict, Any
from pathlib import Path
from cspy.db import CspDataStore
from collections import deque, namedtuple
import logging
import threading
import time
from pprint import pformat
from cspy.cspympi import ReoptTaskTag
from cspy.cspympi import WorkQueue
from cspy.cspympi import ReoptWorker
from cspy.configuration import CspyConfiguration
from cspy.apps.setup_app import check_multipoles_valid
from cspy.apps.dma import generate_combined_name
from cspy import Molecule
import pandas as pd
import sys
import numpy as np
import json
from contextlib import nullcontext
from typing import Union

LOG = logging.getLogger(__name__)


MinimizationStructure = namedtuple(
    "MinimizationStructure", "id spacegroup energy trial_number filename file_content"
)
QUERY_TEXT_INI = (
    "select C.*, T.minimization_step, T.trial_number, T.minimization_time "
    "from crystal C join "
    "(select * from trial_structure) "
    "T on C.id = T.id where C.id like '%-0-0'"
)
QUERY_TEXT = (
    "select C.*, T.minimization_step, T.trial_number, T.minimization_time "
    "from crystal C join "
    "(select * from trial_structure where minimization_step = 0 and valid = 1) "
    "T on C.id = T.id "
)


class ThresholdMinimizationManager():
    """
    Class to handle minimization of structures in
    threshold trajectories.
    """
    def __init__(
            self, 
            databases: List[Path], 
            workers: range, 
            skip: int = 0, 
            only_increasing: bool = False,
            trial_num: Union[int, None] = None,
            name: str = "thresh-minimization"
        ) -> None:
        if skip < 0:
            skip = 0
        self.skip = skip
        self.name = name
        self.structures: Dict[Path, deque] = {}
        self.work_queue = WorkQueue(workers)
        self.successful_minimizations = {}
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
        self.errors_file = Path("errors.txt")
        self.trials = {}

        for database in databases:
            # Read all valid structures from the threshold trajectories
            # and add them to the list of structures to minimize
            ds = CspDataStore(str(database))
            structures = deque()

            db_trials = ds.query("select * from trial").fetchall()
            trials = []
            counts = {}
            previous = {}
            for i, trial in enumerate(db_trials):
                if trial_num is not None:
                    if db_trials[i][1] != trial_num:
                        continue
                trials.append(list(trial))
                meta = json.loads(trial[-1])
                trials[-1][-1] = meta
                counts[trials[-1][1]] = 0

            ini_structures = ds.query(QUERY_TEXT_INI).fetchall()
            for id, sg, _density, energy, _mol_id, file_content, _min_step, trial_number, _min_time in ini_structures:
                if trial_num is not None:
                    if trial_number != trial_num:
                        continue
                structures.append(
                    MinimizationStructure(
                        id=id,
                        spacegroup=sg,
                        energy=energy,
                        trial_number=trial_number,
                        filename=database.parent / f"{database.stem}",
                        file_content=file_content
                    )
                )
                counts[trial_number] += 1
                previous[trial_number] = -1e10

            db_structures = ds.query(QUERY_TEXT).fetchall()
            for id, sg, _density, energy, _mol_id, file_content, _min_step, trial_number, _min_time in db_structures:
                if trial_num is not None:
                    if trial_number != trial_num:
                        continue
                if only_increasing:
                    # Add structures that have higher energy than the previous structure
                    if previous[trial_number] < energy:
                        structures.append(
                            MinimizationStructure(
                                id=id,
                                spacegroup=sg,
                                energy=energy,
                                trial_number=trial_number,
                                filename=database.parent / f"{database.stem}",
                                file_content=file_content
                            )
                        )
                    previous[trial_number] = energy
                    continue

                if counts[trial_number] >= skip:
                    structures.append(
                        MinimizationStructure(
                            id=id,
                            spacegroup=sg,
                            energy=energy,
                            trial_number=trial_number,
                            filename=database.parent / f"{database.stem}",
                            file_content=file_content
                        )
                    )
                    counts[trial_number] = 0
                else:
                    counts[trial_number] += 1
            LOG.info(f"Adding {len(structures)} crystals from {database} to minimization queue")

            ds.disconnect()
            self.structures[database.parent / f"{database.stem}"] = structures
            self.successful_minimizations[database.parent / f"{database.stem}"] = deque()
            self.trials[database.parent / f"{database.stem}"] = trials
        
        self.create_databases()

    
    def create_databases(self) -> None:
        """
        Create the output databases. It will delete any database
        that conflicts with the name of output databases.
        """
        from cspy.db.datastore_writer import DatastoreWriter

        LOG.info("Setting up database files")
        for database in self.structures:
            out_db = database.parent / f"{database.stem}.minimized.db"
            if out_db.exists():
                out_db.unlink()

            ds = CspDataStore.create_and_connect(str(out_db))
            ds.disconnect()

            # Add initial data to database
            ds = CspDataStore(str(out_db))
            ds.add_trials(self.trials[database.parent / f"{database.stem}"])
            ds.disconnect()

        self.db_writer = DatastoreWriter(
            self.successful_minimizations,
            filename_format=f"{{db_id}}.minimized.db",
            interval=0.5,
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f"{self.name}-dbworker"
        )
        self.db_thread.start()

    
    def run(self) -> None:
        """
        Add the crystal structures to the queue and wait for workers
        to finish all the jobs.
        """
        LOG.info("Running")
        for _, v in self.structures.items():
            for structure in v:
                self.work_queue.append_job((ReoptTaskTag.OPT, structure))

        self.process_remaining_work()
        LOG.info("All minimizations complete!")
        self.shutdown()


    def process_remaining_work(self) -> None:
        """
        Get results from workers and process them.
        """
        while not self.work_queue.done:
            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(1)

            # if a known exception occurs, valid will be a string, otherwise bool
            for task, (valid, crystals), ttime in self.work_queue.results:
                LOG.debug("Successful %s: %s s %d", task, ttime, len(crystals))
                if valid == True:
                    LOG.info(f"Successful minimization of crystal {crystals[0].id} from database {crystals[0].filename}.db")
                    for c in crystals:
                        self.successful_minimizations[c.filename].append(c)
                else:
                    c = crystals[0]
                    LOG.warning(f"Unsuccessful minimization of crystal {c.id} from database {c.filename}.db")
                    self.successful_minimizations[c.filename].append(c)
                    self.errors["summary"]["total"] += 1
                    if valid == "timeout":
                        self.errors["summary"]["timeout"] += 1
                    elif valid == 'MaxIts':
                        self.errors["summary"]["max_its"] += 1
                    elif valid == 'BuckCat':
                        self.errors["summary"]["buck_cat"] += 1
                    elif valid == 'MolClash':
                        self.errors["summary"]["mol_clash"] += 1
                    else:
                        self.errors["summary"]["unknown"] += 1
            self.update_errors()


    def update_errors(self) -> None:
        """
        Update the errors file.
        """
        errors_string = self.errors_table_string()
        LOG.debug("Current status:\n%s", errors_string)
        with open(self.errors_file, "w") as f:
            f.write(errors_string + "\n")

    
    def errors_table_string(self) -> str:
        """
        Generate the string that is saved to the errors file.

        Returns
        -------

        str
            The error string
        """
        df = pd.DataFrame.from_dict(self.errors, orient="index")
        # If minor version is greater than 11, filter out: DataFrame.applymap has been deprecated
        context = warnings.catch_warnings(action="ignore", category=FutureWarning) if sys.version_info[1] >= 11 else nullcontext()
        with context:
            df = df.applymap(str)
        return df.to_string()


    def shutdown(self) -> None:
        """
        Shutdown the process after all minimizations done.
        """
        LOG.info("Shutting down...")
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
        self.work_queue.terminate_workers()
        LOG.info("Shutdown complete")


class ThresholdMinimizationWorker(ReoptWorker):
    """
    Handle work sent by the ThresholdMinimizationManager.
    Inherits ReoptWorker, adds the get_worker_data() method.
    """
    @classmethod
    def get_worker_data(cls, **kwargs) -> Dict[str, Any]:
        """
        Create the worker data dictionary from kwargs.

        Arguments
        ---------

        **kwargs
            Used to create the worker_data

        Returns
        -------

        worker_data : Dict[str, Any]
            The worker data

        Raises
        ------

        KeyError
            If any required kwarg is missing. Required ones change depending on what 
            minimization steps are added to CSPy configuration, read note below

        ValueError
            When no valid multipoles are found when needed

        TypeError
            If passed `cutoff` argument is not able to be parsed as a float

        Notes
        -----

        Required kwargs depending on `csp_minimization_step` kind:

        - `vasp` or `dftb`: kwargs are ignored and all data is read from CSPy configuration
        - `dmacrys` or `pmin`: `xyz_files`, `potential`, `keep_files`, `cutoff`. It is also recommended
        to pass `axis`, `charges` and `multipoles` or default values will be checked that 
        may result in a raised ValueError
        """
        config = CspyConfiguration()
        reopt_steps = [step.get('kind') for step in config.get("csp_minimization_step")]
        if "dftb" in reopt_steps or "vasp" in reopt_steps:
            default_name = "reopt_structures"
            worker_data = {
                "charges": None,
                "multipoles": None,
                "axis": None,
                "bondlength_cutoffs": None,
                "minimization": {
                    "minimization_steps": config.get("csp_minimization_step"),
                },
                "check_single_point_energy":
                    config.get("csp.check_single_point_energy")
            }
            LOG.info("Minimization settings: %s", pformat(worker_data["minimization"]))
        else:
            xyz_files = kwargs["xyz_files"]
            default_name = generate_combined_name(xyz_files)
            if kwargs.get("axis", None) is None:
                kwargs["axis"] = default_name + ".mols"
                LOG.info("No axis file provided, trying %s", kwargs["axis"])
            if kwargs.get("charges", None) is None:
                kwargs["charges"] = default_name + "_rank0.dma"
                LOG.info("No charges file provided, trying %s", kwargs["charges"])
            if kwargs.get("multipoles", None) is None:
                kwargs["multipoles"] = default_name + ".dma"
                LOG.info("No multipole file provided, trying %s", kwargs["multipoles"])
            if not check_multipoles_valid(
                xyz_files, kwargs["axis"], kwargs["charges"], kwargs["multipoles"],
                kwargs["potential"]
            ):
                LOG.error("No good matches for given multipoles! Exiting...")
                raise ValueError("No good matches for given multipoles! Exiting...")

            cutoff = kwargs.get("cutoff", None)
            if cutoff == "calculate":
                cutoff = 15.0
                from scipy.spatial.distance import pdist
                for x in xyz_files:
                    mol = Molecule.load(x)
                    if len(mol) == 1:
                        continue
                    cutoff = max(cutoff, 1.5 * np.max(pdist(mol.positions)))
            else:
                cutoff = float(cutoff)

            bondlength_cutoffs = {}
            for x in xyz_files:
                mol = Molecule.load(x)
                for k, v in mol.neighcrys_bond_cutoffs().items():
                    if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                        bondlength_cutoffs[k] = v

            config = CspyConfiguration()
            config.set("neighcrys.potential", kwargs["potential"])
            worker_data = {
                "charges": Path(kwargs["charges"]).read_text(),
                "multipoles": Path(kwargs["multipoles"]).read_text(),
                "axis": Path(kwargs["axis"]).read_text(),
                "bondlength_cutoffs": bondlength_cutoffs,
                "asymmetric_unit": [Path(x).read_text() for x in xyz_files],
                "minimization": {
                    "neighcrys": {
                        "vdw_cutoff": cutoff,
                        "potential": kwargs["potential"],
                    },
                    "pmin": {"timeout": config.get("pmin.timeout")},
                    "dmacrys": {"timeout": config.get("dmacrys.timeout")},
                    "minimization_steps": config.get("csp_minimization_step"),
                },
                "check_single_point_energy":
                    config.get("csp.check_single_point_energy"),
                "descriptors" : config.get("descriptors"),
                "keep_files" : kwargs["keep_files"],
                "Zp": "calculate"
            }
            LOG.info("Minimization settings: %s", pformat(worker_data["minimization"]))

        LOG.debug(f'worker data: %s', pformat(worker_data))
        return worker_data
        