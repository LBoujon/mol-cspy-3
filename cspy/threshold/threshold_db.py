"""
Module that provides an interface to cluster, merge, minimize and
dump structures from threshold databases.

Functions
---------

- `minimize_databases`: Minimize structures from a database, similar to what
`cspy-reoptimize` does but specific to threshold databases
- `parse_db`: Parse the contents of a database and modify all IDs to the new
trial numbers provided. Used when merging databases with `join_databases`
- `join_databases`: Merge the contents of multiple databases into a single one, 
modifying the trial numbers in all IDs so that multiple databases do not
overlap. Uses the `parse_db` function to extract the information from the 
databases
- `dump_db`: Dump the unique or trajectory structures from a database to a zipped folder in 
CIF or SHELX format
- `dump_databases`: Call the `dump_db` function for a list of databases and save 
the unique structures in separate zipped folders, one for each database
- `cluster`: Cluster the contents of a database, for now only clustering method
available is compack. It can cluster structures only within the superbasin
that includes all initial trajectories
- `get_data_by_trial`: Extract the data associated with a trial from
a database
- `split_trials`: Save the data from each trial into separate databases 

Constants
---------

- `UNIQUE_STRUCTURES_SQL`: SQL query to obtain the data from all the unique
structures in the database
- `TRAJECTORY_STRUCTURES_SQL`: SQL query to obtain all the trajectory structures
from a database 
"""

from typing import List, Union, Tuple
from typing_extensions import Literal
import argparse
from argparse import Namespace
from pathlib import Path
from concurrent import futures
import logging
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.db import CspDataStore
from cspy import Crystal
from cspy.configuration import CONFIG
from cspy.potentials import available_potentials
from cspy.threshold.reoptimization_classes import ThresholdMinimizationManager, ThresholdMinimizationWorker
import json
import zipfile
import pandas as pd
import sys


LOG = logging.getLogger(__name__)


def minimize_databases(databases: List[Path], skip: int = 0, increasing: bool = False, trial: Union[int, None] = None, **kwargs) -> None:
    """
    Minimize the crystals of all databases, skipping `skip`
    crystals between consecutive minimizations.
    The `kwargs` are passed to ThresholdMinimizationWorker.get_worker_data()
    in order to read the necessary xyz, charges, multipoles, axis, ... data to run
    the minimization jobs.

    Arguments
    ---------

    databases : List[Path]
        The databases to get the data from

    skip : int
        The number of holding points in the MC trajectory to skip between minimizations

    increasing : bool
        Overrides the skip value and minimizes those structures whose SPE is higher than
        the previous step

    trial : Union[int, None]
        Filter which trial to minimize from. If set to None all trials in the database
        are minimized. Default is None

    **kwargs
        Passed to ThresholdMinimizationWorker.get_worker_data() to obtain the worker_data. 
        Check the method's docs to see which kwargs you need to pass

    Raises
    ------

    ValueError
        When there are not enough MPI ranks to run the minimization jobs or when no valid
        worker_data can be created
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        LOG.error("Run with at least 2 MPI ranks")
        raise ValueError("Not enough ranks")

    try:
        worker_data = ThresholdMinimizationWorker.get_worker_data(**kwargs)
    except:
        LOG.error(f"Could not create valid worker data with following kwargs: {kwargs}")
        raise ValueError("Invalid kwargs for worker_data")
    
    if rank == 0:
        comm.Split(0)
        minimization_man = ThresholdMinimizationManager(
            databases,
            workers=range(1, size),
            skip=skip,
            only_increasing=increasing,
            trial_num=trial
        )
        minimization_man.run()
    else:
        comm.Split(1)
        ThresholdMinimizationWorker(worker_data).run()


def parse_db(database: Path, trial_numbers: range, name: str = "threshutils", only_valid: bool = True) -> \
    Tuple[str, 
          List[Tuple[str, int, int, str, str]], 
          List[Tuple[str, int, Union[float, None], Union[float, None], str]], 
          List[Tuple[str, int, int, int, Union[float, None], Union[str, None]]], 
          List[Tuple[str, str, bytes, str]]
    ]:
    """
    Extracts the contents of a threshold database and renames 
    all the ids of crystals, descriptors, trials, trial_structures
    into new ones so they can be merged into another database 
    without causing conflicts.

    Arguments
    ---------

    database : Path
        Path to the database file

    trial_nubers : range
        Range of trial numbers to substitute the trials in the database

    name : str
        The name that will be added to the first part of the crystal IDs. It
        must not have any '-'. Default is 'threshutils'

    only_valid : bool
        Add only the necessary structures to the returned data, essentially
        skips all minimization steps except 0 and last. Default is True

    Returns
    -------

    (trials, crystals, trial_structures, descriptors) : Tuple[str, List, List, List, List]
        First the name of the database, a list of trials data, list of crystals data,
        list of trial_structures data and a list of descriptors data

    Raises
    ------
    AssertionError
        If the `name` argument contains any '-'
    """
    assert not "-" in name, f"'name' argument cannot contain '-': {name}"

    # get database contents and dump them into output database
    # The ids must be modified so that they have increasing
    # value of trial_number
    # must dump: 
    # 1 - crystals
    # 2 - descriptor
    # 3 - trial
    # 4 - trial_structure
    db = CspDataStore(str(database))

    # initial db contents
    crystals = db.crystals().fetchall()
    descriptors = db.get_descriptor_data()
    trials = db.get_trial_data()
    trial_structures = db.get_trial_structure_data()
    max_min_step: int = db.final_minimization_step()
    db.disconnect()

    for i, trial in enumerate(trials):
        trials[i] = list(trial)
    new_trial_ids = {}
    new_trial_numbers = {}
    for i, trial_number in enumerate(trial_numbers):
        new_trial_id = "{}-Thre-1-{}".format(name, trial_number)
        new_trial_ids[trials[i][1]] = new_trial_id
        new_trial_numbers[trials[i][1]] = trial_number
        trials[i] = list(trials[i])
        meta = json.loads(trial[-1])
        trials[i][-1] = meta
        trials[i][0] = "{}-1-{}".format(name, trial_number)
        trials[i][1] = trial_number

    LOG.debug(f"Read {database} contents: {len(crystals)} crystals, {len(descriptors)} descriptors, {len(trial_structures)} trial_structures, {len(trials)} trials")

    for i, crystal in enumerate(crystals):
        crystals[i] = list(crystal)
    for i, descriptor in enumerate(descriptors):
        descriptors[i] = list(descriptor)
    for i, trial_structure in enumerate(trial_structures):
        trial_structures[i] = list(trial_structure)

    descriptor_idx = 0
    for i, crystal in enumerate(crystals):
        if i % 10000 == 0 and i != 0:
            LOG.info(f"[{database}] Processed {i}/{len(crystals)} crystals")

        crystal = list(crystal)
        old_id: str = crystal[0]

        split_id = old_id.split("-")
        if len(split_id) == 6:
            mc_step = split_id[-1]
            minimization_step = split_id[-2]
            new_id = "{}-{}-{}".format(new_trial_ids[int(split_id[-3])], minimization_step, mc_step)
        elif len(split_id) == 8:
            mc_step = split_id[-3]
            minimization_step = split_id[-1]
            new_id = "{}-0-{}-OPT-{}".format(new_trial_ids[int(split_id[-5])], mc_step, minimization_step)
        else:
            LOG.error(f"[{database}] Crystal id not formatted as expected: {old_id}, skipping...")
            return (str(database), [], [], [], [])

        # change to new values
        crystals[i][0] = new_id

        if mc_step == "0" and minimization_step == str(max_min_step):
            if descriptors[descriptor_idx][0] == old_id:
                descriptors[descriptor_idx][0] = new_id
                descriptor_idx += 1
            else:
                LOG.warning(f"descriptor {descriptor_idx+1} does not match expected id: {old_id}")
        elif minimization_step == str(max_min_step):
            if descriptor_idx >= len(descriptors): # Database has missing descriptors
                LOG.warning("Expecting descriptor but there don't seem to be enough in the database table... Skipping and 'cspy-db cluster' can recalculate them")
                continue
            if descriptors[descriptor_idx][0] == old_id:
                descriptors[descriptor_idx][0] = new_id
                descriptor_idx += 1
            else:
                LOG.warning(f"descriptor {descriptor_idx+1} does not match expected id: {old_id}")

    for i, trial_structure in enumerate(trial_structures):
        if i % 10000 == 0 and i != 0:
            LOG.info(f"[{database}] Processed {i}/{len(crystals)} trial_structures")

        trial_structure = list(trial_structure)
        old_id: str = trial_structure[0]

        split_id = old_id.split("-")
        if len(split_id) == 6:
            mc_step = split_id[-1]
            minimization_step = split_id[-2]
            new_trial_num = new_trial_numbers[int(split_id[-3])]
            new_id = "{}-{}-{}".format(new_trial_ids[int(split_id[-3])], minimization_step, mc_step)
        elif len(split_id) == 8:
            mc_step = split_id[-3]
            minimization_step = split_id[-1]
            new_trial_num = new_trial_numbers[int(split_id[-5])]
            new_id = "{}-0-{}-OPT-{}".format(new_trial_ids[int(split_id[-5])], mc_step, minimization_step)
        else:
            LOG.error(f"[{database}] Trial_structure id not formatted as expected: {old_id}, skipping...")
            return (str(database), [], [], [], [])
        
        trial_structures[i][0] = new_id
        trial_structures[i][2] = new_trial_num

    if only_valid:
        idxs_to_delete = []
        for i, crystal in enumerate(crystals):
            split_id = crystal[0].split("-")
            if len(split_id) == 6:
                mc_step = split_id[-1]
                minimization_step = split_id[-2]
            elif len(split_id) == 8:
                mc_step = split_id[-3]
                minimization_step = split_id[-1]
            else:
                LOG.error(f"[{database}] Crystal id not formatted as expected: {old_id}, skipping...")
                return (str(database), [], [], [], [])

            if minimization_step != str(max_min_step) and minimization_step != "0" and mc_step != "0":
                idxs_to_delete.append(i)
        
        for i in reversed(sorted(idxs_to_delete)):
            del crystals[i]

        idxs_to_delete = []
        for i, trial_structure in enumerate(trial_structures):
            split_id = trial_structure[0].split("-")
            if len(split_id) == 6:
                mc_step = split_id[-1]
                minimization_step = split_id[-2]
            elif len(split_id) == 8:
                mc_step = split_id[-3]
                minimization_step = split_id[-1]
            else:
                LOG.error(f"[{database}] Trial_structure id not formatted as expected: {old_id}, skipping...")
                return (str(database), [], [], [], [])

            if minimization_step != str(max_min_step) and minimization_step != "0" and mc_step != "0":
                idxs_to_delete.append(i)
        
        for i in reversed(sorted(idxs_to_delete)):
            del trial_structures[i]
        
        if len(crystals) != len(trial_structures):
            LOG.warning(f"Crystals and trial_structures do not have the same number of entries {len(crystals)} != {len(trial_structures)}")

    return str(database), trials, crystals, trial_structures, descriptors


def join_databases(databases: List[Path], output_db: Path, workers: int = 3, serial: bool = True, 
                   name: str = "threshutils", only_valid: bool = True) -> None:
    """
    Join multiple threshold databases into a single one `output_db`.
    The individual databases can have multiple trial runs.

    By default it will add them sequentially to the output database. If `serial` is set to False,
    the databases can be read in parallel using `workers` number of jobs, but if the databases are large
    this will raise errors.

    Arguments
    ---------

    databases : List[Path]
        A list of database paths

    output_db : Path
        Path to the output database

    workers : int
        Number of parallel jobs to run when not running in serial. Default is 3

    serial : bool
        Merge databases to output one after the other. Default is True

    name : str
        Name to substitute into the database IDs. Default is 'threshutils'

    only_valid : bool
        Add only the necessary structures to the returned data, essentially
        skips all minimization steps except 0 and last. Default is True
    """
    LOG.info(f"The following databases will be combined into {output_db}: {[str(d) for d in databases]}")

    # count the trials in each database to know the total
    # dict { database: [trial_nums]}
    trial_nums = {}
    count = 0
    if serial:
        for database in databases:
            db = CspDataStore(str(database))
            trials = db.get_trial_data(["trial_id"])
            db.disconnect()

            trial_nums = range(count, count+len(trials))
            LOG.debug(f"{len(trials)} trials in {database}")
            count += len(trials)
            
            db_name, trials, crystals, trial_structures, descriptors = parse_db(database, trial_nums, name, only_valid)

            out_db = CspDataStore(str(output_db))
            LOG.info(f"Adding contents to ouput database from {db_name}")
            out_db.add_trials(trials)
            out_db.add_crystal_structures(crystals)
            out_db.insert_many("trial_structure", trial_structures)
            out_db.insert_many("descriptor", descriptors)
            out_db.disconnect()
    else:
        for database in databases:
            db = CspDataStore(str(database))
            trials = db.get_trial_data(["trial_id"])
            db.disconnect()

            trial_nums[database] = range(count, count+len(trials))
            LOG.debug(f"{len(trials)} trials in {database}")
            count += len(trials)

        with futures.ProcessPoolExecutor(max_workers=workers) as e:
            fs = [ e.submit(parse_db, d, trials, name, only_valid) for d, trials in trial_nums.items()]

            for f in futures.as_completed(fs):
                result = f.result()

                out_db = CspDataStore(str(output_db))
                name, trials, crystals, trial_structures, descriptors = result

                LOG.info(f"Adding contents to ouput database from {name}")
                out_db.add_trials(trials)
                out_db.add_crystal_structures(crystals)
                out_db.insert_many("trial_structure", trial_structures)
                out_db.insert_many("descriptor", descriptors)
                
                out_db.disconnect()


UNIQUE_STRUCTURES_SQL = (
    "select crystal.* "
    "from crystal join "
    "(select distinct unique_id from equivalent_to) "
    "on crystal.id = unique_id join "
    "(select distinct(id), minimization_step, "
    "trial_number, minimization_time, metadata from trial_structure) T "
    " on T.id = crystal.id "
)

TRAJECTORY_STRUCTURES_SQL = (
    "select C.* "
    "from crystal C join "
    "(select * from trial_structure where minimization_step = 0 and (valid = 1 or id like '%-0-0')) "
    "T on C.id = T.id"
)


def dump_db(database: Path, format: Literal["cif", "res"] = "cif", zipped_folder: Union[Path, None] = None,
            extract_kind: Literal["uniques", "trajectory"] = "uniques") -> None:
    """
    Dump the unqiue or trajectory structures of the database into a zipped folder
    in SHELX or CIF format.

    Arguments
    ---------

    database : Path
        Path to the database

    format : Literal["cif", "res"]
        Which format to save the crystals as. Default is 'cif'

    zipped_folder : Union[Path, None]
        The output zip folder. If not provided the output will be formatted as '{database.stem}.zip'.
        Default is None

    extract_kind : Literal["uniques", "trajectory"]
        Which structures to extract. Default is 'uniques'
    """
    if zipped_folder is None:
        out_zip = f"{database.stem}.zip"
    else:
        out_zip = str(zipped_folder) if zipped_folder.suffix == ".zip" else f"{zipped_folder.name}.zip"

    ds = CspDataStore(str(database))
    if extract_kind == "uniques":
        dataframe = pd.read_sql(UNIQUE_STRUCTURES_SQL, ds.connection)
    else:
        dataframe = pd.read_sql(TRAJECTORY_STRUCTURES_SQL, ds.connection)
    structures = dataframe.pop("file_content")
    ids = dataframe.pop("id")
    ds.disconnect()

    LOG.info(f"Read {len(ids)} unique crystals from {database}")

    with zipfile.ZipFile(out_zip, mode="w") as zf:
        for structure_id, res_content in zip(ids, structures):
            if format == "cif":
                c = Crystal.from_shelx_string(res_content)
                c.titl = structure_id
                zf.writestr(f"{structure_id}.cif", c.to_cif_string() + "\n")
            else:
                zf.writestr(f"{structure_id}.res", res_content)


def dump_databases(databases: List[Path], format: Literal["cif", "res"] = "cif", workers: int = 1, extract_kind: Literal["uniques", "trajectory"] = "uniques") -> None:
    """
    Dump the unqiue or trajectory structures of the databases into zipped folders
    in SHELX or CIF format.

    Arguments
    ---------

    databases : List[Path]
        The databases to dump

    format : format: Literal["cif", "res"]
        The format to save the structures. Default is 'cif'

    workers : int
        The number of processes to use to save the databases

    extract_kind : Literal["uniques", "trajectory"]
        Which structures to extract. Default is 'uniques'
    """
    LOG.info(f"The following databases will be dumped: {[str(d) for d in databases]}")

    with futures.ProcessPoolExecutor(max_workers=workers) as e:
        [ e.submit(dump_db, d, format) for d in databases]


def cluster(databases: List[Path], superbasin: bool = True, **kwargs) -> None:
    """
    Cluster the structures in a Threshold database further. The only 
    available clustering method right now is compack.

    The clustering setting defaults are:
    - calculate_missing: True
    - cluster_from_equivalent: True (As the threshold database should already
    have some degree of clustering)
    - method: compack
    - cluster_energy_threshold: 1.0
    - cluster_density_threshold: 0.05
    - jobs: 1
    - cluster_rms_threshold: 0.03

    Arguments
    ---------
    
    databases : List[Path]
        The databases to cluster

    superbasin : bool 
        Cluster only the structures up to the basin connecting all trajectories. 
        Default is True

    **kwargs : Dict[str, Any]
        Containing the clustering settings that want to be modified from the 
        default ones

    Raises
    ------

    NotImplementedError
        When the clustering method selected is not valid
    """
    from cspy.db.clustering import structure_rows, read_equivalent_table, find_duplicates_compack
    from cspy.configuration import CspyConfiguration
    from cspy.threshold.disconnectivity_graph import DisconnectivityGraph
    import numpy as np
    import time
    for database in databases:
        db = CspDataStore(str(database))

        args_dict = {
            "calculate_missing": True,
            "cluster_from_equivalent": True,
            "method": "compack",
            "cluster_energy_threshold": 1.0,
            "cluster_density_threshold": 0.05,
            "jobs": 1,
            "cluster_rms_threshold": 0.03
        }
        args_dict.update(kwargs) # Override defaults to user-defined ones
        args = Namespace(**args_dict)

        t1 = time.time()
        ids, eds, _, molids, contents, _ = structure_rows(
            db, 
            cluster_from_equivalent=args.cluster_from_equivalent, 
            kind=args.method, 
            calculate_missing=args.calculate_missing,
        )
        if superbasin:
            disconn = DisconnectivityGraph(db)
            # find highest lid connecting all trajectories
            highest_lid_level = 0
            for level in reversed(range(1, len(disconn.yticks)+1)):
                if len(disconn.lid_ids_from_level(level)) != 1:
                    highest_lid_level = level+1
                    break
            LOG.info(f"Lowest lid connecting all trajectories: {level}")
            uni_ids_list = []
            for level in reversed(range(1, highest_lid_level)):
                for thre_id in disconn.lid_ids_from_level(level):
                    for struct_id in disconn.structures_in_lid(thre_id):
                        data = disconn.graph.nodes[struct_id]
                        uni_ids_list.append(data["uni_id"])
            
            ids_to_remove = []
            for i, cid in enumerate(ids):
                if cid not in uni_ids_list:
                    ids_to_remove.append(i)

            eds = np.delete(eds, ids_to_remove, 0)
            for i in sorted(ids_to_remove, reverse=True):
                del ids[i]
                del molids[i]
                del contents[i]
        equivalent_table = read_equivalent_table(db) 
        t2 = time.time()
        LOG.debug("%s: loading data took %.3fs", database, t2 - t1)

        if args.method == "compack":
            config = CspyConfiguration()
            compack_settings = config['compack']
            duplicates = find_duplicates_compack(database, ids, eds, molids, contents, compack_settings, args)
        else:
            raise NotImplementedError(
                f"Removing duplicates not supported for method='{args.method}'"
            )
        LOG.info("%s: %d unique structures total", database, len(duplicates))

        for unique_id, equivalent_ids in duplicates.items():
            equivalent_table[unique_id] += equivalent_ids
            for equivalent_id in equivalent_ids:
                equivalent_table[unique_id] += equivalent_table[equivalent_id]
                equivalent_table.pop(equivalent_id)

        db.add_equivalent_structures(equivalent_table)
        db.commit()
        db.disconnect()


def get_data_by_trial(database: Path, trial: int) -> \
    Tuple[List[Tuple[str, int, int, str, str]], 
          List[Tuple[str, int, Union[float, None], Union[float, None], str]], 
          List[Tuple[str, int, int, int, Union[float, None], Union[str, None]]], 
          List[Tuple[str, str, bytes, str]]
    ]:
    """
    Extract the data associated to a specific trial number from
    a database.

    Pulls the contents of the tables: crystal, descriptor, trial, 
    and trial_structure.

    Arguments
    ---------

    database : Path
        The path to the database

    trial : int
        The trial number from which to extract the data

    Returns
    -------

    (trials, crystals, trial_structures, descriptors) : Tuple[List, List, List, List]
        List of trials data, list of crystals data,
        list of trial_structures data and a list of descriptors data
    """
    db = CspDataStore(str(database))
    crystals = db.query(f"select C.* from crystal C join (select trial_number, id from trial_structure where trial_number = {trial}) T on T.id = C.id").fetchall()
    descriptors = db.query(f"select D.* from descriptor D join (select trial_number, id from trial_structure where trial_number = {trial}) T on T.id = D.id").fetchall()
    trials = db.query(f"select T.* from trial T where T.trial_number = {trial}").fetchall()
    trial_structures = db.query(f"select T.* from trial_structure T where T.trial_number = {trial}").fetchall()
    db.disconnect()

    return trials, crystals, trial_structures, descriptors 


def split_trials(databases: List[Path]) -> None:
    """
    Split the database into one database for each trial.

    Arguments
    ---------

    databases : List[Path]
        The databases to split
    """

    for database in databases:
        ds = CspDataStore(str(database))
        trials_data = ds.get_trial_data(["trial_number"])
        ds.disconnect()

        for trial_number in trials_data:
            LOG.info(f"Parsing trial {trial_number[0]}")
            trial_data = get_data_by_trial(database, trial_number[0])
            trial_db_name = f"{database.stem}_t{trial_number[0]}.db"

            LOG.info(f"Adding contents to ouput database {trial_db_name} from trial {trial_number[0]}")
            trials, crystals, trial_structures, descriptors = trial_data
            if len(trials) == 0:
                LOG.info(f"Trial {trial_number[0]} does not exist in {database}")
                continue

            out_db = CspDataStore(trial_db_name)
            out_db.add_trials(trials)
            out_db.add_crystal_structures(crystals)
            out_db.insert_many("trial_structure", trial_structures)
            out_db.insert_many("descriptor", descriptors)
            out_db.disconnect()
    

class ArgparseTypeValidation():
    @staticmethod
    def valid_in_db(file_str: str) -> Path:
        """
        Function to check that the input file exists and has a
        `.db` extension.
        """
        file = Path(file_str)

        if file.is_file():
            if file.name.endswith(".db"):
                return file
            
            raise argparse.ArgumentTypeError(
                f"Database file does not have correct extension: .db"
            )

        raise argparse.ArgumentTypeError(f"Database file does not exist: {file_str}")
    
    @staticmethod
    def valid_out_db(file_str: str) -> Path:
        """
        Function to check that the output file does not exist and has a
        `.db` extension.
        """
        file = Path(file_str)

        if not file.is_file():
            if file.name.endswith(".db"):
                return file
            
            raise argparse.ArgumentTypeError(
                f"Database file does not have correct extension: .db"
            )

        raise argparse.ArgumentTypeError(f"Database file already exists: {file_str}")


class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def parse_args(args: Union[List[str], None] = None, name: Union[str, None] = None) -> argparse.Namespace:
    """
    Parse the arguments of the script.
    """
    if name is None:
        name = __file__
        
    parser = argparse.ArgumentParser(
        prog=name,
        description="""
Program to interact with CSPy mc-threshold databases. It can:
 - Join multiple databases into a single one. Useful if
 running threshold jobs in different databases and you
 want to create a disconnectivity graph from their 
 trajectories (the disconnectivity graph code only
 accepts one single database as input).
 - Dump the structures of the leaf nodes or trajectories 
 from a database, useful to cluster them and add different 
 colours on the vertices of the disconnectivity graph. 
 (This command works more or less the same as 'cspy-db dump').
 - Minimize structures from trajectories again, with the
 option to skip holding points between minimizations. You have 
 to provide xyz, charges, multipoles and axis files when
 calling this command. Minimized databases will be named
 like: {dbname}.minimized.db. This command must be called
 with mpiexec/mpirun and at least 2 ranks or it will stop
 execution.
 - Cluster the threshold database further using COMPACK.
 - Split the contents of a database into one database per trial.""",
        epilog=f"""
Examples:
# Join all databases starting with 'phenan_*'
> {name} --combine phenan_*
# The default output database is called 'combined.db'. Now we can
# cluster it with pxrd and COMPACK for best results
> cspy-db cluster -m cdtw_cos -o output_cdtw_cos.db combined.db
> cspy-db cluster -cfe -m compack -o output_compack.db combined.db

# Dump the leaf node structures in a .zip file in the .cif format
> {name} --dump --dump-format cif combined.db
# This will create a combined.zip folder with all the crystals
# in .cif format. This is useful in order to tag the crystals
# so that the disconnectivity graph can be coloured.

# Minimize structures of all databases, skipping 2
# holding points between minimizations
> mpiexec -np 2 {name} --minimize --minimize-skip 2 -x XYZ -c CHARGES \\
  -m MULTIPOLES -a AXIS *.db 
        """,
        formatter_class=MyFormatter
    )
    parser.add_argument(
        "databases",
        nargs="+",
        type=ArgparseTypeValidation.valid_in_db,
        help="Input databases (.db)",
    )
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize the structures in the databases",
    )
    cmd_group.add_argument(
        "--combine",
        action="store_true",
        help="Combine the databases into a single one",
    )
    cmd_group.add_argument(
        "--dump",
        action="store_true",
        help="""Dump the min structures of a database.
The output will be a .zip file with the 
name of the input database(s)""",
    )
    cmd_group.add_argument(
        "--cluster",
        action="store_true",
        help="""Further cluster the structures in a 
threshold database, right now the only supported 
clustering method is COMPACK""",
    )
    cmd_group.add_argument(
        "--split",
        action="store_true",
        help="Split the input databases into their trials",
    )
    min_group = parser.add_argument_group(
        title="Minimize",
        description="Configure how the minimization of the db will be done"
    )
    min_group.add_argument(
        "--minimize-skip",
        type=int,
        default=0,
        help="""Number of steps to skip between
minimization of structures""",
    )
    min_group.add_argument(
        "--minimize-increasing",
        action="store_true",
        help="""Minimize structures that have higher
SPE than the previous MC step""",
    )
    min_group.add_argument(
        "--minimize-trial",
        type=int,
        default=None,
        help="""Filter which trial to minimize""",
    )
    min_group.add_argument(
        "-x",
        "--xyz-files",
        type=str,
        nargs="+",
        help="Xyz files containing molecules for generation",
    )
    min_group.add_argument(
        "-c",
        "--charges",
        type=str,
        default=CONFIG.get("csp.charges_file"),
        help="Rank0 multipole file",
    )
    min_group.add_argument(
        "-m",
        "--multipoles",
        type=str,
        default=CONFIG.get("csp.multipoles_file"),
        help="RankN multipole file",
    )
    min_group.add_argument(
        "-a",
        "--axis",
        type=str,
        default=None,
        help="Axis filename for structure minimization",
    )
    min_group.add_argument(
        "-p",
        "--potential",
        default=CONFIG.get("csp.potential"),
        choices=available_potentials.keys(),
        help="intermolecular potential name",
    )
    min_group.add_argument(
        "--cutoff",
        default=CONFIG.get("csp.cutoff"),
        help="dmacrys real space/repulsion-dispersion cutoff",
    )
    min_group.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep DMACRYS and NEIGHCRYS files which, for each structure, are stored in a new directory in the pwd."
    )
    comb_group = parser.add_argument_group(
        title="Combine",
        description="Combine databases into single one"
    )
    comb_group.add_argument(
        "--combine-db-name",
        type=ArgparseTypeValidation.valid_out_db,
        default=Path("combined.db"),
        help="Name of the combined output database",
    )
    comb_group.add_argument(
        "--combine-only-valid",
        action="store_true",
        help="""Reduce amount of data added to new database""",
    )
    comb_group.add_argument(
        "--combine-serial",
        action="store_true",
        help="""Combine the databases in serial instead of parallel""",
    )
    dump_group = parser.add_argument_group(
        title="Dump",
        description="Dump structures of a database into a folder"
    )
    dump_group.add_argument(
        "--dump-format",
        choices=("cif", "res"),
        default="cif",
        help="Set the format to dump the crystal files into",
    )
    dump_group.add_argument(
        "--dump-kind",
        choices=("unique", "trajectory"),
        default="cif",
        help="Which structures to dump from the database",
    )
    cluster_group = parser.add_argument_group(
        title="Cluster",
        description="Further cluster threshold database"
    )
    cluster_group.add_argument(
        "--cluster-superbasin",
        action="store_true",
        help="""Cluster structures in smallest basin
containing all trajectory starts""",
    )
    cluster_group.add_argument(
        "--cluster-method",
        choices=("compack",),
        default="compack",
        help="Cluster algorithm",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of workers to use for some actions",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        choices=("INFO", "DEBUG", "WARNING", "ERROR"),
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args(args)


def main(args: Union[List[str], None] = None, name: Union[str, None] = None) -> None:
    """
    App code
    """
    args = parse_args(args, name)
    logging.basicConfig(
        format=FORMATS[args.log_level], datefmt=DATEFMT, level=args.log_level
    )

    if args.minimize:
        try:
            minimize_databases(args.databases, args.minimize_skip, args.minimize_increasing, args.minimize_trial,
                               xyz_files=args.xyz_files, potential=args.potential, keep_files=args.keep_files,
                               axis=args.axis, charges=args.charges, multipoles=args.multipoles, cutoff=args.cutoff)
        except ValueError:
            sys.exit(1)
    elif args.combine:
        join_databases(args.databases, args.combine_db_name, args.workers, args.combine_serial, "threshutils", args.combine_only_valid)
    elif args.dump:
        dump_databases(args.databases, args.dump_format, args.workers, args.dump_kind)
    elif args.cluster:
        cluster(args.databases, args.cluster_superbasin, args.workers, args.cluster_method)
    elif args.split:
        split_trials(args.databases)


if __name__ == "__main__":
    main()
