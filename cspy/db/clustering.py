from cspy.db import CspDataStore
from cspy.configuration import CspyConfiguration
from cspy.db.clustering_loops import find_duplicates
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.crystal import Crystal
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from collections import defaultdict
import logging
import numpy as np
import time
import os
from typing import Tuple, Union, Optional
import sys

LOG = logging.getLogger(__name__)


def calculate_missing_xrd(cid, res_content):
    LOG.info("trying to calculate missing powder pattern for %s", cid)
    c = Crystal.from_shelx_string(res_content)
    pp = c.calculate_powder_pattern()
    i = pp.pattern if pp is not None else None
    LOG.info("powder pattern for %s: %s", cid, "FAILED" if i is None else "SUCCESS")
    return i


def read_equivalent_table(ds):
    """Read existing equivalent_table from previous clustering"""
    equivalent_table = defaultdict(list)
    for unique_id, equivalent_id in ds.query(
        "select unique_id, equivalent_id from equivalent_to"
    ):
        equivalent_table[unique_id].append(equivalent_id)
    if not equivalent_table:
        LOG.warning("%s: no existing equivalent table")
    return equivalent_table


def structure_rows(ds, cluster_from_equivalent=False, kind="cdtw", calculate_missing=True):
    LOG.info("%s: loading crystals, descriptors", ds.filename)
    ids = []
    eds = []
    xs = []
#RC
    molids = []
    contents = []
    missing_descriptors = {}
    unique_structures = []
    if cluster_from_equivalent:
        rows = ds.unique_structures(with_file_content=True).fetchall()
        descriptors = dict(ds.query(
            "select distinct(id), value from descriptor join equivalent_to on "
            "descriptor.id==equivalent_to.unique_id where name='xrd'"
        ))
    else:
        rows = ds.final_minimizations().fetchall()
        descriptors = dict(ds.query("select id, value from descriptor where name='xrd'"))
    LOG.debug("%s: loaded %d descriptors", ds.filename, len(descriptors))

    if kind == "compack" or kind == "pymatgen":
        from cspy.db.compack_clustering import res_to_cif 
#RC
#        for cid, sg, density, energy, content in rows:
        try:
            for cid, sg, density, energy, molid, content in rows:
                ids.append(cid)
                eds.append((energy, density))
                molids.append(molid)  #RC
                contents.append(res_to_cif(content))
        except:
            for cid, sg, density, energy, content in rows:
                ids.append(cid)
                eds.append((energy, density))
                molids.append('None')  #RC
                contents.append(res_to_cif(content))

    elif kind in ("cdtw_cos", "cdtw", "cos"):
#RC
#        for cid, sg, density, energy, content in rows:
        try:
            for cid, sg, density, energy, molid, content in rows:
                if cid in descriptors:
                    a = np.frombuffer(descriptors[cid])
                    x = a / simps(a, dx=0.02)
                else:
                    x = calculate_missing_xrd(cid, content) if calculate_missing else None
                    if x is not None:
                        missing_descriptors[cid] = x
                        x = x / simps(x, dx=0.02)

                if x is None:
                    unique_structures.append(cid)
                else:
                    ids.append(cid)
                    eds.append((energy, density))
                    xs.append(x)
                    molids.append(molid)  #RC
                    contents.append(content)
        except:
            for cid, sg, density, energy, content in rows:
                if cid in descriptors:
                    a = np.frombuffer(descriptors[cid])
                    x = a / simps(a, dx=0.02)
                else:
                    x = calculate_missing_xrd(cid, content) if calculate_missing else None
                    if x is not None:
                        missing_descriptors[cid] = x
                        x = x / simps(x, dx=0.02)

                if x is None:
                    unique_structures.append(cid)
                else:
                    ids.append(cid)
                    eds.append((energy, density))
                    xs.append(x)
                    molids.append('None')  #RC
                    contents.append(content)

        if missing_descriptors:
            ds.add_descriptors("xrd", missing_descriptors,
                               metadata="{'two_theta': (0, 20), 'sep': 0.02}")
    else:
        raise NotImplementedError(
            f"Removing duplicates not supported for method='{kind}'"
        )

    if len(ids) == 0:
        LOG.warning("%s: no structures retrieved")
    LOG.debug(
        "%s: retrieved %d structures, %d had descriptors",
        ds.filename,
        len(ids),
        len(descriptors),
    )
    return (
        ids,
        np.array(eds, dtype=np.float64),
        np.asarray(xs, dtype=np.float64),
        molids, #RC
        contents,
        unique_structures,
    )



def find_duplicates_powder(dbname, ids, eds, xs, no_desc, args):
    nstructs = len(ids)
    LOG.info(
        "%s: finding unique structures by '%s', N=%d", dbname, args.method, nstructs
    )
    t1 = time.time()
    duplicates_array = find_duplicates(
        eds,
        xs,
        method=args.method,
        ethresh=args.cluster_energy_threshold,
        dthresh=args.cluster_density_threshold,
        cdtw_thresh=args.cluster_xrdcdtw_threshold,
        cos_thresh=args.cluster_xrdcos_threshold
    )
    t2 = time.time()
    LOG.debug("%s: clustering took %.3fs", dbname, t2 - t1)
    duplicates = {}
    for i, n in enumerate(duplicates_array):
        if n == -1:
            duplicates[i] = []
        else:
            duplicates[n].append(i)
    duplicates = {ids[k]: [ids[x] for x in v] for k, v in duplicates.items()}

    if args.method == "cdtw":
        process = "e->d->cdtw"
    elif args.method == "cdtw_cos":
        process = "e->d->cos->cdtw"
    elif args.method == "cos":
        process = "e->d->cos"
    LOG.info(
        "%s: %d unique structures from %s", dbname, len(duplicates), process
    )

    for cid in no_desc:
        duplicates[cid] = []
    if len(no_desc) > 0:
        LOG.info("%s: keeping %d stuctures w/o descriptors", dbname, len(no_desc))
    return duplicates


def find_duplicates_compack(dbname, ids, eds, molids, contents, settings, args):
    from cspy.db.compack_clustering import iterative_compack_batch, iterative_pymatgen_batch

    duplicates = {}
    LOG.info("%s: clustering %d structures using compack", dbname, len(ids))

    existing_rmsds = {}
    if os.path.exists('rmsds.dat'):
        LOG.info('Restarting duplicate removal from rmsds.dat file.')
        with open('rmsds.dat', 'r') as f:
            for line in f.readlines():
                parts = line.split()
                id1, id2, rmsd_ = parts
                existing_rmsds[tuple(sorted([id1, id2]))] = float(rmsd_)
        LOG.info(f'{len(existing_rmsds.keys())} structure pairs were read from rmsds.dat file.')

    equivalent_to = np.ones(len(ids), dtype=int) * -1
    for i in range(len(ids)):
        LOG.debug("Reference structure %s (%d)", ids[i], i)
        if equivalent_to[i] > -1:
            LOG.debug(
                "%s (%d) == %s (%d), skipping",
                ids[i],
                i,
                ids[equivalent_to[i]],
                equivalent_to[i],
            )
            continue
        ed = eds[i]
        ed_a = np.abs(np.array(eds) - ed)
        idxs = np.where(
            (ed_a[:, 0] < args.cluster_energy_threshold)
            & (ed_a[:, 1] < args.cluster_density_threshold)
            & (equivalent_to < 0)
        )[0]
        idxs = idxs[idxs > i]
        if len(idxs) == 0:
            LOG.debug(
                "%s (%d) had no structures in energy/density window for comparison",
                ids[i],
                i,
            )
            duplicates[i] = []
            continue

        # Removing those idxs that already compared to the reference structure.
        to_remove = []
        precalculated_rmsds = {}
        for k, idx in enumerate(idxs):
            if tuple(sorted([ids[i], ids[idx]])) in existing_rmsds.keys():
                precalculated_rmsds[idx] = existing_rmsds[tuple(sorted([ids[i], ids[idx]]))]
                to_remove.append(k)
        if to_remove:
            LOG.info(f'Found {len(to_remove)} pairs in rmsds.dat')
            for idx in to_remove[::-1]:
                idxs = np.delete(idxs, idx)

        LOG.info("Comparing %d structures to %s", len(idxs), ids[i])
        if args.method == 'compack':
            rmsds = iterative_compack_batch(
                contents[i],
                [contents[j] for j in idxs],
                ids[i],
                [ids[j] for j in idxs],
                args.jobs,
                settings,
            )
        elif args.method == 'pymatgen':
            rmsds = iterative_pymatgen_batch(
                contents[i],
                [contents[j] for j in idxs],
                ids[i],
                [ids[j] for j in idxs],
                args.jobs,
            )
        LOG.debug("RMSD: %s", rmsds)
        dups = [idxs[j] for j, v in rmsds.items() if v < args.cluster_rms_threshold]
        dups += [idx for idx, v in precalculated_rmsds.items() if v < args.cluster_rms_threshold]

        # updatign existing_rmsds dictionary
        for j, v in rmsds.items():
            existing_rmsds[tuple(sorted([ids[i], ids[j]]))] = v

        if len(dups) > 0:
            equivalent_to[dups] = i
        LOG.info("Found %d equivalent structures to %s", len(dups), ids[i])
        duplicates[i] = dups
        LOG.info(
            "%d structures remaining", len(np.where(equivalent_to[i + 1:] < 0)[0])
        )

    LOG.info("%s: %d unique structures from compack", dbname, len(duplicates))
    duplicates = {ids[k]: [ids[x] for x in v] for k, v in duplicates.items()}
    return duplicates


def copy_to_db(inputdbname : str, outputdbname : str, duplicates : list):
    """ Check if the input and output databases have the same name.
    If they don't, write the unique structures to a new (unless it already exists) 
    database of name outputdbname.

    Parameters
    ----------
    inputdbname : str
        CSPy input database name/path
    outputdbname : str
        CSPy output database name/path
    duplicates : list
        List of duplicate structures
    
    """
    if inputdbname != outputdbname:
        ds = CspDataStore(inputdbname)
        LOG.info(
            "%s: copying %d structures to %s", inputdbname, len(duplicates), outputdbname
        )
        ds.copy_unique_structures_to(outputdbname)
        ds.close()
    else:
        LOG.info(
            "Input database name (%s) is the same as output database name (%s). Unique structures have not been copied anywhere new.", inputdbname, outputdbname
        )


def find_equivalent_structures(dbname : str, 
                               args : dict, 
                               calculate_missing: bool = True) -> Tuple[str, list, str]:
    """ Read database, iterate over structures, and find which are equivalent.

    Parameters
    ----------
    dbname : str
        CSPy database name/path
    calculate_missing : bool
        Whether or not to calculate PXRD patterns where missing
    
    Returns
    -------
    num_equivalents : dict()
        keys are ids of structures 
        values are the number of equivalent structures
    dbname : str
        CSPy database name/path
    duplicates : list
        List of duplicate structures
    
    """
    ds = CspDataStore(dbname)
    t1 = time.time()

    rows = structure_rows(
        ds, 
        cluster_from_equivalent=args.cluster_from_equivalent, 
        kind=args.method, 
        calculate_missing=calculate_missing,
    )
    if len(rows) == 5:
        LOG.error("Input database has only 5 columns. It may use the old schema and need converting. Quitting...")
        sys.exit()
    ids, eds, xs, molids, contents, no_desc = rows
    if args.cluster_from_equivalent:
        equivalent_table = read_equivalent_table(ds) 
    t2 = time.time()
    LOG.debug("%s: loading data took %.3fs", dbname, t2 - t1)

    if args.method == "compack" or args.method == "pymatgen":
        config = CspyConfiguration()
        compack_settings = config.get('compack', {})
        duplicates = find_duplicates_compack(dbname, ids, eds, molids, contents, compack_settings, args)
    elif args.method in ("cdtw_cos", "cdtw", "cos"):
        duplicates = find_duplicates_powder(dbname, ids, eds, xs, no_desc, args)
    else:
        raise NotImplementedError(
            f"Removing duplicates not supported for method='{args.method}'"
        )
    LOG.info("%s: %d unique structures total", dbname, len(duplicates))

    if args.cluster_from_equivalent:
        for unique_id, equivalent_ids in duplicates.items():
            equivalent_table[unique_id] += equivalent_ids
            for equivalent_id in equivalent_ids:
                equivalent_table[unique_id] += equivalent_table[equivalent_id]
                equivalent_table.pop(equivalent_id)
    else:
        equivalent_table = duplicates

    ds.add_equivalent_structures(equivalent_table)
    ds.commit()
    ds.close()
    return dbname, duplicates


def structure_search(dbname, args):
    from cspy.db.compack_clustering import iterative_compack_batch, iterative_pymatgen_batch
    ds = CspDataStore(dbname)
    t1 = time.time()
    ids, eds, xs, molids, contents, no_desc = structure_rows(
        ds,
        cluster_from_equivalent=args.cluster_from_equivalent,
        kind=args.method,
    )
    if args.cluster_from_equivalent:
        equivalent_table = read_equivalent_table(ds)

    t2 = time.time()
    LOG.debug("%s: loading data took %.3fs", dbname, t2 - t1)
    LOG.debug(f'Reference structure {args.compack_exp_str}')
    LOG.info("Comparing %d structures to %s", len(ids), args.compack_exp_str)

    if os.path.exists('rmsds.dat'):
        LOG.info('Restarting structure search from rmsds.dat file.')
        existing_rmsds = {}
        with open('rmsds.dat', 'r') as f:
            for line in f.readlines():
                parts = line.split()
                id1, id2, rmsd_ = parts
                existing_rmsds[tuple(sorted([id1, id2]))] = float(rmsd_)
        LOG.info(f'{len(existing_rmsds.keys())} structure pairs were read from rmsds.dat file.')
        LOG.debug(f'number of comparison structures before pruning: {len(contents)}')
        to_remove = []
        for i, id_ in enumerate(ids):
            if tuple(sorted([args.compack_exp_str, id_])) in existing_rmsds.keys():
                to_remove.append(i)
        for i in to_remove[::-1]:
            ids.pop(i)
            contents.pop(i)
        LOG.info(f'Number of comparison structures after pruning: {len(contents)}')

    try:
        from ccdc.io import CrystalReader
        csd = CrystalReader('csd')
        crystal = csd.crystal(args.compack_exp_str)
        exp_cif_string = crystal.to_string('cif')
        LOG.info(f"Reference structure {args.compack_exp_str} located in csd database")
    except (ModuleNotFoundError, RuntimeError, NameError):
        try:
            exp_cif_string = Crystal.load(args.compack_exp_str).to_cif_string()
        except ValueError:
            from pymatgen.core.structure import Structure
            crystal = Structure.from_file(args.compack_exp_str)
            exp_cif_string = crystal.to(fmt='cif')
        LOG.info(f"Reference structure {args.compack_exp_str} read from file")
    config = CspyConfiguration()
    time1 = time.time()
    if args.method == 'compack':
        settings = config['compack']
        rmsds = iterative_compack_batch(
            exp_cif_string,
            contents,
            args.compack_exp_str,
            ids,
            args.jobs,
            settings,
        )
    elif args.method == 'pymatgen':
        pymatgen_settings = config.get('structurematcher', {})
        LOG.info('Following pymatgen StructureMatcher settings are overridden:')
        LOG.info(f'{pymatgen_settings}')
        rmsds = iterative_pymatgen_batch(
                exp_cif_string,
                contents,
                args.compack_exp_str,
                ids,
                args.jobs,
            )
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented for structure search: ")
        
    matches = {idx:rmsd for idx, rmsd in rmsds.items() if rmsd < args.cluster_rms_threshold}
    with open('rmsd_matches.txt', 'w') as f:
        for idx, rmsd in matches.items():
            f.write(f'{ids[idx]} {rmsd}\n')
    LOG.info(
        "Found %d equivalent structures to %s in %s seconds",
        len(matches), args.compack_exp_str, time.time()-time1
    )


def main(sys_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "databases",
        nargs="+",
        type=str,
        help="Databases to process given space separated."
    )
    # Clustering parameters
    parser.add_argument(
        "-cfe",
        "--cluster-from-equivalent",
        action="store_true",
        default=False,
        help="Cluster the unique structures from existing equivalent_table",
    )
    parser.add_argument(
        "-cdt",
        "--cluster-density-threshold",
        type=float,
        help="The density threshold used in clustering, "
        "within which structures are considered the "
        "same.",
        default=0.05
    )
    parser.add_argument(
        "-cet",
        "--cluster-energy-threshold",
        type=float,
        help="The energy threshold used in clustering, "
        "within which structures are considered the "
        "same.",
        default=1.0
    )
    parser.add_argument(
        "-ccdtwt",
        "--cluster-xrdcdtw-threshold",
        type=float,
        help="The difference in XRD patterns as "
        "measured by the constrained dynamic "
        "time warping method, within which "
        "structures are considered the same.",
        default=10.00
    )
    parser.add_argument(
        "-ccost",
        "--cluster-xrdcos-threshold",
        type=float,
        help="The similarity in XRD patterns as "
        "measured by a cosine difference within which "
        "structures are considered the same.",
        default=0.80
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="Number of parallel processes/threads to use for "
        "xrd/compack/pymatgen clustering",
        default=1
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("INFO", "DEBUG", "ERROR", "WARN"),
        default="INFO",
        help="Control level of logging output"
    )
    parser.add_argument(
        "-rms",
        "--cluster-rms-threshold",
        type=float,
        help="RMS difference threshold used in " "compack clustering.",
        default=0.3
    )
    parser.add_argument(
        "--compack-timeout",
        type=float,
        help="Timeout for compack clustering in seconds.",
        default=30.0
    )
    parser.add_argument(
        "--skip-calculate-missing-pxrd",
        action="store_true",
        help="Skip the calculation of missing pxrd patterns.",
        default=False
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default="output.db", 
        help="Database for output"
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="cdtw_cos",
        choices=("cdtw_cos", "cdtw", "cos", "compack", "pymatgen"),
        help="How to perform clustering"
    )
    parser.add_argument(
        "--compack_exp_str",
        type=str,
        default=None,
        help="File name containing the experimental crystal structure."
             "Alternatively, specify the CCDC reference code."
             "e.g. name.cif, name.res, ACETAC01"
        )


    args = parser.parse_args(sys_args)
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s'
    )

    LOG.info("%d databases to process", len(args.databases))

    LOG.info("Creating output database: %s", args.output)
    output_db = CspDataStore(args.output)
    output_db.close()

    calculate_missing = False if args.skip_calculate_missing_pxrd else True
    n_uniques = 0

    # compack calculation likely going to be run on single database, therefore parallelise
    # over structures rather than databases. This is done by setting job_pool to 1 and then
    # using args.jobs to set the number of parellel compack calculations 
    # in iterative_compack_batch
    if not args.compack_exp_str:
        LOG.info(f'Task: Duplicate removal (clustering).')
        if args.method =='compack':
            job_pool = 1
        elif args.method == 'pymatgen':
            job_pool = 1
            config = CspyConfiguration()
            pymatgen_settings = config.get('structurematcher', {})
            LOG.info('Following pymatgen StructureMatcher settings are overridden:')
            LOG.info(f'{pymatgen_settings}')
        else:
            job_pool = args.jobs
        with ThreadPoolExecutor(job_pool) as e:
            futures = [
                e.submit(
                    find_equivalent_structures, 
                    dbname, 
                    args, 
                    calculate_missing=calculate_missing
                )
                for dbname in args.databases
            ]

            for future in as_completed(futures):
                inputdbname, duplicates = future.result()
                outputdbname = args.output
                if inputdbname != outputdbname:
                    ds = CspDataStore(inputdbname)
                    LOG.info(
                        "%s: copying %d structures to %s", inputdbname, len(duplicates), outputdbname
                    )
                    ds.copy_unique_structures_to(outputdbname)
                    ds.close()
                else:
                    LOG.info(
                        "Input database name (%s) is the same as output database name (%s). " \
                        "Unique structures have not been copied anywhere new.", inputdbname, outputdbname
                    )
                n_uniques += len(duplicates)

        LOG.info("Found %d unique structures in total", n_uniques)
        if len(args.databases) == 1:
            return

        LOG.info("Clustering output database %s. Unique structures will not be copied anywhere new.", args.output)
        find_equivalent_structures(args.output, args, calculate_missing=False)
    else:
        LOG.info(f'Task: Finding structure match(es) to {args.compack_exp_str}')
        if args.method not in ['compack', 'pymatgen']:
            raise NotImplementedError(f"Method {args.method} is not implemented for "
                            "structure search. Use either 'compack' or 'pymatgen'.")
        LOG.debug(f'Searching for matches to {args.compack_exp_str}.')
        for db in args.databases:
            structure_search(dbname=db,
                             args=args,
                            )


if __name__ == "__main__":
    main()
