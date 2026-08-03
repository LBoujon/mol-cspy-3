from cspy.db import CspDataStore
from cspy.configuration import CspyConfiguration
from cspy.db.clustering_loops import find_duplicates
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.crystal import Crystal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from collections import defaultdict
import json
import logging
import numpy as np
import time
import os
from typing import Tuple, Union, Optional, Any
import sys
from copy import copy


LOG = logging.getLogger(__name__)
PXRD_CHECKPOINT_BATCH_SIZE = 100


def calculate_missing_xrd(
    cid, res_content, backend="auto", return_backend=False
):
    """Calculate PXRD with an explicit backend and report the backend used."""

    backend = backend.lower()
    if backend not in {"auto", "platon", "pymatgen"}:
        raise ValueError(f"Unknown PXRD backend: {backend}")
    LOG.info("trying to calculate missing powder pattern for %s", cid)
    c = Crystal.from_shelx_string(res_content)
    methods = ["platon", "pymatgen"] if backend == "auto" else [backend]
    pp = None
    selected_backend = None
    for method in methods:
        try:
            if method == "platon":
                pp = c.calculate_powder_pattern()
            else:
                pp = c.calculate_powder_pattern(method="pymatgen")
        except Exception as exc:
            LOG.warning("%s powder pattern failed for %s: %s", method, cid, exc)
            pp = None
        if pp is not None:
            selected_backend = method
            break
        if backend == "auto" and method == "platon":
            LOG.warning(
                "falling back to pymatgen for %s; mixed PXRD backends can "
                "require recalibrating clustering thresholds",
                cid,
            )

    i = pp.pattern if pp is not None else None
    LOG.info(
        "powder pattern for %s: %s%s",
        cid,
        "FAILED" if i is None else "SUCCESS",
        "" if selected_backend is None else f" ({selected_backend})",
    )
    return (i, selected_backend) if return_backend else i


def _calculate_missing_xrd_task(task):
    """Worker entry point for parallel, restartable PXRD generation."""

    cid, res_content, backend = task
    try:
        pattern, selected_backend = calculate_missing_xrd(
            cid,
            res_content,
            backend=backend,
            return_backend=True,
        )
    except Exception as exc:
        LOG.warning("PXRD generation failed for %s: %s", cid, exc)
        pattern, selected_backend = None, None
    return cid, pattern, selected_backend


def _pxrd_descriptor_metadata(backend):
    metadata = {
        "two_theta": [0, 20],
        "separation": 0.02,
        "backend": backend,
    }
    if backend == "pymatgen":
        metadata.update({"profile": "lorentzian", "fwhm": 0.05})
    return json.dumps(metadata, sort_keys=True)


def _checkpoint_pxrd_descriptors(ds, pending):
    count = 0
    for backend, values in pending.items():
        if not values:
            continue
        ds.add_descriptors(
            "xrd",
            values,
            metadata=_pxrd_descriptor_metadata(backend),
            replace=True,
        )
        count += len(values)
    pending.clear()
    if count:
        LOG.info("%s: checkpointed %d generated PXRD descriptors", ds.filename, count)


def _generate_missing_xrd_descriptors(
    ds,
    rows,
    descriptors,
    backend,
    jobs,
    batch_size=PXRD_CHECKPOINT_BATCH_SIZE,
):
    """Generate absent PXRD descriptors in parallel and checkpoint batches."""

    tasks = [
        (row[0], row[-1], backend)
        for row in rows
        if row[0] not in descriptors
    ]
    if not tasks:
        return {}

    LOG.info(
        "%s: generating %d missing PXRD descriptors using %d process(es)",
        ds.filename,
        len(tasks),
        jobs,
    )
    if jobs > 1:
        chunksize = max(1, min(50, len(tasks) // (jobs * 8)))
        executor = ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(_calculate_missing_xrd_task, tasks, chunksize=chunksize)
    else:
        executor = None
        results = map(_calculate_missing_xrd_task, tasks)

    generated = {}
    generated_backends = set()
    pending = defaultdict(dict)
    pending_count = 0
    try:
        for cid, pattern, selected_backend in results:
            if pattern is None:
                continue
            generated[cid] = pattern
            generated_backends.add(selected_backend)
            pending[selected_backend][cid] = pattern
            pending_count += 1
            if pending_count >= batch_size:
                _checkpoint_pxrd_descriptors(ds, pending)
                pending_count = 0
    finally:
        _checkpoint_pxrd_descriptors(ds, pending)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if len(generated_backends) > 1:
        LOG.warning(
            "%s: generated PXRD descriptors with multiple backends (%s). "
            "Use --pxrd-backend to keep a homogeneous descriptor set.",
            ds.filename,
            ", ".join(sorted(generated_backends)),
        )
    return generated


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


def structure_rows(
    ds,
    cluster_from_equivalent=False,
    kind="cdtw",
    calculate_missing=True,
    pxrd_backend="auto",
    jobs=1,
):
    LOG.info("%s: loading crystals, descriptors", ds.filename)
    ids = []
    eds = []
    xs = []
#RC
    molids = []
    contents = []
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
        generated_descriptors = {}
        if calculate_missing:
            generated_descriptors = _generate_missing_xrd_descriptors(
                ds,
                rows,
                descriptors,
                backend=pxrd_backend,
                jobs=jobs,
            )
#RC
#        for cid, sg, density, energy, content in rows:
        try:
            for cid, sg, density, energy, molid, content in rows:
                if cid in descriptors:
                    a = np.frombuffer(descriptors[cid])
                    x = a / simps(a, dx=0.02)
                elif cid in generated_descriptors:
                    a = generated_descriptors[cid]
                    x = a / simps(a, dx=0.02)
                else:
                    x = None

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
                elif cid in generated_descriptors:
                    a = generated_descriptors[cid]
                    x = a / simps(a, dx=0.02)
                else:
                    x = None

                if x is None:
                    unique_structures.append(cid)
                else:
                    ids.append(cid)
                    eds.append((energy, density))
                    xs.append(x)
                    molids.append('None')  #RC
                    contents.append(content)

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
        pxrd_backend=getattr(args, "pxrd_backend", "auto"),
        jobs=getattr(args, "jobs", 1),
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
        jobs=getattr(args, "jobs", 1),
    )
    if args.cluster_from_equivalent:
        equivalent_table = read_equivalent_table(ds)

    all_ids = ids
    all_contents = contents
    content_by_id = dict(zip(all_ids, all_contents))
    pattern_content_by_id = content_by_id
    if args.critic2_patterns:
        if args.cluster_from_equivalent:
            pattern_rows = ds.unique_structures(with_file_content=True).fetchall()
        else:
            pattern_rows = ds.final_minimizations(with_file_content=True).fetchall()
        pattern_content_by_id = {row[0]: row[-1] for row in pattern_rows}
    ds.close()

    t2 = time.time()
    LOG.debug("%s: loading data took %.3fs", dbname, t2 - t1)
    LOG.debug(f'Reference structure {args.compack_exp_str}')
    LOG.info("Comparing %d structures to %s", len(ids), args.compack_exp_str)

    cached_rmsds = {}
    if os.path.exists('rmsds.dat'):
        LOG.info('Restarting structure search from rmsds.dat file.')
        existing_rmsds = {}
        with open('rmsds.dat', 'r') as f:
            for line in f.readlines():
                parts = line.split()
                id1, id2, rmsd_ = parts
                existing_rmsds[tuple(sorted([id1, id2]))] = float(rmsd_)
        LOG.info(f'{len(existing_rmsds.keys())} structure pairs were read from rmsds.dat file.')
        LOG.debug(f'number of comparison structures before pruning: {len(all_contents)}')
        pending = []
        for i, id_ in enumerate(all_ids):
            key = tuple(sorted([args.compack_exp_str, id_]))
            if key in existing_rmsds:
                cached_rmsds[id_] = existing_rmsds[key]
            else:
                pending.append(i)
        ids = [all_ids[i] for i in pending]
        contents = [all_contents[i] for i in pending]
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
    time1 = time.time()
    rmsds = {}
    if ids:
        config = CspyConfiguration()
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
            raise NotImplementedError(
                f"Method {args.method} is not implemented for structure search: "
            )

    structural_scores = cached_rmsds.copy()
    structural_scores.update({ids[idx]: rmsd for idx, rmsd in rmsds.items()})
    matches = {
        structure_id: rmsd
        for structure_id, rmsd in structural_scores.items()
        if rmsd < args.cluster_rms_threshold
    }
    with open('rmsd_matches.txt', 'w') as f:
        for structure_id, rmsd in sorted(matches.items(), key=lambda item: item[1]):
            f.write(f'{structure_id} {rmsd}\n')

    if args.critic2_patterns:
        from cspy.db.critic2_patterns import (
            compare_patterns,
            read_pattern_comparisons,
            write_pattern_comparisons,
        )

        pattern_candidates = [
            (structure_id, rmsd, pattern_content_by_id[structure_id])
            for structure_id, rmsd in matches.items()
        ]
        pattern_reference = exp_cif_string
        reference_suffix = ".cif"
        if os.path.isfile(args.compack_exp_str):
            with open(args.compack_exp_str, encoding="utf-8") as handle:
                pattern_reference = handle.read()
            reference_suffix = os.path.splitext(args.compack_exp_str)[1] or ".cif"
        critic2_config = CspyConfiguration().get("critic2", {})
        critic2_timeout = (
            args.critic2_timeout
            if args.critic2_timeout is not None
            else float(critic2_config.get("timeout", 300.0))
        )
        previous = []
        if args.critic2_resume:
            previous = read_pattern_comparisons(args.critic2_output)
            if previous:
                LOG.info(
                    "Resuming critic2 comparisons from %s (%d rows)",
                    args.critic2_output,
                    len(previous),
                )
        comparisons = compare_patterns(
            pattern_reference,
            pattern_candidates,
            jobs=args.jobs,
            executable=args.critic2_executable,
            timeout=critic2_timeout,
            reference_suffix=reference_suffix,
            previous=previous,
            retry_errors=args.critic2_retry_errors,
            fail_fast=args.critic2_fail_fast,
            checkpoint=args.critic2_output,
        )
        write_pattern_comparisons(comparisons, args.critic2_output)
        failures = sum(row.status != "ok" for row in comparisons)
        if failures:
            LOG.warning(
                "%d critic2 comparisons failed; details are recorded in %s",
                failures,
                args.critic2_output,
            )
    LOG.info(
        "Found %d equivalent structures to %s in %s seconds",
        len(matches), args.compack_exp_str, time.time()-time1
    )




def compack_compare_db(dbname:str,args:dict[str, Any])-> None:
    """ search for matches between the supplied database and another database.

    Parameters
    ----------
    dbname : str
        CSPy database name/path

    args: dict[str, Any]
          The arguments passed to cspy-db cluster
 
    """

    from cspy.db.compack_clustering import iterative_compack_batch

    eng_thresh = args.cluster_energy_threshold
    den_thresh = args.cluster_density_threshold

    ds=CspDataStore(dbname)
    comp_ds=CspDataStore(args.compack_compdbname)

    ids, eds, xs, molids, contents, no_desc = structure_rows(ds,kind='compack')
    comp_ids, comp_eds, comp_xs, comp_molids, comp_contents, comp_no_desc = structure_rows(comp_ds,kind='compack')
    config = CspyConfiguration()
    settings = config['compack']
    for i in range(len(ids)):
        ed = eds[i]
        comp_ed_a = np.abs(np.array(comp_eds) - ed)
        idxs = np.where(
            (comp_ed_a[:, 0] < eng_thresh)
            & (comp_ed_a[:, 1] < den_thresh)
        )[0]

        refname=ids[i]
        rmsds = iterative_compack_batch(contents[i],
            [comp_contents[j] for j in idxs],
            refname,comp_ids,
            args.jobs,
            settings)

        matches = {j:rmsd for j,rmsd in rmsds.items() if rmsd < args.cluster_rms_threshold}
        with open('comp_db_matches.txt', 'a') as f:
             for j, rmsd in matches.items():
                 f.write(f'{refname} {comp_ids[idxs[j]]} {rmsd}\n')


def write_completeness_report(args) -> None:
    """Write completeness estimates after duplicate-removal clustering."""

    from cspy.db.completeness import (
        combine_database_observations,
        convergence,
        summarize,
        write_convergence,
        write_summary,
    )

    global_clusters = args.output if len(args.databases) > 1 else None
    observations = combine_database_observations(
        args.databases,
        global_clusters=global_clusters,
        strict_assignments=not args.completeness_allow_incomplete_clusters,
    )
    rows = summarize(
        observations,
        windows=args.completeness_windows,
        by_spacegroup=not args.completeness_total_only,
        bootstrap_samples=args.completeness_bootstrap_samples,
        confidence_level=args.completeness_confidence_level,
        random_seed=args.completeness_random_seed,
    )
    with open(
        args.completeness_output, "w", newline="", encoding="utf-8"
    ) as handle:
        write_summary(rows, handle)
    LOG.info("Wrote completeness report to %s", args.completeness_output)
    if args.completeness_convergence_output:
        curve = convergence(
            observations,
            windows=args.completeness_windows,
            points=args.completeness_convergence_points,
            by_spacegroup=not args.completeness_total_only,
        )
        with open(
            args.completeness_convergence_output,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            write_convergence(curve, handle)
        LOG.info(
            "Wrote completeness convergence curve to %s",
            args.completeness_convergence_output,
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
        "--pxrd-backend",
        choices=("auto", "platon", "pymatgen"),
        default="auto",
        help=(
            "backend for missing PXRD descriptors; choose an explicit backend "
            "to keep descriptors homogeneous (default: auto)"
        ),
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
    parser.add_argument(
        "--critic2-patterns",
        action="store_true",
        help=(
            "after structural matching, compare GPWDF and GVCPWDF patterns with "
            "critic2 only for structures below --cluster-rms-threshold"
        ),
    )
    parser.add_argument(
        "--critic2-executable",
        type=str,
        default=None,
        help=(
            "critic2 executable name or path; otherwise use "
            "CSPY_CRITIC2_EXECUTABLE or resolve critic2 from the job PATH "
            "after conda activation/module loading"
        ),
    )
    parser.add_argument(
        "--critic2-output",
        type=str,
        default="critic2_pattern_matches.csv",
        help=(
            "CSV for structural and critic2 pattern scores "
            "(default: critic2_pattern_matches.csv)"
        ),
    )
    parser.add_argument(
        "--critic2-timeout",
        type=float,
        default=None,
        help="timeout in seconds for each comparison (configured default: 300)",
    )
    parser.add_argument(
        "--critic2-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse successful rows already present in --critic2-output",
    )
    parser.add_argument(
        "--critic2-retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retry checkpoint rows whose status is error",
    )
    parser.add_argument(
        "--critic2-fail-fast",
        action="store_true",
        help="abort the pattern run when one candidate fails",
    )

    parser.add_argument(
        "--compack_compdbname",
        type=str,
        default=None,
        help="File name of a comparison database"
    )
    parser.add_argument(
        "--completeness",
        action="store_true",
        help=(
            "calculate Good--Turing/Chao estimates after clustering and write "
            "them to a CSV file"
        ),
    )
    parser.add_argument(
        "--completeness-windows",
        nargs="*",
        type=float,
        default=(5.0, 10.0, 15.0),
        metavar="KJ_MOL",
        help="energy windows for --completeness (default: 5 10 15)",
    )
    parser.add_argument(
        "--completeness-total-only",
        action="store_true",
        help="omit per-space-group rows from the completeness report",
    )
    parser.add_argument(
        "--completeness-output",
        type=str,
        default="completeness.csv",
        help="CSV written by --completeness (default: completeness.csv)",
    )
    parser.add_argument(
        "--completeness-bootstrap-samples",
        type=int,
        default=1000,
        help="bootstrap resamples for completeness intervals (default: 1000)",
    )
    parser.add_argument(
        "--completeness-confidence-level",
        type=float,
        default=0.95,
        help="confidence level for completeness intervals (default: 0.95)",
    )
    parser.add_argument(
        "--completeness-random-seed",
        type=int,
        default=0,
        help="random seed for completeness intervals (default: 0)",
    )
    parser.add_argument(
        "--completeness-allow-incomplete-clusters",
        action="store_true",
        help="treat final structures absent from equivalent_to as singletons",
    )
    parser.add_argument(
        "--completeness-convergence-output",
        type=str,
        help="CSV for the trial-ordered completeness curve",
    )
    parser.add_argument(
        "--completeness-convergence-points",
        type=int,
        default=20,
        help="number of convergence checkpoints (default: 20)",
    )


    args = parser.parse_args(sys_args)
    if args.jobs < 1:
        parser.error("--jobs must be greater than zero")
    if args.critic2_patterns and not args.compack_exp_str:
        parser.error("--critic2-patterns requires --compack_exp_str")
    if args.critic2_timeout is not None and args.critic2_timeout <= 0:
        parser.error("--critic2-timeout must be greater than zero")
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s'
    )

    LOG.info("%d databases to process", len(args.databases))

    calculate_missing = False if args.skip_calculate_missing_pxrd else True
    n_uniques = 0

    # Pair matching and missing-PXRD generation parallelise over structures
    # within one database. Process databases sequentially to avoid creating
    # args.jobs ** 2 workers.
    if not args.compack_exp_str and not args.compack_compdbname:
        LOG.info("Creating output database: %s", args.output)
        output_db = CspDataStore(args.output)
        output_db.close()
        LOG.info(f'Task: Duplicate removal (clustering).')
        if args.method == 'pymatgen':
            config = CspyConfiguration()
            pymatgen_settings = config.get('structurematcher', {})
            LOG.info('Following pymatgen StructureMatcher settings are overridden:')
            LOG.info(f'{pymatgen_settings}')

        executor = None
        if args.method in ("cdtw_cos", "cdtw", "cos") and len(args.databases) > 1:
            # Preserve the existing database-level parallelism for multi-SG
            # inputs without starting a full process pool inside every thread.
            per_database_args = copy(args)
            per_database_args.jobs = 1
            executor = ThreadPoolExecutor(max_workers=args.jobs)
            futures = [
                executor.submit(
                    find_equivalent_structures,
                    dbname,
                    per_database_args,
                    calculate_missing=calculate_missing,
                )
                for dbname in args.databases
            ]
            results = (future.result() for future in as_completed(futures))
        else:
            results = (
                find_equivalent_structures(
                    dbname,
                    args,
                    calculate_missing=calculate_missing,
                )
                for dbname in args.databases
            )

        try:
            for inputdbname, duplicates in results:
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
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        LOG.info("Found %d unique structures in total", n_uniques)
        if len(args.databases) == 1:
            if args.completeness:
                write_completeness_report(args)
            return

        LOG.info("Clustering output database %s. Unique structures will not be copied anywhere new.", args.output)
        find_equivalent_structures(args.output, args, calculate_missing=False)
        if args.completeness:
            write_completeness_report(args)
    elif args.compack_exp_str:
        LOG.info(f'Task: Finding structure match(es) to {args.compack_exp_str}')
        if args.method not in ['compack', 'pymatgen']:
            raise NotImplementedError(f"Method {args.method} is not implemented for "
                            "structure search. Use either 'compack' or 'pymatgen'.")
        LOG.debug(f'Searching for matches to {args.compack_exp_str}.')
        for db in args.databases:
            structure_search(dbname=db,
                             args=args,
                            )

    elif args.compack_compdbname:
        if args.method != 'compack':
            raise NotImplementedError(f"Method {args.method} is not implemented for "
                            "comparison of databases. Use 'compack'")
        LOG.debug(f'Searching for matches between {args.databases} and {args.compack_compdbname}.')
        for db in args.databases:
            compack_compare_db(dbname=db,args=args)


if __name__ == "__main__":
    main()
