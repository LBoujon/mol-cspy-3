import logging
import numpy as np
import time
import logging
from cspy.crystal import Crystal as cspy_Crystal
#from cspy.datastore import CspDataStore
from cspy.db.datastore import CspDataStore
from multiprocessing import Pool
from collections import defaultdict
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from cspy.configuration import CspyConfiguration


LOG = logging.getLogger(__name__)

try:
    from ccdc.crystal import PackingSimilarity
    from ccdc.crystal import Crystal as ccdc_Crystal
except ImportError:
    import sys
    LOG.error("COMPACK clustering is not available because the csd-pyhon-api package is not installed. "
              "Follow installation instructions here: https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html")
    sys.exit(1)


CONFIG = CspyConfiguration()


def res_to_cif(res_string):
    """Change res string to cif string since ccdc could not read res string. If
    ccdc is installed in another conda environment rather than cspy, this part
    need to be changed"""
    c = cspy_Crystal.from_shelx_string(res_string)
    return c.to_cif_string()

def pymatgen_structurematcher(reference_structure,
                    comparison_structures,
                    ref_id,
                    comp_ids,
                    start_batch=0,
                    **kwargs
                   ):
    """
     Calculate RMS displacement between two structures using pymatgen's StructureMatcher.

    Args:
        reference_structure (str): file content of the reference structure's CIF file
        comparison_structures (str): file content of the comparison structures' CIF files

    StructureMatcher arguments (set in configuration file):
        ltol (float): Fractional length tolerance. Default is 0.2.
        stol (float): Site tolerance. Defined as the fraction of the
            average free length per atom := ( V / Nsites ) ** (1/3)
            Default is 0.3.
        angle_tol (float): Angle tolerance in degrees. Default is 5 degrees.
        primitive_cell (bool): If true: input structures will be reduced to
            primitive cells prior to matching. Default to True.
        scale (bool): Input structures are scaled to equivalent volume if
        true; For exact matching, set to False.
        attempt_supercell (bool): If set to True and number of sites in
            cells differ after a primitive cell reduction (divisible by an
            integer) attempts to generate a supercell transformation of the
            smaller cell which is equivalent to the larger structure.
        allow_subset (bool): Allow one structure to match to the subset of
            another structure. Eg. Matching of an ordered structure onto a
            disordered one, or matching a delithiated to a lithiated
            structure. This option cannot be combined with
            attempt_supercell, or with structure grouping.
        comparator (Comparator): A comparator object implementing an equals
            method that declares equivalency of sites. Default is
            SpeciesComparator, which implies rigid species
            mapping, i.e., Fe2+ only matches Fe2+ and not Fe3+.
        
            Other comparators are provided, e.g. ElementComparator which
            matches only the elements and not the species.

            The reason why a comparator object is used instead of
            supplying a comparison function is that it is not possible to
            pickle a function, which makes it otherwise difficult to use
            StructureMatcher with Python's multiprocessing.
        supercell_size (str or list): Method to use for determining the
            size of a supercell (if applicable). Possible values are
            'num_sites', 'num_atoms', 'volume', or an element or list of elements
            present in both structures.
        ignored_species (list): A list of ions to be ignored in matching.
            Useful for matching structures that have similar frameworks
            except for certain ions, e.g. Li-ion intercalation frameworks.
            This is more useful than allow_subset because it allows better
            control over what species are ignored in the matching.
    
    Returns: 
        rms displacement normalized by (Vol / nsites) ** (1/3)
        and maximum distance between paired sites. If no matching
        lattice is found None is returned.
    """
    pymatgen_settings = CONFIG.get('structurematcher', {})
    sm = StructureMatcher(**pymatgen_settings)
    LOG.debug(f'pymatgen.StructureMatcher settings are: {sm.as_dict()}')
    BIG_RMS = 1e10

    results = []
    reference = Structure.from_str(reference_structure, fmt='cif')
    for j, comparison in enumerate(comparison_structures, start=start_batch):
        c_j = Structure.from_str(comparison, fmt='cif')
        out_ = sm.get_rms_dist(reference, c_j)
        
        if out_ is None:
            results.append((j, BIG_RMS))
        else:
            rmsd_, max_dist_ = out_
            results.append((j, rmsd_))
    with open('rmsds.dat', 'a') as f:
        for result in results:
            j, rmsd = result
            if rmsd < 1e3:
                LOG.info(f'Overlay of {ref_id}-{comp_ids[j]} has RMSD = {rmsd}')
            f.write(f'{ref_id} {comp_ids[j]} {rmsd}\n')

    return results

def mercury_compack(reference_structure,
                    comparison_structures,
                    ref_id,
                    comp_ids,
                    settings,
                    start_batch=0,
                   ):
    """Compare a list of structures in comparison_structures to single
    reference_struture"""
    similarity_engine = PackingSimilarity()
    nmolecules = settings['packing_shell_size']
    for k, v in settings.items():
        setattr(similarity_engine.settings, k, v)

    BIG_RMS = 1e10

    results = []
    c_i = ccdc_Crystal.from_string(reference_structure, format='cif')
    for j, comparison in enumerate(comparison_structures, start=start_batch):
        n = 6
        c_j = ccdc_Crystal.from_string(comparison, format='cif')

        #Number of nmolecules to be matched is iterated from small to large
        #to rule out pairs structures with few number of molecules matched. This
        #increase the comparison efficiency by a lot.
        while n < nmolecules:
            similarity_engine.settings.packing_shell_size = n
            compare = similarity_engine.compare(c_i, c_j)
            if compare is None:
                results.append((j, BIG_RMS))
                break
            if compare.nmatched_molecules < n:
                results.append((j, BIG_RMS))
                break
            n = round(n * 1.618)
        else:
            similarity_engine.settings.packing_shell_size = nmolecules
            compare = similarity_engine.compare(c_i, c_j)
            if compare.nmatched_molecules == nmolecules:
                results.append((j, compare.rmsd))
            else:
                results.append((j, BIG_RMS))
    with open('rmsds.dat', 'a') as f:
        for result in results:
            j, rmsd = result
            if rmsd < 1e3:
                LOG.info(f'Overlay of {ref_id}-{comp_ids[j]} has RMSD = {rmsd}')
            f.write(f'{ref_id} {comp_ids[j]} {rmsd}\n')
    return results

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

def iterative_pymatgen_batch(ref, comp, ref_id, comp_ids, nprocs):

    length = len(comp)
    chunksize = min(40, int(length / nprocs / nprocs) + 1)
    if length % chunksize == 0:
        chunknum = int(length / chunksize)
    else:
        chunknum = int(length / chunksize) + 1
    results = []
    pool = Pool(processes=nprocs)
    for i in range(chunknum):
        start = i * chunksize
        end = (i + 1) * chunksize
        LOG.debug("Start %s to %s structures", start, end)
        results.append(
            pool.apply_async(
                func=pymatgen_structurematcher,
                args=(ref, comp[start:end], ref_id, comp_ids, start),
            )
        )
    pool.close()
    pool.join()
    del pool

    rmsds = {}
    for result in results:
        for j, rmsd in result.get():
            rmsds[j] = rmsd
    return rmsds

def iterative_compack_batch(ref, comp, ref_id, comp_ids, nprocs, settings):
    """Multiprocessing pair comparison. Pool.map had some wired issue with a
    long list on iridis 5. One could change the code to use pool.map after
    testing pool.map."""

    for k, v in settings.items():
        LOG.info(f'{k} = {v}')
    length = len(comp)
    chunksize = min(10000, int(length / nprocs / nprocs) + 1)
    if length % chunksize == 0:
        chunknum = int(length / chunksize)
    else:
        chunknum = int(length / chunksize) + 1
    results = []
    pool = Pool(processes=nprocs)
    for i in range(chunknum):
        start = i * chunksize
        end = (i + 1) * chunksize
        LOG.debug("Start %s to %s structures", start, end)
        results.append(
            pool.apply_async(
                func=mercury_compack,
                args=(ref, comp[start:end], ref_id, comp_ids, settings, start),
            )
        )
    pool.close()
    pool.join()
    del pool

    rmsds = {}
    for result in results:
        for j, rmsd in result.get():
            rmsds[j] = rmsd
    return rmsds
