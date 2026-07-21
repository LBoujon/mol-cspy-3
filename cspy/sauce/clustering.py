from cspy.db import CspDataStoreMPS
from cspy.sauce.extract_mps import create_mps_database, add_to_mps_database, set_molecules
from cspy.flex.flex_molecule import FlexMolecule
from cspy.util.misc import mp_check_reservation
import ccdc
from ccdc.descriptors import MolecularDescriptors as MD
import multiprocessing
import itertools
import numpy as np
import argparse
import time
import logging
from typing import Optional

LOG = logging.getLogger(__name__)

def query_and_execute(cursor, sql_query : str) -> dict:
        cursor.execute(sql_query)
        fetch = cursor.fetchall()

        return fetch


def get_mol2(cursor, pair : list) -> str:
    mol2_coords = query_and_execute(cursor, "select res from pairs where id = '" + pair[0] + "';")[0]

    return mol2_coords[0]


def consolidate_properties(properties_dict1 : dict, properties_dict2 : dict) -> dict:
    """ if two pairs are determined to be equivalent, we needn't keep both their
    property dictionaries. We'll create a new property dictionary that summarises
    the information from both input dictionaries.

    """

    properties_dict = {"min_density" : None,
                       "max_density" : None,
                       "median_density" : None,}
    # update min energy now because we use it during clustering, but deal with density later
    if properties_dict1["min_energy"] < properties_dict2["min_energy"]:
        properties_dict["min_energy"] = properties_dict1["min_energy"]
        properties_dict["xyz_coordinates"] = properties_dict1["xyz_coordinates"]
        properties_dict["mol2_coordinates"] = properties_dict1["mol2_coordinates"]
        properties_dict["molecule1"] = properties_dict1["molecule1"]
        properties_dict["molecule2"] = properties_dict1["molecule2"]
    else:
        properties_dict["min_energy"] = properties_dict2["min_energy"]
        properties_dict["xyz_coordinates"] = properties_dict2["xyz_coordinates"]
        properties_dict["mol2_coordinates"] = properties_dict2["mol2_coordinates"]
        properties_dict["molecule1"] = properties_dict2["molecule1"]
        properties_dict["molecule2"] = properties_dict2["molecule2"]

    properties_dict["energy_list"] = properties_dict1["energy_list"] + properties_dict2["energy_list"]
    properties_dict["density_list"] = properties_dict1["density_list"] + properties_dict2["density_list"]
    properties_dict["frequency"] = properties_dict1["frequency"] + properties_dict2["frequency"]

    if properties_dict1["asymmetric"] or properties_dict2["asymmetric"]:
        properties_dict["asymmetric"] = True
    else:
        properties_dict["asymmetric"] = False
    if properties_dict1["symmetric"] or properties_dict2["symmetric"]:
        properties_dict["symmetric"] = True
    else:
        properties_dict["symmetric"] = False
     
    return properties_dict


def get_nearest_neighbours(atoms1 : list, atoms2 : list) -> list:
    """ Which two atoms between the two molecules are the closest? """
    atoms1_coords = []
    for atom in atoms1:
        x, y, z = atom.coordinates
        atoms1_coords.append(np.array([x, y, z]))
    atoms2_coords = []
    for atom in atoms2:
        x, y, z = atom.coordinates
        atoms2_coords.append(np.array([x, y, z]))

    fake_bond_pair_inds = [999999999, 9999999999]

    min_distance = 999999
    for ind1, coord1 in enumerate(atoms1_coords):
        for ind2, coord2 in enumerate(atoms2_coords):
            dist = np.linalg.norm(coord1-coord2)
            if dist < min_distance:
                # if statement creates consistency if two neighbours have competing distances
                if ind1 > fake_bond_pair_inds[0] and ind2 > fake_bond_pair_inds[1] and (min_distance - dist) < 0.01:
                    pass
                else:
                    min_distance = dist
                    fake_bond_pair_inds = [ind1, ind2]

    return fake_bond_pair_inds


def make_fake_bond(pair_obj : ccdc.molecule.Molecule, fake_bond_pair_inds : list) -> ccdc.molecule.Molecule:
    """ Define a bond between the two molecules to trick the CCDC into thinking that it's one molecule. """
    atoms = pair_obj.atoms
    atoms_1 = pair_obj.components[0].atoms

    fake_bond_pair = [atoms[fake_bond_pair_inds[0]], atoms[fake_bond_pair_inds[1] + len(atoms_1)]]
    pair_obj.add_bond("Single", fake_bond_pair[0], fake_bond_pair[1])

    return pair_obj


def get_pair_obj(db_name : str, pair : str, mol2_string : Optional[str] = None, fake_bond_pair_inds : Optional[list] = None) -> tuple:
    """ Represent a molecular pair as a CCDC Molecule object.
    We can construct the Molecule from a mol2 string. This can
    either be provided directly or sourced from a database if a
    db_name is provided.

    We can't store the CCDC Molecule obj, so we store the information
    as a mol2 string.

    Parameters
    ----------
    db_name : str
        Name of database

    pair : str
        Name of pair in database

    mol2_string : str
        mol2 string representation of molecular pair

    fake_bond_pair_inds : list of ints
        List containing indexes of molecules that we will define
        a fake bond between.

    Returns
    -------
    pair_obj : ccdc.molecule.Molecule
        molecular pair as a CCDC Molecule object

    mol2_string : str
        mol2 string representation of molecular pair

    xyz_string : str
        xyz string representation of molecular pair

    fake_bond_pair_inds : list of ints
        List containing indexes of molecules that we will define
        a fake bond between.

    cog_diff : int
        Difference (in Angstrom) between center of geometry of the
        two molecules in the molecular pair.

    """
    if not mol2_string:
        db = CspDataStoreMPS(db_name)
        mol2_string, xyz_string = db.query("select mol2_coordinates,xyz_coordinates from pairs where id = '" + pair + "';").fetchall()[0]
        db.disconnect()
    else:
        xyz_string = None
    pair_obj = ccdc.molecule.Molecule.from_string(mol2_string)
    components = pair_obj.components
    if not len(components) == 2:
        pair_obj = None
        fake_bond_pair_inds = []
        cog_diff = None
    else:
        comp1 = components[0]
        atoms_1 = comp1.atoms
        comp2 = components[1]
        atoms_2 = comp2.atoms

        if not fake_bond_pair_inds:
            fake_bond_pair_inds = get_nearest_neighbours(atoms_1, atoms_2)
            cog1 = np.array(comp1.centre_of_geometry())
            cog2 = np.array(comp2.centre_of_geometry())
            cog_diff = np.linalg.norm(cog2-cog1)
        else:
            cog_diff = 0

    return pair_obj, mol2_string, xyz_string, fake_bond_pair_inds, cog_diff


def compare_pair(pair_pair : list, pair_data : dict, duplicates : dict, properties : dict, rmsd_tol : float = 0.6) -> None:
    """ Overlay two pairs and calculate their rmsd.
    If rmsd is less than 0.6, we treat them as equivalent.

    Parameters
    ----------
    pair_pair : list of strings
        Two-element list containing the IDs of two molecular pairs.

    pair_data : dict
        Dictionary where the keys are the names of pairs and the values 
        are another dictionary containing mol2 strings and the indices of
        fake bond pairs

    duplicates : list of strings
        List containing IDs of molecular pairs that are duplicates and will
        be removed

    properties : dict
        Dictionary where the keys are names of molecular pairs and the values
        are dictionary. This dictionary contains various properties such
        as xyz_coordinates and min_energy. See consolidate_properties.

    rmsd_tol : float
        Tolerance for RMSD difference between two pairs in order for them
        to be considered the same.

    """
    pair_data1 = pair_data[pair_pair[0]]
    pair1, mol2_string, xyz_string, fake_bond_pair_inds1, cog_diff = get_pair_obj(None, None, pair_data1['mol2'], pair_data1['fake_bond_pair_inds'])
    pair1 = make_fake_bond(pair1, fake_bond_pair_inds1)
    pair_data2 = pair_data[pair_pair[1]]
    pair2, mol2_string, xyz_string, fake_bond_pair_inds2, cog_diff = get_pair_obj(None, None, pair_data2['mol2'], pair_data2['fake_bond_pair_inds'])
    pair2 = make_fake_bond(pair2, fake_bond_pair_inds2)

    try:
        molecule, rmsd, rmsd_tanimoto, transformation = MD.overlay_rmsds_and_transformation(pair1, pair2, with_symmetry=True, atoms=None)
    except:
        LOG.info("Failed to overlay. Setting rmsd to 1000.")
        rmsd = 1000

    if rmsd < rmsd_tol:
        if not pair_pair[1] in duplicates and not pair_pair[0] in duplicates:
            properties_dict1 = properties[pair_pair[0]]
            properties_dict2 = properties[pair_pair[1]]
            properties_dict = consolidate_properties(properties_dict1, properties_dict2)
            if properties_dict1["min_energy"] < properties_dict2["min_energy"]:
                properties[pair_pair[0]] = properties_dict
                duplicates.append(pair_pair[1])
            else:
                properties[pair_pair[1]] = properties_dict
                duplicates.append(pair_pair[0])


def compare_pairs_phase2(pair : str, pair_data : dict, candidate_duplicates : dict, duplicates : list, properties : dict) -> None:
    """ Compare a molecular pair with all of it's potential equivalent
    pairs by performing molecular overlays.

    Parameters
    ----------
    pair : str
        ID of molecular pair

    pair_data : dict
        Dictionary where the keys are the names of pairs and the values 
        are another dictionary containing mol2 strings and the indices of
        fake bond pairs

    candidate_duplicates : dict
        Keys are IDs of pairs. Values are lists of IDs of pairs that may 
        me equivalent o the pair in the key.

    duplicates : list of strings
        List containing IDs of molecular pairs that are duplicates and will
        be removed

    properties : dict
        Dictionary where the keys are names of molecular pairs and the values
        are dictionary. This dictionary contains various properties such
        as xyz_coordinates and min_energy. See consolidate_properties.

    """
    if not pair in duplicates:
        pair_pairs = itertools.product([pair], candidate_duplicates[pair])
        for pair_pair in pair_pairs:
            if not pair_pair[0] in duplicates and not pair_pair[1] in duplicates:
                    compare_pair(pair_pair, pair_data, duplicates, properties)
            else:
                if pair_pair[0] in duplicates:
                    break


def compare_pairs_phase1(pairs : str, metrics_float : dict, metrics_int : dict, candidate_duplicates : dict, Q : str, workers : str) -> None:
    """ Iterate over pairs and compare each pair to all other pairs
    simultaneously. This occurs via a numpy subtraction between
    two vectors of properties. 
    Where the difference is within a tolerance (tol_vector_float),
    the matching pair is added to the candidate_duplicates dictionary.

    Parameters
    ----------
    pairs : list of strings
        List of IDs of molecular pairs

    metrics_float : array of float
        Numpy array of floats to be compared to other pair's metrics_float.
        Matches are determined by difference being within a tolerance.

    metrics_int : array of int
        Numpy array of floats to be compared to other pair's metrics_int.
        Matches are determined by vectors matching exactly.

    candidate_duplicates : dict
        Keys are IDs of pairs. Values are lists of IDs of pairs that may 
        me equivalent o the pair in the key.

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function

    """
    for ind, pair in enumerate(pairs):
        # this if statement checks the pair_pair was intended for this worker
        if mp_check_reservation(ind, Q, workers):
            duplicates = []
            ref_metrics_int = metrics_int[ind]
            ref_metrics_float = metrics_float[ind]
            # box lengths A B C, CoG, MoI, FBPI A and B, Mols A and B
            tol_vector_float = np.asarray([1.75, 1.75, 1.75, 1.5, 0.2 * ref_metrics_float[4], 0.2 * ref_metrics_float[5], 0.2 * ref_metrics_float[6]])
            tol_vector_int = np.asarray([0, 0, 0, 0])

            #dif_metrics_int = metrics_int - ref_metrics_int
            bool_metrics_int = (metrics_int == ref_metrics_int)
            bool_metrics_mask = np.all(bool_metrics_int, axis=1)

            dif_metrics_float = metrics_float[bool_metrics_mask] - ref_metrics_float
            bool_metrics_float = (dif_metrics_float >= (tol_vector_float * -1)) & (dif_metrics_float <= tol_vector_float)

            pair_indices = np.where(bool_metrics_mask)[0]
            for local_ind, bool_metric in enumerate(bool_metrics_float):
                # correct ind2 to real ind
                ind2 = pair_indices[local_ind]
                if not ind2 == ind:
                    if np.all(bool_metric):
                        duplicates.append(pairs[ind2])
            candidate_duplicates[pair] = duplicates


def prepare_data(db_name : str, pairs : list, Q : str, workers : int, properties : dict, unique_molecules : dict, pair_data : dict) -> None:
    """ Iterate over pairs and compare each pair to all other pairs
    simultaneously. This occurs via a numpy subtraction between
    two vectors of properties. 
    Where the difference is within a tolerance (tol_vector_float),
    the matching pair is added to the candidate_duplicates dictionary.

    We can't store the CCDC Molecule obj, so we store the information
    as a mol2 string.

    Parameters
    ----------
    db_name : str
        Name of mol-CSPy database to read

    pairs : list of strings
        List of IDs of molecular pairs

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function

    properties : dict
        Dictionary where the keys are names of molecular pairs and the values
        are dictionary. This dictionary contains various properties such
        as xyz_coordinates and min_energy. See consolidate_properties.

    unique_molecules : dict
        Dictionary of unique molecules where keys are molecules name and the values 
        are a dictionary containing the molecules xyz_coordinates and equivalent_atoms

    pair_data : dict
        Dictionary where the keys are the names of pairs and the values 
        are another dictionary containing mol2 strings and the indices of
        fake bond pairs

    """
    unique_mol_keys = list(unique_molecules.keys())
    for ind, pair in enumerate(pairs):
        if mp_check_reservation(ind, Q, workers):
            pair_obj, mol2_string, xyz_string, fake_bond_pair_inds, cog_diff = get_pair_obj(db_name, pair)
            if not pair_obj == None:
                cog_diff = round(cog_diff, 2)
                cspyMol = FlexMolecule()
                cspyMol.init_from_xyz_str(xyz_string)
                cspyMol.calculate_moments_inertia()
                MoI_val = cspyMol.moments_inertia_eig_vals
                MoI_val = np.round(MoI_val, 1)
                box_lengths = cspyMol.set_box_lengths()
                box_lengths.sort()
                box_lengths = np.asarray(box_lengths)
                box_lengths = np.round(box_lengths, 1)

                if len(unique_mol_keys) == 0:
                    mol1_ind = 0
                    mol2_ind = 0
                else:
                    mol1_ind = unique_mol_keys.index(properties[pair]['molecule1'])
                    mol2_ind = unique_mol_keys.index(properties[pair]['molecule2'])
                molecules = [mol1_ind, mol2_ind]
                molecules = np.asarray(molecules)

                pair_data[pair] = {'mol2' : mol2_string,
                               'fake_bond_pair_inds' : fake_bond_pair_inds,
                               'cog_diff' : cog_diff,
                               'box_lengths' : box_lengths,
                               'MoI_val' : MoI_val,
                               'mols' : molecules}


def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-mps cluster',
                description='Read all molecular pairs from database and cluster them')
        parser.add_argument('database', nargs=1, type=str,
                help='Specify database file to read')
        parser.add_argument('-np', '--numproc', nargs=1, type=int, default=[1],
                help='Specify number of paralllel process')
        parser.add_argument(
                "--log-level", type=str,
                choices=("INFO", "DEBUG", "ERROR", "WARN"),
                default="INFO",
                help="Control level of logging output")
        
        args = parser.parse_args(sys_args)

        logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s')

        start = time.time()

        workers = args.numproc[0]

        db_name = args.database[0]
        db = CspDataStoreMPS(db_name)

        manager = multiprocessing.Manager()
        pair_data = manager.dict()
        candidate_duplicates = manager.dict()
        duplicates = manager.list()
        properties = manager.dict()
        unique_molecules = dict()
        for row in db.select("pairs",
                                ['id','min_energy','min_density',
                                'xyz_coordinates','mol2_coordinates',
                                'asymmetric','symmetric', "molecule1", "molecule2"]):
            properties[row[0]]={"min_energy" : row[1], "energy_list" : [row[1]],
                        "min_density" : None, "max_density" : None, 
                        "median_density" : None, "density_list" : [row[2]],
                        "frequency" : 1, "xyz_coordinates" : row[3], "mol2_coordinates" : row[4], 
                        "asymmetric" : row[5], "symmetric" : row[6], 
                        "molecule1" : row[7], "molecule2" : row[8]}

        for row in db.select("molecules",
                                ['id', 'xyz_coordinates','equivalent_atoms']):
            unique_molecules[row[0]] = {"xyz_coordinates": row[1], "equivalent_atoms": row[2]} 
            
        db.disconnect()

        pairs = list(properties.keys())

        LOG.info("Converting molecular pairs into a CCDC-compatible format and collecting properties.")
        with multiprocessing.Pool(workers) as pool:
                pool.starmap(prepare_data, [[db_name, pairs, Q, workers, properties, unique_molecules, pair_data] for Q in range(workers)])

        # update list of pairs because we might have had to skip bad pairs
        pairs = list(pair_data.keys())
        
        sorted_pairs = sorted(pairs, key=lambda pair: pair_data[pair]['cog_diff'])

        all_box_lengths = np.asarray([pair_data[pair]["box_lengths"] for pair in sorted_pairs])
        all_CoG_diffs = np.asarray([[pair_data[pair]["cog_diff"]] for pair in sorted_pairs])
        all_MoI_vals = np.asarray([pair_data[pair]["MoI_val"] for pair in sorted_pairs])
        # need to make sure this check handles symmetry properly
        all_FBPIs = np.asarray([sorted(pair_data[pair]["fake_bond_pair_inds"]) for pair in sorted_pairs], dtype=np.uint8)
        all_mols = np.asarray([sorted(pair_data[pair]["mols"]) for pair in sorted_pairs], dtype=np.uint8)

        metrics_float = np.concatenate((all_box_lengths, all_CoG_diffs, all_MoI_vals), axis=1, dtype=np.float32)
        metrics_int = np.concatenate((all_FBPIs, all_mols), axis=1, dtype=np.uint8)

        LOG.info("Starting first phase of clustering: numerical comparison of vector properties")
        split_sorted_pairs = np.array_split(sorted_pairs, workers)
        # make array-wise comparisons with various metrics (see above)
        with multiprocessing.Pool(workers) as pool:
                pool.starmap(compare_pairs_phase1, [[sorted_pairs, metrics_float, metrics_int, candidate_duplicates, Q, workers] for Q in range(workers)])


        LOG.info("Starting first phase of clustering: molecular overlays")
        # make comparisons via molecular overlays
        with multiprocessing.Pool(workers) as pool:
                pool.starmap(compare_pairs_phase2, [[pair, pair_data, candidate_duplicates, duplicates, properties] for pair in sorted_pairs])

        LOG.info("Writing results to database %s", db_name)
        db = CspDataStoreMPS(db_name)
        pair_properties = {}
        uniques = []
        for pair in pairs:
            if not pair in duplicates:
                uniques.append(pair)
                pair_properties_dict = properties[pair]
                density_array = np.array(pair_properties_dict["density_list"])
                pair_properties_dict["min_density"] = np.min(density_array)
                pair_properties_dict["max_density"] = np.max(density_array)
                pair_properties_dict["median_density"] = np.median(density_array)
                pair_properties[pair] = pair_properties_dict
        db.disconnect()

        create_mps_database("clustered_pairs")
        set_molecules(unique_molecules,"clustered_pairs")
        add_to_mps_database(pair_properties,"clustered_pairs")

        LOG.info("Num uniques: %d", len(uniques))
        LOG.info("Runtime: %d", time.time() - start)
