from cspy.db import CspDataStoreMPS
from cspy.crystal.generate_crystal import CrystalGenerator, inverse_matrix
from cspy.crystal.util import get_contacts
from cspy.linalg.kabsch import kabsch_rotation_matrix
from cspy.chem.molecule import Molecule
import multiprocessing
import argparse
import math
import random
import numpy as np
import copy
import itertools
import json
import time
from cspy.sample.sobol import sobol_vector
from math import sin, cos, pi, sqrt
import logging
from typing import Tuple

LOG = logging.getLogger(__name__)

def get_cell_parameters(clg : CrystalGenerator, seed : int) -> Tuple:
    """
    For an initialised CrystalGenerator with a given sobol seed,
    return the cell parameters

    Parameters
    ----------
    clg : CrystalGenerator
        CrystalGenerator object that is configured with the crystal
        generator settings

    seed : int
        Random seed

    Returns
    -------
    a : float 
        Cell length `a` of unit cell

    b : float 
        Cell length `b` of unit cell

    c : float 
        Cell length `c` of unit cell

    alpha : float 
        Cell angle `alpha` of unit cell

    beta : float 
        Cell angle `beta` of unit cell

    gamma : float 
        Cell angle `gamma` of unit cell

    ca : float 
        Cosine of alpha

    cb : float 
        Cosine of beta

    cg : float 
        Cosine of gamma

    sg : int 
        Space group number

    v : float 
        Unit cell volume

    x : float 
        Variable used for matrix operations on cell

    translations : np.arraylike()
        Vector of translations to be applied to molecule in unit cell

    """

    # generate a quasirandom vector and map to the parameters
    ndim, dim = clg.search_dimensions

    npos = dim["positions"]
    nrot = dim["orientations"]
    nang = len(dim["uc_angles"])
    vector = sobol_vector(seed, ndim)

    # set up molecular translations 
    translations = vector[0: npos].reshape(-1, 3)

    # map cell angles
    angle_params = {}
    for i, k in enumerate(dim["uc_angles"]):
        angle_params[k] = vector[i + npos + nrot]
    alpha, beta, gamma = clg.convert_angles(angle_params)

    ca = cos(alpha)
    cb = cos(beta)
    cg = cos(gamma)
    sg = sin(gamma)
    x = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
    if x <= 0:
        return (None, )*13

    # rotate the molecules
    rotations = vector[npos: npos + nrot].reshape(-1, 3)

    # rotations and translations are required for this step so we can get mol_pos and
    # subsequently, lengths
    mol_pos, mol_pos_translated = clg.translate_and_rotate_molecules(translations, rotations)

    if not np.any(translations):
        translations = [np.zeros(3) for m in range(clg.n_mols)]

    # map cell lengths
    sqrt_x = sqrt(x)
    unit_inverse = inverse_matrix(1, 1, 1, ca, cb, cg, sg, sqrt_x)
    lengths = clg.convert_lengths(
        vector[npos + nrot + nang:], mol_pos, unit_inverse, sqrt_x, seed
    )
    if lengths is None:
        return None, None, None, None, None, None, None, None, None, None, None, None, None
    a, b, c = lengths

    v = a * b * c * sqrt_x

    return a, b, c, alpha, beta, gamma, ca, cb, cg, sg, v, x, translations[0]

 
def check_collision_in_chain(pairChain : Molecule) -> bool:
    """
    Check if there's a collision between any of the molecules in the chain.
    If there is, return True.
    Otherwise, return False.

    Parameters
    ----------
    pairChain : Molecule
        Molecule object representing a chain of molecules where each
        molecule is a different component of the molecule

    Returns
    -------
    collision : bool
        Whether or not there is a collision

    """
    num_components = len(pairChain.components)
    neighs_pos = []
    neighs_cov = []

    for ind, component in enumerate(pairChain.components):
        if ind < (num_components - 1):
            neighs_pos.append(component.positions)
            neighs_cov.append(np.array([i.cov for i in component.elements]))

    neighs_pos = np.concatenate(neighs_pos)
    neighs_cov = np.concatenate(neighs_cov)

    molC = pairChain.components[-1]
    molC_pos = molC.positions
    molC_cov = np.concatenate([[i.cov for i in molC.elements]])


    contacts, distances, box_idxs, at_col_vectors = get_contacts(molC_pos, molC_cov, neighs_pos, neighs_cov, tolerance=1.0)

    if contacts == []:
        collision = False
    else:
        if len(np.concatenate(contacts)) > 0:
            collision = True
        else:
            collision = False

    return collision


def get_required_pair_types(xyz_files : list, pair_types : dict) -> list:
    """
    Iterate over list of xyz_files and create a chain of pairs of molecules.
    The pairs required to form the chain are then stored in a list.
    We check this list that each of these pairs exists somewhere in our databases.
    

    Parameters
    ----------
    xyz_files : list
        List should be in a random order and
        each molecule in the asymmetric unit should be represented by one xyz_file.
        If a molecule appears in the asymmetric unit multiple times, it should
        appear in xyz_files multiple times.

    pair_types : dict
        Dictionary where the keys are pair types, and the values are IDs of pairs.

    Returns
    -------
    collision : bool
        Whether or not there is a collision

    """
    required_pair_types = []
    for ind, xyz_file in enumerate(xyz_files):
        if not ind == len(xyz_files)-1:
            pair_type = tuple(sorted([xyz_file, xyz_files[ind+1]]))
            required_pair_types.append(pair_type)

    # make sure required_pairs list only includes valid pairs
    unique_molecules = []
    for ind, pair_type in enumerate(required_pair_types):
        if pair_type in pair_types.keys():
            if not pair_type[0] in unique_molecules:
                unique_molecules.append(pair_type[0])
            if not pair_type[1] in unique_molecules:
                unique_molecules.append(pair_type[1])
        else:
            new_mol = xyz_files[ind+1]
            for unique_mol in unique_molecules:
                candidate_pair_type = tuple(sorted([new_mol, unique_mol]))
                if candidate_pair_type in pair_types.keys():
                    required_pair_types[ind] = candidate_pair_type
                    if not new_mol in unique_molecules:
                        unique_molecules.append(new_mol)
                    break

    return required_pair_types


def select_pair(pair_types : dict, pair_type : str, weights : dict, seed : int) -> str:
    """
    Randomly select a unique pair from the list that matches the pair type.
    

    Parameters
    ----------
    pair_types : dict
        Dictionary where the keys are pair types, and the values are IDs of pairs.

    pair_type : str
        Type of pair, e.g. acetone and water, water and water, etc.

    weights : dict
        Dictionary where the keys are pair types, and the values are 
        random weights of pairs.

    seed : int
        Random seed

    Returns
    -------
    pair : str
        Pair ID

    """
    random.seed(seed)
    pair = random.choices(pair_types[pair_type], weights = weights[pair_type], k = 1)
    return pair[0]


def generate_single_pair_crystal(pair_types : dict, properties : dict, xyz_files : list, weights : dict, seed : int) -> Tuple:
    """
    Randomly select a unique pair from the list that matches the pair type.
    

    Parameters
    ----------
    pair_types : dict
        Dictionary where the keys are pair types, and the values are IDs of pairs.

    properties : dict
        Dictionary where the keys are pair IDs and values are the properties of
        those pairs.

    xyz_files : list
        List should be in a random order and
        each molecule in the asymmetric unit should be represented by one xyz_file.
        If a molecule appears in the asymmetric unit multiple times, it should
        appear in xyz_files multiple times.

    weights : dict
        Dictionary where the keys are pair types, and the values are 
        random weights of pairs.

    seed : int
        Random seed

    Returns
    -------
    cspyMol : Molecule
        A pair of molecules stored as a mol-CSPy Molecule object

    pair : str
        The ID of the pair

    None : None
        This exists for symmetry with the generate_multi_pair_crystal function
    """

    pair_type = tuple(sorted([xyz_files[0], xyz_files[1]]))
    pair = select_pair(pair_types, pair_type, weights, seed)
    xyz_string = properties[pair]['xyz_coordinates']
    cspyMol = Molecule.from_xyz_string(xyz_string)

    return cspyMol, pair, None


def get_pair_info(properties : dict, molecules : dict, pair : str) -> Tuple:
    """
    Retrieve information related to a specific pair
    

    Parameters
    ----------
    properties : dict
        Dictionary where the keys are pair IDs and values are the properties of
        those pairs.

    molecules : dict
        Dictionary where the keys are molecule IDs and values are the properties of
        those molecules.

    pair : str
        Pair ID

    Returns
    -------
    molecule1 : str
        ID of first molecule in pair

    molecule2 : str
        ID of second molecule in pair

    mol1_atom_ids : str
        IDs of atoms in pair that belong to molecule 1

    molecule2 : str
        IDs of atoms in pair that belong to molecule 2
    """

    molecule1 = properties[pair]['molecule1']
    molecule2 = properties[pair]['molecule2']
    mol1_num_atoms = molecules[molecule1]['num_atoms']
    mol2_num_atoms = molecules[molecule2]['num_atoms']
    mol1_atom_ids = [i for i in range(mol1_num_atoms)]
    mol2_atom_ids = [i for i in range(mol1_num_atoms, mol1_num_atoms + mol2_num_atoms)]
                     
    return molecule1, molecule2, mol1_atom_ids, mol2_atom_ids


def make_chain(pairChain : Molecule, 
               properties : dict, 
               molecules : dict, 
               xyz_files : list, 
               pair : str, 
               existing_components : dict, 
               ind : int, 
               seed : int) -> Tuple:
    """
    Append a new molecule to an existing chain by taking a pair and superimposing
    one molecule in the pair onto an existing molecule in the chain.
    

    Parameters
    ----------
    pairChain : Molecule
        Molecule object representing a chain of molecules where each
        molecule is a different component of the molecule

    properties : dict
        Dictionary where the keys are pair IDs and values are the properties of
        those pairs.

    molecules : dict
        Dictionary where the keys are molecule IDs and values are the properties of
        those molecules.

    xyz_files : list
        List should be in a random order and
        each molecule in the asymmetric unit should be represented by one xyz_file.
        If a molecule appears in the asymmetric unit multiple times, it should
        appear in xyz_files multiple times.

    pair : str
        Pair ID

    existing_components : dict
        Dictionary where the keys are the IDs of molecules and the values are a list
        containing the indexes of molecules in the chain that match the key.

    ind : int
        Index of molecule that has been added. (This is just a count)

    seed : int
        Random seed


    Returns
    -------
    pairChain : Molecule
        Molecule object representing a chain of molecules where each
        molecule is a different component of the molecule

    existing_components : dict
        Dictionary where the keys are the IDs of molecules and the values are a list
        containing the indexes of molecules in the chain that match the key.

    chain_connect : int
        Index of molecule in the chain that the new pair was superimposed onto
    """

    random.seed(seed)
    xyz_string = properties[pair]['xyz_coordinates']
    new_pair = Molecule.from_xyz_string(xyz_string)
    molecule1, molecule2, mol1_atom_ids, mol2_atom_ids = get_pair_info(properties, molecules, pair)
    try:
        new_pair.def_components([mol1_atom_ids, mol2_atom_ids])
    except:
        return None, None, None

    new_mol_type = xyz_files[ind+2]
    if molecule1 == new_mol_type:
        if molecule2 == new_mol_type:
            new_pair_connect = random.randint(0, 1)
        else:
            new_pair_connect = 1
    else:
        new_pair_connect = 0

    if new_pair_connect == 0:
        new_pair_connect_mol = molecule1
        new_pair_other_mol = molecule2
        new_pair_other = 1
    else:
        new_pair_connect_mol = molecule2
        new_pair_other_mol = molecule1
        new_pair_other = 0

    molA = new_pair.components[new_pair_connect]
    
    if not new_pair_connect_mol == 'unknownid':
        equivalent_positions = random.choice(molecules[new_pair_connect_mol]['equivalent_atoms'])
        molA.exchange_symmetry_equivalent_atoms(equivalent_positions)

    molA_pos = molA.positions
    molA_cent = molA.centroid

    if not new_pair_connect_mol in existing_components.keys():
        return None, None
    chain_connect = random.choice(existing_components[new_pair_connect_mol])
    molB = pairChain.components[chain_connect]
    
    molB_pos = molB.positions
    molB_cent = molB.centroid
    
    molC  = new_pair.components[new_pair_other]
    molC_pos = molC.positions
    rotation_matrix = kabsch_rotation_matrix(molA_pos - molA_cent, molB_pos - molA_cent)

    #apply rotation
    molC_pos = np.dot(molC_pos - molA_cent, rotation_matrix) + molB_cent
    molC.positions = molC_pos

    pairChain_test = copy.deepcopy(pairChain)
    pairChain_test.add_component(molC)

    if not check_collision_in_chain(pairChain_test):
        pairChain.add_component(molC)
        if not new_pair_other_mol in existing_components.keys():
            existing_components[new_pair_other_mol] = []
        existing_components[new_pair_other_mol].append(ind+2)
    else:
        # abandon crystal entirely if this fails.
        # inefficient but good for testing
        return None, None, None
    
    return pairChain, existing_components, chain_connect


def generate_multi_pair_crystal(pair_types : dict, 
                                properties : dict, 
                                clg : CrystalGenerator, 
                                molecules : dict, 
                                unshuffled_xyz : list, 
                                weights : dict, 
                                sample_size : int, 
                                seed : int) -> Tuple:
    """
    Generate a crystal where the asymmetry unit is composed of a chain of more
    than one pair of molecules.
    We do this by initially selecting one pair of molecules as our base.
    Then we select the next pair randomly and append it to the chain. We check
    for collisions and if there are none, we build a supercell from this chain
    and check for collisions there. If there are collisions at either step, we 
    replace the latest pair with a different random pair. If we pass all
    collision checks, we move on to add the next pair in the chain.
    When the chain represents the entire asymmetric unit, we sort the chain so
    that the molecules match the order expected by DMACRYS, and then we return
    the clg object.
    

    Parameters
    ----------
    pair_types : dict
        Dictionary where the keys are pair types, and the values are IDs of pairs.

    properties : dict
        Dictionary where the keys are pair IDs and values are the properties of
        those pairs.
    
    clg : CrystalGenerator
        CrystalGenerator object that is configured with the crystal
        generator settings

    molecules : dict
        Dictionary where the keys are molecule IDs and values are the properties of
        those molecules.

    unshuffled_xyz : list
        List should be in a random order and
        each molecule in the asymmetric unit should be represented by one unshuffled_xyz.
        If a molecule appears in the asymmetric unit multiple times, it should
        appear in unshuffled_xyz multiple times.

    weights : dict
        Dictionary where the keys are pair types, and the values are 
        random weights of pairs.

    sample_size : int
        Number of times to attempt to generate a crystal

    seed : int
        Random seed


    Returns
    -------
    clg : CrystalGenerator
        CrystalGenerator object that is configured with the crystal
        generator settings

    overrides : dict
        Dictionary of that allows overwriting of variables in CrystalGenerator class
    """

    xyz_files = copy.deepcopy(unshuffled_xyz)
    random.seed(seed)
    random.shuffle(xyz_files)

    # Check that the first pair is in the dictionary of allowed pair types
    # Try to fix it if not
    check_passed = False
    pass_attempts = 0
    while check_passed == False:
        if pass_attempts == len(xyz_files):
            return None, None
        xyz0 = xyz_files[0]
        xyz1 = xyz_files[1]
        pair0 = tuple(sorted([xyz0, xyz1]))
        if not pair0 in pair_types.keys():
            xyz_files.pop(0)
            xyz_files.append(xyz0)
        else:
            check_passed = True
        pass_attempts += 1

    # sort single atom "molecules" to the end of the list
    for _ in range(len(xyz_files)-2):
        xyz2 = xyz_files[2]
        xyz_num_atoms = molecules[xyz2]["num_atoms"]
        if xyz_num_atoms == 1:
            xyz_files.pop(2)
            xyz_files.append(xyz2)

    pairChain, pair0, rotations = generate_single_pair_crystal(pair_types, properties, xyz_files, weights, seed+1)

    required_pair_types = get_required_pair_types(xyz_files, pair_types)

    existing_components = dict()

    molecule1, molecule2, mol1_atom_ids, mol2_atom_ids = get_pair_info(properties, molecules, pair0)
    existing_components[molecule1] = [0]
    if not molecule2 in existing_components.keys():
        existing_components[molecule2] = []
    existing_components[molecule2].append(1)

    try:
        pairChain.def_components([mol1_atom_ids, mol2_atom_ids])
    except:
        LOG.error("Cant define components")
        return None, None
    
    a, b, c, alpha, beta, gamma, ca, cb, cg, sg, v, x, sobol_translation  = get_cell_parameters(clg, seed)
    if a == None:
        return None, None
    
    comp1 = pairChain.components[0]
    comp1_centroid = comp1.centroid
    pairChain.translate(-1 * comp1_centroid)
    for component in pairChain.components:
        component.translate(-1 * comp1_centroid)
    
    del required_pair_types[0]

    # check that initial pair doesn't have a clash
    num_components = len(pairChain.components)
    translations = [sobol_translation for component in range(num_components)]
    nudged_translations = [[0, 0, 0] for component in range(num_components)]
    mol_pos = []
    for component in pairChain.components:
        mol_pos.append(component.positions)
    clg.n_mols = num_components
    clg.n_atoms = [len(component) for component in pairChain.components]
    sc_pos, asym_pos = clg.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, translations)
    covs = [[i.cov for i in mol.elements] for mol in pairChain.components]
    clg.sc_covs = []
    for cov in covs:
        clg.sc_covs += cov * clg.nsymops * 125
    clg.sc_covs = np.array(clg.sc_covs)
    collision, mol_collisions, react_col_vectors, sum_col_vectors = clg.collision(sc_pos, [], tol=0.5)
    if collision:
        return None, None

    overrides = {'a' : a, 'b' : b, 'c' : c,
                 'alpha' : alpha, 'beta' : beta, 'gamma' : gamma}

    # ::::::::::::: 1 point MPS START :::::::::::::
    for ind, pair_type in enumerate(required_pair_types):
        valid = False
        for sample_count, sample in enumerate(range(sample_size)):
            pair = select_pair(pair_types, pair_type, weights, seed+ind+sample+1)
            candidate_pairChain, candidate_existing_components, candidate_chain_connect = make_chain(copy.deepcopy(pairChain), properties, molecules, xyz_files, pair, copy.deepcopy(existing_components), ind, seed+ind+sample+1)
            if candidate_pairChain:
                num_components = len(candidate_pairChain.components)
                translations = [sobol_translation for component in range(num_components)]
                nudged_translations = [[0, 0, 0] for component in range(num_components)]
                mol_pos = []
                for component in candidate_pairChain.components:
                    mol_pos.append(component.positions)
                clg.n_mols = num_components
                clg.n_atoms = [len(component) for component in candidate_pairChain.components]
                covs = [[i.cov for i in mol.elements] for mol in candidate_pairChain.components]
                clg.sc_covs = []
                for cov in covs:
                    clg.sc_covs += cov * clg.nsymops * 125
                clg.sc_covs = np.array(clg.sc_covs)
                sc_pos, asym_pos = clg.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, translations)
                collision, mol_collisions, react_col_vectors, sum_col_vectors = clg.collision(sc_pos, [], tol=0.5)
                if collision:
                    pass
                else:
                    valid = True
                    break               
                    
            else:
                continue
        if not valid:
            return None, None
        pairChain = candidate_pairChain

        existing_components = candidate_existing_components
    # ::::::::::::: 1 point MPS END :::::::::::::

    # ordering of molecules must be correct for DMACRYS to find multipoles
    if not unshuffled_xyz == xyz_files:
        new_order = []
        for xyz in unshuffled_xyz:
            ind = existing_components[xyz][0]
            del existing_components[xyz][0]
            new_order.append(ind)
        pairChain.sort_components(new_order)

    clg.molecules = [pairChain]
    clg.n_mols = 1
    clg.n_atoms = [len(pairChain)]
    covs = [[i.cov for i in pairChain.elements]]
    clg.sc_covs = []
    for cov in covs:
        clg.sc_covs += cov * clg.nsymops * 125
    clg.sc_covs = np.array(clg.sc_covs)


    overrides['translations'] = [sobol_translation]
    overrides['rotations'] = [[0, 0, 0]]

    return clg, overrides

def generate_crystal(pair_types : dict, 
                     properties : dict, 
                     molecules : dict, 
                     xyz_files : list, 
                     sg : int, 
                     weights : dict, 
                     sample_size : int, 
                     Q : int, 
                     workers : int, 
                     num_crystals : int) -> None:
    """
    Iterate over seeds and generate QR crystals using the MPS method until reaching
    the target number of crystal (num_crystals).
    

    Parameters
    ----------
    pair_types : dict
        Dictionary where the keys are pair types, and the values are IDs of pairs.

    properties : dict
        Dictionary where the keys are pair IDs and values are the properties of
        those pairs.

    molecules : dict
        Dictionary where the keys are molecule IDs and values are the properties of
        those molecules.

    xyz_files : list
        List should be in a random order and
        each molecule in the asymmetric unit should be represented by one xyz_files.
        If a molecule appears in the asymmetric unit multiple times, it should
        appear in xyz_files multiple times.

    sg : int
        Crystal sapce group

    weights : dict
        Dictionary where the keys are pair types, and the values are 
        random weights of pairs.

    sample_size : int
        Number of times to attempt to generate a crystal

    seed : int
        Random seed

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function
    """

    from cspy.sauce.extract_mps import check_reservation
    num_valid = 0
    for seed in range(1, 99999999):
        if check_reservation(seed, Q, workers):
            crystals = []
            LOG.debug("Seed: %d", seed)
            reference_molecules = [Molecule.from_xyz_string(molecules[xyz_file]['xyz_coordinates']) for xyz_file in xyz_files]
            clg = CrystalGenerator(reference_molecules, sg, tvp=2.5, nudge=0,  adaptcell=True, asi=False)
            if len(xyz_files) > 2:

                clg, overrides = generate_multi_pair_crystal(pair_types, properties, clg, molecules, xyz_files, weights, sample_size, seed)
                if clg:
                    generated = clg.generate(seed, overrides=overrides)
                    if generated:
                        crystals.append(generated)

            else:
                clg, generated, cspyMol, pair0, rotations = generate_single_pair_crystal(pair_types, properties, xyz_files, weights, seed)
                if cspyMol:
                    num_cells = 10
                    for cell_index in range(num_cells):
                        mutator = seed * cell_index
                        generated, rotations = clg.generate_and_opt_rot(seed+mutator, maxiter=10, overrides=overrides)
                        if generated:
                            crystals.append(generated)

            num_valid += len(crystals)
            LOG.info("Num valid: %d", num_valid)

            for ind, generated in enumerate(crystals):
                generated.to_shelx_file(str(seed) + '_' + str(ind) + '.res')

            if num_valid == num_crystals or num_valid > num_crystals:
                break
    

def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-mps clg',
                description='Read all molecular pairs from database and generate candidate crystals')
        parser.add_argument('database', type=str,
                help='Specify database file to read')
        parser.add_argument("-x", "--xyz-files", nargs="+", type=str,
                help="Xyz files containing molecules for generation")
        parser.add_argument('-spg', '--spacegroup', type=int,
                help='Specify spacegroup of crystal to generate')
        parser.add_argument('-np', '--numproc', type=int,
                help='Specify number of paralllel process')
        parser.add_argument('-nc', '--numcrys', type=int,
                help='Specify number of crystals to sample')
        parser.add_argument('-ss', '--samplesize', type=int,
                help='Specify number of candidate chains to sample per crystal')
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

        workers = args.numproc
        num_crystals = args.numcrys
        if args.samplesize:
            sample_size = args.samplesize
        else:
            sample_size = args.samplesize
        sg = args.spacegroup
        xyz_files = args.xyz_files
        unique_xyz = set(sorted(xyz_files))


        crys_per_worker = math.floor(num_crystals / workers)
        num_crys_distribution = [crys_per_worker for n in range(workers)]
        for ind, _ in enumerate(range(num_crystals - (crys_per_worker * workers))):
            num_crys_distribution[ind] += 1

        db_name = args.database[0]
        db = CspDataStoreMPS(db_name)

        manager = multiprocessing.Manager()
        properties = manager.dict()
        for row in db.select("pairs",
                                ['id','min_energy','min_density',
                                'xyz_coordinates','mol2_coordinates',
                                'asymmetric','symmetric', 'molecule1', 'molecule2']):
            properties[row[0]]={"min_energy" : row[1], "energy_list" : [row[1]],
                        "min_density" : None, "max_density" : None, 
                        "median_density" : None, "density_list" : [row[2]],
                        "frequency" : 1, "xyz_coordinates" : row[3], "mol2_coordinates" : row[4], 
                        "asymmetric" : row[5], "symmetric" : row[6], 
                        "molecule1" : row[7], "molecule2" : row[8]}

        molecules = dict()
        for row in db.select("molecules", ['id','xyz_coordinates', 'equivalent_atoms']):
            molecules[row[0]] = {'num_atoms' : int(row[1].split('\n')[0]), 
                                 'equivalent_atoms' : json.loads(row[2]),
                                 'xyz_coordinates' : row[1]}

        pairs = list(properties.keys())
        pair_types = dict()
        weights = dict()

        for combo in itertools.combinations(unique_xyz, 2):
            combo = tuple(sorted(combo))
            pair_types[combo] = []
        for xyz in unique_xyz:
            combo = tuple([xyz, xyz])
            pair_types[combo] = []

        for xyz in unique_xyz:
            mol_num_atoms = molecules[xyz]['num_atoms']
            # don't allow pairs of single atoms
            if mol_num_atoms == 1:
                pair_types.pop(tuple([xyz, xyz]), None)

        for pair in pairs:
            pair_type = tuple(sorted([properties[pair]["molecule1"], properties[pair]["molecule2"]]))
            if pair_type in pair_types.keys():
                pair_types[pair_type].append(pair)
                if not pair_type in weights.keys():
                    weights[pair_type] = []
                weights[pair_type].append(properties[pair]['frequency'])

        start = time.time()

        with multiprocessing.Pool(workers) as pool:
            pool.starmap(generate_crystal, [[pair_types, properties, molecules, xyz_files, sg, weights, 
                                             sample_size, Q, workers, 
                                             num_crys_distribution[Q]] for Q in range(workers)])
            
        LOG.info("Runtime: %d", time.time() - start)