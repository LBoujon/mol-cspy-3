import logging
import math
import numpy as np
from spglib import standardize_cell, get_symmetry_dataset, niggli_reduce
from cspy.chem.element import Element
from .space_group import SpaceGroup
from typing import Tuple, List, Literal, Optional
from cspy.chem import Molecule

LOG = logging.getLogger(__name__)


def get_lengths_and_angles(lattice: np.ndarray) -> tuple:
    """Return the lengths and angles of a lattice.

    Parameters
    ----------
    lattice : np.ndarray
        The lattice vectors of a unit cell.

    Returns
    -------
    lengths : np.ndarray
        The lengths of the lattice vectors.
    angles : np.ndarray
        The angles between the lattice vectors.
    """
    a, b, c = np.linalg.norm(lattice, axis=1)
    alpha = np.arccos(np.dot(lattice[1], lattice[2]) / (b * c))
    beta = np.arccos(np.dot(lattice[0], lattice[2]) / (a * c))
    gamma = np.arccos(np.dot(lattice[0], lattice[1]) / (a * b))
    return np.array([a, b, c]), np.array([alpha, beta, gamma])


def niggli_primitive_crystal(crystal, **kwargs):
    from cspy.crystal import Crystal, AsymmetricUnit, UnitCell

    lattice = crystal.unit_cell.direct
    uc_dict = crystal.unit_cell_atoms()
    positions = uc_dict["frac_pos"]
    elements = uc_dict["element"]
    asym_atoms = uc_dict["asym_atom"]
    asym_labels = uc_dict["label"]
    cell = lattice, positions, elements

    # reduced_cell = find_primitive(cell, **kwargs)
    reduced_cell = standardize_cell(cell, to_primitive=True, no_idealize=True, **kwargs)

    if reduced_cell is None:
        LOG.warning("Could not find reduced cell for crystal %s", crystal.titl)
        return None
    reduced_lattice, positions, elements = reduced_cell
    unit_cell = UnitCell(reduced_lattice)
    cart_positions = unit_cell.to_cartesian(positions)

    # tranform to niggli cell so don't end up with flat cells
    niggli_lattice = niggli_reduce(reduced_lattice)
    if niggli_lattice is None:
        LOG.warning("Could not find niggli lattice for crystal %s", crystal.titl)
        return None
    unit_cell = UnitCell(niggli_lattice)
    positions = unit_cell.to_fractional(cart_positions)
    positions -= np.floor(positions)

    unit_cell = UnitCell.from_lengths_and_angles(
        unit_cell.lengths, unit_cell.angles
    )  # make lattice vectors consistent with cspy

    sg = SpaceGroup(1)
    asym = AsymmetricUnit(
        [Element[x] for x in elements],
        positions,
        # labels=asym_labels[:len(positions)], # if unit cell has more molecules than asym cell then will yield duplicate labels
    )
    return Crystal(unit_cell, sg, asym, titl=crystal.titl + "_primitive")


def standardize_crystal(crystal, **kwargs):
    from cspy.crystal import Crystal, AsymmetricUnit, UnitCell

    name = crystal.titl
    lattice = crystal.unit_cell.direct
    uc_dict = crystal.unit_cell_atoms()
    positions = uc_dict["frac_pos"]
    elements = uc_dict["element"]
    asym_atoms = uc_dict["asym_atom"]
    asym_labels = uc_dict["label"]
    cell = lattice, positions, elements

    reduced_cell = standardize_cell(cell, **kwargs)

    if reduced_cell is None:
        LOG.warning("Could not find reduced cell for crystal %s", crystal)
        return None
    dataset = get_symmetry_dataset(reduced_cell)
    asym_idx = np.unique(dataset.equivalent_atoms)
    asym_idx = asym_idx[np.argsort(asym_atoms[asym_idx])]
    sg = SpaceGroup(dataset.number, choice=dataset.choice)

    reduced_lattice, positions, elements = reduced_cell
    unit_cell = UnitCell(reduced_lattice)
    asym = AsymmetricUnit(
        [Element[x] for x in elements[asym_idx]],
        positions[asym_idx],
        labels=asym_labels[asym_idx],
    )
    c_standard = Crystal(unit_cell, sg, asym)
    c_standard.titl = name
    return c_standard


def detect_symmetry(crystal, **kwargs):
    from cspy.crystal import Crystal, AsymmetricUnit, UnitCell

    lattice = crystal.unit_cell.direct
    uc_dict = crystal.unit_cell_atoms()
    positions = uc_dict["frac_pos"]
    elements = uc_dict["element"]
    asym_atoms = uc_dict["asym_atom"]
    asym_labels = uc_dict["label"]
    cell = lattice, positions, elements
    dataset = get_symmetry_dataset(cell, **kwargs)
    if dataset.number == crystal.space_group.international_tables_number:
        return None
    asym_idx = np.unique(dataset.equivalent_atoms)
    asym_idx = asym_idx[np.argsort(asym_atoms[asym_idx])]
    sg = SpaceGroup(dataset.number, choice=dataset.choice)
    asym = AsymmetricUnit(
        [Element[x] for x in dataset.std_types[asym_idx]],
        dataset.std_positions[asym_idx],
    )
    unit_cell = UnitCell(dataset.std_lattice)
    return Crystal(unit_cell, sg, asym)


def crystal_expansions(crystal, target_num_uc_mols, ratios=(1, 1, 1)):
    """Returns the expansions required to generate a supercell with a
    target number of unit cell molecules. This method tries to
    increase each cell axis length equally.

    Parameters
    ----------
    crystal : cspy.crystal.Crystal
        The input crystal to generate the supercell from.
    target_num_uc_mols : int
        The target number of unit cell molecules that the supercell
        should have.
    ratios : tuple
        The ratio the unit cell should be expanded with.
    Returns
    -------
        A tuple of the expansions.
    """
    num_uc_mols = len(crystal.unit_cell_molecules())
    if target_num_uc_mols <= num_uc_mols:
        return 1, 1, 1

    if target_num_uc_mols % num_uc_mols != 0:
        new_target = math.ceil(target_num_uc_mols / num_uc_mols) * num_uc_mols
        LOG.debug(
            "Unable to create an expanded crystal structure "
            "with %s molecules in the unit cell, will expand "
            "to a crystal structure with %s molecules in the "
            "unit cell instead",
            target_num_uc_mols,
            new_target,
        )
        target_num_uc_mols = new_target

    a, b, c = crystal.unit_cell.lengths
    r_a, r_b, r_c = ratios
    a = np.inf if r_a <= 1e-10 else a / r_a
    b = np.inf if r_b <= 1e-10 else b / r_b
    c = np.inf if r_c <= 1e-10 else c / r_c
    i, j, k = 1, 1, 1
    for prime in prime_factors(target_num_uc_mols // num_uc_mols):
        if a == min(a, b, c):
            i *= prime
            a *= prime
        elif b == min(a, b, c):
            j *= prime
            b *= prime
        elif c == min(a, b, c):
            k *= prime
            c *= prime
    return i, j, k


def prime_factors(n):
    """Return a list of prime factors of n.

    Parameters
    ----------
    n : int
        An integer that you want to break down into its prime factors.

    Returns
    -------
    factors : list of int
        A list of prime factors of n.
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    factors.sort(reverse=True)
    return factors


def get_contacts(mol_pos : np.array, 
                 mol_radii : list[float], 
                 neighs_pos : np.array, 
                 neighs_radii : list[float], 
                 tolerance : float=0.5,
                 get_vectors : bool=False
                 ) -> Tuple[list[int], list[float], np.array, list[np.array]]:
    
    """There is a contact between a molecule and its neighbours if the distance
        between any two atoms are less then the sum of their covalent
        radii + 0.5 ang which is slightly greater then CCDC
        recommendations for deciding whether two atoms are bonded,
        the sum of covalent radii + 0.4.

        Parameters
        ----------
        mols_pos : np.array
            An array of atoms positions of a molecule.

        mol_radii : list
            An list of covalent radii of the atoms in the molecule

        neighs_pos : np.array
            An array of positions of neighbouring atoms.

        neighs_radii : list
            An list of covalent radii of neighbouring atoms.

        tolerance : float
            Two atoms are in contact if the interatomic distance
            is less than the sum of their radii plus the tolerance.

        get_vectors : bool
            If true, calculate collision vectors. This adds cost and 
            isn't always needed.

        Returns
        -------
        contacts : list[int]
            List of indices of neighbouring atoms which are in 
            contact with molecule.

        distances : list[float]
            List of distances between neighbouring atoms which 
            are in contact with molecule and the molecule.

        box_idxs : np.array
            Indices of neighbouring atoms that are within a box
            of the molecule which is equal to the size of the 
            molecule plus 5 Angstrom.

        at_col_vectors : list[np.array]
            An atomic collision vector.
            This is the negative of the vector of the smallest 
            magnitude that would bring two atoms out of contact.
        """
    from scipy.spatial import cKDTree as KDTree

    contacts = []
    distances = []
    at_col_vectors = []

    # only generate neighs kd-tree for points within a box
    # around the test molecule
    mol_min = np.min(mol_pos, axis=0) - 5
    mol_max = np.max(mol_pos, axis=0) + 5

    box_idxs = np.all(
                np.logical_and(mol_min < neighs_pos, neighs_pos < mol_max), axis=1
            )
    
    if np.any(box_idxs):
        near_neighs_pos = neighs_pos[box_idxs]
        near_neighs_radii = neighs_radii[box_idxs]

        tree_1 = KDTree(mol_pos, compact_nodes=False, balanced_tree=False)
        tree_2 = KDTree(near_neighs_pos, compact_nodes=False, balanced_tree=False)
        rough_contacts = tree_1.query_ball_tree(tree_2, 5)

        for j, idxs in enumerate(rough_contacts):
            if len(idxs) == 0:
                continue
            diff = mol_pos[j] - near_neighs_pos[idxs]
            dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            # extra 0.1 A to accomodate a little variance
            sum_radii = mol_radii[j] + near_neighs_radii[idxs] + tolerance

            contacts.append(np.array(idxs)[dist < sum_radii])
            distances.append(dist[dist < sum_radii])

            if np.any(dist < sum_radii) and get_vectors:
                # if no collision, below gives float between 0 and 1
                col_factor = sum_radii / dist
                # set < 1 values to 1 so target_diff is unchanged
                col_factor[col_factor < 1] = 1
                col_factor = \
                    np.stack((col_factor, col_factor, col_factor), axis=1)
                target_diff = diff * col_factor
                neigh_col_vector = target_diff - diff
                # if atom dist = 0, set vector to sum_radii
                neigh_col_vector = np.nan_to_num(neigh_col_vector, \
                                                nan=np.stack((sum_radii, sum_radii, sum_radii), axis=1))
                neigh_x_vectors, neigh_y_vectors, neigh_z_vectors = np.hsplit(neigh_col_vector, 3)
                x_comp = np.min(np.minimum(0, neigh_x_vectors)) + np.max(np.maximum(0, neigh_x_vectors))
                y_comp = np.min(np.minimum(0, neigh_y_vectors)) + np.max(np.maximum(0, neigh_y_vectors))
                z_comp = np.min(np.minimum(0, neigh_z_vectors)) + np.max(np.maximum(0, neigh_z_vectors))
                at_col_vector = np.array([x_comp, y_comp, z_comp])
            else:
                at_col_vector = np.zeros(3)

            at_col_vectors.append(at_col_vector)

    return contacts, distances, box_idxs, at_col_vectors


def find_formula_unit(molecules: List[Molecule], metric: Literal["chemical", "conformational"] ="chemical") -> Tuple[List[Molecule], List[Molecule], int]:
    """Takes a list of cspy.chem.Molecule objects, finds which ones are unique (chemically or conformationally).
    This is then used to find the number of formula units in the list.

    Parameters
    ----------
    molecules : List[Molecule]
        A list of cspy Molecules
    metric : str
        "chemical" or "conformational"
        If 'chemical', compare connectivity graphs between molecules to see if
        they are chemically unique. (tautomers are treated as different molecules)
        If 'conformational', overlay molecules to see if molecules are
        conformationally unique.

    Returns
    -------
    formula_unit : List[Molecule]
        A list of molecules that define the formula unit

    unique_molecules : List[Molecule]
        From the list of molecules provided, which have unique connectivity

    Z : int
        The number of formula units in the list provided
    """
    from collections import Counter

    if metric not in ["chemical", "conformational"]:
        raise ValueError(
            "Invalid uniqueness metric, %s. Supported metrics are chemical and conformational.",
            metric,
        )

    if metric == "conformational":
        molecules_count_dict = Counter(molecules)
    elif metric == "chemical":
        molecules_count_dict = {}
        for mol1 in molecules:
            is_new = True
            for mol2 in molecules_count_dict.keys():
                if not sorted(mol1.elements) == sorted(mol2.elements):
                    continue
                if mol1.has_similar_molecular_graph_to(mol2):
                    is_new = False
                    molecules_count_dict[mol2] += 1
                    break
            if is_new:
                molecules_count_dict[mol1] = 1

    LOG.info(
        "Found %d unique molecules in the list of %d molecules",
        len(molecules_count_dict),
        len(molecules),
    )
    LOG.debug("molecules_count_dict: %s", molecules_count_dict)

    unique_molecules = list(molecules_count_dict.keys())

    Z = int(np.gcd.reduce(list(molecules_count_dict.values())))

    formula_unit = []
    for mol, count in molecules_count_dict.items():
        if count % Z != 0:
            LOG.warning(
                "Molecule %s has a count of %d, which is not divisible by the GCD %d. "
                "This may lead to unexpected results.",
                getattr(mol, "titl", str(mol)),
                count,
                Z,
            )
        formula_unit += [mol] * (count // Z)

    return formula_unit, unique_molecules, Z
