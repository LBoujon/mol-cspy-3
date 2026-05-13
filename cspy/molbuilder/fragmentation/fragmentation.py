import copy
import itertools as it
from rdkit import Chem
from rdkit.Chem import BondType
import cspy.molbuilder.utility as util
from cspy.molbuilder.exceptions import MoleculeFragmentationFailure


def fragmentation(molecule, atom_positions, dummies_i=True, dummies_j=True):
    """Fragments input molecule_i along the specified atom_positions.
    For two point fragmentations the part of the molecule with the
    lowest atom index out of the two rings between the fragmentation
    bond will be the first of the two fragments in the tuple that is
    returned from this function. For three or four point fragmentations
    the male part (the part that will be used to attach to another
    molecule) will be always returned in the second position in the
    returned tuple of fragments.

    For three point fragmentations errors may occur if
    dummies_molecule_i or dummies_molecule_j is set to false due to
    kekulization errors.

    :param molecule: rdkit Mol object of the molecule to be fragmented
    :param atom_positions: atoms position along where the fragmentation
        will occur
    :param dummies_i: boolean to select whether dummies will be added to
        the first fragment
    :param dummies_j: boolean to select whether dummies will be added to
        the second fragment
    :return: a tuple of the two rdkit Mol objects of the fragments
    """

    # find the atom indexes of the fragmentation ring, this will be any
    # of the rings that has all atom_positions in it, for two point
    # fragmentation this is the ring with the larger atom index, for
    # three and four point there is only one ring
    fragment_ring_atom_idxs = None
    for ring_atom_idxs in reversed(util.joined_symm_sssr(molecule)):
        if set(atom_positions).issubset(ring_atom_idxs):
            fragment_ring_atom_idxs = ring_atom_idxs
            break

    # get the atoms of the bonds we need to break
    bonds_to_break_i = []
    bonds_to_break_j = []
    for atom_idx in atom_positions:
        for neighbor in molecule.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in atom_positions:
                if neighbor_idx in fragment_ring_atom_idxs:
                    bonds_to_break_i.append([atom_idx, neighbor_idx])
                else:
                    bonds_to_break_j.append([atom_idx, neighbor_idx])

    bonds_to_break_i, bonds_to_break_j = sort_bonds_to_break(
        molecule, bonds_to_break_i, bonds_to_break_j
    )

    molecule_j = copy.deepcopy(molecule)
    molecule_i = Chem.RWMol(molecule)
    molecule_j = Chem.RWMol(molecule_j)

    if dummies_i:
        util.set_dummies(molecule_i, atom_positions)
    if dummies_j:
        util.set_dummies(molecule_j, atom_positions)

    for bond_to_break in bonds_to_break_i:
        molecule_i.RemoveBond(*bond_to_break)
    for bond_to_break in bonds_to_break_j:
        molecule_j.RemoveBond(*bond_to_break)

    if len(atom_positions) == 2:
        correct_bonding(molecule_i, atom_positions)
        correct_bonding(molecule_j, atom_positions)

    molecule_i = remove_fragment(molecule_i, bonds_to_break_i[0][1])
    molecule_j = remove_fragment(molecule_j, bonds_to_break_j[0][1])

    try:
        Chem.SanitizeMol(molecule_i)
        Chem.SanitizeMol(molecule_j)
    except Exception as e:
        raise MoleculeFragmentationFailure(
            'Molecule fragmentation failure with exception ({}) for '
            'fragmentation of molecule {}'.format(e,
            Chem.MolToSmiles(molecule))
        )

    return molecule_i, molecule_j


def correct_bonding(molecule, atom_idxs):
    """For certain molecules like c1ccc2=NC(=O)N=c2c1 fragmentation
    causes the bonds to not be the correct type.

    :param molecule: rdkit Mol object
    :param atom_idxs: the fragmentation position
    """
    bonds = list(it.chain.from_iterable(
        [molecule.GetAtomWithIdx(i).GetBonds() for i in atom_idxs]
    ))

    neighbor_bond_test = False
    for bond in bonds:
        bond_atom_idxs = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if (sorted(bond_atom_idxs) != sorted(atom_idxs)
        and bond.GetBondType() == BondType.DOUBLE):
            neighbor_bond_test = True
            break

    if neighbor_bond_test:
        molecule.GetBondBetweenAtoms(*atom_idxs).SetBondType(BondType.SINGLE)


def sort_bonds_to_break(molecule, bonds_to_break_i, bonds_to_break_j):
    """Sort the bonds_to_break to ensure that after the bonds are broken
    the fragments are returned in the correct order in the fragmentation
    function.

    :param molecule: rdkit Mol object
    :param bonds_to_break_i: a list of list of atoms index of the bonds
        to be broken
    :param bonds_to_break_j: a list of list of atoms index of the bonds
        to be broken
    :return: a tuple of bonds_to_break in the correct order
    """
    atoms_i = set(it.chain.from_iterable(bonds_to_break_i))
    atoms_j = set(it.chain.from_iterable(bonds_to_break_j))

    min_ring_atom_idx_i = None
    min_ring_atom_idx_j = None

    for ring_atom_idxs in util.joined_symm_sssr(molecule):
        if ((min_ring_atom_idx_i is None
        or min(ring_atom_idxs) < min_ring_atom_idx_i)
        and atoms_i.issubset(ring_atom_idxs)):
            min_ring_atom_idx_i = min(ring_atom_idxs)

    for ring_atom_idxs in util.joined_symm_sssr(molecule):
        if ((min_ring_atom_idx_j is None
        or min(ring_atom_idxs) < min_ring_atom_idx_j)
        and atoms_j.issubset(ring_atom_idxs)):
            min_ring_atom_idx_j = min(ring_atom_idxs)

    if min_ring_atom_idx_i is None:
        return bonds_to_break_j, bonds_to_break_i
    elif min_ring_atom_idx_j is None:
        return bonds_to_break_i, bonds_to_break_j
    elif min_ring_atom_idx_i < min_ring_atom_idx_j:
        return bonds_to_break_j, bonds_to_break_i
    else:
        return bonds_to_break_i, bonds_to_break_j


def remove_fragment(combo, atom_idx):
    """Removes a fragment by its atom index.

    :param combo: rdkit Mol object of the combo
    :param atom_idx: the atom index of the fragment that is required to
        be removed
    :return: an rdkit Mol object with one of the fragments removed
    """
    fragment_id = None
    for i, fragment_atom_idxs in enumerate(Chem.GetMolFrags(combo)):
        if atom_idx in fragment_atom_idxs:
            fragment_id = i
            break

    combo = Chem.RWMol(combo)
    for atom_idx in reversed(Chem.GetMolFrags(combo)[fragment_id]):
        combo.RemoveAtom(atom_idx)

    return combo


def fragmentation_positions(molecule):
    """Returns two, three and four point fragmentation positions.

    :param molecule: rdkit Mol object
    :return: a list of all fragmentation positions
    """
    return two_point_fragmentation_positions(molecule) \
           + three_point_fragmentation_positions(molecule) \
           + four_point_fragmentation_positions(molecule)


def two_point_fragmentation_positions(molecule):
    """Returns two point fragmentation positions.

    :param molecule: rdkit Mol object
    :return: a list of all two point fragmentation positions
    """
    positions = two_point_fragmentation_positions_subroutine(
        molecule, '[#6R2](@[#6R2](@[*])@[*])(@[*])@[*]', 2
    )
    positions += two_point_fragmentation_positions_subroutine(
        molecule, '[#6R3](@[#6R3](@[*])@[*])(@[*])@[*]', 3
    )
    positions += two_point_fragmentation_positions_subroutine(
        molecule, '[#6R4](@[#6R4](@[*])@[*])(@[*])@[*]', 4
    )
    return sorted(positions)


def two_point_fragmentation_positions_subroutine(molecule, pattern,
                                                 num_bond_rings):
    """A subroutine used in the two_point_fragmentation_positions
    function, includes some tests to ensure the correct fragmentation
    positions are returned.

    :param molecule: rdkit Mol object
    :param pattern: smarts pattern match string
    :param num_bond_rings: the number of rings the bond with the atoms
        indexes from the first two atom indexes in the pattern match
        should be in
    :return: a list of two point fragmentation positions
    """
    positions = []
    for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
        atom_idx_i, atom_idx_j = match[:2]
        bond_in_num_rings = 0
        for ring_atom_idx in Chem.GetSymmSSSR(molecule):
            if atom_idx_i in ring_atom_idx and atom_idx_j in ring_atom_idx:
                bond_in_num_rings += 1

        if bond_in_num_rings == num_bond_rings:
            positions.append((atom_idx_i, atom_idx_j))

    return positions


def three_point_fragmentation_positions(molecule):
    """Returns three point fragmentation positions.

    :param molecule: rdkit Mol object
    :return: a list of three point fragmentation positions
    """
    pattern = Chem.MolFromSmarts('[cR2&H0]@[cR3&H0]@[cR2&H0]')
    matched = list(molecule.GetSubstructMatches(pattern))

    positions = []
    for ring_atom_idxs in Chem.GetSymmSSSR(molecule):
        # to ensure fragments are kekulizable for six membered rings the
        # ring attached to the three point fragmentation positions
        # should have two different fragmentation positions
        if len(ring_atom_idxs) == 6:
            for position_i, position_j in it.combinations(matched, 2):
                total_positions = position_i + position_j

                ring_size_five = [molecule.GetAtomWithIdx(i).IsInRingSize(5)
                                  for i in total_positions]

                if (set(total_positions).issubset(ring_atom_idxs)
                and not any(ring_size_five)):
                    positions.append(position_i)
                    positions.append(position_j)

        if len(ring_atom_idxs) == 5:
            for pattern_matched_position in matched:
                if set(pattern_matched_position).issubset(ring_atom_idxs):
                    positions.append(pattern_matched_position)

    # sort the list so that the the middle atom is in the middle of the
    # tuple and the lowest atom index is at the beginning
    for i, atoms_idxs in enumerate(positions):
        if atoms_idxs[0] < atoms_idxs[2]:
            positions[i] = atoms_idxs
        else:
            positions[i] = (atoms_idxs[2], atoms_idxs[1], atoms_idxs[0])

    return sorted(positions)


def four_point_fragmentation_positions(molecule):
    """Returns four point fragmentation positions.

    :param molecule: rdkit Mol object
    :return: a list of four point fragmentation positions
    """
    pattern = Chem.MolFromSmarts('[cR2&H0]@[cR3&H0]@[cR3&H0]@[cR2&H0]')

    positions = []
    for ring_atom_idx in Chem.GetSymmSSSR(molecule):
        for atom_idxs in molecule.GetSubstructMatches(pattern):

            # the smarts pattern matches for ring double check this for
            # cages instead
            if (util.atom_in_num_joined_symm_sssr(molecule, atom_idxs[0]) != 2
            or util.atom_in_num_joined_symm_sssr(molecule, atom_idxs[1]) != 3
            or util.atom_in_num_joined_symm_sssr(molecule, atom_idxs[2]) != 3
            or util.atom_in_num_joined_symm_sssr(molecule, atom_idxs[3]) != 2):
                continue

            if set(atom_idxs).issubset(ring_atom_idx):
                # if there are two adjacent five membered ring the match
                # pattern is not a four point fragmentation position but
                # actually two adjacent three point fragmentation
                # positions
                if ((molecule.GetAtomWithIdx(atom_idxs[0]).IsInRingSize(5)
                or molecule.GetAtomWithIdx(atom_idxs[3]).IsInRingSize(5))
                and len(ring_atom_idx) == 6):
                    continue

                # sort the atoms indexes so the index with the lowest
                # value is at the beginning with the remaining atom
                # indexes sorted by following around the position by
                # bond
                if atom_idxs[0] > atom_idxs[3]:
                    atom_idxs = (atom_idxs[3], atom_idxs[2],
                                 atom_idxs[1], atom_idxs[0])

                positions.append(atom_idxs)

    return sorted(positions)
