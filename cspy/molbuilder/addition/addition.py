from random import choice
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Chem.rdqueries import AtomNumEqualsQueryAtom
import cspy.molbuilder.utility as util
from cspy.molbuilder.exceptions import FragmentAdditionFailure


def addition(molecule, fragments, max_size):
    """Adds a fragment from the fragments list to the molecule if
    possible.

    This is the function is a part of the 'addition operator' as
    explained in.

    Evolutionary Chemical Space Exploration for Functional
    Materials: Computational Organic Semiconductor Discovery

    Note that the paper above refers to a single fragment (a list of a
    single fragment) while the code below is more general and works on
    the addition of any fragment from the list of fragments.

    :param molecule: rdkit Mol object of the molecule
    :param fragments: a list of rdkit Mol object of the fragments
    :param max_size: the maximum molecule size that addition can make
    :return: None if no additions can be carried out and a rdkit
        molecule object if an addition was carried out
    """
    mol_size = util.num_joined_symm_sssr(molecule)

    # get all fragment addition possibilities
    addition_possibilities = []
    for fragment in fragments:
        frag_size = util.num_joined_symm_sssr(fragment)

        if mol_size + frag_size > max_size:
            continue

        for position in attachment_positions(molecule, fragment):
            addition_possibilities.append((fragment, position))

    # if there are no possibilities return None
    if len(addition_possibilities) == 0:
        return None

    fragment, attach_point = choice(addition_possibilities)
    molecule = add_fragment(molecule, fragment, attach_point)
    return molecule


def add_fragment(molecule, fragment, attach_idxs, flip=None):
    """Adds together a molecule and fragment using the fragments dummy
    atom positions and the molecules atom positions from the attach_idxs
    tuple.

    :param molecule: rdkit Mol object of the molecule
    :param fragment: rdkit Mol object of the fragment
    :param attach_idxs: tuple of atom positions on the molecule the
        fragment will bond to
    :param flip: a boolean which switches the attachment atom positions
        of the molecule so the molecule is flipped and may produce a
        different combined molecule
    :return: rdkit Mol object of the combined molecule
    """
    if flip is None:
        flip = choice([True, False])

    if flip:
        mol_attach_idx_0 = attach_idxs[-1]
        mol_attach_idx_1 = attach_idxs[0]
    else:
        mol_attach_idx_0 = attach_idxs[0]
        mol_attach_idx_1 = attach_idxs[-1]

    combo = Chem.CombineMols(molecule, fragment)
    combo = Chem.RWMol(combo)

    # get all dummy atoms indexes of the fragment part of the combined
    # molecule
    dummy_atoms = combo.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0))
    frag_attach_idxs = []
    for atom_idx in [i.GetIdx() for i in dummy_atoms]:
        if atom_idx >= molecule.GetNumAtoms():
            frag_attach_idxs.append(atom_idx)

    # get the fragment attachment atoms and the bonding atoms that are
    # next to them
    frag_attach_and_bonding_idxs = []
    for attach_idx in frag_attach_idxs:
        neighbor_atoms = combo.GetAtomWithIdx(attach_idx).GetNeighbors()
        for neighbor_idx in [i.GetIdx() for i in neighbor_atoms]:
            if neighbor_idx not in frag_attach_idxs:
                frag_attach_and_bonding_idxs.append((attach_idx, neighbor_idx))

    frag_bond_0 = combo.GetBondBetweenAtoms(*frag_attach_and_bonding_idxs[0])
    frag_bond_1 = combo.GetBondBetweenAtoms(*frag_attach_and_bonding_idxs[1])

    frag_attach_idx_0, frag_bonding_idx_0 = frag_attach_and_bonding_idxs[0]
    frag_attach_idx_1, frag_bonding_idx_1 = frag_attach_and_bonding_idxs[1]

    # create the combined molecule and fragment then remove the excess
    # dummies
    if (frag_bond_0.GetBondType() == BondType.SINGLE
    and frag_bond_1.GetBondType() == BondType.SINGLE):
        combo.AddBond(mol_attach_idx_0, frag_bonding_idx_0, BondType.SINGLE)
        combo.AddBond(mol_attach_idx_1, frag_bonding_idx_1, BondType.SINGLE)
    else:
        combo.AddBond(mol_attach_idx_0, frag_bonding_idx_0, BondType.AROMATIC)
        combo.AddBond(mol_attach_idx_1, frag_bonding_idx_1, BondType.AROMATIC)

        if len(attach_idxs) == 2:
            mol_attach_bond = combo.GetBondBetweenAtoms(*attach_idxs)
            mol_attach_bond.SetIsAromatic(True)

    combo.RemoveBond(frag_attach_idx_0, frag_bonding_idx_0)
    combo.RemoveBond(frag_attach_idx_1, frag_bonding_idx_1)
    util.remove_smallest_fragment(combo)
    util.replace_dummies(combo)

    try:
        Chem.SanitizeMol(combo)
    except Exception as e:
        raise FragmentAdditionFailure(
            'Fragment addition failure with exception ({}) for addition of '
            'fragment {} to molecule {}'.format(e, Chem.MolToSmiles(fragment),
            Chem.MolToSmiles(molecule))
        )

    return Chem.MolFromSmiles(Chem.MolToSmiles(combo))


def attachment_positions(molecule, fragment):
    """Returns a list of all attachment positions of the molecule for
    the inputted fragment.

    :param molecule: rdkit Mol object of the molecule
    :param fragment: rdkit Mol object of the fragment
    :return: a list of all attachment positions of the molecule the
        fragment can attach to
    """
    number_of_dummy_atoms \
        = len(fragment.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0)))

    if number_of_dummy_atoms == 2:
        return two_point_attachment_positions(molecule, fragment)
    elif number_of_dummy_atoms == 3:
        return three_point_attachment_positions(molecule)
    elif number_of_dummy_atoms == 4:
        return four_point_attachment_positions(molecule, fragment)


def two_point_attachment_positions(molecule, fragment):
    """Returns a list of atom indexes that are able to undergo a two
    point fragment addition. The two point attachment position is given
    by a sorted tuple of two atom positions.

    :param molecule: rdkit Mol object of the molecule
    :param fragment: rdkit Mol object of the fragment
    :return: a list of the two point attachment positions
    """
    patterns = ['[#6R1&H]@[#6R1&H]', '[CR2&H]=[CR2&H]']

    Chem.Kekulize(fragment)
    Chem.Kekulize(molecule)

    positions = []
    for pattern in [Chem.MolFromSmarts(i) for i in patterns]:

        dummy_atoms = fragment.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0))
        dummy_idxs = [i.GetIdx() for i in dummy_atoms]
        dummy_bond = fragment.GetBondBetweenAtoms(*dummy_idxs)

        for attach_idxs in molecule.GetSubstructMatches(pattern):
            attach_bond = molecule.GetBondBetweenAtoms(*attach_idxs)

            if (attach_bond.GetBondType() == BondType.SINGLE
            and dummy_bond.GetBondType() == BondType.SINGLE
            and util.heteroatom_at_equal_dist(molecule, *attach_idxs)
            and util.heteroatom_at_equal_dist(fragment, *dummy_idxs)):
                continue

            positions.append(attach_idxs)

    Chem.SanitizeMol(fragment)
    Chem.SanitizeMol(molecule)
    return sorted(positions)


def three_point_attachment_positions(molecule):
    """Returns a list of three point attachment position of a molecule.
    The three point attachment position is given by a sorted tuple of
    three atom indexes, starting from the lowest atom index and moving
    around the attachment position by bond. These positions are between
    two six membered ring. This is used for the determination of
    positions where we can add fragments to in something like a
    naphthalene ring system. E.g.

    naphthalene + five_membered_ring_fragment -> thacenaphthylene

    :param molecule: rdkit Mol object of the molecule
    :return: a list of the three point attachment positions
    """
    pattern = Chem.MolFromSmarts('[cR1r6&H]@[cR2r6&H0]@[cR1r6&H]')
    positions = list(molecule.GetSubstructMatches(pattern))
    return sorted(positions)


def four_point_attachment_positions(molecule, fragment):
    """Returns a list of four point attachment atom positions of a
    molecule. The four point attachment position is given by a sorted
    tuple of four atom positions, starting from the lowest atom index
    and moving around the attachment position by bond. This is
    constrained so that only the center ring can have a size of five or
    six. If the center ring is size five then a fragment with a ring of
    size five cannot undergo four point addition to it.

    :param molecule: rdkit Mol object of the molecule
    :param fragment: rdkit Mol object of the fragment
    :return: a list of the four point attachment positions
    """
    smarts = Chem.MolFromSmarts('[cR1r6&H]@[cR2&H0]@[cR2&H0]@[cR1r6&H]')

    dummy_atoms = fragment.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0))
    frag_ring_size_five = any(i.IsInRingSize(5) for i in dummy_atoms)

    # add a attachment positions if bond is touching a total of three
    # different ring systems and both the center ring and fragment rings
    # are not size five
    positions = []
    for atom_idxs in molecule.GetSubstructMatches(smarts):

        in_total_number_rings = 0
        for ring_atom_idxs in Chem.GetSymmSSSR(molecule):

            if (set(atom_idxs[1:3]).issubset(ring_atom_idxs)
            and frag_ring_size_five and len(ring_atom_idxs) == 5):
                continue

            if set(atom_idxs[1:3]).intersection(ring_atom_idxs):
                in_total_number_rings += 1

        if in_total_number_rings == 3:
            positions.append(atom_idxs)

    return sorted(positions)
