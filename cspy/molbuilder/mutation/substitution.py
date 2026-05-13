from rdkit import Chem
from rdkit.Chem import BondType
import cspy.molbuilder.utility as util
from cspy.molbuilder.exceptions import MoleculeMutationFailure


def type0_mutation(molecule, fragment, atom_position):
    """Carries out a type0 substitution mutation on the molecule.

    :param molecule: rdkit Mol object of the molecule to be mutated
    :param fragment: rdkit Mol object of the type0 mutation fragment
    :param atom_position: the position of the molecule to be mutated
    :return: rdkit Mol object of the molecule after mutation
    """
    combo = Chem.CombineMols(molecule, fragment)
    combo = Chem.RWMol(combo)

    substitution_neighbor_idxs = []
    for neighbors in combo.GetAtomWithIdx(atom_position).GetNeighbors():
        if neighbors.IsInRing():
            substitution_neighbor_idxs.append(neighbors.GetIdx())

    bond_types = {}
    for neighbor_idx in substitution_neighbor_idxs:

        bond_types[neighbor_idx] = molecule.GetBondBetweenAtoms(
            atom_position, neighbor_idx).GetBondType()

        combo.RemoveBond(atom_position, neighbor_idx)

    for neighbor_idx in substitution_neighbor_idxs:
        combo.AddBond(neighbor_idx, molecule.GetNumAtoms(),
                      bond_types[neighbor_idx])

    util.remove_smallest_fragment(combo)

    try:
        combo = Chem.RemoveHs(combo)
    except Exception as e:
        raise MoleculeMutationFailure(
            'Mutation failure with exception ({}) for mutation of molecule {} '
            'with fragment {}'.format(
                e, Chem.MolToSmiles(molecule), Chem.MolToSmiles(fragment)
            )
        )

    return Chem.MolFromSmiles(Chem.MolToSmiles(combo))


def type1_mutation(molecule, fragment, atom_position):
    """Carries out a type1 substitution mutation on the molecule.

    :param molecule: rdkit Mol object of the molecule to be mutated
    :param fragment: rdkit Mol object of the type1 mutation fragment
    :param atom_position: the position of the molecule to be mutated
    :return: rdkit Mol object of the molecule after mutation
    """
    combo = Chem.CombineMols(molecule, fragment)
    combo = Chem.RWMol(combo)

    substitution_neighbor_idxs = []
    for neighbors in combo.GetAtomWithIdx(atom_position).GetNeighbors():
        if neighbors.IsInRing():
            substitution_neighbor_idxs.append(neighbors.GetIdx())

    for neighbor_idx in substitution_neighbor_idxs:
        combo.RemoveBond(atom_position, neighbor_idx)

    for neighbor_idx in substitution_neighbor_idxs:
        combo.AddBond(neighbor_idx, molecule.GetNumAtoms(), BondType.SINGLE)

    util.remove_smallest_fragment(combo)

    try:
        combo = Chem.RemoveHs(combo)
    except Exception as e:
        raise MoleculeMutationFailure(
            'Mutation failure with exception ({}) for mutation of molecule {} '
            'with fragment {}'.format(
                e, Chem.MolToSmiles(molecule), Chem.MolToSmiles(fragment)
            )
        )

    return Chem.MolFromSmiles(Chem.MolToSmiles(combo))


def mutable_positions(molecule, fragment, fragments_list):
    """Returns all mutation positions for a given mutation fragment, the
    fragment and smarts string in the fragments_list should be the
    same type.

    :param molecule: rdkit Mol object of the molecule
    :param fragment: rdkit Mol object of the mutation fragment
    :param fragments_list: list of smarts string of all
        mutation possibilities
    :return: a tuple containing the atom indexes of the mutation
        positions for a given mutation fragment
    """
    already_substituted_positions = set(
        [i[0] for i in molecule.GetSubstructMatches(fragment)]
    )

    positions = []
    for fragment_smarts in fragments_list:
        positions += [
            i[0] for i in molecule.GetSubstructMatches(fragment_smarts)
        ]
    positions = set(positions)

    return sorted(list(positions - already_substituted_positions))
