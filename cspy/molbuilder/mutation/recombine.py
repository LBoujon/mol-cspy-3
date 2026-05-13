import itertools as it
from random import choice
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Chem.rdqueries import AtomNumEqualsQueryAtom
import cspy.molbuilder.addition as ad
import cspy.molbuilder.fragmentation as fg
import cspy.molbuilder.utility as util
from cspy.molbuilder.exceptions import MoleculeRecombineFailure


def recombine(molecule, fragmentation_position):
    """Carries out a recombine mutation on the molecule by fragmenting
    the molecule moving the dummy position of where the fragmentation
    occurred and recombining them together.

    :param molecule: rdkit Mol object of the molecule to be mutated
    :param fragmentation_position: the fragmentation position
    :return: rdkit Mol object of the recombined molecule
    """
    frag_i, frag_j = fg.fragmentation(molecule, fragmentation_position)

    positions = None
    if len(fragmentation_position) == 2:
        positions = choice(recombine_two_point_positions(frag_i, frag_j))

    elif len(fragmentation_position) == 3:
        positions = choice(recombine_three_point_positions(frag_i, frag_j))

    elif len(fragmentation_position) == 4:
        positions = choice(recombine_four_point_positions(frag_i, frag_j))

    frag_i = recombine_dummy_position_switch(frag_i, positions[0])
    frag_j = recombine_dummy_position_switch(frag_j, positions[1])
    molecule = ad.add_fragment(frag_i, frag_j, util.bonding_atom_idxs(frag_i))

    return molecule


def recombine_two_point_positions(fragment_i, fragment_j):
    """Returns recombine two point positions.

    :param fragment_i: rdkit Mol object of fragment_i
    :param fragment_j: rdkit Mol object of fragment_j
    :return: a list of tuple containing two tuples which are the
        position the dummies on each fragments will be switched with
    """
    patterns = ['[#0R1,#6R1r6,#7R1r6,#6R1r5&!$([*R1]1-*@*@*@*-1),'
                '#7R1r5&!$([*R1]1-*@*@*@*-1)]@[#0R1,#6R1r6,#7R1r6,'
                '#6R1r5&!$([*R1]1-*@*@*@*-1),#7R1r5&!$([*R1]1-*@*@*@*-1)]',
                '[#0R2,CR2r6,NR2r6]@[#0R2,CR2r6,NR2r6]']

    Chem.Kekulize(fragment_i)
    Chem.Kekulize(fragment_j)

    positions_i = []
    positions_j = []

    for pattern in patterns:
        smarts = Chem.MolFromSmarts(pattern)
        positions_i += fragment_i.GetSubstructMatches(smarts)
        positions_j += fragment_j.GetSubstructMatches(smarts)

    positions = []
    for atom_idxs_i, atom_idxs_j in it.product(positions_i, positions_j):

        bond_i = fragment_i.GetBondBetweenAtoms(*atom_idxs_i)
        bond_j = fragment_j.GetBondBetweenAtoms(*atom_idxs_j)

        if (bond_i.GetBondType() == BondType.SINGLE
        and bond_j.GetBondType() == BondType.SINGLE
        and util.heteroatom_at_equal_dist(fragment_i, *atom_idxs_i)
        and util.heteroatom_at_equal_dist(fragment_j, *atom_idxs_j)):
            continue

        positions.append((atom_idxs_i, atom_idxs_j))

    Chem.SanitizeMol(fragment_i)
    Chem.SanitizeMol(fragment_j)
    return sorted(positions)


def recombine_three_point_positions(fragment_i, fragment_j):
    """Returns recombine three point positions. This function expects
    the male part (the part that will be used to attach to another
    molecule) from the three point fragmentation to be inputted through
    variable fragment_j.

    :param fragment_i: rdkit Mol object of fragment_i
    :param fragment_j: rdkit Mol object of fragment_j
    :return: a list of tuple containing two tuples which are the
        position the dummies on each fragments will be switched with
    """
    smarts_i = Chem.MolFromSmarts(
        '[#0R1r6,cR1r6&H]@[#0R2r6,cR2r6&H0]@[#0R1r6,cR1r6&H]'
    )
    smarts_j = Chem.MolFromSmarts('[#0R1,#6R1]@[#0R1,#6R1]@[#0R1,#6R1]')

    positions_i = fragment_i.GetSubstructMatches(smarts_i)
    positions_j = fragment_j.GetSubstructMatches(smarts_j)

    positions = []
    for atom_idxs_i, atom_idxs_j in it.product(positions_i, positions_j):

        ring_size_five_i = [fragment_i.GetAtomWithIdx(i).IsInRingSize(5)
                            for i in atom_idxs_i]
        ring_size_five_j = [fragment_j.GetAtomWithIdx(j).IsInRingSize(5)
                            for j in atom_idxs_j]

        if any(ring_size_five_i) and any(ring_size_five_j):
            continue

        positions.append((atom_idxs_i, atom_idxs_j))

    return sorted(positions)


def recombine_four_point_positions(fragment_i, fragment_j):
    smarts_i = Chem.MolFromSmarts(
        '[#0R1r6,cR1r6&H]@[#0R2,cR2&H0]@[#0R2,cR2&H0]@[#0R1r6,cR1r6&H]'
    )
    smarts_j = Chem.MolFromSmarts(
        '[#0R1,#6R1]@[#0R1,#6R1]@[#0R1,#6R1]@[#0R1,#6R1]'
    )

    positions_i = fragment_i.GetSubstructMatches(smarts_i)
    positions_j = fragment_j.GetSubstructMatches(smarts_j)

    positions = []
    for atom_idxs_i, atom_idxs_j in it.product(positions_i, positions_j):

        ring_size_five = [fragment_j.GetAtomWithIdx(j).IsInRingSize(5)
                          for j in atom_idxs_j]

        in_total_number_rings = 0
        for ring_atom_idxs in Chem.GetSymmSSSR(fragment_i):

            if (set(atom_idxs_i[1:3]).issubset(ring_atom_idxs)
            and any(ring_size_five) and len(ring_atom_idxs) == 5):
                continue

            if set(atom_idxs_i[1:3]).intersection(ring_atom_idxs):
                in_total_number_rings += 1

        if in_total_number_rings == 3:
            positions.append((atom_idxs_i, atom_idxs_j))

    return sorted(positions)


def recombine_dummy_position_switch(fragment, switch_idxs, flip=None):
    """Switches the three point dummy positions with the positions
    in switch_atom_idxs.

    :param fragment: rdkit Mol object of the fragment, must have three
        dummy positions
    :param switch_idxs: a tuple of the switch positions on
        the fragment
    :param flip: a boolean which changes how the dummies are switched
    :return: rdkit Mol object of the fragment with the dummy
        positions switched
    """
    dummy_atoms_idxs = util.bonding_atom_idxs(fragment)
    num_common_atoms = len(set(switch_idxs).intersection(dummy_atoms_idxs))

    # if the dummies and switch positions are the same return
    # the fragment
    if num_common_atoms == len(switch_idxs):
        return fragment

    switch_atom_neighbors, common_atom_neighbors, dummy_atom_neighbors \
        = neighbors_subroutine(fragment, switch_idxs, dummy_atoms_idxs)

    num_switch_neighbors = len(switch_atom_neighbors)
    num_common_neighbors = len(common_atom_neighbors)
    num_dummy_neighbors = len(dummy_atom_neighbors)
    neighbor_count = [num_switch_neighbors, num_common_neighbors,
                      num_dummy_neighbors]

    if flip is None:
        flip = choice([True, False])

    if num_dummy_neighbors == 0:
        return fragment

    elif neighbor_count == [1, 0, 1]:
        fragment = two_bond_switch_subroutine(
            fragment, switch_atom_neighbors, dummy_atom_neighbors
        )

    elif neighbor_count == [1, 1, 1]:
        fragment = three_bond_switch_subroutine(
            fragment, switch_atom_neighbors, common_atom_neighbors,
            dummy_atom_neighbors, flip
        )

    elif neighbor_count == [2, 0, 2]:
        fragment = four_bond_switch_subroutine(
            fragment, switch_atom_neighbors, dummy_atom_neighbors, flip
        )

    elif neighbor_count == [2, 1, 2]:
        fragment = five_bond_switch_subroutine(
            fragment, switch_atom_neighbors, common_atom_neighbors,
            dummy_atom_neighbors, switch_idxs, dummy_atoms_idxs, flip
        )

    elif neighbor_count == [3, 0, 3]:
        fragment = six_bond_switch_subroutine(
            fragment, switch_atom_neighbors, dummy_atom_neighbors, flip
        )

    elif neighbor_count == [3, 1, 3]:
        fragment = seven_bond_switch_subroutine(
            fragment, switch_atom_neighbors, common_atom_neighbors,
            dummy_atom_neighbors, switch_idxs, dummy_atoms_idxs, flip
        )

    elif neighbor_count == [4, 0, 4]:
        fragment = eight_bond_switch_subroutine(
            fragment, switch_atom_neighbors, dummy_atom_neighbors, flip
        )

    try:
        Chem.SanitizeMol(fragment)
    except Exception as e:
        raise MoleculeRecombineFailure(
            'Molecule recombination failure with exception ({}) for dummy '
            'switch of fragment {}'.format(e, Chem.MolToSmiles(fragment))
        )

    return fragment


def neighbors_subroutine(fragment, switch_atom_idxs, dummy_atoms_idxs):
    switch_atom_neighbors = []
    common_atom_neighbors = []
    dummy_atom_neighbors = []

    # get the atoms of the bonds we need to break for the neighbors of
    # the switch atoms
    for switch_idx in switch_atom_idxs:
        neighbors = fragment.GetAtomWithIdx(switch_idx).GetNeighbors()

        for neighbor_idx in [i.GetIdx() for i in neighbors]:
            bond = fragment.GetBondBetweenAtoms(switch_idx, neighbor_idx)

            if (not bond.IsInRing() or neighbor_idx in switch_atom_idxs
            or {switch_idx, neighbor_idx}.issubset(dummy_atoms_idxs)):
                continue

            if neighbor_idx not in dummy_atoms_idxs:
                switch_atom_neighbors.append([switch_idx, neighbor_idx])
            else:
                common_atom_neighbors.append([switch_idx, neighbor_idx])

    atom_neighbors = [sorted(i) for i in switch_atom_neighbors] + \
                     [sorted(i) for i in common_atom_neighbors]

    # get the atoms of the bonds we need to break for the neighbors of
    # the dummy atoms
    for dummy_idx in dummy_atoms_idxs:
        neighbors = fragment.GetAtomWithIdx(dummy_idx).GetNeighbors()

        for neighbor_idx in [i.GetIdx() for i in neighbors]:
            bond = fragment.GetBondBetweenAtoms(dummy_idx, neighbor_idx)

            if (not bond.IsInRing() or neighbor_idx in switch_atom_idxs
            or {dummy_idx, neighbor_idx}.issubset(dummy_atoms_idxs)
            or sorted([dummy_idx, neighbor_idx]) in atom_neighbors):
                continue

            dummy_atom_neighbors.append([dummy_idx, neighbor_idx])

    return switch_atom_neighbors, common_atom_neighbors, dummy_atom_neighbors


def two_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                               dummy_neighbor_idxs):
    # if the dummies are switched with a neighboring atom so there is
    # one common atom, this requires rotating the three atoms around
    # e.g. -*-*-1- to -1-*-*-, flip has no effect for this type
    # of switch
    switch_neigh_idxs = switch_neighbor_idxs[0]
    dummy_neigh_idxs = dummy_neighbor_idxs[0]

    fragment = Chem.RWMol(fragment)
    fragment.RemoveBond(*switch_neigh_idxs)
    fragment.RemoveBond(*dummy_neigh_idxs)

    dummy_neigh_idxs[1], switch_neigh_idxs[1] \
        = switch_neigh_idxs[1], dummy_neigh_idxs[1]

    fragment.AddBond(*switch_neigh_idxs, BondType.AROMATIC)
    fragment.AddBond(*dummy_neigh_idxs, BondType.AROMATIC)
    return fragment


def three_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                 common_neighbor_idxs, dummy_neighbor_idxs,
                                 flip):

    switch_neigh_idxs = switch_neighbor_idxs[0]
    common_neigh_idxs = common_neighbor_idxs[0]
    dummy_neigh_idxs = dummy_neighbor_idxs[0]

    switch_neigh_bond_type = fragment.GetBondBetweenAtoms(
        *switch_neigh_idxs
    ).GetBondType()

    common_neigh_bond_type = fragment.GetBondBetweenAtoms(
        *common_neigh_idxs
    ).GetBondType()

    dummy_neigh_bond_type = fragment.GetBondBetweenAtoms(
        *dummy_neigh_idxs
    ).GetBondType()

    fragment = Chem.RWMol(fragment)
    fragment.RemoveBond(*switch_neigh_idxs)
    fragment.RemoveBond(*common_neigh_idxs)
    fragment.RemoveBond(*dummy_neigh_idxs)

    if flip:
        switch_neigh_idxs[1], common_neigh_idxs[1], \
        dummy_neigh_idxs[0], dummy_neigh_idxs[1] \
            = dummy_neigh_idxs[0], dummy_neigh_idxs[1], \
              common_neigh_idxs[1], switch_neigh_idxs[1]
    else:
        dummy_neigh_idxs[1], switch_neigh_idxs[1] \
            = switch_neigh_idxs[1], dummy_neigh_idxs[1]

    fragment.AddBond(*switch_neigh_idxs, switch_neigh_bond_type)
    fragment.AddBond(*common_neigh_idxs, common_neigh_bond_type)
    fragment.AddBond(*dummy_neigh_idxs, dummy_neigh_bond_type)
    return fragment


def four_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                dummy_neighbor_idxs, flip):

    switch_neigh_idxs_0 = switch_neighbor_idxs[0]
    switch_neigh_idxs_1 = switch_neighbor_idxs[-1]
    dummy_neigh_idxs_0 = dummy_neighbor_idxs[0]
    dummy_neigh_idxs_1 = dummy_neighbor_idxs[-1]

    switch_neigh_bond_type = fragment.GetBondBetweenAtoms(
        *switch_neigh_idxs_0
    ).GetBondType()

    dummy_neigh_bond_type = fragment.GetBondBetweenAtoms(
        *dummy_neigh_idxs_0
    ).GetBondType()

    fragment = Chem.RWMol(fragment)
    fragment.RemoveBond(*switch_neigh_idxs_0)
    fragment.RemoveBond(*switch_neigh_idxs_1)
    fragment.RemoveBond(*dummy_neigh_idxs_0)
    fragment.RemoveBond(*dummy_neigh_idxs_1)

    if flip:
        switch_neigh_idxs_1[1], switch_neigh_idxs_0[1], \
        dummy_neigh_idxs_1[1], dummy_neigh_idxs_0[1] \
            = dummy_neigh_idxs_0[1], dummy_neigh_idxs_1[1], \
              switch_neigh_idxs_0[1], switch_neigh_idxs_1[1]
    else:
        switch_neigh_idxs_0[1], switch_neigh_idxs_1[1], \
        dummy_neigh_idxs_0[1], dummy_neigh_idxs_1[1] \
            = dummy_neigh_idxs_0[1], dummy_neigh_idxs_1[1], \
              switch_neigh_idxs_0[1], switch_neigh_idxs_1[1]

    fragment.AddBond(*switch_neigh_idxs_0, dummy_neigh_bond_type)
    fragment.AddBond(*switch_neigh_idxs_1, dummy_neigh_bond_type)
    fragment.AddBond(*dummy_neigh_idxs_0, switch_neigh_bond_type)
    fragment.AddBond(*dummy_neigh_idxs_1, switch_neigh_bond_type)

    if len(fragment.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0))) == 2:

        switch_bond = fragment.GetBondBetweenAtoms(switch_neigh_idxs_0[0],
                                                   switch_neigh_idxs_1[0])
        dummy_bond = fragment.GetBondBetweenAtoms(dummy_neigh_idxs_0[0],
                                                  dummy_neigh_idxs_1[0])

        if dummy_neigh_bond_type == BondType.AROMATIC:
            switch_bond.SetBondType(BondType.AROMATIC)
            switch_bond.SetIsAromatic(True)
        elif dummy_neigh_bond_type == BondType.DOUBLE:
            switch_bond.SetBondType(BondType.SINGLE)
            switch_bond.SetIsAromatic(False)
        elif dummy_neigh_bond_type == BondType.SINGLE:
            switch_bond.SetBondType(BondType.DOUBLE)
            switch_bond.SetIsAromatic(False)

        if switch_neigh_bond_type == BondType.AROMATIC:
            dummy_bond.SetBondType(BondType.AROMATIC)
            dummy_bond.SetIsAromatic(True)
        elif switch_neigh_bond_type == BondType.DOUBLE:
            dummy_bond.SetBondType(BondType.SINGLE)
            dummy_bond.SetIsAromatic(False)
        elif switch_neigh_bond_type == BondType.SINGLE:
            dummy_bond.SetBondType(BondType.DOUBLE)
            dummy_bond.SetIsAromatic(False)

    return fragment


def five_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                common_neighbor_idxs, dummy_neighbor_idxs,
                                switch_idxs, dummy_idxs, flip):

    if (common_neighbor_idxs[0][0] == switch_idxs[1]
    and common_neighbor_idxs[0][1] == dummy_idxs[1]):
        return four_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                           dummy_neighbor_idxs, flip)

    bonds_to_break = []
    for i, bond in enumerate(switch_neighbor_idxs):
        if bond[0] == switch_idxs[1]:
            bonds_to_break.append(switch_neighbor_idxs.pop(i))
            break

    for i, bond in enumerate(dummy_neighbor_idxs):
        if bond[0] == dummy_idxs[1]:
            bonds_to_break.append(dummy_neighbor_idxs.pop(i))
            break

    fragment = Chem.RWMol(fragment)

    fragment.RemoveBond(*bonds_to_break[0])
    fragment.RemoveBond(*bonds_to_break[1])

    fragment.AddBond(bonds_to_break[0][0], bonds_to_break[1][1],
                     BondType.AROMATIC)
    fragment.AddBond(bonds_to_break[1][0], bonds_to_break[0][1],
                     BondType.AROMATIC)

    return three_bond_switch_subroutine(
        fragment, [switch_neighbor_idxs[0]], common_neighbor_idxs,
        [dummy_neighbor_idxs[0]], flip
    )


def six_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                               dummy_neighbor_idxs, flip):

    fragment = Chem.RWMol(fragment)
    fragment.RemoveBond(*switch_neighbor_idxs[1])
    fragment.RemoveBond(*dummy_neighbor_idxs[1])

    fragment.AddBond(switch_neighbor_idxs[1][0], dummy_neighbor_idxs[1][1],
                     BondType.AROMATIC)
    fragment.AddBond(switch_neighbor_idxs[1][1], dummy_neighbor_idxs[1][0],
                     BondType.AROMATIC)

    return four_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                       dummy_neighbor_idxs, flip)


def seven_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                 common_neighbor_idxs, dummy_neighbor_idxs,
                                 switch_idxs, dummy_idxs, flip):

    if switch_neighbor_idxs[1][0] != switch_idxs[1]:
        switch_neighbor_idxs.reverse()

    if dummy_neighbor_idxs[1][0] != dummy_idxs[1]:
        dummy_neighbor_idxs.reverse()

    fragment = three_bond_switch_subroutine(
        fragment, [switch_neighbor_idxs[1]], common_neighbor_idxs,
        [dummy_neighbor_idxs[1]], flip
    )

    return four_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                       dummy_neighbor_idxs, flip)


def eight_bond_switch_subroutine(fragment, switch_neighbor_idxs,
                                 dummy_neighbor_idxs, flip):

    fragment = four_bond_switch_subroutine(
        fragment, [switch_neighbor_idxs[1], switch_neighbor_idxs[2]],
        [dummy_neighbor_idxs[1], dummy_neighbor_idxs[2]], flip
    )

    fragment = four_bond_switch_subroutine(
        fragment, [switch_neighbor_idxs[0], switch_neighbor_idxs[3]],
        [dummy_neighbor_idxs[0], dummy_neighbor_idxs[3]], flip
    )

    return fragment
