import itertools as it
from rdkit import Chem
from rdkit.Chem import BondType
import cspy.molbuilder.fragmentation as fg
import cspy.molbuilder.utility as util


def crossover_positions(molecule_i, molecule_j, min_size, max_size):
    """Find all the crossover positions for molecule_i and molecule_j
    with a restriction on the molecule size.

    :param molecule_i: rdkit Mol object of molecule_i
    :param molecule_j: rdkit Mol object of molecule_j
    :param min_size: minimum size of the child molecules
    :param max_size: maximum size of the child molecules
    :return: a list of tuples of two tuples and one boolean where the
        first tuple is the fragmentation position of molecule_i, the
        second tuple is the fragmentation position of molecule_j and a
        boolean which states which fragments are to be combined
    """
    positions_i = fg.fragmentation_positions(molecule_i)
    positions_j = fg.fragmentation_positions(molecule_j)

    crossover_count_0 = {
        (i, j): 0 for i, j in it.product(positions_i, positions_j)
    }
    crossover_count_1 = {
        (i, j): 0 for i, j in it.product(positions_i, positions_j)
    }

    for position_i, position_j in it.product(positions_i, positions_j):

        # crossover positions must be the same type both two point,
        # three point or four point
        if len(position_i) != len(position_j):
            continue

        main_ring_idx_i = util.main_joined_ring_idx(molecule_i, position_i)
        main_ring_idx_j = util.main_joined_ring_idx(molecule_j, position_j)

        frag_ring_idxs_i = util.joined_ring_connection_idxs(
            molecule_i, main_ring_idx_i, position_i
        )
        frag_ring_idxs_j = util.joined_ring_connection_idxs(
            molecule_j, main_ring_idx_j, position_j
        )

        # for each fragmentation position on a given molecule a set of
        # two fragment ring indexes are created, here we iterate through
        # each fragment ring index and test if they can be recombined
        for (q, ring_idxs_i), (r, ring_idxs_j) in it.product(
                enumerate(frag_ring_idxs_i), enumerate(frag_ring_idxs_j)):

            # child molecules sizes after crossover should not be above
            # or below the size limits
            if (len(ring_idxs_i + ring_idxs_j) < min_size
            or len(ring_idxs_i + ring_idxs_j) > max_size):
                continue

            # ensure that child molecules are valid for two point
            # crossovers this is important when crossing over molecules
            # with five membered rings with heteroatoms
            if len(position_i) == 2:

                Chem.Kekulize(molecule_i)
                Chem.Kekulize(molecule_j)

                bond_type_i = molecule_i.GetBondBetweenAtoms(
                    *position_i
                ).GetBondType()
                bond_type_j = molecule_j.GetBondBetweenAtoms(
                    *position_j
                ).GetBondType()

                hetero_equal_dist_i = util.heteroatom_at_equal_dist_for_conn(
                    molecule_i, ring_idxs_i, *position_i
                )
                hetero_equal_dist_j = util.heteroatom_at_equal_dist_for_conn(
                    molecule_j, ring_idxs_j, *position_j
                )

                Chem.SanitizeMol(molecule_i)
                Chem.SanitizeMol(molecule_j)

                if (bond_type_i == BondType.SINGLE and hetero_equal_dist_i
                and bond_type_j == BondType.SINGLE and hetero_equal_dist_j):
                    continue

            # for three and four point crossover child molecules should
            # not have the main ring, the ring that contains all
            # fragment positions, of both parent molecules this is
            # because of the asymmetry in the three and four point
            # fragmentation
            if ((len(position_i) == 3 or len(position_i) == 4)
            and main_ring_idx_i in ring_idxs_i
            and main_ring_idx_j in ring_idxs_j):
                continue

            # keep track of the crossover count so we know it is
            # possible to create two child molecules if the count
            # reaches two
            if q == r:
                crossover_count_0[(position_i, position_j)] += 1
            else:
                crossover_count_1[(position_i, position_j)] += 1

    positions = []
    for key in [key for key, value in crossover_count_0.items() if value == 2]:
        positions.append((*key, True))
    for key in [key for key, value in crossover_count_1.items() if value == 2]:
        positions.append((*key, False))

    return sorted(positions)
