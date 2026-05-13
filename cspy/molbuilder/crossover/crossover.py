import logging
from random import choice
from rdkit import Chem
import cspy.molbuilder.addition as ad
import cspy.molbuilder.fragmentation as fg
import cspy.molbuilder.utility as util
from .positions import crossover_positions


LOG = logging.getLogger(__name__)


class Crossover:

    def __init__(self, min_size, max_size):
        """Crossover object used to crossover two parent molecules.

        :param min_size: the minimum molecule size
        :param max_size: the maximum molecule size
        """
        self.min_size = min_size
        self.max_size = max_size

    def crossover(self, molecule_i, molecule_j):
        """Creates two child molecules by fragmenting molecule_i and
        molecule_j and recombining the fragments with each other.

        This is the function is the 'crossover operator' as
        explained in.

        Evolutionary Chemical Space Exploration for Functional
        Materials: Computational Organic Semiconductor Discovery

        :param molecule_i: rdkit Mol object of molecule_i
        :param molecule_j: rdkit Mol object of molecule_j
        :return: a tuple of two rdkit Mol object of the child molecules
        """
        if (util.num_joined_symm_sssr(molecule_i) == 1
        or util.num_joined_symm_sssr(molecule_j) == 1):
            return molecule_i, molecule_j

        positions = crossover_positions(molecule_i, molecule_j, self.min_size,
                                        self.max_size)

        if not positions:
            LOG.info('no possible crossover positions found for %s and %s '
                     'returning the initial molecules',
                     Chem.MolToSmiles(molecule_i),
                     Chem.MolToSmiles(molecule_j))
            return molecule_i, molecule_j

        position = choice(positions)

        frag_i_l, frag_i_r = fg.fragmentation(molecule_i, position[0])
        frag_j_l, frag_j_r = fg.fragmentation(molecule_j, position[1])

        # swap the molecules ordering for one set of the molecules so
        # the correct fragments are crossed over
        if position[2]:
            frag_j_l, frag_j_r = frag_j_r, frag_j_l

        bonding_idxs_0 = util.bonding_atom_idxs(frag_i_l)
        bonding_idxs_1 = util.bonding_atom_idxs(frag_j_l)

        # crossover the fragments to create the child molecules
        child_0 = ad.add_fragment(frag_i_l, frag_j_r, bonding_idxs_0)
        child_1 = ad.add_fragment(frag_j_l, frag_i_r, bonding_idxs_1)
        return child_0, child_1
