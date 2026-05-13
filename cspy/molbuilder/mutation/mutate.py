from random import choice
from rdkit import Chem
from cspy.molbuilder.fragmentation import fragmentation_positions
from .recombine import recombine
from .substitution import mutable_positions
from .substitution import type0_mutation
from .substitution import type1_mutation


class Mutator:

    def __init__(self, type0_frag_smarts, type1_frag_smarts, protected_smarts):
        """Mutator object used to mutate a molecule, the type of the
        fragments smarts are made to ensure the mutation occur at the
        correct positions in the molecule.

        :param type0_frag_smarts: a list of smarts string used for
            matching type0 mutatble positions and also used to carry
            out the mutation
        :param type1_frag_smarts: a list of smarts string used for
            matching type1 mutatble positions and also used to carry out
            the mutation
        :param protect_smarts: a list of smarts strings used for
            matching positions that should not be mutated
        """
        self.type0_frag = [Chem.MolFromSmarts(i) for i in type0_frag_smarts]
        self.type1_frag = [Chem.MolFromSmarts(i) for i in type1_frag_smarts]
        self.protected = [Chem.MolFromSmarts(i) for i in protected_smarts]

    def multiple_mutations(self, molecule, number_of_mutations):
        """A convenience method to carry out multiple random mutations.

        :param molecule: rdkit Mol object of the molecule to be mutated
        :param number_of_mutations: number of mutations to carry out on
            the molecule
        :return: rdkit Mol object of the mutated molecule
        """
        for i in range(number_of_mutations):
            molecule = self.mutation(molecule)
        return molecule

    def mutation(self, molecule):
        """Carries out a random mutation on the molecule, this matches a
        mutation position using the smarts strings from
        self.type0_fragments or self.type0_fragments and then replaces
        the position using one of the smarts strings.

        This is the function is the 'mutation operator' as explained in.

        Evolutionary Chemical Space Exploration for Functional
        Materials: Computational Organic Semiconductor Discovery

        :param molecule: rdkit Mol object of the molecule to be mutated
        :return: rdkit Mol object of the mutated molecule
        """
        protected_positions = set()
        for i, protected in enumerate(self.protected):
            for positions in molecule.GetSubstructMatches(protected):
                protected_positions.update(set(positions))

        mutatable_position_list = []
        for i, frag in enumerate(self.type0_frag):
            for position in mutable_positions(molecule, frag, self.type0_frag):
                if position in protected_positions:
                    continue
                mutatable_position_list.append([0, i, position])

        for i, frag in enumerate(self.type1_frag):
            for position in mutable_positions(molecule, frag, self.type1_frag):
                if position in protected_positions:
                    continue
                mutatable_position_list.append([1, i, position])

        if len(mutatable_position_list) == 0:
            return molecule

        random_mutation_choice = choice(mutatable_position_list)
        atom_position = random_mutation_choice[2]

        if random_mutation_choice[0] == 0:
            frag = self.type0_frag[random_mutation_choice[1]]
            return type0_mutation(molecule, frag, atom_position)
        elif random_mutation_choice[0] == 1:
            frag = self.type1_frag[random_mutation_choice[1]]
            return type1_mutation(molecule, frag, atom_position)

    def recombination(self, molecule):
        """Carries out a random recombine mutation on the molecule by
        fragmenting the molecule, moving the dummy position of where the
        fragmentation occurred and recombing them together.

        This is the function is the 'recombination operator' as
        explained in.

        Evolutionary Chemical Space Exploration for Functional
        Materials: Computational Organic Semiconductor Discovery

        :param molecule: rdkit Mol object of the molecule to be mutated
        :return: rdkit Mol object of the mutated molecule
        """
        position = choice(fragmentation_positions(molecule))
        return recombine(molecule, position)
