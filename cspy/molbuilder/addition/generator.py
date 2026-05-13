from random import choice
from random import randint
from rdkit import Chem
import cspy.molbuilder.utility as util
from .addition import addition


class Generator:

    def __init__(self, molecules, fragments, min_size, max_size):
        """Generator object is used to create molecules from user
        inputted molecule and fragment smiles up to a given minimum and
        maximum size determined by the number of joined rings (rings and
        cages) in the molecule.

        :param fragment_smiles: a list of smiles string of the fragments
        :param min_size: the minimum molecule size
        :param max_size: the maximum molecule size
        """
        self.molecules = [Chem.MolFromSmiles(i) for i in molecules]
        self.fragments = [Chem.MolFromSmiles(i) for i in fragments]
        self.min_size = min_size
        self.max_size = max_size

    def create_molecule(self):
        """Creates a molecule from the initial molecules and fragments.

        :return: rdkit Mol object of the randomly generated molecule
        """
        rand_mol_size = randint(self.min_size, self.max_size)

        while True:
            molecule = choice(self.molecules)

            if util.num_joined_symm_sssr(molecule) == rand_mol_size:
                return molecule

            while True:
                molecule = addition(molecule, self.fragments, rand_mol_size)

                # if there are no more addition possibilities retry
                # from a new initial molecule
                if molecule is None:
                    break

                mol_size = util.num_joined_symm_sssr(molecule)
                # if we have reached the desire size return the molecule
                if mol_size == rand_mol_size:
                    return molecule
