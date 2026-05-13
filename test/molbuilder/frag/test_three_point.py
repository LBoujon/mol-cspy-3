from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.fragmentation.fragmentation import fragmentation
from cspy.molbuilder.fragmentation.fragmentation import three_point_fragmentation_positions


class TestThreePointFragmentation(TestCase):

    def setUp(self):
        self.pyracyclene = Chem.MolFromSmiles('c1cc2c3c(ccc4c3c1C=C4)C=C2')
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')

    def test_fragmentation_of_pyracyclene_returns_a_five_membered_ring_fragmemnt_and_a_acenaphthylene_molecule(self):
        acenaphthylene, fragment = fragmentation(self.pyracyclene, (2, 3, 4), False, True)
        self.assertEqual(Chem.MolToInchi(acenaphthylene, options='-SNon'), 'InChI=1S/C12H8/c1-3-9-4-2-6-11-8-7-10(5-1)12(9)11/h1-8H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*cc*:1')

    def test_fragmentation_of_perylene_returns_fused_six_membered_ring_fragmemnt_and_a_naphthalene_molecule(self):
        naphthalene, fragment = fragmentation(self.perylene, (6, 7, 8), False, True)
        self.assertEqual(Chem.MolToInchi(naphthalene, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*c2cccc3cccc(*:1)c23')


class TestThreePointFragmentationPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.pyracyclene = Chem.MolFromSmiles('c1cc2c3c(ccc4c3c1C=C4)C=C2')
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')
        self.cyclopenta_cd_perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4ccc5c6c4c3ccc6C=C5')
        self.aceanthrylene = Chem.MolFromSmiles('c1ccc2c(c1)cc3cccc4c3c2C=C4')
        self.cyclopent_hi_aceanthrylene = Chem.MolFromSmiles('c1cc2ccc3c2c(c1)c4ccc5c4c3ccc5')
        self.molecule_0 = Chem.MolFromSmiles('C1=CC2C=CC1c1c2c2ccccc2c2ccccc12')

    def test_fragmentation_positions_for_benzene(self):
        positions = three_point_fragmentation_positions(self.benzene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_phenanthrene(self):
        positions = three_point_fragmentation_positions(self.phenanthrene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_pyracyclene(self):
        positions = three_point_fragmentation_positions(self.pyracyclene)
        self.assertEqual([(2, 3, 4), (7, 8, 9)], positions)

    def test_fragmentation_positions_for_perylene(self):
        positions = three_point_fragmentation_positions(self.perylene)
        self.assertEqual([(6, 7, 8), (10, 15, 16)], positions)

    def test_fragmentation_positions_for_cyclopenta_cd_perylene(self):
        positions = three_point_fragmentation_positions(self.cyclopenta_cd_perylene)
        self.assertEqual([(6, 7, 8), (10, 15, 16), (13, 14, 19)], positions)

    def test_fragmentation_positions_for_aceanthrylene(self):
        positions = three_point_fragmentation_positions(self.aceanthrylene)
        self.assertEqual([(11, 12, 13)], positions)

    def test_fragmentation_positions_for_cyclopent_hi_aceanthrylene(self):
        positions = three_point_fragmentation_positions(self.cyclopent_hi_aceanthrylene)
        self.assertEqual([(2, 6, 5), (9, 13, 12)], positions)

    def test_fragmentation_positions_for_molecule_0(self):
        positions = three_point_fragmentation_positions(self.molecule_0)
        self.assertEqual([], positions)
