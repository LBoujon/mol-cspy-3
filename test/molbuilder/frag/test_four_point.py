from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.fragmentation.fragmentation import fragmentation
from cspy.molbuilder.fragmentation.fragmentation import four_point_fragmentation_positions


class TestFourPointFragmentation(TestCase):

    def setUp(self):
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.coronene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        self.phenanthro_4_5_bcd_thiophene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)s4')

    def test_fragmentation_of_coronene_returns_a_six_membered_ring_fragmemnt_and_a_phenanthrene_molecule(self):
        phenanthrene, fragment = fragmentation(self.pyrene, (2, 11, 10, 5), False, True)
        self.assertEqual(Chem.MolToInchi(phenanthrene, options='-SNon'), 'InChI=1S/C14H10/c1-3-7-13-11(5-1)9-10-12-6-2-4-8-14(12)13/h1-10H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*:*cc*:1')

    def test_fragmentation_of_coronene_returns_a_six_membered_ring_fragmemnt_and_a_benzoperylene_molecule(self):
        benzoperylene, fragment = fragmentation(self.coronene, (2, 19, 20, 5), False, True)
        self.assertEqual(Chem.MolToInchi(benzoperylene, options='-SNon'), 'InChI=1S/C22H12/c1-3-13-7-9-15-11-12-16-10-8-14-4-2-6-18-17(5-1)19(13)21(15)22(16)20(14)18/h1-12H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*:*cc*:1')

    def test_fragmentation_of_phenanthro_4_5_bcd_thiophene_returns_a_six_membered_ring_fragmemnt_and_a_dibenzothiophene(self):
        dibenzothiophene, fragment = fragmentation(self.phenanthro_4_5_bcd_thiophene, (2, 11, 10, 5), False, True)
        self.assertEqual(Chem.MolToInchi(dibenzothiophene, options='-SNon'), 'InChI=1S/C12H8S/c1-3-7-11-9(5-1)10-6-2-4-8-12(10)13-11/h1-8H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*:*cc*:1')

    def test_fragmentation_of_phenanthro_4_5_bcd_thiophene_returns_a_sulfur_ring_fragmemnt_and_a_phenanthrene(self):
        phenanthrene, fragment = fragmentation(self.phenanthro_4_5_bcd_thiophene, (9, 10, 11, 12), False, True)
        self.assertEqual(Chem.MolToInchi(phenanthrene, options='-SNon'), 'InChI=1S/C14H10/c1-3-7-13-11(5-1)9-10-12-6-2-4-8-14(12)13/h1-10H')
        self.assertEqual(Chem.MolToSmiles(fragment), '*1:*:*s*:1')


class TestFourPointFragmentationPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.pyracyclene = Chem.MolFromSmiles('c1cc2c3c(ccc4c3c1C=C4)C=C2')
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.coronene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        self.phenanthro_4_5_bcd_thiophene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)s4')
        self.molecule_0 = Chem.MolFromSmiles('C1=CC2C=CC1c1c2c2ccccc2c2ccccc12')

    def test_fragmentation_positions_for_benzene(self):
        positions = four_point_fragmentation_positions(self.benzene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_phenanthrene(self):
        positions = four_point_fragmentation_positions(self.phenanthrene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_pyracyclene(self):
        positions = four_point_fragmentation_positions(self.pyracyclene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_pyrene(self):
        positions = four_point_fragmentation_positions(self.pyrene)
        self.assertEqual([(2, 11, 10, 5), (9, 10, 11, 12)], positions)

    def test_fragmentation_positions_for_coronene(self):
        positions = four_point_fragmentation_positions(self.coronene)
        self.assertEqual([(2, 19, 18, 17), (2, 19, 20, 5), (5, 20, 21, 8), (8, 21, 22, 11), (11, 22, 23, 14),
                          (14, 23, 18, 17)], positions)

    def test_fragmentation_positions_for_phenanthro_4_5_bcd_thiophene(self):
        positions = four_point_fragmentation_positions(self.phenanthro_4_5_bcd_thiophene)
        self.assertEqual([(2, 11, 10, 5), (9, 10, 11, 12)], positions)

    def test_fragmentation_positions_for_molecule_0(self):
        positions = four_point_fragmentation_positions(self.molecule_0)
        self.assertEqual([], positions)
