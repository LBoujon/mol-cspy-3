from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.fragmentation.fragmentation import fragmentation
from cspy.molbuilder.fragmentation.fragmentation import two_point_fragmentation_positions


class TestTwoPointFragmentation(TestCase):

    def setUp(self):
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.benzothiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccs2')
        self.isoquinoline = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        self.naphtho_2_3_c_thiophene = Chem.MolFromSmiles('s2cc1cc3c(cc1c2)cccc3')
        self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = Chem.MolFromSmiles('c1ccc2c(c1)C3C=CC2C=C3')
        self.benzimidazolone = Chem.MolFromSmiles('c1ccc2=NC(=O)N=c2c1')

    def test_fragment_molecule_napthalene_returns_two_benzene_rings(self):
        benzene_1, benzene_2 = fragmentation(self.napthalene, (3, 8), False, False)
        self.assertEqual(Chem.MolToInchi(benzene_1, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
        self.assertEqual(Chem.MolToInchi(benzene_2, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')

    def test_fragment_molecule_benzothiophene_returns_benzene_and_thiophene(self):
        benzene, thiophene  = fragmentation(self.benzothiophene, (3, 4), False, False)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
        self.assertEqual(Chem.MolToInchi(thiophene, options='-SNon'), 'InChI=1S/C4H4S/c1-2-4-5-3-1/h1-4H')

    def test_fragment_molecule_isoquinoline_returns_benzene_and_pyridine(self):
        benzene, pyridine = fragmentation(self.isoquinoline, (3, 8), False, False)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
        self.assertEqual(Chem.MolToInchi(pyridine, options='-SNon'), 'InChI=1S/C5H5N/c1-2-4-6-5-3-1/h1-5H')

    def test_fragment_molecule_naphtho_2_3_c_thiophene_returns_napthalene_and_thiophene(self):
        thiophene, napthalene  = fragmentation(self.naphtho_2_3_c_thiophene, (2, 7), False, False)
        self.assertEqual(Chem.MolToInchi(thiophene, options='-SNon'), 'InChI=1S/C4H4S/c1-2-4-5-3-1/h1-4H')
        self.assertEqual(Chem.MolToInchi(napthalene, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')

    def test_fragment_molecule_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_and_returns_benzene_and_barrelene(self):
        benzene, barrelene  = fragmentation(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene, (3, 4), False, False)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
        self.assertEqual(Chem.MolToInchi(barrelene, options='-SNon'), 'InChI=1S/C8H8/c1-2-8-5-3-7(1)4-6-8/h1-8H')

    def test_fragment_molecule_benzimidazolone_and_returns_benzene_and_imidazolone(self):
        benzene, imidazolone  = fragmentation(self.benzimidazolone, (3, 8), False, False)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
        self.assertEqual(Chem.MolToInchi(imidazolone, options='-SNon'), 'InChI=1S/C3H2N2O/c6-3-4-1-2-5-3/h1-2H')


class TestTwoPointFragmentationPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.benzothiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccs2')
        self.isoquinoline = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        self.naphtho_2_3_c_thiophene = Chem.MolFromSmiles('s2cc1cc3c(cc1c2)cccc3')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = Chem.MolFromSmiles('c1ccc2c(c1)C3C=CC2C=C3')
        self.dibarrelene = Chem.MolFromSmiles('C1=CC2C=CC1C1=C2C2C=CC1C=C2')
        self.fluorobarrelene = Chem.MolFromSmiles('FC1=C(F)C2C(F)=C(F)C1C(F)=C2F')
        self.molecule_0 = Chem.MolFromSmiles('C1=CC2C=CC1c1c2c2ccccc2c2ccccc12')

    def test_fragmentation_positions_for_benzene(self):
        positions = two_point_fragmentation_positions(self.benzene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_napthalene(self):
        positions = two_point_fragmentation_positions(self.napthalene)
        self.assertEqual([(3, 8)], positions)

    def test_fragmentation_positions_for_thiophene(self):
        positions = two_point_fragmentation_positions(self.thiophene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_benzothiophene(self):
        positions = two_point_fragmentation_positions(self.benzothiophene)
        self.assertEqual([(3, 4)], positions)

    def test_fragmentation_positions_for_isoquinoline(self):
        positions = two_point_fragmentation_positions(self.isoquinoline)
        self.assertEqual([(3, 8)], positions)

    def test_fragmentation_positions_for_naphtho_2_3_c_thiophene(self):
        positions = two_point_fragmentation_positions(self.naphtho_2_3_c_thiophene)
        self.assertEqual([(2, 7), (4, 5)], positions)

    def test_fragmentation_positions_for_phenanthrene(self):
        positions = two_point_fragmentation_positions(self.phenanthrene)
        self.assertEqual([(3, 4), (8, 9)], positions)

    def test_fragmentation_positions_for_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene(self):
        positions = two_point_fragmentation_positions(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene)
        self.assertEqual([(3, 4)], positions)

    def test_fragmentation_positions_for_dibarrelene(self):
        positions = two_point_fragmentation_positions(self.dibarrelene)
        self.assertEqual([(6, 7)], positions)

    def test_fragmentation_positions_for_fluorobarrelene(self):
        positions = two_point_fragmentation_positions(self.fluorobarrelene)
        self.assertEqual([], positions)

    def test_fragmentation_positions_for_molecule_0(self):
        positions = two_point_fragmentation_positions(self.molecule_0)
        self.assertEqual([(6, 7), (8, 13), (14, 19)], positions)
