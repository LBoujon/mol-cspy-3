from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.addition.addition import add_fragment
from cspy.molbuilder.addition.addition import four_point_attachment_positions


class TestAddFragment(TestCase):

    def setUp(self):
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')
        self.benzo_ghi_perylene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5cccc6c5c4c3c2c6c1')
        self.fluoranthene = Chem.MolFromSmiles('c1ccc-2c(c1)-c3cccc4c3c2ccc4')
        self.benzo_ghi_fluoranthene = Chem.MolFromSmiles('c1cc2ccc3ccc4cccc-5c4c3c2c5c1')

        self.benzene_fragment = Chem.MolFromSmiles('c1c****1')
        self.sulfur_fragment = Chem.MolFromSmiles('*1**s*1')

    # Phenanthrene
    def test_add_fragment_adds_a_benzene_fragment_to_phenanthrene_to_create_pyrene(self):
        pyrene = add_fragment(self.phenanthrene, self.benzene_fragment, (2, 3, 9, 10), True)
        self.assertEqual(Chem.MolToInchi(pyrene, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')
        pyrene = add_fragment(self.phenanthrene, self.benzene_fragment, (2, 3, 9, 10), False)
        self.assertEqual(Chem.MolToInchi(pyrene, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')

    def test_add_fragment_adds_a_sulfur_fragment_to_phenanthrene_to_create_phenanthro_4_5_bcd_thiophene(self):
        phenanthro_4_5_bcd_thiophene = add_fragment(self.phenanthrene, self.sulfur_fragment, (2, 3, 9, 10), True)
        self.assertEqual(Chem.MolToInchi(phenanthro_4_5_bcd_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-7-8-10-4-2-6-12-14(10)13(9)11(5-1)15-12/h1-8H')
        phenanthro_4_5_bcd_thiophene = add_fragment(self.phenanthrene, self.sulfur_fragment, (2, 3, 9, 10), False)
        self.assertEqual(Chem.MolToInchi(phenanthro_4_5_bcd_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-7-8-10-4-2-6-12-14(10)13(9)11(5-1)15-12/h1-8H')

    # Perylene
    def test_add_fragment_adds_a_benzene_fragment_to_perylene_to_create_benzo_ghi_perylene(self):
        benzo_ghi_perylene = add_fragment(self.perylene, self.benzene_fragment, (5, 6, 16, 17), True)
        self.assertEqual(Chem.MolToInchi(benzo_ghi_perylene, options='-SNon'), 'InChI=1S/C22H12/c1-3-13-7-9-15-11-12-16-10-8-14-4-2-6-18-17(5-1)19(13)21(15)22(16)20(14)18/h1-12H')
        benzo_ghi_perylene = add_fragment(self.perylene, self.benzene_fragment, (5, 6, 16, 17), False)
        self.assertEqual(Chem.MolToInchi(benzo_ghi_perylene, options='-SNon'), 'InChI=1S/C22H12/c1-3-13-7-9-15-11-12-16-10-8-14-4-2-6-18-17(5-1)19(13)21(15)22(16)20(14)18/h1-12H')

    def test_add_fragment_adds_a_sulfur_fragment_to_perylene_to_create_perylo_1_12_bcd_thiophene(self):
        perylo_1_12_bcd_thiophene = add_fragment(self.perylene, self.sulfur_fragment, (5, 6, 16, 17), True)
        self.assertEqual(Chem.MolToInchi(perylo_1_12_bcd_thiophene, options='-SNon'), 'InChI=1S/C20H10S/c1-3-11-7-9-15-19-17(11)13(5-1)14-6-2-4-12-8-10-16(21-15)20(19)18(12)14/h1-10H')
        perylo_1_12_bcd_thiophene = add_fragment(self.perylene, self.sulfur_fragment, (5, 6, 16, 17), False)
        self.assertEqual(Chem.MolToInchi(perylo_1_12_bcd_thiophene, options='-SNon'), 'InChI=1S/C20H10S/c1-3-11-7-9-15-19-17(11)13(5-1)14-6-2-4-12-8-10-16(21-15)20(19)18(12)14/h1-10H')

    # Benzo[ghi]perylene
    def test_add_fragment_adds_a_benzene_fragment_to_benzo_ghi_perylene_to_create_coronene(self):
        coronene = add_fragment(self.benzo_ghi_perylene, self.benzene_fragment, (14, 15, 20, 21), True)
        self.assertEqual(Chem.MolToInchi(coronene, options='-SNon'), 'InChI=1S/C24H12/c1-2-14-5-6-16-9-11-18-12-10-17-8-7-15-4-3-13(1)19-20(14)22(16)24(18)23(17)21(15)19/h1-12H')
        coronene = add_fragment(self.benzo_ghi_perylene, self.benzene_fragment, (14, 15, 20, 21), False)
        self.assertEqual(Chem.MolToInchi(coronene, options='-SNon'), 'InChI=1S/C24H12/c1-2-14-5-6-16-9-11-18-12-10-17-8-7-15-4-3-13(1)19-20(14)22(16)24(18)23(17)21(15)19/h1-12H')

    # Fluoranthene
    def test_add_fragment_adds_a_benzene_fragment_to_fluoranthene_to_create_benzo_ghi_fluoranthene(self):
        benzo_ghi_fluoranthene = add_fragment(self.fluoranthene, self.benzene_fragment, (2, 3, 12, 13), True)
        self.assertEqual(Chem.MolToInchi(benzo_ghi_fluoranthene, options='-SNon'), 'InChI=1S/C18H10/c1-3-11-7-9-13-10-8-12-4-2-6-15-14(5-1)16(11)18(13)17(12)15/h1-10H')
        benzo_ghi_fluoranthene = add_fragment(self.fluoranthene, self.benzene_fragment, (2, 3, 12, 13), False)
        self.assertEqual(Chem.MolToInchi(benzo_ghi_fluoranthene, options='-SNon'), 'InChI=1S/C18H10/c1-3-11-7-9-13-10-8-12-4-2-6-15-14(5-1)16(11)18(13)17(12)15/h1-10H')

    # Benzo[ghi]fluoranthene
    def test_add_fragment_adds_a_benzene_fragment_to_benzo_ghi_fluoranthene_to_create_to_corannulene(self):
        corannulene = add_fragment(self.benzo_ghi_fluoranthene, self.benzene_fragment, (11, 12, 16, 17), True)
        self.assertEqual(Chem.MolToInchi(corannulene, options='-SNon'), 'InChI=1S/C20H10/c1-2-12-5-6-14-9-10-15-8-7-13-4-3-11(1)16-17(12)19(14)20(15)18(13)16/h1-10H')
        corannulene = add_fragment(self.benzo_ghi_fluoranthene, self.benzene_fragment, (11, 12, 16, 17), False)
        self.assertEqual(Chem.MolToInchi(corannulene, options='-SNon'), 'InChI=1S/C20H10/c1-2-12-5-6-14-9-10-15-8-7-13-4-3-11(1)16-17(12)19(14)20(15)18(13)16/h1-10H')


class TestFourPointAttachmentPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.anthracene = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.naphtho_1_2_c_thiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2csc3')
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')
        self.isochrysene = Chem.MolFromSmiles('c1ccc2c(c1)c3ccccc3c4c2cccc4')
        self.benzo_ghi_perylene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5cccc6c5c4c3c2c6c1')
        self.fluoranthene = Chem.MolFromSmiles('c1ccc-2c(c1)-c3cccc4c3c2ccc4')
        self.benzo_ghi_fluoranthene = Chem.MolFromSmiles('c1cc2ccc3ccc4cccc-5c4c3c2c5c1')
        self.dibenzothiophene = Chem.MolFromSmiles('c1ccc2c(c1)c3ccccc3s2')

        self.benzene_fragment = Chem.MolFromSmiles('c1c****1')
        self.sulfur_fragment = Chem.MolFromSmiles('*1**s*1')

    # Benzene
    def test_four_point_attachment_positions_for_benzene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.benzene, self.benzene_fragment)
        self.assertEqual([], positions)

    # Napthalene
    def test_four_point_attachment_positions_for_napthalene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.napthalene, self.benzene_fragment)
        self.assertEqual([], positions)

    # Anthracene
    def test_four_point_attachment_positions_for_anthracene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.anthracene, self.benzene_fragment)
        self.assertEqual([], positions)

    # Phenanthrene
    def test_four_point_attachment_positions_for_phenanthrene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.phenanthrene, self.benzene_fragment)
        self.assertEqual([(2, 3, 9, 10)], positions)

    # Naphtho[1,2-c]thiophene
    def test_four_point_attachment_positions_for_naphtho_1_2_c_thiophene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.naphtho_1_2_c_thiophene, self.benzene_fragment)
        self.assertEqual([], positions)

    # Perylene
    def test_four_point_attachment_positions_for_perylene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.perylene, self.benzene_fragment)
        self.assertEqual([(5, 6, 16, 17), (9, 8, 10, 11)], positions)

    # Isochrysene
    def test_four_point_attachment_positions_for_isochrysene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.isochrysene, self.benzene_fragment)
        self.assertEqual([(2, 3, 13, 14), (5, 4, 6, 7), (10, 11, 12, 17)], positions)

    # Benzo[ghi]perylene
    def test_four_point_attachment_positions_for_benzo_ghi_perylene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.benzo_ghi_perylene, self.benzene_fragment)
        self.assertEqual([(14, 15, 20, 21)], positions)

    # Fluoranthene
    def test_four_point_attachment_positions_for_fluoranthene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.fluoranthene, self.benzene_fragment)
        self.assertEqual([(2, 3, 12, 13), (5, 4, 6, 7)], positions)

    # Benzo[ghi]fluoranthene
    def test_four_point_attachment_positions_for_benzo_ghi_fluoranthene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.benzo_ghi_fluoranthene, self.benzene_fragment)
        self.assertEqual([(11, 12, 16, 17)], positions)

    def test_four_point_attachment_positions_for_benzo_ghi_fluoranthene_with_a_sulfur_fragment(self):
        positions = four_point_attachment_positions(self.benzo_ghi_fluoranthene, self.sulfur_fragment)
        self.assertEqual([], positions)

    # Dibenzothiophene
    def test_four_point_attachment_positions_for_dibenzothiophene_with_a_benzene_fragment(self):
        positions = four_point_attachment_positions(self.dibenzothiophene, self.benzene_fragment)
        self.assertEqual([(5, 4, 6, 7)], positions)

    def test_four_point_attachment_positions_for_dibenzothiophene_with_a_sulfur_fragment(self):
        positions = four_point_attachment_positions(self.dibenzothiophene, self.sulfur_fragment)
        self.assertEqual([], positions)
