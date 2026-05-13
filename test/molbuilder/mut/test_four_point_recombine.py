from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.mutation.recombine import recombine_four_point_positions
from cspy.molbuilder.mutation.recombine import recombine_dummy_position_switch


class TestRecombineFourPointPositions(TestCase):

    def setUp(self):
        self.phenanthrene_fragment = Chem.MolFromSmiles('c1c**2c(c1)ccc3*2*ccc3')
        self.perylene_fragment = Chem.MolFromSmiles('c1cc2cc**3c2c(c1)c4cccc5c4*3*cc5')
        self.isochrysene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*3*cccc3c4c2cccc4')
        self.benzo_ghi_perylene_fragment = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5cc**6c5c4c3c2*6*1')
        self.fluoranthene_fragment = Chem.MolFromSmiles('c1ccc-2*(*1)-*3*ccc4c3c2ccc4')
        self.benzo_ghi_fluoranthene_fragment = Chem.MolFromSmiles('c1cc2ccc3ccc4cc**-5c4c3c2*5*1')
        self.dibenzothiophene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*3*cccc3s2')

        self.benzene_fragment = Chem.MolFromSmiles('c1c****1')
        self.sulfur_fragment = Chem.MolFromSmiles('*1**s*1')

    # Phenanthrene
    def test_recombine_four_point_positions_for_a_phenanthrene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.phenanthrene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((2, 3, 9, 10), (0, 1, 2, 3)), ((2, 3, 9, 10), (0, 5, 4, 3)),
                                    ((2, 3, 9, 10), (1, 0, 5, 4)), ((2, 3, 9, 10), (1, 2, 3, 4)),
                                    ((2, 3, 9, 10), (2, 1, 0, 5)), ((2, 3, 9, 10), (2, 3, 4, 5))])

    def test_recombine_four_point_positions_for_a_phenanthrene_fragment_and_a_sulfur_fragment(self):
        bondable = recombine_four_point_positions(self.phenanthrene_fragment, self.sulfur_fragment)
        self.assertEqual(bondable, [((2, 3, 9, 10), (2, 1, 0, 4))])

    # Perylene
    def test_recombine_four_point_positions_for_a_perylene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.perylene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((5, 6, 16, 17), (0, 1, 2, 3)), ((5, 6, 16, 17), (0, 5, 4, 3)),
                                    ((5, 6, 16, 17), (1, 0, 5, 4)), ((5, 6, 16, 17), (1, 2, 3, 4)),
                                    ((5, 6, 16, 17), (2, 1, 0, 5)), ((5, 6, 16, 17), (2, 3, 4, 5)),
                                    ((9, 8, 10, 11), (0, 1, 2, 3)), ((9, 8, 10, 11), (0, 5, 4, 3)),
                                    ((9, 8, 10, 11), (1, 0, 5, 4)), ((9, 8, 10, 11), (1, 2, 3, 4)),
                                    ((9, 8, 10, 11), (2, 1, 0, 5)), ((9, 8, 10, 11), (2, 3, 4, 5))])

    def test_recombine_four_point_positions_for_a_perylene_fragment_and_a_sulfur_fragment(self):
        bondable = recombine_four_point_positions(self.perylene_fragment, self.sulfur_fragment)
        self.assertEqual(bondable, [((5, 6, 16, 17), (2, 1, 0, 4)), ((9, 8, 10, 11), (2, 1, 0, 4))])

    # Isochrysene
    def test_recombine_four_point_positions_for_a_isochrysene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.isochrysene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((2, 3, 13, 14), (0, 1, 2, 3)), ((2, 3, 13, 14), (0, 5, 4, 3)),
                                    ((2, 3, 13, 14), (1, 0, 5, 4)), ((2, 3, 13, 14), (1, 2, 3, 4)),
                                    ((2, 3, 13, 14), (2, 1, 0, 5)), ((2, 3, 13, 14), (2, 3, 4, 5)),
                                    ((5, 4, 6, 7), (0, 1, 2, 3)), ((5, 4, 6, 7), (0, 5, 4, 3)),
                                    ((5, 4, 6, 7), (1, 0, 5, 4)), ((5, 4, 6, 7), (1, 2, 3, 4)),
                                    ((5, 4, 6, 7), (2, 1, 0, 5)), ((5, 4, 6, 7), (2, 3, 4, 5)),
                                    ((10, 11, 12, 17), (0, 1, 2, 3)), ((10, 11, 12, 17), (0, 5, 4, 3)),
                                    ((10, 11, 12, 17), (1, 0, 5, 4)), ((10, 11, 12, 17), (1, 2, 3, 4)),
                                    ((10, 11, 12, 17), (2, 1, 0, 5)), ((10, 11, 12, 17), (2, 3, 4, 5))])

    def test_recombine_four_point_positions_for_a_isochrysene_fragment_and_a_sulfur_fragment(self):
        bondable = recombine_four_point_positions(self.isochrysene_fragment, self.sulfur_fragment)
        self.assertEqual(bondable, [((2, 3, 13, 14), (2, 1, 0, 4)), ((5, 4, 6, 7), (2, 1, 0, 4)),
                                    ((10, 11, 12, 17), (2, 1, 0, 4))])

    # Benzo[ghi]perylene
    def test_recombine_four_point_positions_for_a_benzo_ghi_perylene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.benzo_ghi_perylene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((14, 15, 20, 21), (0, 1, 2, 3)), ((14, 15, 20, 21), (0, 5, 4, 3)),
                                    ((14, 15, 20, 21), (1, 0, 5, 4)), ((14, 15, 20, 21), (1, 2, 3, 4)),
                                    ((14, 15, 20, 21), (2, 1, 0, 5)), ((14, 15, 20, 21), (2, 3, 4, 5))])

    def test_recombine_four_point_positions_for_a_benzo_ghi_perylene_fragment_and_a_sulfur_fragment(self):
        bondable = recombine_four_point_positions(self.benzo_ghi_perylene_fragment, self.sulfur_fragment)
        self.assertEqual(bondable, [((14, 15, 20, 21), (2, 1, 0, 4))])

    # Fluoranthene
    def test_recombine_four_point_positions_for_a_fluoranthene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.fluoranthene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((2, 3, 12, 13), (0, 1, 2, 3)), ((2, 3, 12, 13), (0, 5, 4, 3)),
                                    ((2, 3, 12, 13), (1, 0, 5, 4)), ((2, 3, 12, 13), (1, 2, 3, 4)),
                                    ((2, 3, 12, 13), (2, 1, 0, 5)), ((2, 3, 12, 13), (2, 3, 4, 5)),
                                    ((5, 4, 6, 7), (0, 1, 2, 3)), ((5, 4, 6, 7), (0, 5, 4, 3)),
                                    ((5, 4, 6, 7), (1, 0, 5, 4)), ((5, 4, 6, 7), (1, 2, 3, 4)),
                                    ((5, 4, 6, 7), (2, 1, 0, 5)), ((5, 4, 6, 7), (2, 3, 4, 5))])

    # Benzo[ghi]fluoranthene
    def test_recombine_four_point_positions_for_a_benzo_ghi_fluoranthene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.benzo_ghi_fluoranthene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((11, 12, 16, 17), (0, 1, 2, 3)), ((11, 12, 16, 17), (0, 5, 4, 3)),
                                    ((11, 12, 16, 17), (1, 0, 5, 4)), ((11, 12, 16, 17), (1, 2, 3, 4)),
                                    ((11, 12, 16, 17), (2, 1, 0, 5)), ((11, 12, 16, 17), (2, 3, 4, 5))])

    # Dibenzothiophene
    def test_recombine_four_point_positions_for_a_dibenzothiophene_fragment_and_a_benzene_fragment(self):
        bondable = recombine_four_point_positions(self.dibenzothiophene_fragment, self.benzene_fragment)
        self.assertEqual(bondable, [((5, 4, 6, 7), (0, 1, 2, 3)), ((5, 4, 6, 7), (0, 5, 4, 3)),
                                    ((5, 4, 6, 7), (1, 0, 5, 4)), ((5, 4, 6, 7), (1, 2, 3, 4)),
                                    ((5, 4, 6, 7), (2, 1, 0, 5)), ((5, 4, 6, 7), (2, 3, 4, 5))])


class TestFourPointRecombineDummyPositionSwitch(TestCase):

    def setUp(self):
        self.phenanthrene_fragment = Chem.MolFromSmiles('c1c**2c(c1)ccc3*2*ccc3')
        self.perylene_fragment = Chem.MolFromSmiles('c1cc2cc**3c2c(c1)c4cccc5c4*3*cc5')
        self.isochrysene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*3*cccc3c4c2cccc4')
        self.benzo_ghi_perylene_fragment = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5cc**6c5c4c3c2*6*1')
        self.fluoranthene_fragment = Chem.MolFromSmiles('c1ccc-2*(*1)-*3*ccc4c3c2ccc4')
        self.benzo_ghi_fluoranthene_fragment = Chem.MolFromSmiles('c1cc2ccc3ccc4cc**-5c4c3c2*5*1')
        self.dibenzothiophene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*3*cccc3s2')
        self.dibenzo_f_h_quinoxaline_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*3*cccc3c4c2nccn4')
        self.benzene_fragment = Chem.MolFromSmiles('c1c****1')
        self.sulfur_fragment = Chem.MolFromSmiles('*1**s*1')

    # Benzene
    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_0(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (0, 1, 2, 3), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_1(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (0, 5, 4, 3), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_2(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (1, 0, 5, 4), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_3(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (1, 2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_4(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (2, 1, 0, 5), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_benzene_fragment_returns_a_benzene_fragment_5(self):
        benzene_fragment = recombine_dummy_position_switch(self.benzene_fragment, (2, 3, 4, 5), True)
        self.assertEqual(Chem.MolToSmiles(benzene_fragment), '*1:*:*cc*:1')

    # Sulfur
    def test_recombine_dummy_position_switch_for_a_sulfur_fragment_returns_a_sulfur_fragment(self):
        sulfur_fragment = recombine_dummy_position_switch(self.sulfur_fragment, (2, 1, 0, 4), True)
        self.assertEqual(Chem.MolToSmiles(sulfur_fragment), '*1:*:*s*:1')

    # Phenanthrene
    def test_recombine_dummy_position_switch_for_a_phenanthrene_fragment_returns_a_phenanthrene_fragment(self):
        phenanthrene_fragment = recombine_dummy_position_switch(self.phenanthrene_fragment, (2, 3, 9, 10), True)
        self.assertEqual(Chem.MolToSmiles(phenanthrene_fragment), '*1cccc2ccc3ccc*:*3:*:12')

    # Perylene
    def test_recombine_dummy_position_switch_for_a_perylene_fragment_returns_a_perylene_fragment_0(self):
        perylene_fragment = recombine_dummy_position_switch(self.perylene_fragment, (5, 6, 16, 17), True)
        self.assertEqual(Chem.MolToSmiles(perylene_fragment), '*1ccc2cccc3c4cccc5cc*:*(:*:1c23)c54')

    def test_recombine_dummy_position_switch_for_a_perylene_fragment_returns_a_perylene_fragment_1(self):
        perylene_fragment = recombine_dummy_position_switch(self.perylene_fragment, (9, 8, 10, 11), False)
        self.assertEqual(Chem.MolToSmiles(perylene_fragment), '*1ccc2cccc3c4cccc5cc*:*(:*:1c23)c54')

    def test_recombine_dummy_position_switch_for_a_perylene_fragment_returns_a_perylene_fragment_2(self):
        perylene_fragment = recombine_dummy_position_switch(self.perylene_fragment, (9, 8, 10, 11), True)
        self.assertEqual(Chem.MolToSmiles(perylene_fragment), '*1ccc2cccc3c4cccc5cc*:*(:*:1c23)c54')

    # Isochrysene
    def test_recombine_dummy_position_switch_for_a_isochrysene_fragment_returns_a_isochrysene_fragment_0(self):
        isochrysene_fragment = recombine_dummy_position_switch(self.isochrysene_fragment, (2, 3, 13, 14), False)
        self.assertEqual(Chem.MolToSmiles(isochrysene_fragment), '*1cccc2*:1:*1:*cccc1c1ccccc21')

    def test_recombine_dummy_position_switch_for_a_isochrysene_fragment_returns_a_isochrysene_fragment_1(self):
        isochrysene_fragment = recombine_dummy_position_switch(self.isochrysene_fragment, (2, 3, 13, 14), True)
        self.assertEqual(Chem.MolToSmiles(isochrysene_fragment), '*1cccc2*:1:*1:*cccc1c1ccccc21')

    def test_recombine_dummy_position_switch_for_a_isochrysene_fragment_returns_a_isochrysene_fragment_2(self):
        isochrysene_fragment = recombine_dummy_position_switch(self.isochrysene_fragment, (5, 4, 6, 7), True)
        self.assertEqual(Chem.MolToSmiles(isochrysene_fragment), '*1cccc2*:1:*1:*cccc1c1ccccc21')

    def test_recombine_dummy_position_switch_for_a_isochrysene_fragment_returns_a_isochrysene_fragment_4(self):
        isochrysene_fragment = recombine_dummy_position_switch(self.isochrysene_fragment, (10, 11, 12, 17), False)
        self.assertEqual(Chem.MolToSmiles(isochrysene_fragment), '*1cccc2*:1:*1:*cccc1c1ccccc21')

    def test_recombine_dummy_position_switch_for_a_isochrysene_fragment_returns_a_isochrysene_fragment_5(self):
        isochrysene_fragment = recombine_dummy_position_switch(self.isochrysene_fragment, (10, 11, 12, 17), True)
        self.assertEqual(Chem.MolToSmiles(isochrysene_fragment), '*1cccc2*:1:*1:*cccc1c1ccccc21')

    # Benzo[ghi]perylene
    def test_recombine_dummy_position_switch_for_a_benzo_ghi_perylene_fragment_returns_a_benzo_ghi_perylene_fragment(self):
        benzo_ghi_perylene_fragment = recombine_dummy_position_switch(self.benzo_ghi_perylene_fragment, (14, 15, 20, 21), True)
        self.assertEqual(Chem.MolToSmiles(benzo_ghi_perylene_fragment), '*1ccc2ccc3ccc4ccc5cc*:*6:*:1c2c3c4c65')

    # Fluoranthene
    def test_recombine_dummy_position_switch_for_a_fluoranthene_fragment_returns_a_fluoranthene_fragment_0(self):
        fluoranthene_fragment = recombine_dummy_position_switch(self.fluoranthene_fragment, (2, 3, 12, 13), False)
        self.assertEqual(Chem.MolToSmiles(fluoranthene_fragment), '*1cccc2*:1:*1:*ccc3cccc2c13')

    def test_recombine_dummy_position_switch_for_a_fluoranthene_fragment_returns_a_fluoranthene_fragment_1(self):
        fluoranthene_fragment = recombine_dummy_position_switch(self.fluoranthene_fragment, (2, 3, 12, 13), True)
        self.assertEqual(Chem.MolToSmiles(fluoranthene_fragment), '*1cccc2*:1:*1:*ccc3cccc2c13')

    def test_recombine_dummy_position_switch_for_a_fluoranthene_fragment_returns_a_fluoranthene_fragment_2(self):
        fluoranthene_fragment = recombine_dummy_position_switch(self.fluoranthene_fragment, (5, 4, 6, 7), True)
        self.assertEqual(Chem.MolToSmiles(fluoranthene_fragment), '*1cccc2*:1:*1:*ccc3cccc2c13')

    # Benzo[ghi]fluoranthene
    def test_recombine_dummy_position_switch_for_a_benzo_ghi_fluoranthene_fragment_returns_a_benzo_ghi_fluoranthene_fragment(self):
        benzo_ghi_fluoranthene_fragment = recombine_dummy_position_switch(self.benzo_ghi_fluoranthene_fragment, (11, 12, 16, 17), True)
        self.assertEqual(Chem.MolToSmiles(benzo_ghi_fluoranthene_fragment), '*1ccc2ccc3ccc4cc*:*5:*:1c2c3c54')

    # Dibenzothiophene
    def test_recombine_dummy_position_switch_for_a_dibenzothiophene_fragment_returns_a_dibenzothiophene_fragment(self):
        dibenzothiophene_fragment = recombine_dummy_position_switch(self.dibenzothiophene_fragment, (5, 4, 6, 7), True)
        self.assertEqual(Chem.MolToSmiles(dibenzothiophene_fragment), '*1cccc2sc3ccc*:*3:*:12')

    # Dibenzo[f,h]quinoxaline
    def test_recombine_dummy_position_switch_for_a_dibenzo_f_h_quinoxaline_fragment_returns_a_dibenzo_f_h_quinoxaline_fragment(self):
        dibenzo_f_h_quinoxaline_fragment = recombine_dummy_position_switch(self.dibenzo_f_h_quinoxaline_fragment, (5, 4, 6, 7), True)
        self.assertEqual(Chem.MolToSmiles(dibenzo_f_h_quinoxaline_fragment), '*1cccc2*:1:*1:*cccc1c1nccnc21')

    def test_recombine_dummy_position_switch_for_a_dibenzo_f_h_quinoxaline_fragment_returns_a_fragment_0(self):
        fragment = recombine_dummy_position_switch(self.dibenzo_f_h_quinoxaline_fragment, (2, 3, 13, 14), False)
        self.assertEqual(Chem.MolToSmiles(fragment), '*1cccc2*:1:*1:*ccnc1c1cccnc21')

    def test_recombine_dummy_position_switch_for_a_dibenzo_f_h_quinoxaline_fragment_returns_a_fragment_1(self):
        fragment = recombine_dummy_position_switch(self.dibenzo_f_h_quinoxaline_fragment, (2, 3, 13, 14), True)
        self.assertEqual(Chem.MolToSmiles(fragment), '*1ccnc2*:1:*1:*ccnc1c1ccccc21')

    def test_recombine_dummy_position_switch_for_a_dibenzo_f_h_quinoxaline_fragment_returns_a_fragment_2(self):
        fragment = recombine_dummy_position_switch(self.dibenzo_f_h_quinoxaline_fragment, (10, 11, 12, 17), False)
        self.assertEqual(Chem.MolToSmiles(fragment), '*1cccc2*:1:*1:*ccnc1c1cccnc21')

    def test_recombine_dummy_position_switch_for_a_dibenzo_f_h_quinoxaline_fragment_returns_a_fragment_3(self):
        fragment = recombine_dummy_position_switch(self.dibenzo_f_h_quinoxaline_fragment, (10, 11, 12, 17), True)
        self.assertEqual(Chem.MolToSmiles(fragment), '*1ccnc2*:1:*1:*ccnc1c1ccccc21')
