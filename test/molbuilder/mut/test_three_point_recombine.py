from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.mutation.recombine import recombine_three_point_positions
from cspy.molbuilder.mutation.recombine import recombine_dummy_position_switch


class TestRecombineThreePointPositions(TestCase):

    def setUp(self):
        self.napthalene_fragment = Chem.MolFromSmiles('c1c**2*cccc2c1')
        self.anthracene_fragment = Chem.MolFromSmiles('c1c**2*c3ccccc3cc2c1')
        self.phenanthrene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*cc3c2cccc3')
        self.naphtho_2_3_b_thiophene_fragment = Chem.MolFromSmiles('c1c**2*c3c(ccs3)cc2c1')
        self.acenaphthylene_fragment = Chem.MolFromSmiles('c1**2*ccc3c2c(c1)C=C3')

        self.five_membered_ring_fragment = Chem.MolFromSmiles('c1c***1')
        self.poly_aromatic_ring_fragment = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)***3')

    def test_recombine_three_point_positions_for_a_napthalene_fragment_and_a_five_membered_ring_fragment(self):
        bondable = recombine_three_point_positions(self.napthalene_fragment, self.five_membered_ring_fragment)
        self.assertEqual(bondable, [((2, 3, 4), (0, 1, 2)), ((2, 3, 4), (0, 4, 3)), ((2, 3, 4), (1, 0, 4)), ((2, 3, 4), (1, 2, 3)), ((2, 3, 4), (2, 3, 4)),
                                    ((7, 8, 9), (0, 1, 2)), ((7, 8, 9), (0, 4, 3)), ((7, 8, 9), (1, 0, 4)), ((7, 8, 9), (1, 2, 3)), ((7, 8, 9), (2, 3, 4))])

    def test_recombine_three_point_positions_for_a_napthalene_fragment_and_a_poly_aromatic_ring_fragment(self):
        bondable = recombine_three_point_positions(self.napthalene_fragment, self.poly_aromatic_ring_fragment)
        self.assertEqual(bondable, [((2, 3, 4), (1, 0, 9)), ((2, 3, 4), (3, 4, 5)), ((2, 3, 4), (10, 11, 12)),
                                    ((7, 8, 9), (1, 0, 9)), ((7, 8, 9), (3, 4, 5)), ((7, 8, 9), (10, 11, 12))])

    def test_recombine_three_point_positions_for_a_anthracene_fragment_and_a_five_membered_ring_fragment(self):
        bondable = recombine_three_point_positions(self.anthracene_fragment, self.five_membered_ring_fragment)
        self.assertEqual(bondable, [((2, 3, 4), (0, 1, 2)), ((2, 3, 4), (0, 4, 3)), ((2, 3, 4), (1, 0, 4)), ((2, 3, 4), (1, 2, 3)), ((2, 3, 4), (2, 3, 4)),
                                    ((4, 5, 6), (0, 1, 2)), ((4, 5, 6), (0, 4, 3)), ((4, 5, 6), (1, 0, 4)), ((4, 5, 6), (1, 2, 3)), ((4, 5, 6), (2, 3, 4)),
                                    ((9, 10 ,11), (0, 1, 2)), ((9, 10 ,11), (0, 4, 3)), ((9, 10 ,11), (1, 0, 4)), ((9, 10 ,11), (1, 2, 3)), ((9, 10 ,11), (2, 3, 4)),
                                    ((11, 12, 13), (0, 1, 2)), ((11, 12, 13), (0, 4, 3)), ((11, 12, 13), (1, 0, 4)), ((11, 12, 13), (1, 2, 3)), ((11, 12, 13), (2, 3, 4))])

    def test_recombine_three_point_positions_for_a_phenanthrene_fragment_and_a_five_membered_ring_fragment(self):
        bondable = recombine_three_point_positions(self.phenanthrene_fragment, self.five_membered_ring_fragment)
        self.assertEqual(bondable, [((5, 4, 6), (0, 1, 2)), ((5, 4, 6), (0, 4, 3)), ((5, 4, 6), (1, 0, 4)), ((5, 4, 6), (1, 2, 3)), ((5, 4, 6), (2, 3, 4)),
                                    ((7, 8, 13), (0, 1, 2)), ((7, 8, 13), (0, 4, 3)), ((7, 8, 13), (1, 0, 4)), ((7, 8, 13), (1, 2, 3)), ((7, 8, 13), (2, 3, 4))])

    def test_recombine_three_point_positions_for_a_naphtho_2_3_b_thiophene_fragment_and_a_five_membered_ring_fragment(self):
        bondable = recombine_three_point_positions(self.naphtho_2_3_b_thiophene_fragment, self.five_membered_ring_fragment)
        self.assertEqual(bondable, [((2, 3, 4), (0, 1, 2)), ((2, 3, 4), (0, 4, 3)), ((2, 3, 4), (1, 0, 4)), ((2, 3, 4), (1, 2, 3)), ((2, 3, 4), (2, 3, 4)),
                                    ((10, 11, 12), (0, 1, 2)), ((10, 11, 12), (0, 4, 3)), ((10, 11, 12), (1, 0, 4)), ((10, 11, 12), (1, 2, 3)), ((10, 11, 12), (2, 3, 4))])

    def test_recombine_three_point_positions_for_a_naphtho_2_3_b_thiophene_fragment_and_a_poly_aromatic_ring_fragment(self):
        bondable = recombine_three_point_positions(self.naphtho_2_3_b_thiophene_fragment, self.poly_aromatic_ring_fragment)
        self.assertEqual(bondable, [((2, 3, 4), (1, 0, 9)), ((2, 3, 4), (3, 4, 5)), ((2, 3, 4), (10, 11, 12)),
                                    ((10, 11, 12), (1, 0, 9)), ((10, 11, 12), (3, 4, 5)), ((10, 11, 12), (10, 11, 12))])


class TestThreePointRecombineDummyPositionSwitch(TestCase):

    def setUp(self):
        self.napthalene_fragment = Chem.MolFromSmiles('c1c**2*cccc2c1')
        self.anthracene_fragment = Chem.MolFromSmiles('c1c**2*c3ccccc3cc2c1')
        self.phenanthrene_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*cc3c2cccc3')
        self.naphtho_2_3_b_thiophene_fragment = Chem.MolFromSmiles('c1c**2*c3c(ccs3)cc2c1')
        self.acenaphthylene_fragment = Chem.MolFromSmiles('c1**2*ccc3c2c(c1)C=C3')
        self.cinnoline_fragment = Chem.MolFromSmiles('c1ccc2*(*1)*cnn2')
        self.five_membered_ring_fragment = Chem.MolFromSmiles('c1c***1')
        self.six_membered_ring_fragment = Chem.MolFromSmiles('*1*C(=O)NC(=O)*1')
        self.seven_membered_ring_fragment = Chem.MolFromSmiles('c1***ccc1')
        self.poly_aromatic_ring_fragment = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)***3')

    # five membered ring fragment
    def test_recombine_dummy_position_switch_for_a_five_membered_ring_fragment_returns_a_five_membered_ring_fragment_0(self):
        five_membered_ring_fragment = recombine_dummy_position_switch(self.five_membered_ring_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(five_membered_ring_fragment), '*1:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_five_membered_ring_fragment_returns_a_five_membered_ring_fragment_1(self):
        five_membered_ring_fragment = recombine_dummy_position_switch(self.five_membered_ring_fragment, (1, 2, 3), True)
        self.assertEqual(Chem.MolToSmiles(five_membered_ring_fragment), '*1:*cc*:1')

    def test_recombine_dummy_position_switch_for_a_five_membered_ring_fragment_returns_a_five_membered_ring_fragment_2(self):
        five_membered_ring_fragment = recombine_dummy_position_switch(self.five_membered_ring_fragment, (0, 1, 2), True)
        self.assertEqual(Chem.MolToSmiles(five_membered_ring_fragment), '*1:*cc*:1')

    # six membered ring fragment
    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_0(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (1, 0, 7), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*c(=O)[nH]1')

    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_1(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (0, 7, 5), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*[nH]c1=O')

    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_2(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (4, 5, 7), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*c(=O)[nH]1')

    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_3(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (2, 4, 5), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*c(=O)[nH]1')

    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_4(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (1, 2, 4), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*c(=O)[nH]1')

    def test_recombine_dummy_position_switch_for_a_six_membered_ring_fragment_returns_a_six_membered_ring_fragment_5(self):
        six_membered_ring_fragment = recombine_dummy_position_switch(self.six_membered_ring_fragment, (0, 1, 2), True)
        self.assertEqual(Chem.MolToSmiles(six_membered_ring_fragment), 'O=c1*:*:*[nH]c1=O')

    # seven membered ring fragment
    def test_recombine_dummy_position_switch_for_a_seven_membered_ring_fragment_returns_a_seven_membered_ring_fragment_0(self):
        seven_membered_ring_fragment = recombine_dummy_position_switch(self.seven_membered_ring_fragment, (0, 1, 2), True)
        self.assertEqual(Chem.MolToSmiles(seven_membered_ring_fragment), '*1:*cccc*:1')

    def test_recombine_dummy_position_switch_for_a_seven_membered_ring_fragment_returns_a_seven_membered_ring_fragment_1(self):
        seven_membered_ring_fragment = recombine_dummy_position_switch(self.seven_membered_ring_fragment, (0, 6, 5), True)
        self.assertEqual(Chem.MolToSmiles(seven_membered_ring_fragment), '*1:*cccc*:1')

    def test_recombine_dummy_position_switch_for_a_seven_membered_ring_fragment_returns_a_seven_membered_ring_fragment_2(self):
        seven_membered_ring_fragment = recombine_dummy_position_switch(self.seven_membered_ring_fragment, (1, 2, 3), True)
        self.assertEqual(Chem.MolToSmiles(seven_membered_ring_fragment), '*1:*cccc*:1')

    def test_recombine_dummy_position_switch_for_a_seven_membered_ring_fragment_returns_a_seven_membered_ring_fragment_3(self):
        seven_membered_ring_fragment = recombine_dummy_position_switch(self.seven_membered_ring_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(seven_membered_ring_fragment), '*1:*cccc*:1')

    def test_recombine_dummy_position_switch_for_a_seven_membered_ring_fragment_returns_a_seven_membered_ring_fragment_4(self):
        seven_membered_ring_fragment = recombine_dummy_position_switch(self.seven_membered_ring_fragment, (4, 5, 6), True)
        self.assertEqual(Chem.MolToSmiles(seven_membered_ring_fragment), '*1:*cccc*:1')

    # poly aromatic ring fragment
    def test_recombine_dummy_position_switch_for_a_poly_aromatic_ring_fragment_returns_a_poly_aromatic_ring_fragment_0(self):
        poly_aromatic_ring_fragment = recombine_dummy_position_switch(self.poly_aromatic_ring_fragment, (10, 11, 12), True)
        self.assertEqual(Chem.MolToSmiles(poly_aromatic_ring_fragment), '*1:*c2cccc3cccc(*:1)c23')

    def test_recombine_dummy_position_switch_for_a_poly_aromatic_ring_fragment_returns_a_poly_aromatic_ring_fragment_1(self):
        poly_aromatic_ring_fragment = recombine_dummy_position_switch(self.poly_aromatic_ring_fragment, (1, 0, 9), True)
        self.assertEqual(Chem.MolToSmiles(poly_aromatic_ring_fragment), '*1:*c2cccc3cccc(*:1)c23')

    def test_recombine_dummy_position_switch_for_a_poly_aromatic_ring_fragment_returns_a_poly_aromatic_ring_fragment_2(self):
        poly_aromatic_ring_fragment = recombine_dummy_position_switch(self.poly_aromatic_ring_fragment, (3, 4, 5), True)
        self.assertEqual(Chem.MolToSmiles(poly_aromatic_ring_fragment), '*1:*c2cccc3cccc(*:1)c23')

    # Napthalene
    def test_recombine_dummy_position_switch_for_a_napthalene_fragment_returns_a_napthalene_fragment_0(self):
        napthalene_fragment = recombine_dummy_position_switch(self.napthalene_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(napthalene_fragment), '*1cccc2ccc*:*:12')

    def test_recombine_dummy_position_switch_for_a_napthalene_fragment_returns_a_napthalene_fragment_1(self):
        napthalene_fragment = recombine_dummy_position_switch(self.napthalene_fragment, (7, 8, 9), True)
        self.assertEqual(Chem.MolToSmiles(napthalene_fragment), '*1cccc2ccc*:*:12')

    # Anthracene
    def test_recombine_dummy_position_switch_for_a_anthracene_fragment_returns_a_anthracene_fragment_0(self):
        anthracene_fragment = recombine_dummy_position_switch(self.anthracene_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(anthracene_fragment), '*1cccc2cc3ccccc3*:*:12')

    def test_recombine_dummy_position_switch_for_a_anthracene_fragment_returns_a_anthracene_fragment_1(self):
        anthracene_fragment = recombine_dummy_position_switch(self.anthracene_fragment, (9, 10, 11), True)
        self.assertEqual(Chem.MolToSmiles(anthracene_fragment), '*1cccc2cc3ccccc3*:*:12')

    def test_recombine_dummy_position_switch_for_a_anthracene_fragment_returns_a_anthracene_fragment_2(self):
        anthracene_fragment = recombine_dummy_position_switch(self.anthracene_fragment, (11, 12, 13), True)
        self.assertEqual(Chem.MolToSmiles(anthracene_fragment), '*1cccc2cc3ccccc3*:*:12')

    # Phenanthrene
    def test_recombine_dummy_position_switch_for_a_phenanthrene_fragment_returns_a_phenanthrene_fragment_0(self):
        phenanthrene_fragment = recombine_dummy_position_switch(self.phenanthrene_fragment, (5, 4, 6), True)
        self.assertEqual(Chem.MolToSmiles(phenanthrene_fragment), '*1cccc2*:1:*cc1ccccc12')

    def test_recombine_dummy_position_switch_for_a_phenanthrene_fragment_returns_a_phenanthrene_fragment_1(self):
        phenanthrene_fragment = recombine_dummy_position_switch(self.phenanthrene_fragment, (7, 8, 13), True)
        self.assertEqual(Chem.MolToSmiles(phenanthrene_fragment), '*1cccc2*:1:*cc1ccccc12')

    # Naphtho[2,3-c]thiophene
    def test_recombine_dummy_position_switch_for_a_naphtho_2_3_b_thiophene_fragment_fragment_returns_a_naphtho_2_3_b_thiophene_fragment_0(self):
        naphtho_2_3_b_thiophene_fragment = recombine_dummy_position_switch(self.naphtho_2_3_b_thiophene_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(naphtho_2_3_b_thiophene_fragment), '*1cccc2cc3ccsc3*:*:12')

    def test_recombine_dummy_position_switch_for_a_naphtho_2_3_b_thiophene_fragment_fragment_returns_a_naphtho_2_3_b_thiophene_fragment_1(self):
        naphtho_2_3_b_thiophene_fragment = recombine_dummy_position_switch(self.naphtho_2_3_b_thiophene_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToSmiles(naphtho_2_3_b_thiophene_fragment), '*1cccc2cc3ccsc3*:*:12')

    # Acenaphthylene
    def test_recombine_dummy_position_switch_for_a_acenaphthylene_fragment_fragment_returns_a_acenaphthylene_fragment_0(self):
        acenaphthylene_fragment = recombine_dummy_position_switch(self.acenaphthylene_fragment, (1, 2, 3), True)
        self.assertEqual(Chem.MolToSmiles(acenaphthylene_fragment), '*1ccc2ccc3cc*:*:1c23')

    def test_recombine_dummy_position_switch_for_a_acenaphthylene_fragment_fragment_returns_a_acenaphthylene_fragment_1(self):
        acenaphthylene_fragment = recombine_dummy_position_switch(self.acenaphthylene_fragment, (5, 6, 11), True)
        self.assertEqual(Chem.MolToSmiles(acenaphthylene_fragment), '*1ccc2cccc3c*:*:1c32')

    def test_recombine_dummy_position_switch_for_a_acenaphthylene_fragment_fragment_returns_a_acenaphthylene_fragment_2(self):
        acenaphthylene_fragment = recombine_dummy_position_switch(self.acenaphthylene_fragment, (9, 8, 10), True)
        self.assertEqual(Chem.MolToSmiles(acenaphthylene_fragment), '*1ccc2cccc3c*:*:1c32')

    # Cinnoline
    def test_recombine_dummy_position_switch_for_a_cinnoline_fragment_fragment_returns_a_cinnoline_fragment_0(self):
        cinnoline_fragment = recombine_dummy_position_switch(self.cinnoline_fragment, (5, 4, 6), True)
        self.assertEqual(Chem.MolToSmiles(cinnoline_fragment), '*1cccc2nnc*:*:12')

    def test_recombine_dummy_position_switch_for_a_cinnoline_fragment_fragment_returns_a_1_6_naphthyridine_fragment(self):
        one_6_naphthyridine_fragment = recombine_dummy_position_switch(self.cinnoline_fragment, (2, 3, 9), True)
        self.assertEqual(Chem.MolToSmiles(one_6_naphthyridine_fragment), '*1ccnc2ccn*:*:12')

    def test_recombine_dummy_position_switch_for_a_cinnoline_fragment_fragment_returns_a_quinazoline_fragment(self):
        quinazoline_fragment = recombine_dummy_position_switch(self.cinnoline_fragment, (2, 3, 9), False)
        self.assertEqual(Chem.MolToSmiles(quinazoline_fragment), '*1cccc2ncn*:*:12')
