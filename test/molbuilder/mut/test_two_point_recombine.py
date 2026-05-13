from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from rdkit import Chem
from cspy.molbuilder.mutation.recombine import recombine
from cspy.molbuilder.mutation.recombine import recombine_two_point_positions
from cspy.molbuilder.mutation.recombine import recombine_dummy_position_switch


class TestTwoPointRecombineMutation(TestCase):

    def setUp(self):
        self.indole = Chem.MolFromSmiles('c1ccc2c(c1)cc[nH]2')
        self.one_h_inden_1_one = Chem.MolFromSmiles('c1ccc2c(c1)C=CC2=O')
        self.thieno_3_4_e_2_benzothiophene = Chem.MolFromSmiles('c1cc2cscc2c3c1csc3')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')

        self.mock_random = Mock()

    def test_recombine_mutation_for_indole_returns_indole_0(self):
        self.mock_random.side_effect = [((0, 1), (0, 1)), True, True, True]
        with patch('cspy.molbuilder.mutation.recombine.choice', self.mock_random):
            indole = recombine(self.indole, (3, 4))
            self.assertEqual(Chem.MolToInchi(indole, options='-SNon'), 'InChI=1S/C8H7N/c1-2-4-8-7(3-1)5-6-9-8/h1-6,9H')

    def test_recombine_mutation_for_indole_returns_isoindole_0(self):
        self.mock_random.side_effect = [((0, 1), (1, 2)), True, True, True]
        with patch('cspy.molbuilder.mutation.recombine.choice', self.mock_random):
            isoindole = recombine(self.indole, (3, 4))
            self.assertEqual(Chem.MolToInchi(isoindole, options='-SNon'), 'InChI=1S/C8H7N/c1-2-4-8-6-9-5-7(8)3-1/h1-6,9H')

    def test_recombine_mutation_for_indole_returns_indole_1(self):
        self.mock_random.side_effect = [((0, 1), (2, 3)), True, True, True]
        with patch('molbuilder.src.mutation.recombine.choice', self.mock_random):
            indole = recombine(self.indole, (3, 4))
            self.assertEqual(Chem.MolToInchi(indole, options='-SNon'), 'InChI=1S/C8H7N/c1-2-4-8-7(3-1)5-6-9-8/h1-6,9H')

    def test_recombine_mutation_for_phenanthrene_returns_phenanthrene(self):
        self.mock_random.side_effect = [((0, 1), (0, 1)), True, True, True]
        with patch('cspy.molbuilder.mutation.recombine.choice', self.mock_random):
            phenanthrene = recombine(self.phenanthrene, (3, 4))
            self.assertEqual(Chem.MolToInchi(phenanthrene, options='-SNon'), 'InChI=1S/C14H10/c1-3-7-13-11(5-1)9-10-12-6-2-4-8-14(12)13/h1-10H')

    def test_recombine_mutation_for_phenanthrene_returns_anthracene(self):
        self.mock_random.side_effect = [((1, 2), (7, 8)), True, True, True]
        with patch('cspy.molbuilder.mutation.recombine.choice', self.mock_random):
            anthracene = recombine(self.phenanthrene, (3, 4))
            self.assertEqual(Chem.MolToInchi(anthracene, options='-SNon'), 'InChI=1S/C14H10/c1-2-6-12-10-14-8-4-3-7-13(14)9-11(12)5-1/h1-10H')


class TestRecombineTwoPointPositions(TestCase):

    def setUp(self):
        self.benzene_fragment = Chem.MolFromSmiles('c1c**cc1')
        self.thiophene_fragment = Chem.MolFromSmiles('c1**sc1')
        self.cyclopentadienone_fragment = Chem.MolFromSmiles('*1=*C(=O)C=C1')
        self.two_benzothiophene = Chem.MolFromSmiles('*1*cc2cscc2c1')
        self.thieno_3_4_b_pyridine_fragment = Chem.MolFromSmiles('c1scc2n**cc12')
        self.barrelene_fragment = Chem.MolFromSmiles('*1=*C([H])2C=CC([H])1C=C2')
        self.maleimide_fragment = Chem.MolFromSmiles('*1=*C(=O)NC1=O')
        self.naphthalene_fragment = Chem.MolFromSmiles('*1*cc2ccccc2c1')
        self.pyridine_fragment = Chem.MolFromSmiles('*1*cncc1')
        self.spiro_4_4_nona_1_3_6_8_tetraene_fragment = Chem.MolFromSmiles('*1=*C2(C=C1)C=CC=C2')

    def test_recombine_two_point_positions_for_a_benzene_fragment_and_a_thiophene_fragment(self):
        bondable = recombine_two_point_positions(self.benzene_fragment, self.thiophene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (0, 4)), ((0, 1), (1, 2)),
            ((0, 5), (0, 1)), ((0, 5), (0, 4)), ((0, 5), (1, 2)),
            ((1, 2), (0, 1)), ((1, 2), (0, 4)), ((1, 2), (1, 2)),
            ((2, 3), (0, 1)), ((2, 3), (0, 4)), ((2, 3), (1, 2)),
            ((3, 4), (0, 1)), ((3, 4), (0, 4)), ((3, 4), (1, 2)),
            ((4, 5), (0, 1)), ((4, 5), (0, 4)), ((4, 5), (1, 2))
        ])

    def test_recombine_two_point_positions_for_a_benzene_fragment_and_a_cyclopentadienone_fragment(self):
        bondable = recombine_two_point_positions(self.benzene_fragment, self.cyclopentadienone_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (0, 5)), ((0, 1), (4, 5)),
            ((0, 5), (0, 1)), ((0, 5), (0, 5)), ((0, 5), (4, 5)),
            ((1, 2), (0, 1)), ((1, 2), (0, 5)), ((1, 2), (4, 5)),
            ((2, 3), (0, 1)), ((2, 3), (0, 5)), ((2, 3), (4, 5)),
            ((3, 4), (0, 1)), ((3, 4), (0, 5)), ((3, 4), (4, 5)),
            ((4, 5), (0, 1)), ((4, 5), (0, 5)), ((4, 5), (4, 5))
        ])

    def test_recombine_two_point_positions_for_a_benzene_fragment_and_a_maleimide_fragment(self):
        bondable = recombine_two_point_positions(self.benzene_fragment, self.maleimide_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 5), (0, 1)), ((1, 2), (0, 1)), ((2, 3), (0, 1)), ((3, 4), (0, 1)), ((4, 5), (0, 1))
        ])

    def test_recombine_two_point_positions_for_a_benzene_fragment_and_a_naphthalene_fragment(self):
        bondable = recombine_two_point_positions(self.benzene_fragment, self.naphthalene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (0, 9)), ((0, 1), (1, 2)), ((0, 1), (4, 5)), ((0, 1), (5, 6)), ((0, 1), (6, 7)),
            ((0, 5), (0, 1)), ((0, 5), (0, 9)), ((0, 5), (1, 2)), ((0, 5), (4, 5)), ((0, 5), (5, 6)), ((0, 5), (6, 7)),
            ((1, 2), (0, 1)), ((1, 2), (0, 9)), ((1, 2), (1, 2)), ((1, 2), (4, 5)), ((1, 2), (5, 6)), ((1, 2), (6, 7)),
            ((2, 3), (0, 1)), ((2, 3), (0, 9)), ((2, 3), (1, 2)), ((2, 3), (4, 5)), ((2, 3), (5, 6)), ((2, 3), (6, 7)),
            ((3, 4), (0, 1)), ((3, 4), (0, 9)), ((3, 4), (1, 2)), ((3, 4), (4, 5)), ((3, 4), (5, 6)), ((3, 4), (6, 7)),
            ((4, 5), (0, 1)), ((4, 5), (0, 9)), ((4, 5), (1, 2)), ((4, 5), (4, 5)), ((4, 5), (5, 6)), ((4, 5), (6, 7))
        ])

    def test_recombine_two_point_positions_for_a_two_benzothiophene_and_a_thiophene_fragment(self):
        bondable = recombine_two_point_positions(self.two_benzothiophene, self.thiophene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 4)), ((0, 1), (1, 2)), ((0, 8), (0, 1)), ((0, 8), (0, 4)), ((0, 8), (1, 2)), ((1, 2), (0, 1)),
            ((1, 2), (0, 4)), ((1, 2), (1, 2))
        ])

    def test_recombine_two_point_positions_for_a_thieno_3_4_b_pyridine_fragment_and_a_thiophene_fragment(self):
        bondable = recombine_two_point_positions(self.thieno_3_4_b_pyridine_fragment, self.thiophene_fragment)
        self.assertEqual(bondable, [
            ((4, 5), (0, 1)), ((4, 5), (0, 4)), ((4, 5), (1, 2)), ((5, 6), (0, 4)), ((5, 6), (1, 2)), ((6, 7), (0, 1)),
            ((6, 7), (0, 4)), ((6, 7), (1, 2))
        ])

    def test_recombine_two_point_positions_for_a_barrelene_fragment_and_a_thiophene_fragment(self):
        bondable = recombine_two_point_positions(self.barrelene_fragment, self.thiophene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (0, 4)), ((0, 1), (1, 2)), ((3, 4), (0, 1)), ((3, 4), (0, 4)), ((3, 4), (1, 2)),
            ((6, 7), (0, 1)), ((6, 7), (0, 4)), ((6, 7), (1, 2))
        ])

    def test_recombine_two_point_positions_for_a_benzene_fragment_and_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment(self):
        bondable = recombine_two_point_positions(self.benzene_fragment, self.spiro_4_4_nona_1_3_6_8_tetraene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (0, 4)), ((0, 1), (3, 4)), ((0, 1), (5, 6)), ((0, 1), (6, 7)), ((0, 1), (7, 8)),
            ((0, 5), (0, 1)), ((0, 5), (0, 4)), ((0, 5), (3, 4)), ((0, 5), (5, 6)), ((0, 5), (6, 7)), ((0, 5), (7, 8)),
            ((1, 2), (0, 1)), ((1, 2), (0, 4)), ((1, 2), (3, 4)), ((1, 2), (5, 6)), ((1, 2), (6, 7)), ((1, 2), (7, 8)),
            ((2, 3), (0, 1)), ((2, 3), (0, 4)), ((2, 3), (3, 4)), ((2, 3), (5, 6)), ((2, 3), (6, 7)), ((2, 3), (7, 8)),
            ((3, 4), (0, 1)), ((3, 4), (0, 4)), ((3, 4), (3, 4)), ((3, 4), (5, 6)), ((3, 4), (6, 7)), ((3, 4), (7, 8)),
            ((4, 5), (0, 1)), ((4, 5), (0, 4)), ((4, 5), (3, 4)), ((4, 5), (5, 6)), ((4, 5), (6, 7)), ((4, 5), (7, 8))
        ])

    def test_recombine_two_point_positions_for_a_thiophene_fragment_and_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment(self):
        bondable = recombine_two_point_positions(self.thiophene_fragment, self.spiro_4_4_nona_1_3_6_8_tetraene_fragment)
        self.assertEqual(bondable, [
            ((0, 1), (0, 1)), ((0, 1), (3, 4)), ((0, 1), (5, 6)), ((0, 1), (7, 8)),
            ((0, 4), (0, 1)), ((0, 4), (0, 4)), ((0, 4), (3, 4)), ((0, 4), (5, 6)), ((0, 4), (6, 7)), ((0, 4), (7, 8)),
            ((1, 2), (0, 1)), ((1, 2), (0, 4)), ((1, 2), (3, 4)), ((1, 2), (5, 6)), ((1, 2), (6, 7)), ((1, 2), (7, 8))
        ])


class TestTwoPointRecombineDummyPositionSwitch(TestCase):

    def setUp(self):
        self.cyclobutadiene_fragment = Chem.MolFromSmiles('*1=*C=C1')
        self.pyrrole_fragment = Chem.MolFromSmiles('*1*c[nH]c1')
        self.pyridazine_fragment = Chem.MolFromSmiles('*1*cnnc1')
        self.one_2_5_thiadiazole_fragment = Chem.MolFromSmiles('*1*nsn1')
        self.barralene_fragment = Chem.MolFromSmiles('*1=*C2C=CC1C=C2')
        self.difluorobenzene_fragment_1 = Chem.MolFromSmiles('Fc1c**cc1F')
        self.difluorobenzene_fragment_2 = Chem.MolFromSmiles('Fc1**ccc1F')
        self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment = Chem.MolFromSmiles('*1*cc2c(c1)C([H])3C=CC([H])2C=C3')
        self.spiro_4_4_nona_1_3_6_8_tetraene_fragment = Chem.MolFromSmiles('*1=*C2(C=C1)C=CC=C2')

    # Cyclobutadiene
    def test_recombine_dummy_position_switch_for_a_cyclobutadiene_fragment_returns_a_cyclobutadiene_fragment_0(self):
        cyclobutadiene_fragment = recombine_dummy_position_switch(self.cyclobutadiene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToSmiles(cyclobutadiene_fragment), '*1:*cc1')

    def test_recombine_dummy_position_switch_for_a_cyclobutadiene_fragment_returns_a_cyclobutadiene_fragment_1(self):
        cyclobutadiene_fragment = recombine_dummy_position_switch(self.cyclobutadiene_fragment, (2, 3), True)
        self.assertEqual(Chem.MolToSmiles(cyclobutadiene_fragment), '*1:*cc1')

    # Pyrrole
    def test_recombine_dummy_position_switch_for_a_pyrrole_fragment_returns_a_pyrrole_fragment_0(self):
        pyrrole_fragment = recombine_dummy_position_switch(self.pyrrole_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToSmiles(pyrrole_fragment), '*1:*c[nH]c1')

    def test_recombine_dummy_position_switch_for_a_pyrrole_fragment_returns_a_pyrrole_fragment_1(self):
        pyrrole_fragment = recombine_dummy_position_switch(self.pyrrole_fragment, (1, 2), True)
        self.assertEqual(Chem.MolToSmiles(pyrrole_fragment), '*1:*[nH]cc1')

    def test_recombine_dummy_position_switch_for_a_pyrrole_fragment_returns_a_pyrrole_fragment_2(self):
        pyrrole_fragment = recombine_dummy_position_switch(self.pyrrole_fragment, (0, 4), True)
        self.assertEqual(Chem.MolToSmiles(pyrrole_fragment), '*1:*[nH]cc1')

    # Pyridazine
    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyridazine_fragment_0(self):
        pyridazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToSmiles(pyridazine_fragment), '*1:*cnnc1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyridazine_fragment_1(self):
        pyridazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (0, 5), True)
        self.assertEqual(Chem.MolToSmiles(pyridazine_fragment), '*1:*nncc1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyridazine_fragment_2(self):
        pyridazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (1, 2), True)
        self.assertEqual(Chem.MolToSmiles(pyridazine_fragment), '*1:*nncc1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyrimidine_fragment(self):
        pyrimidine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (2, 3), False)
        self.assertEqual(Chem.MolToSmiles(pyrimidine_fragment), '*1:*ncnc1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyrazine_fragment(self):
        pyrazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (2, 3), True)
        self.assertEqual(Chem.MolToSmiles(pyrazine_fragment), '*1:*nccn1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyridazine_fragment_3(self):
        pyridazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (3, 4), True)
        self.assertEqual(Chem.MolToSmiles(pyridazine_fragment), '*1:*cnnc1')

    def test_recombine_dummy_position_switch_for_a_pyridazine_fragment_returns_a_pyridazine_fragment_4(self):
        pyridazine_fragment = recombine_dummy_position_switch(self.pyridazine_fragment, (3, 4), False)
        self.assertEqual(Chem.MolToSmiles(pyridazine_fragment), '*1:*cnnc1')

    # 1,2,5-Thiadiazole
    def test_recombine_dummy_position_switch_for_a_1_2_5_thiadiazole_fragment_returns_a_1_2_5_thiadiazole_fragment(self):
        one_2_5_thiadiazole_fragment = recombine_dummy_position_switch(self.one_2_5_thiadiazole_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToSmiles(one_2_5_thiadiazole_fragment), '*1:*nsn1')

    def test_recombine_dummy_position_switch_for_a_1_2_5_thiadiazole_fragment_returns_a_1_2_3_thiadiazole_0(self):
        one_2_3_thiadiazole_fragment = recombine_dummy_position_switch(self.one_2_5_thiadiazole_fragment, (1, 2), False)
        self.assertEqual(Chem.MolToSmiles(one_2_3_thiadiazole_fragment), '*1:*snn1')

    def test_recombine_dummy_position_switch_for_a_1_2_5_thiadiazole_fragment_returns_a_1_2_3_thiadiazole_1(self):
        one_2_3_thiadiazole_fragment = recombine_dummy_position_switch(self.one_2_5_thiadiazole_fragment, (0, 4), False)
        self.assertEqual(Chem.MolToSmiles(one_2_3_thiadiazole_fragment), '*1:*snn1')

    # Barralene
    def test_recombine_dummy_position_switch_for_a_barralene_fragment_returns_barralene_a_fragment_0(self):
        barralene_fragment = recombine_dummy_position_switch(self.barralene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToSmiles(barralene_fragment), '*1=*C2C=CC1C=C2')

    def test_recombine_dummy_position_switch_for_a_barralene_fragment_returns_a_barralene_fragment_1(self):
        barralene_fragment = recombine_dummy_position_switch(self.barralene_fragment, (3, 4), False)
        self.assertEqual(Chem.MolToSmiles(barralene_fragment), '*1=*C2C=CC1C=C2')

    def test_recombine_dummy_position_switch_for_a_barralene_fragment_returns_a_barralene_fragment_2(self):
        barralene_fragment = recombine_dummy_position_switch(self.barralene_fragment, (6, 7), False)
        self.assertEqual(Chem.MolToSmiles(barralene_fragment), '*1=*C2C=CC1C=C2')

    # 1,2-Difluorobenzene
    def test_recombine_dummy_position_switch_for_a_difluorobenzene_fragment_1_returns_a_difluorobenzene_fragment(self):
        difluorobenzene_fragment = recombine_dummy_position_switch(self.difluorobenzene_fragment_1, (1, 6), False)
        self.assertEqual(Chem.MolToSmiles(difluorobenzene_fragment), 'Fc1c*:*cc1F')

    def test_recombine_dummy_position_switch_for_a_difluorobenzene_fragment_2_returns_a_difluorobenzene_fragment(self):
        difluorobenzene_fragment = recombine_dummy_position_switch(self.difluorobenzene_fragment_2, (1, 2), False)
        self.assertEqual(Chem.MolToSmiles(difluorobenzene_fragment), 'Fc1*:*c(F)cc1')

    # Tricyclo[6.2.2.0~2,7~]dodeca-2,4,6,9,11-pentaene
    def test_recombine_dummy_position_switch_for_a_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment_returns_a_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment_0(self):
        tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment = recombine_dummy_position_switch(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment, (7, 8), True)
        self.assertEqual(Chem.MolToSmiles(tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment), '*1=*C2C=CC1c1ccccc12')

    def test_recombine_dummy_position_switch_for_a_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment_returns_a_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment_1(self):
        tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment = recombine_dummy_position_switch(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment, (10, 11), True)
        self.assertEqual(Chem.MolToSmiles(tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_fragment), '*1=*C2C=CC1c1ccccc12')

    # Spiro[4.4]nona-1,3,6,8-tetraene
    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_0(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')

    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_1(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (0, 4), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=CC2(C=*1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (0, 4), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=CC2(C=*1)C=CC=C2')

    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_2(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (3, 4), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (3, 4), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')

    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_3(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (5, 6), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (5, 6), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')

    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_4(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (6, 7), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=CC2(C=*1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (6, 7), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=CC2(C=*1)C=CC=C2')

    def test_recombine_dummy_position_switch_for_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_returns_a_spiro_4_4_nona_1_3_6_8_tetraene_fragment_5(self):
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (7, 8), False)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')
        spiro_4_4_nona_1_3_6_8_tetraene_fragment = recombine_dummy_position_switch(self.spiro_4_4_nona_1_3_6_8_tetraene_fragment, (7, 8), True)
        self.assertEqual(Chem.MolToSmiles(spiro_4_4_nona_1_3_6_8_tetraene_fragment), '*1=*C2(C=C1)C=CC=C2')
