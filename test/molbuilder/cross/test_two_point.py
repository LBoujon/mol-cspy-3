from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from rdkit import Chem
from cspy.molbuilder.crossover.positions import crossover_positions
from cspy.molbuilder.crossover.crossover import Crossover


@patch('cspy.molbuilder.crossover.crossover.crossover_positions', Mock(return_value=True))
class TestCrossover(TestCase):

    def setUp(self):
        self.isoquinoline = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        self.anthyridine = Chem.MolFromSmiles('C1=CC2=CC3=C(N=CC=C3)N=C2N=C1')
        self.naphtho_2_3_c_thiophene = Chem.MolFromSmiles('s2cc1cc3c(cc1c2)cccc3')
        self.tetracene = Chem.MolFromSmiles('c1ccc2cc3cc4ccccc4cc3cc2c1')
        self.triptycene = Chem.MolFromSmiles('c1ccc2c(c1)C3c4ccccc4C2c5c3cccc5')
        self.benztriptycene = Chem.MolFromSmiles('c1ccc2c(c1)C3C=CC2C=C3')
        self.t1 = Chem.MolFromSmiles('O=C1NC(=O)c2cc3c(cc21)C1c2cc4c(cc2C3c2cc3c(cc21)C(=O)NC3=O)C(=O)NC4=O')
        self.t2 = Chem.MolFromSmiles('O=c1[nH]c2cc3c(cc2[nH]1)C1c2cc4[nH]c(=O)[nH]c4cc2C3c2cc3[nH]c(=O)[nH]c3cc21')

        self.crossover = Crossover(Mock(), Mock())
        self.mock_random = Mock()

    # Isoquinoline
    def test_crossover_of_two_isoquinoline_and_returns_naphthalene_and_2_6_naphthyridine(self):
        self.mock_random.side_effect = [((3, 8), (3, 8), True), False, True]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.isoquinoline, self.isoquinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C8H6N2/c1-3-9-6-8-2-4-10-5-7(1)8/h1-6H')

    def test_crossover_of_two_isoquinoline_and_returns_two_isoquinolines(self):
        self.mock_random.side_effect = [((3, 8), (3, 8), False), True, True]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.isoquinoline, self.isoquinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C9H7N/c1-2-4-9-7-10-6-5-8(9)3-1/h1-7H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C9H7N/c1-2-4-9-7-10-6-5-8(9)3-1/h1-7H')

    def test_crossover_of_two_isoquinoline_and_returns_naphthalene_and_2_7_naphthyridine(self):
        self.mock_random.side_effect = [((3, 8), (3, 8), True), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.isoquinoline, self.isoquinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C8H6N2/c1-3-9-5-8-6-10-4-2-7(1)8/h1-6H')

    # Pyrido[2,3-b][1,8]naphthyridine
    def test_crossover_of_two_anthyridines_returns_two_anthyridines(self):
        self.mock_random.side_effect = [((2, 11), (2, 11), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.anthyridine, self.anthyridine)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C11H7N3/c1-3-8-7-9-4-2-6-13-11(9)14-10(8)12-5-1/h1-7H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C11H7N3/c1-3-8-7-9-4-2-6-13-11(9)14-10(8)12-5-1/h1-7H')

    def test_crossover_of_two_anthyridines_returns_an_anthyridines_and_a_pyrido_2_3_b_1_5_naphthyridine(self):
        self.mock_random.side_effect = [((4, 5), (4, 5), False), False, True]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.anthyridine, self.anthyridine)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C11H7N3/c1-3-8-7-9-4-2-6-13-11(9)14-10(8)12-5-1/h1-7H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C11H7N3/c1-3-8-7-10-9(4-2-5-12-10)14-11(8)13-6-1/h1-7H')

    # Naphtho[2,3-c]thiophene
    def test_crossover_of_two_naphtho_2_3_c_thiophenes_and_returns_two_naphtho_2_3_c_thiophenes(self):
        crossover_positions = [((2, 7), (2, 7), False), ((4, 5), (4, 5), False)]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.naphtho_2_3_c_thiophene, self.naphtho_2_3_c_thiophene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C12H8S/c1-2-4-10-6-12-8-13-7-11(12)5-9(10)3-1/h1-8H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C12H8S/c1-2-4-10-6-12-8-13-7-11(12)5-9(10)3-1/h1-8H')

    # Tetracene
    def test_crossover_of_two_tetracene_and_returns_two_tetracene(self):
        crossover_positions = [
            ((3, 16), (3, 16), False), ((3, 16), (7, 12), True), ((5, 14), (5, 14), False), ((5, 14), (5, 14), True),
            ((7, 12), (3, 16), True), ((7, 12), (7, 12), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.tetracene, self.tetracene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C18H12/c1-2-6-14-10-18-12-16-8-4-3-7-15(16)11-17(18)9-13(14)5-1/h1-12H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C18H12/c1-2-6-14-10-18-12-16-8-4-3-7-15(16)11-17(18)9-13(14)5-1/h1-12H')

    # Triptycene
    def test_crossover_of_two_triptycenes_returns_two_triptycenes(self):
        crossover_positions = [
            ((3, 4), (3, 4), False), ((3, 4), (7, 12), True), ((3, 4), (14, 15), True), ((7, 12), (3, 4), True),
            ((7, 12), (7, 12), False), ((7, 12), (14, 15), False), ((14, 15), (3, 4), True), ((14, 15), (7, 12), False),
            ((14, 15), (14, 15), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.triptycene, self.triptycene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C20H14/c1-2-8-14-13(7-1)19-15-9-3-5-11-17(15)20(14)18-12-6-4-10-16(18)19/h1-12,19-20H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C20H14/c1-2-8-14-13(7-1)19-15-9-3-5-11-17(15)20(14)18-12-6-4-10-16(18)19/h1-12,19-20H')

    # Tricyclo[6.2.2.0~2,7~]dodeca-2,4,6,9,11-pentaene
    def test_crossover_of_two_benztriptycene_returns_anthracene_and_a_dibarrelene(self):
        self.mock_random.side_effect = [((3, 4), (3, 4), True), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.benztriptycene, self.benztriptycene)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C14H12/c1-2-10-4-3-9(1)13-11-5-7-12(8-6-11)14(10)13/h1-12H')

    def test_crossover_of_two_benztriptycene_returns_two_benztriptycene(self):
        self.mock_random.side_effect = [((3, 4), (3, 4), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.benztriptycene, self.benztriptycene)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C12H10/c1-2-4-12-10-7-5-9(6-8-10)11(12)3-1/h1-10H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C12H10/c1-2-4-12-10-7-5-9(6-8-10)11(12)3-1/h1-10H')

    # t1
    def test_crossover_of_two_t1_molecules_returns_two_t1_molecules(self):
        crossover_positions = [
            ((5, 10), (5, 10), False), ((5, 10), (14, 15), True), ((5, 10), (21, 22), True), ((7, 8), (7, 8), False),
            ((7, 8), (12, 17), True), ((7, 8), (19, 24), True), ((12, 17), (7, 8), True), ((12, 17), (12, 17), False),
            ((12, 17), (19, 24), False), ((14, 15), (5, 10), True), ((14, 15), (14, 15), False),
            ((14, 15), (21, 22), False), ((19, 24), (7, 8), True), ((19, 24), (12, 17), False),
            ((19, 24), (19, 24), False), ((21, 22), (5, 10), True), ((21, 22), (14, 15), False),
            ((21, 22), (21, 22), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.t1, self.t1)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C26H11N3O6/c30-21-13-1-7-8(2-14(13)22(31)27-21)20-11-5-17-15(23(32)28-25(17)34)3-9(11)19(7)10-4-16-18(6-12(10)20)26(35)29-24(16)33/h1-6,19-20H,(H,27,30,31)(H,28,32,34)(H,29,33,35)')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C26H11N3O6/c30-21-13-1-7-8(2-14(13)22(31)27-21)20-11-5-17-15(23(32)28-25(17)34)3-9(11)19(7)10-4-16-18(6-12(10)20)26(35)29-24(16)33/h1-6,19-20H,(H,27,30,31)(H,28,32,34)(H,29,33,35)')

    # t2
    def test_crossover_of_two_t2_molecules_returns_two_t2_molecules(self):
        crossover_positions = [
            ((3, 8), (3, 8), False), ((3, 8), (13, 18), True), ((3, 8), (24, 29), True), ((5, 6), (5, 6), False),
            ((5, 6), (11, 20), True), ((5, 6), (22, 31), True), ((11, 20), (5, 6), True), ((11, 20), (11, 20), False),
            ((11, 20), (22, 31), False), ((13, 18), (3, 8), True), ((13, 18), (13, 18), False),
            ((13, 18), (24, 29), False), ((22, 31), (5, 6), True), ((22, 31), (11, 20), False),
            ((22, 31), (22, 31), False), ((24, 29), (3, 8), True), ((24, 29), (13, 18), False),
            ((24, 29), (24, 29), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.t2, self.t2)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C23H14N6O3/c30-21-24-13-1-7-8(2-14(13)25-21)20-11-5-17-15(26-22(31)28-17)3-9(11)19(7)10-4-16-18(6-12(10)20)29-23(32)27-16/h1-6,19-20H,(H2,24,25,30)(H2,26,28,31)(H2,27,29,32)')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C23H14N6O3/c30-21-24-13-1-7-8(2-14(13)25-21)20-11-5-17-15(26-22(31)28-17)3-9(11)19(7)10-4-16-18(6-12(10)20)29-23(32)27-16/h1-6,19-20H,(H2,24,25,30)(H2,26,28,31)(H2,27,29,32)')


class TestCrossoverPositions(TestCase):

    def setUp(self):
        self.isoquinoline = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        self.benzimidazolone = Chem.MolFromSmiles('c1ccc2=NC(=O)N=c2c1')
        self.naphtho_2_3_c_thiophene = Chem.MolFromSmiles('s2cc1cc3c(cc1c2)cccc3')
        self.anthracene = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        self.tetracene = Chem.MolFromSmiles('c1ccc2cc3cc4ccccc4cc3cc2c1')
        self.triptycene = Chem.MolFromSmiles('c1ccc2c(c1)C3c4ccccc4C2c5c3cccc5')
        self.benztriptycene = Chem.MolFromSmiles('c1ccc2c(c1)C3C=CC2C=C3')
        self.t1 = Chem.MolFromSmiles('O=C1NC(=O)c2cc3c(cc21)C1c2cc4c(cc2C3c2cc3c(cc21)C(=O)NC3=O)C(=O)NC4=O')
        self.t2 = Chem.MolFromSmiles('O=c1[nH]c2cc3c(cc2[nH]1)C1c2cc4[nH]c(=O)[nH]c4cc2C3c2cc3[nH]c(=O)[nH]c3cc21')
        self.two_benzothiophene = Chem.MolFromSmiles('c1ccc2cscc2c1')
        self.four_h_thieno_2_3_c_pyrrole = Chem.MolFromSmiles('C1=CNC2=C1SC=C2')
        self.five_h_thieno_2_3_c_pyrrole = Chem.MolFromSmiles('c1csc2c1c[nH]c2')
        self.bisthieno_2_3_b_3_2_d_thiophene = Chem.MolFromSmiles('c1csc2c1c3ccsc3s2')

    # Isoquinoline
    def test_crossover_positions_for_isoquinoline_with_min_and_max_size_of_2(self):
        positions = crossover_positions(self.isoquinoline, self.isoquinoline, 2, 2)
        self.assertEqual(positions, [((3, 8), (3, 8), False), ((3, 8), (3, 8), True)])

    def test_crossover_positions_for_isoquinoline_and_naphtho_2_3_c_thiophene_with_min_and_max_size_of_2_and_3(self):
        positions = crossover_positions(self.isoquinoline, self.naphtho_2_3_c_thiophene, 2, 3)
        self.assertEqual(positions, [
            ((3, 8), (2, 7), False), ((3, 8), (2, 7), True), ((3, 8), (4, 5), False), ((3, 8), (4, 5), True)
        ])

    # Benzimidazolone
    def test_crossover_positions_for_benzimidazolone_and_benztriptycene_with_min_and_max_size_of_2(self):
        positions = crossover_positions(self.benzimidazolone, self.benztriptycene, 2, 2)
        self.assertEqual(positions, [((3, 8), (3, 4), False), ((3, 8), (3, 4), True)])

    # Naphtho[2,3-c]thiophene
    def test_crossover_positions_for_naphtho_2_3_c_thiophene_with_min_and_max_size_of_3(self):
        positions = crossover_positions(self.naphtho_2_3_c_thiophene, self.naphtho_2_3_c_thiophene, 3, 3)
        self.assertEqual(positions, [((2, 7), (2, 7), False), ((4, 5), (4, 5), False)])

    def test_crossover_positions_for_naphtho_2_3_c_thiophene_with_min_and_max_size_of_2_and_4(self):
        positions = crossover_positions(self.naphtho_2_3_c_thiophene, self.naphtho_2_3_c_thiophene, 2, 4)
        self.assertEqual(positions, [
            ((2, 7), (2, 7), False), ((2, 7), (4, 5), False), ((4, 5), (2, 7), False), ((4, 5), (4, 5), False)
        ])

    # Anthracene
    def test_crossover_positions_for_anthracene_with_min_and_max_size_of_3(self):
        positions = crossover_positions(self.anthracene, self.anthracene, 3, 3)
        self.assertEqual(positions, [
            ((3, 12), (3, 12), False), ((3, 12), (5, 10), True), ((5, 10), (3, 12), True), ((5, 10), (5, 10), False)
        ])

    # Tetracene
    def test_crossover_positions_for_tetracene_with_min_and_max_size_of_4(self):
        positions = crossover_positions(self.tetracene, self.tetracene, 4, 4)
        self.assertEqual(positions, [
            ((3, 16), (3, 16), False), ((3, 16), (7, 12), True), ((5, 14), (5, 14), False), ((5, 14), (5, 14), True),
            ((7, 12), (3, 16), True), ((7, 12), (7, 12), False)
        ])

    # Triptycene
    def test_crossover_positions_for_triptycene_with_min_and_max_size_of_4(self):
        positions = crossover_positions(self.triptycene, self.triptycene, 4, 4)
        self.assertEqual(positions, [
            ((3, 4), (3, 4), False), ((3, 4), (7, 12), True), ((3, 4), (14, 15), True), ((7, 12), (3, 4), True),
            ((7, 12), (7, 12), False), ((7, 12), (14, 15), False), ((14, 15), (3, 4), True), ((14, 15), (7, 12), False),
            ((14, 15), (14, 15), False)
        ])

    def test_crossover_positions_for_triptycene_with_min_and_max_size_of_2_and_6(self):
        positions = crossover_positions(self.triptycene, self.triptycene, 2, 6)
        self.assertEqual(positions, [
            ((3, 4), (3, 4), False), ((3, 4), (3, 4), True), ((3, 4), (7, 12), False), ((3, 4), (7, 12), True),
            ((3, 4), (14, 15), False), ((3, 4), (14, 15), True), ((7, 12), (3, 4), False), ((7, 12), (3, 4), True),
            ((7, 12), (7, 12), False), ((7, 12), (7, 12), True), ((7, 12), (14, 15), False), ((7, 12), (14, 15), True),
            ((14, 15), (3, 4), False), ((14, 15), (3, 4), True), ((14, 15), (7, 12), False), ((14, 15), (7, 12), True),
            ((14, 15), (14, 15), False), ((14, 15), (14, 15), True)
        ])

    # Tricyclo[6.2.2.0~2,7~]dodeca-2,4,6,9,11-pentaene
    def test_crossover_positions_for_benztriptycene_with_min_and_max_size_of_2(self):
        positions = crossover_positions(self.benztriptycene, self.benztriptycene, 2, 2)
        self.assertEqual(positions, [((3, 4), (3, 4), False), ((3, 4), (3, 4), True)])

    # t1
    def test_crossover_positions_for_t1_with_min_and_max_size_of_7(self):
        positions = crossover_positions(self.t1, self.t1, 7, 7)
        self.assertEqual(positions, [
            ((5, 10), (5, 10), False), ((5, 10), (14, 15), True), ((5, 10), (21, 22), True), ((7, 8), (7, 8), False),
            ((7, 8), (12, 17), True), ((7, 8), (19, 24), True), ((12, 17), (7, 8), True), ((12, 17), (12, 17), False),
            ((12, 17), (19, 24), False), ((14, 15), (5, 10), True), ((14, 15), (14, 15), False),
            ((14, 15), (21, 22), False), ((19, 24), (7, 8), True), ((19, 24), (12, 17), False),
            ((19, 24), (19, 24), False), ((21, 22), (5, 10), True), ((21, 22), (14, 15), False),
            ((21, 22), (21, 22), False)
        ])

    def test_crossover_positions_for_t1_with_min_and_max_size_of_2_and_12(self):
        positions = crossover_positions(self.t1, self.t1, 2, 12)
        self.assertEqual(positions, [
            ((5, 10), (5, 10), False), ((5, 10), (5, 10), True), ((5, 10), (7, 8), False), ((5, 10), (7, 8), True),
            ((5, 10), (12, 17), False), ((5, 10), (12, 17), True), ((5, 10), (14, 15), False),
            ((5, 10), (14, 15), True), ((5, 10), (19, 24), False), ((5, 10), (19, 24), True),
            ((5, 10), (21, 22), False), ((5, 10), (21, 22), True), ((7, 8), (5, 10), False), ((7, 8), (5, 10), True),
            ((7, 8), (7, 8), False), ((7, 8), (7, 8), True), ((7, 8), (12, 17), False), ((7, 8), (12, 17), True),
            ((7, 8), (14, 15), False), ((7, 8), (14, 15), True), ((7, 8), (19, 24), False), ((7, 8), (19, 24), True),
            ((7, 8), (21, 22), False), ((7, 8), (21, 22), True), ((12, 17), (5, 10), False), ((12, 17), (5, 10), True),
            ((12, 17), (7, 8), False), ((12, 17), (7, 8), True), ((12, 17), (12, 17), False),
            ((12, 17), (12, 17), True), ((12, 17), (14, 15), False), ((12, 17), (14, 15), True),
            ((12, 17), (19, 24), False), ((12, 17), (19, 24), True), ((12, 17), (21, 22), False),
            ((12, 17), (21, 22), True), ((14, 15), (5, 10), False), ((14, 15), (5, 10), True),
            ((14, 15), (7, 8), False), ((14, 15), (7, 8), True), ((14, 15), (12, 17), False),
            ((14, 15), (12, 17), True), ((14, 15), (14, 15), False), ((14, 15), (14, 15), True),
            ((14, 15), (19, 24), False), ((14, 15), (19, 24), True), ((14, 15), (21, 22), False),
            ((14, 15), (21, 22), True), ((19, 24), (5, 10), False), ((19, 24), (5, 10), True),
            ((19, 24), (7, 8), False), ((19, 24), (7, 8), True), ((19, 24), (12, 17), False),
            ((19, 24), (12, 17), True), ((19, 24), (14, 15), False), ((19, 24), (14, 15), True),
            ((19, 24), (19, 24), False), ((19, 24), (19, 24), True), ((19, 24), (21, 22), False),
            ((19, 24), (21, 22), True), ((21, 22), (5, 10), False), ((21, 22), (5, 10), True),
            ((21, 22), (7, 8), False), ((21, 22), (7, 8), True), ((21, 22), (12, 17), False),
            ((21, 22), (12, 17), True), ((21, 22), (14, 15), False), ((21, 22), (14, 15), True),
            ((21, 22), (19, 24), False), ((21, 22), (19, 24), True), ((21, 22), (21, 22), False),
            ((21, 22), (21, 22), True)
        ])

    # t2
    def test_crossover_positions_for_t2_with_min_and_max_size_of_7(self):
        positions = crossover_positions(self.t2, self.t2, 7, 7)
        self.assertEqual(positions, [
            ((3, 8), (3, 8), False), ((3, 8), (13, 18), True), ((3, 8), (24, 29), True), ((5, 6), (5, 6), False),
            ((5, 6), (11, 20), True), ((5, 6), (22, 31), True), ((11, 20), (5, 6), True), ((11, 20), (11, 20), False),
            ((11, 20), (22, 31), False), ((13, 18), (3, 8), True), ((13, 18), (13, 18), False),
            ((13, 18), (24, 29), False), ((22, 31), (5, 6), True), ((22, 31), (11, 20), False),
            ((22, 31), (22, 31), False), ((24, 29), (3, 8), True), ((24, 29), (13, 18), False),
            ((24, 29), (24, 29), False)
        ])

    # 2-Benzothiophene
    def test_crossover_positions_for_2_benzothiophene_and_4h_thieno_2_3_c_pyrrole_with_min_and_max_size_of_2(self):
        positions = crossover_positions(self.two_benzothiophene, self.four_h_thieno_2_3_c_pyrrole, 2, 2)
        self.assertEqual(positions, [((3, 7), (3, 4), False), ((3, 7), (3, 4), True)])

    def test_crossover_positions_for_2_benzothiophene_and_5h_thieno_2_3_c_pyrrole_with_min_and_max_size_of_2(self):
        positions = crossover_positions(self.two_benzothiophene, self.five_h_thieno_2_3_c_pyrrole, 2, 2)
        self.assertEqual(positions, [((3, 7), (3, 4), False)])

    def test_crossover_positions_for_2_benzothiophene_and_bisthieno_2_3_b_3_2_d_thiophene_with_min_and_max_size_of_2_and_3(self):
        positions = crossover_positions(self.two_benzothiophene, self.bisthieno_2_3_b_3_2_d_thiophene, 2, 3)
        self.assertEqual(positions, [
            ((3, 7), (3, 4), False), ((3, 7), (3, 4), True), ((3, 7), (5, 9), False), ((3, 7), (5, 9), True)
        ])
