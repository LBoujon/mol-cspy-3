from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from rdkit import Chem
from cspy.molbuilder.crossover.positions import crossover_positions
from cspy.molbuilder.crossover.crossover import Crossover


@patch('cspy.molbuilder.crossover.crossover.crossover_positions', Mock(return_value=True))
class TestCrossover(TestCase):

    def setUp(self):
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.naphtho_8_1_2_cde_cinnoline = Chem.MolFromSmiles('c1cc2cccc3nnc4cccc1c4c23')
        self.coronene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')

        self.crossover = Crossover(Mock(), Mock())
        self.mock_random = Mock()

    def test_crossover_of_two_pyrene_and_returns_two_pyrene(self):
        crossover_positions = [
            ((2, 11, 10, 5), (2, 11, 10, 5), False), ((2, 11, 10, 5), (9, 10, 11, 12), False),
            ((9, 10, 11, 12), (2, 11, 10, 5), False), ((9, 10, 11, 12), (9, 10, 11, 12), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.pyrene, self.pyrene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')

    def test_crossover_of_pyrene_and_naphtho_8_1_2_cde_cinnoline_returns_pyrene_and_naphtho_8_1_2_cde_cinnoline(self):
        self.mock_random.side_effect = [((2, 11, 10, 5), (2, 15, 14, 13), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.pyrene, self.naphtho_8_1_2_cde_cinnoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C14H8N2/c1-3-9-7-8-10-4-2-6-12-14(10)13(9)11(5-1)15-16-12/h1-8H')

    def test_crossover_of_pyrene_and_naphtho_8_1_2_cde_cinnoline_returns_naphtho_8_1_2_cde_cinnoline_and_pyrene(self):
        self.mock_random.side_effect = [((2, 11, 10, 5), (6, 15, 14, 9), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.pyrene, self.naphtho_8_1_2_cde_cinnoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C14H8N2/c1-3-9-7-8-10-4-2-6-12-14(10)13(9)11(5-1)15-16-12/h1-8H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C16H10/c1-3-11-7-9-13-5-2-6-14-10-8-12(4-1)15(11)16(13)14/h1-10H')

    def test_crossover_of_two_coronene_and_returns_two_coronene(self):
        crossover_positions = [
            ((2, 19, 18, 17), (2, 19, 18, 17), False), ((2, 19, 18, 17), (2, 19, 20, 5), False),
            ((2, 19, 18, 17), (5, 20, 21, 8), False), ((2, 19, 18, 17), (8, 21, 22, 11), False),
            ((2, 19, 18, 17), (11, 22, 23, 14), False), ((2, 19, 18, 17), (14, 23, 18, 17), False),
            ((2, 19, 20, 5), (2, 19, 18, 17), False), ((2, 19, 20, 5), (2, 19, 20, 5), False),
            ((2, 19, 20, 5), (5, 20, 21, 8), False), ((2, 19, 20, 5), (8, 21, 22, 11), False),
            ((2, 19, 20, 5), (11, 22, 23, 14), False), ((2, 19, 20, 5), (14, 23, 18, 17), False),
            ((5, 20, 21, 8), (2, 19, 18, 17), False), ((5, 20, 21, 8), (2, 19, 20, 5), False),
            ((5, 20, 21, 8), (5, 20, 21, 8), False), ((5, 20, 21, 8), (8, 21, 22, 11), False),
            ((5, 20, 21, 8), (11, 22, 23, 14), False), ((5, 20, 21, 8), (14, 23, 18, 17), False),
            ((8, 21, 22, 11), (2, 19, 18, 17), False), ((8, 21, 22, 11), (2, 19, 20, 5), False),
            ((8, 21, 22, 11), (5, 20, 21, 8), False), ((8, 21, 22, 11), (8, 21, 22, 11), False),
            ((8, 21, 22, 11), (11, 22, 23, 14), False), ((8, 21, 22, 11), (14, 23, 18, 17), False),
            ((11, 22, 23, 14), (2, 19, 18, 17), False), ((11, 22, 23, 14), (2, 19, 20, 5), False),
            ((11, 22, 23, 14), (5, 20, 21, 8), False), ((11, 22, 23, 14), (8, 21, 22, 11), False),
            ((11, 22, 23, 14), (11, 22, 23, 14), False), ((11, 22, 23, 14), (14, 23, 18, 17), False),
            ((14, 23, 18, 17), (2, 19, 18, 17), False), ((14, 23, 18, 17), (2, 19, 20, 5), False),
            ((14, 23, 18, 17), (5, 20, 21, 8), False), ((14, 23, 18, 17), (8, 21, 22, 11), False),
            ((14, 23, 18, 17), (11, 22, 23, 14), False), ((14, 23, 18, 17), (14, 23, 18, 17), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.coronene, self.coronene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C24H12/c1-2-14-5-6-16-9-11-18-12-10-17-8-7-15-4-3-13(1)19-20(14)22(16)24(18)23(17)21(15)19/h1-12H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C24H12/c1-2-14-5-6-16-9-11-18-12-10-17-8-7-15-4-3-13(1)19-20(14)22(16)24(18)23(17)21(15)19/h1-12H')


class TestCrossoverPositions(TestCase):

    def setUp(self):
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.coronene = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')

    def test_crossover_positions_for_pyrene_with_min_and_max_size_of_4(self):
        positions = crossover_positions(self.pyrene, self.pyrene, 4, 4)
        self.assertEqual(positions, [
            ((2, 11, 10, 5), (2, 11, 10, 5), False), ((2, 11, 10, 5), (9, 10, 11, 12), False),
            ((9, 10, 11, 12), (2, 11, 10, 5), False), ((9, 10, 11, 12), (9, 10, 11, 12), False)
        ])

    def test_crossover_positions_for_pyrene_with_min_and_max_size_of_2_and_6(self):
        positions = crossover_positions(self.pyrene, self.pyrene, 2, 6)
        self.assertEqual(positions, [
            ((2, 11, 10, 5), (2, 11, 10, 5), False), ((2, 11, 10, 5), (9, 10, 11, 12), False),
            ((9, 10, 11, 12), (2, 11, 10, 5), False), ((9, 10, 11, 12), (9, 10, 11, 12), False)
        ])

    def test_crossover_positions_for_coronene_with_min_and_max_size_of_7(self):
        positions = crossover_positions(self.coronene, self.coronene, 7, 7)
        self.assertEqual(positions, [
            ((2, 19, 18, 17), (2, 19, 18, 17), False), ((2, 19, 18, 17), (2, 19, 20, 5), False),
            ((2, 19, 18, 17), (5, 20, 21, 8), False), ((2, 19, 18, 17), (8, 21, 22, 11), False),
            ((2, 19, 18, 17), (11, 22, 23, 14), False), ((2, 19, 18, 17), (14, 23, 18, 17), False),
            ((2, 19, 20, 5), (2, 19, 18, 17), False), ((2, 19, 20, 5), (2, 19, 20, 5), False),
            ((2, 19, 20, 5), (5, 20, 21, 8), False), ((2, 19, 20, 5), (8, 21, 22, 11), False),
            ((2, 19, 20, 5), (11, 22, 23, 14), False), ((2, 19, 20, 5), (14, 23, 18, 17), False),
            ((5, 20, 21, 8), (2, 19, 18, 17), False), ((5, 20, 21, 8), (2, 19, 20, 5), False),
            ((5, 20, 21, 8), (5, 20, 21, 8), False), ((5, 20, 21, 8), (8, 21, 22, 11), False),
            ((5, 20, 21, 8), (11, 22, 23, 14), False), ((5, 20, 21, 8), (14, 23, 18, 17), False),
            ((8, 21, 22, 11), (2, 19, 18, 17), False), ((8, 21, 22, 11), (2, 19, 20, 5), False),
            ((8, 21, 22, 11), (5, 20, 21, 8), False), ((8, 21, 22, 11), (8, 21, 22, 11), False),
            ((8, 21, 22, 11), (11, 22, 23, 14), False), ((8, 21, 22, 11), (14, 23, 18, 17), False),
            ((11, 22, 23, 14), (2, 19, 18, 17), False), ((11, 22, 23, 14), (2, 19, 20, 5), False),
            ((11, 22, 23, 14), (5, 20, 21, 8), False), ((11, 22, 23, 14), (8, 21, 22, 11), False),
            ((11, 22, 23, 14), (11, 22, 23, 14), False), ((11, 22, 23, 14), (14, 23, 18, 17), False),
            ((14, 23, 18, 17), (2, 19, 18, 17), False), ((14, 23, 18, 17), (2, 19, 20, 5), False),
            ((14, 23, 18, 17), (5, 20, 21, 8), False), ((14, 23, 18, 17), (8, 21, 22, 11), False),
            ((14, 23, 18, 17), (11, 22, 23, 14), False), ((14, 23, 18, 17), (14, 23, 18, 17), False)
        ])
