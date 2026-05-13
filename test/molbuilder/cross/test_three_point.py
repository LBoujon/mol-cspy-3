from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from rdkit import Chem
from cspy.molbuilder.crossover.positions import crossover_positions
from cspy.molbuilder.crossover.crossover import Crossover


@patch('cspy.molbuilder.crossover.crossover.crossover_positions', Mock(return_value=True))
class TestCrossover(TestCase):

    def setUp(self):
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')
        self.benzo_de_naphtho_1_8_gh_quinoline = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ncc5')

        self.crossover = Crossover(Mock(), Mock())
        self.mock_random = Mock()

    def test_crossover_of_two_perylene_and_returns_two_perylene_with_mocked_random_for_all_possibilities(self):
        self.mock_random.side_effect = [((6, 7, 8), (6, 7, 8), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.perylene, self.perylene)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C20H12/c1-5-13-6-2-11-17-18-12-4-8-14-7-3-10-16(20(14)18)15(9-1)19(13)17/h1-12H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C20H12/c1-5-13-6-2-11-17-18-12-4-8-14-7-3-10-16(20(14)18)15(9-1)19(13)17/h1-12H')

    def test_crossover_of_perylene_and_benzo_de_naphtho_1_8_gh_quinoline_returns_benzo_de_naphtho_1_8_gh_quinoline_and_perylene(self):
        self.mock_random.side_effect = [((6, 7, 8), (6, 7, 8), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.perylene, self.benzo_de_naphtho_1_8_gh_quinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C19H11N/c1-4-12-5-3-9-16-17(12)14(7-1)15-8-2-6-13-10-11-20-19(16)18(13)15/h1-11H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C20H12/c1-5-13-6-2-11-17-18-12-4-8-14-7-3-10-16(20(14)18)15(9-1)19(13)17/h1-12H')

    def test_crossover_of_two_benzo_de_naphtho_1_8_gh_quinoline_returns_two_benzo_de_naphtho_1_8_gh_quinoline(self):
        self.mock_random.side_effect = [((6, 7, 8), (6, 7, 8), False), False, False]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.benzo_de_naphtho_1_8_gh_quinoline, self.benzo_de_naphtho_1_8_gh_quinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C19H11N/c1-4-12-5-3-9-16-17(12)14(7-1)15-8-2-6-13-10-11-20-19(16)18(13)15/h1-11H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C19H11N/c1-4-12-5-3-9-16-17(12)14(7-1)15-8-2-6-13-10-11-20-19(16)18(13)15/h1-11H')

    def test_crossover_of_two_benzo_de_naphtho_1_8_gh_quinoline_returns_perylene_and_diazaperylene(self):
        self.mock_random.side_effect = [((6, 7, 8), (10, 15, 16), False), False, True]
        with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
            child_i, child_j = self.crossover.crossover(self.benzo_de_naphtho_1_8_gh_quinoline, self.benzo_de_naphtho_1_8_gh_quinoline)
            self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C20H12/c1-5-13-6-2-11-17-18-12-4-8-14-7-3-10-16(20(14)18)15(9-1)19(13)17/h1-12H')
            self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C18H10N2/c1-3-11-7-9-19-17-15(11)13(5-1)14-6-2-4-12-8-10-20-18(17)16(12)14/h1-10H')


class TestCrossoverPositions(TestCase):

    def setUp(self):
        self.perylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)c4cccc5c4c3ccc5')

    def test_crossover_positions_for_perylene_with_min_and_max_size_of_5(self):
        positions = crossover_positions(self.perylene, self.perylene, 5, 5)
        self.assertEqual(positions, [
            ((6, 7, 8), (6, 7, 8), False), ((6, 7, 8), (10, 15, 16), False), ((10, 15, 16), (6, 7, 8), False),
            ((10, 15, 16), (10, 15, 16), False)
        ])

    def test_crossover_positions_for_perylene_with_min_and_max_size_of_4_and_6(self):
        positions = crossover_positions(self.perylene, self.perylene, 4, 6)
        self.assertEqual(positions, [
            ((6, 7, 8), (6, 7, 8), False), ((6, 7, 8), (10, 15, 16), False), ((10, 15, 16), (6, 7, 8), False),
            ((10, 15, 16), (10, 15, 16), False)
        ])
