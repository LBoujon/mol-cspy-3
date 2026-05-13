from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from rdkit import Chem
from cspy.molbuilder.crossover.positions import crossover_positions
from cspy.molbuilder.crossover.crossover import Crossover


@patch('cspy.molbuilder.crossover.crossover.crossover_positions', Mock(return_value=True))
class TestCrossover(TestCase):

    def setUp(self):
        self.benzopyrene = Chem.MolFromSmiles('c1ccc2c(c1)c3cccc4c3c5c2cccc5cc4')

        self.crossover = Crossover(Mock(), Mock())
        self.mock_random = Mock()

    def test_crossover_of_two_benzopyrene_and_returns_two_benzopyrene(self):
        crossover_positions = [
            ((3, 4), (3, 4), False), ((6, 11, 12, 13), (6, 11, 12, 13), False),
            ((10, 11, 12, 17), (10, 11, 12, 17), False)
        ]

        for crossover_position in crossover_positions:
            self.mock_random.side_effect = [crossover_position, False, False]
            with patch('cspy.molbuilder.addition.addition.choice', self.mock_random), patch('cspy.molbuilder.crossover.crossover.choice', self.mock_random):
                child_i, child_j = self.crossover.crossover(self.benzopyrene, self.benzopyrene)
                self.assertEqual(Chem.MolToInchi(child_i, options='-SNon'), 'InChI=1S/C20H12/c1-2-8-16-15(7-1)17-9-3-5-13-11-12-14-6-4-10-18(16)20(14)19(13)17/h1-12H')
                self.assertEqual(Chem.MolToInchi(child_j, options='-SNon'), 'InChI=1S/C20H12/c1-2-8-16-15(7-1)17-9-3-5-13-11-12-14-6-4-10-18(16)20(14)19(13)17/h1-12H')


class TestCrossoverPositions(TestCase):

    def setUp(self):
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.benzopyrene = Chem.MolFromSmiles('c1ccc2c(c1)c3cccc4c3c5c2cccc5cc4')

    def test_crossover_positions_for_benzopyrene_and_benzopyrene_with_min_and_max_size_of_5(self):
        positions = crossover_positions(self.benzopyrene, self.benzopyrene, 5, 5)
        self.assertEqual(positions, [
            ((3, 4), (3, 4), False), ((6, 11, 12, 13), (6, 11, 12, 13), False),
            ((10, 11, 12, 17), (10, 11, 12, 17), False)
        ])

    def test_crossover_positions_for_benzopyrene_and_benzopyrene_with_min_and_max_size_of_4_and_6(self):
        positions = crossover_positions(self.benzopyrene, self.benzopyrene, 4, 6)
        self.assertEqual(positions, [
            ((3, 4), (3, 4), False), ((6, 11, 12, 13), (6, 11, 12, 13), False),
            ((6, 11, 12, 13), (10, 11, 12, 17), False), ((10, 11, 12, 17), (6, 11, 12, 13), False),
            ((10, 11, 12, 17), (10, 11, 12, 17), False)
        ])
