from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from cspy.molbuilder.population.generational import Generational


class TestTournamentSelection(TestCase):

    @patch.object(Generational, "__init__", lambda x, y, z: None)
    def setUp(self):
        Generational.__abstractmethods__ = frozenset()
        self.population = Generational(Mock(), Mock())
        self.population.calculated_properties = {'a': 10, 'b': 1, 'c': 1}
        self.population.tournament_win_rate = 0.75

        self.mock_random = Mock()

    def test_tournament_selection_for_maximize_and_an_random_number_of_0_returns_a(self):
        self.population.maximize = True
        self.mock_random.side_effect = [0]

        with patch('cspy.molbuilder.population.generational.uniform', self.mock_random):
            a = self.population.tournament_selection('a', 'b')
            self.assertEqual(a, 'a')

    def test_tournament_selection_for_maximize_and_an_random_number_of_1_returns_b(self):
        self.population.maximize = True
        self.mock_random.side_effect = [1]

        with patch('cspy.molbuilder.population.generational.uniform', self.mock_random):
            b = self.population.tournament_selection('a', 'b')
            self.assertEqual(b, 'b')

    def test_tournament_selection_for_minimize_and_an_random_number_of_0_returns_b(self):
        self.population.maximize = False
        self.mock_random.side_effect = [0]

        with patch('cspy.molbuilder.population.generational.uniform', self.mock_random):
            b = self.population.tournament_selection('a', 'b')
            self.assertEqual(b, 'b')

    def test_tournament_selection_for_minimize_and_an_random_number_of_1_returns_b(self):
        self.population.maximize = False
        self.mock_random.side_effect = [1]

        with patch('cspy.molbuilder.population.generational.uniform', self.mock_random):
            a = self.population.tournament_selection('a', 'b')
            self.assertEqual(a, 'a')

    def test_tournament_selection_b_and_c_called_the_choice_function_once_with_a_list_of_b_and_c(self):
        self.population.maximize = False
        self.mock_random.side_effect = [1]

        with patch('cspy.molbuilder.population.generational.choice', self.mock_random):
            self.population.tournament_selection('b', 'c')
            self.mock_random.assert_called_once_with(['b', 'c'])
