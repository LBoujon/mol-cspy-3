import unittest
from unittest import TestCase
from unittest.mock import Mock, patch
from cspy.sample.mc.mc_change import calc_beta, mc_accept, get_shift, MC
from cspy.crystal import Crystal
from os.path import dirname, join
import numpy as np

class TestMCFunc(TestCase):
    def setUp(self):
        self.mock_random = Mock()

    def test_calc_beta(self):
        self.assertAlmostEqual(calc_beta(0), 100000)
        self.assertAlmostEqual(calc_beta(100), 0.001202723944)
        self.assertAlmostEqual(calc_beta(500), 0.000240544791)

    def test_mc_accept(self):
        self.mock_random.side_effect = [0.3, 0.5]
        with patch('cspy.sample.mc.mc_change.default_rng') as MockClass:
            instance = MockClass()
            instance.random = self.mock_random
            assert mc_accept(-6, -5, 0.001) == (True, True)
            assert mc_accept(-5, -6, 0.001) == (True, False)
            assert mc_accept(-5, -6, 0.001) == (False, False)

    def test_get_shift(self):
        self.assertAlmostEqual(get_shift(80.0, 90.0, 135.0, 45.0), 0.222222222222)
        self.assertAlmostEqual(get_shift(110.0, 90.0, 135.0, 45.0), -0.4444444444444)


class TestMCMove(TestCase):
    MOVE_LIST = [
        ("tra", 0.2, 0.1),
        ("rot", 0.2, 0.1),
        ("u_l", 0.2, 0.1),
        ("u_a", 0.2, 0.1),
        ("vol", 0.1, 10.0),
        ("ref", 0.1, 0.0),
    ]
    
    with open(join(dirname(__file__), "ACEMID07.res"), 'r') as f:
        ACEMID_RES = f.read()

    SG = 14

    def setUp(self):
        self.mock_random = Mock()
        self.mock_randint = Mock()
        self.mc = MC(self.MOVE_LIST, auto_prob=True, auto_cutoff=True)
        self.mc.rng = Mock()
        self.mc.update_sg(self.ACEMID_RES)

    def test_get_moves(self):
        mc_tmp = MC(self.MOVE_LIST)
        moves_refer = [
            ['tra', [0.2, 0.1]],
            ['rot', [0.4, 0.1]],
            ['u_l', [0.6, 0.1]],
            ['u_a', [0.8, 0.1]],
            ['vol', [0.9, 10.0]],
            ['ref', [1.0, 0.0]],
        ]
        for i_move in moves_refer:
            print(i_move)
            move_type, move_params = i_move
            self.assertTrue(move_type in mc_tmp.moves)
            np.testing.assert_almost_equal(mc_tmp.moves[move_type].cutoff, move_params[1])


        mc_tmp = MC(self.MOVE_LIST, cut_off_scale=1.2)
        moves_refer = [
            ['tra', [0.2, 0.12]],
            ['rot', [0.4, 0.12]],
            ['u_l', [0.6, 0.12]],
            ['u_a', [0.8, 0.12]],
            ['vol', [0.9, 12.0]],
            ['ref', [1.0, 0.0]],
        ]
        for i_move in moves_refer:
            move_type, move_params = i_move
            self.assertTrue(move_type in mc_tmp.moves)
            np.testing.assert_almost_equal(mc_tmp.moves[move_type].cutoff, move_params[1])

    def test_update_sg(self):
        self.assertEqual(self.mc.sg, self.SG)
        self.assertEqual(self.mc.index, 2)
        self.assertEqual(self.mc.z, 8)
        self.assertEqual(self.mc.num_mol, 2)
        moves_refer = [
            ['tra', [0.3333333333, 0.1]],
            ['rot', [0.6666666666, 0.1]],
            ['u_l', [0.8333333333, 0.1]],
            ['u_a', [0.8888888888, 0.1]],
            ['vol', [0.9444444444, 80.0]],
            ['ref', [1.0, 0.0]],
        ]
        for i_move in moves_refer:
            move_type, move_params = i_move
            self.assertTrue(move_type in self.mc.moves)
            np.testing.assert_almost_equal(self.mc.moves[move_type].probability, move_params[0])
            np.testing.assert_almost_equal(self.mc.moves[move_type].cutoff, move_params[1])

    def test_choose_move(self):
        moves_refer = ['tra', 'rot', 'u_l', 'u_a', 'vol', 'ref']
        self.mock_random.side_effect = [0.2, 0.5, 0.8, 0.85, 0.9, 0.95]
        self.mc.rng.random = self.mock_random
        #with patch(self.mc.rng.random, self.mock_random): # as MockClass:
        #with patch('cspy.sample.mc.mc_change.random.random', self.mock_random):
        for move_refer in moves_refer:
            self.assertEqual(self.mc.choose_move(), move_refer)

    def test_translation(self):
        acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
        acemid_asymmetric_unit = acemid_crys.asym_mols()
        trans = (-0.007071067812,  0.01224744871, 0.014142135624)
        coord_0_refer = acemid_asymmetric_unit[0].positions
        coord_1_refer = acemid_asymmetric_unit[1].positions + trans
        lattice_parameter_refer = acemid_crys.unit_cell.parameters

        self.mock_randint.side_effect = [1]
        self.mock_random.side_effect = [0.25, 0.3333333333333, 0.2]
        self.mc.rng.random = self.mock_random
        self.mc.rng.integers = self.mock_randint
        # with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
        #      patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
        new_crystal, rand = self.mc.translation(acemid_crys)
        asym_mols = new_crystal.asym_mols()
        self.assertEqual(rand, 0.2)
        np.testing.assert_almost_equal(
        asym_mols[0].positions,
        coord_0_refer
        )
        np.testing.assert_almost_equal(
        asym_mols[1].positions,
        coord_1_refer
        )
        np.testing.assert_almost_equal(
        new_crystal.unit_cell.parameters,
        lattice_parameter_refer
        )

    def test_rotation(self):
        acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
        acemid_asymmetric_unit = acemid_crys.asym_mols()
        rotation_matrix = np.array([
            [0.989227298, -0.113281423, 0.09271824],
            [0.107950319, 0.992305213, 0.06063904],
            [-0.098874069, -0.049976831, 0.99384417]
        ])
        coord_0_refer = acemid_asymmetric_unit[0].positions
        coord_1_center = np.average(acemid_asymmetric_unit[1].positions, axis=0)
        coord_1_refer = acemid_asymmetric_unit[1].positions - coord_1_center
        coord_1_refer = np.dot(coord_1_refer, np.transpose(rotation_matrix))
        coord_1_refer += coord_1_center
        lattice_parameter_refer = acemid_crys.unit_cell.parameters

        self.mock_randint.side_effect = [1]
        self.mock_random.side_effect = [0.25, 0.3333333333333, 0.75]
        self.mc.rng.random = self.mock_random
        self.mc.rng.integers = self.mock_randint
        # with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
        #      patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
        new_crystal, rand = self.mc.rotation(acemid_crys)
        asym_mols = new_crystal.asym_mols()
        self.assertEqual(rand, 0.75)
        np.testing.assert_almost_equal(
        asym_mols[0].positions,
        coord_0_refer
        )
        np.testing.assert_almost_equal(
        asym_mols[1].positions,
        coord_1_refer
        )
        np.testing.assert_almost_equal(
        new_crystal.unit_cell.parameters,
        lattice_parameter_refer
        )

    def test_unit_cell_length(self):
        acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
        acemid_asymmetric_unit = acemid_crys.asym_mols()
        coord_0_refer = acemid_asymmetric_unit[0].positions
        coord_1_refer = acemid_asymmetric_unit[1].positions

        u_l_change = np.array([0, 0, 0.05, 0, 0, 0])
        lattice_parameter_refer = acemid_crys.unit_cell.parameters + u_l_change

        self.mock_randint.side_effect = [2]
        self.mock_random.side_effect = [0.75]
        self.mc.rng.random = self.mock_random
        self.mc.rng.integers = self.mock_randint
        # with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
        #      patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
        new_crystal, rand = self.mc.unit_length(acemid_crys)
        asym_mols = new_crystal.asym_mols()
        self.assertEqual(rand, 0.75)
        np.testing.assert_almost_equal(
        asym_mols[0].positions,
        coord_0_refer
        )
        np.testing.assert_almost_equal(
        asym_mols[1].positions,
        coord_1_refer
        )
        np.testing.assert_almost_equal(
        new_crystal.unit_cell.parameters,
        lattice_parameter_refer
        )

    def test_unit_cell_angle(self):
        acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
        acemid_asymmetric_unit = acemid_crys.asym_mols()
        coord_0_refer = acemid_asymmetric_unit[0].positions
        coord_1_refer = acemid_asymmetric_unit[1].positions

        u_a_change = np.array([0, 0, 0, 0, 0.017211111111, 0])
        lattice_parameter_refer = acemid_crys.unit_cell.parameters + u_a_change

        self.mock_random.side_effect = [0.75]
        self.mc.rng.random = self.mock_random
        # with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
        #      patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
        new_crystal, rand = self.mc.unit_angle(acemid_crys)
        asym_mols = new_crystal.asym_mols()
        self.assertEqual(rand, 0.75)
        np.testing.assert_almost_equal(
        asym_mols[0].positions,
        coord_0_refer
        )
        np.testing.assert_almost_equal(
        asym_mols[1].positions,
        coord_1_refer
        )
        np.testing.assert_almost_equal(
        new_crystal.unit_cell.parameters,
        lattice_parameter_refer
        )

    def test_volume(self):
        acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
        acemid_asymmetric_unit = acemid_crys.asym_mols()

        volume_change = np.array([1.02119340114, 1.02119340114, 1.02119340114, 1.0, 1.0, 1.0])

        coord_change = volume_change[:3] - np.array([1.0, 1.0, 1.0])

        coord_0_refer = acemid_asymmetric_unit[0].positions
        coord_0_center = np.average(acemid_asymmetric_unit[0].positions, axis=0)
        coord_0_center_frac = acemid_crys.to_fractional(coord_0_center)
        coord_0_move_frac = coord_0_center_frac * coord_change
        coord_0_move = acemid_crys.to_cartesian(coord_0_move_frac)
        coord_0_refer += coord_0_move

        coord_1_refer = acemid_asymmetric_unit[1].positions
        coord_1_center = np.average(acemid_asymmetric_unit[1].positions, axis=0)
        coord_1_center_frac = acemid_crys.to_fractional(coord_1_center)
        coord_1_move_frac = coord_1_center_frac * coord_change
        coord_1_move = acemid_crys.to_cartesian(coord_1_move_frac)
        coord_1_refer += coord_1_move

        lattice_parameter_refer = acemid_crys.unit_cell.parameters * volume_change

        self.mock_random.side_effect = [0.75]
        self.mc.rng.random = self.mock_random
        # with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
        #      patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
        new_crystal, rand = self.mc.volume(acemid_crys)
        asym_mols = new_crystal.asym_mols()
        self.assertEqual(rand, 0.75)
        np.testing.assert_almost_equal(
        asym_mols[0].positions,
        coord_0_refer
        )
        np.testing.assert_almost_equal(
        asym_mols[1].positions,
        coord_1_refer
        )
        np.testing.assert_almost_equal(
        new_crystal.unit_cell.parameters,
        lattice_parameter_refer
        )

    #def test_reflect(self):
    #    acemid_crys = Crystal.from_shelx_string(self.ACEMID_RES)
    #    acemid_asymmetric_unit = acemid_crys.asym_mols()
    #    coord_0_refer = acemid_asymmetric_unit[0].positions
    #    coord_1_center = np.average(acemid_asymmetric_unit[1].positions, axis=0)
    #    coord_1_refer = acemid_asymmetric_unit[1].positions - coord_1_center
    #    for atom in coord_1_refer:
    #        atom[2] *= -1
    #    coord_1_refer += coord_1_center
    #    lattice_parameter_refer = acemid_crys.unit_cell.parameters

    #    self.mock_randint.side_effect = [1]
    #    with patch('cspy.sample.mc.mc_change.random.random', self.mock_random),\
    #         patch('cspy.sample.mc.mc_change.random.randint', self.mock_randint):
    #         new_crystal, im = self.mc.reflect(acemid_crys)
    #         asym_mols = new_crystal.asym_mols()
    #         self.assertEqual(im, 1)
    #         np.testing.assert_almost_equal(
    #            asym_mols[0].positions,
    #            coord_0_refer
    #         )
    #         np.testing.assert_almost_equal(
    #            asym_mols[1].positions,
    #            coord_1_refer
    #         )
    #         np.testing.assert_almost_equal(
    #            new_crystal.unit_cell.parameters,
    #            lattice_parameter_refer
    #         )
