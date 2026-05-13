import unittest
from unittest import TestCase
from cspy.sample.mc.thre_interval import Interval_Fixed, Interval_Max_Ratio
from cspy.sample.mc.thre_increase import Increase_Fixed
import numpy as np

class TestIncreaseFunc(TestCase):
    INCREASE_PARA = [["fixed", "5.0"], ["fixed", "2.5"], ["fixed", "50.0"]]

    def test_increase_fixed(self):
        self.assertAlmostEqual(Increase_Fixed(self.INCREASE_PARA[0]).increase_energy, 5.0)
        self.assertAlmostEqual(Increase_Fixed(self.INCREASE_PARA[1]).increase_energy, 2.5)
        self.assertAlmostEqual(Increase_Fixed(self.INCREASE_PARA[2]).increase_energy, 50.0)

class TestIntervalFunc(TestCase):
    INTERVAL_FIXED_PARA = ["fixed", "2"]
    INTERVAL_MAX_RATIO_PARA = ["max_ratio", "sqrt", "2", "3", "2", "5.0", "0.01"]
    INITIAL_ENERGY = -50.0
    INCREASE_ENERGY = 5.0
    UPDATE_ENERGY_1 = -46.4
    UPDATE_ENERGY_2 = -36.2
    
    def test_interval_fixed(self):
        obj = Interval_Fixed(
            self.INTERVAL_FIXED_PARA,
            self.INITIAL_ENERGY, 
            self.INCREASE_ENERGY,
        )
        self.assertAlmostEqual(obj.lid_energy, -45.0)
        self.assertEqual(obj.iteration_tmp, 0)

        valid, lift_lid = obj.update_step(self.UPDATE_ENERGY_1)
        self.assertTrue(valid)
        self.assertFalse(lift_lid)
        self.assertEqual(obj.iteration_tmp, 1)
        self.assertAlmostEqual(obj.lid_energy, -45.0)

        valid, lift_lid = obj.update_step(self.UPDATE_ENERGY_2)
        self.assertFalse(valid)
        self.assertTrue(lift_lid)
        self.assertEqual(obj.iteration_tmp, 0)
        self.assertAlmostEqual(obj.lid_energy, -45.0)

        obj.update_lid_energy(self.INCREASE_ENERGY)
        self.assertAlmostEqual(obj.lid_energy, -40.0)

    def test_interval_max_ratio(self):
        obj = Interval_Max_Ratio(
            self.INTERVAL_MAX_RATIO_PARA, 
            self.INITIAL_ENERGY, 
            self.INCREASE_ENERGY,
        )
        self.assertAlmostEqual(obj.lid_energy, -45.0)
        self.assertEqual(obj.iteration_tmp, 0)
        self.assertEqual(obj.num_max, 0)
        self.assertEqual(obj.max_ratio, [])
        self.assertAlmostEqual(obj.energy_bin, 0.4472135955)

        valid, lift_lid = obj.update_step(self.UPDATE_ENERGY_1)
        self.assertTrue(valid)
        self.assertFalse(lift_lid)
        self.assertEqual(obj.num_max, 0)
        self.assertEqual(obj.max_ratio, [0.0])
        self.assertEqual(obj.iteration_tmp, 1)
        self.assertAlmostEqual(obj.lid_energy, -45.0)
        self.assertAlmostEqual(obj.energy_bin, 0.4472135955)

        valid, lift_lid = obj.update_step(self.UPDATE_ENERGY_2)
        self.assertFalse(valid)
        self.assertTrue(lift_lid)
        self.assertEqual(obj.num_max, 0)
        self.assertEqual(obj.max_ratio, [])
        self.assertEqual(obj.iteration_tmp, 0)
        self.assertAlmostEqual(obj.lid_energy, -45.0)
        self.assertAlmostEqual(obj.energy_bin, 0.4472135955)

        obj.update_lid_energy(self.INCREASE_ENERGY)
        self.assertAlmostEqual(obj.lid_energy, -40.0)
        self.assertAlmostEqual(obj.energy_bin, 0.6324555320)
