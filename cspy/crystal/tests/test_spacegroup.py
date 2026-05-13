from unittest import TestCase
from cspy.crystal.symmetry_operation import SymmetryOperation
from cspy.crystal.space_group import SpaceGroup
import numpy as np


TEST_SPACEGROUPS_REDUCED = {
    33: (-1, ("x,y,z", "1/2-x,1/2+y,1/2+z", "1/2+x,1/2-y,z", "-x,-y,1/2+z")),
    14: (1, ("x,y,z", "-x,1/2+y,1/2-z")),
}


class TestSpaceGroup(TestCase):
    def test_initialization_from_reduced_symops(self):
        for n, (latt, symops) in TEST_SPACEGROUPS_REDUCED.items():
            sg_symops = [SymmetryOperation.from_string_code(x) for x in symops]
            sg = SpaceGroup.from_symmetry_operations(sg_symops, expand_latt=latt)
            self.assertEqual(sg.international_tables_number, n)

    def test_apply_all_symops(self):
        sg = SpaceGroup(33)
        asym = np.array(
            [
                [0.165100, 0.285800, 0.170900],
                [0.089400, 0.376200, 0.348100],
                [0.182000, 0.051000, -0.116000],
                [0.128000, 0.510000, 0.491000],
                [0.033000, 0.540000, 0.279000],
                [0.053000, 0.168000, 0.421000],
                [0.128700, 0.107500, 0.000000],
                [0.252900, 0.370300, 0.176900],
            ]
        )
        sym, transformed = sg.apply_all_symops(asym)
        np.testing.assert_equal(asym, transformed[: len(asym)])
        # assert that the unity symmetry operation is the first set of atoms
        np.testing.assert_equal(sym[: len(asym)], 16484)
        self.assertEqual(len(transformed), len(sg) * len(asym))
