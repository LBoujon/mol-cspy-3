import unittest
import numpy as np
from cspy.crystal.symmetry_operation import (
    decode_symm_int,
    encode_symm_int,
    decode_symm_str,
    encode_symm_str,
    SymmetryOperation,
)


class TestSymmetryOperation(unittest.TestCase):
    def test_encode_int_self_consistent(self):
        values = (17140694, 3360, 16484, 18426648, 18439772)
        for v in values:
            self.assertEqual(encode_symm_int(*decode_symm_int(v)), v)

    def test_encode_str_self_consistent(self):
        values = ("x,y,z", "+x,+y,z", "X,y,Z", "x,+y,z")
        for v in values:
            self.assertEqual(encode_symm_str(*decode_symm_str(v)), "+x,+y,+z")

    def test_decode_str(self):
        values = {
            "+1/2+x,-y+6/12,-z": (np.diag([1, -1, -1]), np.array([0.5, 0.5, 0.0])),
            "-x+1/3,y-2/3,z+1/6": (
                np.diag([-1, 1, 1]),
                np.array([1 / 3, 1 / 3, 1 / 6]),
            ),
            "-X+1/3,+Y-5/3,z+2/12": (
                np.diag([-1, 1, 1]),
                np.array([1 / 3, 1 / 3, 1 / 6]),
            ),
        }
        for string, (rotation, translation) in values.items():
            r2, t2 = decode_symm_str(string)
            np.testing.assert_almost_equal(r2, rotation)
            np.testing.assert_almost_equal(t2, translation)

    def test_encode_str(self):
        values = {
            "1/2+x,1/2-y,-z": (np.diag([1, -1, -1]), np.array([0.5, 0.5, 0.0])),
            "1/3-x,1/3+y,1/6+z": (np.diag([-1, 1, 1]), np.array([1 / 3, 1 / 3, 1 / 6])),
        }
        for string, (rotation, translation) in values.items():
            s2 = encode_symm_str(rotation, translation)
            self.assertEqual(string, s2)

    def test_encode_int(self):
        values = {
            16484: (np.eye(3), np.zeros(3)),
            18439608: (np.diag([1, -1, -1]), np.array([0.5, 0.5, 0])),
            208697: (
                np.reshape([0, 1, 0, -1, 1, 0, 0, 0, 1], (3, 3)),
                np.array([0, 0, 10 / 12]),
            ),
        }
        for c1, (rotation, translation) in values.items():
            c2 = encode_symm_int(rotation, translation)
            self.assertEqual(c1, c2)

    def test_decode_int(self):
        values = {
            16484: (np.eye(3), np.zeros(3)),
            18439608: (np.diag([1, -1, -1]), np.array([0.5, 0.5, 0])),
            208697: (
                np.reshape([0, 1, 0, -1, 1, 0, 0, 0, 1], (3, 3)),
                np.array([0, 0, 10 / 12]),
            ),
        }
        for c1, (rotation, translation) in values.items():
            r2, t2 = decode_symm_int(c1)
            np.testing.assert_almost_equal(r2, rotation)
            np.testing.assert_almost_equal(t2, translation)

    def test_apply_symop(self):
        p21_c_symop2 = SymmetryOperation.from_string_code("-x,y+1/2,-z+1/2")
        pts = np.array([[0.5, 0.5, 0], [0, 0.5, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]])
        expected = np.array(
            [[-0.5, 1.0, 0.5], [0.0, 1.0, 0.5], [-0.5, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        transformed = p21_c_symop2(pts)
        np.testing.assert_almost_equal(expected, transformed)
