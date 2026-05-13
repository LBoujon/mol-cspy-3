from unittest import TestCase
from cspy.crystal import Crystal
from os.path import abspath, join, dirname
from tempfile import TemporaryDirectory
import os

EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))
EXPERIMENTAL = join(EXAMPLES_DIR, "experimental_structures")
GENERATED = join(EXAMPLES_DIR, "generated_structures")

TCYETY_LABELS = (
    "C_F1_1____",
    "C_F1_2____",
    "C_F1_3____",
    "C_F1_4____",
    "C_F1_5____",
    "C_F1_6____",
    "N_F1_1____",
    "N_F1_2____",
    "N_F1_3____",
    "N_F1_4____",
)

TCYETY_PROJECTION_AXES = (
    (-0.762449, 0.417888, -0.494006),
    (0.123702, 0.843534, 0.52263),
    (0.635114, 0.337374, -0.694845),
)

TCYETY_PROJECTED_POSITIONS = (
    (0.358213, -0.582544, -0.000181),
    (1.784295, -0.582544, -0.000181),
    (-0.285639, -1.854960, -0.000181),
    (-0.358213, 0.582544, 0.000180),
    (-1.784295, 0.582544, 0.000180),
    (0.285638, 1.854960, 0.000180),
    (2.938602, -0.604767, -0.000271),
    (-0.786893, -2.894928, -0.000411),
    (-2.938602, 0.604766, 0.000272),
    (0.786893, 2.894928, 0.000411),
)


class NeighcrysAxisTests(TestCase):

    tcyety_sg2 = None

    def load_tcyety_sg2(self):
        if self.tcyety_sg2 is not None:
            self.tcyety_sg2 = Crystal.load(join(GENERATED, "tcyety_sg2.res"))

    def test_tcyety_sg2_labels(self):
        self.load_tcyety_sg2()
