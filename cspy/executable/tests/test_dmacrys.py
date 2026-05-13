from cspy.executable import Dmacrys
from os.path import dirname, join
import logging
import sys
import unittest
import pytest
import numpy as np
from cspy.formats import DmacrysSummary
from cspy.formats.shelx import parse_shelx_file_content
from cspy.crystal.space_group import SpaceGroup


if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)


class DmacrysTestCase(unittest.TestCase):

    def run_dmacrys(self, dmain, symmetry_file):
        with open(dmain) as f:
            dmain_contents = f.read()
        with open(symmetry_file) as f:
            symmetry_contents = f.read()

        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            exe = Dmacrys(
                dmain_contents,
                symmetry_contents,
                name="test",
                working_directory=tmpdirname,
                timeout=1800,
                adjust_input=False
            )
            exe.run()
            return exe

    @pytest.mark.external_binaries
    def test_dmacrys(self):
        DMAIN_FILE = join(dirname(__file__), "all.dmain")
        SYMMETRY_FILE = join(dirname(__file__), "all_symmetry_file")
        output = self.run_dmacrys(DMAIN_FILE, SYMMETRY_FILE).result()
        assert output is not None

    @pytest.mark.external_binaries
    def test_dmacrys_16_1(self):
        DMAIN_FILE = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/reason/PAPTUX_CO1.dmain"
        )
        SYMMETRY_FILE = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/reason/fort.20"
        )
        output = self.run_dmacrys(DMAIN_FILE, SYMMETRY_FILE)
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-134.1598, summary.initial_energy)
        self.assertEqual(-134.1608, summary.final_energy)
        self.assertEqual(2341.8328, summary.initial_volume)
        self.assertEqual(1.3062, summary.initial_density)
        self.assertEqual(2343.7655, summary.final_volume)
        self.assertEqual(1.3051, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-19.1140, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-1.1124, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-21.0849, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(24.8682, a)
        self.assertEqual(17.0814, b)
        self.assertEqual(5.5176, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "O1", "O2", "O3", "H1", "H2",
             "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11",
             "H12", "H13", "H14"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.35923214, 0.34898461, 0.25553648],
            [0.37429842, 0.39361492, 0.04932947],
            [0.33523769, 0.41711617, -0.10750312],
            [0.23892299, 0.42533209, -0.22544358],
            [0.18653103, 0.41244264, -0.17360657],
            [0.17173594, 0.37050810, 0.03898701],
            [0.21025544, 0.34270406, 0.19132295],
            [0.30685015, 0.33163740, 0.30031231],
            [0.43326575, 0.41236912, 0.00226327],
            [0.45818894, 0.45018567, 0.22398075],
            [0.46539731, 0.34040121, -0.07784483],
            [0.09917343, 0.31682191, 0.26236181],
            [0.28032936, 0.39944664, -0.06632466],
            [0.26553971, 0.35734001, 0.14192156],
            [0.48734097, 0.41861397, 0.36412647],
            [0.44319545, 0.52432926, 0.24989850],
            [0.11831211, 0.36135134, 0.06761637],
            [0.38982479, 0.32945048, 0.37950397],
            [0.34565458, 0.45077227, -0.26575373],
            [0.24977027, 0.45699626, -0.38684409],
            [0.15491882, 0.43256160, -0.29076918],
            [0.20010613, 0.31080012, 0.35248996],
            [0.29616820, 0.29854897, 0.45917707],
            [0.43468751, 0.45615390, -0.13998534],
            [0.50673929, 0.35593815, -0.11799615],
            [0.44738166, 0.31566276, -0.23950400],
            [0.46587757, 0.29601693, 0.06218406],
            [0.05638657, 0.31043852, 0.23494904],
            [0.11734071, 0.25905563, 0.26518566],
            [0.10664707, 0.34577495, 0.43424319],
            [0.45862235, 0.54399352, 0.39475390]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=9
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(61, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_dmacrys_16_2(self):
        DMAIN_FILE = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/final_minimization/PAPTUX_CO1_Z2.dmain"
        )
        SYMMETRY_FILE = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/final_minimization/fort.20"
        )
        output = self.run_dmacrys(DMAIN_FILE, SYMMETRY_FILE)
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-134.7351, summary.initial_energy)
        self.assertEqual(-134.7351, summary.final_energy)
        self.assertEqual(2338.1172, summary.initial_volume)
        self.assertEqual(1.3082, summary.initial_density)
        self.assertEqual(2338.1247, summary.final_volume)
        self.assertEqual(1.3082, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-19.0455, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-1.6815, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-20.6591, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(24.9232, a)
        self.assertEqual(17.0383, b)
        self.assertEqual(5.5060, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
             "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
             "C27", "C28", "O1", "O2", "O3", "O4", "O5", "O6", "H1", "H2",
             "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11",
             "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20",
             "H21", "H22", "H23", "H24", "H25", "H26", "H27", "H28"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.37558831, 0.34902719, 0.26334084],
            [0.39109635, 0.39342324, 0.05670310],
            [0.35247533, 0.41682111, -0.10246697],
            [0.25663298, 0.42506937, -0.22489265],
            [0.20423788, 0.41233180, -0.17500383],
            [0.18898963, 0.37064754, 0.03800566],
            [0.22707590, 0.34293026, 0.19270977],
            [0.32321364, 0.33180981, 0.30621979],
            [0.45004769, 0.41203448, 0.01179042],
            [0.47448642, 0.45023634, 0.23447428],
            [0.48220504, 0.33969853, -0.06605124],
            [0.11607014, 0.31731416, 0.25951737],
            [0.29758725, 0.39928013, -0.06329619],
            [0.28235338, 0.35741667, 0.14535979],
            [0.84296746, 0.15138658, 0.74789080],
            [0.85742208, 0.10659211, 0.95526870],
            [0.81800784, 0.08295065, 1.11004950],
            [0.72157702, 0.07459786, 1.22253538],
            [0.66944958, 0.08749264, 1.16754857],
            [0.65528501, 0.12958673, 0.95377924],
            [0.69414459, 0.15754002, 0.80348498],
            [0.79082784, 0.16874820, 0.69999312],
            [0.91612589, 0.08782469, 1.00584479],
            [0.94164648, 0.05000195, 0.78503565],
            [0.94792419, 0.15997605, 1.08821734],
            [0.58351190, 0.18341778, 0.72584195],
            [0.76333717, 0.10063219, 1.06560871],
            [0.74916542, 0.14289747, 0.85619633],
            [0.50324824, 0.41874196, 0.37661740],
            [0.45954426, 0.52463603, 0.25871662],
            [0.13561728, 0.36161852, 0.06454048],
            [0.97112342, 0.08172089, 0.64641601],
            [0.92679152, -0.02433322, 0.75796486],
            [0.60206262, 0.13873136, 0.92197514],
            [0.40583310, 0.32957294, 0.38915793],
            [0.36323325, 0.45029823, -0.26108848],
            [0.26782500, 0.45654386, -0.38662287],
            [0.17296212, 0.43238484, -0.29405433],
            [0.21658054, 0.31121365, 0.35423941],
            [0.31219051, 0.29890325, 0.46543702],
            [0.45180701, 0.45570827, -0.13132988],
            [0.52355313, 0.35513026, -0.10474235],
            [0.46454502, 0.31468595, -0.22844985],
            [0.48234752, 0.29541652, 0.07493502],
            [0.07343100, 0.31095824, 0.23031021],
            [0.13413454, 0.25937068, 0.26397132],
            [0.12319456, 0.34658868, 0.43164600],
            [0.47465169, 0.54454107, 0.40424062],
            [0.87383980, 0.17103374, 0.62553132],
            [0.82795772, 0.04916912, 1.26913303],
            [0.73194642, 0.04281267, 1.38480637],
            [0.63757987, 0.06726044, 1.28302773],
            [0.68447079, 0.18956609, 0.64148975],
            [0.78061551, 0.20196060, 0.54027692],
            [0.91715144, 0.04388643, 1.14833485],
            [0.98906173, 0.14442159, 1.13083682],
            [0.92947373, 0.18471197, 1.24921507],
            [0.94878967, 0.20451620, 0.94806517],
            [0.54074153, 0.18977343, 0.75081077],
            [0.60162277, 0.24134581, 0.72425173],
            [0.59147567, 0.15445107, 0.55396859],
            [0.94260869, -0.04398949, 0.61366737]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=9
        )
        # no space group test as cspy does not support non standard
        # space groups

    @pytest.mark.external_binaries
    def test_dmacrys_18(self):
        DMAIN_FILE = join(
            dirname(__file__),
            "test_suite/18.defaults_AXOSOW/DMACRYS_input/AXOSOW.dmain"
        )
        SYMMETRY_FILE = join(
            dirname(__file__),
            "test_suite/18.defaults_AXOSOW/DMACRYS_input/fort.20"
        )
        output = self.run_dmacrys(DMAIN_FILE, SYMMETRY_FILE)
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-44.6224, summary.initial_energy)
        self.assertEqual(-44.8841, summary.final_energy)
        self.assertEqual(700.0617, summary.initial_volume)
        self.assertEqual(1.0639, summary.initial_density)
        self.assertEqual(694.9194, summary.final_volume)
        self.assertEqual(1.0717, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-11.5855, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(1.4023, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-10.3241, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(9.3410, a)
        self.assertEqual(10.6476, b)
        self.assertEqual(6.9870, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "O1", "H1", "H2", "H3", "H4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.79923409, -0.00108079, 0.12579764],
            [0.82633126, 0.11901967, 0.08061070],
            [0.95951738, 0.17981543, 0.14465412],
            [0.99127588, 0.28904846, 0.10868870],
            [0.87461708, -0.05508524, 0.21003549],
            [0.70224437, -0.04835839, 0.08083767],
            [0.75295352, 0.17526649, -0.00338363],
            [1.02988049, 0.12033942, 0.22809104],
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=9
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(61, space_group.international_tables_number)
