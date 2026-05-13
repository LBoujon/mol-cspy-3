from cspy.executable import Dmacrys
from cspy.executable import Neighcrys
from os.path import dirname, join, exists
from pathlib import Path
import logging
import sys
import unittest
import pytest
import numpy as np
import shutil
from cspy.formats import DmacrysSummary
from cspy.formats.shelx import parse_shelx_file_content
from cspy.crystal.space_group import SpaceGroup


if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)


class NeighcrysDmacrysTestCase(unittest.TestCase):

    def run_neighcrys_dmacrys(self, res, neighcrys_kwargs, dmacrys_kwargs):
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            tmp_res = shutil.copy(res, tmpdirname)
            assert exists(tmp_res)
            LOG.debug("Copied %s -> %s", res, tmp_res)
            res_path = Path(tmp_res)
            exe = Neighcrys(res_path.name, name=res_path.stem, **neighcrys_kwargs)

            pots = shutil.copy(exe.args["potential_filename"], tmpdirname)
            LOG.debug("Copied %s -> %s", exe.args["potential_filename"], pots)
            assert exists(pots)
            exe.args["potential_filename"] = Path(pots).name

            if "labels_filename" in neighcrys_kwargs:
                labels = shutil.copy(exe.args["labels_filename"], tmpdirname)
                LOG.debug("Copied %s -> %s", exe.args["labels_filename"], labels)
                assert exists(labels)
                exe.args["labels_filename"] = Path(labels).name

            if "polarizabilities_filename" in neighcrys_kwargs:
                polarize = shutil.copy(exe.args["polarizabilities_filename"], tmpdirname)
                LOG.debug("Copied %s -> %s", exe.args["polarizabilities_filename"], polarize)
                assert exists(polarize)
                exe.args["polarizabilities_filename"] = Path(polarize).name

            if "hessian_filename" in neighcrys_kwargs:
                hess = shutil.copy(exe.args["hessian_filename"], tmpdirname)
                LOG.debug("Copied %s -> %s", exe.args["hessian_filename"], hess)
                assert exists(hess)
                exe.args["hessian_filename"] = Path(hess).name

            if neighcrys_kwargs["paste_molecular_structure_filename"]:
                paste = shutil.copy(exe.args["paste_molecular_structure_filename"], tmpdirname)
                LOG.debug("Copied %s -> %s", exe.args["paste_molecular_structure_filename"], paste)
                assert exists(paste)
                exe.args["paste_molecular_structure_filename"] = Path(paste).name

            bond = shutil.copy(exe.args["bondlength_filename"], tmpdirname)
            LOG.debug("Copied %s -> %s", exe.args["bondlength_filename"], bond)
            assert exists(bond)
            exe.args["bondlength_filename"] = Path(bond).name

            axis = shutil.copy(exe.args["axis_filename"], tmpdirname)
            LOG.debug("Copied %s -> %s", exe.args["axis_filename"], axis)
            exe.args["axis_filename"] = Path(axis).name
            assert exists(axis)

            mults = shutil.copy(exe.args["multipole_filename"], tmpdirname)
            LOG.debug("Copied %s -> %s", exe.args["multipole_filename"], mults)
            exe.args["multipole_filename"] = Path(mults).name
            assert exists(mults)

            exe.working_directory = tmpdirname
            exe.run()
            exe = Dmacrys(
                exe.dmain_contents,
                exe.symmetry_contents,
                name="test",
                working_directory=tmpdirname,
                timeout=1800,
                adjust_cutm_nbur=False,
                **dmacrys_kwargs
            )
            exe.run()
            return exe

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_01(self):
        res = join(
            dirname(__file__),
            "test_suite/01.lem_fit_AXOSOW/NEIGHCRYS_input/AXOSOW_cb106.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/01.lem_fit_AXOSOW/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/01.lem_fit_AXOSOW/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/01.lem_fit_AXOSOW/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/01.lem_fit_AXOSOW/NEIGHCRYS_input/fit.pots"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-43.1566, summary.initial_energy)
        self.assertEqual(-43.1619, summary.final_energy)
        self.assertEqual(700.0617, summary.initial_volume)
        self.assertEqual(1.0639, summary.initial_density)
        self.assertEqual(700.2127, summary.final_volume)
        self.assertEqual(1.0636, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-14.4947, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(4.7078, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-9.0871, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(9.8857, a)
        self.assertEqual(10.0951, b)
        self.assertEqual(7.0163, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "O1", "H1", "H2", "H3", "H4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.78712332, 0.00856541, 0.11842367],
            [0.82602893, 0.13205763, 0.07405215],
            [0.95912390, 0.18197091, 0.13484899],
            [1.00126043, 0.29353891, 0.09940752],
            [0.85274431, -0.05619372, 0.19960310],
            [0.69014818, -0.03083133, 0.07590810],
            [0.76259538, 0.19892392, -0.00686581],
            [1.02070687, 0.11044322, 0.21708070]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(61, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_02(self):
        res = join(
            dirname(__file__),
            "test_suite/02.lem_will01_AXOSOW/NEIGHCRYS_input/AXOSOW_cb106.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/02.lem_will01_AXOSOW/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/02.lem_will01_AXOSOW/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/02.lem_will01_AXOSOW/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/02.lem_will01_AXOSOW/NEIGHCRYS_input/will01.pots"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance=": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "w",
            "foreshorten_hydrogens": True,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-44.6153, summary.initial_energy)
        self.assertEqual(-44.8553, summary.final_energy)
        self.assertEqual(700.0617, summary.initial_volume)
        self.assertEqual(1.0639, summary.initial_density)
        self.assertEqual(694.7546, summary.final_volume)
        self.assertEqual(1.0720, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-11.6110, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(1.4655, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-10.2693, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(9.3994, a)
        self.assertEqual(10.5737, b)
        self.assertEqual(6.9904, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "O1", "H1", "H2", "H3", "H4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.79793384, -0.00096073, 0.12488780],
            [0.82623882, 0.11974376, 0.07980709],
            [0.95932063, 0.17973103, 0.14377989],
            [0.99213502, 0.28944712, 0.10790370],
            [0.87224131, -0.05604774, 0.20897767],
            [0.70126223, -0.04753784, 0.08010391],
            [0.75399340, 0.17703574, -0.00398847],
            [1.03010326, 0.11784593, 0.22890433]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(61, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_03(self):
        res = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/SOXLEX_af395.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/03.lem_custom_spli_SOXLEX/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-84.6923, summary.initial_energy)
        self.assertEqual(-85.1785, summary.final_energy)
        self.assertEqual(760.8962, summary.initial_volume)
        self.assertEqual(2.5170, summary.initial_density)
        self.assertEqual(780.5834, summary.final_volume)
        self.assertEqual(2.4535, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(87.2387, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-76.1616, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-19.9111, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(3.8430, a)
        self.assertEqual(14.0048, b)
        self.assertEqual(14.5186, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(92.5989, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["Br1", "Br2", "Cl1", "F1", "C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.42684645, 0.59839474, 0.13233834],
            [1.01794197, 0.85697093, 0.40438501],
            [0.70784835, 0.64809183, 0.34270436],
            [0.76268049, 0.94813930, 0.06239314],
            [0.85448667, 0.82745653, 0.28267957],
            [0.72391640, 0.73618717, 0.26038101],
            [0.60697688, 0.71918614, 0.16876555],
            [0.61884393, 0.79008629, 0.10149031],
            [0.75000755, 0.87896541, 0.12699837],
            [0.86869728, 0.89936447, 0.21651232],
            [0.52843031, 0.77644717, 0.03120591],
            [0.96937743, 0.96932199, 0.23416050]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_04(self):
        res = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/SOXLEX_af395.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/04.lem_custom_nospli_SOXLEX/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "spline": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-85.8975, summary.initial_energy)
        self.assertEqual(-86.1616, summary.final_energy)
        self.assertEqual(760.8962, summary.initial_volume)
        self.assertEqual(2.5170, summary.initial_density)
        self.assertEqual(775.8528, summary.final_volume)
        self.assertEqual(2.4685, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(92.2682, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-81.0162, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-21.5697, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(3.8286, a)
        self.assertEqual(13.8841, b)
        self.assertEqual(14.6405, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(94.4913, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["Br1", "Br2", "Cl1", "F1", "C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.41219415, 0.59767311, 0.13029203],
            [1.01168768, 0.85875465, 0.40431476],
            [0.69096696, 0.64871492, 0.34142592],
            [0.77003368, 0.94841494, 0.06194507],
            [0.84947779, 0.82845923, 0.28216132],
            [0.71415090, 0.73670815, 0.25934624],
            [0.59845166, 0.71913981, 0.16741984],
            [0.61607057, 0.78995214, 0.10033838],
            [0.75177400, 0.87932110, 0.12635869],
            [0.86947177, 0.90028942, 0.21619554],
            [0.52657382, 0.77587545, 0.02981239],
            [0.97378438, 0.97061918, 0.23424110]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_05(self):
        res = join(
            dirname(__file__),
            "test_suite/05.largecell_KAXXAI/NEIGHCRYS_input/KAXXAI02.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/05.largecell_KAXXAI/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/05.largecell_KAXXAI/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/05.largecell_KAXXAI/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/05.largecell_KAXXAI/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": True,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-268.4195, summary.initial_energy)
        self.assertEqual(-276.3210, summary.final_energy)
        self.assertEqual(2419.1969, summary.initial_volume)
        self.assertEqual(1.4371, summary.initial_density)
        self.assertEqual(2490.2762, summary.final_volume)
        self.assertEqual(1.3961, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-47.9499, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-13.7040, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-25.2003, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.7417, a)
        self.assertEqual(11.5427, b)
        self.assertEqual(27.8747, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(91.2818, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
             "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
             "C27", "C28", "N1", "N2", "O1", "O2", "O3", "O4", "Cl1",
             "Cl2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
             "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17",
             "H18", "H19", "H20", "H21", "H22", "H23", "H24"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.41323647, 0.85568520, 0.03268595],
            [0.54412106, 0.93105048, 0.05571462],
            [0.71304939, 0.92944654, 0.03888062],
            [0.84004361, 1.00011244, 0.05858473],
            [0.79757923, 1.07319577, 0.09646170],
            [0.63326653, 1.07587104, 0.11417211],
            [0.50058979, 1.00580129, 0.09382377],
            [0.26797895, 1.09327498, 0.14146837],
            [0.29391826, 1.21012833, 0.13139327],
            [0.22574711, 1.29457515, 0.16042212],
            [0.12645335, 1.26360700, 0.19905751],
            [0.09633803, 1.14696863, 0.20757540],
            [0.16675808, 1.05883404, 0.18012555],
            [0.13722235, 0.93275235, 0.19031037],
            [0.05830988, 0.65261692, 0.96895321],
            [-0.06200665, 0.56655728, 0.94804198],
            [-0.23400577, 0.56574146, 0.96243010],
            [-0.35231129, 0.48815259, 0.94286586],
            [-0.29767765, 0.41045623, 0.90812368],
            [-0.12936869, 0.40875765, 0.89318196],
            [-0.00615107, 0.48633273, 0.91303796],
            [0.23913826, 0.41841103, 0.86361082],
            [0.23662100, 0.29804227, 0.86612071],
            [0.32271581, 0.23338045, 0.83229661],
            [0.41513995, 0.28734924, 0.79683958],
            [0.41849375, 0.40762698, 0.79525301],
            [0.32968452, 0.47594769, 0.82762307],
            [0.32844025, 0.60584745, 0.82550747],
            [0.33723435, 1.00721067, 0.11176427],
            [0.16179830, 0.48694819, 0.89883970],
            [0.46970707, 0.79286621, -0.00310822],
            [0.26124986, 0.85024350, 0.04552226],
            [0.20909964, 0.66378553, 0.95609038],
            [-0.00555018, 0.71875054, 1.00272089],
            [-0.03745716, 1.11236697, 0.25468628],
            [0.53896178, 0.47276292, 0.75072191],
            [0.36782935, 0.74902245, -0.01774441],
            [0.74489949, 0.87161673, 0.00978952],
            [0.96988866, 0.99890675, 0.04491966],
            [0.89604432, 1.12900092, 0.11217392],
            [0.60492908, 1.13222516, 0.14404663],
            [0.25926202, 0.94018983, 0.10283436],
            [0.36802140, 1.23479893, 0.10052020],
            [0.25004455, 1.38516456, 0.15303586],
            [0.07324523, 1.32919448, 0.22225589],
            [0.02510400, 0.90291516, 0.16992348],
            [0.11572011, 0.92106036, 0.22826119],
            [0.24946815, 0.88336041, 0.18006056],
            [0.09150166, 0.76899269, 1.01624197],
            [-0.27524874, 0.62689838, 0.98932216],
            [-0.48492913, 0.48783450, 0.95443373],
            [-0.38991848, 0.34994741, 0.89252645],
            [-0.09110045, 0.34742543, 0.86600469],
            [0.24048976, 0.54395277, 0.91593149],
            [0.16756638, 0.25499926, 0.89440027],
            [0.31758319, 0.13968259, 0.83370195],
            [0.48387153, 0.23670543, 0.77070518],
            [0.19831068, 0.63717239, 0.83067566],
            [0.37268337, 0.63400398, 0.79081588],
            [0.41336162, 0.64004957, 0.85347461]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_06(self):
        res = join(
            dirname(__file__),
            "test_suite/06.largecell_KAXXAI_nospli/NEIGHCRYS_input/KAXXAI02.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/06.largecell_KAXXAI_nospli/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/06.largecell_KAXXAI_nospli/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/06.largecell_KAXXAI_nospli/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/06.largecell_KAXXAI_nospli/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": True,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "spline": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-267.3524, summary.initial_energy)
        self.assertEqual(-275.3161, summary.final_energy)
        self.assertEqual(2419.1969, summary.initial_volume)
        self.assertEqual(1.4371, summary.initial_density)
        self.assertEqual(2494.4118, summary.final_volume)
        self.assertEqual(1.3937, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-47.8301, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-13.5833, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-25.0430, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.7493, a)
        self.assertEqual(11.5629, b)
        self.assertEqual(27.8446, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(91.2531, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
             "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
             "C27", "C28", "N1", "N2", "O1", "O2", "O3", "O4", "Cl1",
             "Cl2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
             "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17",
             "H18", "H19", "H20", "H21", "H22", "H23", "H24"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.41270224, 0.85580373, 0.03274713],
            [0.54352142, 0.93102930, 0.05575820],
            [0.71232327, 0.92923925, 0.03891609],
            [0.83925663, 0.99976377, 0.05860191],
            [0.79685640, 1.07289593, 0.09646958],
            [0.63266521, 1.07575541, 0.11418859],
            [0.50005742, 1.00582772, 0.09385717],
            [0.26769572, 1.09350255, 0.14148365],
            [0.29381997, 1.21009406, 0.13132328],
            [0.22577802, 1.29455488, 0.16032469],
            [0.12643703, 1.26386206, 0.19901621],
            [0.09614505, 1.14748224, 0.20761735],
            [0.16642396, 1.05934451, 0.18019913],
            [0.13669189, 0.93354384, 0.19047514],
            [0.05818955, 0.65285227, 0.96900177],
            [-0.06190986, 0.56684377, 0.94811304],
            [-0.23377775, 0.56604311, 0.96250634],
            [-0.35187605, 0.48849576, 0.94296102],
            [-0.29716480, 0.41082479, 0.90823237],
            [-0.12898070, 0.40911272, 0.89328618],
            [-0.00597586, 0.48664838, 0.91312387],
            [0.23923769, 0.41872421, 0.86370049],
            [0.23677928, 0.29857389, 0.86628594],
            [0.32291172, 0.23392465, 0.83247038],
            [0.41530954, 0.28769453, 0.79694855],
            [0.41860133, 0.40775741, 0.79528751],
            [0.32975900, 0.47605346, 0.82764509],
            [0.32845355, 0.60571873, 0.82544834],
            [0.33681997, 1.00742369, 0.11180739],
            [0.16184585, 0.48724883, 0.89892051],
            [0.46910613, 0.79291565, -0.00304146],
            [0.26082420, 0.85053265, 0.04559275],
            [0.20885945, 0.66398846, 0.95612779],
            [-0.00573052, 0.71897529, 1.00276133],
            [-0.03769024, 1.11322016, 0.25479395],
            [0.53903340, 0.47264908, 0.75067691],
            [0.36729472, 0.74916736, -0.01767007],
            [0.74412215, 0.87137546, 0.00983319],
            [0.96900630, 0.99841219, 0.04492981],
            [0.89527484, 1.12859136, 0.11216743],
            [0.60437125, 1.13214650, 0.14405688],
            [0.25883967, 0.94054308, 0.10290743],
            [0.36796586, 1.23454865, 0.10040515],
            [0.25021367, 1.38493985, 0.15287299],
            [0.07332791, 1.32946310, 0.22219396],
            [0.02468632, 0.90376314, 0.17008007],
            [0.11509852, 0.92203361, 0.22847288],
            [0.24877463, 0.88411623, 0.18025231],
            [0.09116492, 0.76919792, 1.01627233],
            [-0.27508235, 0.62717883, 0.98938738],
            [-0.48439411, 0.48819049, 0.95453320],
            [-0.38924316, 0.35034717, 0.89264954],
            [-0.09064709, 0.34780037, 0.86611955],
            [0.24038605, 0.54423069, 0.91600078],
            [0.16774109, 0.25569052, 0.89461746],
            [0.31782944, 0.14039422, 0.83393375],
            [0.48406839, 0.23706210, 0.77082130],
            [0.19842145, 0.63697930, 0.83059497],
            [0.37272898, 0.63371379, 0.79070545],
            [0.41320116, 0.63997744, 0.85342977]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_07(self):
        res = join(
            dirname(__file__),
            "test_suite/07.largecell_KAXXAI_30ang/NEIGHCRYS_input/KAXXAI02.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/07.largecell_KAXXAI_30ang/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/07.largecell_KAXXAI_30ang/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/07.largecell_KAXXAI_30ang/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/07.largecell_KAXXAI_30ang/NEIGHCRYS_input/pote.dat"
        )
        # using the extended mode as cspy does not support editing the
        # RDMA and CUTO in the dmain directly
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": True,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": True,
            "wcal_ngcv": False,
            "step_size": 0.5,
            "max_iterations": 1000,
            "vdw_cutoff": 30,
            "connectivity_search_max_cells": 3,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "spline": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-271.8905, summary.initial_energy)
        self.assertEqual(-279.7492, summary.final_energy)
        self.assertEqual(2419.1969, summary.initial_volume)
        self.assertEqual(1.4371, summary.initial_density)
        self.assertEqual(2489.6114, summary.final_volume)
        self.assertEqual(1.3964, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-47.8678, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-13.8497, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-25.6842, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.7436, a)
        self.assertEqual(11.5457, b)
        self.assertEqual(27.8533, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(91.2849, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
             "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
             "C27", "C28", "N1", "N2", "O1", "O2", "O3", "O4", "Cl1",
             "Cl2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
             "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17",
             "H18", "H19", "H20", "H21", "H22", "H23", "H24"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.41302497, 0.85576498, 0.03269994],
            [0.54388870, 0.93110069, 0.05574920],
            [0.71279391, 0.92944841, 0.03891897],
            [0.83976955, 1.00008422, 0.05864137],
            [0.79730881, 1.07318672, 0.09653374],
            [0.63301770, 1.07590975, 0.11424137],
            [0.50036184, 1.00587065, 0.09387362],
            [0.26779619, 1.09340760, 0.14152073],
            [0.29379390, 1.21021744, 0.13142538],
            [0.22564713, 1.29467916, 0.16045890],
            [0.12632175, 1.26376977, 0.19911816],
            [0.09615079, 1.14717454, 0.20765468],
            [0.16654255, 1.05902675, 0.18020206],
            [0.13694491, 0.93299181, 0.19040805],
            [0.05808572, 0.65268375, 0.96897465],
            [-0.06211115, 0.56659082, 0.94804521],
            [-0.23408569, 0.56572602, 0.96242619],
            [-0.35228016, 0.48810328, 0.94284396],
            [-0.29755864, 0.41042205, 0.90809053],
            [-0.12927079, 0.40877147, 0.89315545],
            [-0.00616572, 0.48638231, 0.91303019],
            [0.23917589, 0.41852861, 0.86359961],
            [0.23674928, 0.29819189, 0.86612586],
            [0.32291335, 0.23355326, 0.83229295],
            [0.41531549, 0.28751556, 0.79681204],
            [0.41857743, 0.40776219, 0.79520992],
            [0.32969848, 0.47605684, 0.82758694],
            [0.32835661, 0.60592107, 0.82545371],
            [0.33702693, 1.00732808, 0.11181177],
            [0.16175995, 0.48704555, 0.89883881],
            [0.46949112, 0.79292279, -0.00310800],
            [0.26105843, 0.85036679, 0.04553198],
            [0.20884614, 0.66389243, 0.95611661],
            [-0.00584995, 0.71880140, 1.00275327],
            [-0.03767847, 1.11264641, 0.25479302],
            [0.53901885, 0.47289217, 0.75064962],
            [0.36763298, 0.74910074, -0.01775962],
            [0.74464061, 0.87160447, 0.00981619],
            [0.96959762, 0.99884081, 0.04497864],
            [0.89575950, 1.12896861, 0.11226034],
            [0.60468128, 1.13227823, 0.14412870],
            [0.25905174, 0.94033427, 0.10287610],
            [0.36792351, 1.23484222, 0.10053274],
            [0.24998839, 1.38523427, 0.15305767],
            [0.07313194, 1.32936936, 0.22232064],
            [0.02486114, 0.90317046, 0.16999853],
            [0.11540214, 0.92133657, 0.22838730],
            [0.24915207, 0.88358099, 0.18016769],
            [0.09112412, 0.76907309, 1.01628881],
            [-0.27539772, 0.62687132, 0.98932695],
            [-0.48487983, 0.48774767, 0.95440665],
            [-0.38971247, 0.34988660, 0.89247902],
            [-0.09093232, 0.34744970, 0.86596898],
            [0.24036859, 0.54407405, 0.91594502],
            [0.16771109, 0.25515566, 0.89442495],
            [0.31785269, 0.13987887, 0.83371027],
            [0.48410059, 0.23689059, 0.77067116],
            [0.19822769, 0.63719676, 0.83060823],
            [0.37260824, 0.63406177, 0.79073688],
            [0.41319852, 0.64016259, 0.85344703]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.skip
    def test_neighcrys_dmacrys_08(self):
        # this test fails when using neighcrys 2.2.1
        res = join(
            dirname(__file__),
            "test_suite/08.2comp_KONTIQ/NEIGHCRYS_input/KONTIQ_A2312.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/08.2comp_KONTIQ/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/08.2comp_KONTIQ/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/08.2comp_KONTIQ/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/08.2comp_KONTIQ/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-182.8457, summary.initial_energy)
        self.assertEqual(-182.8622, summary.final_energy)
        self.assertEqual(3124.6643, summary.initial_volume)
        self.assertEqual(1.5997, summary.initial_density)
        self.assertEqual(3115.8722, summary.final_volume)
        self.assertEqual(1.6042, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-40.7019, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-42.3102, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-60.0451, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(25.9069, a)
        self.assertEqual(32.4064, b)
        self.assertEqual(3.7114 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "O1", "O2", "O3",
             "O4", "O5", "O6", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
             "H8"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.35440991, 0.69632763, 0.61586946],
            [0.35580154, 0.65373485, 0.56877453],
            [0.39903395, 0.63607205, 0.41550668],
            [0.44104986, 0.65997376, 0.30774636],
            [0.43925465, 0.70268839, 0.35621622],
            [0.39602262, 0.72054249, 0.50969388],
            [0.30959583, 0.71732916, 0.77747455],
            [0.40479290, 0.59476074, 0.35598213],
            [0.48304526, 0.64255555, 0.15857811],
            [0.48099689, 0.72445407, 0.24717359],
            [0.27133154, 0.69120198, 0.87131682],
            [0.30612182, 0.75403995, 0.82525132],
            [0.30759969, 0.55747715, 0.47655797],
            [0.37465366, 0.58022836, 0.43586993],
            [0.47728271, 0.61319981, 0.14716855],
            [0.47517917, 0.75320828, 0.29427583],
            [0.24440062, 0.70818688, 0.97281823],
            [0.32344047, 0.63483250, 0.65129814],
            [0.39389520, 0.75363298, 0.54957477],
            [0.29873850, 0.55489255, 0.22613383],
            [0.28248002, 0.57653989, 0.56927472]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(43, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_09(self):
        res = join(
            dirname(__file__),
            "test_suite/09.salt_FINVAZ/NEIGHCRYS_input/FINVAZ_A395.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/09.salt_FINVAZ/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/09.salt_FINVAZ/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/09.salt_FINVAZ/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/09.salt_FINVAZ/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "w",
            "foreshorten_hydrogens": True,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-626.1229, summary.initial_energy)
        self.assertEqual(-626.1234, summary.final_energy)
        self.assertEqual(2117.8138, summary.initial_volume)
        self.assertEqual(1.1774, summary.initial_density)
        self.assertEqual(2118.7804, summary.final_volume)
        self.assertEqual(1.1769, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-581.6984, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-14.7279, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-26.7762, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(23.2600, a)
        self.assertEqual(11.0730, b)
        self.assertEqual(9.8245 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(123.1401, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
             "C10", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8",
             "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16",
             "H17", "H18", "N1", "Cl1"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.10047845, 0.70862964, 0.15517766],
            [0.07648606, 0.82540504, 0.19132332],
            [0.11001128, 0.93150609, 0.15858174],
            [0.08766736, 0.92855275, -0.01992681],
            [0.11105471, 0.80974148, -0.05499760],
            [0.07753124, 0.70331688, -0.02282976],
            [0.17819999, 0.69502911, 0.26704234],
            [0.21145931, 0.80148088, 0.23410271],
            [0.18934937, 0.79822296, 0.05576924],
            [0.18831195, 0.92027368, 0.26984811],
            [0.01482686, 0.60817007, 0.11499817],
            [0.08120158, 0.60270729, 0.30611258],
            [0.08189152, 0.52221157, 0.16491379],
            [0.02045783, 0.83254934, 0.11357654],
            [0.09152047, 0.82670309, 0.31819168],
            [0.09309487, 1.01554340, 0.18418149],
            [0.03191758, 0.93810347, -0.10001155],
            [0.11039888, 1.00470932, -0.04488919],
            [0.09487737, 0.80660814, -0.18228364],
            [0.09330561, 0.61764092, -0.04850508],
            [0.02150948, 0.70967087, -0.10195017],
            [0.19389340, 0.69548480, 0.39439749],
            [0.19462733, 0.60930400, 0.24323987],
            [0.26714738, 0.79242709, 0.31375170],
            [0.20648036, 0.71435836, 0.02994323],
            [0.21359090, 0.87244011, 0.03193769],
            [0.21253473, 0.99630316, 0.24920773],
            [0.20469111, 0.92388203, 0.39745503],
            [0.06729198, 0.60309377, 0.18755795],
            [0.40731814, 1.12806953, 0.96096772]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(15, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_10(self):
        res = join(
            dirname(__file__),
            "test_suite/10.largesalt_WEMGEK/NEIGHCRYS_input/WEMGEK_59.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/10.largesalt_WEMGEK/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/10.largesalt_WEMGEK/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/10.largesalt_WEMGEK/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/10.largesalt_WEMGEK/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-762.4537, summary.initial_energy)
        self.assertEqual(-762.4587, summary.final_energy)
        self.assertEqual(785.0863, summary.initial_volume)
        self.assertEqual(1.3339, summary.initial_density)
        self.assertEqual(785.1735, summary.final_volume)
        self.assertEqual(1.3337, summary.final_density)
        self.assertEqual(2, summary.z)
        self.assertEqual(-580.8779, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-75.1771, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-67.6442, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(6.0404, a)
        self.assertEqual(7.1999, b)
        self.assertEqual(18.5784 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(103.6460, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
             "C10", "C11", "C12", "C13", "C14", "N1", "O1", "O2", "O3",
             "O4", "O5", "O6", "O7", "H1", "H2", "H3", "H4", "H5", "H6",
             "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15",
             "H16", "H17", "H18", "H19", "H20", "H21"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.20887579, 0.16281505, 0.27673733],
            [0.09597351, -0.01258726, 0.29881374],
            [0.15919272, -0.04596470, 0.38167039],
            [0.01133072, 0.00180324, 0.42559221],
            [0.07381888, -0.02848518, 0.50136755],
            [0.28337269, -0.10640691, 0.53353017],
            [0.43085731, -0.15537696, 0.48975043],
            [0.36846939, -0.12580848, 0.41406557],
            [0.13407529, 0.34484830, 0.30528070],
            [0.35418939, 0.22514670, 0.16081607],
            [0.77648887, 0.30328368, 0.95626168],
            [0.86048159, 0.18957167, 0.89926166],
            [1.11101178, 0.22406908, 0.90231353],
            [1.20151387, 0.10021415, 0.84684097],
            [0.15813860, 0.17052432, 0.19224255],
            [-0.13878582, 0.00648668, 0.26878525],
            [0.61215127, 0.40047225, 0.94383398],
            [0.89839728, 0.27683806, 1.02529600],
            [0.72854151, 0.23008145, 0.82898842],
            [1.13578839, 0.41617804, 0.89350407],
            [1.27681675, 0.18178097, 0.79987143],
            [1.18592742, -0.06719102, 0.85932831],
            [0.39175259, 0.14734929, 0.29482205],
            [0.16127371, -0.12964179, 0.27275433],
            [-0.15238762, 0.06023224, 0.40065531],
            [-0.04167902, 0.00812957, 0.53510726],
            [0.33103707, -0.13015448, 0.59228252],
            [0.59269577, -0.21774943, 0.51434210],
            [0.48291163, -0.16733545, 0.38039845],
            [-0.04791662, 0.36540810, 0.28549738],
            [0.17465252, 0.34266025, 0.36511878],
            [0.22259958, 0.46195611, 0.28820564],
            [0.42097075, 0.35609753, 0.18473091],
            [0.48329727, 0.11888718, 0.17457287],
            [0.29624431, 0.23809974, 0.10143057],
            [0.02639672, 0.25803634, 0.17427237],
            [0.10017207, 0.04590140, 0.17134882],
            [-0.20222745, -0.10182937, 0.24359875],
            [0.83030584, 0.34918953, 1.05677746],
            [0.84660595, 0.04398152, 0.91324255],
            [0.57344155, 0.21360203, 0.82846568],
            [1.20416052, 0.18143300, 0.95740886],
            [1.29174386, 0.43933177, 0.89766442]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(4, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_11(self):
        res = join(
            dirname(__file__),
            "test_suite/11.properties_CBMZPN/NEIGHCRYS_input/CBMZPN_opt_III.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/11.properties_CBMZPN/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/11.properties_CBMZPN/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/11.properties_CBMZPN/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/11.properties_CBMZPN/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "STAR": "PROP",
            "ACCM": 100000000,
            "NOPR": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-124.4958, summary.initial_energy)
        self.assertEqual(-124.5098, summary.final_energy)
        self.assertEqual(1184.9861, summary.initial_volume)
        self.assertEqual(1.3243, summary.initial_density)
        self.assertEqual(1180.8192, summary.final_volume)
        self.assertEqual(1.3290, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-21.2804, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(9.1164, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-18.7019, summary.lattice_energy_contributions["Higher multipole interaction energy"])
        np.testing.assert_almost_equal(np.array([
            0.0000, 0.0000, 0.0000, 42.3756, 45.5384, 47.4784, 50.1256,
            53.7867, 53.7961, 57.7942, 67.5939, 69.7281, 76.6065,
            82.4652, 84.4618, 86.8428, 91.6871, 100.1495, 102.2815,
            105.3979, 108.1949, 127.8800, 130.4624, 134.3770
        ]), summary.phonon_frequencies, decimal=5)

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.8724, a)
        self.assertEqual(11.0294, b)
        self.assertEqual(13.6106 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(92.3105, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
             "C10", "C11", "C12", "C13", "C14", "C15", "N1", "N2", "O1",
             "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
             "H10", "H11", "H12"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.17273677, 0.32510081, 0.34783344],
            [0.01714606, 0.38357641, 0.32981396],
            [-0.01901507, 0.43719323, 0.23877121],
            [0.10119392, 0.43354188, 0.16585724],
            [0.25621560, 0.37567683, 0.18417051],
            [0.29576861, 0.32092672, 0.27562910],
            [0.46011466, 0.26153794, 0.29175605],
            [0.55336813, 0.25313328, 0.37743943],
            [0.50977719, 0.30319000, 0.47238876],
            [0.63983844, 0.34476235, 0.53790701],
            [0.60297037, 0.39589449, 0.62805022],
            [0.43394625, 0.40714791, 0.65484238],
            [0.30290349, 0.36655979, 0.59134700],
            [0.34034863, 0.31351863, 0.50126824],
            [0.11251943, 0.16338073, 0.46227693],
            [0.20670303, 0.26586511, 0.43938763],
            [0.16102003, 0.11005555, 0.55135365],
            [-0.00721390, 0.12678122, 0.40985809],
            [-0.07505514, 0.38320619, 0.38662445],
            [-0.14045480, 0.48111721, 0.22442730],
            [0.07332217, 0.47494853, 0.09491815],
            [0.34931580, 0.37269465, 0.12765932],
            [0.51746467, 0.22700313, 0.22613273],
            [0.67884073, 0.21273949, 0.37460418],
            [0.77075206, 0.33683460, 0.51693297],
            [0.70520268, 0.42722348, 0.67723440],
            [0.40469896, 0.44706606, 0.72481989],
            [0.17126853, 0.37277754, 0.61089506],
            [0.28378283, 0.11809376, 0.57443862],
            [0.11232779, 0.02590406, 0.55691906]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_12(self):
        res = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/DCLBEN_beta_abi1.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/12.properties_DCLBEN/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "STAR": "PROP",
            "ACCM": 100000000,
            "NOPR": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-72.7683, summary.initial_energy)
        self.assertEqual(-72.7695, summary.final_energy)
        self.assertEqual(149.6047, summary.initial_volume)
        self.assertEqual(1.6316, summary.initial_density)
        self.assertEqual(149.5668, summary.final_volume)
        self.assertEqual(1.6321, summary.final_density)
        self.assertEqual(1, summary.z)
        self.assertEqual(12.7639, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-8.3179, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-11.0154, summary.lattice_energy_contributions["Higher multipole interaction energy"])
        np.testing.assert_almost_equal(np.array([
            0.0000, 0.0000, 0.0000, 49.1063, 66.8034, 103.7450
        ]), summary.phonon_frequencies, decimal=5)

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.3156, a)
        self.assertEqual(5.6533, b)
        self.assertEqual(3.8479 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(92.8923, alpha)
        self.assertEqual(109.0915, beta)
        self.assertEqual(93.9925, gamma)

        self.assertEqual(
            ["Cl1", "Cl2", "C1", "C2", "C3", "C4", "C5", "C6", "H1",
             "H2", "H3", "H4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.65752699,   0.29460980, 0.01765558],
            [1.34247301, -0.29460980, -0.01765558],
            [0.84803921, 0.13055532, 0.00850833],
            [1.03882686, 0.22472117, 0.19038327],
            [0.80852476, -0.09387537, -0.18062983],
            [1.15196079, -0.13055532, -0.00850833],
            [0.96117314, -0.22472117, -0.19038327],
            [1.19147524, 0.09387537, 0.18062983],
            [1.06762741, 0.39897076, 0.33612589],
            [0.65995872, -0.16499214, -0.32062205],
            [0.93237259, -0.39897076, -0.33612589],
            [1.34004128, 0.16499214, 0.32062205]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(1, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_13(self):
        res = join(
            dirname(__file__),
            "test_suite/13.properties_FINVAZ/NEIGHCRYS_input/FINVAZ_A395.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/13.properties_FINVAZ/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/13.properties_FINVAZ/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/13.properties_FINVAZ/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/13.properties_FINVAZ/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "w",
            "foreshorten_hydrogens": True,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "STAR": "PROP",
            "ACCM": 100000000,
            "NOPR": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-626.1263, summary.initial_energy)
        self.assertEqual(-626.1272, summary.final_energy)
        self.assertEqual(2117.8138, summary.initial_volume)
        self.assertEqual(1.1774, summary.initial_density)
        self.assertEqual(2118.6039, summary.final_volume)
        self.assertEqual(1.1770, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-581.7066, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-14.7209, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-26.7778, summary.lattice_energy_contributions["Higher multipole interaction energy"])
        np.testing.assert_almost_equal(np.array([
            0.0000, 0.0000, 0.0000, 29.6841, 35.8839, 36.8223, 42.9154,
            46.6563, 49.7245, 52.7854, 56.2845, 56.6495, 57.6731,
            66.4173, 68.8675, 74.3252, 75.4281, 79.1519, 82.6188,
            88.4994, 93.3421, 97.0813, 110.3716, 110.9669, 124.0169,
            134.7954, 144.9962, 149.4570, 160.8888, 161.0403, 169.7963,
            177.0508, 179.5408, 180.0014, 182.0893, 184.9637
        ]), summary.phonon_frequencies, decimal=5)

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(23.2596, a)
        self.assertEqual(11.0722, b)
        self.assertEqual(9.8252 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(123.1460, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
             "C10", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8",
             "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16",
             "H17", "H18", "N1", "Cl1"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.10047545, 0.70863039, 0.15517842],
            [0.07648732, 0.82541753, 0.19132478],
            [0.11001391, 0.93152276, 0.15858438],
            [0.08766036, 0.92856821, -0.01992281],
            [0.11104345, 0.80974513, -0.05499433],
            [0.07751857, 0.70331632, -0.02282767],
            [0.17820367, 0.69502331, 0.26704216],
            [0.21146435, 0.80147930, 0.23410371],
            [0.18934479, 0.79822014, 0.05577159],
            [0.18832130, 0.92028391, 0.26984983],
            [0.01481804, 0.60817082, 0.11499834],
            [0.08120354, 0.60270458, 0.30611124],
            [0.08188428, 0.52220026, 0.16491288],
            [0.02045436, 0.83256640, 0.11357864],
            [0.09152855, 0.82671645, 0.31819217],
            [0.09310056, 1.01556847, 0.18418464],
            [0.03190580, 0.93812362, -0.10000688],
            [0.11039277, 1.00472789, -0.04488434],
            [0.09485922, 0.80661088, -0.18227941],
            [0.09328983, 0.61763194, -0.04850351],
            [0.02149200, 0.70967480, -0.10194744],
            [0.19390390, 0.69547976, 0.39439632],
            [0.19462799, 0.60928975, 0.24323915],
            [0.26715720, 0.79242087, 0.31375203],
            [0.20647272, 0.71434712, 0.02994508],
            [0.21358724, 0.87244012, 0.03194087],
            [0.21254520, 0.99631641, 0.24921027],
            [0.20470738, 0.92389319, 0.39745580],
            [0.06728760, 0.60309035, 0.18755754],
            [0.40730259, 1.12806835, 0.96096212]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(15, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_14(self):
        res = join(
            dirname(__file__),
            "test_suite/14.press_YICDIZ/NEIGHCRYS_input/YICDIZ.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/14.press_YICDIZ/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/14.press_YICDIZ/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/14.press_YICDIZ/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/14.press_YICDIZ/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": True,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "PRES": "0.5 GPa"
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-11.4386, summary.initial_energy)
        self.assertEqual(-11.9699, summary.final_energy)
        self.assertEqual(567.7725, summary.initial_volume)
        self.assertEqual(1.2884, summary.initial_density)
        self.assertEqual(572.6596, summary.final_volume)
        self.assertEqual(1.2774, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-8.4252, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(5.4905, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-2.5268, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(13.5921, a)
        self.assertEqual(8.8802, b)
        self.assertEqual(4.7445 , c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "F1", "H1", "H2",
             "H3", "H4", "H5", "H6", "H7"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.93969899, 0.74887928, 0.55967403],
            [0.89859876, 0.61370118, 0.65718825],
            [0.82481519, 0.61263342, 0.84885722],
            [0.78433857, 0.74807439, 0.93387810],
            [0.81842528, 0.88030019, 0.83850094],
            [0.89721832, 0.87923471, 0.65319229],
            [1.02548826, 0.74558482, 0.36195464],
            [0.70830737, 0.74658528, 1.12008606],
            [0.92648017, 0.50773331, 0.57727109],
            [0.79800885, 0.50730380, 0.93454108],
            [0.78525307, 0.98535216, 0.90492940],
            [0.92623357, 0.98577041, 0.57947268],
            [1.08467943, 0.68092909, 0.45525900],
            [1.05035648, 0.85952794, 0.32262056],
            [1.00346791, 0.69393661, 0.16502742]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(19, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_15(self):
        res = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/OBEQUJ.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/dmacrys.dma"
        )
        pols = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/dmacrys.dma.pol"
        )
        pots = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/15.induction_OBEQUJ/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "polarizabilities_filename": pols,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-68.3145, summary.initial_energy)
        self.assertEqual(-70.2546, summary.final_energy)
        self.assertEqual(1152.2180, summary.initial_volume)
        self.assertEqual(1.3848, summary.initial_density)
        self.assertEqual(1206.8094, summary.final_volume)
        self.assertEqual(1.3221, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-11.6502, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-12.0597, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-0.9971, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(11.0377, a)
        self.assertEqual(6.8310, b)
        self.assertEqual(16.0058, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "N1", "N2", "O1", "H1",
             "H2", "H3", "H4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.29000556, 0.12220636, 0.56307405],
            [0.41674055, 0.19061354, 0.55326558],
            [0.50552343, 0.18989850, 0.61792430],
            [0.47299841, 0.12188430, 0.69471152],
            [0.35192774, 0.05369210, 0.70838327],
            [0.26574957, 0.05321667, 0.64733524],
            [0.44365955, 0.25515295, 0.47757164],
            [0.45428679, 0.30562042, 0.41092208],
            [0.21637774, 0.12528176, 0.50562530],
            [0.59672621, 0.24317881, 0.60486087],
            [0.53864169, 0.11930742, 0.74554758],
            [0.32801360, -0.00018008, 0.77053122],
            [0.17400872, 0.00052247, 0.65918265]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(61, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_16(self):
        res = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/NEIGHCRYS_input/PAPTUX_CO1.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/16.symmred_PAPTUX/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [3, 0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-134.1608, summary.initial_energy)
        self.assertEqual(-134.7345, summary.final_energy)
        self.assertEqual(2343.7864, summary.initial_volume)
        self.assertEqual(1.3051, summary.initial_density)
        self.assertEqual(2338.1331, summary.final_volume)
        self.assertEqual(1.3082, summary.final_density)
        self.assertEqual(8, summary.z)
        self.assertEqual(-19.0460, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-1.6809, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-20.6588, summary.lattice_energy_contributions["Higher multipole interaction energy"])

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
            [0.37558754, 0.34902684, 0.26334263],
            [0.39109578, 0.39342299, 0.05670493],
            [0.35247488, 0.41682112, -0.10246569],
            [0.25663261, 0.42506986, -0.22489251],
            [0.20423745, 0.41233247, -0.17500421],
            [0.18898901, 0.37064812, 0.03800526],
            [0.22707515, 0.34293057, 0.19270989],
            [0.32321281, 0.33180964, 0.30622103],
            [0.45004718, 0.41203402, 0.01179287],
            [0.47448587, 0.45023576, 0.23447744],
            [0.48220443, 0.33969784, -0.06604884],
            [0.11606930, 0.31731487, 0.25951633],
            [0.29758674, 0.39928033, -0.06329549],
            [0.28235268, 0.35741678, 0.14536047],
            [0.84296764, 0.15138622, 0.74789134],
            [0.85742218, 0.10659179, 0.95526991],
            [0.81800788, 0.08295054, 1.11005104],
            [0.72157703, 0.07459814, 1.22253697],
            [0.66944962, 0.08749307, 1.16754988],
            [0.65528513, 0.12958711, 0.95377990],
            [0.69414477, 0.15754020, 0.80348528],
            [0.79082805, 0.16874800, 0.69999336],
            [0.91612596, 0.08782418, 1.00584632],
            [0.94164651, 0.05000112, 0.78503721],
            [0.94792436, 0.15997563, 1.08821849],
            [0.58351212, 0.18341830, 0.72584173],
            [0.76333724, 0.10063226, 1.06560995],
            [0.74916557, 0.14289749, 0.85619692],
            [0.50324756, 0.41874114, 0.37662095],
            [0.45954385, 0.52463563, 0.25871994],
            [0.13561662, 0.36161930, 0.06453953],
            [0.97112351, 0.08171990, 0.64641718],
            [0.92679144, -0.02433416, 0.75796693],
            [0.60206275, 0.13873190, 0.92197558],
            [0.40583223, 0.32957238, 0.38916014],
            [0.36323295, 0.45029832, -0.26108718],
            [0.26782477, 0.45654442, -0.38662272],
            [0.17296179, 0.43238572, -0.29405513],
            [0.21657964, 0.31121388, 0.35423953],
            [0.31218954, 0.29890301, 0.46543825],
            [0.45180666, 0.45570794, -0.13132745],
            [0.52355257, 0.35512944, -0.10473952],
            [0.46454444, 0.31468536, -0.22844796],
            [0.48234675, 0.29541570, 0.07493746],
            [0.07343016, 0.31095913, 0.23030866],
            [0.13413358, 0.25937121, 0.26397025],
            [0.12319369, 0.34658934, 0.43164539],
            [0.47465125, 0.54454058, 0.40424438],
            [0.87384003, 0.17103321, 0.62553159],
            [0.82795770, 0.04916905, 1.26913508],
            [0.73194637, 0.04281300, 1.38480845],
            [0.63757986, 0.06726104, 1.28302930],
            [0.68447103, 0.18956622, 0.64148955],
            [0.78061578, 0.20196036, 0.54027666],
            [0.91715143, 0.04388596, 1.14833693],
            [0.98906188, 0.14442104, 1.13083823],
            [0.92947392, 0.18471179, 1.24921622],
            [0.94878992, 0.20451574, 0.94806578],
            [0.54074175, 0.18977413, 0.75081046],
            [0.60162308, 0.24134637, 0.72425109],
            [0.59147586, 0.15445138, 0.55396837],
            [0.94260859, -0.04399063, 0.61366942]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        # no space group test as cspy does not support non standard
        # space groups

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_17(self):
        res = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/TEVSOD.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/fit.pots"
        )
        paste = join(
            dirname(__file__),
            "test_suite/17.pasting_TEVSOD/NEIGHCRYS_input/coord_to_paste_au"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": paste
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-133.4879, summary.initial_energy)
        self.assertEqual(-136.4097, summary.final_energy)
        self.assertEqual(1203.3924, summary.initial_volume)
        self.assertEqual(1.3098, summary.initial_density)
        self.assertEqual(1220.4251, summary.final_volume)
        self.assertEqual(1.2915, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-35.0721, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(17.2292, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-23.2976, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(5.6437, a)
        self.assertEqual(9.3846, b)
        self.assertEqual(23.3169, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(98.7977, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "O1", "N1", "H1",
             "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
             "H11", "H12", "H13", "H14", "H15"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.33523308, 0.86712578, 0.21274447],
            [0.11430230, 0.92810974, 0.17278132],
            [0.03240120, 0.82932837, 0.12197733],
            [-0.14587842, 0.73073999, 0.13003790],
            [-0.23805975, 0.63397180, 0.08749257],
            [-0.15134273, 0.63454388, 0.03478071],
            [0.02598276, 0.73108243, 0.02632587],
            [0.12181515, 0.82918215, 0.06896157],
            [0.31414674, 0.92668900, 0.05214859],
            [0.46162490, 1.01615068, 0.09945126],
            [0.31266057, 1.12660526, 0.12397486],
            [0.33641866, 1.27112705, 0.11217180],
            [0.19310978, 1.37299125, 0.13353384],
            [0.02262158, 1.33051223, 0.16720439],
            [-0.00268442, 1.18644658, 0.17923249],
            [0.14007580, 1.08361442, 0.15791452],
            [0.39478022, 0.74278173, 0.21140775],
            [0.44683299, 0.96167268, 0.25205540],
            [0.59261784, 0.92762854, 0.27889695],
            [0.41073090, 1.06662718, 0.24887788],
            [-0.02645678, 0.92330721, 0.20046499],
            [-0.21292292, 0.73020251, 0.17134342],
            [-0.37607951, 0.55907133, 0.09543150],
            [-0.22042359, 0.56010542, 0.00061067],
            [0.09512858, 0.73113957, -0.01475538],
            [0.22985231, 0.99908903, 0.01794895],
            [0.43803999, 0.86035422, 0.03179844],
            [0.60506865, 1.06893358, 0.08071749],
            [0.55018482, 0.94588805, 0.13376932],
            [0.46986598, 1.30417762, 0.08581164],
            [0.21483488, 1.48502504, 0.12386663],
            [-0.09062654, 1.40880850, 0.18404901],
            [-0.13630728, 1.15284886, 0.20546288]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_19(self):
        res = join(
            dirname(__file__),
            "test_suite/19.largecubic_BTXXV/NEIGHCRYS_input/BTXXV_5721.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/19.largecubic_BTXXV/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/19.largecubic_BTXXV/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/19.largecubic_BTXXV/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/19.largecubic_BTXXV/NEIGHCRYS_input/pote.dat"
        )
        # using the extended mode as cspy does not support editing the
        # RDMA and CUTO in the dmain directly
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": True,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "wcal_ngcv": False,
            "step_size": 0.5,
            "max_iterations": 1000,
            "vdw_cutoff": 30,
            "connectivity_search_max_cells": 3,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-206.0935, summary.initial_energy)
        self.assertEqual(-206.4054, summary.final_energy)
        self.assertEqual(13981.3227, summary.initial_volume)
        self.assertEqual(1.3182, summary.initial_density)
        self.assertEqual(13983.5413, summary.final_volume)
        self.assertEqual(1.3180, summary.final_density)
        self.assertEqual(24, summary.z)
        self.assertEqual(-27.7567, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-17.0388, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(5.2789, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(24.0920, a)
        self.assertEqual(24.0920, b)
        self.assertEqual(24.0920, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
             "C20", "C21", "C22", "C23", "C24", "N1", "N2", "N3", "N4",
             "O1", "O2", "O3", "O4", "O5", "O6", "H1", "H2", "H3", "H4",
             "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13",
             "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.99133211, 0.42838994, 0.33055854],
            [1.04332100, 0.43997665, 0.30778387],
            [1.05968664, 0.41175076, 0.26034369],
            [1.02632264, 0.37242441, 0.23464760],
            [0.97493764, 0.36200334, 0.25845354],
            [0.95662678, 0.38912122, 0.30591359],
            [0.97468538, 0.45922532, 0.38151298],
            [0.10566374, 0.35548631, 0.74998046],
            [0.16026522, 0.34528367, 0.76354054],
            [0.17466584, 0.29931940, 0.79434728],
            [0.13524982, 0.26061527, 0.81066807],
            [0.08124638, 0.27183550, 0.79769704],
            [0.06426850, 0.31780682, 0.76703831],
            [0.00467362, 0.32913670, 0.75408236],
            [-0.00772732, 0.42630999, 0.77335888],
            [-0.05542958, 0.43236857, 0.80439466],
            [-0.05955195, 0.47243618, 0.84385669],
            [-0.01585459, 0.51013134, 0.85303288],
            [0.03098418, 0.50318547, 0.82164310],
            [0.03695620, 0.46229628, 0.78195552],
            [0.09031100, 0.45478992, 0.74975303],
            [0.03811157, 0.39656357, 0.69098055],
            [0.14960362, 0.20886821, 0.84247936],
            [-0.02155574, 0.55504527, 0.89724528],
            [1.11445159, 0.42378467, 0.23614421],
            [0.93853840, 0.32056294, 0.23227964],
            [0.09237556, 0.40363151, 0.71712188],
            [-0.00608105, 0.38430418, 0.73053633],
            [0.92402894, 0.44470992, 0.39959552],
            [1.00343459, 0.49352332, 0.40417365],
            [1.14291503, 0.45847358, 0.25949608],
            [1.12743515, 0.39819224, 0.19447435],
            [0.95582208, 0.29751861, 0.19069558],
            [0.89362762, 0.31220258, 0.25405455],
            [1.07029643, 0.47051676, 0.32697122],
            [1.03984872, 0.35080821, 0.19759079],
            [0.91612148, 0.37962509, 0.32305487],
            [0.91643813, 0.46656868, 0.43267151],
            [0.19217126, 0.37380589, 0.74978561],
            [0.21759617, 0.29347150, 0.80632719],
            [0.04935904, 0.24349653, 0.81188028],
            [-0.01935661, 0.32453350, 0.79179647],
            [-0.00979656, 0.29824275, 0.72481263],
            [-0.09032331, 0.40530708, 0.79601087],
            [-0.09664772, 0.47548958, 0.86905482],
            [0.06543265, 0.53129158, 0.82826869],
            [0.09633999, 0.49037509, 0.72295490],
            [0.12443419, 0.45362111, 0.77900018],
            [0.02760077, 0.43424284, 0.66883021],
            [0.04065372, 0.36312749, 0.66105666],
            [0.13750687, 0.17266800, 0.81874717],
            [0.12789655, 0.20892977, 0.88184664],
            [0.19394521, 0.20773394, 0.84984118],
            [0.01434137, 0.58206137, 0.89600462],
            [-0.02443046, 0.53584832, 0.93779291],
            [-0.05870942, 0.57908428, 0.88932012]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(205, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_20(self):
        res = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/BTXXV_5721.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/pote.dat"
        )
        hess = join(
            dirname(__file__),
            "test_suite/20.largecubic_HESS_BTXXV/NEIGHCRYS_input/dmahessian"
        )
        # using the extended mode as cspy does not support editing the
        # RDMA and CUTO in the dmain directly
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "hessian_filename": hess,
            "extended_mode": True,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "wcal_ngcv": False,
            "step_size": 0.5,
            "max_iterations": 1000,
            "vdw_cutoff": 30,
            "connectivity_search_max_cells": 3,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "HESS": 1
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-206.0935, summary.initial_energy)
        self.assertEqual(-206.4054, summary.final_energy)
        self.assertEqual(13981.3227, summary.initial_volume)
        self.assertEqual(1.3182, summary.initial_density)
        self.assertEqual(13983.5413, summary.final_volume)
        self.assertEqual(1.3180, summary.final_density)
        self.assertEqual(24, summary.z)
        self.assertEqual(-27.7567, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-17.0388, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(5.2789, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(24.0920, a)
        self.assertEqual(24.0920, b)
        self.assertEqual(24.0920, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
             "C20", "C21", "C22", "C23", "C24", "N1", "N2", "N3", "N4",
             "O1", "O2", "O3", "O4", "O5", "O6", "H1", "H2", "H3", "H4",
             "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13",
             "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.99133211, 0.42838994, 0.33055854],
            [1.04332100, 0.43997665, 0.30778387],
            [1.05968664, 0.41175076, 0.26034369],
            [1.02632264, 0.37242441, 0.23464760],
            [0.97493764, 0.36200334, 0.25845354],
            [0.95662678, 0.38912122, 0.30591359],
            [0.97468538, 0.45922532, 0.38151298],
            [0.10566374, 0.35548631, 0.74998046],
            [0.16026522, 0.34528367, 0.76354054],
            [0.17466584, 0.29931940, 0.79434728],
            [0.13524982, 0.26061527, 0.81066807],
            [0.08124638, 0.27183550, 0.79769704],
            [0.06426850, 0.31780682, 0.76703831],
            [0.00467362, 0.32913670, 0.75408236],
            [-0.00772732, 0.42630999, 0.77335888],
            [-0.05542958, 0.43236857, 0.80439466],
            [-0.05955195, 0.47243618, 0.84385669],
            [-0.01585459, 0.51013134, 0.85303288],
            [0.03098418, 0.50318547, 0.82164310],
            [0.03695620, 0.46229628, 0.78195552],
            [0.09031100, 0.45478992, 0.74975303],
            [0.03811157, 0.39656357, 0.69098055],
            [0.14960362, 0.20886821, 0.84247936],
            [-0.02155574, 0.55504527, 0.89724528],
            [1.11445159, 0.42378467, 0.23614421],
            [0.93853840, 0.32056294, 0.23227964],
            [0.09237556, 0.40363151, 0.71712188],
            [-0.00608105, 0.38430418, 0.73053633],
            [0.92402894, 0.44470992, 0.39959552],
            [1.00343459, 0.49352332, 0.40417365],
            [1.14291503, 0.45847358, 0.25949608],
            [1.12743515, 0.39819224, 0.19447435],
            [0.95582208, 0.29751861, 0.19069558],
            [0.89362762, 0.31220258, 0.25405455],
            [1.07029643, 0.47051676, 0.32697122],
            [1.03984872, 0.35080821, 0.19759079],
            [0.91612148, 0.37962509, 0.32305487],
            [0.91643813, 0.46656868, 0.43267151],
            [0.19217126, 0.37380589, 0.74978561],
            [0.21759617, 0.29347150, 0.80632719],
            [0.04935904, 0.24349653, 0.81188028],
            [-0.01935661, 0.32453350, 0.79179647],
            [-0.00979656, 0.29824275, 0.72481263],
            [-0.09032331, 0.40530708, 0.79601087],
            [-0.09664772, 0.47548958, 0.86905482],
            [0.06543265, 0.53129158, 0.82826869],
            [0.09633999, 0.49037509, 0.72295490],
            [0.12443419, 0.45362111, 0.77900018],
            [0.02760077, 0.43424284, 0.66883021],
            [0.04065372, 0.36312749, 0.66105666],
            [0.13750687, 0.17266800, 0.81874717],
            [0.12789655, 0.20892977, 0.88184664],
            [0.19394521, 0.20773394, 0.84984118],
            [0.01434137, 0.58206137, 0.89600462],
            [-0.02443046, 0.53584832, 0.93779291],
            [-0.05870942, 0.57908428, 0.88932012]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(205, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_21(self):
        res = join(
            dirname(__file__),
            "test_suite/21.longcell_BTXXIII/NEIGHCRYS_input/BTXXIII_16.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/21.longcell_BTXXIII/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/21.longcell_BTXXIII/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/21.longcell_BTXXIII/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/21.longcell_BTXXIII/NEIGHCRYS_input/fit.pots"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": True,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "wcal_ngcv": False,
            "step_size": 0.5,
            "max_iterations": 1000,
            "vdw_cutoff": 15,
            "connectivity_search_max_cells": 4,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-128.0126, summary.initial_energy)
        self.assertEqual(-129.4032, summary.final_energy)
        self.assertEqual(4711.8247, summary.initial_volume)
        self.assertEqual(1.0890, summary.initial_density)
        self.assertEqual(4631.4271, summary.final_volume)
        self.assertEqual(1.1079, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(7.4656, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-9.7653, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-6.9167, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(14.5659, a)
        self.assertEqual(69.8043, b)
        self.assertEqual(4.5551, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
             "C20", "C21", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
             "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15",
             "H16", "H17", "N1", "O1", "O2", "Cl1", "Cl2"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.85528942, 0.92647622, 0.37358603],
            [0.92346020, 0.91635649, 0.52715211],
            [1.01087924, 0.92454554, 0.54528189],
            [1.03032137, 0.94221582, 0.41442215],
            [0.96163752, 0.95209502, 0.26179476],
            [0.87387782, 0.94404548, 0.24269512],
            [0.90051536, 0.89715306, 0.66444255],
            [0.97864544, 0.88774837, 0.83472890],
            [0.86665029, 0.83979330, 1.05174033],
            [0.93197735, 0.83232201, 1.24665000],
            [1.00987519, 0.84344170, 1.31170266],
            [1.02220445, 0.86113449, 1.17752210],
            [0.95794842, 0.86867298, 0.97853172],
            [0.87946595, 0.85771068, 0.92108342],
            [0.96162461, 0.78647441, 1.67726618],
            [1.03154944, 0.77334889, 1.75169086],
            [1.11741432, 0.77384452, 1.62116219],
            [1.13438991, 0.78769685, 1.40486250],
            [1.06751575, 0.80087153, 1.32720285],
            [0.98011312, 0.80094224, 1.46288412],
            [0.87427856, 0.78366401, 1.83643162],
            [0.78631243, 0.92045892, 0.35577716],
            [1.06646358, 0.91740683, 0.66178906],
            [0.82078174, 0.95179557, 0.12300985],
            [0.87740108, 0.88751092, 0.48743699],
            [0.84069204, 0.89901861, 0.81013296],
            [1.03772722, 0.88573899, 0.68612159],
            [1.00257936, 0.89767765, 1.00670000],
            [0.80577200, 0.83133804, 0.99925918],
            [1.06005644, 0.83825764, 1.47058222],
            [1.08322868, 0.86954105, 1.23152966],
            [0.82724604, 0.86291984, 0.76897537],
            [0.85017164, 0.81188693, 1.44995523],
            [1.01492590, 0.76265437, 1.91776286],
            [1.17010542, 0.76350869, 1.68275301],
            [1.20059410, 0.78808635, 1.29246189],
            [1.08132175, 0.81125656, 1.15432391],
            [0.75060002, 0.79146690, 1.86482176],
            [0.91550285, 0.81445622, 1.38128159],
            [0.86172103, 0.77233124, 2.03703992],
            [0.80239465, 0.79500903, 1.73908389],
            [1.14053764, 0.95150666, 0.44588962],
            [0.98284677, 0.97403306, 0.09586273],
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(41, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_22(self):
        res = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/BTXXII_20.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/pote_same_as_CP.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/22.damped_C6_dispersion_BTXXII/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-129.1484, summary.initial_energy)
        self.assertEqual(-133.0953, summary.final_energy)
        self.assertEqual(1014.1676, summary.initial_volume)
        self.assertEqual(1.6263, summary.initial_density)
        self.assertEqual(968.9021, summary.final_volume)
        self.assertEqual(1.7023, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-41.2898, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(0.0000, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(0.0000, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(8.4802, a)
        self.assertEqual(7.6326, b)
        self.assertEqual(15.2869, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(78.3025, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "S1", "S2",
             "S3", "N1", "N2", "N3", "N4"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.95186423, 0.67783239, 0.88352291],
            [1.05099892, 0.76097540, 0.93597756],
            [1.20439652, 0.66757162, 0.74255717],
            [1.30042105, 0.74625098, 0.79210390],
            [0.81610672, 0.61068465, 0.93867036],
            [1.27064451, 0.57592346, 0.66228682],
            [1.47152235, 0.74000539, 0.76540918],
            [0.68589236, 0.51959220, 0.91448289],
            [0.99090026, 0.67191117, 0.76617941],
            [1.23390808, 0.86961237, 0.89092209],
            [0.82294350, 0.65957243, 1.04858127],
            [1.00056850, 0.76000254, 1.02290052],
            [1.31899137, 0.50263552, 0.59558608],
            [1.61123447, 0.74153595, 0.74518611],
            [0.57748690, 0.44568701, 0.89580845]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_23(self):
        res = join(
            dirname(__file__),
            "test_suite/23.largecell_KAXXAI_30ang/NEIGHCRYS_input/KAXXAI02.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/23.largecell_KAXXAI_30ang/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/23.largecell_KAXXAI_30ang/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/23.largecell_KAXXAI_30ang/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/23.largecell_KAXXAI_30ang/NEIGHCRYS_input/pote.dat"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "extended_mode": True,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": True,
            "wcal_ngcv": False,
            "step_size": 0.5,
            "max_iterations": 1000,
            "vdw_cutoff": 30,
            "connectivity_search_max_cells": 3,
            "potential_type": "f",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {
            "spline": False
        }
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-271.8905, summary.initial_energy)
        self.assertEqual(-279.7492, summary.final_energy)
        self.assertEqual(2419.1969, summary.initial_volume)
        self.assertEqual(1.4371, summary.initial_density)
        self.assertEqual(2489.6114, summary.final_volume)
        self.assertEqual(1.3964, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-47.8678, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-13.8497, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-25.6842, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(7.7436, a)
        self.assertEqual(11.5457, b)
        self.assertEqual(27.8533, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(91.2849, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
             "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
             "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
             "C27", "C28", "N1", "N2", "O1", "O2", "O3", "O4", "Cl1",
             "Cl2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
             "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17",
             "H18", "H19", "H20", "H21", "H22", "H23", "H24"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.41302497, 0.85576498, 0.03269994],
            [0.54388870, 0.93110069, 0.05574920],
            [0.71279391, 0.92944841, 0.03891897],
            [0.83976955, 1.00008422, 0.05864137],
            [0.79730881, 1.07318672, 0.09653374],
            [0.63301769, 1.07590975, 0.11424137],
            [0.50036184, 1.00587065, 0.09387362],
            [0.26779619, 1.09340760, 0.14152073],
            [0.29379390, 1.21021744, 0.13142538],
            [0.22564713, 1.29467916, 0.16045890],
            [0.12632175, 1.26376977, 0.19911816],
            [0.09615079, 1.14717454, 0.20765468],
            [0.16654255, 1.05902675, 0.18020206],
            [0.13694491, 0.93299181, 0.19040805],
            [0.05808572, 0.65268375, 0.96897465],
            [-0.06211115, 0.56659082, 0.94804521],
            [-0.23408569, 0.56572602, 0.96242619],
            [-0.35228016, 0.48810328, 0.94284396],
            [-0.29755864, 0.41042205, 0.90809053],
            [-0.12927079, 0.40877147, 0.89315545],
            [-0.00616572, 0.48638231, 0.91303019],
            [0.23917589, 0.41852861, 0.86359961],
            [0.23674928, 0.29819189, 0.86612586],
            [0.32291335, 0.23355326, 0.83229295],
            [0.41531549, 0.28751556, 0.79681204],
            [0.41857743, 0.40776219, 0.79520992],
            [0.32969848, 0.47605684, 0.82758694],
            [0.32835661, 0.60592107, 0.82545371],
            [0.33702693, 1.00732808, 0.11181177],
            [0.16175995, 0.48704555, 0.89883881],
            [0.46949112, 0.79292279, -0.00310800],
            [0.26105843, 0.85036679, 0.04553198],
            [0.20884614, 0.66389243, 0.95611661],
            [-0.00584995, 0.71880140, 1.00275327],
            [-0.03767847, 1.11264641, 0.25479302],
            [0.53901885, 0.47289217, 0.75064962],
            [0.36763298, 0.74910074, -0.01775962],
            [0.74464061, 0.87160447, 0.00981619],
            [0.96959762, 0.99884081, 0.04497864],
            [0.89575950, 1.12896861, 0.11226034],
            [0.60468128, 1.13227823, 0.14412870],
            [0.25905174, 0.94033427, 0.10287610],
            [0.36792351, 1.23484222, 0.10053274],
            [0.24998838, 1.38523427, 0.15305767],
            [0.07313194, 1.32936936, 0.22232064],
            [0.02486114, 0.90317046, 0.16999853],
            [0.11540214, 0.92133657, 0.22838730],
            [0.24915207, 0.88358099, 0.18016769],
            [0.09112412, 0.76907309, 1.01628881],
            [-0.27539772, 0.62687132, 0.98932695],
            [-0.48487983, 0.48774767, 0.95440665],
            [-0.38971247, 0.34988660, 0.89247902],
            [-0.09093232, 0.34744970, 0.86596898],
            [0.24036859, 0.54407405, 0.91594502],
            [0.16771109, 0.25515566, 0.89442495],
            [0.31785269, 0.13987887, 0.83371027],
            [0.48410059, 0.23689059, 0.77067116],
            [0.19822769, 0.63719676, 0.83060823],
            [0.37260824, 0.63406177, 0.79073688],
            [0.41319852, 0.64016259, 0.85344703]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(14, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_24(self):
        res = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/PYR1opt.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/dmacrys.dma"
        )
        pots = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/24.dampedispersion_PYRDIN/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-58.1316, summary.initial_energy)
        self.assertEqual(-61.9302, summary.final_energy)
        self.assertEqual(451.9388, summary.initial_volume)
        self.assertEqual(1.1625, summary.initial_density)
        self.assertEqual(401.6851, summary.final_volume)
        self.assertEqual(1.3080, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-10.8588, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-1.9319, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-4.0349, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(5.3784, a)
        self.assertEqual(6.7129, b)
        self.assertEqual(11.1254, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["N1", "C1", "C2", "C3", "C4", "C5", "H1", "H2", "H3",
             "H4", "H5"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.86633488, 0.56562298, 0.11891954],
            [0.86057613, 0.42084583, 0.20035203],
            [0.66964905, 0.28266585, 0.20929580],
            [0.47338346, 0.29556553, 0.12905499],
            [0.47681579, 0.44570917, 0.04366279],
            [0.67682910, 0.57620498, 0.04252441],
            [1.01709604, 0.41471788, 0.26194235],
            [0.67556597, 0.16798557, 0.27768613],
            [0.32058184, 0.19055182, 0.13299594],
            [0.32778667, 0.46203575, -0.02103660],
            [0.68499405, 0.69551311, -0.02331380]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(19, space_group.international_tables_number)

    @pytest.mark.external_binaries
    def test_neighcrys_dmacrys_25(self):
        res = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/PYR1opt.res"
        )
        bondlengths = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/bondlengths"
        )
        axis = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/dmacrys.mols"
        )
        mults = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/dmacrys.dma"
        )
        pols = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/dmacrys.dma.pol"
        )
        pots = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/pote.dat"
        )
        labels = join(
            dirname(__file__),
            "test_suite/25.polarizability_PYRDIN/NEIGHCRYS_input/labels"
        )
        # values taken from neighcrys_answers see NEIGHCRYS_input directory
        neighcrys_kwargs = {
            "bondlength_filename": bondlengths,
            "axis_filename": axis,
            "multipole_filename": mults,
            "polarizabilities_filename": pols,
            "potential_filename": pots,
            "labels_filename": labels,
            "extended_mode": False,
            "max_intermolecular_distance": 4.0,
            "insert_bond_centre_sites": False,
            "standardise_bonds": False,
            "potential_type": "c",
            "foreshorten_hydrogens": False,
            "symmetry_subgroup": [0],
            "paste_molecular_structure_filename": ""
        }
        dmacrys_kwargs = {}
        output = self.run_neighcrys_dmacrys(
            res, neighcrys_kwargs, dmacrys_kwargs
        )
        summary = DmacrysSummary(output.summary_contents)
        shelx = parse_shelx_file_content(output.shelx_contents)

        # values taken from test suite see DMACRYS_output directory
        self.assertTrue(summary.valid)
        self.assertEqual(-61.9760, summary.initial_energy)
        self.assertEqual(-66.7578, summary.final_energy)
        self.assertEqual(451.9388, summary.initial_volume)
        self.assertEqual(1.1625, summary.initial_density)
        self.assertEqual(401.6851, summary.final_volume)
        self.assertEqual(1.3080, summary.final_density)
        self.assertEqual(4, summary.z)
        self.assertEqual(-10.8588, summary.lattice_energy_contributions["Inter-molecular charge-charge energy"])
        self.assertEqual(-1.9319, summary.lattice_energy_contributions["Total charge-dipole+dipole-dipole energy"])
        self.assertEqual(-4.0349, summary.lattice_energy_contributions["Higher multipole interaction energy"])

        a, b, c = shelx["CELL"]["lengths"]
        self.assertEqual(5.3784, a)
        self.assertEqual(6.7129, b)
        self.assertEqual(11.1254, c)

        alpha, beta, gamma = shelx["CELL"]["angles"]
        self.assertEqual(90.0000, alpha)
        self.assertEqual(90.0000, beta)
        self.assertEqual(90.0000, gamma)

        self.assertEqual(
            ["N1", "C1", "C2", "C3", "C4", "C5", "H1", "H2", "H3",
             "H4", "H5"],
            [r["label"] for r in shelx["ATOM"]]
        )
        np.testing.assert_almost_equal(np.array([
            [0.86633488, 0.56562298, 0.11891954],
            [0.86057613, 0.42084583, 0.20035203],
            [0.66964905, 0.28266585, 0.20929580],
            [0.47338346, 0.29556553, 0.12905499],
            [0.47681579, 0.44570917, 0.04366279],
            [0.67682910, 0.57620498, 0.04252441],
            [1.01709604, 0.41471788, 0.26194235],
            [0.67556597, 0.16798557, 0.27768613],
            [0.32058184, 0.19055182, 0.13299594],
            [0.32778667, 0.46203575, -0.02103660],
            [0.68499405, 0.69551311, -0.02331380]
        ]),
            np.asarray([r["position"] for r in shelx["ATOM"]]),
            decimal=8
        )
        space_group = SpaceGroup.from_symmetry_operations(
            shelx["SYMM"], expand_latt=shelx["LATT"]
        )
        self.assertEqual(19, space_group.international_tables_number)
