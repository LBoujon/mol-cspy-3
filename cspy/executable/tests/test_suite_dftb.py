from cspy.crystal import Crystal
from cspy.executable import Dftb
from cspy.configuration import CONFIG
from cspy.formats.dftb_input import write_dftb_inputs
from cspy.formats.dftb_output import DftbOutput
from os.path import dirname, join, abspath
from os import remove
import logging
import sys
import unittest
import pytest
from filecmp import cmp

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)


class DftbTestCase(unittest.TestCase):
    def run_dftb(self, cif_file, settings):
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)          
            self.writing_inputs(cif_file, tmpdirname, settings) 
            exe = Dftb('dftb_in.hsd',
                       'geometry.gen',
                       working_directory=tmpdirname)
            exe.run()
        return exe


    def writing_inputs(self, cif_file, working_directory, settings):
        structure = Crystal.load(cif_file)
        write_dftb_inputs(structure, working_directory, settings)


    @pytest.mark.external_binaries
    def test_dftb_01(self):
        settings = {
            "skf_set":"3ob-3-1",
            "skf_path":join(dirname(__file__),'test_suite_dftb/parameter_sets/'),
            "disp_coeff":"3ob_brandenburg",
            "groups":1,
            "max_steps":2500,
            "max_force":0.00058,
            "alg":"LBFGS",
            "output_prefix":"geom.out",
            "kpoint_spacing":0.05,
            "scc_tol":1e-5,
            "timeout":1200,
            "single_point": True,
            "fixed_lattice_opt": False,
            "lattice_opt": False,
        }
        cif_file = join(dirname(__file__),'test_suite_dftb/01.singlepoint_ACETAC01_crystal/inputs/ACETAC01.cif')
        calc = self.run_dftb(cif_file, settings)
        output = DftbOutput(calc.output_contents)
        properties = output._parse_file()
        self.assertAlmostEqual(-46.3853370962, float(properties['final_energy']))
        

    @pytest.mark.external_binaries
    def test_dftb_02(self):
        settings = {
            "skf_set":"3ob-3-1",
            "skf_path":join(dirname(__file__),'test_suite_dftb/parameter_sets/'),
            "disp_coeff":"3ob_brandenburg",
            "groups":1,
            "max_steps":2500,
            "max_force":0.00058,
            "alg":"LBFGS",
            "output_prefix":"geom.out",
            "kpoint_spacing":0.05,
            "scc_tol":1e-5,
            "timeout":1200,
            "single_point": False,
            "fixed_lattice_opt": False,
            "lattice_opt": True,
        }
        cif_file = join(dirname(__file__),'test_suite_dftb/02.optimization_ACETAC01_crystal/inputs/ACETAC01.cif')
        calc = self.run_dftb(cif_file, settings)
        output = DftbOutput(calc.output_contents)
        self.assertTrue(output._valid())
        properties = output._parse_file()
        self.assertAlmostEqual(-46.4063421694, float(properties['final_energy']))


