from cspy.executable import Gdma
from cspy.executable.gdma import GdmaException
from os.path import dirname, join, abspath
import logging
import sys
import unittest
from cspy.chem.multipole import DistributedMultipoles
import pytest
import numpy as np

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))

ACETAC01_FCHK = join(EXAMPLES_DIR, "ACETAC01/ACETAC01_A.fchk")
ACETAC01_MULTS = join(EXAMPLES_DIR, "ACETAC01/ACETAC01.dma")
LOG = logging.getLogger(__name__)


class GdmaTestCase(unittest.TestCase):
    def run_gdma(self, formchk):
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            exe = Gdma(formchk)
            exe.working_directory = tmpdirname
            exe.run()
            return exe.result()

    def step1(self):
        res = self.run_gdma(ACETAC01_FCHK)
        self.mults = res
        assert self.mults is not None

    def step2(self):
        reference_mults = DistributedMultipoles.from_dma_file(ACETAC01_MULTS)
        ref_coords = reference_mults.molecules[0].positions
        coords = self.mults.molecules[0].positions
        LOG.debug("%s\n%s", ref_coords, coords)
        np.testing.assert_allclose(ref_coords, coords)

    @pytest.mark.external_binaries
    def test_gdma(self):
        self.step1()
        self.step2()
