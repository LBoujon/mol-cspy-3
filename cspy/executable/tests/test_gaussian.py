from cspy.executable import Gaussian
from cspy.executable.gaussian import GaussianException
from os.path import dirname, join, abspath
import logging
import sys
import unittest
import pytest

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)
TEST_INPUT = """
%chk=job
#P B3LYP/3-21G

test

0 1
H 1.4 0.0 0.0
H 0.0 0.0 0.0

"""


class GaussianTestCase(unittest.TestCase):
    def run_gaussian(self):
        with TemporaryDirectory() as tmpdirname:
            tmpdirname = "."
            LOG.debug("created temp directory: %s", tmpdirname)
            out_file = join(tmpdirname, "test.log")

            exe = Gaussian(TEST_INPUT)
            exe.working_directory = tmpdirname
            exe.run()
            return exe.result()

    @pytest.mark.external_binaries
    def test_gaussian_works(self):
        res = self.run_gaussian()
        print(res)
        assert res
