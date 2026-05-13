from cspy.executable import Compack
from cspy.executable.compack import CompackException
from os.path import dirname, join, abspath
import logging
import sys
import unittest
import pytest

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))

ACETAC01_RES = join(EXAMPLES_DIR, "ACETAC01_experiment.res")
LOG = logging.getLogger(__name__)


class CompackTestCase(unittest.TestCase):
    def run_compack(self, refs, comps):
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            out_file = join(tmpdirname, "compack.out")
            exe = Compack(
                reference_structures=refs,
                comparison_structures=comps,
                output_file=out_file,
            )
            exe.working_directory = tmpdirname
            exe.run()
            return exe.result

    @pytest.mark.skip
    def test_compack_works(self):
        with open(ACETAC01_RES) as f:
            res_str = f.read()
        refs = [res_str]
        comps = [res_str]
        res = self.run_compack(refs, comps)
        for ref, comp, rmsd in res:
            self.assertTrue(rmsd < 1e-6)

    @pytest.mark.skip
    def test_compack_fails_on_bad_inputs(self):
        res_str = "FAILURE OF A RES FILE"
        refs = [res_str]
        comps = [res_str]
        with self.assertRaises(CompackException):
            res = self.run_compack(refs, comps)
