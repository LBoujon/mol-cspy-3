from cspy.executable import Neighcrys
from os.path import dirname, join, abspath, exists
import logging
import sys
import unittest
import pytest
import shutil
import difflib

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)
EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))
ACETAC01_RES = join(EXAMPLES_DIR, "ACETAC01/ACETAC01.res")
ACETAC01_AXIS = join(EXAMPLES_DIR, "ACETAC01/ACETAC01.mols")
ACETAC01_MULTS = join(EXAMPLES_DIR, "ACETAC01/ACETAC01_rank0.dma")
ACETAC01_DMAIN = join(EXAMPLES_DIR, "ACETAC01/ACETAC01.res.dmain")
BOND_CUTOFFS = join(EXAMPLES_DIR, "ACETAC01/bond_cutoffs")
DMAIN_FILES = {
    x: join(dirname(__file__), "{}.dmain".format(x)) for x in ("basic", "axis", "all")
}


def assert_same_file_contents(contents, ref_filename):
    assert exists(ref_filename)
    with open(ref_filename) as f:
        correct = f.read()
    delta = difflib.context_diff(
        contents.splitlines(keepends=True)[3:], correct.splitlines(keepends=True)[3:]
    )
    diff_string = "".join(delta)
    if diff_string:
        LOG.error("Diff:\n%s", diff_string)
        with open("bad.dmain", "w") as f:
            f.write(contents)
        assert False
    assert True


class NeighcrysTestCase(unittest.TestCase):
    def run_neighcrys(self, res, **kwargs):
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            tmp_res = join(tmpdirname, "a.res")
            shutil.copy(res, tmp_res)
            assert exists(tmp_res)
            LOG.debug("Copied %s -> %s", res, tmp_res)
            exe = Neighcrys("a.res", name="a", **kwargs)
            pots = join(tmpdirname, "pote.dat")
            shutil.copy(exe.args["potential_filename"], pots)
            LOG.debug("Copied %s -> %s", exe.args["potential_filename"], pots)
            assert exists(pots)
            exe.args["potential_filename"] = "pote.dat"
            shutil.copy(BOND_CUTOFFS, join(tmpdirname, exe.args["bondlength_filename"]))
            LOG.debug(
                "Copied %s -> %s",
                BOND_CUTOFFS,
                join(tmpdirname, exe.args["bondlength_filename"]),
            )
            if exe.args["axis_filename"]:
                axis = join(tmpdirname, "a.mols")
                shutil.copy(exe.args["axis_filename"], axis)
                LOG.debug("Copied %s -> %s", exe.args["axis_filename"], axis)
                exe.args["axis_filename"] = "a.mols"
                assert exists(axis)
            if exe.args["multipole_filename"]:
                mults = join(tmpdirname, "a.dma")
                shutil.copy(exe.args["multipole_filename"], mults)
                LOG.debug("Copied %s -> %s", exe.args["multipole_filename"], mults)
                exe.args["multipole_filename"] = "a.dma"
                assert exists(mults)
            exe.working_directory = tmpdirname
            exe.run()
            return exe.result()

    @pytest.mark.external_binaries
    def test_neighcrys(self):
        res = self.run_neighcrys(ACETAC01_RES)
        assert res is not None
        assert_same_file_contents(res, DMAIN_FILES["basic"])

    @pytest.mark.external_binaries
    def test_neighcrys_with_axis(self):
        res = self.run_neighcrys(ACETAC01_RES, axis_filename=ACETAC01_AXIS)
        assert res is not None
        assert_same_file_contents(res, DMAIN_FILES["axis"])

    @pytest.mark.external_binaries
    def test_neighcrys_with_axis(self):
        res = self.run_neighcrys(
            ACETAC01_RES, axis_filename=ACETAC01_AXIS, multipole_filename=ACETAC01_MULTS
        )
        assert res is not None
        assert_same_file_contents(res, DMAIN_FILES["all"])
