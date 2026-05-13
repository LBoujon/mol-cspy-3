import logging
from cspy.executable import Gdma, Mulfit, ReturnCodeError
from cspy.util.path import Path, TemporaryDirectory
from cspy.chem.multipole import DistributedMultipoles

LOG = logging.getLogger(__name__)


def gdma(fchk_contents, **kwargs):
    LOG.debug("Running GDMA")
    mults = None
    with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
        LOG.debug("created temp directory: %s", tmpdirname)
        fchk = Path(tmpdirname) / "input.fchk"
        fchk.write_text(fchk_contents)
        try:
            assert fchk.exists(), "Fchk file not found"
            exe = Gdma(fchk, working_directory=tmpdirname, **kwargs)
        except ReturnCodeError as e:
            LOG.error("Error, GDMA exited with returncode=%s", exe.returncode)
            return None
        except Exception as e:
            LOG.exception("Error when running GDMA: %s", e)
            return None
        exe.working_directory = tmpdirname
        exe.run()
        mults = DistributedMultipoles.from_dma_string(exe.punch_contents)
    return mults
