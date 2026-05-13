from unittest import TestCase
from cspy.chem.multipole import DistributedMultipoles
from os.path import dirname, join, exists

CSPY_DIR = dirname(dirname(dirname(dirname(dirname(__file__)))))

ACETAC01_XYZ = join(CSPY_DIR, "examples/ACETAC01/ACETAC01.xyz")
ACETAC01_0 = join(CSPY_DIR, "examples/ACETAC01/ACETAC01_rank0.dma")


class MultipoleTest(TestCase):
    def test_multipoles_parsed(self):
        assert exists(ACETAC01_0)
        mults = DistributedMultipoles.from_dma_file(ACETAC01_0)
        assert len(mults.molecules) == 2
