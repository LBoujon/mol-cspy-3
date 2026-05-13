from cspy.executable import Platon
from cspy.util.path import TemporaryDirectory, Path
from cspy.formats.hklf4 import Hklf4
import numpy as np
import logging
from io import StringIO

LOG = logging.getLogger(__name__)
DEFAULT_SEPARATION = 0.02
DEFAULT_TT_RANGE = (0, 20)


class StructureFactors:
    def __init__(self, hkl, intensities, **kwargs):
        self.hkl = hkl
        self.intensities = intensities
        self.kwargs = kwargs

    @property
    def f2(self):
        return np.sum(self.intensities ** 2, axis=1)

    def q_mag(self, crystal):
        return np.linalg.norm(
            np.dot(self.hkl, crystal.unit_cell.reciprocal_lattice), axis=1
        )

    @classmethod
    def from_file(cls, filename):
        """
        Calculates the HKLF values using platon
        filename(string): path of cif file
        If successful returns numpy array, else None
        """
        if not filename.endswith(".cif"):
            LOG.warn("Please use a cif file when calculating powder patterns!")

        stdin = "ASYM GENERATE"
        exe = Platon(
            filename,
            ["-o"],
            stdin=stdin,
            working_directory=Path(filename).parent.absolute(),
        )
        exe.timeout = 10.0
        exe.run()
        hklf_contents = getattr(exe, "hklf_contents", None)
        if hklf_contents is None:
            print(exe.output_contents)
            return None
        hklf = Hklf4(hklf_contents)
        return cls(hklf.hkl, hklf.intensities)

    @classmethod
    def from_cif_string(cls, contents, **kwargs):
        with TemporaryDirectory() as tempdir:
            res_file = Path(tempdir, "tmp.cif")
            res_file.write_text(contents)
            pattern = cls.from_file(str(res_file), **kwargs)
        return pattern

    def __repr__(self):
        return f"<StructureFactors: n={len(self.hkl)}>"
