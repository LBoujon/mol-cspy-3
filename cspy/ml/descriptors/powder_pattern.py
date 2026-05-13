from cspy.executable import Platon
from cspy.util.path import TemporaryDirectory, Path
from cspy.formats.powder import PowderSietronics, PowderAscii
from subprocess import TimeoutExpired
import numpy as np
import logging

LOG = logging.getLogger(__name__)
#LOG.setLevel(logging.DEBUG)
DEFAULT_SEPARATION = 0.02
DEFAULT_TT_RANGE = (0, 20)


class PowderPattern:
    def __init__(
        self, data, two_theta_range=DEFAULT_TT_RANGE, separation=DEFAULT_SEPARATION
    ):
        self.pattern = np.array(data)
        self.tt_range = two_theta_range
        self.separation = separation
        self.nbins = len(self.pattern)
        assert self._is_valid()

    def _is_valid(self):
        """
        Checks an xrd pattern is a non-empty numpy array.
        If it is returns True, else False.
        """
        return (
            (self.pattern is not None)
            and (isinstance(self.pattern, np.ndarray))
            and (len(self.pattern) > 0)
        )

    @classmethod
    def from_file(
        cls,
        filename,
        two_theta_range=DEFAULT_TT_RANGE,
        separation=DEFAULT_SEPARATION,
        working_directory=None,
        wavelength=None
    ):
        """
        Calculates the powder pattern using platon
        filename(string): path of cif file

        If successful returns a PowderPattern object, else None
        """
        if not filename.endswith(".cif"):
            LOG.warn("Please use a cif file when calculating powder patterns!")

        if wavelength:
            if wavelength in ["Cu","Mo","Ga","Ag","In"]:
                LOG.info("Computing powder pattern from file %s using supplied wavelength %s",filename,wavelength)
            else:
                LOG.error("Specified wavelength argument not available in Platon; try providing the value in the CIF file")
                return None

        tt_max = two_theta_range[1]
        args = ["-o", "-Q{:d}".format(tt_max)]
        if wavelength:
            args += ["-{:s}".format(wavelength)]
        if working_directory is None:
            working_directory = Path(filename).parent.absolute()
        exe = Platon(filename, args, working_directory=working_directory)
        exe.timeout = 10.0
        try:
            exe.run()
        except TimeoutExpired:
            LOG.error("Time out in Platon")
            return None
        cpi_contents = getattr(exe, "cpi_contents", None)
        if cpi_contents is None:
            LOG.error("Empty cpi contents in Platon")
            LOG.error("Output contents\n%s", exe.output_contents)
            return None
        x = PowderSietronics(cpi_contents)
        return PowderPattern(
            x.intensities, two_theta_range=two_theta_range, separation=separation
        )

    @classmethod
    def from_cif_string(cls, contents, **kwargs):
        with TemporaryDirectory() as tempdir:
            cif_file = Path(tempdir, "tmp.cif")
            cif_file.write_text(contents)
            pattern = cls.from_file(str(cif_file), working_directory=tempdir, **kwargs)
        return pattern

    @classmethod
    def from_raw_data_file(cls, 
        filename,
        zero_negatives=False):
       # get these from data itself
       # two_theta_range=DEFAULT_TT_RANGE,
       # separation=DEFAULT_SEPARATION):
        LOG.info("Reading a powder XRD pattern from %s",filename)
        try:
            if filename.endswith(".cpi"):  # assume it's Sietronics format
                LOG.debug("Reading raw Sietronics CPI data from %s",filename)
                x = PowderSietronics.from_file(filename)
                return PowderPattern(
                x.intensities
                )
            else: # assume it's raw ASCII
                LOG.debug("Reading raw ASCII data from %s",filename)
                with open(filename,"r") as f:
                    contents = f.read()
                lines = contents.split("\n")
                raw_data = np.array([[float(d) for d in l.strip().split()] for l in lines if l]) # if takes care of any blank lines
                raw_thetas = raw_data[:,0]
                raw_intensities = raw_data[:,1]
                if zero_negatives:
                    LOG.debug("Zeroing out negative intensities before plotting raw data file")
                    np.clip(raw_intensities,0.0,None,out=raw_intensities)
                two_theta_range=(raw_thetas[0],raw_thetas[-1])
                bins = len(raw_thetas)
                xrd = np.histogram(
                            raw_thetas, weights=raw_intensities, bins=bins, density=False
                        )
                new_intensities, new_two_theta = xrd
                # We will assume that the raw data has consistent intervals between points
                separation=raw_thetas[1]-raw_thetas[0]
                #new_two_theta = 0.5 * (new_two_theta[1:] + self.two_theta[:-1])
                return PowderPattern(
                new_intensities, two_theta_range=two_theta_range, separation=separation
                )
        except:
            LOG.error("Unable to read PXRD from %s",filename)
            LOG.error("Possibly an unrecognised format? Recognised formats are .cpi (Sietronics) and ASCII")
            raise

    def get_subset_of_pattern(self, tt_subrange):
        LOG.debug("Getting subset of pattern using 2theta subrange: %s",tt_subrange)
        # First get the full list of 2theta values from the original pattern
        # Only 2theta range and nbins are defined by pattern data;
        # separation is an independent specified value, so ignore it
        all_tts = np.linspace(self.tt_range[0],self.tt_range[1],self.nbins)

        # Filter 2thetas and intensities based on whether index corresponds to a value in the subrange
        # numpy arrays allow selection based on conditionals as follows:
        new_tts = all_tts[
                          (all_tts >= tt_subrange[0])
                          & (all_tts < tt_subrange[1])
                          ]
        new_intens = self.pattern[
                                 (all_tts >= tt_subrange[0])
                                 & (all_tts < tt_subrange[1])
                                 ]
        # returns new_tts as well to save working them out twice
        return new_tts, new_intens
