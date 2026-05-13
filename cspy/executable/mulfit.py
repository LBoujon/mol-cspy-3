from .executable import AbstractExecutable
from cspy.executable.locations import MULFIT_EXEC
import logging
import os
from os import remove
from os.path import basename, exists, join
from cspy.formats.mulfit import MulfitInput, MulfitOutput

LOG = logging.getLogger("cspy.mulfit")


class MulfitException(Exception):
    pass


class Mulfit(AbstractExecutable):

    _executable_location = MULFIT_EXEC
    _INPUT_FILE_FMT = "{}.mulfit_in"
    _OUTPUT_FILE_FMT = "{}.mulfit_out"

    def __init__(self, mults, **kwargs):
        """
        Parameters
        ----------
        mults : :obj:`DistributedMultipoles`
            Multipoles to fit with mulfit

        Keyword Args
        ------------
        working_directory : str, optional
            directory in which to run the MULFIT executable

        Other kwargs will be parsed to :obj:`~cspy.formats.mulfit.MulfitInput`
        """
        # TODO update using defaults in configuration
        self._working_directory = kwargs.pop("working_directory", os.getcwd())
        self.name = kwargs.get("title")
        self.mults = mults
        self.kwargs = kwargs

    @property
    def input_file(self):
        return self._INPUT_FILE_FMT.format(self.name)

    @property
    def output_file(self):
        return self._OUTPUT_FILE_FMT.format(self.name)

    def write_inputs(self):
        inp = MulfitInput(self.mults, **self.kwargs)
        inp.write_dma_file()
        LOG.debug("Writing mulfit input file to %s", os.path.abspath(self.input_file))
        with open(self.input_file, "w") as f:
            f.write(str(inp))

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def result(self):
        if not self.multipoles:
            self.multipoles = DistributedMultipoles.from_dma_string(self.punch_contents)
        return self.multipoles

    def post_process(self):
        LOG.debug(
            "Reading mulfit output file from %s", os.path.abspath(self.output_file)
        )
        with open(self.output_file) as f:
            self.output_contents = f.read()

        for fname in (self.input_file, self.output_file):
            if exists(fname):
                LOG.debug("Mulfit clean up: %s", fname)
                remove(fname)

    def run(self, *args, **kwargs):
        self.kwargs.update(kwargs)
        self._run_raw_stdin()
        return MulfitOutput(self.output_contents)
