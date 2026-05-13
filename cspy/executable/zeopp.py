from .executable import AbstractExecutable, ReturnCodeError
from cspy.executable.locations import ZEOXX_EXEC
from cspy.util.path import Path
from os.path import exists, join
from os import remove
import logging
import os
from collections import defaultdict

LOG = logging.getLogger(__name__)


class Zeopp(AbstractExecutable):
    """VERY RUDIMENTARY ZEO++ SUPPORT, needs to be rewritten well"""

    _executable_location = ZEOXX_EXEC

    def __init__(self, filename="test.cif", working_directory="."):
        self.channel_radius = 1.7  # methane probe radius
        self.probe_radius = 1.7
        self.nsamples = 2000
        self.filename = filename
        self._working_directory = working_directory

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        pass

    def result(self):
        return self.returncode

    def post_process(self):
        self.stdout_contents = Path(self.output_file).read_text()
        self.sa_contents = Path(self.sa_file).read_text()

    def run_surface_area(
        self, filename, channel_radius=1.7, probe_radius=1.7, nsamples=2000
    ):
        self._output_file = Path(
            self.working_directory, Path(filename).stem + ".stdout"
        )
        self.sa_file = Path(self.working_directory, Path(filename).stem + ".sa")
        try:
            args = [
                "-ha",
                "-sa",
                str(channel_radius),
                str(probe_radius),
                str(nsamples),
                filename,
            ]
            self._run_raw(*args)
        except ReturnCodeError as e:
            LOG.error("zeo++ failed: %s", e)
            raise e

    def run(self):
        self.run_surface_area(
            self.filename,
            channel_radius=self.channel_radius,
            probe_radius=self.probe_radius,
        )
