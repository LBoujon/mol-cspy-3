from cspy.executable.executable import AbstractExecutable, ReturnCodeError
import logging
from os import remove, environ
from os.path import basename, exists, join
from cspy.util.path import working_directory, Path
from cspy.executable.locations import which
from cspy.configuration import CONFIG
import shutil
import copy
from cspy.chem import Element
from tempfile import TemporaryFile
from typing import Tuple, Union, Optional

GULP_EXEC = which("gulp")
LOG = logging.getLogger("gulp")


class Gulp(AbstractExecutable):

    _input_file = "gulp.gin"
    _output_file = "gulp.gout"
    _executable_location = GULP_EXEC
    _timeout = CONFIG.get("gulp.timeout", 60.0)

    def __init__(self, input_contents : str, *args : Tuple, working_directory : Optional[str] = ".", **kwargs):
        """Gulp exe class for running a GULP executable.

        Parameters
        ----------
        input_contents : string
            contents that will be written into GULP input file (.gin)
        working_directory : string
            which directory to run calculations/generate files in

        """
        self._timeout = kwargs.get("timeout", self._timeout)
        self.threads = kwargs.get("threads", 1)
        self.input_contents = input_contents
        self.output_contents = None
        self.cif_contents = None
        self.kwargs = kwargs.copy()
        self.working_directory = working_directory
        LOG.debug(
            "Initializing GULP calculation timeout: %ss",
            self.timeout,
        )
        self.error_contents = None

    @property
    def input_file(self):
        return join(self.working_directory, self._input_file)

    @property
    def output_file(self):
        return join(self.working_directory, self._output_file)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        LOG.debug("Writing input file to %s", self.input_file)
        with open(self.input_file, "w") as f:
            f.write(self.input_contents)

    def result(self):
        return self.output_contents

    def post_process(self):
        with open(self.output_file) as f:
            self.output_contents = f.read()

    def run(self, *args, **kwargs):
        try:
            self._run_raw_stdin()
        except ReturnCodeError as e:
            LOG.error("GULP failed: %s", e)
            self.post_process()
            raise e

if __name__ == "__main__":
    import sys
    logging.basicConfig(level="DEBUG")
    input_contents = Path(sys.argv[1]).read_text()
    gulp = Gulp(input_contents)
    gulp.run()