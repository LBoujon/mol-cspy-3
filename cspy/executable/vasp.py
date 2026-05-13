import copy
from .executable import AbstractExecutable, ReturnCodeError
from os import environ
import logging
from cspy.util.path import Path
from .locations import which

VASP_EXEC = which("vasp_std")
LOG = logging.getLogger(__name__)


class Vasp(AbstractExecutable):
    _executable_location = VASP_EXEC
    def __init__(self,
                 working_directory=".",
                 timeout=86400,
                 mpi_cores=40,
                 **kwargs):
        self.timeout = timeout
        self.working_directory = working_directory
        self.mpi_cores = mpi_cores
        self.output_contents = None

    @property
    def input_file(self):
        return Path(self.working_directory, "INCAR")

    @property
    def input_geometry(self):
        return Path(self.working_directory, "POSCAR")

    @property
    def output_file(self):
        return Path(self.working_directory, "OUTCAR")

    @property
    def output_geometry(self):
        return Path(self.working_directory, "CONTCAR")

    def post_process(self):
        with open(self.output_file) as f:
            self.output_contents = f.read()
        with open(self.input_file) as f:
            self.input_contents = f.read()
        with open(self.input_geometry) as f:
            self.input_geometry_contents = f.read()

    def result(self):
        return self.output_contents

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        pass

    def run(self, *args, **kwargs):
        env = copy.deepcopy(environ)
        env.update({"OMP_NUM_THREADS": "1"})
        try:
            self._run_raw()
        except ReturnCodeError as e:
            LOG.error("VASP failed: %s", e)
            self.post_process()
            LOG.debug("Output log: %s", self.output_contents)
            LOG.debug("Input file: %s", self.input_contents)
            LOG.debug("Input geometry: %s", self.input_geometry_contents)
            raise e



