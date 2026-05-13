import copy
from .executable import AbstractExecutable, ReturnCodeError
from os import environ
import logging
from cspy.util.path import Path
from .locations import which


DFTB_EXEC = which("dftb+")
LOG = logging.getLogger("dftb")

class Dftb(AbstractExecutable):
    _executable_location = DFTB_EXEC
    _OUT_LOGFILE_FMT = "{}_dftbplus.out"
    _OUT_GEOM_FMT = "{}.gen"

    def __init__(self, 
                 input_hsd,
                 input_coords,
                 crys_name="crys",
                 working_directory=".",
                 output_prefix="geom.out",
                 timeout=1200.0,
                 **kwargs):
        
        self.timeout = timeout 
        self.crys_name = crys_name
        self.input_hsd = input_hsd
        self.input_coords = input_coords
        self.working_directory = working_directory
        self.output_prefix = output_prefix
        self.output_contents = None
        self.log_contents = None
        self.opt_coords = None
        self.error_contents = None
        
    @property
    def input_file(self):
        return Path(self.working_directory, self.input_hsd)
    
    @property
    def input_geometry(self):
        return Path(self.working_directory, self.input_coords)
    
    @property
    def output_file(self):
        return Path(self.working_directory, self._OUT_LOGFILE_FMT.format(self.crys_name))
    
    @property
    def output_geometry(self):
        return Path(self.working_directory, self._OUT_GEOM_FMT.format(self.output_prefix)) 
        
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
            self._run_raw_stdin()
        except ReturnCodeError as e:
            LOG.error("Dftb+ failed: %s", e)
            self.post_process()
            LOG.debug("Output log: %s", self.output_contents)
            LOG.debug("Input file: %s", self.input_contents)
            LOG.debug("Input geometry: %s", self.input_geometry_contents)
            raise e
            

