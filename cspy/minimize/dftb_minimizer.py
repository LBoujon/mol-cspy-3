from cspy.crystal import Crystal
from cspy.executable.dftb import Dftb
from cspy.executable import ReturnCodeError, TimeoutExpired
import os
import logging
from tempfile import TemporaryDirectory
from os.path import exists
from cspy.formats.dftb_input import write_dftb_inputs
from cspy.formats.dftb_output import DftbOutput
from cspy.util.timer import Timer

LOG = logging.getLogger(__name__)

class DftbMinimizer:
    
    def __init__(self, 
                 mol_calc=None,
                 **kwargs):
        LOG.debug("Initializing DftbMinimizer.")    
        self.mol_calc = mol_calc
        self.kwargs = kwargs
        self.skf_set = self.kwargs.get("skf_set")
        self.skf_path = self.kwargs.get("skf_path")
        self.output_prefix = self.kwargs.get("output_prefix")
        self.single_point = self.kwargs.get("single_point")
        self.lattice_opt = self.kwargs.get("lattice_opt")
        self.timeout = self.kwargs.get("timeout")
        self.last_log_contents = None
            
        if self.skf_set != "3ob-3-1":
            LOG.error("%s parameter set not yet implemented for use in cspy DFTB minimisation.",
                      self.skf_set)
            raise NotImplementedError
              
        if self.skf_path is not None:
            param_path = os.path.join(self.skf_path, self.skf_set)
            path_check = exists(param_path)
            if not path_check:
                LOG.error("No %s parameters found.", self.skf_set)
                raise RuntimeError("Missing %s parameters.", self.skf_set)
            LOG.debug("Found %s parameter set: %s", self.skf_set, param_path)
    
        else:
            LOG.error("No slater-koster file path has been provided.")
            raise RuntimeError("Config is missing a path to the slater-koster parameters.")
            
    

    def create_inputs(self, structure, working_directory):
        write_dftb_inputs(structure, 
                          working_directory=working_directory,
                          settings=self.kwargs
                          )
        
       
    def run_calculation(self, name, working_directory, output_prefix, timeout):
        exe = Dftb(
            input_hsd="dftb_in.hsd",
            input_coords="geometry.gen",
            crys_name=name,
            working_directory=working_directory,
            output_prefix=output_prefix,
            timeout=timeout,
            )
        timing = Timer()
        with timing:
            return_code = exe.run()
        success = exe.output_contents is not None
        LOG.debug(
            "DFTB+ calculation %s in %.2fs",
            "succeeded" if success else "failed",
            timing.elapsed,
        )       
        if not success:
            LOG.debug(
                "Last 20 lines in DFTB output: %s\n",
                "\n".join(exe.output_contents.splitlines()[-20:]),
            )
            LOG.debug("DFTB input:\n%s", exe.hsd_file.read())
        return exe
        
    def minimize(self, structure):
        name = structure.molecular_formula if self.mol_calc else structure.titl
        if self.single_point:
            LOG.debug("Performing single point energy calculation on %s", name)
        elif self.lattice_opt:
            LOG.debug("Performing full lattice minimization on %s",  name)
        else:
            LOG.debug("Performing fixed lattice minimization on %s",  name)    
        timing = Timer()
        with timing:
            with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
                self.create_inputs(structure, 
                                   working_directory=tmpdirname,
                                   )      
                try:
                    LOG.debug("Starting dftb minimization in %s", tmpdirname)
                    d = self.run_calculation(name=name,
                                               working_directory=tmpdirname,
                                               output_prefix=self.output_prefix,
                                               timeout=self.timeout,
                                               )
                except (ReturnCodeError, TimeoutExpired) as exc:
                    LOG.exception("Error in DFTB+: %s", exc)
                    return None
                except Exception as exc:
                    LOG.exception("Unknown error in DFTB+: %s", exc)
                    return None
    
                #post-process
                self.last_log_contents = d.output_contents
                #LOG.debug(self.last_log_contents)
                self.last_input_contents = d.input_contents
                self._job_output = DftbOutput(self.last_log_contents)    
                if not self.single_point and not self._job_output._valid():
                    LOG.debug("Invalid calculation.")
                    LOG.debug("Output contents:\n%s", self.last_log_contents)
                    return None
                
                self._job_output_properties = self._job_output._parse_file()
                structure.properties["energy"] = self._job_output_properties['initial_energy']    
                if self.single_point and not self.mol_calc:
                     structure.properties["final_energy"] = (float(self._job_output_properties['final_energy'])*2625.5002)/len(structure.unit_cell_molecules())
                    
                if not self.single_point:
                    if self.mol_calc:
                        from cspy.chem.molecule import Molecule
                        new_structure = Molecule.from_gen_file(d.output_geometry)
                        new_structure.properties["initial_energy"] = (float(self._job_output_properties['initial_energy'])*2625.5002)
                        new_structure.properties["final_energy"] = (float(self._job_output_properties['final_energy'])*2625.5002) 
                    else:
                        new_structure = Crystal.from_gen_file(d.output_geometry)
                        new_structure.properties["initial_energy"] = (float(self._job_output_properties['initial_energy'])*2625.5002)/len(new_structure.unit_cell_molecules())
                        new_structure.properties["final_energy"] = (float(self._job_output_properties['final_energy'])*2625.5002)/len(new_structure.unit_cell_molecules()) 
                        new_structure.properties["density"] = float(new_structure.density)
            
        LOG.info(
        'Minimization of "%s" complete in %.2fs', name, timing.elapsed
        )
    
        return structure if self.single_point else new_structure
                           

    def __call__(self, crystal):
        return self.minimize(crystal)

    @property
    def log_summary(self):
        return self.last_log_contents

    @property
    def input_summary(self):
        return self.last_input_contents

