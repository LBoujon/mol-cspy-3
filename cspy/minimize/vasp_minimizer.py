import logging
import os
from tempfile import TemporaryDirectory
from os.path import exists
from distutils.dir_util import copy_tree
from typing import Union

from cspy.crystal import Crystal
from cspy.executable.vasp import Vasp
from cspy.executable import ReturnCodeError, TimeoutExpired
from cspy.formats.vasp_input import create_vasp_inputs
from cspy.formats.vasp_output import VaspOutput
from cspy.util.constants import EV2KJ_PER_MOL
from cspy.util.timer import Timer
from distutils.dir_util import copy_tree
from cspy.crystal.util import find_formula_unit


LOG = logging.getLogger(__name__)

class VaspMinimizer:
    """VaspMinimizer is a class that handles VASP calculations."""
    def __init__(self, **kwargs):
        LOG.debug("Initializing VaspMinimizer.")
        self.kwargs = kwargs
        self.timeout = self.kwargs.get("timeout")
        self.keepoutputs = self.kwargs.get("keepoutputs")
        self.potcar_path = self.kwargs.get("potcar_path")
        self.mpi_cores = self.kwargs['mpi_settings'].get("cores_per_structure")
        self.last_log_contents = None
        self.job_type = (
            "minimize" if int(self.kwargs['incar_settings'].get('NSW')) > 0 
            else "single_point"
        )
        self.save_forces = kwargs.get('save_forces', False)
        LOG.info('job type: {}'.format(self.job_type))

        if exists(self.potcar_path):
            LOG.debug("Found potcar parameter set in %s", self.potcar_path)
        else:
            LOG.error("No potcar parameters found in %s", self.potcar_path)
            raise RuntimeError("Missing potcar parameters.")

    def create_inputs(self, crystal, working_directory) -> None:
        create_vasp_inputs(
            input_crystal=crystal,
            working_directory=working_directory,
            settings=self.kwargs,
        )

    def run_calculation(
        self,
        working_directory: str,
        timeout: float,
        mpi_cores: int
    ) -> Vasp:
        """
        Run the VASP calculation.

        Args:
            working_directory (str): Directory where the VASP calculation will be run.
            timeout (float): Timeout for the VASP calculation.
            mpi_cores (int): Number of MPI cores to use for the calculation.

        Returns:
            Vasp: An instance of the Vasp class containing the results of the calculation.
        """
        LOG.debug(
            "Running VASP calculation in %s with timeout %.2f and %d MPI cores",
            working_directory, timeout, mpi_cores
        )
        exe = Vasp(
            working_directory=working_directory,
            timeout=timeout,
            mpi_cores=mpi_cores,
        )
        timing = Timer()
        with timing:
            return_code = exe.run()
        success = exe.output_contents is not None
        LOG.debug(
            "VASP calculation %s in %.2fs",
            "succeeded" if success else "failed",
            timing.elapsed,
        )
        if not success and exe.output_contents is not None:
            LOG.debug(
                "Last 20 lines in VASP output: %s\n",
                "\n".join(exe.output_contents.splitlines()[-20:]),
            )
            if hasattr(exe, "input_file") and exe.input_file is not None:
                LOG.debug("VASP input:\n%s", exe.input_file.read())
        return exe

    def minimize(
        self,
        crystal: Crystal,
        completed_calcs_folder: str = "completed_calcs",
    ) -> Union[Crystal, None]:
        """
        Perform a VASP minimization on the given crystal structure.

        Args:
            crystal (Crystal): The crystal structure to minimize.
            completed_calcs_folder (str): Folder to save completed calculations.

        Returns:
            Crystal: The minimized crystal structure, or None if the calculation failed.
        """
        name_ = crystal.titl
        LOG.debug("Performing VASP calculation on %s", crystal.titl)
        formula_unit, unique_molecules, Z = find_formula_unit(crystal.unit_cell_molecules())
        timing = Timer()
        with timing:
            with TemporaryDirectory(prefix=f"{os.getcwd()}/") as tmpdirname:
                crystal.save(f"{tmpdirname}/{name_}.res")
                self.create_inputs(
                    crystal,
                    working_directory=tmpdirname,
                )
                try:
                    LOG.debug("Starting VASP minimization in %s", tmpdirname)
                    d = self.run_calculation(
                        working_directory=tmpdirname,
                        timeout=self.timeout,
                        mpi_cores=self.mpi_cores,
                    )
                    if self.keepoutputs:
                        if not os.path.exists(completed_calcs_folder):
                            os.makedirs(completed_calcs_folder)
                        copy_tree(
                            tmpdirname,
                            f"{completed_calcs_folder}/{name_}",
                        )
                except (ReturnCodeError, TimeoutExpired) as exc:
                    LOG.exception("Error in VASP: %s", exc)
                    return None
                except Exception as exc:
                    LOG.exception("Unknown error in VASP: %s", exc)
                    return None

                # post-process
                self.last_log_contents = d.output_contents
                self.last_input_contents = d.input_contents
                self._job_output = VaspOutput(self.last_log_contents)
                if not self._job_output._valid(self.job_type):
                    LOG.debug("Invalid calculation.")
                    LOG.debug("Output contents:\n%s", self.last_log_contents)
                    return None
                self._job_output_properties = self._job_output._parse_file()
                if self.job_type == "single_point":
                    new_crystal = Crystal.from_CONTCAR(d.output_geometry)
                    new_crystal.properties["energy_eV"] = float(
                        self._job_output_properties["initial_energy"]
                    )
                    new_crystal.properties["energy"] = (
                        float(self._job_output_properties["initial_energy"])
                        * EV2KJ_PER_MOL
                    ) / Z
                    # we will be looking for this property later
                    new_crystal.properties["final_energy"] = new_crystal.properties[
                        "energy"
                    ]
                elif self.job_type == "minimize":
                    new_crystal = Crystal.from_CONTCAR(d.output_geometry)
                    new_crystal.properties["initial_energy"] = (
                        float(self._job_output_properties["initial_energy"])
                        * EV2KJ_PER_MOL
                    ) / Z
                    new_crystal.properties["final_energy"] = (
                        float(self._job_output_properties["final_energy"])
                        * EV2KJ_PER_MOL
                    ) / Z
                    new_crystal.properties["density"] = float(crystal.density)
                else:
                    LOG.error("Unknown job type: %s", self.job_type)
                    return None

                if self.save_forces:
                    new_crystal.properties["atom_forces"] = (
                        self._job_output._parse_forces()
                    )

        LOG.info(
            'Minimization of "%s" complete in %.2fs',
            crystal.titl,
            timing.elapsed,
        )

        return new_crystal

    def __call__(self, crystal: Crystal) -> Crystal:
        return self.minimize(crystal)

    @property
    def log_summary(self) -> str:
        return self.last_log_contents

    @property
    def input_summary(self) -> str:
        return self.last_input_contents


