from cspy import Crystal, Molecule
from cspy.executable.gaussian import Gaussian
from cspy.executable import ReturnCodeError, TimeoutExpired
from os.path import join, exists
import copy
from cspy.util.path import Path
import os
import logging
import time
from tempfile import TemporaryDirectory
from cspy.templates import get_template

LOG = logging.getLogger(__name__)
G09_SCF_TEMPLATE = get_template("gaussian_scf")


def find_energies(stdout):
    matches = energy_regex.findall(stdout)
    if not matches:
        return np.nan, np.nan
    return float(matches[0]), float(matches[-1])


class GaussianMinimizer:
    def __init__(self, method="B3LYP", basis_set="6-311G**", **kwargs):
        LOG.debug("Initializing GaussianMinimizer: %s/%s", method, basis_set)
        self.link0_commands = {}
        self.additional_sections = []
        if "threads" in kwargs:
            self.link0_commands["nprocshared"] = kwargs.pop("threads")
        if "memory" in kwargs:
            self.link0_commands["mem"] = kwargs.pop("memory")
        if "additional_sections" in kwargs:
            self.additional_sections = kwargs.pop("additional_sections")
        self.name = kwargs.get("name", "gaussian_opt")
        self.kwargs = kwargs
        self.last_log_contents = None
        self.last_fchk_contents = None
        self.opt = kwargs.get("opt", True)
        self.route_commands = kwargs.get("route_commands", "opt" if self.opt else "")
        if method.lower() == "mp2":
            method = "mp2=semidirect"
            self.route_commands += " density=current"
        self.method = method
        self.basis_set = basis_set

    def generate_input_file(self, molecule, **kwargs):
        name = kwargs.get("name", molecule.molecular_formula)
        link0 = copy.deepcopy(self.link0_commands)
        link0.update({"chk": "{}.chk".format(name)})
        input_file = G09_SCF_TEMPLATE.render(
            link0=link0,
            title=name + self.name,
            method=self.method,
            basis=self.basis_set,
            charge=kwargs.get("charge", 0),
            multiplicity=kwargs.get("multiplicity", 1),
            geometry=molecule.to_xyz_string(header=False),
            route_commands=self.route_commands,
            additional_sections=self.additional_sections
        )
        return input_file

    def generate_fchk(self, molecule, **kwargs):
        mol = self.minimize_molecule(molecule, **kwargs)
        return self.last_fchk_contents

    def minimize_molecule(self, molecule, **kwargs):
        name = kwargs.get("name", molecule.molecular_formula)
        log_file = kwargs.get("log_file", "")
        fchk_file = kwargs.get("fchk_file", "")

        LOG.info(
            "%s molecule %s with Gaussian",
            "Optimizing" if self.opt else "Single point energy for",
            molecule,
        )
        input_file = self.generate_input_file(molecule, **kwargs)

        with TemporaryDirectory() as tmpdirname:
            g09 = Gaussian(
                input_file,
                output_file=name + ".log",
                run_formchk=name + ".chk",
                working_directory=tmpdirname,
            )
            try:
                returnode = g09.run()
            except ReturnCodeError as e:
                returncode = g09.returncode
                LOG.error("Error running gaussian input:\n%s", input_file)
                raise RuntimeError(
                    f"Gaussian exited with return code: {returncode}"
                ) from e

        self.last_log_contents = g09.log_contents
        self.last_fchk_contents = g09.fchk_contents

        if log_file:
            Path(log_file).write_text(g09.log_contents)
        if fchk_file:
            Path(fchk_file).write_text(g09.fchk_contents)

        return Molecule.from_gaussian_optimization_string(self.last_log_contents)

    def minimize(self, obj, **kwargs):
        if isinstance(obj, Molecule):
            return self.minimize_molecule(obj, **kwargs)
        else:
            raise NotImplementedError(
                "GaussianMinimizer only implemented for Molecule objects"
            )

    def __call__(self, obj, **kwargs):
        return self.minimize(obj, **kwargs)

    def __repr__(self):
        link0 = ",".join(f"{k}={v}" for k, v in self.link0_commands.items())
        if link0:
            link0 = f" ({link0})"
        return f"<GaussianMinimizer: {self.method}/{self.basis_set}{link0}>"
