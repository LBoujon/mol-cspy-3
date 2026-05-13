from cspy import Crystal, Molecule
from cspy.executable.xtb import Xtb
from cspy.executable import ReturnCodeError, TimeoutExpired
from os.path import exists
from cspy.util.path import Path
from cspy.configuration import CONFIG
import os
import logging
import time
from tempfile import TemporaryDirectory
import re
from cspy.util.constants import HA2KJ_PER_MOL

LOG = logging.getLogger(__name__)
NUMERIC_CONST_PATTERN = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
energy_regex = re.compile(r"total\s+energy\s+({})\s*Eh".format(NUMERIC_CONST_PATTERN))


def find_energies(stdout):
    matches = energy_regex.findall(stdout)
    if not matches:
        return np.nan, np.nan
    return float(matches[0]), float(matches[-1])


class XtbMinimizer:

    xtb_param_fmt = "param_gfn{}-xtb.txt"

    def __init__(self, gfn=CONFIG.get("xtb.gfn", 0), name="Unknown", **kwargs):
        LOG.debug("Initializing XtbMinimizer: GFN%s-XTB", gfn)
        xtb_home = os.environ.get("XTBHOME", CONFIG.get("xtb.home", os.environ.get("HOME")))
        xtb_path = os.environ.get("XTBPATH", xtb_home)
        self.maxcycle = kwargs.get("maxcycle", CONFIG.get("xtb.maxcycle", 1000))
        self.gfn = gfn
        gfnstr =  str(gfn)
        xtb_param_file = self.xtb_param_fmt.format(gfnstr)
        param_file_loc = os.path.join(xtb_path, xtb_param_file)
        home_param = exists(param_file_loc)
        if not home_param:
            LOG.error("No parameter data for GFN%s-XTB, will likely fail", self.gfn)
            raise RuntimeError(f"Missing parameter file for Xtb: {xtb_param_file}")
        self.name = name
        self.kwargs = kwargs
        self.last_output_contents = None
        self.last_log_contents = None

    def single_point_crystal(self, crystal, **kwargs):
        input_contents = crystal.to_turbomole_string() + "$end"
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        if self.gfn > 1:
            LOG.error(
                "Currently GFN%s-XTB is unsupported for periodic systems", self.gfn
            )
            raise ValueError("Must use GFN0-XTB or GFN-XTB (i.e. 0 or 1)")
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                opt=False,
                **self.kwargs,
            )
            self.last_input_contents = input_contents
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
                crystal.properties["lattice_energy"] = final
        return result

    def minimize_crystal(self, crystal, engine="inertial"):
        input_contents = crystal.to_turbomole_string()
        input_contents += (
            "$opt\n"
            "engine={engine}\n"
            "maxcycle={maxcycle}\n"
            "$end".format(engine=engine, maxcycle=self.maxcycle)
        )
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        if self.gfn > 1:
            LOG.error(
                "Currently GFN%s-XTB is unsupported for periodic systems", self.gfn
            )
            raise ValueError("Must use GFN0-XTB or GFN-XTB (i.e. 0 or 1)")
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            self.last_input_contents = input_contents
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
            if exe.opt and success:
                result = Crystal.from_turbomole_string(exe.opt_coord_contents)
                result.properties["lattice_energy"] = final
                result.properties["density"] = result.density
        return result

    def minimize_molecule(self, 
                          molecule, 
                          engine="rf", 
                          constraints=CONFIG.get("xtb.constraints_file", None)
                         ):
        input_contents = molecule.to_turbomole_string()
        input_contents += (
            "$opt\n"
            "engine={engine}\n"
            "maxcycle={maxcycle}\n"
            "$end".format(engine=engine, maxcycle=self.maxcycle)
        )
        if constraints:
            input_contents += ("\n$constrain")
            with open(constraints, "r") as f:     #JG - needs re-writing
                for line in f:
                    input_contents += ("\n{}".format(line.split("\n")[0]))
            input_contents += ("\n$end")
        # LOG.debug("Input contents:\n%s", input_contents)
        result = None
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            #self.last_output_contents = exe.output_contents
            #self.last_log_contents = exe.opt_log_contents
            #self.last_log_contents = exe.opt_coord_content #check this maps in xtb executable py
            self.last_log_contents = exe.output_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
                LOG.debug("Energy change %.5f -> %.5f Eh", init, final)
            if exe.opt and success:
                result = Molecule.from_turbomole_string(exe.opt_coord_contents)
                result.properties["scf_energy"] = final * HA2KJ_PER_MOL
        return result

    def minimize(self, obj, **kwargs):
        from cspy.flex.flex_molecule import FlexMolecule
        if isinstance(obj, Crystal):
            return self.minimize_crystal(obj, **kwargs)
        elif isinstance(obj, Molecule) or isinstance(obj, FlexMolecule):
            return self.minimize_molecule(obj, **kwargs)
        else:
            raise NotImplementedError(
                f"XtbMinimizer only implemented for Crystal, Molecule types not {obj.__class__.__name__}"
            )

    @property
    def log_summary(self):
        return self.last_log_contents

    def __call__(self, obj, **kwargs):
        return self.minimize(obj)
