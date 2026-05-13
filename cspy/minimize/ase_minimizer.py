import logging
import importlib.util
import sys
import os
from tempfile import TemporaryDirectory
from copy import deepcopy

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.calculators.calculator import Calculator

from cspy import Crystal
from cspy.ase.calculators import custom_calculators, other_models
from cspy.util.timer import Timer
from cspy.util.constants import EV2KJ, KCAL2KJ, NA
from cspy.util.timeout import run_with_timeout, TimeoutException
from cspy.crystal.util import find_formula_unit
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mace.calculators import MACECalculator
    from ase.calculators.dftd3 import PureDFTD3
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

ASE_BUILTIN_CALCULATORS = [
    "TIP3P",
    "TIP4P",
    "Lennard-Jones",
    "Morse",
]

LOG = logging.getLogger(__name__)

class ASEMinimizer:
    def __init__(self, **kwargs):
        """
        Initialize the ASEMinimizer with a specified optimizer and options.

        Args:
            **kwargs:
            fmax (float): Maximum force tolerance after geometry optimization. Default is 0.05.
            steps (int): Maximum number of geometry optimization steps. Default is 1000.
            timeout (int): If set, the maximum time allowed for the minimization in seconds. Defaults to 600 seconds.
            optimizer (str): Optimizer from ase.optimize for geometry optimization. Defaults to 'LBFGS'.
            symmetry (bool): Keep symmetry during optimization. If False, reduce cell to P1.
            calculator_type (str): Type of calculator to use. Defaults to 'mace'.
                Supported values: 'mace', 'ase_builtin', 'custom'.
            calculator (Calculator): If provided, uses this calculator directly.
            mace (dict): Keyword arguments for MACE calculator. See load_mace_calculator().
            ase_builtin (dict): Keyword arguments for ASE built-in calculator. See load_ase_builtin_calculator().
            custom (dict): Keyword arguments for custom calculators.
            model_units (dict): Units of the model used. 

        Example1 (using MACE model of Faraday2024 without dispersion correction):
        ---------
            >>> from cspy.minimize import ASEMinimizer
            >>> mace_settings = {
            ...     'model': '${CSPY_HOME}/cspy/potentials/MACEPots/Faraday2024/Faraday2024_stage1_mace.model',
            ...     'device': 'cpu',
            ...     'dispersion': False,
            ...     'fmax': 0.05,
            ...     'steps': 1000,
            ...     'timeout': 600
            ... }
            >>> minimizer = ASEMinimizer(
            ...     calculator_type='mace',
            ...     mace=mace_settings,
            ...     symmetry=True,
            ... )
            >>> optimized_crystal = my_crystal.minimize_with_mace(minimizer=minimizer)

        Note:
            Since calculator_type is set to 'mace', the mace_settings dictionary
            should be passed via the 'mace' keyword argument, i.e. mace=mace_settings.

        Example2 (using another MACE model from Faraday2024 without dispersion correction):
        ---------
            In cspy.toml file, you can specify the ASEMinimizer settings as follows:
            ------------------------------------
            [ase]
            calculator_type = 'mace'
            symmetry = true
            optimizer = 'BFGS'
            fmax = 0.05
            steps = 100
            timeout = 600

            [ase.model_units]
            energy = 'eV'
            length = 'Ang'
            normalisation = 'cell'
            energy_corr = "none"

            [ase.mace]
            model = '${CSPY_HOME}/cspy/potentials/MACEPots/Faraday2024/Faraday2024_stage2_mace.model'
            dispersion = false
            device = 'cpu'
            --------------------------------------
            Then use the ASEMinimizer as follows:
            >>> from cspy.minimize import ASEMinimizer
            >>> from cspy.configuration import CONFIG
            >>> ase_settings = CONFIG.get("ase", {})
            >>> ase_minimizer = ASEMinimizer(**ase_settings)
            >>> optimized_crystal = ase_minimizer.minimize(my_crystal)

        Example3 (Using mace_mp with D3 dispersion correction):
        ---------
            In cspy.toml file with one of the mol-CSPy apps, you can enable a multi-tiers MACE optimisation via:
            In the first step, the optimizer, fmax, and steps are overridden by [csp_minimization_step.ase].
            In the second step, the optimizer, fmax, and steps default to the values from [ase].
            ------------------------------------
            [[csp_minimization_step]]
            kind = "ase"

            [csp_minimization_step.ase]
            optimizer = 'LBFGS'
            fmax = 0.05
            steps = 1000
            timeout = 600

            [[csp_minimization_step]]
            kind = "ase"

            [ase]
            calculator_type = 'mace'
            symmetry = true
            optimizer = 'BFGS'
            fmax = 0.01
            steps = 100

            [ase.model_units]
            energy = 'eV'
            length = 'Ang'
            normalisation = 'cell'
            energy_corr = "none"

            [ase.mace]
            model = 'mace_mp'
            dispersion = true
            device = 'cpu'


        Example4 (using n2p2 single point after CSP):
        ---------
            In cspy.toml file with one of the mol-CSPy apps, you can enable a n2p2 single point after a CSP via:
            ------------------------------------
            [[csp_minimization_step]]
            kind = "dmacrys"
            electrostatics = "charges"
            PRES = "0.1 GPa"
            opt_scheme = "fast&loose"

            [[csp_minimization_step]]
            kind = "dmacrys"
            electrostatics = "multipoles"
            opt_scheme = "precise"

            [[csp_minimization_step]]
            kind = "ase"

            [csp_minimization_step.ase]
            calculator_type = "n2p2"
            steps = 0

            [csp_minimization_step.ase.n2p2]
            n2p2_model = "Faraday2024"


        Example5 (two steps of ASE minimization using different models/settings):
        ---------
            In cspy.toml file with one of the mol-CSPy apps, you can add two steps of 
            ASE minimization using different models/settings via the below cspy.toml 
            snippet. It uses "mace_mp" without dispersion on GPU in the first step, 
            and "Faraday2024" model with dispersion on CPU in the second step.
            ------------------------------------

            # -- step 1 of minimization --

            [[csp_minimization_step]]
            kind = "ase"

            [csp_minimization_step.ase]
            calculator_type = 'mace'
            symmetry = false
            optimizer = 'BFGS'
            fmax = 0.1
            steps = 100
            timeout = 300

            [csp_minimization_step.ase.model_units]
            energy = 'eV'
            length = 'Ang'
            normalisation = 'cell'
            energy_corr = "none"

            [csp_minimization_step.ase.mace]
            model = 'mace_mp'
            dispersion = false
            device = 'cuda'
            dispersion_cutoff = 22

            # -- step 2 of minimization --

            [[csp_minimization_step]]
            kind = "ase"

            [csp_minimization_step.ase]
            calculator_type = 'mace'
            symmetry = true
            optimizer = 'LBFGS'
            fmax = 0.05
            steps = 100
            timeout = 600

            [csp_minimization_step.ase.model_units]
            energy = 'eV'
            length = 'Ang'
            normalisation = 'cell'
            energy_corr = "none"

            [csp_minimization_step.ase.mace]
            model = '${CSPY_HOME}/cspy/potentials/MACEPots/Faraday2024/Faraday2024_stage2_mace.model'
            dispersion = true
            device = 'cpu'
            dispersion_cutoff = 50        

        """
        LOG.debug("Initializing ASEMinimizer:")
        optimizer_map = {
            "BFGS": "BFGS",
            "LBFGS": "LBFGS",
            "BFGSLineSearch": "BFGSLineSearch",
            "LBFGSLineSearch": "LBFGSLineSearch",
            "GPMin": "GPMin",
            "FIRE": "FIRE"
        }
        step_kwargs = kwargs.get("ase", {})

        LOG.debug("ASEMinimizer ase kwargs: %s", 
                  step_kwargs)

        optimizer = step_kwargs.get("optimizer", kwargs.get("optimizer", "LBFGS"))
        if optimizer in optimizer_map:
            module = importlib.import_module("ase.optimize")
            self.optimizer = getattr(module, optimizer_map[optimizer])
            LOG.debug(
                "Using optimizer: %s in ASEMinimizer.", optimizer_map[optimizer]
            )
        else:
            LOG.error("Unsupported or unrecognised optimizer: %s", optimizer)
            LOG.info(
                "Supported optimizers are: BFGS, LBFGS, BFGSLineSearch, LBFGSLineSearch, GPMin, FIRE."
            )
            raise ValueError("Unsupported or unrecognised optimizer")

        self.symmetry = step_kwargs.get("symmetry", kwargs.get("symmetry", True))
        self.model_units = step_kwargs.get("model_units", kwargs.get("model_units", None))
        
        LOG.debug("Model units are: %s", self.model_units)

        self.fmax = step_kwargs.get("fmax", kwargs.get("fmax", 0.05))
        self.steps = step_kwargs.get("steps", kwargs.get("steps", 1000))
        self.timeout = step_kwargs.get("timeout", kwargs.get("timeout", 600))

        self.calculator_type = step_kwargs.get("calculator_type", kwargs.get("calculator_type", "mace"))
        self.calculator_kwargs = step_kwargs.get(self.calculator_type, kwargs.get(self.calculator_type, {}))
        LOG.debug("Calculator %s kwargs are: %s",
                   self.calculator_type, 
                   self.calculator_kwargs)

        if self.calculator_type == 'mace':
            self.model = self.calculator_kwargs.get("model", "mace_mp")
            LOG.debug(
            "Minimizer is trying to set a(n) %s model: %s...",
            self.calculator_type,
            self.model,
            )
        else:
            self.model = self.calculator_type

        calculator = step_kwargs.get('calculator', kwargs.get('calculator', None))
        if isinstance(calculator, Calculator):
            self.calculator = calculator
        elif self.calculator_type == "mace":
            self.calculator = self.load_mace_calculator(**self.calculator_kwargs)
            LOG.debug("MACE calculator %s was started successfully.", self.model)
        elif self.calculator_type == "ase_builtin":
            self.calculator = self.load_ase_builtin_calculator(**self.calculator_kwargs)
            LOG.debug("Built-in calculator %s was started successfully.", self.model)
        elif self.calculator_type in custom_calculators.keys():
            # check if name of model matches a calculator in ase/calculators
            # if so, load it as a module, and execute setup_custom_calculator
            spec = importlib.util.spec_from_file_location(
                "custom_calculator", custom_calculators[self.model]
            )
            custom_calculator_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_calculator"] = custom_calculator_module
            spec.loader.exec_module(custom_calculator_module)
            self.calculator = custom_calculator_module.setup_custom_calculator(**self.calculator_kwargs)
            self.model_units = custom_calculator_module.calculator_units
        else:
            raise ValueError(
                "Unrecognised model %s for ASE. Cannot initialize calculator.",
                getattr(self, 'model', None)
            )
        
    def load_dftd3_calculator(
        self,
        device: str = "cpu",
        dispersion_xc: str = "pbe",
        damping: str = "bj",
        dispersion_cutoff: float = 50.0,
        default_dtype: str = "float32"
    ) -> Union["PureDFTD3", "TorchDFTD3Calculator"]:
        """Load a D3 dispersion calculator based on the provided device.

        Args:
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
            dispersion_xc (str): Exchange-correlation functional for D3 dispersion. Defaults to 'pbe'.
            damping (str): Damping method for D3 dispersion. Defaults to 'bj'.
            dispersion_cutoff (float): Cutoff distance for D3 dispersion. Defaults to 50.0.
            default_dtype (str): Default data type for calculations. Defaults to 'float32'.
                dtype is only used when device == "cuda".

        Returns:
            PureDFTD3 or TorchDFTD3Calculator: An instance of the D3 dispersion calculator.
        """
        if device == "cuda":
            LOG.debug("Loading TorchDFTD3Calculator for D3 dispersion corrections.")
            gh_url = "https://github.com/pfnet-research/torch-dftd"
            try:
                from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
            except ImportError:
                raise RuntimeError(
                    f"Please install torch-dftd to use dispersion corrections (see {gh_url})"
                )
            import torch
            dtype = torch.float32 if default_dtype == "float32" else torch.float64
            d3_calc = TorchDFTD3Calculator(
                device=device,
                damping=damping,
                dtype=dtype,
                xc=dispersion_xc,
                cutoff=dispersion_cutoff,
            )
        elif device == "cpu":
            LOG.debug("Using ase.calculators.dftd3 for D3 dispersion corrections.")
            import shutil
            if shutil.which("dftd3") is None:
                LOG.error(
                    "'dftd3' executable not found in PATH. "
                    "Please install dftd3 via instructions in "
                    "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/get_dft-d3 "
                    "and ensure 'dftd3' is available."
                )
                sys.exit(1)
            from ase.calculators.dftd3 import PureDFTD3
            from ase.parallel import DummyMPI
            d3_calc = PureDFTD3(
                comm=DummyMPI(),
                xc=dispersion_xc,
                damping=damping,
                cutoff=dispersion_cutoff,
                grad=True
            )
        else:
            raise ValueError(
                f"Device {device} not supported. Please use 'cpu' or 'cuda'."
            )
        LOG.debug(
            "D3 calculator with dispersion_xc=%s, damping=%s, "
            "cutoff=%s, default_dtype=%s was successfully set up.",
            dispersion_xc, damping, dispersion_cutoff, default_dtype
        )
        return d3_calc

    def load_mace_calculator(self, **kwargs) -> "MACECalculator":
        """
        Load a MACE calculator based on the provided model name or file.

        Args:
            **kwargs:
            model (str): Model name or model file. Defaults to 'mace_mp'.
            dispersion (bool): Include D3 correction? Defaults to False.
            dispersion_cutoff (float): Cutoff for D3 correction. Defaults to 50.0.
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
            completed_dir (str): Path for output files. Defaults to current working directory.
            scale_by_element_count (int): Atomic number to scale total energy with.
                Defaults to scale by number of molecules in the cell.

        Returns:
            MACECalculator: An instance of MACECalculator configured with the specified model and parameters.
        """
        import torch
        from ase.calculators.mixing import SumCalculator
        try:
            from mace.calculators import MACECalculator
        except ImportError:
            LOG.error("mace-torch package is not installed. Try installing with 'pip install cspy[nn]'")
            sys.exit(1)

        device = kwargs.get("device", "cpu")
        LOG.debug("Loading MACE model %s on device %s...", self.model, device)
        if device not in ["cpu", "cuda"]:
                raise ValueError(f"Device {device} not supported. Please use 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' or install CUDA.")

        if self.model == "mace_mp":
            try:
                from mace.calculators.foundations_models import mace_mp
                return mace_mp(
                    device=self.calculator_kwargs.get("device", "cpu"),
                    default_dtype=self.calculator_kwargs.get("default_dtype", "float32"),
                    dispersion=self.calculator_kwargs.get("dispersion", False),
                    damping=self.calculator_kwargs.get("damping", "bj"),
                    dispersion_xc=self.calculator_kwargs.get("dispersion_xc", "pbe"),
                    dispersion_cutoff=self.calculator_kwargs.get("dispersion_cutoff", 50.0),
                )
            except ImportError as e:
                raise ImportError(
                    "Loading by mace.calculators.foundations_models import mace_mp failed. "
                    "Try to load MACE model by filename"
                ) from e
        elif self.model == "mace_anicc":
            try:
                from mace.calculators.foundations_models import mace_anicc
                return mace_anicc(device=self.calculator_kwargs.get("device", "cpu"))
            except ImportError as e:
                raise ImportError(
                    "Loading by mace.calculators.foundations_models import mace_anicc failed. "
                    "Try to load MACE model by filename"
                ) from e

        elif self.model == "mace_off":
            try:
                from mace.calculators.foundations_models import mace_off
                return mace_off(device=self.calculator_kwargs.get("device", "cpu"),
                                default_dtype=self.calculator_kwargs.get("default_dtype", "float32"))
            except ImportError as e:
                raise ImportError(
                    "Loading by mace.calculators.foundations_models import mace_off failed. "
                    "Try to load MACE model by filename"
                ) from e
        else:
            LOG.debug("Trying to load MACE model from %s file...", self.model)
            dispersion = kwargs.get("dispersion", False)
            dispersion_cutoff = kwargs.get("dispersion_cutoff", 50.0)
            dispersion_xc = kwargs.get("dispersion_xc", "pbe")
            damping = kwargs.get("damping", "bj")
            default_dtype = kwargs.get("default_dtype", "float32")
            if dispersion:
                LOG.debug(
                    "Using dispersion_xc=%s, dispersion_cutoff=%s, damping=%s, default_dtype=%s (not used cpu)",
                    dispersion_xc,
                    dispersion_cutoff,
                    damping,
                    default_dtype,
                )
                d3_calc = self.load_dftd3_calculator(
                    device=device,
                    dispersion_xc=dispersion_xc,
                    damping=damping,
                    dispersion_cutoff=dispersion_cutoff,
                    default_dtype=default_dtype,
                )

                LOG.debug("%s was successfully set up.", d3_calc)

            try:
                mace_calc = MACECalculator(
                    model_paths=self.model, device=device, default_dtype=default_dtype, 
                )
                if dispersion:
                    LOG.debug(
                        "Using a sum calculator of MACE model %s and D3 dispersion correction %s.",
                        self.model,
                        d3_calc,
                    )
                    calc = SumCalculator([mace_calc, d3_calc])
                else:
                    LOG.debug("Using MACE calculator %s without dispersion correction.", self.model)
                    calc = mace_calc
                return calc
            except ImportError as e:
                raise ImportError("Importing mace model from %s file failed.", self.model) from e

    def load_ase_builtin_calculator(self, **kwargs):
        """
        Load one of ASE's builtin calculators
        https://wiki.fysik.dtu.dk/ase/ase/calculators/others.html
        Args:
            **kwargs:
                - model (str): Defaults to mace_mp.
                - rc (float): Radial cut-off for pairwise interactions (TIP3P, TIP4P, Lennard-Jones)
                - width (float): Width of cut-off function (TIP3P, TIP4P)
                - sigma (float): Potential minimum (Lennard-Jones)
                - epsilon (float): Potential depth (Lennard-Jones, Morse)
                - ro (float): Onset of cutoff function in 'smooth' mode (Lennard-Jones)
                - smooth (Boolean): Enable 'smooth' mode (Lennard-Jones)
                - r0 (float): Minimum distance (Morse)
                - rho0 (float): Expoential prefactor (Morse)
                - rcut1 (float): First radial cutoff distance (Morse)
                - rcut2 (float): Second radial cutoff distance (Morse)
                - neighbor_list (callable): neighbor_list function compatible with ase.neighborlist.neighbor_list (Morse)
        """
        if kwargs.get('model') == 'TIP3P':
            from ase.calculators.tip3p import TIP3P
            return TIP3P(**kwargs)
        elif kwargs.get('model') == 'TIP4P':
            from ase.calculators.tip4p import TIP4P
            return TIP4P(**kwargs)
        elif kwargs.get('model') == 'Lennard-Jones':
            from ase.calculators.lj import LennardJones
            return LennardJones(**kwargs)
        else:
            from ase.calculators.morse import MorsePotential
            return MorsePotential(**kwargs)

    def minimize(self, crystal: Crystal, **kwargs: dict) -> Crystal:
        """
        Takes a Crystal object and performs geometry optimization using ASE.

        Args:
            crystal (Crystal): Crystal object to be optimized.
            **kwargs:
            fmax (float): If set self.fmax is overridden. Maximum force tolerance after geometry optimization.
            steps (int): If set self.steps is overridden. Maximum number of geometry optimization steps.

        Returns:
            Crystal: Optimized Crystal object. Geometry optimization results are in the same order as the input Crystal object.
        """
        if not isinstance(crystal, Crystal):
            raise TypeError("Input must be a Crystal object.")
        LOG.info(
            "Performing ASE geometry minimization using %s and %s "
            "on %s.",
            self.model,
            self.optimizer,
            crystal.titl,
        )
        fmax = kwargs.get("fmax", self.fmax)  # override class variable if provided
        assert fmax >= 0, "Force tolerance must be non-negative."

        steps = kwargs.get("steps", self.steps)  # override class variable if provided
        assert steps >= 0, "Number of steps must be non-negative."

        LOG.debug(
            "Minimizing using %s calculator with symmetry=%s, "
            "fmax=%s, steps=%s.",
            self.calculator,
            self.symmetry,
            fmax,
            steps
        )

        crystal_std = crystal.standardized()
        if crystal_std is None:
            LOG.error("Failed to standardize crystal %s: %s", crystal.titl, e)
            LOG.error("The original crystal will be used.")
            crystal_std = crystal

        timing = Timer()
        cwd = os.getcwd()
        with timing:
            # When using CPU, D3 inputs and outputs are written/read to/from disk.
            with TemporaryDirectory() as tmpdirname:
                LOG.debug(
                    "Running geometry minimization of %s using ASE calculator %s in %s",
                    crystal.titl,
                    self.calculator,
                    tmpdirname,
                )
                os.chdir(tmpdirname)
                atoms = crystal_std.to_ase_atoms()
                atoms.calc = self.calculator

                if steps == 0:
                    LOG.debug("step = 0 in minimize(): Doing a single point calculation.")
                    try:
                        # The original crystal is not modified, as this is a single-point calculation.
                        new_crystal = deepcopy(crystal)  
                        new_crystal.properties["final_energy"] = atoms.get_potential_energy()
                        new_crystal.properties["ase_atom_forces"] = atoms.get_forces()
                    except Exception as e:
                        LOG.error(
                            "Failed to run single point calculation on %s: %s",
                            crystal.titl,
                            e,
                        )
                        os.chdir(cwd)
                        return None
                else:
                    LOG.debug(
                        "Initial energy of %s: ",
                        crystal.titl,
                    )
                    LOG.debug(
                        "%s",
                        atoms.get_potential_energy()
                    )
                    if self.symmetry:
                        LOG.debug("Setting symmetry constraints for ASE atoms.")
                        atoms.set_constraint(FixSymmetry(atoms))
                    ucf = FrechetCellFilter(atoms)
                    optimizer = self.optimizer(ucf, master=True)
                    try:
                        valid = run_with_timeout(
                            optimizer.run,
                            kwargs={"fmax": fmax, "steps": steps},
                            timeout=self.timeout,
                        )
                        LOG.debug(
                            "Optimizer in ASEMinimizer.minimize() finished for %s",
                            crystal.titl,
                        )

                    except TimeoutException:
                        LOG.error(
                            "Optimization of %s timed out after %d seconds.",
                            crystal.titl,
                            self.timeout
                        )
                        os.chdir(cwd)
                        return None

                    except Exception as e:
                        LOG.error(
                            "Failed to optimize %s: %s",
                            crystal.titl,
                            e,
                        )
                        os.chdir(cwd)
                        return None
                    max_force = np.abs(atoms.get_forces()).max()
                    if max_force > fmax:
                        LOG.info(
                            "Minimization of %s failed to reach force tolerance "
                            "%s within %d step limit after %.2fs.",
                            crystal.titl,
                            fmax,
                            steps,
                            timing.elapsed,
                        )
                        os.chdir(cwd)
                        return None  # consider the minimization failed
                    new_titl = kwargs.get("new_titl", f"{crystal.titl}_opt_by_ase")
                    new_crystal = Crystal.from_ase_atoms(atoms, titl=new_titl)
                    new_crystal.properties["final_energy"] = atoms.get_potential_energy()
                    new_crystal.properties["valid_minimization"] = valid
                os.chdir(cwd)
        new_crystal = self.standardise_units(
            crystal_std, new_crystal, model_units=self.model_units, scale_by_element_count=kwargs.get("scale_by_element_count", None)
        )
        new_crystal.properties["ase_metadata"] = {
            "optimizer": self.optimizer.__name__,
            "calculator_type": self.calculator_type,
            "model": self.model,
            "fmax": fmax,
            "steps": steps,
        }
        new_crystal.properties["minimization_time"] = timing.elapsed

        LOG.info(
            "Minimization of %s complete in %.2fs.",
            crystal.titl,
            timing.elapsed,
        )
        LOG.info(
            "The final lattice energy of %s is %.6f kJ/mol.",
            crystal.titl,
            new_crystal.properties["lattice_energy"],
        )

        return new_crystal
    
    def single_point(self, crystal: Crystal, silent: bool = False) -> Crystal:
        """Perform a single point calculation using ASE.

        Args:
            crystal (Crystal): Crystal object.
            silent (bool): If True, suppress output messages. Defaults to False.

        Returns:
            crystal: Crystal object with updated properties dictionary after single point calculation.
            updated properties include 'final_energy' and 'ase_atom_forces'.
        """
        LOG.debug(
            "Performing ASE single point calculation using %s on %s.",
            self.model,
            crystal.titl,
        )
        if not hasattr(self, "calculator"):
            LOG.error(
                "Calculator not set. Please initialize ASEMinimizer with a calculator."
            )
            return None
        return self.minimize(crystal, steps=0, silent=silent)

    def standardise_units(self, crystal : Crystal, 
                          new_crystal : Crystal, 
                          model_units: Optional[dict] = None, 
                          scale_by_element_count: Optional[str] = None) -> Crystal:
        """
        Convert the properties stored in the Crystal's properties dictionary to CSPy/DMACRYS 
        standard units.
        The standard unit of energy in CSPy is kJ/mol.
        'mol' refers to Avogadro's number and is not short for 'molecule'.
        The standard units of length in CSPy is Angstrom (Å).
        Therefore the standard unit of force is kJ/(mol Å) and stress is kJ/(mol Å^2).
        Energies are normalised per formula unit of molecules.
        A crystal purely of molA always has formula unit: [molA].
        A co-crystal of molA and molB in ratios 1:2 always has a formula unit: [molA, molB, molB]
        Energies are *NOT* normalised per molecule or per asymmetric unit.

        The units of the properties returned by an ASE optimisation depend on the model.
        These units are defined in the dictionary model_units. There is a default dictionary
        for each calculator but these are overridden if the user provides a dictionary when
        calling this function.
        Recognised and supported units of energy are: [kJ/mol, kJ eV, kcal]
        Recognised and supported units of length are: [Ang]
        Recognised and supported normalisations are: [molecular_formula_unit, cell, molecule, atom]
        energy_corr is an energy correction for delta-learning models. 
        If set to `lattice_energy`, the lattice_energy of the unoptimised crystal will be added to
        the energy of the energy returned by the ASE model.

        Symbols:
        -----------------------------------------
        Z:              The number of molecular formula units in the cell
        Zp (Z'):        The number of molecular formula units in the asymmetric unit
                        This is equal to Z divided by the number of symmetry operators in the space group setting
        G / Zpp (Z''):  The number of molecules in the asymmetric unit. 
                        G must be a whole number for each unique molecule.
                        Zpp may be a float for each unique molecule.

        model_units example:
        -----------------------------------------
            {'energy' : 'eV',
            'length' : 'Ang',
            'normalisation' : 'cell',
            'energy_corr' : None}

        CSPy standard units:
        ----------------------------------------
            {'energy' : 'kJ/mol',
            'length' : 'Ang',
            'normalisation' : 'molecular_formula_unit',
            'energy_corr' : None}

        Args:
            crystal (Crystal): Unoptimised crystal
            new_crystal (Crystal): Optimised crystal
            model_units (dict): Dictionary of units used by the model
            scale_by_element_count (str):   Symbol of element to scale energy by.
                                            If not None, treat Z as equal to the number of this 
                                            element in the unit cell.
                                            Use this for extended systems where discrete molecules
                                            cannot be defined.

        Returns:
            new_crystal (Crystal): Crystal object with updated properties dictionary 
            standardised to CSPy/DMACRYS style.
        """

        if model_units is None:
            if self.model in custom_calculators.keys():
                #model_units = custom_calculators[self.model]
                model_units = self.model_units
            elif self.calculator_type == "mace":
                model_units = other_models["mace"]
            else:
                LOG.info("Cannot find units for properties. Setting to defaults.")
                model_units = {'energy' : 'eV',
                                'length' : 'Ang',
                                'normalisation' : 'cell',
                                'energy_corr' : None}
                LOG.info("Default model units are: %s", model_units)

        # make sure energy is in kJ/mol
        if model_units['energy'] == 'kJ/mol':
            pass
        elif model_units['energy'] == 'kJ':
            new_crystal.properties["final_energy"] *= NA
        elif model_units['energy'] == 'eV':
            new_crystal.properties["final_energy"] *= EV2KJ * NA
        elif model_units['energy'] == 'kcal':
            new_crystal.properties["final_energy"] *= KCAL2KJ * NA
        else:
            LOG.warning("Model (%s) uses incompatible energy units of %s. Leaving units unchanged", self.model, model_units['energy'])

        # normalise by formula unit
        if model_units['normalisation'] == 'molecular_formula_unit':
            new_crystal.properties["lattice_energy"] = new_crystal.properties["final_energy"]
        else:
            if not scale_by_element_count:
                molecular_formula_unit, unique_molecules, Z = find_formula_unit(new_crystal.unit_cell_molecules())
                mols_per_molecular_formula_unit = len(molecular_formula_unit)
            else:
                Z = crystal.unit_cell_atoms()["element"].tolist().count(scale_by_element_count)
                total_atoms = len(crystal.unit_cell_atoms()["element"])

            if model_units['normalisation'] == 'cell':
                new_crystal.properties["lattice_energy"] = new_crystal.properties["final_energy"] / Z

            # scale_by_element is used when molecules can't be defined
            elif model_units['normalisation'] == 'molecule' and not scale_by_element_count:
                new_crystal.properties["lattice_energy"] = new_crystal.properties["final_energy"] * mols_per_molecular_formula_unit

            elif model_units['normalisation'] == 'atom':
                if not scale_by_element_count:
                    num_atoms_per_FU = 0
                    for molecule in molecular_formula_unit:
                        num_atoms_per_FU += len(molecule.elements)
                    # get energy per formula unit
                    new_crystal.properties["lattice_energy"] = new_crystal.properties["final_energy"] * num_atoms_per_FU
                else:
                    # multiply by total atoms to get energy per unit cell first then divide by Z (element count)
                    new_crystal.properties["lattice_energy"] = new_crystal.properties["final_energy"] * total_atoms / Z

            else:
                LOG.warning("Model (%s) uses incompatible energy units of %s. Leaving units unchanged", self.model, model_units['normalisation'])

        if model_units['energy_corr'] == 'lattice_energy':
            if 'lattice_energy' in crystal.properties.keys():
                new_crystal.properties["lattice_energy"] += crystal.properties['lattice_energy']
            else:
                LOG.error("Initial lattice energy missing from crystal. Storing correction as lattice energy.")

        if not "density" in new_crystal.properties.keys():
            new_crystal.properties["density"] = new_crystal.density

        # final_energy is the one that gets written to the db
        # what's the difference between final_energy and lattice_energy?
        new_crystal.properties["final_energy"] = new_crystal.properties["lattice_energy"]
        return new_crystal

    def __call__(self, obj, **kwargs):
        return self.minimize(obj, **kwargs)

    def __repr__(self):
        return f"<ASEMinimizer: {self.calculator}/{self.optimizer}>"
