__all__ = [
    "DmacrysMinimizer",
    "dmacrys_minimizer",
    "GaussianMinimizer",
    "gaussian_minimizer",
    "GulpMinimizer",
    "gulp_minimizer",
    "PminMinimizer",
    "pmin_minimizer",
    "XtbMinimizer",
    "xtb_minimizer",
    "CompositeMinimizer",
    "DftbMinimizer",
    "dftb_minimizer",
    "single_point_evaluation",
    "calculate_multipoles_and_minimize",
    "dftb_calculator",
    "vasp_calculator",
    "ASEMinimizer",
]
from .dmacrys_minimizer import DmacrysMinimizer
from .gaussian_minimizer import GaussianMinimizer
from .pmin_minimizer import PminMinimizer
from .xtb_minimizer import XtbMinimizer
from .composite_minimizer import CompositeMinimizer
from .gulp_minimizer import GulpMinimizer
from .dftb_minimizer import DftbMinimizer
from .vasp_minimizer import VaspMinimizer
from .ase_minimizer import ASEMinimizer

def single_point_evaluation(
    crystal, 
    mults, 
    axis, 
    potential, 
    bondlength_cutoffs=None,
    **kwargs
):
    if bondlength_cutoffs is not None:
        crystal.properties["bondlength_cutoffs"] = bondlength_cutoffs
    evaluater = DmacrysMinimizer(
        mults,
        axis,
        potential=potential,
        MAXI=0,
        rmsd_error_threshold=0.5,
        **kwargs,
    )
    nan = float("nan")

    try:
        evaluater(crystal)
    except:
        return nan, nan

    dm = evaluater.summary
    try:
        energy = dm.initial_energy
        density = dm.initial_density
        return energy, density
    except AttributeError:
        return nan, nan


def calculate_multipoles_and_minimize(crystal, **kwargs):
    import logging
    import numpy as np
    from cspy.apps.dma import generate_multipoles
    from cspy.configuration import CONFIG
    from cspy.potentials import available_potentials
    from cspy.chem.multipole import DistributedMultipoles
    from pathlib import Path
    import shutil

    LOG = logging.getLogger(__name__)

    # setup settings used in multiple programs
    potential = kwargs.get("potential", "fit")
    foreshorten_hydrogens = kwargs.get("foreshorten_hydrogens", None)
    potential_type, potential_filename = available_potentials[potential]
    if potential_type == "W" and foreshorten_hydrogens is None:
        foreshorten_hydrogens = 0.1
        LOG.info(
            "Reducing hydrogen bond lengths by: %.2f", foreshorten_hydrogens
        )

    # setup gaussian settings
    g09_kwargs = {
        "method": kwargs.get("method", CONFIG.get("gaussian.method")),
        "basis_set": kwargs.get("basis_set", CONFIG.get("gaussian.basis")),
        "nprocshared": kwargs.get("gaussian_cpus", 2),
        "mem": kwargs.get("gaussian_mem", "2GB"),
        "pcm": kwargs.get("pcm"),
        "charges": kwargs.get("charges")
    }

    # setup gdma and mulfit settings
    mulfit_method = kwargs.get("mulfit_method", "cumulative")
    mulfit_rank = kwargs.get("mulfit_rank", None)

    ranks = []
    mf_suffix = ""
    if mulfit_rank is not None:
        ranks.append(mulfit_rank)
        mf_suffix = "_rank{}{}".format(
            mulfit_rank, "o" if mulfit_method == "overall" else ""
        )
    mf_format = crystal.titl + "{}.dma"
    multipole_filename = mf_format.format(mf_suffix)

    dma_kwargs = {
        "potential_type": potential_type,
        "mulfit_method": mulfit_method,
        "dma_switch": kwargs.get("dma_switch", 4),
        "foreshorten_hydrogens": foreshorten_hydrogens,
        "hydrogen_radius": kwargs.get("hydrogen_radius", None)
    }

    # run gaussian and gdma/mulfit if multipole file doesn't exist
    multipole = kwargs.get("multipole", None)
    cleanup_files = []
    if multipole is None and not Path(multipole_filename).exists():
        LOG.info("Calculating multipoles (ranks=%s)", ranks)
        cleanup_files = generate_multipoles(
            crystal, ranks, mf_format, **dma_kwargs, **g09_kwargs
        )
    elif multipole is not None and not Path(multipole_filename).exists():
        LOG.info("Using multipoles in %s", multipole)
        shutil.copy(multipole, multipole_filename)
    else:
        LOG.info("Using multipoles in %s", multipole_filename)
    mults = DistributedMultipoles.from_dma_file(multipole_filename)

    # setup neighcrys/dmacrys settings
    cutoff = kwargs.get("cutoff", "calculate")
    if cutoff == "calculate":
        cutoff = 15.0
        from scipy.spatial.distance import pdist
        for mol in crystal.symmetry_unique_molecules():
            if len(mol) == 1:
                continue
            cutoff = max(cutoff, 1.5 * np.max(pdist(mol.positions)))
    else:
        cutoff = float(cutoff)
    LOG.info("Using cutoff: %f", cutoff)

    dmacrys_kwargs = {
        "potential": potential,
        "timeout": kwargs.get("dmacrys_timeout", CONFIG.get("dmacrys.timeout")),
        "MAXI": kwargs.get("MAXI", CONFIG.get("neighcrys.max_iterations")),
        "vdw_cutoff": cutoff,
        "bondlength_cutoffs": crystal.properties["bondlength_cutoffs"]
    }
    if kwargs.get("single_point", False):
        LOG.info(
            "Calculating single point energy for crystal %s, l_max=%d",
            crystal, mults.max_rank,
        )
        dmacrys_kwargs["MAXI"] = 0
        # dmacrys_kwargs["CONV"] = True
        #LOG.info("Minimizing crystal %s with PROP calculation", crystal)
        # dmacrys_kwargs["ACCM"] = kwargs.get("ACCM", 100000000)
        # dmacrys_kwargs["STAR"] = "PROP"
        # dmacrys_kwargs["WCAL"] = False
        dmacrys_kwargs["NOPR"] = False
    elif kwargs.get("symmetry_reduction", False):
        LOG.info("Minimizing crystal %s with SEIG", crystal)
        dmacrys_kwargs["SEIG"] = kwargs.get("SEIG", 1)
        dmacrys_kwargs["NOPR"] = False
    elif kwargs.get("phonon", False):
        LOG.info("Minimizing crystal %s with PROP calculation", crystal)
        dmacrys_kwargs["ACCM"] = kwargs.get("ACCM", 100000000)
        dmacrys_kwargs["STAR"] = "PROP"
        dmacrys_kwargs["WCAL"] = False
        dmacrys_kwargs["NOPR"] = False
    else:
        LOG.info("Minimizing crystal %s", crystal)

    # setup axis file
    axis_file = kwargs.get("axis", None)
    axis_filename = crystal.titl + ".mols"
    if axis_file is None and not Path(axis_filename).exists():
        LOG.info("creating axis file using neighcrys")
        crystal.neighcrys_setup(potential_type)
        axis_file = Path(axis_filename)
        axis_file.write_text(crystal.neighcrys_axis.to_string())
    elif axis_file is not None and not Path(axis_filename).exists():
        LOG.info("using axes in {}".format(axis_file))
        shutil.copy(axis_file, axis_filename)
    else:
        LOG.info("using axes in {}".format(axis_filename))
    axis = Path(axis_filename).read_text()

    # run neighcrys/dmacrys

    reorder_atoms_if_high_rmsd = kwargs.get("reorder_atoms_if_high_rmsd", False)
    reorder_method = kwargs.get("reorder_method", "molecular_axis")
    minimizer = DmacrysMinimizer(mults,
                                 axis,
                                 reorder_atoms_if_high_rmsd=reorder_atoms_if_high_rmsd,
                                 reorder_method=reorder_method,
                                 **dmacrys_kwargs)
    result = minimizer(crystal)

    # process results and cleanup
    dmacrys_summary_file = crystal.titl + mf_suffix + ".dmacrys_summary"
    dmacrys_output_file = crystal.titl + mf_suffix + ".dmacrys_stdout"
    try:
       dm = minimizer.summary
       LOG.info(
           "Initial energy: %.3f, initial density: %.3f",
           dm.initial_energy, dm.initial_density,
         )
        
       LOG.info(
            "Final energy: %.3f, final density: %.3f",
            dm.final_energy, dm.final_density
         )
    except AttributeError:
       #log error here
       if result=='Timeout':
           LOG.error("Summary file not available due to timeout")
       elif result==None:
           LOG.error("Summary file not available due to unknown error and/or no SHELX file content")
       return None
    crystal.properties["lattice_energy"] = dm.initial_energy
    crystal.properties["density"] = dm.initial_density
    if kwargs.get("atom_forces", False) and kwargs.get("single_point", False):
        crystal.properties["atom_forces"] = minimizer.dmaout.atom_forces(crystal)
    if kwargs.get("stress_tensor", False):
        (stress_tensor_eV, stress_tensor_kjmol) = minimizer.dmaout.stress_tensor(
            fixed_cell=dmacrys_kwargs.get("CONV", False)
        )
        crystal.properties["stress_tensor_eV"] = stress_tensor_eV
        crystal.properties["stress_tensor_kjmol"] = stress_tensor_kjmol
    if result is not None:
        e = result.properties["lattice_energy"]
        d = result.properties["density"]
        result.titl = f"{crystal.titl}.{e}.{d}"
        if kwargs.get("atom_forces", False):
            result.properties["atom_forces"] = minimizer.dmaout.atom_forces(result)
        result.properties["stress_tensor_eV"] = crystal.properties.get("stress_tensor_eV", None)
        result.properties["stress_tensor_kjmol"] = crystal.properties.get("stress_tensor_kjmol", None)
    if kwargs.get("cleanup", True):
        for fname in cleanup_files:
            Path(fname).unlink()
    if kwargs.get("outputs", False):
        Path(dmacrys_summary_file).write_text(dm.contents)
        Path(dmacrys_output_file).write_text(minimizer.dmaout_contents)
    if kwargs.get("single_point", False):
        result = crystal
    return result

def dftb_calculator(structure, **kwargs):
    import logging
    from pathlib import Path
    from cspy.configuration import CONFIG
    from cspy.chem.molecule import Molecule
    LOG = logging.getLogger(__name__)
    
    mol_calc = isinstance(structure, Molecule)
    dftb_keywords = CONFIG.get('dftb')
    if kwargs.get("single_point") or kwargs.get("lattice_opt"):
        dftb_keywords['fixed_lattice_opt'] = False
    dftb_keywords.update(kwargs)
    LOG.debug("dftb_keywords:\n%s", dftb_keywords)
    minimizer = DftbMinimizer(mol_calc=mol_calc,
                              **dftb_keywords)
    result = minimizer(structure)
    dftb_log_file = "dftb_stdout.txt"
    dftb_input_hsd = "dftb_in.txt"

    if result is not None:
        e = float(result.properties["final_energy"])
        LOG.info(f"Final energy = {e} kJ/mol")
        if not mol_calc:
            dens = result.density
            result.properties['density'] = dens
            LOG.info(f"Final density = {dens} cm^3/g")
            result.titl = f"{structure.titl} {e} {dens}"

    if kwargs.get("outputs", False):
        Path(dftb_log_file).write_text(minimizer.log_summary)
        Path(dftb_input_hsd).write_text(minimizer.input_summary)

    return result

def vasp_calculator(structure, **kwargs):
    import logging
    from pathlib import Path
    from cspy.configuration import CONFIG
    LOG = logging.getLogger(__name__)

    vasp_keywords = CONFIG.get('vasp')
    vasp_keywords.update(kwargs)
    LOG.debug("vasp_keywords:\n%s", vasp_keywords)
    minimizer = VaspMinimizer(**vasp_keywords)
    result = minimizer(structure)
    vasp_log_file = "vasp_stdout.txt"

    if result is not None:
        if kwargs.get("single_point", False):
            e = float(result.properties["energy"])
        else:
            e = float(result.properties["final_energy"])
        LOG.info(f"Final energy = {e} kJ/mol")
        dens = result.density
        LOG.info(f"Final density = {dens} cm^3/g")
        result.titl = f"{structure.titl} {e} {dens}"

    if kwargs.get("outputs", False):
        Path(vasp_log_file).write_text(minimizer.log_summary)

    return result

def xtb_calculator(structure, **kwargs):
    import logging
    from pathlib import Path
    from cspy.configuration import CONFIG
    from cspy.chem.molecule import Molecule
    from cspy.minimize import XtbMinimizer
    LOG = logging.getLogger(__name__)

    xtb_keywords = CONFIG.get('xtb')
    LOG.debug("xtb_keywords:\n%s", xtb_keywords)
    xtb_keywords.update(kwargs)
    minimizer = XtbMinimizer(**xtb_keywords)
    result = minimizer(structure) #result is an energy if spe, structure if opt
    xtb_log_file = "xtb_std_out.txt"

    Path(xtb_log_file).write_text(minimizer.log_summary)

    return result
    
class MinimizerFactory:
    kind_mapping = {
        "pmin": PminMinimizer,
        "dmacrys": DmacrysMinimizer,
        "gaussian": GaussianMinimizer,
        "xtb": XtbMinimizer,
        "gulp": GulpMinimizer,
        "dftb": DftbMinimizer,
        "vasp": VaspMinimizer,
        "ase" : ASEMinimizer
    }

    def __init__(self, kind, *args, **kwargs):
        self.kind = kind.lower()
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.create_minimizer(*args, **kwargs)

    def create_minimizer(self, *args, **kwargs):
        args = args + self.args
        cls = self.kind_mapping[self.kind]
        self.kwargs = {k: v for k, v in self.kwargs.items() if k not in kwargs}
        return cls(*args, **self.kwargs, **kwargs)

    @property
    def required_args(self):
        if self.kind in ("pmin", "dmacrys", "gulp"):
            return "multipoles", "axis", "potential"
        else:
            return ()

    def __repr__(self):
        return (
            f"<MinimizerFactory: kind='{self.kind}' required_args={self.required_args}>"
        )
