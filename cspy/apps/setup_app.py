import time
import logging
import sys
import os
import datetime
import toml
import json
import numpy as np
import itertools
import argparse
from pathlib import Path as pathlib_Path
from cspy import __version__
from cspy.configuration import CONFIG
from cspy.apps.dma import generate_combined_name
from cspy.chem.multipole import DistributedMultipoles
from cspy.util.path import Path
from cspy.crystal.util import find_formula_unit
from cspy.formats.dma import parse_dma
from cspy.chem.energy import boltzmann_weighting
from cspy.chem.molecule import Molecule
from cspy.crystal import Crystal
from cspy.potentials import available_potentials
from cspy.db import CspDataStoreMPS
from cspy.db import CspDataStoreFlex
from typing import Any
from cspy.util import recursive_dict_update
import math

LOG = logging.getLogger(__name__)

SETTINGS_IGNORE_LIST = ["SERVER_IP", "PROCID", "REDISPASS", "BROKERPORT", "BACKENDPORT", "VHOST", "net", "celery"]

BANNER = \
    "============================================================================\n"\
    "============================================================================\n"\
    "                        ██████                                              \n"\
    "                  ██████               ███                                  \n"\
    "                 ████          ███████ ███████                              \n"\
    "                 ████       █████      ██  ███████                          \n"\
    "                 ████          █████   ██      ██████                       \n"\
    "                 ████             ████ ██    █████                          \n"\
    "                 ████             ████ █████        ██  ███                 \n"\
    "                 ████       ██ ████    ██             ███                   \n"\
    "                  ███                                  █                    \n"\
    "                     ████                             ███                   \n"\
    "                         ███                       ███                      \n"\
    "============================================================================\n"\
    "============================================================================"

def print_now(string: str) -> None:
    """
    Python buffers print statements and then prints them
    later for efficiency. We want them printed now.
    This is an alias that enforces printing now.

    Parameters
    ----------

    string : str
        The string to print by flushing the buffer
    """
    print(string, end='\n', flush=True)

class AppHeader:
    """Class for setting up and printing a header for mol-CSPy apps.
        This header contains metadata and information about the application
        being run. E.g. settings.

    Parameters
    ----------
    **kwargs
        Optional keyword arguments
    """

    def __init__(self, **kwargs) -> None:
        self.banner = BANNER

        # get versions
        self.cspy_version = __version__
        self.python_version = sys.version.split()[0]
        self.version_summary = f"{'':{11}s} CSPy Version: {self.cspy_version:{15}s} Python Version: {self.python_version:{15}s}"
        
        # conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            self.conda_name = conda_prefix.split('/')[-1]
            self.conda_summary = f"{'':{41}s} Conda Env: {self.conda_name:{15}s}"
        else:
            LOG.debug("Failed to find conda environment. Assuming we're not using one.")
            self.conda_name = None

        # git commit
        head_commit, working_tree_commit = self._check_git_commit()
        if head_commit:
            if working_tree_commit and head_commit != working_tree_commit:
                self.git_summary = f"{'':{11}s} Git Commit: {head_commit:{40}s}\n{'':{11}s} Work Tree:  {working_tree_commit:{40}s}"
            else:
                self.git_summary = f"{'':{11}s} Git Commit: {head_commit:{40}s}"
        else:
            LOG.debug("Failed to find git commit hash. Assuming not a git repo.")
            self.git_summary = None

        # get date and time
        self.update_now()

        self.initialise_print_list()

    def update_now(self) -> None:
        """
        Find the date and time and summarise in a string
        """
        self.date = datetime.datetime.today().strftime("%Y-%m-%d")
        self.time = datetime.datetime.now().time().strftime("%H:%M:%S")
        self.now_summary = f"{'':{11}s} Date: {self.date:{23}s} Time: {self.time:{15}s}"

    def print_header(self) -> None:
        """
        Iterate over list of attributes in to_print
        and print those attributes
        """
        for attribute in self.to_print:
            print_now(getattr(self, attribute))

    def initialise_print_list(self) -> None:
        """
        Create list of essential strings to print.
        If a `to_print` list already exists, 
        we'll reset it.
        """
        self.to_print = ["banner", "version_summary", "now_summary"]
        if self.conda_name:
            self.to_print.insert(2, "conda_summary")
        if self.git_summary:
            self.to_print.insert(2, "git_summary")

    def print_app_info(self, app_name : str, arguments : dict[str, Any]) -> None:
        """
        Print the name of the app and and the CLI arguments
        that were passed to it.

        Parameters
        ----------

        app_name : str
            The name of the app that is running
        
        arguments : dict[str, Any]
            The CLI arguments of the app
        """
        print_now('\n')
        print_now(f"{'':>{15}s}{'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^':^{40}s}{'':<{15}s}")
        print_now(f"{'':>{17}s}{app_name:^{40}s}{'':<{15}s}")
        for arg, value in vars(arguments).items():
            if isinstance(value, list):
                value = ', '.join(value)
            elif not value:
                value = 'None'
            else:
                value = str(value)

            print_now(f"{'':>{17}s}{arg + ':':<{20}s}{value:<{20}s}{'':<{15}s}")

        print_now(f"{'':>{15}s}{'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^':^{40}s}{'':<{15}s}")

    def print_settings(self) -> None:
        """
        Check settings from CONFIG and print them.
        We ignore any settings from SETTINGS_IGNORE_LIST
        as these settings are likely to contain unhelpful
        or sensitive data (e.g. IP address)
        """
        self.settings = CONFIG.__dict__['_settings'].__dict__['maps'][0]
        for key in self.settings:
            if not key in SETTINGS_IGNORE_LIST:
                setting = self.settings[key]
                print_now('\n')
                print_now(key)
                if isinstance(setting, list):
                    for subsetting in setting:
                        print_now(subsetting)
                else:
                    print_now(setting)

    def _parse_minimisation_steps(self, steps: list[dict[str, Any]]) -> None:
        """
        The minimisation steps should be very clear, so
        we put in a bit of extra effort to format them nicely.

        Parameters
        ----------

        steps : list[dict[str, Any]]
            The minimisation steps
        """
        for step_ind, step in enumerate(steps):
            print_now(f"{'':>{15}s}{'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~':^{40}s}{'':<{15}s}")
            if 'kind' in step.keys():
                print_now(f"{'::':>{17}s}{step['kind']:^{40}s}{'::':<{15}s}")
            for setting in step.keys():
                if not setting == 'kind':
                    print_now(f"{'::':>{17}s}{setting + ':':<{20}s}{str(step[setting]):<{20}s}{'::':<{15}s}")
            print_now(f"{'':>{15}s}{'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~':^{40}s}{'':<{15}s}")
            if not step_ind == len(steps) - 1:
                print_now(f"{'':>{15}s}{'V':^{40}s}{'':<{15}s}")
                print_now(f"{'':>{15}s}{'V':^{40}s}{'':<{15}s}")
                print_now(f"{'':>{15}s}{'V':^{40}s}{'':<{15}s}")

    def _parse_settings_dict(self, settings_dict: dict[str, Any]) -> None:
        """
        For a dictionary of settings in cspy.toml (e.g. DMACRYS),
        print those settings to the cmdline.

        Parameters
        ----------

        settings_dict : dict[str, Any]
            The dictionary with the settings in a .toml configuration file
        """
        print_now(f"{'':>{15}s}{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!':^{40}s}{'':<{15}s}")
        for key, value in settings_dict.items():
            if isinstance(value, list):
                try:
                    value = ', '.join(value)
                except:
                    value = [' '.join(subvalue) for subvalue in value]
                    value = ', '.join(value)
            elif not value:
                value = 'None'
            else:
                value = str(value)
            print_now(f"{'':>{17}s}{key + ':':<{25}s}{value:<{20}s}{'':<{15}s}")
        print_now(f"{'':>{15}s}{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!':^{40}s}{'':<{15}s}")

    def _check_git_commit(self) -> tuple[str | None, str | None]:
        """
        Check if we can get the current git commit hash. Check the hash of
        the current HEAD and of the current working tree. If the working tree
        has uncommitted changes, the two hashes will differ.

        Returns
        -------

        tuple[str | None, str | None]
            A tuple of (head_commit, working_tree_commit). If we cannot
            determine the commit hashes, we return None for those values.
        """
        import subprocess

        try:
            head_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], 
                    stderr=subprocess.DEVNULL,
                    cwd=pathlib_Path(__file__).parent.parent
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            LOG.debug("Failed to get git commit hash for HEAD. Assuming not running in a git repo.")
            return None, None

        try:
            working_tree_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD^{tree}"],
                    stderr=subprocess.DEVNULL,
                    cwd=pathlib_Path(__file__).parent.parent
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            LOG.debug("Failed to get git commit hash for working tree.")
            return head_commit, None

        return head_commit, working_tree_commit

    def print_toml(self) -> None:
        """
        Check if a cspy.toml exists and if it does, print
        the contents.
        We ignore any settings from SETTINGS_IGNORE_LIST
        as these settings are likely to contain unhelpful
        or sensitive data (e.g. IP address)
        """
        self.toml_contents = None
        for loc in CONFIG.config_locations:
            with open(loc) as f:
                if not self.toml_contents:
                    self.toml_contents = toml.load(f)
                else:
                    self.toml_contents = recursive_dict_update(self.toml_contents, toml.load(f))
        
        if self.toml_contents:
            for key in self.toml_contents:
                if not key in SETTINGS_IGNORE_LIST:
                    setting = self.toml_contents[key]
                    print_now('\n')
                    print_now(f"{'':>{17}s}{key:^{40}s}{'':<{15}s}")
                    if isinstance(setting, list):
                        # handle minimisation steps more clearly
                        if key == "csp_minimization_step":
                            self._parse_minimisation_steps(setting)
                        else:
                            for subsetting in setting:
                                print_now(subsetting)
                    else:
                        self._parse_settings_dict(setting)
        else:
            print_now("No .toml detected. Using mol-CSPy defaults.")

    def print_raw_inputs(self) -> None:
        """
        Take the input that the user provided to the app 
        (CLI and cspy.toml) and print it to terminal with
        no processing. 
        This can then be copy/pasted to set up other jobs.
        """
        executed_command_list = sys.argv
        executed_command_list[0] = executed_command_list[0].split('/')[-1]
        print_now('\n\n')
        print_now(f"{'':>{17}s}{'Raw Input':^{40}s}{'':<{15}s}")
        print_now(f"{'':>{15}s}{'################# Command ###################':^{40}s}{'':<{15}s}")
        print_now(' '.join(executed_command_list))
        print_now('\n')
        print_now(f"{'':>{15}s}{'################ cspy.toml ##################':^{40}s}{'':<{15}s}")
        if os.path.isfile('cspy.toml'):
            with open('cspy.toml', 'r') as f:
                flines = f.read()
                print_now(flines)
        else:
            print_now("No cspy.toml detected")

def check_multipoles_valid(xyz_files : list[str], axis : str, charges : str, multipoles : str, potential : str) -> bool:
    """Check that DMA file can be read properly and then check that they
    map correctly to the provided molecules.

    Parameters
    ----------
    xyz_files : list[str]
        List of paths to xyz files

    axis : str
        Path to DMACRYS .mols file

    charges : str
        Path to DMACRYS _rank0.dma file

    multipoles : str
        Path to DMACRYS .dma file

    potential : str
        Name of fit or w99 potential in cspy.potentials

    Returns
    -------
    success : bool
        True if multipoles are valid
    """
    
    c = DistributedMultipoles.from_dma_file(charges)
    if len(c.molecules) > 0:
        LOG.info("Succesfully read charges file: %s mols", len(c.molecules))
    m = DistributedMultipoles.from_dma_file(multipoles)
    if len(m.molecules) > 0:
        LOG.info(
            "Succesfully read higher order multipoles file: %s mols", len(m.molecules)
        )
    axis_contents = Path(axis).read_text()
    success = True
    for rank, multipoles in (("charges", c), ("multipoles", m)):
        crys = Crystal.from_xyz_files(xyz_files, titl="tmp")
        potential_type = available_potentials[potential][0]
        crys.neighcrys_setup(potential_type=potential_type, file_content=axis_contents)
        LOG.info("Mapping: %s", rank)
        reduction = 0.1 if potential_type == "W" else None
        mapping = crys.map_multipoles(multipoles, foreshorten_hydrogens=reduction)
        for i, (j, rmsd, inv) in enumerate(mapping):
            LOG.info(
                "%s: %5s idx=%d%s (%.3g): %s",
                xyz_files[i],
                rank,
                j,
                "'" if inv else "",
                rmsd,
                "ok" if rmsd < 1e-4 else "failed",
            )
            if rmsd > 1e-4:
                success = False
    return success


def aut_data(dma_file: str, xyz_files: list[str]) -> tuple[list[list[str]], list[list[float]], dict[str, dict[str, Any]]]:
    """Needed when running SAUCE with the Asymmetric Unit Transplant
    method. Collects data related to molecules in the asymmetric units.

    Parameters
    ----------

    dma_file : str
        Path of a DMA file

    xyz_files : list[str]
        List of paths to xyz files

    Returns
    -------
    pot_atom_types : list[list[str]]
        Atom typings for energy calculations. Each element of
        the top level list is a molecule.

    pot_charges : list[list[float]]
        Point charges for energy calculations. Each element of
        the top level list is a molecule.
    
    molecules : dict[str, dict[str, Any]]
        List where data for each molecule is its own dictionary
    """

    dma_data = parse_dma(dma_file)
    pot_atom_types = []
    pot_charges = []
    for molecule_dma in dma_data:
        pot_atom_types.append(molecule_dma['atom_types'])
        pot_charges.append(molecule_dma['charges'])

    molecules = dict()
    for xyz in xyz_files:
        if not xyz in molecules.keys():
            cspyMol = Molecule.from_xyz_file(xyz)
            xyz_string = cspyMol.to_xyz_string()
            elements = cspyMol.elements
            cspyMol.get_symmetry_equivalent_atoms()

            molecules[xyz] = {'num_atoms' : len(elements), 
                                'equivalent_atoms' : cspyMol.equivalent_atoms,
                                'xyz_coordinates' : xyz_string,
                                'atom_elements' : elements}
        else:
            pass

    return pot_atom_types, pot_charges, molecules


def chomp_data(args : argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], str], dict[tuple[str, str], list[float]], dict[str, dict[str, Any]]]:
    """Scrape database of molecular pairs for ChoMP.

    Parameters
    ----------
    args : args.Namespace
        argparse arguments

    Returns
    -------
    properties : dict[str, dict[str, Any]]
        Dict of dicts where each top level
        dict is a pair and the other dict 
        describes the properties

    pair_types : dict[tuple[str, str], str]
        Dict containing all types of pairs 
        (e.g. mol A and mol B)
        as well as all unique pairs of that type.

    weights : dict[tuple[str, str], list[float]]
        Identical structure to above but instead of 
        IDs of unique pairs, it is the random weights 
        of those pairs.
    
    molecules : dict[str, dict[str, Any]]
        Data concerning molecules included in the pairs 
        where the key is a molecule name and the pair is a
        dict of data.
    """
    db_name = args.clg_chomp
    db = CspDataStoreMPS(db_name)
    properties = dict()
    for row in db.select("pairs",
                            ['id','min_energy','min_density',
                            'xyz_coordinates','mol2_coordinates',
                            'asymmetric','symmetric', 'molecule1', 'molecule2']):
        properties[row[0]]={"min_energy" : row[1], "energy_list" : [row[1]],
                    "min_density" : None, "max_density" : None, 
                    "median_density" : None, "density_list" : [row[2]],
                    "frequency" : 1, "xyz_coordinates" : row[3], "mol2_coordinates" : row[4], 
                    "asymmetric" : row[5], "symmetric" : row[6], 
                    "molecule1" : row[7], "molecule2" : row[8]}
    
    molecules = dict()
    for row in db.select("molecules", ['id','xyz_coordinates', 'equivalent_atoms']):
        molecules[row[0]] = {'num_atoms' : int(row[1].split('\n')[0]), 
                                'equivalent_atoms' : json.loads(row[2]),
                                'xyz_coordinates' : row[1]}

    pairs = list(properties.keys())
    pair_types = dict()
    weights = dict()
    energies = dict()

    xyz_files_with_ext = args.xyz_files
    xyz_files = [xyz.strip('.xyz') for xyz in xyz_files_with_ext]
    unique_xyz = set(sorted(xyz_files))

    for combo in itertools.combinations(unique_xyz, 2):
        combo = tuple(sorted(combo))
        pair_types[combo] = []
    for xyz in unique_xyz:
        combo = tuple([xyz, xyz])
        pair_types[combo] = []

    for xyz in unique_xyz:
        mol_num_atoms = molecules[xyz]['num_atoms']
        # don't allow pairs of single atoms
        if mol_num_atoms == 1:
            pair_types.pop(tuple([xyz, xyz]), None)

    for pair in pairs:
        pair_type = tuple(sorted([properties[pair]["molecule1"], properties[pair]["molecule2"]]))
        if pair_type in pair_types.keys():
            pair_types[pair_type].append(pair)
            if not pair_type in energies.keys():
                energies[pair_type] = []
            energies[pair_type].append(properties[pair]['min_energy'])

    for pair_type in energies.keys():
        weights[pair_type] = boltzmann_weighting(np.asarray(energies[pair_type]), temperature=args.chomp_temperature)

    return properties, pair_types, weights, molecules


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Accepts an argparse parser and adds common arguments to it.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        argparse parser

    Returns
    -------
    parser : argparse.ArgumentParser
        argparse arguments.
        argparse parser
    """
    parser.add_argument(
        "--log-level",
        default=CONFIG.get("csp.log_level"),
        help="Log level"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        default=False,
        help="Keep DMACRYS and NEIGHCRYS files which, for each structure, are stored in a new directory in the pwd."
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        default=False,
        help="Skip the mol-CSPy header at the start of the job."
    )
    parser.add_argument(
        "--status-file",
        default=CONFIG.get("csp.status_file"),
        type=str,
        help="Specify output status file",
    )
    
    return parser

def add_DMACRYS_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Accepts an argparse parser and adds arguments to it
    that relate to the running of DMACRYS.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        argparse parser

    Returns
    -------
    parser : argparse.ArgumentParser
        argparse arguments.
        argparse parser
    """

    parser.add_argument(
        "-c",
        "--charges",
        type=str,
        default=CONFIG.get("csp.charges_file"),
        help="Rank0 multipole file",
    )
    parser.add_argument(
        "-m",
        "--multipoles",
        type=str,
        default=CONFIG.get("csp.multipoles_file"),
        help="RankN multipole file",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=str,
        default=None,
        help="Axis filename for structure minimization",
    )

    return parser

def add_minimisation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Accepts an argparse parser and adds arguments to it
    that relate to geometry optimisations.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        argparse parser

    Returns
    -------
    parser : argparse.ArgumentParser
        argparse arguments.
        argparse parser
    """

    parser.add_argument(
        "-p",
        "--potential",
        default=CONFIG.get("csp.potential"),
        choices=available_potentials.keys(),
        help="intermolecular potential name",
    )
    parser.add_argument(
        "--cutoff",
        default=CONFIG.get("csp.cutoff"),
        help="dmacrys real space/repulsion-dispersion cutoff",
    )

    return parser

def add_CLG_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Accepts an argparse parser and adds arguments to it
    that relate to the crystal landscape generator.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        argparse parser

    Returns
    -------
    parser : argparse.ArgumentParser
        argparse arguments.
        argparse parser
    """

    clg_type = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-g",
        "--spacegroups",
        type=str,
        default=CONFIG.get("csp.spacegroups"),
        help="Spacegroup set for structure generation",
    )
    parser.add_argument(
        "-n",
        "--number-structures",
        type=int,
        default=CONFIG.get("csp.number_structures"),
        help="Number of structures in each spacegroup for structure generation. Should be provided as a single integer",
    )
    parser.add_argument(
        "--nudge",
        type=int,
        default=0,
        help="Nudge molecules in asymmetric unit that fail QR step",
    )
    parser.add_argument(
        "--adaptcell",
        action="store_true",
        help="Adaptively optimise cell parameters",
    )
    parser.add_argument(
        "--asi",
        action="store_true",
        help="Allow molecules to have superimposed centroids (set to true for encapsulation)",
    )
    clg_type.add_argument(
        "--clg-chomp",
        type=str,
        help="Use the chomp CLG with molecular pairs from the provided database",
    )
    clg_type.add_argument(
        "--clg-aut",
        action="store_true",
        help="Use the AUT CLG with molecular pairs. "
        "If not running a CSP, a database must exist of format: [seed]-spacegroup-AU.db",
    )

    return parser

def add_gaussian_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Accepts an argparse parser and adds arguments to it
    that relate to the crystal landscape generator.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        argparse parser

    Returns
    -------
    parser : argparse.ArgumentParser
        argparse arguments.
        argparse parser
    """

    parser.add_argument(
        "-fun",
        "--functional",
        type=str,
        help="Electronic structure functional for Gaussian calculation (a string passed verbatim into "\
            "Gaussian input file, e.g. B3LYP or PBEPBE)",
        default="B3LYP",
    )
    parser.add_argument(
        "-bas",
        "--basis_set",
        type=str,
        help="Basis set to use in each Gaussian run",
        default="6-311G**",
    )
    parser.add_argument(
        "-basfile",
        "--basis_file",
        type=str,
        help="File contains the basis set information that are not standard in Gaussian",
        default="",
    )
    parser.add_argument(
        "-gmem",
        "--gaussian_memory",
        type=str,
        help="Memory to use in each Gaussian run",
        default="2GB",
    )
    parser.add_argument(
        "-gcpu",
        "--gaussian-cpus",
        type=int,
        help="Number of cpus to use for each Gaussian calculation",
        default="1",
    )
    parser.add_argument(
        "--chelpg",
        action="store_true",
        help="Use CHelpG to calculate partial charges ",
        default=False,
    )
    parser.add_argument(
        "--gaussian_cleanup",
        action="store_true",
        help="Clean up gaussian output",
        default=False,
    )
    parser.add_argument(
        "-vol",
        "--molecular_volume",
        action="store_true",
        help="Calculate molecular volume",
        default=False,
    )
    parser.add_argument(
        "-pcm",
        "--polarizable_continuum",
        action="store_true",
        help="Specify the value of epilson to use in the polarisable contiuum model"\
            "e.g. '3' or '9.2'",
        default=False,
    )
    parser.add_argument(
        "-epcm",
        "--external_iteration_pcm",
        action="store_true",
        help="Use a Polarizable Continuum Model",
        default=False,
    )
    parser.add_argument(
        "-esp",
        "--dielectric_constant",
        type=str,
        help="Dielectric constant for Polarizable Continuum",
        default="3.0",
    )
    parser.add_argument(
        "-gopt",
        "--opt",
        type=str,
        help="Options within gaussian optimisation",
        default="ModRedundant,",
    )
    parser.add_argument("--iso", type=str, help="ISO value", default="")
    parser.add_argument(
        "--freq",
        type=str,
        default=False,
        help="Options for frequency, use a blank to turn on default",
    )
    parser.add_argument(
        "--gaussian_all_molecules",
        action="store_true",
        help="Run Gaussian on all molecules.",
        default=False,
    )
    parser.add_argument(
        "--additional_args",
        type=str,
        help="Additional arguments for gaussian",
        default="",
    )
    parser.add_argument(
        "--set_molecular_states",
        type=str,
        default=None,
        help="Set the charge and spin multiplicity for each molecule in the crystal e.g. '0,1 0,3 -1,1' would set the first molecule as a singlet, second as a triplet and third as a negatively charged singlet state, enter as a string",
    )
    sp = parser.add_mutually_exclusive_group()
    sp.add_argument(
        "--force_rerun",
        action="store_true",
        default=True,
        dest="force_rerun",
        help="Force re-running of Gaussian - NOTE: not widely used.",
    )
    sp.add_argument(
        "--no-force_rerun", action="store_false", default=True, dest="force_rerun"
    )

    return parser

def remove_argument(parser: argparse.ArgumentParser, arg: str):
    """
    When provided a parser and the name of an argument, the argument
    will be removed from the parser. Useful if some scripts are
    incompatible with certain arguments set up by the functions above.

    Parameters
    ----------

    parser : argparse.ArgumentParser
        The argparse parser

    arg : str
        The name of the argument to remove
    """
    # note, you might get conflicts if you reintroduce an argument with the same name/flag
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


class CspyApp:
    """Class for setting up mol-CSPy apps.

    Parameters
    ----------
    app_name : str
        Name of app that will be run

    args : argparse.Namespace
        argparse arguments.
        This might be blank when its passed to this class.

    rank : int
        mpi rank

    **kwargs
        Optional keyword arguments
    """

    def __init__(self, app_name : str, args=argparse.Namespace, rank : int = 0, size : int = 1, **kwargs) -> None:
        self.app_name = app_name
        self.args = args
        self.rank = rank
        self.size = size
        self.worker_data = dict()
        self.mc_para = dict()
        self.use_dmacrys = False
        self.log_level: str = "INFO"

        if self.rank > 0:
            # wait a second for master to do its job
            # this keeps the command line output tidy
            time.sleep(1)

    def enable_logging(self, log_level : str | None = None) -> None:
        """Enable logging on this process.

        Parameters
        ----------
        log_level : str
            Amount/type of information returned by log.
            info, debug, error
        """
        if log_level:
            self.log_level = log_level
        logging.basicConfig(
        level=self.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s'
        )

    def configure_molecules(self, configure_electrostatics : bool = False) -> None:
        """
        Configures worker data relating to molecules

        Parameters
        ----------
        configure_electrostatics : bool
            Whether or not to set up axis, charges, and multipoles

        """
        
        self.worker_data['charges'] = None
        self.worker_data['multipoles'] = None
        self.worker_data['axis'] = None
        if configure_electrostatics:
            if self.args.xyz_files:
                self.default_name = generate_combined_name(self.args.xyz_files)
            else:
                self.default_name = "reopt_structures"

            if self.use_dmacrys:
                # only relevant if dmacrys is being used for minimisations
                if self.args.axis is None:
                    self.args.axis = self.default_name + ".mols"
                    LOG.info("No axis file provided, trying %s", self.args.axis)
                if self.args.charges is None:
                    self.args.charges = self.default_name + "_rank0.dma"
                    LOG.info("No charges file provided, trying %s", self.args.charges)
                if self.args.multipoles is None:
                    self.args.multipoles = self.default_name + ".dma"
                    LOG.info("No multipole file provided, trying %s", self.args.multipoles)
                if not check_multipoles_valid(
                    self.args.xyz_files, self.args.axis, self.args.charges, self.args.multipoles,
                    self.args.potential
                ):
                    LOG.error("No good matches for given multipoles! Exiting...")
                    sys.exit(1)

                self.worker_data['charges'] = Path(self.args.charges).read_text()
                self.worker_data['multipoles'] = Path(self.args.multipoles).read_text()
                self.worker_data['axis'] = Path(self.args.axis).read_text()

            else:
                # these DMACRYS input files are sometimes used, even if we're not using DMACRYS
                if self.args.charges:
                    self.worker_data['charges'] = Path(self.args.charges).read_text()
                if self.args.multipoles:
                    self.worker_data['multipoles'] = Path(self.args.multipoles).read_text()
                if self.args.axis:
                    self.worker_data['axis'] = Path(self.args.axis).read_text()


        try:
            elec_option = CONFIG.get("csp_minimization_step")[-1]["electrostatics"]
        except:
            elec_option = None
            
        if elec_option == "charges":
            self.worker_data['electrostatics'] = self.worker_data['charges']
        elif elec_option == "multipoles":
            self.worker_data['electrostatics'] = self.worker_data['multipoles']


        cutoff = getattr(self.args, "cutoff", CONFIG.get("csp.cutoff"))
        mol_for_cutoff = []
        if self.flex:
            emin = []
            for id, data in enumerate(self.args.databases):
                db = CspDataStoreFlex(data)
                emin += [
                    row[0] for row in db.select("distorted_molecules", ["min(energy)"])
                ]
                if cutoff == "calculate":
                    distorted_mols = [
                        row[0]
                        for row in db.select("distorted_molecules", ["xyz_coordinates"], "limit 1")
                    ]
                    for dist_mol in distorted_mols:
                        mol_for_cutoff.append(Molecule.from_xyz_string(dist_mol))
                db.disconnect()
                self.worker_data["min_energy"] = emin

        if cutoff == "calculate":
            self.cutoff = 15.0
            from scipy.spatial.distance import pdist
            if self.flex:
                # we handled this above already
                pass
            else:
                for x in self.args.xyz_files:
                    mol_for_cutoff.append(Molecule.load(x))

            for mol in mol_for_cutoff:
                if len(mol) == 1:
                    continue
                self.cutoff = max(self.cutoff, 1.5 * np.max(pdist(mol.positions)))
        else:
            self.cutoff = float(cutoff)

        # If we were given xyz files, we can get information about the molecules
        self.args.xyz_files = getattr(self.args, "xyz_files", [])
        if self.args.xyz_files:
            self.bonds = []
            self.bondlength_cutoffs = {}
            for x in self.args.xyz_files:
                mol = Molecule.load(x)
                mol.guess_bonds()
                self.bonds.append(mol.bonds.todense())
                for k, v in mol.neighcrys_bond_cutoffs().items():
                    if k not in self.bondlength_cutoffs or v > self.bondlength_cutoffs[k]:
                        self.bondlength_cutoffs[k] = v

            self.molecules = [Molecule.from_xyz_file(xyz) for xyz in self.args.xyz_files]
            self.formula_unit, self.unique_molecules, self.Zp = find_formula_unit(self.molecules)
            self.asymmetric_unit = [Path(x).read_text() for x in self.args.xyz_files]
            self.worker_data["xyz_names"] = self.args.xyz_files

            # get a list of list where each list contains the unique numerical id of each atom
            self.atom_ids = []
            for xyz in self.args.xyz_files:
                mol_num_atoms = len(Molecule.from_xyz_file(xyz).elements)
                if len(self.atom_ids) == 0:
                    prev_mol_num_atoms = 0
                else:
                    prev_mol_num_atoms = self.atom_ids[-1][-1] + 1
                mol_atom_ids = [i for i in range(prev_mol_num_atoms, prev_mol_num_atoms + mol_num_atoms)]
                self.atom_ids.append(mol_atom_ids)

        else:
            self.bondlength_cutoffs = None
            self.molecules = None
            self.formula_unit = None
            self.unique_molecules = None
            self.Zp = 1
            LOG.warning("No .xyz files provided so assuming Z'= 1 This may not be a problem for you." \
            "If your cell will be standardised, Z' will be calculated on the final crystal anyway.")
            self.atom_ids = None
            self.asymmetric_unit = []

        self.worker_data["bondlength_cutoffs"] = self.bondlength_cutoffs
        self.worker_data["Zp"] = self.Zp
        self.worker_data["atom_ids"] = self.atom_ids
        self.worker_data["asymmetric_unit"] = self.asymmetric_unit

    def configure_minimisation(self) -> None:
        """
        Configures worker data relating to
        geometry optimisations
        """
        CONFIG.set("neighcrys.potential", self.args.potential)
        self.worker_data["check_single_point_energy"] = CONFIG.get("csp.check_single_point_energy")
        self.minimization_settings = {"neighcrys" : {"vdw_cutoff" : self.cutoff,
                                                     "potential" : getattr(self.args, "potential", CONFIG.get("csp.potential"))},
                                    "pmin": {"timeout": CONFIG.get("pmin.timeout")},
                                    "dmacrys": {"timeout": CONFIG.get("dmacrys.timeout")},
                                    "gulp": {"timeout": CONFIG.get("gulp.timeout")},
                                    "minimization_steps": CONFIG.get("csp_minimization_step")}
    
        self.worker_data["minimization"] = self.minimization_settings
        self.energy_cap = getattr(self.args, "energy_cap", None)
        self.energy_window = getattr(self.args, "energy_window", None)

    def configure_clg(self) -> None:
        """
        Configures worker data relating to
        the crystal landscape generator
        """
        self.cluster_properties, self.clusters, self.cluster_weights, self.molecule_properties, self.pot_atom_types, self.pot_charges = [None, None, None, None, None, None]
        if self.args.clg_chomp:
            self.worker_data['mps'] = True
            self.worker_data['aut'] = False
            # hacky fix that offsets reading of databases 
            time.sleep(int(self.rank))
            self.cluster_properties, self.clusters, self.cluster_weights, self.molecule_properties = chomp_data(self.args)
        elif self.args.clg_aut:
            self.worker_data['mps'] = False
            self.worker_data['aut'] = True
            self.pot_atom_types, self.pot_charges, self.molecule_properties = aut_data(self.args.charges, self.args.xyz_files)
        else:
            self.worker_data['mps'] = False
            self.worker_data['aut'] = False

        if self.flex:
            self.worker_data["flex"] = True
        else:
            self.worker_data["flex"] = False
        self.worker_data["clusters"] = self.clusters
        self.worker_data["cluster_properties"] = self.cluster_properties
        self.worker_data["cluster_weights"] = self.cluster_weights
        self.worker_data["molecule_properties"] = self.molecule_properties
        self.worker_data["pot_atom_types"] = self.pot_atom_types
        self.worker_data["pot_charges"] = self.pot_charges
        self.worker_data["nudge"] =  getattr(self.args, "nudge", False)
        self.worker_data["adaptcell"] = getattr(self.args, "adaptcell", False)
        self.worker_data["asi"] = getattr(self.args, "asi", False)

        self.spacegroups = getattr(self.args, "spacegroups", CONFIG.get("csp.spacegroups"))
        self.number_structures = getattr(self.args, "number_structures", CONFIG.get("csp.number_structures"))

    def configure_mc(self) -> None:
        """
        Configures worker data relating to
        monte carlo steps
        """
        self.mc_para["temper"] = CONFIG.get("mc.initial_temper")  # Temperature for basin hopping
        self.mc_para["trial_step"] = CONFIG.get("mc.num_steps")  # Number of maximum steps for one threshold trial
        self.mc_para["num_trials"] = CONFIG.get("mc.num_trials")  # Number of basin hopping trials
        self.mc_para["sat_expand"] = CONFIG.get("mc.sat_expand")  # Whether to sat-expand after perturbation
        self.mc_para["move_all"] = CONFIG.get("mc.move_all")  # Whether to apply all available types of move at one perturbation, don't use unless specific interest
        self.mc_para["auto_prob"] = CONFIG.get("mc.auto_prob")  # Calculate probability of choosing move type according to degrees of freedom
        self.mc_para["auto_cutoff"] = CONFIG.get("mc.auto_cutoff")  # Calculate cutoff of move type, currently only applied to volume expansion and contraction based on number of molecules in unit cell
        self.mc_para["continue_running"] = CONFIG.get("mc.continue_running")  # If true, trial would keep running even after finish
        self.mc_para["on_the_fly"] = CONFIG.get("mc.on_the_fly")  # Whether to use on-the-fly clustering, if True, continue_running will always be False
        self.mc_para["move"] = CONFIG.get("mc.move")  # Specified move details
        self.mc_para["move_scale"] = CONFIG.get("mc.move_scale")  # Scale of move step size with lid states increasing
        self.mc_para["distributed"] = CONFIG.get("mc.distributed", True)  # run Monte-Carlo Trajectories distributed over cores, only for Threshold
        self.mc_para["bonds"] = self.bonds

    def configure_bh(self) -> None:
        """
        Configures worker data relating to
        basin hopping steps
        """
        self.mc_para["raw_s"] = CONFIG.get("basin_hopping.raw_structure")  # Perturb from unminimized structure if True
        self.mc_para["opt_s"] = CONFIG.get("basin_hopping.basin_hopping")  # Minimize after perturbation, meaningless here
        self.mc_para["cluster_s"] = CONFIG.get("mc.on_the_fly") # Whether to use on-the-fly clustering, if True, continue_running will always be False
        self.mc_para["dump_accept"] = CONFIG.get("basin_hopping.dump_accept")  # Only dump accepted structures if True
        self.mc_para["niggli_s"] = CONFIG.get("basin_hopping.niggli_cell")  # Niggli_cell after perturbation

    def configure_thresh(self) -> None:
        """
        Configures worker data relating to
        thresholding steps
        """
        self.mc_para["interval_para"] = CONFIG.get("threshold.interval_para")  # Parameters for deciding intervals (number of steps) under each lid state
        self.mc_para["increase_para"] = CONFIG.get("threshold.increase_para")  # Parameters for deciding increase energy at each lid
        self.mc_para["minimize_s"] = CONFIG.get("threshold.minimize")  # Minimize after perturbation for basin information
        self.mc_para["min_energy"] = CONFIG.get("threshold.min_energy")  # Set minimal energy for threshold, would use minimized energy of initial structure if not


    def configure_mpi_cores(self) -> None:
        """
        Configures how mpi cores are allocated.
        If each worker requires more than 1 core (e.g. when using VASP)
        we should have fewer workers than cores.
        This function will prevent too many workers from spinning up.
        """

        parallel_subprocess = False

        if hasattr(self, "minimization_settings"):
            # check if vasp is being used
            for step in self.minimization_settings["minimization_steps"]:
                if step['kind'] == 'vasp':
                    parallel_subprocess = True
                    break

        if hasattr(self.args, "gaussian-cpus"):
            if self.args.gaussian_cpus > 1:
                parallel_subprocess = True
                LOG.error("Parallel apps in CSPy do not currently support parralel g09 subprosses. Try setting gaussian-cpus to 1.")
                sys.exit(1)

        
        if self.rank == 0:
            self.non_worker_core = True
        else:
            self.non_worker_core = False

        if parallel_subprocess == True:
            cores_per_structure = int(CONFIG.get("vasp.mpi_settings.cores_per_structure"))
            num_non_master_cores = self.size - 1
            if cores_per_structure > num_non_master_cores:
                LOG.error("Cores per structure (%s) exceeds number of non-master cores (%s). Cannot proceed...", cores_per_structure, num_non_master_cores)
                sys.exit()
            # this is the number of structures we can optimise simultaneously
            self.num_workers = math.floor(num_non_master_cores / cores_per_structure)

            LOG.info("Running %s parallel task(s) across %s cores.", self.num_workers, self.size)

            if self.rank > self.num_workers:
                self.non_worker_core = True
                LOG.debug("Core %s is reserved for another process. Will not run worker here.", str(self.rank))

    def configure_app(
            self,
            *,
            clg : bool = False, 
            minimisation : bool = False,
            mc: bool = False,
            bh: bool = False,
            thresh: bool = False,
            flex: bool = False,
            skip_header : bool = False
        ) -> None:
        """
        Configure worker data for a given application.
        Choose which data to configure based on a series
        of booleans passed to the function.

        Parameters
        ----------
        clg : bool
            App uses crystal landscape generator

        minimisation : bool
            App performs geometry optimisations

        mc : bool
            App takes monte carlo steps

        bh : bool
            App takes basin hopping steps

        thresh : bool
            App takes thresholding steps

        flex : bool
            App treats molecules as flexible
        
        skip_header : bool
            Don't print the app header

        """
        self.log_level = getattr(self.args, "log_level", CONFIG.get("csp.log_level"))
        if not skip_header and self.rank==0:
            app_header = AppHeader()
            app_header.print_header()
            app_header.print_app_info(app_name=self.app_name, arguments=self.args)
            app_header.print_toml()
            app_header.print_raw_inputs()

        if self.rank == 0:
            self.enable_logging()
        else:
            # this sets the maximum logging level to CRITICAL
            # without it, logs appear but without their prefix
            logging.disable(logging.CRITICAL)
            pass

        if getattr(self.args, "database_files", None):
            for db_file in self.args.database_files:
                if not os.path.exists(db_file):
                    LOG.error("Database file %s does not exist. Exiting...", db_file)
                    sys.exit()
            self.database_files = self.args.database_files

        if minimisation:
            steps = CONFIG.get("csp_minimization_step")
            for step in steps:
                if step['kind'] == 'dmacrys':
                    self.use_dmacrys = True
                    break

        if flex:
            self.flex = True
        else:
            self.flex = False

        # if xyz_files aren't provided as input, there probably aren't molecules to configure
        if hasattr(self.args, "xyz_files") or self.flex:
            if self.flex or not minimisation:
                self.configure_molecules(configure_electrostatics=False)
            else:
                self.configure_molecules(configure_electrostatics=True)

        if minimisation:
            self.configure_minimisation()
            
        if clg:
            self.configure_clg()

        self.worker_data["keep_files"] = getattr(self.args, "keep_files", False)
        self.worker_data["descriptors"] = CONFIG.get("descriptors")
        self.status_file = getattr(self.args, "status_file", CONFIG.get("csp.status_file"))
        self.chomp_temperature = getattr(self.args, "chomp_temperature", None)

        if mc or bh or thresh:
            self.configure_mc()
        
        if bh or thresh:
            self.configure_bh()

        if thresh:
            self.configure_thresh()
        
        self.worker_data["mc"] = self.mc_para

        # set a default number of workers based on the size of the MPI array
        self.num_workers = self.size - 1
        self.configure_mpi_cores()

        if self.rank > 0:
            # this sets the maximum logging level to self.log_level
            logging.disable(self.log_level)
            self.enable_logging()
