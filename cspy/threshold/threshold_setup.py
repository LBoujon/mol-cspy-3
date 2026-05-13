"""
Module to aid with the setup of crystal structures to run `cspy-threshold` jobs, 
mainly creating supercells in a consistent way and with the same number of molecules
accross all initial crystals.

Functions
---------

- `save_crystals`: Save a list of crystal objects to .res files
- `calculate_bonding`: Calculates the bonds of a list of molecules and returns them
as `numpy.matrix` objects
- `check_bonding`: Check that the molecules in a list of crystals match the 
bonding of some template molecules. If they do not match it is possible to try
and fix it
- `check_multipoles`: Check that a multipoles file can be mapped to a list of
crystals 
- `load_crystals`: Load crystals from files or cspy databases
- `load_molecules`: Load molecules from files
"""

from cspy import Molecule, Crystal
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.chem.multipole.distributed_multipoles import DistributedMultipoles
from cspy.db import CspDataStore
from cspy.crystal.supercell_creator import SuperCellsCreatorBC, METHOD_CLASSES
from cspy.crystal import NeighcrysAxis
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union
from typing_extensions import Literal
import numpy as np
import sys


LOG = logging.getLogger(__name__)


def save_crystals(crystals: List[Crystal], add_to_end: str) -> None:
    """
    Save the crystals to .res files. The filenames are of the
    form: "{crystal.titl}{add_to_end}.res".

    Arguments
    ---------

    crystals : List[Crystal]
        List of crystals to save

    add_to_end : str
        Append this text to the end of the filename, before the .res
    """
    for crystal in crystals:
        filename = f"{crystal.titl}{add_to_end}.res"
        LOG.info(f"Saving {crystal.titl} to {filename}")
        crystal.save(filename)


def calculate_bonding(molecules: List[Molecule]) -> List[np.matrix]:
    """
    Get the bonds of the input molecules as `numpy.matrix` objects,
    the molecular bonds are calulated with `Molecule.guess_bonds` unless
    `Molecule.bonds` is already set.

    Arguments
    ---------

    molecules : List[Molecule]
        The list of molecules

    Returns
    -------

    bonds : List[numpy.matrix]
        A list of the bonds of the input molecules, as a numpy matrix
    """
    LOG.debug(f"Calculating bonding of molecules: {molecules}")
    bonds = []
    for mol in molecules:
        if mol.bonds is None:
            mol.guess_bonds()
        bonds.append(mol.bonds.todense())

    return bonds


def check_bonding(crystals: List[Crystal], molecules: List[Molecule], tol: float = 1e-2, 
                  fix: bool = False, fix_method: Literal["match_ordering", "replace_molecules"] = "replace_molecules") -> Tuple[bool, List[Crystal]]:
    """
    Check that all molecules in all crystals match the bonding
    of the molecules supplied. If the `fix` argument is set to True,
    the code will try to fix any molecules that do not match the 
    template ones.

    Arguments
    ---------

    crystals : List[Crystal]
        The list of crystals to check

    molecules : List[Molecule]
        List of template molecules

    tol : float
        The tolerance in bond length to accept. Default is 1e-2 (same
        as in the MC threshold move method of ThresholdWorker)

    fix : bool
        Try and fix any bonding issues. Default is False

    fix_method : Literal["match_ordering", "replace_molecules"]
        The fix method to use. Default is replace_molecules

    Returns
    -------

    all_valid, checked_crystals : Tuple[bool, List[Crystals]]
        Returns whether all crystals pass the check of comparing with the template molecules, 
        and returns the list of checked crystals, which if the `fix` argument is set to True will
        include the modified crystals 
    """
    if not fix_method in ["match_ordering", "replace_molecules"]:
        raise ValueError("fix_method must be one of: match_ordering, replace_molecules")

    bonds = calculate_bonding(molecules)
    all_valid = True
    checked_crystals = []
    for crystal in crystals:
        crys_molecules = crystal.symmetry_unique_molecules()
        valid = True
        for mol in crys_molecules:
            for bond in bonds:
                try:
                    if np.allclose(mol.bonds.todense(), bond, atol=tol):
                        break
                except ValueError:
                    continue
            else:
                valid = False
        
        if valid:
            LOG.info(f"No bonding issues found with {crystal.titl}")
            checked_crystals.append(crystal)
        else:
            LOG.warning(f"Molecular bonding issue found with {crystal.titl}")
            all_valid = False
            if fix:
                LOG.info(f"Attempting to fix bonding by replacing molecules in crystal")
                try:
                    if fix_method == "replace_molecules":
                        new_crystal = crystal.replace_molecules(molecules, reorder_to="other")
                    else:
                        new_crystal = crystal.match_atom_ordering_to(molecules)
                    new_crystal.titl = crystal.titl
                    checked_crystals.append(new_crystal)
                    all_valid = True
                except ValueError:
                    LOG.warning("Failed to fix bonding")
    
    return all_valid, checked_crystals


def check_multipoles(crystals: List[Crystal], multipoles: Path, tol: float = 1e-3) -> bool:
    """
    Check that multipoles can be mapped to crystals and resulting
    RMSD is below or equal to the set `tol`.

    Arguments
    ---------

    crystals : List[Crystal]
        A list of crystals

    multipoles : Path
        Path pointing to the multipoles file

    tol : float
        The tolerance to accept a multipoles mapping as correct. Default is 1e-3

    Returns
    -------

    all_valid : bool
        Returns if all crystals can be mapped to the multipoles file
    """
    mults = DistributedMultipoles.from_dma_file(multipoles)
    all_valid = True
    for crystal in crystals:
        mapping: List[Tuple[int, float, bool]] = crystal.map_multipoles(mults)
        max_rmsd = max(mapping, key=lambda x: x[1])[1]
        if max_rmsd > tol:
            all_valid = False
        else:
            LOG.info(f"Multipoles can be mapped to {crystal.titl} with a max RMSD of {max_rmsd}")

    return all_valid


def check_molecular_axis(crystals: List[Crystal], axis: Path) -> bool:
    """
    Check that a Neighcrys axis can be used with all the crystals passed.

    Arguments
    ---------

    crystals : List[Crystal]
        A list of crystals

    axis : Path
        Path pointing to the axis file

    Returns
    -------

    all_valid : bool
        Returns if the axis file has no issues being read and mapped to the crystals
    """
    axis_contents = axis.read_text()
    all_valid = True
    for crystal in crystals:
        try:
            nc_axis = NeighcrysAxis(crystal, file_content=axis_contents, gen=True) # This is too basic and there are no checks whether the atoms are
                                                                                   # at the distance that the file defines. Also it will generate a 
                                                                                   # valid axis if the labels in the input one do not match any molecules
            if len(nc_axis.molecular_axes) != len(crystal.symmetry_unique_molecules()):
                LOG.warning(f"Number of molecular axes in {axis.name} does not match number of symmetry unique molecules in {crystal.titl}")
                all_valid = False
            LOG.info(f"No issues with molecular axis in crystal: {crystal.titl}")
        except:
            LOG.info(f"Issue with molecular axis {axis.name} in crystal: {crystal.titl}")
            all_valid = False
            
    return all_valid


def load_crystals(files: List[Path], excluded_spgs: Union[List[int], None] = None, 
                  max_energy: Union[float, None] = None, num_structures: Union[int, None] = None) -> List[Crystal]:
    """
    Load crystals from file or from databases.

    Crystals in space groups `excluded_spgs` will be ignored and excluded from the 
    return data.

    If both `max_energy` and `num_structures` is provided, only up to
    `num_structures` many crystals will be taken within `max_energy` of the global minimum.

    Arguments
    ---------

    files : List[Path]
        Path to a crystal file or cspy database

    excluded_spgs : Union[List[int], None] 
        List of space group numbers that will be excluded from crystals
        read. Values must be in the range [0-230]. If None is passed 
        no crystals are ignored. Default is None
    
    max_energy : Union[float, None]
        Only used when reading data from databases, take the crystals from
        a database that have energy below that from the global minimum. Default
        is None

    num_structures : Union[float, None]
        Only used when reading data from databases, take this many number of
        structures from the database in order of increasing energy. Default
        is None

    Returns
    -------

    crystals : List[Crystal]
        A list of the loaded crystals

    Raises
    ------

    ValueError
        When both `max_energy` and `num_structures` are None and a database
        file is passed to read crystals from. When one of the values passed in
        `excluded_spgs` is not in the range [0-230]. When one of the items in
        `files` does not have a valid file extension (.db , .res, .cif)

    TypeError
        If any space group value in `excluded_spgs` is not of type int, or if `excluded_spgs`
        is not a list
    """
    if excluded_spgs is not None:
        if not isinstance(excluded_spgs, list):
            raise TypeError("excluded_spgs must be of type List")
        for sg in excluded_spgs:
            if not isinstance(sg, int):
                raise TypeError(f"{sg} is not int: {type(sg)}")
            if sg > 230 or sg < 1:
                raise ValueError(f"List of excluded space groups contains a value outside of the accepted range [0-230]: {sg}")
        LOG.info(f"Excluding crystals in space groups: {excluded_spgs}")
    crystals = []
    for file in files:
        if file.name.endswith(".db"):
            if max_energy is None and num_structures is None:
                raise ValueError("`max_energy` and `num_structures` are both None so I don't know how to get the data from the database")
            
            db = CspDataStore(str(file))
            structs_cursor = db.unique_structures(with_file_content=True, fallback=True, with_trial_data=False,
                                                  max_energy=max_energy, excluded_spgs=excluded_spgs)
            if num_structures is None:
                selected = structs_cursor.fetchall()
            else:
                selected = structs_cursor.fetchmany(num_structures)
            LOG.debug(f"Obtained {len(selected)} crystals from database {db.filename}")
            db.disconnect()
            
            for result in selected:
                crystal_txt = result[-1]
                c: Crystal = Crystal.from_shelx_string(crystal_txt)
                LOG.debug(f"Read crystal ({c}) from database ({file})")
                crystals.append(c)
        elif file.suffix in [".cif", ".res"]:
            c: Crystal = Crystal.load(file)
            # set the name to the filename
            c.titl = file.name[:-4]
            LOG.debug(f"Read crystal: {c}")
            if excluded_spgs is not None and c.space_group.international_tables_number in excluded_spgs:
                LOG.debug(f"Crystal ({c}) is in one of the excluded space groups, not adding to crystal list")
                continue
            crystals.append(c)
        else:
            raise ValueError(f"File does not have a valid extension: {file}")

    return crystals


def load_molecules(files: List[Path]) -> List[Molecule]:
    """
    Load molecules from files.

    Arguments
    ---------

    files : List[Path]
        The Paths of the files containing the molecules

    Returns
    -------

    mols : List[Molecule]
        A list of the loaded molecules
    """
    mols = []
    for file in files:
        mols.append(Molecule.load(file))
    
    return mols


class ArgparseTypeValidation():
    @staticmethod
    def valid_crystal_file(file_str: str) -> Path:
        """
        Function to check that the input file exists and has a
        `.cif`, `.res`, or `.db` extension.
        """
        file = Path(file_str)

        if file.is_file():
            if file.name.endswith(".res") or file.name.endswith(".cif") or file.name.endswith(".db"):
                return file
            
            raise argparse.ArgumentTypeError(
                f"Crystal file does not have correct extension: .cif, .res, or .db"
            )

        raise argparse.ArgumentTypeError(f"Crystal file does not exist: {file_str}")


    @staticmethod
    def valid_molecule(file_str: str) -> Path:
        """
        Function to check that the input file exists and has a
        `.cif` or `.res` extension.
        """
        file = Path(file_str)

        if file.is_file():
            if file.name.endswith(".xyz"):
                return file
            
            raise argparse.ArgumentTypeError(
                f"Molecule file does not have correct extension: .xyz"
            )

        raise argparse.ArgumentTypeError(f"Molecule file does not exist: {file_str}")


    @staticmethod
    def valid_file(file_str: str) -> Path:
        """
        Function to check that the input file exists.
        """
        file = Path(file_str)

        if file.is_file():
            return file

        raise argparse.ArgumentTypeError(f"File does not exist: {file_str}")


    @staticmethod
    def valid_csv(file_str: str) -> str:
        """
        Check that the file ends with .csv and add it if necessary
        """
        if not file_str.endswith(".csv"):
            file_str += ".csv"

        return file_str

    
    @staticmethod
    def valid_spg_num(value: str) -> int:
        """
        Check value is int and in range [0-230]
        """
        try:
            val = int(value)
        except:
            raise argparse.ArgumentTypeError(f"{value} is not an integer")

        if val < 1 or val > 230:
            raise argparse.ArgumentTypeError(f"{value} is not in the accepted range [0-230]")
        
        return val


class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def parse_args(args: Union[List[str], None], name: Union[str, None]) -> argparse.Namespace:
    """
    Parse the arguments of the script.
    """
    if name is None:
        name = __file__
        
    parser = argparse.ArgumentParser(
        prog=name,
        description=f"""
Program that sets up the crystals to carry out a cspy-threshold job.
P1 supercells will be created for all input crystal files and databases.

It will check that the atom ordering in the crystals matches that of the 
molecule(s) supplied ('--xyz'). If a multipoles file is passes ('-m') the
program will check that mapping the multipoles gives low RMSD.

It can also be used to check which supercell sizes are available with
different generation methods by setting the '--table' flag.""",
        epilog=f"""
Examples:
# Create the smallest supercell of all crystals with up to 50 molecules
# using the 'double_shortest' method
> {name} --smallest --limit 50 --method double_shortest [CRYSTALS]

# Create supercells with 8 molecules if possible and check that the bonding
# between the given molecule and the crystals doesn't change and that 
# multipoles can be mapped correctly
> {name} -z 8 --xyz [MOLECULE] -m [MULTIPOLES] [CRYSTALS]

# Print to screen the number of molecules up to 100 for which supercells
# for all input crystals can be created using the default method.
> {name} --dry --table -tz 100 [CRYSTALS]
        """,
        formatter_class=MyFormatter
    )
    parser.add_argument(
        "crystals",
        nargs="+",
        type=ArgparseTypeValidation.valid_crystal_file,
        help="Crystal files (.res or .cif) or CSP databases (.db)",
    )
    parser.add_argument(
        "--excluded-spgs",
        nargs="+",
        type=ArgparseTypeValidation.valid_spg_num,
        default=None,
        help="""Crystals in these space groups will be ignored. 
Values must be in the range [0-230]""",
    )
    db_read_group = parser.add_argument_group(
        "Read Crystals from Databases",
        """Settings to modify how crystals
are extracted from a database"""
    )
    db_read_group.add_argument(
        "-e",
        "--max-energy",
        type=float,
        default=None,
        help="""Select structures within this energy 
of the global minimum from any database files""",
    )
    db_read_group.add_argument(
        "-n",
        "--num-structures",
        type=int,
        default=None,
        help="""Select this many number of structures
at most from a database, in order of 
increasing energy""",
    )
    check_group = parser.add_argument_group(
        "Check Validity", 
        """Provide these files to check if they can be mapped to the crystals. 
Bonding issues may be fixed using the --fix argument"""
    )
    check_group.add_argument(
        "--xyz",
        nargs="+",
        type=ArgparseTypeValidation.valid_molecule,
        help=""".xyz file of molecule(s) that will be 
supplied to cspy-threshold""",
    )
    check_group.add_argument(
        "-c",
        "--charges",
        type=ArgparseTypeValidation.valid_file,
        help="Rank0 multipole file",
    )
    check_group.add_argument(
        "-m",
        "--multipoles",
        type=ArgparseTypeValidation.valid_file,
        help="RankN multipole file",
    )
    check_group.add_argument(
        "-a",
        "--axis",
        type=ArgparseTypeValidation.valid_file,
        help="Molecular axis file",
    )
    check_group.add_argument(
        "--fix",
        action="store_true",
        help="""Try and fix some possible issues""",
    )
    mol_num_group = parser.add_argument_group(
        "Number of Molecules",
        "Modify the target number of molecules in the final supercells"
    )
    mol_num_group.add_argument(
        "-z",
        "--molecules",
        type=int,
        default=None,
        help="Number of molecules in the P1 unit cells",
    )
    mol_num_group.add_argument(
        "--smallest",
        action="store_true",
        help="""Create the smallest supercell 
that accommodates all initial structures provided""",
    )
    mol_num_group.add_argument(
        "--limit",
        type=int,
        default=200,
        help="""Limit of number of molecules 
in supercell to check when setting --smallest""",
    )
    mol_num_group.add_argument(
        "--niggli",
        action="store_true",
        help="""Calculate niggli cells of input
crystals before finding smallest supercells""",
    )
    table_group = parser.add_argument_group(
        "Table Output",
        """Check multiple target number of molecules and save results to csv.
Ignores the --smallest and --molecules arguments if set."""
    )
    table_group.add_argument(
        "--table",
        action="store_true",
        help="""Create .csv file with permitted 
supercell Z values. If this flag is set, 
all other flags are ignored except 
--table-output and --table-z""",
    )
    table_group.add_argument(
        "-tz",
        "--table-z",
        type=int,
        default=100,
        help="Max number Z to test in table output",
    )
    table_group.add_argument(
        "-to",
        "--table-output",
        type=ArgparseTypeValidation.valid_csv,
        default="table.csv",
        help="""Table output filename ending. The name 
of the method will be included at the beginning""",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CLASSES.keys(),
        default="grow_shortest",
        help="""Set the method to create the supercells:
- grow_shortest: Each step grow the supercell by 1 
  along the shortest direction.
- double_shortest: Each step double the length of 
  the supercell along the shortest diresction.
- split: Not very stable but can create more
  supercells than grow_shortest and 
  double_shortest.
- prime_fact: Like split method but has more
  reasonable outputs in general.
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="_thresh_setup",
        help="String to add to the end of output filenames",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="""Dry run, do not save the 
P1 crystals to output files""",
    )
    parser.add_argument(
        "--ignore",
        action="store_true",
        help="""Ignore any errors raised by bond 
and multipole mapping checks""",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        choices=("INFO", "DEBUG", "WARNING", "ERROR"),
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args(args)


def main(args: Union[List[str], None] = None, name: Union[str, None] = None) -> None:
    """
    App code
    """
    args: argparse.Namespace = parse_args(args, name)
    logging.basicConfig(
        format=FORMATS[args.log_level], datefmt=DATEFMT, level=args.log_level
    )

    # load the crystals and get number of molecules in the unit cell
    LOG.info(f"Reading crystals from: {[str(c) for c in args.crystals]}")
    try:
        initial_crystals: List[Crystal] = load_crystals(args.crystals, args.excluded_spgs, args.max_energy, args.num_structures)
    except ValueError:
        LOG.error("If you are reading crystals from a database you must set either or both '--max-energy' or '--num-structures'!")
        sys.exit(1)
    LOG.info(f"Read {len(initial_crystals)} crystals")

    if len(initial_crystals) == 0:
        LOG.warning("There are no crystals to process. Doing nothing")
        sys.exit(0)

    # create instance of supercell method class
    supercell_creator: SuperCellsCreatorBC = METHOD_CLASSES[args.method](initial_crystals, args.niggli)

    if args.table:
        # create a table 
        LOG.info(f"Calculating possible supercells with up to {args.table_z} molecules with method '{args.method}'")

        # calculate supercells
        supercells = supercell_creator.check_targets(
            start=supercell_creator.lcm,
            end=args.table_z,
            increment=supercell_creator.max_num_molecules
        )

        # save the results to csv file
        if args.dry:
            LOG.info("Dry run, no output files created, printing to screen instead")
            for mols, supercell in supercells:
                if not supercell is None:
                    LOG.info(f"{mols} mols: {supercell}")
        else:
            supercell_creator.save_mutliple_results_to_csv(supercells, args.table_output)

        # finish execution
        sys.exit(0)

    # find supercells
    if args.smallest:
        LOG.debug("Iterating over target number of molecules in supercell until no issues are found with any starting crystal")
        target = supercell_creator.lcm
        increase = supercell_creator.max_num_molecules
        if target > args.limit:
            LOG.warning(f"Increase the '--limit' value (currently {args.limit})")
            sys.exit(1)

        while target < args.limit:
            LOG.debug(f"Trying {target} molecules")
            supercells = supercell_creator.check_target(target)
            if supercells != None:
                break

            target += increase

        if not supercells is None:
            LOG.info(f"Found supercells for all crystals with {target} molecules")
    else:
        if not args.molecules is None:
            target = args.molecules
        else:
            target = supercell_creator.lcm
        LOG.info(f"Checking for supercells with {target} molecules")
        supercells = supercell_creator.check_target(target)

    if supercells is None:
        LOG.warning("Unable to find valid supercells for all crystals")
        LOG.info("Try setting '--molecules MOLECULES' or '--smallest'")
        sys.exit(1)
    else:
        for crystal, supercell in zip(supercell_creator.crystals, supercells):
            LOG.info(f"Valid {supercell} found for {crystal.titl}")

    # create P1 crystals
    p1_crystals = supercell_creator.get_P1_supercell_crystals()

    issues_found = False
    if args.xyz:
        # read molecules
        if not isinstance(args.xyz,list):
            args.xyz = [args.xyz]
        molecules = load_molecules(args.xyz)

        # Check that the input crystals have same molecular bonding
        all_valid, p1_crystals = check_bonding(p1_crystals, molecules, tol = 1e-2, fix = args.fix)
        
        if not all_valid:
            issues_found = True
            if args.ignore:
                LOG.warning("Bonding issues found, ignoring them...")
            else:
                LOG.info("Stopping execution due to bonding issues found...")
                if not args.fix:
                    LOG.info("Try setting the '--fix' option")
                sys.exit(1)
    else:
        LOG.info("No molecules provided to check bonding")

    if args.multipoles:
        # check that multipoles have low RMSD with crystals
        all_valid = check_multipoles(p1_crystals, args.multipoles)

        if not all_valid:
            issues_found = True
            if args.ignore:
                LOG.warning("Multipole mapping issues found, ignoring them...")
            else:
                LOG.info("Stopping execution due to high RMSD when mapping multipoles...")
                sys.exit(1)
    else:
        LOG.info("No multipoles file provided")

    if args.axis:
        all_valid = check_molecular_axis(p1_crystals, args.axis)

        if not all_valid:
            issues_found = True
            if args.ignore:
                LOG.warning("Molecular axis issues found, ignoring them...")
            else:
                LOG.info(f"Stopping execution due to molecular axis issues with {args.axis}...")
                sys.exit(1)
    else:
        LOG.info("No molecular axis file provided")

    # If execution reaches this point files should be properly setup
    if not issues_found:
        LOG.info("No issues found with the crystals")

    if not args.axis and not args.multipoles:
        LOG.warning(f"Remember that the output crystals are in the P1 space group and have {target} molecules")
        LOG.warning(f"You might need to run `cspy-dma` again to get the correct molecular axis, multipoles and charges files")

    # save crystals
    if args.dry:
        LOG.info("Dry run, no output files created")
    else:
        save_crystals(p1_crystals, args.output)


if __name__ == "__main__":
    main()
    