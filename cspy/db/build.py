from cspy.db.datastore import CspDataStore
import argparse
import logging
import os
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.util.path import Path
from cspy.formats.shelx import parse_shelx_file_content
from cspy.crystal.space_group import SpaceGroup

LOG = logging.getLogger(__name__)


def build_db_from_shelx_string(file_contents: list[str], 
                               dbname: str, 
                               molid: str = "No mol_id exists", 
                               allow_dummy: bool = False) -> None:
    
    """ Iterative over a list of crystal structures in shelx format
    and add them to a database.

    Parameters
    ----------
    file_contents : list[str]
        A list where each element is a shelx string 
        describing a crystal structure

    dbname : str
        Name of database to be created or added to
    
    molid : str
        Id of molecule in database

    allow_dummy : bool
        Logic that defines what to do if an 
        energy and/or density can't be read.
        If True, set energy and density to 0
        If False, skip structure.

    """

    crystals = {
            "id":[],
            "spacegroup":[],
            "energy":[],
            "density":[],
            "molecule_id":[],
            "file_content":[],
            }
    
    for file_content in file_contents:
        shelx_dict = parse_shelx_file_content(file_content)
        space_group = SpaceGroup.from_symmetry_operations(
            shelx_dict["SYMM"], expand_latt=shelx_dict["LATT"]
        )
        spacegroup_number = space_group.international_tables_number

        header = file_content.split('\n')[0].split()
        titl = header[1]
        try:
            energy = float(header[2])
            density = float(header[3])
        except:
            if allow_dummy:
                energy = float(0)
                density = float(0)
            else:
                LOG.error("Can't process energy and/or density from file, and allow-dummy is diabled. Skipping...")
                continue

        crystals["id"].append(titl)
        crystals["spacegroup"].append(spacegroup_number)
        crystals["energy"].append(energy)
        crystals["density"].append(density)
        crystals["molecule_id"].append(molid)
        crystals["file_content"].append(file_content)

    num_crystals = len(crystals["id"])

    trial_data = {
        "id":crystals["id"],
        "minimization_step":[0 for _ in range(num_crystals)],
        "trial_number":[0 for _ in range(num_crystals)],
        "valid":[True for _ in range(num_crystals)],
        "minimization_time":[0.0 for _ in [1 for _ in range(num_crystals)]],
        "metadata":["None" for _ in [1 for _ in range(num_crystals)]]
    }

    if os.path.isfile(dbname):
        LOG.warning("%s already existins. Appending new data to existing database. May overwrite existing data if file names clash.", dbname)

    new_db = CspDataStore(dbname)
    new_db.insert_many("crystal", crystals, replace=True)
    new_db.insert_many("trial_structure", trial_data, replace=True)
    new_db.close()


def build_db_from_shelx_files(files: list[str], 
                               dbname: str, 
                               molid: str = "No mol_id exists", 
                               allow_dummy: bool = False) -> None:
    
    """ Iterative over a list of res files, scrape their contents
    and add them to a database. This function calls build_db_from_shelx_string
    for the construction of the database.

    Parameters
    ----------
    files : list[str]
        A list where each element is a filename
        of a shelx file containing  a crystal structure

    dbname : str
        Name of database to be created or added to
    
    molid : str
        Id of molecule in database

    allow_dummy : bool
        Logic that defines what to do if an 
        energy and/or density can't be read.
        If True, set energy and density to 0
        If False, skip structure.

    """

    file_contents = []
    for filename in files:
        ext = filename.split('.')[-1]
        if not ext == "res":
            LOG.error("File type %s is not shelx. Try .res instead", ext)
        
        p = Path(filename)
        file_contents.append(p.read_text())

    build_db_from_shelx_string(file_contents, dbname, molid, allow_dummy)
    

def main(sys_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", type=str, help="Databases to process.")
    parser.add_argument(
        "-f",
        "--files",
        nargs='*', 
        default=None,
        help="Res files to input to database",
    )
    parser.add_argument(
        "-m",
        "--molid",
        type=str,
        default="No mol_id exists",
        help="Set value for mold ID if known.",
    )
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="Allow dummy data if energy, density, or spacegroup are missing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("INFO", "DEBUG", "ERROR", "WARN"),
        default="INFO",
        help="Control level of logging output",
    )

    args = parser.parse_args(sys_args)
    logging.basicConfig(
        level=args.log_level, format=FORMATS[args.log_level], datefmt=DATEFMT
    )

    build_db_from_shelx_files(args.files, args.dbname, args.molid, args.allow_dummy)