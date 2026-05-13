from cspy.db.datastore import CspDataStore
from cspy.util.path import Path
import pandas as pd
import argparse
import logging
from cspy.util.logging_config import FORMATS, DATEFMT

LOG = logging.getLogger(__name__)

ALL_STRUCTURES_SQL = (
    "select crystal.*, minimization_step, "
    "trial_number, minimization_time, metadata "
    "from crystal join "
    "(select distinct(id), minimization_step, "
    "trial_number, minimization_time, metadata from trial_structure) T "
    "on crystal.id = T.id "
)

UNIQUE_STRUCTURES_SQL = (
    "select crystal.*, minimization_step, "
    "trial_number, minimization_time, metadata "
    "from crystal join "
    "(select distinct unique_id from equivalent_to) "
    "on crystal.id = unique_id join "
    "(select distinct(id), minimization_step, "
    "trial_number, minimization_time, metadata from trial_structure) T "
    " on T.id = crystal.id "
)

STRUCTURES_ENERGY_SQL = (
    "where crystal.energy <= (select min(crystal.energy) from crystal) + {}"
)


def write_structures_to_zip(filename, ids, structures):
    import zipfile

    with zipfile.ZipFile(filename, mode="w") as zf:
        for structure_id, res_content in zip(ids, structures):
            zf.writestr(f"{structure_id}.res", res_content)


def write_structures_to_res(filename, ids, structures):
    with open(filename, "w") as f:
        f.write("\nEND\n".join(structures))


def write_structures_to_cif(filename, ids, structures):
    from cspy.crystal import Crystal

    LOG.info("Converting SHELX strings to CIF, this might take a while...")
    with open(filename, "w") as f:
        for n, structure in zip(ids, structures):
            c = Crystal.from_shelx_string(structure)
            c.titl = n
            f.write(c.to_cif_string() + "\n")


def main(sys_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("databases", nargs="+", type=str, help="Databases to process.")
    parser.add_argument(
        "-t", "--table-output", type=str, default="structures.csv", help="Output file "
    )
    parser.add_argument(
        "-r",
        "--structure-output",
        type=str,
        default="structures.zip",
        help="Output file ",
    )
    parser.add_argument(
        "-f",
        "--structure-filetype",
        choices=("cif", "res"),
      #  default="res",
        default="cif",
        help="File type for compressed structures",
    )
    parser.add_argument(
        "-d",
        "--include-duplicates",
        action="store_true",
        default=False,
        help="Dump duplicate structures also",
    )
    parser.add_argument(
        "-e",
        "--energy",
        type=float,
        default=None,
        help="Dump structures that are a within the inputted energy from the "
        "global minimum",
    )
    parser.add_argument("--parse-metadata", action="store_true", default=False)
    parser.add_argument(
        "-s", "--sort-by", type=str, default="energy", help="Sort by this column"
    )
    parser.add_argument(
        "--copy-db", 
        action="store_true", 
        default=False, 
        help="Use this option to copy structures within a specified energy window to a new database."
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
    if args.copy_db:
        energy = 10 if args.energy is None else args.energy
        LOG.info("Copying structures within {} kJmol of GM".format(energy))
        dataframes = []
        for dbname in args.databases: #JG - this could be changd as typically only used with 1 db
            ds = CspDataStore(dbname)
            ds.copy_unique_structures_within_range(
                "{}_{}kJmol_window.db".format(dbname[:-3], energy), energy_max = energy
            )
            ds.close()
        exit()

    dataframes = []
    query_text = UNIQUE_STRUCTURES_SQL
    if args.include_duplicates:
        query_text = ALL_STRUCTURES_SQL
    if args.energy is not None:
        query_text += STRUCTURES_ENERGY_SQL.format(args.energy)
    for dbname in args.databases:
        ds = CspDataStore(dbname)
        dataframe = pd.read_sql(query_text, ds.connection)
        if not len(dataframe) == 0:
            pass
        else:
            LOG.warning("Database %s contains no unique structures. Trying all structures instead.", dbname)
            query_text = ALL_STRUCTURES_SQL
            if args.energy is not None:
                query_text += STRUCTURES_ENERGY_SQL.format(args.energy)
            dataframe = pd.read_sql(query_text, ds.connection)
        dataframes.append(dataframe)
        ds.close()
    LOG.info("Total rows: %d", sum(len(x) for x in dataframes))
    all_data = pd.concat(dataframes)
    all_data.sort_values("energy", inplace=True)
    structure_files = all_data.pop("file_content")

    if args.parse_metadata:
        import json

        metadata = all_data.pop("metadata")
        new_cols = pd.io.json.json_normalize(metadata.apply(json.loads))
        all_data = pd.concat((all_data, new_cols.reset_index()), axis=1)

    table_file = Path(args.table_output)
    structure_file = Path(args.structure_output)

    def to_latex(filename, **kwargs):
        Path(filename).write_text(all_data.to_latex(**kwargs))

    table_dispatch = {
        ".csv": all_data.to_csv,
        ".xlsx": all_data.to_excel,
        ".h5": all_data.to_hdf,
        ".hdf5": all_data.to_hdf,
        ".hdf": all_data.to_hdf,
        ".pickle": all_data.to_pickle,
        ".tex": to_latex,
    }
    structure_dispatch = {
        ".zip": write_structures_to_zip,
        ".res": write_structures_to_res,
        ".cif": write_structures_to_cif,
    }
    LOG.info("Writing %d rows to %s", len(all_data), args.table_output)
    table_dispatch[table_file.suffix](args.table_output, index=False)
    LOG.info("Writing %d structures to %s", len(all_data), args.structure_output)
    structure_dispatch[structure_file.suffix](
        args.structure_output, all_data["id"], structure_files
    )
