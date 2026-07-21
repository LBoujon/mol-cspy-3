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
    parser.add_argument("dbname", nargs=1, type=str, help="Databases to process.")
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
        "-e",
        "--energy-range",
        nargs=2,
        default=[None, None],
        help="Constrain plot to only structures with energy greater than value 1\
              and less than value 2. Value should be float or None.",
    )
    parser.add_argument(
        "-d",
        "--density-range",
        nargs=2,
        default=[None, None],
        help="Constrain plot to only structures with density greater than value 1\
              and less than value 2. Value should be float or None.",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        default=None,
        help="Dump structure matching provided ID.",
    )
    parser.add_argument(
        "--spg",
        type=int,
        default=None,
        help="Dump structure matching provided space group number.",
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
        "--include-duplicates", 
        action="store_true", 
        default=False, 
        help="Dump duplicate structures also",
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

    dbname = args.dbname[0]
    db = CspDataStore(dbname)

    clustered = False
    num_equivalent_to = db.query("select count(*) from equivalent_to").fetchone()[0]
    if num_equivalent_to > 0:
        clustered = True
    if clustered:
        LOG.info("%s is clustered. Scraping only unique structures", dbname)
    else:
        LOG.info("%s is likely unclustered. Scraping all structures.", dbname)

    if args.energy_range[0]:
        LOG.info("Filtering to structures with relative lattice energy between %s and %s kJ/mol.", args.energy_range[0], args.energy_range[1])
    if args.density_range[0]:
        LOG.info("Filtering to structures with density between %s and %s gcm^{3}.", args.density_range[0], args.density_range[1])
    if args.spg:
        LOG.info("Filtering to structures in spacegroup %s.", args.spg)

    # copy structures from this db to another one
    if args.copy_db:
        if args.id is not None:
            LOG.error("Filtering by ID not supported with copy_db. Ignoring this.")

        new_dbname = dbname[:-3]
        if args.energy_range[0]:
            new_dbname += "_" + str(args.energy_range[0]) + "-" + str(args.energy_range[1]) + "kJmol_window"

        if args.density_range[0]:
            new_dbname += "_" + str(args.density_range[0]) + "-" + str(args.density_range[1]) + "gcm3_window"

        if args.spg is not None:
            new_dbname += "_spg" + str(args.spg)

        new_dbname += ".db"

        LOG.info("Copying rows to %s", new_dbname)
        db.copy_unique_structures_within_range(new_dbname, 
                                               energy_min=args.energy_range[0], energy_max=args.energy_range[1], 
                                               density_min=args.density_range[0], density_max=args.density_range[1], 
                                               spacegroup=args.spg, include_duplicates=args.include_duplicates)
        
        db.close()
        
    # dump structures to csv and zip
    else:
        conditions = []
        if args.energy_range[0]:
            conditions.append(("crystal.energy >= (select min(crystal.energy) from crystal) + {} "
                                ).format(args.energy_range[0]))
            conditions.append(("crystal.energy <= (select min(crystal.energy) from crystal) + {} "
                                ).format(args.energy_range[1]))
        if args.density_range[0]:
            conditions.append(("crystal.density >= {} "
                                ).format(args.density_range[0]))
            conditions.append(("crystal.density <= {} "
                                ).format(args.density_range[1]))
        if args.spg is not None:
            conditions.append(("crystal.spacegroup = {} "
                                ).format(args.spg))
        if args.id is not None:
            conditions.append(("crystal.id = \"{}\" "
                                ).format(args.id))

        if conditions:
            property_range_sql = " where " + "and ".join(conditions)
        else:
            property_range_sql = ''

        
        if clustered and not args.include_duplicates:
            query_text = UNIQUE_STRUCTURES_SQL + property_range_sql
        else:
            query_text = ALL_STRUCTURES_SQL + property_range_sql

        dataframe = pd.read_sql(query_text, db.connection)

        db.close()

        LOG.info("Total rows: %d", len(dataframe))
        dataframe.sort_values("energy", inplace=True)
        structure_files = dataframe.pop("file_content")

        if args.parse_metadata:
            import json

            metadata = dataframe.pop("metadata")
            new_cols = pd.io.json.json_normalize(metadata.apply(json.loads))
            dataframe = pd.concat((dataframe, new_cols.reset_index()), axis=1)

        table_file = Path(args.table_output)
        structure_file = Path(args.structure_output)

        def to_latex(filename, **kwargs):
            Path(filename).write_text(dataframe.to_latex(**kwargs))

        table_dispatch = {
            ".csv": dataframe.to_csv,
            ".xlsx": dataframe.to_excel,
            ".h5": dataframe.to_hdf,
            ".hdf5": dataframe.to_hdf,
            ".hdf": dataframe.to_hdf,
            ".pickle": dataframe.to_pickle,
            ".tex": to_latex,
        }
        structure_dispatch = {
            ".zip": write_structures_to_zip,
            ".res": write_structures_to_res,
            ".cif": write_structures_to_cif,
        }
        LOG.info("Writing %d rows to %s", len(dataframe), args.table_output)
        table_dispatch[table_file.suffix](args.table_output, index=False)
        LOG.info("Writing %d structures to %s", len(dataframe), args.structure_output)
        structure_dispatch[structure_file.suffix](
            args.structure_output, dataframe["id"], structure_files
        )
