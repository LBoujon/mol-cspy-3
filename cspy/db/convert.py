import logging
from cspy.db import DataStore
from cspy.util.logging_config import FORMATS, DATEFMT

LOG = logging.getLogger(__name__)

def add_molecule_id_column(filename: str) -> None:
    """
    Adds a new column "molecule_id" to the crystal table in the specified database file.
    If the column already exists, it does nothing.

    Parameters
    ----------
    filename : str
        The name of the database file to modify.
    """
    import shutil
    from os import getcwd
    from os.path import join

    ds = DataStore(filename)
    if "molecule_id" in ds.get_table_columns("crystal"):
        LOG.info(
            'Column "molecule_id" already exists in %s. No changes made.',
            filename,
        )
        ds.close()
        return

    work_dir = getcwd()
    name_new_db = join(
        work_dir, f'{filename.split(".db")[0]}_molid_updated.db'
    )
    try:
        shutil.copy(join(work_dir, filename), name_new_db)
    except Exception as e:
        LOG.info("Error with shutil.copy - %s", e)

    db = DataStore(name_new_db)
    query_inputs = [
        (
            'CREATE TABLE temp('
            '"id" string PRIMARY KEY, '
            '"spacegroup" integer DEFAULT 1, '
            '"density" float, '
            '"energy" float, '
            '"molecule_id" string DEFAULT "Empty - column added to old database", '
            '"file_content" string);'
        ),
        (
            'INSERT INTO temp("id", "spacegroup", "density", "energy", "file_content") '
            "SELECT * from crystal;"
        ),
        "DROP TABLE crystal;",
        "ALTER TABLE temp RENAME TO crystal;",
    ]
    for item in query_inputs:
        db.query(query_text=item)
    db.close()
    LOG.info(
        "Successfully added 'molecule_id' column to %s. Output file is %s.",
        filename,
        name_new_db,
    )


def main(sys_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database_files",
        nargs="+",
        help=(
            "Database files to convert from the old cspy format to new format "
            "(add molecule_id column to)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("INFO", "WARN", "DEBUG", "ERROR"),
        help="Level of logging output",
    )
    args = parser.parse_args(sys_args)
    logging.basicConfig(
        level=args.log_level,
        format=FORMATS[args.log_level],
        datefmt=DATEFMT,
    )
    for filename in args.database_files:
        LOG.info("Trying to add molecule_id column to database file: %s", filename)
        add_molecule_id_column(filename)

if __name__ == '__main__':
    main()
