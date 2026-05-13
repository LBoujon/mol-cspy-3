from cspy.db import CspDataStore
import argparse


def main(sys_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "databases",
        nargs="+",
        type=str,
        help="Databases to process given space separated.",
    )
    crystal_columns = {
        "energy": ("e", "Lattice energy"),
        "density": ("d", "Density (g/cm^3)"),
        "id": ("i", "Identifier"),
        "spacegroup": ("g", "Space group"),
        "file_content": ("f", "Res file contents"),
    }
    trial_structure_columns = {
        "minimization_step": ("s", "Minimization step"),
        "trial_number": ("n", "Trial number (seed)"),
        "minimization_time": ("t", "Time for minimization"),
        "metadata": ("md", "metadata"),
    }
    tables = {}
    all_columns = []
    for columns, table in (
        (crystal_columns, "crystal"),
        (trial_structure_columns, "trial_structure"),
    ):
        for col, (short, desc) in columns.items():
            all_columns.append(col)
            long_name = col.replace("_", "-")
            parser.add_argument(
                f"-{short}",
                f"--{long_name}",
                action="store_true",
                default=False,
                help=desc,
            )
            tables[col] = table
    args = parser.parse_args(sys_args)
    query_columns = []
    query_tables = []
    for col in all_columns:
        if not getattr(args, col):
            continue
        table = tables[col]
        if not table in query_tables:
            query_tables.append(table)
        query_columns.append(f"{table}.{col}")

    query_text = f"select {', '.join(query_columns)} from "
    for i in range(len(query_tables)):
        if i == 0:
            query_text += query_tables[i]
        else:
            n = query_tables[i]
            p = query_tables[i - 1]
            query_text += f" join {n} on {n}.id = {p}.id"
    for db in args.databases:
        ds = CspDataStore(db)
        for row in ds.query(query_text):
            print(",".join(str(x) for x in row))


if __name__ == "__main__":
    main()
