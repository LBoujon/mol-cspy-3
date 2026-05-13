import logging
from cspy.db import CspDataStore
from cspy.util.logging_config import FORMATS, DATEFMT
import pandas as pd
import numpy as np
from typing import Optional

LOG = logging.getLogger(__name__)

def print_and_store_string(output_file_lines : list, string : str) -> list:
    output_file_lines.append(string)
    LOG.info(string)

    return output_file_lines


def print_database_info(dbfile : str, log_level : str, coupon_collectors : bool = False, output_fname : Optional[str] = None) -> None:
    """ Scrape information about cspy database and return it to the user

    Parameters
    ----------
    dbfile : str
        Path to database file to read

    log_level : string
        Amount of information to return
    
    coupon_collectors : bool
        enable analysis of coupon collectors problem

    output_fname : str
        Save output to name with this file.
        If none, don't save output.

    """

    output_file_lines = []

    db = CspDataStore(dbfile)
    print_and_store_string(output_file_lines, dbfile)
    counts = {}
    for table in db.table_names:
        if table == "sqlite_sequence":
            continue
        query = f"select count(*) from {table}"
        num_rows = db.query(query).fetchone()[0]
        counts[table] = num_rows

    if not log_level == 'INFO':
        string_batch = []
        string_batch.append("\nTable contents\n".upper())
        string_batch.append("\n{:<18s} {:>10s}".format("table".upper(), "count".upper()))
        for k, v in counts.items():
           string_batch.append(f"{k:<18s} {v:10d}")

        output_file_lines = print_and_store_string(output_file_lines, '\n'.join(string_batch) + '\n\n')

    if counts["equivalent_to"] > 0:
        string_batch = []
        string_batch.append("\nStructure redundancy information\n".upper())
        redundancy = {}
        num_final_mins = db.number_of_final_minimizations()
        num_unique = db.number_of_unique_structures()
        perc_unique = (num_unique / num_final_mins) * 100
        redundancy["Num Final Minimisations"] = int(num_final_mins)
        redundancy["Num Uniques"] = int(num_unique)
        redundancy["Percentage Uniques (%)"] = float(perc_unique)
        for k, v in redundancy.items():
            if isinstance(v, float):
                string_batch.append(f"{k:<23s} {v:>5.2f}")
            else:
                string_batch.append(f"{k:<23s} {v:>5d}")

        output_file_lines = print_and_store_string(output_file_lines, '\n'.join(string_batch) + '\n\n')

        string_batch = []
        string_batch.append("\nUnique structure information\n".upper())
        unique_structures = db.unique_structures(with_file_content=False, with_trial_data=True).fetchall()
        df = pd.DataFrame.from_records(
            unique_structures,
            columns=[
                "id",
                "Space Group",
                "Density (g/cm^3)",
                "Energy (kJ/mol)",
                "Molecule id",
                "step",
                "Sobol seed",
                "Final Minimization Time (s)",
            ],
        )
        string_batch.append(df[["Energy (kJ/mol)", "Density (g/cm^3)"]].describe().round(2).to_string())
        output_file_lines = print_and_store_string(output_file_lines, '\n'.join(string_batch) + '\n\n')

        equiv = np.array(db.equivalent_trial_numbers().fetchall())
        duplicates = {}
        for k, v in equiv:
            if k not in duplicates:
                duplicates[k] = []
            if v > -1:
                duplicates[k].append(v)
        N = len(df)

        if coupon_collectors:
            #https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
            string_batch = []
            string_batch.append("\nCoupon collectors problem\n".upper())
            m = 1
            expected = int(np.log(N) * N + (m - 1) * np.log(np.log(N)) * N + N / 2 + 0.5)
            string_batch.append(f"Expected trials to find {N} structures at least {m} times each ")
            string_batch.append(f"    N log N + {m - 1} N log log N + N / 2 = {expected} trials")
            trial_numbers = np.sort(np.unique(equiv.ravel()))
            trial_numbers = trial_numbers[trial_numbers > 0][:expected]
            max_trial = np.max(trial_numbers)
            df["keep"] = True
            for k, v in duplicates.items():
                if k > max_trial:
                    if all(x > max_trial for x in v):
                        df.loc[df["Sobol seed"] == k, "keep"] = False
            filtered = df[df.keep]
            string_batch.append(
                f"No. unique structures if truncated at {expected} successful trials = {len(filtered)}"
            )
            string_batch.append("\nRetained structures\n".upper())
            string_batch.append(filtered[["Energy (kJ/mol)", "Density (g/cm^3)"]].describe().round(2).to_string())
            string_batch.append("\nMissed structures\n".upper())
            string_batch.append( 
                df.loc[~df.keep, ["Energy (kJ/mol)", "Density (g/cm^3)"]]
                .describe()
                .round(2).to_string()
            )
            
            output_file_lines = print_and_store_string(output_file_lines, '\n'.join(string_batch) + '\n\n')
            
    else:
        string_batch = []
        string_batch.append("\nUnclustered structure information\n".upper())
        structures = db.final_minimizations(with_file_content=False, with_trial_data=True).fetchall()
        df = pd.DataFrame.from_records(
            structures,
            columns=[
                "id",
                "Space Group",
                "Density (g/cm^3)",
                "Energy (kJ/mol)",
                "Molecule id",
                "step",
                "Sobol seed",
                "Final Minimization Time (s)",
            ],
        )
        string_batch.append(df[["Energy (kJ/mol)", "Density (g/cm^3)"]].describe().round(2).to_string())
        output_file_lines = print_and_store_string(output_file_lines, ''.join(string_batch) + '\n\n')

        if coupon_collectors:
            output_file_lines = print_and_store_string(output_file_lines, "Coupon collectors problem analysis only available for clustered database")


    string_batch = []
    string_batch.append("\nFive lowest energy structures".upper())
    df_5_smallest = df.nsmallest(5, 'Energy (kJ/mol)')
    string_batch.append(df_5_smallest[["id", "Space Group", "Energy (kJ/mol)", "Density (g/cm^3)"]].round(2).to_string())

    output_file_lines = print_and_store_string(output_file_lines, '\n'.join(string_batch) + '\n\n')
    db.close()

    if output_fname:
        with open(output_fname, 'w') as f:
            f.writelines(''.join(output_file_lines))

    return df_5_smallest


def main(sys_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "databases",
        nargs="+",
        type=str,
        help="Databases to process given space separated.",
    )
    parser.add_argument(
        "--coupon-collectors",
        action="store_true",
        help="Enable coupon collector analysis",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Write output to file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("INFO", "DEBUG", "ERROR", "WARN"),
        default="INFO",
        help="Control level of logging output",
    )
    args = parser.parse_args(sys_args)
    # logging.basicConfig(
    #     level=args.log_level, format=FORMATS[args.log_level], datefmt=DATEFMT
    # )
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s'
    )
    LOG.info("%d databases", len(args.databases))

    all_df_5_smallest = []

    for db in args.databases:
        if args.keep_output:
            output_fname = db.split('.db')[0] + '-INFO.txt'
        else:
            output_fname = None
        df_5_smallest = print_database_info(db, log_level=args.log_level, coupon_collectors=args.coupon_collectors, output_fname=output_fname)
        all_df_5_smallest.append(df_5_smallest)

    if len(all_df_5_smallest) > 1:
        df_merged = pd.concat(all_df_5_smallest, ignore_index=True, sort=False)
        string_batch = []
        string_batch.append("\nSUMMARISING ALL DATABASES\n".upper())
        string_batch.append("\nFive lowest energy structures".upper())
        df_merged_5_smallest = df_merged.nsmallest(5, 'Energy (kJ/mol)')
        string_batch.append(df_merged_5_smallest[["id", "Space Group", "Energy (kJ/mol)", "Density (g/cm^3)"]].round(2).to_string())

        output_file_lines = print_and_store_string([], '\n'.join(string_batch) + '\n\n')

        if output_fname:
            with open(output_fname, 'w') as f:
                f.writelines(''.join(output_file_lines))

if __name__ == "__main__":
    main()
