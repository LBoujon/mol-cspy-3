import logging
from cspy.db import CspDataStore
from cspy.util.logging_config import FORMATS, DATEFMT
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, Union, Optional

def count_equivalents(db: CspDataStore) -> dict:
    """ Count the number of structures that are equivalent to 
    each unique structure in the db.

    Parameters
    ----------
    db : CspDataStore type
        CSP database

    Returns
    -------
    num_equivalents : dict()
        keys are ids of structures 
        values are the number of equivalent structures
    """
    num_equivalents = dict()
    results = db.equivalent_ids()
    for result in results:
        unique, equivalent = result
        if not unique in num_equivalents.keys():
            num_equivalents[unique] = 0
        if equivalent == '-1':
            num_equivalents[unique] = 1
        else:
            num_equivalents[unique] += 1

    return num_equivalents


def scrape_structures_from_dbs(databases: list, 
                               plot_equivalents: bool, 
                               spg_to_plot: Union[None, list], 
                               ignore_clustering : bool) -> dict:
    """ scrape structure data from database files

    Parameters
    ----------
    plot_equivalents : bool
        scrape number of equivalents so that structures can be coloured by number of equivalent structures

    spg_to_plot : bool
        collect structures by spacegroup number instead of by database name

    Returns
    -------
    database_content : dict()
        keys are database file names or spacegroup numbers
        values are dataframes containing structure data
    """

    database_content = {}

    for dbfile in databases:
        db = CspDataStore(dbfile)

        clustered = False
        # checking to see if equivalent_to entries exist
        num_equivalent_to = db.query("select count(*) from equivalent_to").fetchone()[0]
        if num_equivalent_to > 0:
            clustered = True
        if clustered and not ignore_clustering:
            LOG.info("%s is clustered. Plotting only unique crystals", dbfile)
            structures = db.unique_structures(with_file_content=False, with_trial_data=False).fetchall()
        else:
            if clustered:
                LOG.info("%s is clustered, but ignoring it. Plotting all final minimisations", dbfile)
            else:
                LOG.info("%s is likely unclustered. Plotting all final minimisations", dbfile)
            structures = db.final_minimizations(with_file_content=False, with_trial_data=False).fetchall()

        df = pd.DataFrame.from_records(
            structures,
            columns=[
                "id",
                "Space Group",
                "Density (g/cm^3)",
                "Energy (kJ/mol)",
                "Molecule id"
            ],
        )

        if plot_equivalents:
            if not clustered:
                LOG.error("Database isn't clustered. Can't plot equivalent crystals.")
            else:
                num_equivalents = count_equivalents(db)
                df['num_equivalents'] = df['id'].map(num_equivalents)

        database_content[dbfile] = df

    if isinstance(spg_to_plot, list):
        # get list of all spacegroups which appear in dataset
        if len(spg_to_plot) == 0:
            all_spgs = []
            for dbfile in database_content.keys():
                df = database_content[dbfile]
                spgs = df['Space Group'].unique()
                for spg in spgs:
                    if not spg in all_spgs:
                        all_spgs.append(spg)
        else:
            all_spgs = [int(spg) for spg in spg_to_plot]

        database_content_by_spg = {}
        for spg in all_spgs:
            df_spg = None
            for dbfile in database_content.keys():
                df = database_content[dbfile]
                df['keep'] = df['Space Group'].apply(lambda x: True if x == spg else False)
                df_db_spg = df[df['keep']==True]
                df = df.drop('keep', axis=1)
                df_db_spg = df_db_spg.drop('keep', axis=1)
                if not isinstance(df_spg, pd.DataFrame):
                    df_spg = deepcopy(df_db_spg)
                else:
                    df_spg = pd.concat([df_spg, df_db_spg])
            database_content_by_spg[spg] = df_spg
        return database_content_by_spg

    else:
        return database_content


def filter_dataframe(database_content: dict, 
                     column: str, 
                     min_value: Optional[float] = None, 
                     max_value: Optional[float] = None) -> dict:
    """ Remove rows from dataframe that do not satisfy criteria for column

    Parameters
    ----------
    database_content : dict
        keys are labels for data series
        values are dataframes containing structure data

    column : string
        name of column in dataframe
    
    min_value : float
        minimum allowed value of column for any given row

    max_value : float
        maximum allowed value of column for any given row

    Returns
    -------
    database_content : dict()
        keys are labels for data series
        values are dataframes containing structure data
    """
    if min_value:
        min_value = float(min_value)
        for dbfile in database_content.keys():
            df = database_content[dbfile]
            df['keep'] = df[column].apply(lambda x: True if x >= min_value else False)
            df = df[df['keep']==True]
            df = df.drop('keep', axis=1)
            database_content[dbfile] = df
    if max_value:
        max_value = float(max_value)
        for dbfile in database_content.keys():
            df = database_content[dbfile]
            df['keep'] = df[column].apply(lambda x: True if x <= max_value else False)
            df = df[df['keep']==True]
            df = df.drop('keep', axis=1)
            database_content[dbfile] = df

    return database_content


def plot_landscape(database_content : dict, 
                   plot_equivalents: bool = False, 
                   show_plot: bool = True, 
                   save_plot: bool = False) -> Tuple:
    """ Remove rows from dataframe that do not satisfy criteria for column

    Parameters
    ----------
    database_content : dict
        keys become labels for each data series in the plot
        values are dataframes containing structure data

    plot_equivalents : bool
        colour structures by number of equivalent structures

    show_plot : bool
        render a live plot

    save_plot : bool
        save the plot to a file

    """
    plt.style.use('bmh')

    glob_min_energy = 0
    glob_max_energy = -1000
    for dbfile in database_content.keys():
        df = database_content[dbfile]
        db_min_energy = df.loc[df['Energy (kJ/mol)'].idxmin()]['Energy (kJ/mol)']
        if db_min_energy < glob_min_energy:
            glob_min_energy = db_min_energy
        db_max_energy = df.loc[df['Energy (kJ/mol)'].idxmax()]['Energy (kJ/mol)']
        if db_max_energy > glob_max_energy:
            glob_max_energy = db_max_energy

    #fig, ax1 = plt.subplots(figsize=(19.20, 10.80))
    fig, ax1 = plt.subplots()
    for dbfile in database_content.keys():
        df = database_content[dbfile]
        if plot_equivalents:
            plt.scatter(df['Density (g/cm^3)'], df['Energy (kJ/mol)'], c=df['num_equivalents'], s=30, edgecolors=['black'], linewidth=0.7, alpha=0.8, vmin=0, vmax=10)
        else:
            plt.scatter(df['Density (g/cm^3)'], df['Energy (kJ/mol)'], s=30, edgecolors=['black'], linewidth=0.7, alpha=0.8, label=str(dbfile))

    plt.xlabel('Density (g/cm$^{3}$)')
    plt.ylabel('Energy (kJ/mol)')
    ax1.set_ylim([glob_min_energy - 1, glob_max_energy + 1])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Relative Energy (kJ/mol)', rotation=270, labelpad=10)
    ax2.set_ylim([-1, glob_max_energy - glob_min_energy + 1])
    ax2.grid(None)

    if plot_equivalents:
        cbar = plt.colorbar()
        cbar.set_label('Num equivalent crystals', rotation=270)
    else:
        ax1.legend(loc='lower left')

    if save_plot:
        plt.savefig('landscape.png')
    if show_plot:
        plt.show()

    return fig, ax1, ax2


LOG = logging.getLogger(__name__)


def main(sys_args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "databases",
        nargs="+",
        type=str,
        help="Databases to plot the contents of as a landscape.",
    )
    parser.add_argument(
        "-e",
        "--energy-range",
        nargs='*',
        help="Constrain plot to only structures with energy greater than value 1\
              and less than value 2. Value should be float or None.",
    )
    parser.add_argument(
        "-d",
        "--density-range",
        nargs='*',
        help="Constrain plot to only structures with density greater than value 1\
              and less than value 2. Value should be float or None.",
    )
    parser.add_argument(
        "--spg",
        nargs='*', 
        default=None,
        help="Plot by spacegroup instead of db file. List spacegroups to plot only those.",
    )
    parser.add_argument(
        "--ignore-clustering",
        action="store_true",
        help="Ignore any clustering and plot all final minimisations.",
    )
    parser.add_argument(
        "--equivalents",
        action="store_true",
        help="Colour points according to the number of equivalent structures.\
		The maximum value of the colour spectrum is always 10.\
		Two structures with 10 and 10,000 equivalents will have the same\
		colour.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the plot to current working directory",
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
    LOG.info("%d databases", len(args.databases))

    # const can only be used with nargs='?' but nargs='?' wont support multiple inputs.
    # this is a workaround
    if args.spg is not None and len(args.spg) == 0:
        args.spg = []

    database_content = scrape_structures_from_dbs(databases=args.databases, plot_equivalents=args.equivalents, spg_to_plot=args.spg, ignore_clustering=args.ignore_clustering)

    if args.energy_range: 
        database_content = filter_dataframe(database_content, "Energy (kJ/mol)", min_value=args.energy_range[0], max_value=args.energy_range[1])

    if args.density_range:
        database_content = filter_dataframe(database_content, "Density (g/cm^3)", min_value=args.density_range[0], max_value=args.density_range[1])

    fig, ax1, ax2 = plot_landscape(database_content, plot_equivalents=args.equivalents, show_plot=True, save_plot=args.save)


if __name__ == "__main__":
    main()
