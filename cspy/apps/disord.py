import argparse
import logging
from pathlib import Path

from cspy.util.logging_config import DATEFMT, FORMATS


LOG = logging.getLogger("cspy.disord")


def main(sys_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Workflow tools for enumeration, optimisation, "
            "and ensemble free-energy calculations"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available cspy-disord commands.",
    )

    prep_parser = subparsers.add_parser(
        "prepare-cif",
        help="Create a disordered CIF",
    )

    prep_parser.add_argument(
        "host",
        type=Path,
        help="Host crystal file",
    )

    prep_parser.add_argument(
        "--component",
        nargs=2,
        action="append",
        metavar=("LABEL", "SOURCE"),
        default=None,
        help=(
            "Disorder component. For example: "
            "--component A1 host --component A2 guest.xyz"
        ),
    )

    prep_parser.add_argument(
        "--existing-disorder",
        action="store_true",
        help=(
            "Input CIF already contains disorder assembly/group labels."
        ),
    )

    prep_parser.add_argument(
        "--existing-component",
        default="A1",
        help=(
            "Default CSPy component label for states from an existing-disorder CIF."
        ),
    )

    prep_parser.add_argument(
        "--group-components",
        default=None,
        help=(
            "Optional mapping from original CIF states to CSPy chemical " 
            "components, for example 'A:1=A1,A:2=A2' or '1=A1,2=A2'. "
        ),
    )

    prep_parser.add_argument(
        "--joint-state",
        action="append",
        default=[],
        metavar="ASSEMBLY:GROUP,...",
        help=(
            "State for a molecule with multiple disorder assemblies. "
            "--joint-state 'A:1,B:2' --joint-state 'A:2,B:1'."
        ),
    )

    prep_parser.add_argument(
        "--multi-assembly-mode",
        choices=["auto", "explicit", "independent"],
        default="auto",
        help=(
            "How prepare-cif resolves several original disorder assemblies in one molecule. " 
            "'auto' pairs uniquely matching occupancies; "
            "'explicit' requires --joint-state; "
            "'independent' generates the Cartesian product. "
        ),
    )

    prep_parser.add_argument(
        "--joint-occupancy-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum absolute occupancy difference "
        ),
    )

    prep_parser.add_argument(
        "--host-label",
        default="A1",
        help="Label used for host sites. Default: A1.",
    )

    prep_parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=None,
        metavar=("NA", "NB", "NC"),
        help="Create a P1 supercell before adding disorder.",
    )

    prep_parser.add_argument(
        "--guest-molecule-index",
        type=int,
        default=0,
        help=(
            "Guest molecule index when there are multiple different "
            "molecules in the asymmetric unit. Default: 0."
        ),
    )

    prep_parser.add_argument(
        "--overlay-method",
        choices=[
            "auto",
            "kabsch",
            "principal",
            "centroid",
            "mapped-kabsch",
        ],
        default="auto",
        help="How to overlay guest molecules onto host sites.",
    )

    prep_parser.add_argument(
        "--all-isomorphic-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate all graph-isomorphic atom mappings as disorder groups."
        ),
    )

    prep_parser.add_argument(
        "--atom-match-mode",
        choices=["exact", "heavy", "any"],
        default="heavy",
        help=(
            "Atom matching mode for all-isomorphic overlays. "
            "'exact' requires atomic numbers to match; "
            "'heavy' ignores H atoms "
            "'any' ignores atom numbers." 
        ),
    )

    prep_parser.add_argument(
        "--duplicate-detection",
        choices=["none", "rmsd", "pxrd"],
        default="rmsd",
        help=(
            "Method used to remove duplicate overlays."
        ),
    )

    prep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("disordered_input.cif"),
        help="Output disordered CIF.",
    )

    prep_parser.add_argument(
        "--overlay-report",
        dest="overlay_report",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    prep_parser.add_argument(
        "--host-map",
        dest="host_map",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    prep_parser.add_argument(
        "--guest-map",
        dest="guest_map",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    prep_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Control level of logging output.",
    )

    enum_parser = subparsers.add_parser(
        "enumerate",
        help="Enumerate structures",
    )

    enum_parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Prepared input structure file.",
    )

    enum_parser.add_argument(
        "-s",
        "--supercell",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        metavar=("A", "B", "C"),
        help="Supercell size, for example -s 2 1 1. Default: 1 1 1.",
    )

    enum_parser.add_argument(
        "--total-sites",
        type=int,
        default=None,
        help=(
            "Total number of disordered sites in the generated "
            "supercell. By default this is inferred."
        ),
    )

    enum_parser.add_argument(
        "--ratio-counts-file",
        type=Path,
        default=None,
        help=(
            "CSV file containing composition counts or percentages. "
            "Use n_A1,n_A2,... or pct_A1,pct_A2,... columns. "
            "The n_samples column is optional unless --samples-from csv is used."
        ),
    )

    enum_parser.add_argument(
        "--samples-from",
        choices=["csv", "lnW", "10lnW", "log10W",   "10log10W",],
        default="10lnW",
        help=(
            "How to choose the number of samples for each composition. "
        ),
    )

    enum_parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor used with specified sampling."
        ),
    )

    enum_parser.add_argument(
        "-N",
        "--maxiters",
        type=int,
        default=512,
        help=(
            "Maximum number of configurations generated for each "
            "composition. Default: 512."
        ),
    )

    enum_parser.add_argument(
        "--no-pure-components",
        dest="include_pure_components",
        action="store_false",
        default=True,
        help=(
            "Do not automatically add pure-component compositions."
        ),
    )

    enum_parser.add_argument(
        "--orientation-degeneracy",
        nargs=2,
        action="append",
        default=[],
        metavar=("COMPONENT", "FACTOR"),
        help=(
            "Optional orientation degeneracy for theoretical W. "
            "COMPONENT can be A1, A2, A3, etc. Repeat as required."
        ),
    )

    enum_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for exact-composition sampling.",
    )

    enum_parser.add_argument(
        "--prefix",
        type=str,
        default="orgdisord",
        help="Prefix for output file names. Default: orgdisord.",
    )

    enum_parser.add_argument(
        "--output-db",
        type=Path,
        default=None,
        help="Output CSPy database. Default: <prefix>.db.",
    )

    enum_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output database, including associated "
            "SQLite WAL and shared-memory files."
        ),
    )

    enum_parser.add_argument(
        "--no-write",
        dest="no_write",
        action="store_true",
        help="Suppress writing generated structure files.",
    )

    enum_parser.add_argument(
        "--no-reformat-cif",
        action="store_true",
        help=(
            "Do not reformat generated CIF files before loading them "
            "into CSPy."
        ),
    )

    enum_parser.add_argument(
        "--keep-orgdisord-outputs",
        action="store_true",
        help=(
            "Keep the orgdisord results directory and CSV file after "
            "writing the CSPy database."
        ),
    )

    enum_parser.add_argument(
        "--keep-primitive",
        action="store_true",
        help="Keep orgdisord primitive-cell files.",
    )

    enum_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress orgdisord output.",
    )


    enum_parser.add_argument(
        "--exclude-ordered",
        dest="exclude_ordered",
        action="store_true",
        help="Exclude the ordered part of the input structure.",
    )

    enum_parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        help="Merge equivalent structures.",
    )

    enum_parser.add_argument(
        "-a",
        "--algo",
        choices=["symm", "rematch", "ewald"],
        default="symm",
        help=(
            "Algorithm used for merging equivalent structures. "
            "Default: symm."
        ),
    )

    enum_parser.add_argument(
        "-d",
        "--use-disordered-only",
        dest="use_disordered_only",
        action="store_true",
        help=(
            "Use only the disordered part when merging equivalent "
            "structures."
        ),
    )

    enum_parser.add_argument(
        "-p",
        "--symprec",
        type=float,
        default=0.0001,
        help=(
            "Symmetry precision used for merging equivalent "
            "structures. Default: 0.0001."
        ),
    )

    enum_parser.add_argument(
        "--ox",
        nargs=2,
        action="append",
        metavar=("SPECIES", "OXIDATION_STATE"),
        help=(
            "Specify oxidation states, for example "
            "--ox H 1 --ox O -2."
        ),
    )

    enum_parser.add_argument(
        "--ignore-species",
        "--ignore_species",
        dest="ignore_species",
        action="store_true",
        help="Ignore atomic species when merging structures.",
    )

    enum_parser.add_argument(
        "--not-molecular-crystal",
        action="store_true",
        help=argparse.SUPPRESS,
    )


    enum_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    opt_parser = subparsers.add_parser(
        "optimise",
        help=(
            "Optimise structures from a cspy-disord database."
        ),
    )

    opt_parser.add_argument(
        "database",
        type=Path,
        help="database",
    )

    opt_parser.add_argument(
        "--output-db",
        type=Path,
        default=None,
        help="Output database.",
    )

    opt_parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="For optimisation files.",
    )

    opt_parser.add_argument(
        "--fresh-workdir",
        dest="fresh_workdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Delete and recreate the optimisation work directory before starting. Default: True."
        ),
    )

    opt_parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=False,
        help=(
            "Skip structures already present in the output .opt.db."
        ),
    )

    opt_parser.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep optimisation directories.",
    )

    opt_parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Only optimise the first N structures.",
    )

    opt_parser.add_argument(
        "--errors-file",
        type=Path,
        default=Path("optimise_errors.csv"),
        help="CSV file for failed optimisations.",
    )

    opt_parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("optimise_status.txt"),
        help="Optimisation progress status file.",
    )

    opt_parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Number of cspy-opt subprocesses to run in parallel. Default SLURM_CPUS_PER_TASK."
        ),
    )

    opt_parser.add_argument(
        "-j",
        "--gaussian-cpus",
        type=int,
        default=None,
        help="Number of Gaussian CPUs used by each cspy-opt process.",
    )

    opt_parser.add_argument(
        "--cspy-toml",
        dest="cspy_config",
        type=Path,
        default=None,
        help=(
            "Path to cspy.toml."
        ),
    )

    opt_parser.add_argument(
        "--calculation-type",
        default="dmacrys",
        choices=["dmacrys", "mace"],
        help=(
            "Calculation type for cspy-opt. MACE has not yet been fully tested with cspy-disord."
        ),
    )

    opt_parser.add_argument(
        "-p",
        "--potential",
        default="fit",
        help="Potential for cspy-opt. Default: fit.",
    )

    opt_parser.add_argument(
        "--basis-set",
        default=None,
        help="Basis set for cspy-opt.",
    )

    opt_parser.add_argument(
        "--method",
        default=None,
        help="Electronic-structure method for cspy-opt.",
    )

    opt_parser.add_argument(
        "--single-point",
        action="store_true",
        default=False,
        help="Pass --single-point to cspy-opt.",
    )

    # MACE support still needs testing.
    #
    # opt_parser.add_argument(
    #     "--mace-model",
    #     dest="mace_model",
    #     default="mace_mp",
    #     help="MACE model name or path.",
    # )

    opt_parser.add_argument(
        "--multipole",
        type=Path,
        default=None,
        help="Multipole file for cspy-opt.",
    )

    opt_parser.add_argument(
        "--axis",
        type=Path,
        default=None,
        help="Axis file for cspy-opt.",
    )

    opt_parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "DMACRYS real-space potential cutoff passed to cspy-opt. "
            "Default: use the cspy-opt setting."
        ),
    )

    opt_parser.add_argument(
        "--dma-switch",
        dest="dma_switch",
        type=int,
        default=None,
        help=(
            "GDMA switch value passed to cspy-opt during multipole "
            "generation. This is not the DMACRYS RDMA cutoff."
        ),
    )

    opt_parser.add_argument(
        "--reorder-atoms-if-high-rmsd",
        action="store_true",
        default=False,
        help=(
            "Allow cspy-opt to reorder atoms when multipole mapping "
            "RMSD is high."
        ),
    )

    opt_parser.add_argument(
        "--reorder-method",
        default="molecular_axis",
        choices=["molecular_axis", "rdkit"],
        help=(
            "Atom-reordering method used by cspy-opt. "
            "Default: molecular_axis."
        ),
    )

    opt_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )

    calc_parser = subparsers.add_parser(
        "calc",
        help=(
            "Calculate ensemble free energies from an optimised database."
        ),
    )

    calc_parser.add_argument(
        "database",
        type=Path,
        help="database",
    )

    calc_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV file.",
    )

    calc_parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in K. Default: 298.15.",
    )

    calc_parser.add_argument(
        "--energy-input",
        choices=["per_site", "total"],
        default="per_site",
        help=(
            "Interpret database energies as per-site/per-molecule "
            "energies or total-cell energies."
        ),
    )

    calc_parser.add_argument(
        "--reference-energy",
        nargs=2,
        action="append",
        metavar=("COMPONENT", "ENERGY"),
        help=(
            "Reference pure-component energies, "
            "--reference-energy A1 -98.5 --reference-energy A2 -102.0"
        ),
    )

    calc_parser.add_argument(
        "--orientation-degeneracy",
        nargs=2,
        action="append",
        default=[],
        metavar=("COMPONENT", "FACTOR"),
        help=(
            "Orientation degeneracy factor for a component Overrides database metadata. "
            "--orientation-degeneracy A2 2. "
        ),
    )

    calc_parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples.",
    )

    calc_parser.add_argument(
        "--ci",
        type=float,
        default=98.0,
        help=(
            "Bootstrap confidence interval."
        ),
    )

    calc_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for bootstrap resampling.",
    )

    calc_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )


    args = parser.parse_args(sys_args)

    logging.basicConfig(
        format=FORMATS[args.log_level],
        datefmt=DATEFMT,
        level=args.log_level,
    )

    if args.command == "prepare-cif":
        LOG.info("Preparing CIF")

        from progs.orgdisord.prepare_cif import run_prepare_cif

        run_prepare_cif(args)
        LOG.info("Finished preparing CIF")

    elif args.command == "enumerate":
        LOG.info("Starting orgdisord enumeration")

        from progs.orgdisord.enumerate import (
            run_orgdisord_enumerate,
        )

        run_orgdisord_enumerate(args)
        LOG.info("Finished orgdisord enumeration")

    elif args.command == "optimise":
        LOG.info("Starting cspy-opt minimisation")

        from progs.orgdisord.optimise import (
            run_cspy_disord_optimise,
        )

        run_cspy_disord_optimise(args)
        LOG.info("Finished cspy-opt minimisation")

    elif args.command == "calc":
        LOG.info(
            "Starting ensemble free-energy calculation"
        )

        from progs.orgdisord.calc import run_cspy_disord_calc

        run_cspy_disord_calc(args)

        LOG.info(
            "Finished ensemble free-energy calculation"
        )


if __name__ == "__main__":
    main()