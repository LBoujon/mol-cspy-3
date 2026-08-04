import logging
from cspy.flex.flex_molecule import create_flex_database, add_to_flex_database, FlexMolecule
from cspy.db import CspDataStoreFlex
import argparse
from cspy.apps.setup_app import CspyApp, add_common_arguments, add_gaussian_arguments
import sys

LOG = logging.getLogger(__name__)

def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="An application for generating conformational databases for flexible-molecule CSP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="*",
        help="Names of the files containing molecular coordinates. These can be .xyz, .fchk,"\
        "or a mixture of both file types. Alternatively, you can specify conformational databases" \
        "along with the --jointdb to combine these into  one (see jointdb flag)",
    )
    parser.add_argument(
        "--scan_sobol",
        type=int,
        default=0,
        help="Uses a sobol vector for the DOF scan. Specify the number of scan points required",
    )
    parser.add_argument(
        "--scan_dofs",
        nargs='+',
        default="",
        help="A list containing the coordinates to scan. Please see documentation for information. " \
        "Atoms should be indexed starting from 1.",
    )
    parser.add_argument(
        "--coupled_scan_dofs",
        "--coupled-scan-dofs",
        action="store_true",
        default=False,
        help="Advance all scan DOFs with one shared coordinate instead of "
        "sampling their Cartesian product. Step signs control the relative "
        "numerical direction of each coordinate.",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        default=[],
        help="A list of atom numbers you wish to constrain during a gaussian optimisation." \
        "Atoms should be indexed starting from 1."
        "e.g --constraints '1 2 3' '5 6 7 8' '9' would lead to the fixing of the angle between" \
        "atoms 1,2, and 3, the dihedral angle between atoms 5, 6, 7 and 8, and the atomic" \
        "position of atom '9'."
    )
    parser.add_argument(
        "--jointdb",
        action="store_true",
        default=False,
        help="Join a list of databases into a single database",
    )
    parser.add_argument(
        "--charges",
        default="0",
        help="Charge of each molecule given in args.filenames as a single Python string in the same order as the filenames."\
        "(e.g. '0 -1 1') Note: If all molecules are neutral, then there is no need to specify "\
        "this flag.",
    )
    parser.add_argument(
        "--potential",
        default="F",
        choices=["F", "W"],
        help="Intermolecular potential type",
    )
    parser.add_argument(
        "--multiplicities",
        default="1",
        help="Multiplicty of each molecule given in args.filenames as a single Python string "\
        "(e.g. '1 3 1') Note: If all molecules have a multiplicty of 1, then there is no need to "\
        "specify this flag.",
    )
    parser.add_argument(
        "--xtb",
        action="store_true",
        default=False,
        help="Do a scan with xtb",
    )
    parser.add_argument(
        "--foreshorten_hydrogens",
        default=None,
        help="Length to foreshorten hydrogens for dma files after an xtb scan. Usually None for FIT, 0.1 for W99",
    )
    parser = add_gaussian_arguments(parser)
    parser = add_common_arguments(parser)

    args = parser.parse_args(arguments)

    # Importing mpi4py initializes the MPI runtime on some installations.  Keep
    # it behind argument parsing so serial operations such as ``--help`` work on
    # login nodes and machines without an active MPI fabric.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    app_name = sys.argv[0].split('/')[-1]
    app = CspyApp(app_name=app_name, args=args, rank=rank, size=size)
    app.configure_app(skip_header=args.skip_header)

    redundant = 'opt=ModRedundant' if args.constraints else ""

    if args.jointdb:
        try:
            assert len(args.filenames) > 1
            for file in args.filenames:
                assert file.rsplit(".", 1)[-1] == "db"
        except AssertionError:
            raise (
                "Please make sure you have specified multiple database files (.db) to join when using the --joint-db flag"
            )
        if rank == 0:
            molecules = {}
            columns = [
                "id",
                "energy",
                "xyz_coordinates",
                "mults",
                "axes",
                "charges",
                "res",
            ]
            for file in args.filenames:
                db = CspDataStoreFlex(file)
                for row in db.select("distorted_molecules", columns):
                    molecules[row[0]] = [row[2], row[1], row[3], row[4], row[5], row[6]]
                db.disconnect()
            create_flex_database("jointDB")
            add_to_flex_database(molecules, "jointDB")

    else:
        if rank == 0:
            LOG.info(f"Using the following files: {', '.join(args.filenames)}")

        if len(args.filenames) > 1:
            if args.charges == "0":
                args.charges = ' '.join(["0" for mol in args.filenames])
            if args.multiplicities == "1":
                args.multiplicities = ' '.join(["1" for mol in args.filenames])
                
        for idx, file in enumerate(args.filenames):
            mymol = FlexMolecule()
            if file.rsplit(".", 1)[-1] == "xyz":
                mymol.init_from_xyz(file)
            elif file.rsplit(".", 1)[-1] == "fchk":
                mymol.init_from_fchk(file)
            else:
                raise TypeError(
                    "Invalid input file type: Please supply .xyz or .fchk files."
                )
            g09_settings = {
                "nprocshared": args.gaussian_cpus,
                "mem": args.gaussian_memory,
                "pcm": args.polarizable_continuum,
            }
            flex_label = file.rsplit(".", 1)[0]
            mymol.name = file
            charge = args.charges.split(" ")[idx]
            multiplicity = args.multiplicities.split(" ")[idx]
            internals = mymol.eval_user_dofs_str(args.scan_dofs)
            if args.potential == "W" and args.foreshorten_hydrogens is None:
                if rank == 0:
                    LOG.info(
                        f"Changing the foreshorten_hydrogens argument from None to 0.1. \
                        This is because the user has specified a Williams potential"
                    )
                args.foreshorten_hydrogens = 0.1
            if args.xtb:
                mymol.xtb_scan(
                    internals=internals,
                    comm=comm,
                    filename_prefix=flex_label,
                    foreshorten_hydrogens=args.foreshorten_hydrogens,
                )
            else:
                mymol.gaussian_scan(
                    internals=internals,
                    comm=comm,
                    filename_prefix=flex_label,
                    gaussian_args=g09_settings,
                    method=args.functional,
                    charges=charge,
                    potential=args.potential,
                    multiplicities=multiplicity,
                    basis_set=args.basis_set,
                    sobol_points=args.scan_sobol,
                    coupled_scan_dofs=args.coupled_scan_dofs,
                    constraints=args.constraints,
                    redundant=redundant,
                )


if __name__ == "__main__":
    main()
