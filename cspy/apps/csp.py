import logging
from mpi4py import MPI
import sys
from cspy.cspympi import CSPyWorker
from cspy.distributed import QuasiRandomCSP, AUTCSP
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_CLG_arguments, add_minimisation_arguments
from cspy.util.path import Path

LOG = logging.getLogger(__name__)

def main():
    import argparse
    from cspy.configuration import CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files",
        type=str,
        nargs="+",
        help="Xyz files containing molecules for generation",
    )
    parser = add_DMACRYS_arguments(parser)
    parser = add_CLG_arguments(parser)
    parser = add_minimisation_arguments(parser)
    parser.add_argument(
        "-ct",
        "--chomp_temperature",
        type=int,
        default=10000,
        help="Temperature for Boltzmann weighting for ChoMP",
    )
    parser = add_common_arguments(parser)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        LOG.error("This program requires at least 2 MPI processes. " \
                "It's because one process is the manager. " \
                "Exiting...")
        sys.exit(1)

    app_name = sys.argv[0].split('/')[-1]
    app = CspyApp(app_name=app_name, args=args, rank=rank, size=size)
    app.configure_app(clg=True, minimisation=True, skip_header=args.skip_header)

    worker_data = app.worker_data
    
    if rank == 0:
        comm.Split(0)
        if args.clg_aut:
            csp_manager = AUTCSP(
            workers=range(1, app.num_workers+1),
            name=app.default_name,
            spacegroups=app.spacegroups,
            asymmetric_unit=[Path(x).read_text() for x in args.xyz_files],
            number_structures=app.number_structures,
            status_file=app.status_file,
            utilisation_file="utilisation.txt"
            )
        else:
            csp_manager = QuasiRandomCSP(
                workers=range(1, app.num_workers+1),
                name=app.default_name,
                spacegroups=app.spacegroups,
                number_structures=app.number_structures,
                status_file=app.status_file
            )
        csp_manager.run()

    elif not app.non_worker_core:
        LOG.debug("Core %s is a worker core. Running CSP here.", rank)
        comm.Split(1)
        CSPyWorker(worker_data).run()

    else:
        comm.Split(1)
        LOG.debug("Core %s is a non-worker core. Not running CSP here.", rank)


if __name__ == "__main__":
    main()
