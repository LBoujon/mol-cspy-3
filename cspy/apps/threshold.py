import logging
import sys
from mpi4py import MPI
from cspy.cspympi.threshold_tasks import ThresholdWorker
from cspy.distributed.threshold_manager import Threshold
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_CLG_arguments, add_minimisation_arguments, remove_argument

LOG = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyz_files",
        type=str,
        nargs="+",
        help="Xyz files containing molecules for generation",
    )
    parser.add_argument(
        "-r",
        "--res_files",
        type=str,
        nargs="+",
        help="Res files containing molecules for generation",
    )
    parser = add_DMACRYS_arguments(parser)
    parser = add_CLG_arguments(parser)
    # incompatible arguments
    remove_argument(parser, "nudge")
    remove_argument(parser, "adaptcell")
    remove_argument(parser, "asi")
    remove_argument(parser, "clg_chomp")
    remove_argument(parser, "clg_aut")
    remove_argument(parser, "number_structures")
    parser = add_minimisation_arguments(parser)
    parser.add_argument(
        "-t",
        "--nthreads",
        type=int,
        default=1,
        help="Number of dask worker threads per process to use for local scheduler",
    )
    parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Specify seed for random generator. Should be left blank for first runs."
            "The entropy/seed will be logged in the output and can be used to repeat "
            "simulations if needed."
        )
    )
    parser.add_argument(
        "--trials-per-res",
        type=int,
        default=1,
        help="number of trials/trajectories initiated for each res provided",
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
    app.configure_app(minimisation=True, thresh=True, skip_header=args.skip_header)

    worker_data = app.worker_data
    
    res_files = []
    for res in args.res_files:
        res_files.extend([res for _ in range(args.trials_per_res)])
    
    if rank == 0:
        comm.Split(0)
        csp_manager = Threshold(
            workers=range(1, app.num_workers+1),
            name=app.default_name,
            spacegroups=args.spacegroups,
            res_files=res_files,
            mc_para=app.mc_para,
            status_file=args.status_file,
            random_seed=args.random_seed,
        )
        csp_manager.run()

    elif not app.non_worker_core:
        LOG.debug("Core %s is a worker core. Running thresholding here.", rank)
        comm.Split(1)
        ThresholdWorker(worker_data).run()

    else:
        comm.Split(1)
        LOG.debug("Core %s is a non-worker core. Not running thresholding here.", rank)

if __name__ == "__main__":
    main()
