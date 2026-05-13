import logging
import sys
import numpy as np
from mpi4py import MPI
from numpy.random import SeedSequence
from cspy.distributed.qrbh_manager import QRBHCSP
from cspy.cspympi.qrbh_tasks import QRBHWorker
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_CLG_arguments, add_minimisation_arguments

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
    parser = add_DMACRYS_arguments(parser)
    parser = add_CLG_arguments(parser)
    parser = add_minimisation_arguments(parser)
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        default=None,
        help="Specify seed for random generator",
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
    app.configure_app(clg=True, minimisation=True, bh=True, skip_header=args.skip_header)

    worker_data = app.worker_data

    # set up random seeds across processes
    if rank == 0:
        sq = SeedSequence(args.random_seed)
        seeds = sq.generate_state(comm.Get_size())
        LOG.info("random seed/entropy to repeat run: %d", sq.entropy)
    else:
        seeds = None
        
    random_seed = comm.scatter(seeds, root=0)
    np.random.seed(random_seed)
    
    if rank == 0:
        comm.Split(0)
        csp_manager = QRBHCSP(
            workers=range(1, app.num_workers+1),
            name=app.default_name,
            spacegroups=app.spacegroups,
            mc_para=app.mc_para,
            number_structures=app.number_structures,
            status_file=app.status_file
        )
        csp_manager.run()

    elif not app.non_worker_core:
        LOG.debug("Core %s is a worker core. Running QRBH here.", rank)
        comm.Split(1)
        QRBHWorker(worker_data).run()

    else:
        comm.Split(1)
        LOG.debug("Core %s is a non-worker core. Not running QRBH here.", rank)


if __name__ == "__main__":
    main()
