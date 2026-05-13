import logging
from mpi4py import MPI
import sys
from cspy.cspympi import CSPyWorker
from cspy.distributed import QuasiRandomCSP
from collections import Counter
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_CLG_arguments, add_minimisation_arguments

LOG = logging.getLogger(__name__)


def main():
    import argparse
    from cspy.configuration import CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "databases",
        type=str,
        nargs="+",
        help="Databases containing molecular conformations with multipoles generated from cspy-moldis",
    )
    parser = add_DMACRYS_arguments(parser)
    parser = add_CLG_arguments(parser)
    parser = add_minimisation_arguments(parser)
    parser.add_argument(
        "--conf_energy_window",
        type=float,
        default=CONFIG.get("csp.conf_energy_window"),
        help="""When selecting conformers to generate crystals, the conf_energy_window defines the 
        energy window above the minimum energy conformation from within which a conformation may be 
        randomly selected.""",
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
    app = CspyApp(app_name=app_name, args=args, rank=rank)
    app.configure_app(clg=True, minimisation=True, flex=True, skip_header=args.skip_header)

    worker_data = app.worker_data

    db_names = [db.rsplit(".", 1)[0] for db in args.databases]
    # Below returns a dictionary counting the number of times each db was specified
    # e.g. 3-component system A.2B would be: counts ={A:1, B:2}
    # remove paths from names of databases
    db_names_sanitised = [db_name.split('/')[-1] for db_name in db_names]
    counts = dict(Counter(db_names_sanitised))
    default_name = ".".join(
        "{name}{count}".format(
            name=name, count="x{}".format(count) if count > 1 else ""
        )
        for name, count in counts.items()
    )

    worker_data["conf_energy_window"] = args.conf_energy_window
    worker_data["file"] = default_name
    worker_data["dbs"] = db_names

    if rank == 0:
        csp_manager = QuasiRandomCSP(
            workers=range(1, MPI.COMM_WORLD.Get_size()),
            name=default_name,
            spacegroups=app.spacegroups,
            number_structures=app.number_structures,
            status_file=app.status_file,
        )
        csp_manager.run()
    else:
        CSPyWorker(worker_data).run()


if __name__ == "__main__":
    main()
