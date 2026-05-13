import argparse
import sys
import logging

LOG = logging.getLogger("cspy-db")
USAGE = """cspy-db <command> [<args>]
For each command, run 'cspy-db <command> --help' for more information.

Available commands are:
    prune            Remove duplicate structures from databases
    cluster          synonymous with 'prune'
    dump             Extract data from databases into other formats
    plot             Plot a landscape from a database
    info             Return information about the contents of a database
    remove_outliers  Remove gapped structures and undetected Buckingham catastrophies from a database
    convert          Convert an old 5-column database to a new 6-column format (adds molecule_id column)
"""


class DatabaseCLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Process cspy databases", usage=USAGE
        )
        parser.add_argument("command", help="sub command to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            LOG.error("Unrecognized command '%s'", args.command)
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()

    def prune(self):
        self.cluster()

    def info(self):
        from cspy.db.info import main

        main(sys.argv[2:])

    def plot(self):
        from cspy.db.plot import main

        main(sys.argv[2:])

    def cluster(self):
        from cspy.db.clustering import main
        main(sys.argv[2:])

    def convert(self):
        from cspy.db.convert import main

        main(sys.argv[2:])

    def extract(self):
        from cspy.db.extract import main

        main(sys.argv[2:])

    def dump(self):
        from cspy.db.dump import main

        main(sys.argv[2:])

    def remove_outliers(self):
        from cspy.db.outlier_removal import main

        main(sys.argv[2:])


def main():
    DatabaseCLI()


if __name__ == "__main__":
    main()
