import argparse
import sys
import logging

# SAUCE : Sensible Asymmetric Units for Crystal Exploration

LOG = logging.getLogger("cspy-sauce")
USAGE = """cspy-sauce <command> [<args>]

Available commands are:
    extract_mps         Extract molecular pairs from crystal database into pair database
    extract_aut         Extract asymmetric units from crystal database into asymmetric unit database
    extract_uc2au       Extract unit cells from crystal database into asymmetric unit database
    cluster             Remove duplicates from pair database and consolidate properties
    crossref            Check to see whether pairs in database B appear in database A
    clg_mps             Generate crystals via MPS algorithm using pairs from pair database
    clg_aut             Generate crystals via AUT algorithm using AUs from AU database
    energy              Calculate energy of clusters (pairs of AUs) in database and update database with energies
"""

class SAUCECLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Prepare for SAUCE", usage=USAGE
        )
        parser.add_argument("command", help="sub command to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            LOG.error("Unrecognized command '%s'", args.command)
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()

    
    def extract_mps(self):
        from cspy.sauce.extract_mps import main

        main(sys.argv[2:])

    def extract_aut(self):
        from cspy.sauce.extract_aut import main

        main(sys.argv[2:])

    def extract_uc2au(self):
        from cspy.sauce.extract_uc2au import main

        main(sys.argv[2:])

    def cluster(self):
        from cspy.sauce.clustering import main

        main(sys.argv[2:])

    def crossref(self):
        from cspy.sauce.crossref import main

        main(sys.argv[2:])

    def clg_mps(self):
        from cspy.sauce.clg_mps import main

        main(sys.argv[2:])

    def clg_aut(self):
        from cspy.sauce.clg_aut import main

        main(sys.argv[2:])

    def energy(self):
        from cspy.sauce.energy import main

        main(sys.argv[2:])


def main():
    SAUCECLI()


if __name__ == "__main__":
    main()
