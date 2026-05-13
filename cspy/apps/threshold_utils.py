from cspy.threshold import threshold_setup, threshold_db, disconnectivity_graph
from argparse import ArgumentParser
from typing import Union, List
import sys
import logging


NAME = "cspy-threshold-utils"
LOG = logging.getLogger(NAME)


class ThreshUtilsCLI:
    def __init__(self, args: Union[List[str], None]) -> None:
        self.args = args
        parser = ArgumentParser(
            prog=NAME,
            description="Utilities to manage MC Threshold jobs",
            usage="cspy-threshold-utils [-h] {command} [args]",

        )
        parser.add_argument(
            "command", 
            help="Available commands",
            type=str,
            choices=[func for func in dir(ThreshUtilsCLI) if callable(getattr(ThreshUtilsCLI, func)) and not func.startswith("__")]
        )
        args = parser.parse_args(args[1:2])

        getattr(self, args.command)()

    def setup(self) -> None:
        threshold_setup.main(self.args[2:], f"{NAME} {self.args[1]}")

    def disconn(self) -> None:
        disconnectivity_graph.main(self.args[2:], f"{NAME} {self.args[1]}")

    def db(self) -> None:
        threshold_db.main(self.args[2:], f"{NAME} {self.args[1]}")


def main() -> None:
    ThreshUtilsCLI(sys.argv)


if __name__ == "__main__":
    main()
    