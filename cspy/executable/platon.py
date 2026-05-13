from .executable import AbstractExecutable, ReturnCodeError
from cspy.executable.locations import PLATON_EXEC
from cspy.util.path import Path
from os.path import exists, join
from os import remove
import logging
import os
from collections import defaultdict

LOG = logging.getLogger(__name__)


class Platon(AbstractExecutable):

    _executable_location = PLATON_EXEC
    _FILENAMES = {
        "check": "check.def",
        "cpi": "{name}.cpi",
        "lis": "{name}.lis",
        "spf": "{name}_pl.spf",
        "ps": "{name}.ps",
        "hklf": "{name}_gener.hkl",
        "output": "{name}.platon_stdout",
    }

    def __init__(self, filename, args, stdin=None, working_directory="."):
        self.name = Path(filename).stem
        self._working_directory = working_directory
        self.args = [] if args is None else args
        self.args.append(filename)
        self._output_file = Path(
            self.working_directory, f"{self.name}.platon_stdout"
        ).absolute()
        self.input_file = None
        self.stdin = stdin
        if self.stdin is not None:
            self.input_file = f"{self.name}.platon_stdin"
            p = Path(self.working_directory, self.input_file)
            p.write_text(self.stdin)
            self.input_file = p.absolute()
        for name in self._FILENAMES:
            setattr(self, name + "_contents", None)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        pass

    def result(self):
        return self.returncode

    def post_process(self):
        files_to_read = {
            k: v.format(name=self.name) for k, v in self._FILENAMES.items()
        }

        if self.stdin and exists(self.input_file):
            remove(self.input_file)
        for name, location in files_to_read.items():
            location = join(self.working_directory, location)
            if exists(location):
                LOG.debug("Reading %s", location)
                with open(location) as f:
                    setattr(self, name + "_contents", f.read())
                remove(location)

    def run(self, *args, **kwargs):
        try:
            if self.stdin is None:
                self._run_raw(*self.args)
            else:
                self._run_raw_stdin(*self.args)
        except ReturnCodeError as e:
            LOG.error("Platon failed: %s", e)
            self.post_process()
            raise e
