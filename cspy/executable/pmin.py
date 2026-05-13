from .executable import AbstractExecutable, ReturnCodeError
import logging
from os import remove
from os.path import basename, exists, join
from cspy.util.path import working_directory
from cspy.executable.locations import PMIN_EXEC
from cspy.configuration import CONFIG
import shutil
import copy


LOG = logging.getLogger("pmin")


class Pmin(AbstractExecutable):
    _input_file = "PMIN.inp"
    _output_file = "stdout"
    _executable_location = PMIN_EXEC
    _timeout = CONFIG.get("pmin.timeout")
    _FILENAMES = {
        "fort8": "fort.8",
        "fort18": "fort.18",
        "fort20": "fort.20",
        "fort21": "fort.21",
        "fort28": "fort.28",
        "fort31": "fort.31",
        "fort61": "fort.61",
        "fort62": "fort.62",
        "fort99": "fort.99",
        "rss_save": "pmin_RSS.save",
        "lsq_save": "pmin_LSQ.save",
        "save": "PMIN.save",
        "summary": "PMIN.summary_last",
    }

    def __init__(self, input_contents, name="pmin", working_directory=".", **kwargs):
        for name in self._FILENAMES:
            setattr(self, name + "_contents", None)
        self._timeout = kwargs.get("timeout", self._timeout)
        self.input_contents = input_contents
        self.name = name
        self.output_contents = None
        self.kwargs = kwargs.copy()
        self.working_directory = working_directory

    @property
    def input_file(self):
        return join(self.working_directory, self._input_file)

    @property
    def output_file(self):
        return join(self.working_directory, self._output_file)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        with open(self.input_file, "w") as f:
            f.write(self.input_contents)

    def result(self):
        return self.output_contents

    def post_process(self):
        files_to_read = {name: self._FILENAMES[name] for name in ("summary", "save")}

        for name, location in files_to_read.items():
            location = join(self.working_directory, location)
            if exists(location):
                with open(location) as f:
                    setattr(self, name + "_contents", f.read())
                remove(location)

        if exists(self.output_file):
            with open(self.output_file) as f:
                self.output_contents = f.read()
            remove(self.output_file)

    def run(self, *args, **kwargs):
        try:
            self._run_raw_stdin()
        except ReturnCodeError as e:
            LOG.error("pmin failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            raise e
