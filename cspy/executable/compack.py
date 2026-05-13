from .executable import AbstractExecutable
from cspy.executable.locations import COMPACK_EXEC
from cspy.util.path import Path
import logging
import os
from collections import defaultdict

LOG = logging.getLogger("compack")


class CompackException(Exception):
    pass


class Compack(AbstractExecutable):

    _executable_location = COMPACK_EXEC

    def __init__(
        self,
        reference_structures,
        comparison_structures,
        output_file="compack.stdout",
        timeout=10.0,
        nmolecules=30,
        working_directory=".",
    ):
        """
        Parameters
        ----------
        reference_structures : str OR list
            res string or strings for reference
        comparison_structures : str OR list
            res string or strings for comparison to reference
        output_file : str, optional
            output_file to store stdout in by default will be a tempfile
        """
        if isinstance(reference_structures, str):
            reference_structures = [reference_structures]
        if isinstance(comparison_structures, str):
            comparison_structures = [comparison_structures]

        assert isinstance(reference_structures, list)
        assert isinstance(comparison_structures, list)

        self._output_file = output_file
        self.reference = reference_structures
        self.comparison = comparison_structures
        self.working_directory = working_directory
        self.n = nmolecules
        self.timeout = timeout
        self.reference_names = {}
        self.comparison_names = {}
        self.d = 20
        self.h = False
        self.b = False

    @property
    def output_file(self):
        return Path(self.working_directory, self._output_file)

    def write_inputs(self):
        for i, r in enumerate(self.reference):
            name = "ref{}.res".format(i)
            Path(self.working_directory, name).write_text(r)
            self.reference_names[name] = i

        for i, r in enumerate(self.comparison):
            name = "comp{}.res".format(i)
            Path(self.working_directory, name).write_text(r)
            self.comparison_names[name] = i

        Path(self.working_directory, "reference.txt").write_text(
            "\n".join(self.reference_names)
        )

        Path(self.working_directory, "comparison.txt").write_text(
            "\n".join(self.comparison_names)
        )

    @property
    def result(self):
        """Return the result of this calculation"""
        if not hasattr(self, "_rmsd"):
            return None
        return self._rmsd

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def post_process(self):
        # If compack fails, just return a large rms
        contents = self.output_file.read_text()
        BIG_RMS = 1e10
        rms = []
        try:
            for line in contents.splitlines():
                if "number molecules" in line:
                    tokens = line.split()
                    key1, key2 = (
                        self.reference_names[tokens[0]],
                        self.comparison_names[tokens[1]],
                    )
                    if "rms deviation" in line:
                        deviation = float(line.split("rms deviation")[1])
                        rms.append((key1, key2, deviation))
                    else:
                        rms.append((key1, key2, BIG_RMS))
            # Clean up
        except Exception as e:
            LOG.exception("Error reading compack output: %s", e)

        if not rms:
            raise CompackException(
                "Error in compack: output contents\n{}".format(contents)
            )
        self._rmsd = rms

    def run(self, *args, **kwargs):
        n = "-n{}".format(self.n)
        d = "-d{}".format(self.d)
        h = "-h" if self.h else ""
        b = "-b" if self.b else ""
        self._run_raw(n, d, h, "reference.txt", "comparison.txt")
