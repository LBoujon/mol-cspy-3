from .executable import AbstractExecutable
from cspy.executable.locations import GDMA_EXEC
from cspy.util.path import Path
import logging
import os
from cspy.chem.multipole import DistributedMultipoles
from cspy.templates import get_template

LOG = logging.getLogger("cspy.gdma")


class GdmaException(Exception):
    pass


class GdmaInput(object):
    def __init__(self):
        pass


class Gdma(AbstractExecutable):

    _gdma_template = get_template("gdma")
    _executable_location = GDMA_EXEC
    _INPUT_FILE_FMT = "{}.gdma_in"
    _OUTPUT_FILE_FMT = "{}.gdma_out"
    _PUNCH_FILE_FMT = "{}.mult"

    def __init__(
        self,
        fchk,
        name="molecule",
        limit=4,
        dma_switch=4.0,
        units="angstrom",
        output_file=None,
        density="SCF",
        working_directory=".",
        foreshorten_hydrogens=None,
        hydrogen_radius=None,
        timeout=72000.0,
    ):
        """
        Parameters
        ----------
        input_file : str
            string of GDMA input format
        """
        assert os.path.exists(fchk)
        self.name = name
        self.fchk = fchk
        self.timeout = timeout
        self.dma_switch = dma_switch
        self.limit = limit
        self.multipoles = None
        self.density = density
        self.units = units
        self.foreshorten_hydrogens = foreshorten_hydrogens
        self.modified_sites = []
        if self.foreshorten_hydrogens is not None:
            from cspy.chem.molecule import Molecule

            mol = Molecule.load(self.fchk)
            pos = mol.foreshortened_hydrogen_positions(
                reduction=self.foreshorten_hydrogens
            )
            self.modified_sites = [
                (el, x, y, z) for el, (x, y, z) in zip(mol.elements, pos)
            ]
        self.hydrogen_radius = hydrogen_radius
        self.working_directory = working_directory

    @property
    def input_file(self):
        return Path(self.working_directory, self._INPUT_FILE_FMT.format(self.name))

    @property
    def output_file(self):
        return Path(self.working_directory, self._OUTPUT_FILE_FMT.format(self.name))

    @property
    def punch_file(self):
        return Path(self.working_directory, self._PUNCH_FILE_FMT.format(self.name))

    def input_string(self):
        return self._gdma_template.render(
            formchk=self.fchk,
            verbose=True,
            units=self.units,
            dma_switch=self.dma_switch,
            limit=self.limit,
            punch=self.punch_file,
            density=self.density,
            modified_sites=self.modified_sites,
            hydrogen_radius=self.hydrogen_radius,
        )

    def write_inputs(self):
        input_str = self.input_string()
        LOG.debug("Writing input file to %s", self.input_file.absolute())
        self.input_file.write_text(input_str)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def result(self):
        if not self.multipoles:
            self.multipoles = DistributedMultipoles.from_dma_string(self.punch_contents)
        return self.multipoles

    def post_process(self):
        LOG.debug("Reading output file from %s", os.path.abspath(self.output_file))
        self.output_contents = self.output_file.read_text()
        LOG.debug("Reading punch file from %s", os.path.abspath(self.punch_file))
        if not self.punch_file.exists():
            LOG.error(
                "Could not read punch file: %s, output_contents:\n%s",
                self.punch_file,
                self.output_contents,
            )
        self.punch_contents = self.punch_file.read_text()

    def run(self, *args, **kwargs):
        self._run_raw_stdin()
