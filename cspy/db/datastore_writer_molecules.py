from cspy.db import CspDataStoreMolecules
from time import sleep
import logging

LOG = logging.getLogger(__name__)


class DatastoreWriterMolecules:

    def __init__(
            self,
            structure_queue,
            filename="molecules.db",
    ):
        self.filename = filename
        self.structure_queue = structure_queue
        self.structures = {}
        self.complete = False

    def run(self):
        ds = CspDataStoreMolecules(self.filename)
        ds.add_molecules(self.structure_queue)
        ds.close()
