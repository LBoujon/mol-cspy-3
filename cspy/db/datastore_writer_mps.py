from cspy.db import CspDataStoreMPS
import logging

LOG = logging.getLogger(__name__)


class DatastoreWriterMPS:

    def __init__(
            self,
            structure_queue,
            filename="mps.db",
    ):
        self.filename = filename
        self.structure_queue = structure_queue
        self.structures = {}
        self.complete = False

    def run(self):
        ds = CspDataStoreMPS(self.filename)
        ds.add_mps_properties(self.structure_queue)
        ds.close()
