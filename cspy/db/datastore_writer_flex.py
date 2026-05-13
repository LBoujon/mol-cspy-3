from cspy.db import CspDataStoreFlex
from time import sleep
import logging

LOG = logging.getLogger(__name__)


class DatastoreWriterFlex:

    def __init__(
            self,
            structure_queue,
            filename="flex.db",
    ):
        self.filename = filename
        self.structure_queue = structure_queue
        self.structures = {}
        self.complete = False

    def run(self):
        ds = CspDataStoreFlex(self.filename)
        ds.add_flex_results(self.structure_queue)
        ds.close()
