from cspy.db import CspDataStoreAUs
from time import sleep
import logging

LOG = logging.getLogger(__name__)


class DatastoreWriterAUs:

    def __init__(
            self,
            structure_queue,
            filename_format="csp_{db_id}-AU.db",
            interval=0.1
    ):
        self.filename_fmt = filename_format
        self.structure_queue = structure_queue
        self.interval = interval
        self.structures = {}
        self.complete = False

    def pop_structures(self):
        self.structures = {}
        for db_id, l in self.structure_queue.items():
            self.structures[db_id] = []
            while len(l) > 0:
                self.structures[db_id].append(l.popleft())

    def run(self):

        while True:
            self.pop_structures()
            if self.complete and all(len(x) == 0 for x in self.structures.values()):
                LOG.info("Ending DatastoreWriterAUs thread")
                break

            for db_id in self.structures:
                structures = self.structures[db_id]
                if len(structures) < 1:
                    LOG.debug("Nothing to process in db_id %s", db_id)
                    continue
                filename = self.filename_fmt.format(db_id=db_id)
                try:
                    ds = CspDataStoreAUs(filename)
                    LOG.debug("%s Saving %d structures", db_id, len(structures))
                    ds.add_au_properties(structures)
                    LOG.debug(
                        "%s successfully saved %d structures", db_id, len(structures)
                    )
                    self.structures[db_id] = []
                except Exception as e:
                    LOG.error(
                        "Exception when adding %d csp results to %s: %s",
                        len(structures),
                        filename,
                        e,
                    )
                finally:
                    ds.close()
            sleep(self.interval)

    def run_once(self):
        ds = CspDataStoreAUs(self.filename_fmt)
        ds.add_au_properties(self.structure_queue)
        ds.close()