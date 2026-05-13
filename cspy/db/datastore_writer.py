from cspy.db import CspDataStore
from time import sleep
import logging

LOG = logging.getLogger(__name__)


class DatastoreWriter:

    def __init__(
            self,
            structure_queue,
            filename_format="csp_{db_id}.db",
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
                LOG.info("Ending DatastoreWriter thread")
                break

            for db_id in self.structures:
                structures = self.structures[db_id]
                if len(structures) < 1:
                    LOG.debug("Nothing to process in db_id %s", db_id)
                    continue
                filename = self.filename_fmt.format(db_id=db_id)
                try:
                    ds = CspDataStore(filename)
                    LOG.debug("%s Saving %d structures", db_id, len(structures))
                    ds.add_csp_results(structures)
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


class QRBH_DatastoreWriter(DatastoreWriter):

    def __init__(
        self,
        structure_queue,
        trial_queue,
        filename_format="csp_{sg}.db",
        interval=0.1,
    ):
        super().__init__(
            structure_queue,
            filename_format=filename_format,
            interval=interval,
        )

        self.trial_queue = trial_queue
        self.trials = {}

    def pop_trials(self):
        for sg, l in self.trial_queue.items():
            if sg not in self.trials:
                self.trials[sg] = []
            while len(l) > 0:
                seed, trial = l.popleft()
                meta = {
                    "res_old": trial.res_old,
                    "en_old": trial.en_old,
                }
                self.trials[sg].append(
                    (trial.id, seed, trial.iterations, trial.state, meta)
                )

    def run(self):
        while True:
            self.pop_structures()
            self.pop_trials()
            if ( self.complete
                and all(len(x) == 0 for x in self.structures.values())
                and all(len(x) == 0 for x in self.trials.values())
            ):
                LOG.info("Ending DatastoreWriter thread")
                break
            for sg in self.structures:
                structures = self.structures[sg]
                trials = self.trials[sg]
                if len(structures) < 1 and len(trials) < 1:
                    LOG.debug("Nothing to process in sg %s", sg)
                    continue
                filename = self.filename_fmt.format(sg=sg)
                try:
                    ds = CspDataStore(filename)

                    if len(trials) > 0:
                        ds.add_trials(trials)
                        self.trials[sg] = []
                        LOG.debug(
                            "%s succesfully updated trial", sg
                        )

                    if len(structures) > 0:
                        LOG.debug("%s Saving %d structures", sg, len(structures))
                        ds.add_qrbh_results(structures)
                        LOG.debug(
                            "%s succesfully saved %d structures", sg, len(structures)
                        )
                        self.structures[sg] = []
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


class Threshold_DatastoreWriter(QRBH_DatastoreWriter):
    def __init__(
        self,
        structure_queue,
        trial_queue,
        filename_format="csp_{sg}.db",
        interval=0.1,
    ):
        super().__init__(
            structure_queue,
            trial_queue,
            filename_format=filename_format,
            interval=interval,
        )
        self.trial_step = {}

    def pop_trials(self):
        for sg, trial_state_dict in self.trial_queue.items():
            if sg not in self.trials:
                self.trials[sg] = []
                self.trial_step[sg] = 0
            for seed, trial in trial_state_dict.items():
                trial_id, iterations, state, meta = trial.generate_restart()
                if self.trial_step[sg] != iterations:
                    self.trials[sg].append(
                        (trial_id, seed, iterations, state, meta)
                    )
                    self.trial_step[sg] = iterations

class Threshold_Reopt_DatastoreWriter(DatastoreWriter):
	def __init__(
			self,
			structure_queue,
			filename_format="csp_{db_id}.db",
			interval=0.1
	):
		super().__init__(
			structure_queue, 
			filename_format, 
			interval
		)
	
	def run(self):
		while True:
			self.pop_structures()
			if self.complete and all(len(x) == 0 for x in self.structures.values()):
				LOG.info("Ending DatastoreWriter thread")
				break
			for db_id in self.structures:
				structures = self.structures[db_id]
				if len(structures) < 1:
					LOG.debug("Nothing to process in db_id %s", db_id)
					continue
				filename = self.filename_fmt.format(db_id=db_id)
				try:
					ds = CspDataStore(filename)
					LOG.debug("%s Saving %d structures", db_id, len(structures))
					ds.add_qrbh_results(structures)
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
