import os
import logging
import threading
import ast
import shutil
import numpy as np
from collections import deque
from rdkit import Chem
from rdkit.Chem.rdqueries import AtomNumEqualsQueryAtom
from cspy.cspympi import WorkQueue
from cspy.db.datastore_writer import DatastoreWriter
from .mob_jobs import MobTaskTag


LOG = logging.getLogger(__name__)


class MobWorkQueue(WorkQueue):

    def __init__(self, mpi4py_workers, config, sub_comm_nums):
        super().__init__(mpi4py_workers)
        self.start_job_tag = MobTaskTag.REORG
        self.sleep_time = 1
        self.batch_size = 500
        self.sub_comm_nums = sub_comm_nums
        self.space_groups = ast.literal_eval(config['CSP']['space_groups'])
        self.sampling = ast.literal_eval(config['CSP']['sampling'])
        self.energy_scaling = float(config['MOBBUILDER']['energy_scaling'])
        self.workers_to_sleep = set()
        self.results_dict = {}
        self.successful_minimizations = {}
        self.db_writer = DatastoreWriter(
            self.successful_minimizations,
            filename_format=f'{{db_id}}',
            interval=0.5,
        )
        self.db_thread = threading.Thread(
            target=self.db_writer.run, name=f'dbworker'
        )
        self.db_thread.start()

    @property
    def results(self):
        """Yield the results from the master for a worker that has
        completed its calculations.

        Returns
        -------
        tuple
            Yields a tuple of the results.
        """
        for task, (inchi, results) in self.master.results:

            if task == MobTaskTag.FAILED:
                inchi_key = Chem.InchiToInchiKey(inchi)
                shutil.make_archive(
                    os.getcwd() + '/molbuilderlogs/' + inchi_key, 'zip',
                    os.getcwd() + '/molbuilderlogs/' + inchi_key
                )
                shutil.rmtree(os.getcwd() + '/molbuilderlogs/' + inchi_key)
                if inchi in self.results_dict:
                    del self.results_dict[inchi]
                yield inchi, None

            elif task == MobTaskTag.REORG:
                self.results_dict[inchi] = MobFitnessResults(inchi,
                    self.space_groups, self.energy_scaling)
                reorg, affinity = results
                self.results_dict[inchi].reorg = reorg
                self.results_dict[inchi].affinity = affinity
                self.append_job((MobTaskTag.MULT, (inchi, None)))

            elif task == MobTaskTag.MULT:
                qr_info = self.results_dict[inchi].qr_info
                for sg, sample in zip(self.space_groups, self.sampling):
                    for min_seed in range(1, sample + 1, self.batch_size):
                        max_seed = min(sample, min_seed + self.batch_size)
                        self.append_job((
                            MobTaskTag.QR,
                            (inchi, (sg, (min_seed, max_seed)))
                        ))
                    qr_info[sg] = sample

            elif task == MobTaskTag.QR:
                qr_info = self.results_dict[inchi].qr_info

                sg, crystals = results
                failed = 0
                for crystal in crystals:
                    if crystal is None:
                        failed += 1
                    else:
                        self.append_job((MobTaskTag.OPT, (inchi, crystal)))

                last_seed = qr_info[sg]
                for min_seed in range(last_seed + 1, last_seed + failed + 1, self.batch_size):
                    max_seed = min(last_seed + failed, min_seed + self.batch_size)
                    self.append_job((
                        MobTaskTag.QR,
                        (inchi, (sg, (min_seed, max_seed)))
                    ))
                qr_info[sg] = last_seed + failed

            elif task == MobTaskTag.OPT:
                opt_info = self.results_dict[inchi].opt_info
                qr_info = self.results_dict[inchi].qr_info
                csp_comp = self.results_dict[inchi].csp_complete

                sg, db_id, crystals, valid = results
                if valid:
                    opt_info[sg] += 1
                    if db_id not in self.successful_minimizations:
                        self.successful_minimizations[db_id] = deque()
                    for crystal in crystals:
                        self.successful_minimizations[db_id].append(
                            crystal)
                else:
                    last_seed = qr_info[sg]
                    self.append_job((
                        MobTaskTag.QR,
                        (inchi, (sg, (last_seed + 1, last_seed + 1)))
                    ))
                    qr_info[sg] += 1

                for i, sg in enumerate(self.space_groups):
                    if opt_info[sg] == self.sampling[i]:
                        csp_comp[sg] = True

                if self.results_dict[inchi].csp_all_complete:
                    self.prepend_job((MobTaskTag.CP2K, (inchi, None)))

            elif task == MobTaskTag.CP2K:
                inchi_key = Chem.InchiToInchiKey(inchi)
                for i, sg in enumerate(self.space_groups):
                    db_id = os.getcwd() + '/molbuilderlogs/{}/csp/{}_{}.db'.format(
                        inchi_key, inchi_key, sg)
                    del self.successful_minimizations[db_id]

                mo_number, structures = results
                reorg = self.results_dict[inchi].reorg
                crystal_data = self.results_dict[inchi].unique_crystals
                for structure in structures:
                    crystal_data[structure.id] = {
                        'energy': structure.energy,
                        'mobility': None
                    }
                    self.append_job((
                        MobTaskTag.MOB, (inchi, (reorg, mo_number, structure))
                    ))

            elif task == MobTaskTag.MOB:
                cid, mobility = results
                crystal_data = self.results_dict[inchi].unique_crystals
                crystal_data[cid]['mobility'] = mobility

                if self.results_dict[inchi].mob_all_complete:
                    inchi_key = Chem.InchiToInchiKey(inchi)
                    shutil.make_archive(
                        os.getcwd() + '/molbuilderlogs/' + inchi_key, 'zip',
                        os.getcwd() + '/molbuilderlogs/' + inchi_key
                    )
                    shutil.rmtree(os.getcwd() + '/molbuilderlogs/' + inchi_key)
                    results = self.results_dict[inchi]
                    fitness = results.fitness
                    LOG.info('%s, reorganisation_energy: %s, '
                             'electron_affinity: %s, '
                             'number_of_nitrogens: %s, '
                             'mobility: %s '
                             'fitness: %s', inchi,
                             results.reorg,
                             results.affinity,
                             results.number_of_nitrogens,
                             fitness, fitness)
                    del self.results_dict[inchi]
                    yield inchi, fitness

    def do_work(self):
        """Sends a workers to sleep or assigns a job stored in the work
        queue to each idle worker (if any) and runs the calculation.

        """
        while True:
            if len(self.workers_to_sleep) != 0:
                self.send_to_sleep()

            if len(self.master.ready) == 0 or self.empty:
                break

            worker = next(iter(self.master.ready))
            job = self.work_queue.pop(0)
            if job[0] == MobTaskTag.REORG:
                self.master.run_job(worker, job)
                for rank in self.sub_comm_workers(worker):
                    if rank != worker:
                        self.workers_to_sleep.add(rank)
            else:
                self.master.run_job(worker, job)

    def send_to_sleep(self):
        """Sends out a sleep job to the workers when the multicore
        gaussian calculation are running if they are in the same sub
        node group.

        """
        for worker in self.master.ready:
            if worker in self.workers_to_sleep:
                self.master.run_job(worker, (MobTaskTag.SLEEP, (None, None)))
                self.workers_to_sleep.remove(worker)

    def sub_comm_workers(self, rank):
        """Get the workers all the workers in the same sub node group.

        :param rank: the rank of the worker
        :return: a list of worker that are in the same sub node group
        """
        sub_comm_num = self.sub_comm_nums[rank]

        workers = []
        for key, value in self.sub_comm_nums.items():
            if value == sub_comm_num:
                workers.append(key)

        return workers

    def terminate_workers(self):
        LOG.info('Waiting to finish writing database')
        if self.db_thread is not None:
            self.db_writer.complete = True
            self.db_thread.join()
        LOG.info('Done writing databases')
        self.master.terminate_workers()


class MobFitnessResults:

    def __init__(self, inchi, space_groups, energy_scaling):
        self.inchi = inchi
        self.space_groups = space_groups
        self.energy_scaling = energy_scaling
        self.number_of_nitrogens = len(
            Chem.MolFromInchi(inchi).GetAtomsMatchingQuery(
                AtomNumEqualsQueryAtom(7)
            ))
        self.reorg = None
        self.affinity = None
        self.qr_info = {}
        self.opt_info = {sg: 0 for sg in space_groups}
        self.csp_complete = {sg: False for sg in space_groups}
        self.unique_crystals = {}

    @property
    def csp_all_complete(self):
        return all([self.csp_complete[sg] for sg in self.space_groups])

    @property
    def mob_all_complete(self):
        return all([
            crystal['mobility'] is not None
            for crystal in self.unique_crystals.values()
        ])

    @property
    def fitness(self):
        energies = [
            crystal['energy'] for crystal in self.unique_crystals.values()
        ]
        mobilities = [
            crystal['mobility'] for crystal in self.unique_crystals.values()
        ]

        summation = sum([np.exp(-k / 2.70) for k in energies])
        avg = 0
        for j in range(len(energies)):
            avg += mobilities[j] * np.exp(-energies[j] / self.energy_scaling)
        avg /= summation
        return avg
