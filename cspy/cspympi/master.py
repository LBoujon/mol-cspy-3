"""
MIT License

Copyright (c) 2018 Luca Scarabello

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
import time
from mpi4py import MPI
from .worker import WorkerCommTag


LOG = logging.getLogger(__name__)


class WorkerNotReady(Exception):
    """Raise exception when attempting to give a non-ready worker a job.

    This is an edited version of the SlaveNotReady class from
    the mpi-master-slave library,

    https://github.com/luca-s/mpi-master-slave

    """

class Master:

    def __init__(self, workers=None):
        """The main process creates one or more of these masters which
        handles a group of worker processes.

        This is an edited version of the Master class from the
        mpi-master-slave library,

        https://github.com/luca-s/mpi-master-slave

        Parameters
        ----------
        workers: iterable
            An iterable of the rank numbers of the worker processes.
        """
        self.comm = MPI.COMM_WORLD
        self.name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        self.status = MPI.Status()

        LOG.info('MPI master %s rank %s is starting', self.name, self.rank)

        if workers is None:
            workers = []

        self.workers = set(workers)
        self._ready = set()
        self.running = set()

        self.num_workers = len(self.workers)

        # sleep until all workers are init and ready
        while self.num_ready != self.num_workers:
            time.sleep(1)

    @property
    def ready(self):
        """Checks if there are any workers that have become ready
        since the last time this method was called and returns an
        updated set of ready workers. Workers are ready if they are not
        running any jobs and have sent the READY tag.

        Returns
        -------
        set
            A set of workers that are ready to to do work.
        """

        for i in self.workers - (self._ready | self.running):

            if not self.comm.Iprobe(source=i, tag=WorkerCommTag.READY):
                continue

            self.comm.recv(source=i, tag=WorkerCommTag.READY,
                           status=self.status)
            worker = self.status.Get_source()
            self._ready.add(worker)
            LOG.debug('MPI master %s rank %s has received READY from worker '
                      'rank %s', self.name, self.rank, worker)

        return self._ready - self.running

    @property
    def num_ready(self):
        """Returns the number of workers that are ready.

        Returns
        -------
        int
            Number of workers that are ready.
        """
        return len(self.ready)

    @property
    def idle(self):
        """Returns a set of workers that are not running any jobs.

        Returns
        -------
        set
            A set of idle workers.
        """
        return self.workers - self.running

    @property
    def results(self):
        """Yields the results from a worker that has completed
        a job and removes the worker from the running set.

        Returns
        -------
        tuple
            Yields the results sent from a worker.
        """
        for i in set(self.running):

            if not self.comm.Iprobe(source=i, tag=WorkerCommTag.DONE):
                continue

            results = self.comm.recv(source=i, tag=WorkerCommTag.DONE,
                                     status=self.status)
            worker = self.status.Get_source()
            LOG.debug('MPI master %s rank %s has received results '
                      '%s from worker rank %s', self.name, self.rank,
                      results, worker)

            self.running.remove(worker)

            yield results

    def run_job(self, worker, job):
        """Sends an job to the worker causing the worker to run the
        job and removes it from the _ready set and add it to
        the running set.

        Parameters
        ----------
        worker: int
            The rank of the MPI worker.
        job: tuple
            Job for the worker to run.
        """
        if worker in self.ready:
            LOG.debug('MPI master %s rank %s is sending %s and tag START to '
                      'worker rank %s', self.name, self.rank, job, worker)
            self.comm.send(obj=job, dest=worker, tag=WorkerCommTag.START)
            self._ready.remove(worker)
            self.running.add(worker)

        # raise an exception if attempting to run a job on a busy worker
        else:
            raise WorkerNotReady(
                'Attempting to run a job on worker rank {} which is not '
                'READY.'.format(worker)
            )

    def terminate_workers(self):
        """Send EXIT tag to all workers to end their run loops and
        receives the EXIT tag to confirm termination.

        """
        for worker in self.workers:
            LOG.debug('MPI master %s rank %s is sending the tag EXIT to '
                      'worker rank %s', self.name, self.rank, worker)
            self.comm.send(obj=None, dest=worker, tag=WorkerCommTag.EXIT)

        for worker in self.workers:
            self.comm.recv(source=worker, tag=WorkerCommTag.EXIT)
            LOG.debug('MPI master %s rank %s has received the tag EXIT from '
                      'worker rank %s', self.name, self.rank, worker)
