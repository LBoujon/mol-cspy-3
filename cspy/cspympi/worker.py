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
import psutil
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI
from enum import IntEnum


LOG = logging.getLogger(__name__)


class WorkerCommTag(IntEnum):
    READY = 1
    START = 2
    DONE = 3
    EXIT = 4


class Worker(ABC):

    def __init__(self):
        """A worker abstract class, extend this class, create an
        instance and invoke the run process.

        This is an edited version of the Slave class from the
        mpi-master-slave library,

        https://github.com/luca-s/mpi-master-slave.

        """
        self.comm = MPI.COMM_WORLD
        self.name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()

    def run(self):
        """Invoke this method when ready to put this worker to work.

        """
        LOG.info('starting MPI worker %s rank %s property calculator with a '
                 'cpu affinity of %s',
                 self.name, self.rank, psutil.Process().cpu_affinity())
        status = MPI.Status()

        while True:
            LOG.debug('MPI worker %s rank %s is sending status tag READY',
                      self.name, self.rank)
            self.comm.send(None, dest=0, tag=WorkerCommTag.READY)

            job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            LOG.debug('MPI worker %s rank %s has received %s and status tag '
                      '%s', self.name, self.rank, job, WorkerCommTag(tag).name)

            if tag == WorkerCommTag.START:

                LOG.debug('MPI worker %s rank %s is starting calculation on '
                          '%s', self.name, self.rank, job)

                result = self.calculate(job)
                LOG.debug('MPI worker %s rank %s has completed calculation '
                          'for %s with result %s, sending results to master',
                          self.name, self.rank, job, result)

                self.comm.send(result, dest=0, tag=WorkerCommTag.DONE)

            elif tag == WorkerCommTag.EXIT:
                break

        LOG.debug('MPI worker %s rank %s is sending status tag EXIT',
                  self.name, self.rank)
        self.comm.send(None, dest=0, tag=WorkerCommTag.EXIT)
        LOG.info('ending MPI worker %s rank %s property calculator',
                 self.name, self.rank)

    @abstractmethod
    def calculate(self, job):
        """An abstract calculate method, override this method to do
        specific property calculations on the job for the molecule.

        Parameters
        ----------
        job: tuple
            Should be a tuple containing the task to be done as well
            as the args for that task.
        """
        pass
