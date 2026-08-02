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


class WorkQueue:

    def __init__(self, mpi4py_workers):
        """Handles a work queue for a particular master.

        This is an edited version of the WorkQueue class from
        the mpi-master-slave library,

        https://github.com/luca-s/mpi-master-slave

        Parameters
        ----------
        mpi4py_workers: iterable
            An iterable of the rank numbers of the worker processes.
        """
        from .master import Master

        self.master = Master(mpi4py_workers)
        self.work_queue = []

    def __contains__(self, job):
        """Overrides the contains operator to allow logical statements
        be carried out on the WorkQueue objects.

        :param job: The job that is checked.
        :return: A boolean of whether the job is in the work queue.
        """
        if job in self.work_queue:
            return True
        else:
            return False

    @property
    def empty(self):
        """Return True if all the work queues are empty, some workers
        might still be running calculations and there may be results
        to be gathered from the master.

        Returns
        -------
        bool
            Boolean of whether work queues are empty.
        """
        return not self.work_queue

    @property
    def done(self):
        """Return True if the work queues are empty and there are no
        workers running.

        Returns
        -------
        bool
            Boolean if all work is done.
        """
        return self.empty and not self.master.running

    @property
    def results(self):
        """Yield the results from the master for a worker that has
        completed its calculations.

        Returns
        -------
        tuple
            Yields a tuple of the results.
        """
        return self.master.results

    @property
    def num_jobs(self):
        """Number of jobs in the work queue.

        Returns
        -------
        int
            Number of jobs in the work queue.
        """
        return len(self.work_queue)

    @property
    def idle(self):
        """Returns a set of workers that are not running any jobs.

        Returns
        -------
        set
            A set of idle workers.
        """
        return self.master.idle

    @property
    def running(self):
        """Returns a set of workers that are running jobs.

        Returns
        -------
        set
            A set of running workers.
        """
        return self.master.running

    @property
    def num_ready(self):
        """Returns the number of workers that are ready.

        Returns
        -------
        int
            Number of workers that are ready.
        """
        return self.master.num_ready

    def prepend_job(self, job):
        """Add a job to the work queue.

        Parameters
        ----------
        job: tuple
            Job to be prepended to the work queue.
        """
        self.work_queue.insert(0, job)

    def append_job(self, job):
        """Add a job to the work queue.

        Parameters
        ----------
        job: tuple
            Job to be appended to the work queue.
        """
        self.work_queue.append(job)

    def do_work(self):
        """Assign a job stored in the work queue to each idle worker
        (if any) and run the calculation.

        """
        for worker in self.master.ready:

            if self.empty:
                break

            job = self.work_queue.pop(0)
            self.master.run_job(worker, job)

    def terminate_workers(self):
        """Tells master to stop all of its workers.

        """
        self.master.terminate_workers()
