import os
import signal
from subprocess import Popen
from threading import Thread
import logging
import psutil

LOG = logging.getLogger(__name__)


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


class PopenTimeout(object):
    def __init__(self, cmd, input_file=None, output_file=None, shell=False):
        self.cmd = cmd
        self.input_file = input_file
        self.output_file = output_file
        self.process = None
        self.shell = shell

    def run(self, timeout):
        def target():
            self.process = Popen(
                self.cmd,
                stdin=self.input_file,
                stdout=self.output_file,
                shell=self.shell,
            )
            self.process.communicate()

        try:
            thread = Thread(target=target)
            thread.start()

            thread.join(timeout)

            if thread.is_alive():
                kill(self.process.pid)
            return self.process.returncode
        except Exception as e:
            LOG.error("Exception occured in run(): %s", e)
            thread.join()
            return 130
        return 130


class PopenStorePID(object):
    def __init__(self, cmd, input_file=None, output_file=None, shell=False):
        self.cmd = cmd
        self.input_file = input_file
        self.output_file = output_file
        self.process = None
        self.shell = shell

    def run(self, timeout, pid_dict, key, lock):
        def target():
            self.process = Popen(
                self.cmd,
                stdin=self.input_file,
                stdout=self.output_file,
                shell=self.shell,
            )
            self.process.communicate()

        thread = Thread(target=target)
        thread.start()
        lock.acquire()
        pid_dict[key] = self.pid
        lock.release()
        thread.join(timeout)
        if thread.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except OSError as exc:
                LOG.exception("OSERROR in killpg")
            self.process.kill()
            os.kill(self.process.pid, signal.SIGKILL)
            thread.join()
            return 130
        return self.process.returncode
