from os.path import abspath, dirname, join
import os
import platform


def is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(prog):
    fpath, fname = os.path.split(prog)
    if fpath and is_executable(fname):
        return prog
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, prog)
            if is_executable(exe_file):
                return exe_file
    return None


def use_profiling_executables():
    global DMACRYS_EXEC
    if not DMACRYS_EXEC.endswith(".prof"):
        DMACRYS_EXEC = DMACRYS_EXEC + ".prof"


def linux_version():
    """Get the version of linux this system is running
    If it's not linux, return None
    """
    s = platform.platform()
    if not s.startswith("Linux"):
        return None
    version_string = s.split("-")[1]
    return version_string


def libc_version():
    """Get the version of glibc this python was compiled with 
    return None if we don't have any info on it
    """
    try:
        version = platform.libc_ver()[1]
    except Exception as e:
        version = None
    return version


LINUX_VERSION = linux_version()
LIBC_VERSION = libc_version()
DIR = abspath(dirname(__file__))
PROGS_DIR = DIR.replace("cspy/executable", "progs")
COMPACK_DIR = join(PROGS_DIR, "compack_1_1/bin")

COMPACK_EXEC = join(COMPACK_DIR, "compack")
GDMA_EXEC = which("gdma")
MULFIT_EXEC = which("mulfit")
PLATON_EXEC = which("platon")
ZEOXX_EXEC = which("network")
NEIGHCRYS_EXEC = which("neighcrys")
DMACRYS_EXEC = which("dmacrys")
GAUSSIAN_EXEC = which("g09")
FORMCHK_EXEC = which("formchk")
PMIN_EXEC = which("pmin")
DFTB_EXEC = which ("dftb+")

EXECUTABLES = {
    "compack": COMPACK_EXEC,
    "gdma": GDMA_EXEC,
    "mulfit": MULFIT_EXEC,
    "platon": PLATON_EXEC,
    "zeo++": ZEOXX_EXEC,
    "neighcrys": NEIGHCRYS_EXEC,
    "dmacrys": DMACRYS_EXEC,
    "gaussian": GAUSSIAN_EXEC,
    "formchk": FORMCHK_EXEC,
    "pmin": PMIN_EXEC,
    "dftb": DFTB_EXEC,
}

if __name__ == "__main__":
    import os
    from cspy.util.terminal import passfail

    print("--------- CSPy external executables ---------")
    print("Platform: Linux {} | glibc {}".format(LINUX_VERSION, LIBC_VERSION))
    print("CSPy progs dir =", PROGS_DIR)
    for n, loc in EXECUTABLES.items():
        print(passfail("{: <12s} {}".format(n, loc), loc is not None))
