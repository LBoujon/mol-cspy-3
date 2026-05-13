from __future__ import print_function
import os
import errno
import logging
from cspy.cspy_exceptions import CSPyException

file_logger = logging.getLogger("CSPy.file_utils")


def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            file_logger.error("Could not make directory : " + path)
            raise CSPyException("Could not make directory : " + path)
        else:
            file_logger.warn("Directory exists : " + path)


def is_file_readable(filename):
    try:
        f = open(filename, "r")
    except IOError as exc:
        return False
    else:
        f.close()
        return True


def word_replace_in_file(filename, inline, newline, maxfound=99999, tmpfilename="tmp"):
    import shutil

    f = open(filename, "r")
    t = open(tmpfilename, "w")

    found = 0
    for line in f:
        if inline in line and found < maxfound:
            t.write(newline + "\n")
            found += 1
        else:
            t.write(line)

    f.close()
    t.close()
    shutil.move("tmp", filename)
    return found
