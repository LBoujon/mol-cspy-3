import errno
import os
import sys

if sys.version_info >= (3, 4):
    from pathlib import Path as _Path
    from tempfile import TemporaryDirectory
else:
    from backports.tempfile import TemporaryDirectory
    from pathlib2 import Path as _Path

ERR_INVALID_NAME = 123  # windows specific error code


class Path(type(_Path())):
    def copy_to(self, dest):
        if isinstance(dest, str):
            dest = Path(dest)

        if dest.is_dir():
            dest = dest / self.name

        dest.write_bytes(self.read_bytes())

    def append_to(self, dest):
        if isinstance(dest, str):
            dest = Path(dest)

        if dest.is_dir():
            dest = dest / self.name

        with dest.open("a") as f:
            with self.open() as f2:
                f.write(f2.read())


def is_valid_pathname(pathname):
    """ Return `True` if the passed string is a valid
    pathname for the current OS, `False` otherwise """
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip the drivename on windows
        _, pathname = os.path.splitdrive(pathname)

        root_dirname = (
            os.environ.get("HOMEDRIVE", "C:")
            if sys.platform == "win32"
            else os.path.sep
        )
        assert os.path.isdir(root_dirname)

        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as e:
                if hasattr(e, "winerror"):
                    if e.winerror == ERR_INVALID_NAME:
                        return False
                elif e.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError:
        return False

    return True


def is_path_creatable(pathname):
    """Return `True` if we have permissions to create
    the given pathname, `False` otherwise"""
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def path_exists_or_is_creatable(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists or is createable
    by the current user, `False` otherwise."""
    try:
        return is_valid_pathname(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname)
        )
    except OSError:
        return False


def dir_exists_or_is_creatable(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists as a dir or is
    createable by the current user, `False` otherwise."""
    try:
        if not is_valid_pathname(pathname):
            return False
        return (
            os.path.exists(pathname) and os.path.isdir(pathname)
        ) or is_path_creatable(pathname)
    except OSError:
        return False


def dir_exists(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists as a dir,
    `False` otherwise."""
    try:
        if not is_valid_pathname(pathname):
            return False
        return os.path.exists(pathname) and os.path.isdir(pathname)
    except OSError:
        return False


class working_directory:
    """Context manager for temporarily changing workdir """

    previous_workdir = None

    def __init__(self, directory, create=False, remove=False):
        if not is_valid_pathname(directory):
            raise ValueError("{} is not a valid pathname".format(directory))
        self.directory = os.path.expanduser(directory)
        self.create = create
        self.remove = remove

    def __enter__(self):
        self.previous_workdir = os.getcwd()
        if self.directory:
            if not dir_exists(self.directory) and self.create:
                os.mkdir(self.directory)
            os.chdir(self.directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.previous_workdir)
        if dir_exists(self.directory) and self.remove:
            os.rmdir(self.directory)
