from __future__ import print_function


class CSPyException(Exception):
    def __init__(self, msg):
        self.msg = msg
        Exception.__init__(self, msg)  # required for celery/pickling/unpickling


class CSPyDebugException(Exception):
    pass


class DMACRYSInvalidMinimisationException(Exception):
    pass
