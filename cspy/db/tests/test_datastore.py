from cspy.db import DataStore, CspDataStoreFlex
import os
import sys

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory


def test_create_connect():
    with TemporaryDirectory() as tempdir:
        filename = "temp.db"
        dstore = DataStore.create_and_connect(filename)
        assert dstore.schema_version == "unknown"
        dstore.disconnect()
        os.remove(filename)

def test_create_connect_flex():
    with TemporaryDirectory() as tempdir:
        filename = "temp_flex.db"
        dstore = CspDataStoreFlex(filename)
        assert dstore.schema_version == "cspy2_flex"
        dstore.disconnect()
        os.remove(filename)