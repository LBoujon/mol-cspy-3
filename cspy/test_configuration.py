from cspy.configuration import CspyConfiguration, DEFAULT_CONFIG_LOCATIONS
import os
import toml
import sys

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory


def test_init():
    args = {"test_arg": True}
    config = CspyConfiguration(args)
    assert config["test_arg"]
    assert config.get("fkldsjfasfljasflkjas") is None


def test_toml():
    settings = {"test_arg": True}
    toml_location = DEFAULT_CONFIG_LOCATIONS[-1]
    with TemporaryDirectory():
        with open(toml_location, "w") as f:
            toml.dump(settings, f)

        config = CspyConfiguration()
        assert config["test_arg"]
        os.remove(toml_location)