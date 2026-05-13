import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gaussian",
        action="store_true",
        dest="gaussian",
        default=False,
        help="Enable Gaussian tests",
    )


def pytest_runtest_setup(item):
    if "rungaussian" in item.keywords and not item.config.getvalue("gaussian"):
        pytest.skip("need --gaussian option to run")
