import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--external-binaries",
        action="store_true",
        dest="binaries",
        default=False,
        help="Enable tests depending on external binaries",
    )
    parser.addoption(
        "--pynnp",
        action="store_true",
        dest="pynnp",
        default=False,
        help="Enable tests depending on pynnp",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "external_binaries: mark a test as requiring external binaries to run",
    )
    config.addinivalue_line(
        "markers",
        "pynnp: mark a test as requiring pynnp to run",
    )


def pytest_collection_modifyitems(config, items):
    #if config.getoption("--external-binaries"):
    #    return
    skip_binaries = pytest.mark.skip(reason="need --external-binaries option to run")
    skip_pynnp = pytest.mark.skip(reason="need --pynnp option to run")
    for item in items:
        if "external_binaries" in item.keywords and not config.getoption("--external-binaries"):
            item.add_marker(skip_binaries)
        if "pynnp" in item.keywords and not config.getoption("--pynnp"):
            item.add_marker(skip_pynnp)
