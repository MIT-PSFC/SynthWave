"""
Test configuration for SynthWave.

Tests marked ``serial`` must not run concurrently with any other test.
They also trigger explicit garbage collection after each test to ensure
C++ objects (ThinCurr, OFT_env) are finalized before the next test runs.
"""

import gc

import pytest
from OpenFUSIONToolkit import OFT_env


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "serial: mark test as requiring exclusive, sequential execution "
        "(no parallelism). GC is forced after each such test to ensure "
        "C++ resources from OpenFUSIONToolkit are released before the next "
        "test starts.",
    )


@pytest.fixture(autouse=True)
def _serial_gc(request):
    """Force a full GC cycle after every serial-marked test."""
    yield
    if request.node.get_closest_marker("serial"):
        gc.collect()


# Fixture for OFT environment so only one is created for all tests.
@pytest.fixture(scope="session")
def oft_env():
    return OFT_env(nthreads=2)
