import os

# Bound CyteArc worker auto-detection before pytest plugins import NumPy or CyteArc.
os.environ["CYTEARC_WORKERS"] = "2"
# Interactive backends such as TkAgg bind figures to the thread that created
# them. Garbage collection triggered on a Zarr I/O thread then finalizes those
# objects on the wrong thread and crashes the worker with an illegal instruction.
os.environ["MPLBACKEND"] = "Agg"
# OpenMP threads, which run Numba's parallel kernels, spin after each parallel
# region. With a pytest-xdist worker per CPU, the spinning threads of one worker
# held the CPUs that another worker's threads needed to reach their barriers,
# which slowed the Leiden and PARIS tests up to a hundredfold. In the workers,
# idle threads sleep instead; a single process, such as a timed benchmark run,
# keeps OpenMP's default. xdist names the worker count before this module is
# imported, and OpenMP reads the policy when it loads, later.
if int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")) > 1:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import sys

import pytest
from threadpoolctl import threadpool_limits

from cytearc.utils import configure_output, logger

pytest_plugins = [
    "tests.fixtures_agent",
    "tests.fixtures_downloader",
    "tests.fixtures_readers",
    "tests.fixtures_datastore",
    "tests.fixtures_hub",
    "tests.shards",
]


@pytest.fixture(scope="session", autouse=True)
def _limit_blas_threads():
    with threadpool_limits(limits=1, user_api="blas"):
        yield


@pytest.fixture(scope="session", autouse=True)
def _share_numba_threads():
    """Run each xdist worker's Numba kernels on its share of the CPUs by default.

    Tests that pass a thread count still get up to Numba's whole pool, and at
    least two threads keep the kernels' parallel paths in use.
    """
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    if workers <= 1:
        yield
        return
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, max(2, previous // workers)))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


@pytest.fixture(scope="session", autouse=True)
def _quiet_test_logs() -> None:
    """Keep pytest output readable while standalone CLIs retain INFO logs."""
    configure_output(progress=False)
    logger.remove()
    logger.add(sys.stderr, level="ERROR")


@pytest.fixture(autouse=True)
def _reset_zarr_runtime() -> None:
    from tests.storage_helpers import reset_zarr_runtime

    reset_zarr_runtime()
    yield
    reset_zarr_runtime()
