"""Read and set native thread counts in tests, through threadpoolctl itself.

The package limits threads only through ``cytearc.utils.compute`` and
``cytearc.utils.numba``; these helpers read the counts with threadpoolctl's own
controllers, so a test checks the package against an independent reading.
BLAS counts apply to the whole process and OpenMP counts to the calling thread.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from threadpoolctl import ThreadpoolController, threadpool_limits


def blas_threads() -> set[int]:
    """The thread counts of every loaded BLAS library."""
    return {
        library.num_threads
        for library in ThreadpoolController().lib_controllers
        if library.user_api == "blas"
    }


def openmp_threads() -> set[int]:
    """The calling thread's count in every loaded OpenMP runtime."""
    return {
        library.num_threads
        for library in ThreadpoolController().lib_controllers
        if library.user_api == "openmp"
    }


@contextmanager
def raised_threads(blas: int = 2, openmp: int | None = None) -> Iterator[None]:
    """Raise the BLAS count, and the calling thread's OpenMP count, for a test.

    The test session pins BLAS to one thread, so a limit of one shows only
    once the count around it is higher. Skips without a multi-threaded BLAS,
    or without an OpenMP runtime when ``openmp`` is given.
    """
    limits = {"blas": blas} if openmp is None else {"blas": blas, "openmp": openmp}
    with threadpool_limits(limits=limits):
        if blas_threads() != {blas}:
            pytest.skip(f"needs a BLAS that runs {blas} threads")
        if openmp is not None and openmp_threads() != {openmp}:
            pytest.skip(f"needs an OpenMP runtime that runs {openmp} threads")
        yield


def load_openmp_runtime() -> None:
    """Load scikit-learn's OpenMP runtime, as the operations that use it do."""
    import sklearn.cluster  # noqa: F401
