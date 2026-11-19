"""Measure the temporary files of row bucket stores against the free disk that they ask for.

A store checks the free disk before it writes a row, against what it
estimates that its files will take, and again before each block. These
helpers report a chosen free disk to those checks, read the bytes that a
refusal names, and measure a store's files once it is finalized, when they
take the most: every bucket's files and the store's settings and records,
before a read deletes a bucket.
"""

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from cytearc.readers import _sparse
from cytearc.readers._sparse import SparseRowStore


def file_bytes(directory: Path) -> int:
    """Return the bytes of every file under ``directory``."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


@contextmanager
def finalized_bytes() -> Iterator[list[int]]:
    """Record the bytes of the files of each store finalized within, once it is finalized.

    A finalize that splits a bucket holds a copy of it for a while, which
    this would miss, so a split fails the test.
    """
    found: list[int] = []
    finalize = SparseRowStore.finalize

    def recorded(store: SparseRowStore) -> None:
        buckets = store.bucket_starts.size
        finalize(store)
        assert store.bucket_starts.size == buckets, "a bucket was split"
        found.append(file_bytes(store.directory))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SparseRowStore, "finalize", recorded)
        yield found


@contextmanager
def free_disk(free: int) -> Iterator[None]:
    """Report ``free`` bytes free to the disk checks of row bucket stores."""
    usage = shutil.disk_usage(Path.cwd())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            _sparse.shutil, "disk_usage", lambda path: usage._replace(free=free)
        )
        yield


def asked_bytes(refusal: BaseException) -> int:
    """Return the temporary bytes that a store's refusal of a full disk names."""
    found = re.search(r"needs (\d+) temporary bytes", str(refusal))
    assert found is not None, f"No temporary bytes are named in: {refusal}"
    return int(found.group(1))
