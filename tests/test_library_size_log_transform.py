"""Library-size values logged inside their scaling operation.

``normed(log_transform=True)`` of an assay that normalizes with
``norm_lib_size`` takes ``log1p`` in place inside the scaling operation, so a
block of the values holds one float64 array, as the plan of a block now says.
The values are bit-identical to scaling and then logging in a second
operation, as ``normed`` did before. Other normalizers keep ``log1p`` as an
operation of its own.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytearc.assay.normalization import norm_dummy, norm_lib_size
from cytearc.datastore.datastore import DataStore
from tests.storage_helpers import write_count_store

N_CELLS = 40
SUBSET = np.array([1, 2, 4, 7, 8, 11], dtype=np.int64)
CELLS = np.array([0, 3, 5, 9, 12, 13, 20, 31, 39], dtype=np.int64)
COUNT_DTYPES = ["uint8", "uint16", "int32", "float32", "float64"]


def _counts(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    counts = rng.poisson(3.0, size=(N_CELLS, 12)).astype(np.float64)
    if np.dtype(dtype).kind == "f":
        # Fractional counts, such as corrected ones, that float32 holds
        # exactly, so every total is exact in float64.
        counts += rng.integers(0, 4, size=counts.shape) / 4
    # A cell without counts, and one without counts among the subset.
    counts[5] = 0
    counts[9, SUBSET] = 0
    return counts


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict[str, tuple[DataStore, np.ndarray]]:
    opened = {}
    for dtype in COUNT_DTYPES:
        counts = _counts(dtype)
        path = Path(tmp_path_factory.mktemp(f"library_size_log_{dtype}")) / "store.zarr"
        write_count_store(str(path), {"RNA": counts}, dtype)
        # Every cell stays active, including the one without counts.
        store = DataStore(
            str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
        )
        opened[dtype] = (store, counts)
    return opened


@pytest.mark.parametrize("renormalize_subset", [False, True])
@pytest.mark.parametrize("dtype", COUNT_DTYPES)
def test_logged_values_are_those_of_scaling_then_logging(
    stores, dtype, renormalize_subset
):
    store, counts = stores[dtype]
    rna = store.RNA
    assert rna.normMethod is norm_lib_size
    selection: dict[str, Any] = {
        "cell_idx": CELLS,
        "feat_idx": SUBSET,
        "renormalize_subset": renormalize_subset,
    }
    values = rna.normed(log_transform=True, **selection).compute()

    # The chain that normed built before: the scaling, then log1p in float64
    # as an operation of its own.
    two_steps = np.log1p(rna.normed(**selection), dtype=np.float64).compute()
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, two_steps)

    # The arithmetic, independent of the chains: the size factor times the
    # counts, over each cell's total, then log1p, all in float64. The totals
    # are exact, so every step rounds as it does in the operation.
    selected = counts[CELLS][:, SUBSET]
    totals = (selected if renormalize_subset else counts[CELLS]).sum(axis=1)
    totals[totals == 0] = 1
    expected = np.log1p(rna.sf * selected / totals[:, None])
    np.testing.assert_array_equal(values, expected)


def test_library_size_scaling_logs_in_one_operation(stores):
    rna = stores["uint16"][0].RNA
    assert len(rna.normed(log_transform=True)._ops) == 1
    assert len(rna.normed(log_transform=True, renormalize_subset=True)._ops) == 1
    assert len(rna.normed()._ops) == 1


def doubled(_assay: Any, counts: Any) -> Any:
    """A custom normalizer whose values are not logarithms."""
    return 2.0 * counts


@pytest.mark.parametrize("normalizer", [norm_dummy, doubled])
def test_other_normalizers_log_in_an_operation_of_their_own(
    stores, monkeypatch, normalizer
):
    store, counts = stores["uint16"]
    rna = store.RNA
    monkeypatch.setattr(rna, "normMethod", normalizer)
    unlogged = rna.normed()
    logged = rna.normed(log_transform=True)

    assert len(logged._ops) == len(unlogged._ops) + 1
    assert logged._ops[-1].kind == "unary"
    scale = 1.0 if normalizer is norm_dummy else 2.0
    np.testing.assert_array_equal(logged.compute(), np.log1p(scale * counts))


def test_a_logged_block_plans_one_float64_array(stores):
    rna = stores["uint16"][0].RNA
    scaled = rna.normed()
    logged = rna.normed(log_transform=True)
    two_steps = np.log1p(scaled, dtype=np.float64)
    rows = min(scaled._block_size, scaled.shape[0])
    values = rows * scaled.shape[1]

    # Logging adds nothing to what a block holds while it is computed.
    assert logged._block_owned_bytes() == scaled._block_owned_bytes()
    assert logged._block_task_bytes() == scaled._block_task_bytes()
    assert logged.stream_bytes() == scaled.stream_bytes()
    # Scaling and then logging held two float64 arrays of the block at once.
    assert two_steps._block_owned_bytes() == 16 * values
    # One operation holds what the read of the uint16 block holds beside one
    # float64 array: here the block and Zarr's shard-level copy of it.
    held = scaled._read_bytes(2 * values) - scaled._read_bytes(0)
    assert held == 4 * values
    assert logged._block_owned_bytes() == held + 8 * values
