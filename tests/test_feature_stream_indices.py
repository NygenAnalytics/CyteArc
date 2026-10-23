"""Feature streams refuse feature and cell selections they would misread."""

import numpy as np
import pytest

from cytearc.storage.feature_stream import (
    map_feature_cell_bands,
    map_feature_read_groups,
)
from tests.qc_helpers import QC_VALUES, fresh_qc_store, open_qc_store


@pytest.fixture(scope="module")
def counts_t():
    store, _ = fresh_qc_store()
    return open_qc_store(store).RNA.rawDataT


def _groups(counts_t, **selection):
    return list(map_feature_read_groups(counts_t, lambda group: group, **selection))


def _bands(counts_t, **selection):
    return list(map_feature_cell_bands(counts_t, lambda band: band, **selection))


@pytest.mark.parametrize("stream", [_groups, _bands])
@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ({"feat_idx": [-1]}, IndexError, "feat_idx contains an out-of-range index"),
        ({"feat_idx": [6]}, IndexError, "feat_idx contains an out-of-range index"),
        ({"feat_idx": [0, 600]}, IndexError, "feat_idx contains an out-of-range"),
        (
            {"feat_idx": np.ones(6, dtype=bool)},
            TypeError,
            r"feat_idx must hold integer indices, not bool values; pass "
            r"np.flatnonzero\(mask\)",
        ),
        ({"feat_idx": [0.0, 1.0]}, TypeError, "feat_idx must hold integer indices"),
        ({"feat_idx": [[0, 1]]}, ValueError, "feat_idx must be one-dimensional"),
        ({"cell_idx": [-1]}, IndexError, "cell_idx contains an out-of-range index"),
        ({"cell_idx": [6]}, IndexError, "cell_idx contains an out-of-range index"),
        (
            {"cell_idx": np.array([True, False, True, True, False, True])},
            TypeError,
            "cell_idx must hold integer indices, not bool values",
        ),
    ],
)
def test_streams_refuse_selections_they_would_misread(
    counts_t, stream, selection, error, message
):
    with pytest.raises(error, match=message):
        stream(counts_t, **selection)


def test_valid_selections_read_repeated_and_unordered_cells(counts_t):
    cells = np.array([5, 0, 0, 2])
    groups = _groups(counts_t, cell_idx=cells, feat_idx=np.array([3], dtype=np.uint8))
    values = np.concatenate([group.values for group in groups])
    starts = [group.featStart for group in groups]
    # The read group of feature 3 holds the features of its chunk, 2 and 3.
    assert starts == [2]
    np.testing.assert_array_equal(values, QC_VALUES.T[2:4][:, cells])
    assert _groups(counts_t, feat_idx=np.array([], dtype=np.int64)) == []
    assert _bands(counts_t, feat_idx=[]) == []
    bands = _bands(counts_t, cell_idx=[4, 1], feat_idx=[5])
    assert {(band.featStart, band.featEnd) for band in bands} == {(4, 6)}
