"""A count shard lost from a store raises instead of reading as zero counts."""

import shutil

import numpy as np
import pytest
import zarr
from zarr.errors import ChunkNotFoundError
from zarr.storage import MemoryStore

from cytearc.storage.counts_t_contract import validate_count_matrix
from cytearc.storage.feature_stream import _settled, map_feature_read_groups
from cytearc.storage.schema import create_zarr_count_assay
from cytearc.storage.sharding import write_counts_t
from cytearc.tools.repack_zarr import repack_store
from tests.qc_helpers import QC_VALUES, fresh_qc_store, open_qc_store
from tests.storage_helpers import finalize_test_counts, write_count_store


def _prepared_store():
    store, _ = fresh_qc_store()
    open_qc_store(store)
    return store


def test_complete_counts_read_as_stored():
    ds = open_qc_store(_prepared_store())

    np.testing.assert_array_equal(ds.RNA.rawData.compute(), QC_VALUES)
    counts, counts_t = validate_count_matrix(ds.RNA.matrixGroup, require_transpose=True)
    for array in (counts, counts_t, ds.RNA.rawData._backing):
        assert array is not None
        assert array.config.read_missing_chunks is False
        assert array.config.write_empty_chunks is True
    np.testing.assert_array_equal(counts_t[:], QC_VALUES.T)


def test_a_missing_counts_shard_raises_and_names_the_chunk():
    store = _prepared_store()
    # Rows 2 and 3 live in the second counts shard.
    del store._store_dict["RNA/counts/c/1/0"]
    ds = open_qc_store(store)

    with pytest.raises(ChunkNotFoundError, match=r"(?s)RNA/counts'.*chunk 'c/1/0'"):
        ds.RNA.rawData.compute()
    # Rows of the shards that are present still read.
    np.testing.assert_array_equal(ds.RNA.rawData[[0, 1], :].compute(), QC_VALUES[:2])


def test_a_missing_counts_t_shard_raises_for_feature_reads():
    store = _prepared_store()
    # Features 2 and 3 live in the second countsT shard.
    del store._store_dict["RNA/countsT/c/1/0"]
    ds = open_qc_store(store)

    with pytest.raises(ChunkNotFoundError, match=r"(?s)RNA/countsT'.*chunk 'c/1/0'"):
        ds.RNA.rawDataT[:]
    # Feature streams read the feature-major copy.
    with pytest.raises(ChunkNotFoundError, match="chunk 'c/1/0'"):
        list(
            map_feature_read_groups(
                ds.RNA.rawDataT, lambda group: group.values, feat_idx=np.array([3])
            )
        )
    np.testing.assert_array_equal(
        np.concatenate(
            list(
                map_feature_read_groups(
                    ds.RNA.rawDataT, lambda group: group.values, feat_idx=[0, 1]
                )
            )
        ),
        QC_VALUES.T[:2],
    )


def test_count_arrays_write_chunks_of_zeros_and_read_strictly():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    features = ["a", "b", "c"]
    counts = create_zarr_count_assay(
        root, "RNA", None, 4, feat_ids=features, feat_names=features, dtype="uint32"
    )
    # A handle that CyteArc creates stores a chunk of zeros, so every chunk
    # that a strict read fetches exists.
    counts[:] = 0
    assert "RNA/counts/c/0/0" in root.store._store_dict
    finalize_test_counts(counts)
    for array in (counts, write_counts_t(counts, root["RNA"])):
        assert array.config.write_empty_chunks is True
        assert array.config.read_missing_chunks is False
    np.testing.assert_array_equal(
        validate_count_matrix(root["RNA"], require_transpose=True)[1][:],
        np.zeros((3, 4)),
    )
    # A matrix without cells has no chunks to read.
    empty = create_zarr_count_assay(
        root, "ADT", None, 0, feat_ids=features, feat_names=features, dtype="uint32"
    )
    finalize_test_counts(empty)
    counts_t = write_counts_t(empty, root["ADT"])
    assert counts_t.shape == (3, 0)
    assert counts_t.config.read_missing_chunks is False


def test_band_reads_that_fail_alike_raise_one_error():
    lost, again, other = OSError("lost"), OSError("lost"), ValueError("bad")

    assert _settled([1.0, 2.0]) == [1.0, 2.0]
    # Bands that read one lost shard report it once.
    with pytest.raises(OSError) as caught:
        _settled([1.0, lost, again])
    assert caught.value is lost
    with pytest.raises(ExceptionGroup) as group:
        _settled([lost, other, again])
    assert group.value.exceptions == (lost, other)


def test_repacking_a_store_with_a_lost_count_shard_fails_closed(tmp_path):
    source = tmp_path / "source.zarr"
    counts = np.arange(1, 49, dtype=np.uint32).reshape(8, 6)
    write_count_store(str(source), {"ADT": counts}, "uint32")
    open_qc_store(str(source), default_assay="ADT")
    shards = sorted((source / "ADT" / "counts" / "c").iterdir())
    assert shards
    shutil.rmtree(shards[0])

    with pytest.raises(ChunkNotFoundError, match="ADT/counts"):
        repack_store(
            str(source), str(tmp_path / "copy.zarr"), nthreads=1, data_only=True
        )
    assert not (tmp_path / "copy.zarr").exists()
