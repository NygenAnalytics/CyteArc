"""The ``clones`` artifact as every clone source plans, writes and loads it."""

import dataclasses
import io
import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from cytearc.clones.assignments import CloneAssignment, ranked_clones
from cytearc.datastore._operations.clone_artifacts import (
    SOURCE_ARRAYS,
    check_clone_write_budget,
    plan_clones,
    write_clones,
)
from cytearc.metadata.arguments import CloneColumnArguments
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, write_truth_store
from tests.clone_truth import N_CELLS, clone_truth, number_label

TRUTH = clone_truth()


def _assignment() -> CloneAssignment:
    codes, clone_ids, clone_sizes = ranked_clones(
        TRUTH.clones, [number_label(clone) for clone in range(TRUTH.n_clones)]
    )
    return CloneAssignment(
        codes=codes,
        clone_ids=clone_ids,
        clone_sizes=clone_sizes,
        probabilities=None,
        n_cells_in_several_clones=0,
    )


def _plan(store, arguments, **options):
    return plan_clones(
        store.zw,
        arguments,
        arguments.cell_selection,
        N_CELLS,
        with_probability=False,
        required_attributes=("n_clones",),
        **options,
    )


@pytest.fixture
def store(tmp_path: Path):
    return write_truth_store(tmp_path / "store.zarr", TRUTH)


def _arguments(store) -> CloneColumnArguments:
    return CloneColumnArguments(
        cell_selection=every_cell(store),
        labels_fingerprint="0" * 64,
        column="sample",
        invalidate_cache=False,
    )


def test_the_arrays_a_source_adds_are_saved_required_and_loaded(store) -> None:
    arguments = _arguments(store)
    kinds = {"n_barcodes": "i", "doublet": "b", "ambiguous": "b"}
    planned = _plan(store, arguments, source_arrays=kinds)
    assert not planned.reused
    n_barcodes = (np.arange(N_CELLS) % 4).astype(np.int32)
    doublet = np.arange(N_CELLS) % 5 == 0
    ambiguous = np.arange(N_CELLS) % 7 == 1
    assignment = _assignment()
    write_clones(
        store.zw,
        planned,
        assignment,
        {"n_clones": len(assignment.clone_ids)},
        source_arrays={
            "n_barcodes": n_barcodes,
            "doublet": doublet,
            "ambiguous": ambiguous,
        },
    )
    cells = store.clones.load_clones(planned.ref).cells.loc[TRUTH.cell_ids]
    np.testing.assert_array_equal(cells["n_barcodes"], n_barcodes)
    np.testing.assert_array_equal(cells["doublet"], doublet)
    np.testing.assert_array_equal(cells["ambiguous"], ambiguous)
    assert cells["n_barcodes"].dtype == SOURCE_ARRAYS["n_barcodes"]
    assert cells["doublet"].dtype == SOURCE_ARRAYS["doublet"]
    assert cells["ambiguous"].dtype == SOURCE_ARRAYS["ambiguous"]
    assert "clone_probability" not in cells
    # Saved clones are reused with the arrays their source adds, and not once
    # one of them is lost.
    again = _plan(store, arguments, source_arrays=kinds)
    assert again.reused and again.ref == planned.ref
    del artifact_group(store.zw, planned.ref)["doublet"]
    assert not _plan(store, arguments, source_arrays=kinds).reused


def test_planning_invalidates_saved_clones_as_their_arguments_say(store) -> None:
    arguments = _arguments(store)
    planned = _plan(store, arguments)
    assignment = _assignment()
    write_clones(store.zw, planned, assignment, {"n_clones": len(assignment.clone_ids)})
    assert _plan(store, arguments).reused
    again = _plan(store, dataclasses.replace(arguments, invalidate_cache=True))
    assert not again.reused


def test_writing_refuses_arrays_that_no_source_adds(store) -> None:
    planned = _plan(store, _arguments(store))
    assignment = _assignment()
    with pytest.raises(ValueError, match="'barcodes' is not a per-cell array"):
        write_clones(
            store.zw,
            planned,
            assignment,
            {"n_clones": len(assignment.clone_ids)},
            source_arrays={"barcodes": np.zeros(N_CELLS, dtype=np.int64)},
        )
    probable = dataclasses.replace(assignment, probabilities=np.ones(N_CELLS))
    with pytest.raises(
        ValueError, match="'clone_probability' is written from the assignment's"
    ):
        write_clones(
            store.zw,
            planned,
            probable,
            {"n_clones": len(assignment.clone_ids)},
            source_arrays={"clone_probability": np.ones(N_CELLS)},
        )


def test_the_write_check_counts_a_buffer_for_each_write_zarr_runs_at_once() -> None:
    # Zarr writes each chunk and each array's metadata through a buffered
    # file, as many at once as its async concurrency allows, and Python 3.14
    # made each buffer 128 KiB, where it was the file system's block. Three
    # cells of labels five characters long hold 3 * (8 + 4 * 5) bytes.
    labels = np.asarray(["clone", "clone", "other"], dtype=object)

    def bytes_named(concurrency: int) -> int:
        with zarr.config.set({"async.concurrency": concurrency}):
            with pytest.raises(MemoryError, match="width of the longest, 5") as refused:
                check_clone_write_budget(1, 1_000, labels, 3)
        found = re.search(r"holds about (\d+) bytes", str(refused.value))
        assert found is not None
        return int(found.group(1))

    assert bytes_named(1) == 1_000 + 3 * (8 + 4 * 5) + io.DEFAULT_BUFFER_SIZE
    assert bytes_named(20) - bytes_named(10) == 10 * io.DEFAULT_BUFFER_SIZE
