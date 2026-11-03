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
    clone_write_bytes,
    plan_clones,
    write_clones,
)
from cytearc.metadata.arguments import CloneColumnArguments
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, write_cells_store, write_truth_store
from tests.clone_truth import N_CELLS, clone_truth, number_label
from tests.memory_helpers import cost_bound

TRUTH = clone_truth()
# A memory budget that every write of these tests fits, and nothing held.
_ROOMY = {"memory_bytes": 1 << 34, "held_bytes": 0}


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
    kinds = {"n_barcodes": "i", "doublet": "b", "ambiguous": "b", "n_alleles": "i"}
    planned = _plan(store, arguments, source_arrays=kinds)
    assert not planned.reused
    n_barcodes = (np.arange(N_CELLS) % 4).astype(np.int32)
    doublet = np.arange(N_CELLS) % 5 == 0
    ambiguous = np.arange(N_CELLS) % 7 == 1
    n_alleles = (np.arange(N_CELLS) % 3).astype(np.int32)
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
            "n_alleles": n_alleles,
        },
        **_ROOMY,
    )
    cells = store.clones.load_clones(planned.ref).cells.loc[TRUTH.cell_ids]
    np.testing.assert_array_equal(cells["n_barcodes"], n_barcodes)
    np.testing.assert_array_equal(cells["doublet"], doublet)
    np.testing.assert_array_equal(cells["ambiguous"], ambiguous)
    np.testing.assert_array_equal(cells["n_alleles"], n_alleles)
    assert cells["n_barcodes"].dtype == SOURCE_ARRAYS["n_barcodes"]
    assert cells["doublet"].dtype == SOURCE_ARRAYS["doublet"]
    assert cells["ambiguous"].dtype == SOURCE_ARRAYS["ambiguous"]
    assert cells["n_alleles"].dtype == np.int64
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
    write_clones(
        store.zw, planned, assignment, {"n_clones": len(assignment.clone_ids)}, **_ROOMY
    )
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
            **_ROOMY,
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
            **_ROOMY,
        )


def _labelled(width: int, *, probabilities: bool = False) -> CloneAssignment:
    """The truth's clones, each labelled by ``width`` characters."""
    codes, clone_ids, clone_sizes = ranked_clones(
        TRUTH.clones,
        [f"{'c' * (width - 4)}{clone:04d}" for clone in range(TRUTH.n_clones)],
    )
    return CloneAssignment(
        codes=codes,
        clone_ids=clone_ids,
        clone_sizes=clone_sizes,
        probabilities=np.linspace(0.5, 1.0, N_CELLS) if probabilities else None,
        n_cells_in_several_clones=0,
    )


def test_the_save_check_counts_the_labels_width_and_a_buffer_for_each_write(
    store,
) -> None:
    # Zarr writes each chunk and each array's metadata through a buffered
    # file, as many at once as its async concurrency allows, and Python 3.14
    # made each buffer 128 KiB, where it was the file system's block. The
    # labels of these 48 cells are stored at their width, twice as the chunk
    # and Zarr's encoded copy, four bytes a character.
    planned = _plan(store, _arguments(store))

    def bytes_named(width: int, concurrency: int = 10) -> int:
        assignment = _labelled(width)
        with zarr.config.set({"async.concurrency": concurrency}):
            with pytest.raises(
                MemoryError, match=f"width of the longest, {width} characters"
            ) as refused:
                write_clones(
                    store.zw,
                    planned,
                    assignment,
                    {"n_clones": len(assignment.clone_ids)},
                    memory_bytes=1,
                    held_bytes=1_000,
                )
        found = re.search(r"holds about (\d+) bytes", str(refused.value))
        assert found is not None
        return int(found.group(1))

    assert bytes_named(40, 20) - bytes_named(40, 10) == 10 * io.DEFAULT_BUFFER_SIZE
    assert bytes_named(40) - bytes_named(20) == 2 * 4 * 20 * N_CELLS
    assert bytes_named(40) == 1_000 + clone_write_bytes(N_CELLS, TRUTH.n_clones, 40)


def test_the_save_check_counts_the_probabilities(store) -> None:
    # Each cell's probability is held as given and as copied to float64.
    planned = _plan(store, _arguments(store))
    needed = {}
    for probabilities in (False, True):
        assignment = _labelled(12, probabilities=probabilities)
        with pytest.raises(MemoryError) as refused:
            write_clones(
                store.zw,
                planned,
                assignment,
                {"n_clones": len(assignment.clone_ids)},
                memory_bytes=1,
                held_bytes=0,
            )
        found = re.search(r"holds about (\d+) bytes", str(refused.value))
        assert found is not None
        needed[probabilities] = int(found.group(1))
    assert needed[True] - needed[False] == 2 * 8 * N_CELLS


def test_a_save_one_byte_over_its_budget_is_refused_before_it_writes(store) -> None:
    arguments = _arguments(store)
    assignment = _assignment()
    width = max(len(label) for label in assignment.clone_ids.tolist())
    needed = 5_000 + clone_write_bytes(N_CELLS, len(assignment.clone_ids), width)
    planned = _plan(store, arguments)
    with pytest.raises(MemoryError, match=f"holds about {needed} bytes"):
        write_clones(
            store.zw,
            planned,
            assignment,
            {"n_clones": len(assignment.clone_ids)},
            memory_bytes=needed - 1,
            held_bytes=5_000,
        )
    assert not store.artifacts.inspect(planned.ref).exists
    write_clones(
        store.zw,
        planned,
        assignment,
        {"n_clones": len(assignment.clone_ids)},
        memory_bytes=needed,
        held_bytes=5_000,
    )
    assert store.clones.load_clones(planned.ref).clones["n_cells"].sum() == int(
        np.count_nonzero(assignment.codes >= 0)
    )


@pytest.mark.parametrize("probabilities", [False, True])
def test_saving_clones_holds_what_their_cost_states(tmp_path, probabilities) -> None:
    # 60,000 cells in 6,000 clones of 40-character labels, a fifth of the
    # cells without one: the labels as stored outweigh the other arrays.
    n_cells, n_clones = 60_000, 6_000
    store = write_cells_store(
        tmp_path / "store.zarr", [f"cell{index:05d}" for index in range(n_cells)]
    )
    rng = np.random.default_rng(3)
    codes = rng.integers(0, n_clones, n_cells)
    codes[rng.random(n_cells) < 0.2] = -1
    ranked, clone_ids, clone_sizes = ranked_clones(
        codes, [f"{'lineage' * 5}{index:05d}" for index in range(n_clones)]
    )
    assignment = CloneAssignment(
        codes=ranked,
        clone_ids=clone_ids,
        clone_sizes=clone_sizes,
        probabilities=rng.random(n_cells) if probabilities else None,
        n_cells_in_several_clones=0,
    )
    cells = every_cell(store)
    stated = clone_write_bytes(
        n_cells, len(clone_ids), 40, with_probability=probabilities
    )
    attempts = iter(range(10))

    def save() -> None:
        arguments = CloneColumnArguments(
            cell_selection=cells,
            labels_fingerprint=f"{next(attempts):064d}",
            column="lineage",
            invalidate_cache=False,
        )
        planned = plan_clones(
            store.zw,
            arguments,
            cells,
            n_cells,
            with_probability=probabilities,
            required_attributes=("n_clones",),
        )
        write_clones(
            store.zw, planned, assignment, {"n_clones": len(clone_ids)}, **_ROOMY
        )

    save()
    # The assignment is the caller's before the save, and the cost counts it.
    given = ranked.nbytes + clone_ids.nbytes + clone_sizes.nbytes
    if probabilities:
        given += assignment.probabilities.nbytes
    _result, peak = cost_bound(stated - given, save)
    assert peak >= 2 * 4 * 40 * n_cells


def test_saving_clones_of_one_character_holds_what_gathering_their_sizes_holds(
    tmp_path,
) -> None:
    # 200,000 cells in nine clones labelled 1 to 9: gathering each cell's
    # clone size, beside a mask of the cells with a clone and two int64
    # arrays, holds more than writing the labels, which hold four bytes each.
    n_cells = 200_000
    store = write_cells_store(
        tmp_path / "store.zarr", [f"cell{index:06d}" for index in range(n_cells)]
    )
    codes = np.arange(n_cells) % 9
    ranked, clone_ids, clone_sizes = ranked_clones(
        codes, [str(clone + 1) for clone in range(9)]
    )
    assignment = CloneAssignment(
        codes=ranked,
        clone_ids=clone_ids,
        clone_sizes=clone_sizes,
        probabilities=None,
        n_cells_in_several_clones=0,
    )
    cells = every_cell(store)
    attempts = iter(range(10))

    def save() -> None:
        arguments = CloneColumnArguments(
            cell_selection=cells,
            labels_fingerprint=f"{next(attempts):064d}",
            column="lineage",
            invalidate_cache=False,
        )
        planned = plan_clones(
            store.zw,
            arguments,
            cells,
            n_cells,
            with_probability=False,
            required_attributes=("n_clones",),
        )
        write_clones(store.zw, planned, assignment, {"n_clones": 9}, **_ROOMY)

    save()
    given = ranked.nbytes + clone_ids.nbytes + clone_sizes.nbytes
    stated = clone_write_bytes(n_cells, 9, 1) - given
    _result, peak = cost_bound(stated, save)
    # The gathering holds 37 bytes per cell at its peak, where writing holds
    # 20 per cell and its chunks.
    assert peak > 30 * n_cells, peak
