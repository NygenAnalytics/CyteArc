"""The ``clones`` artifact as every clone source plans, writes and loads it."""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from cytearc.clones.assignments import CloneAssignment, ranked_clones
from cytearc.datastore._operations.clone_artifacts import (
    SOURCE_ARRAYS,
    clone_write_bytes,
    plan_clones,
    write_clones,
)
from cytearc.metadata.arguments import CloneColumnArguments
from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, write_truth_store
from tests.clone_truth import N_CELLS, clone_truth, number_label
from cytearc.utils.memory_budget import MemoryLedger

TRUTH = clone_truth()
# A ledger without a budget, which every write of these tests fits.
_ROOMY = {"ledger": MemoryLedger(None, "Saving clones")}


def _holding(budget: int, held: int) -> MemoryLedger:
    """A ledger of ``budget`` bytes that holds ``held`` of them."""
    ledger = MemoryLedger(budget, "Saving clones")
    ledger.hold("what the caller holds", held)
    return ledger


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
                ledger=_holding(1, 0),
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
            ledger=_holding(needed - 1, 5_000),
        )
    assert not store.artifacts.inspect(planned.ref).exists
    write_clones(
        store.zw,
        planned,
        assignment,
        {"n_clones": len(assignment.clone_ids)},
        ledger=_holding(needed, 5_000),
    )
    assert store.clones.load_clones(planned.ref).clones["n_cells"].sum() == int(
        np.count_nonzero(assignment.codes >= 0)
    )
