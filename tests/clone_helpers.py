"""Stores and checks shared by the tests of clones."""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.clone_truth import N_CELLS, CloneTruth, write_cospar
from tests.storage_helpers import write_count_store

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.refs import ArtifactRef


def open_clone_store(path: Path, **options: object) -> "DataStore":
    """Open a store of the truth's cells on one thread."""
    from cytearc.datastore.datastore import DataStore

    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        **{"nthreads": 1, **options},  # type: ignore[arg-type]
    )


def write_truth_store(path: Path, truth: CloneTruth) -> "DataStore":
    """Write a store whose cells are the truth's, with their sample, time and type."""
    counts = np.random.default_rng(5).poisson(3.0, size=(N_CELLS, 6)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=truth.cell_ids)
    store = open_clone_store(path)
    store.cells.insert("sample", truth.samples.astype(object))
    store.cells.insert("time_point", truth.time_points)
    store.cells.insert("cell_type", truth.cell_types.astype(object))
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    return store


def write_cells_store(path: Path, cell_ids: Sequence[str]) -> "DataStore":
    """Write a store of these cells, each with one count of three features.

    The ``every_cell`` column selects every cell.
    """
    counts = np.ones((len(cell_ids), 3), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=list(cell_ids))
    store = open_clone_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store


def copy_clone_store(template: Path, folder: Path) -> "DataStore":
    """Copy a template store into ``folder`` and open the copy."""
    target = folder / template.name
    shutil.copytree(template, target)
    return open_clone_store(target)


def every_cell(store: "DataStore") -> "ArtifactRef":
    return store.snapshot_cell_selection("every_cell")


def ids_read_bytes(store: "DataStore", n_cells: int) -> int:
    """The bytes of reading the IDs of ``n_cells`` cells of ``store`` as text."""
    from cytearc.metadata.rows import text_read_bytes

    return text_read_bytes(store.cells._get_array("ids"), n_cells)


def loaded_partition(store: "DataStore", clones: "ArtifactRef") -> set[frozenset[str]]:
    """The saved clones as sets of cell IDs, from the cells table of load_clones."""
    cells = store.clones.load_clones(clones).cells
    labelled = cells["clone"].dropna()
    return {
        frozenset(str(cell) for cell in members)
        for members in labelled.groupby(labelled, observed=True).groups.values()
    }


def import_truth_clones(
    store: "DataStore",
    truth: CloneTruth,
    folder: Path,
    *,
    cell_selection: "ArtifactRef | None" = None,
) -> "ArtifactRef":
    """Save the truth's clones by importing CoSpar's table of them.

    The table also lists barcodes outside the store, about which the import
    warns, as it does again when it reuses saved clones.
    """
    path = write_cospar(truth, folder / "truth_clones.csv")
    selection = every_cell(store) if cell_selection is None else cell_selection
    with pytest.warns(UserWarning, match="match no selected cell"):
        return store.clones.import_clones(path, cell_selection=selection)
