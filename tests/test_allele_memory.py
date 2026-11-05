"""A recorder import refuses a budget too small for its records before it writes anything."""

from pathlib import Path

import numpy as np
import pytest

from cytearc.utils.memory_budget import MemoryBudgetError
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.slow

_RNG = np.random.default_rng(19)


def _store(folder: Path, cell_ids: list[str]):
    folder.mkdir(parents=True, exist_ok=True)
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((len(cell_ids), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cell_ids,
    )
    store = open_store(folder / "store.zarr")
    return store, store.snapshot_cell_selection()


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """PEtracer's characters of 1,000 cells by 100 integrations: 300,000 records."""
    folder = tmp_path_factory.mktemp("wide")
    cells = [f"cell{index:06d}" for index in range(1_000)]
    header = ["cell_barcode"] + [
        f"intID{k}-{site}" for k in range(100) for site in ("EMX1", "HEK3", "RNF2")
    ]
    tokens = _RNG.choice(np.array(list("*-12345678!"), dtype=object), size=(1_000, 300))
    path = folder / "characters.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + "".join(
            cell + "," + ",".join(row) + "\n"
            for cell, row in zip(cells, tokens.tolist(), strict=True)
        )
    )
    store, cells_ref = _store(folder, cells)
    return store, cells_ref, path, folder


def _import(store, cells, path, **options):
    return store.clones.import_alleles(
        path, cell_selection=cells, invalidate_cache=True, **options
    )


def test_a_budget_too_small_refuses_before_anything_is_written(wide) -> None:
    store, cells, path, _folder = wide
    full = store.memoryBytes
    before = set(store.artifacts.list(kind="lineage_alleles", scope="datastore"))
    try:
        store.memoryBytes = 30_000_000
        with pytest.raises(
            MemoryBudgetError,
            match="Importing alleles for 1000 selected cells holds about",
        ):
            _import(store, cells, path)
    finally:
        store.memoryBytes = full
    assert (
        set(store.artifacts.list(kind="lineage_alleles", scope="datastore")) == before
    )
