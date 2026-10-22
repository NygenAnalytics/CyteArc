"""Stores whose metadata a tool consolidated open with their current contents."""

import ast
from pathlib import Path

import numpy as np
import pytest
import zarr

import cytearc
from cytearc import DataStore
from cytearc.storage.types import read_fresh_group
from tests.storage_helpers import write_count_store

_COUNTS = np.array(
    [[1, 0, 2, 0], [0, 3, 1, 0], [4, 0, 0, 1], [1, 1, 1, 1], [0, 2, 0, 3]],
    dtype=np.uint32,
)


def _open(path: Path, **options) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1, **options
    )


@pytest.mark.filterwarnings("ignore:Consolidated metadata:UserWarning")
def test_writable_operations_and_listings_ignore_consolidated_metadata(tmp_path):
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": _COUNTS}, "uint32")
    ds = _open(path)
    first = ds.snapshot_cell_selection("I")
    # Common advice for object stores; the consolidated copy goes stale as
    # soon as anything is written afterwards.
    zarr.consolidate_metadata(str(path))

    ds = _open(path)
    ds.cells.insert("early", np.array([True, True, False, True, False]))
    early = ds.snapshot_cell_selection("early")
    summary = ds.RNA.attrs.asdict()
    universe = ds.features.universe(from_assay="RNA")

    reader = _open(path, zarr_mode="r")
    selections = reader.artifacts.list(scope="datastore", kind="cell_selection")
    assert set(selections) == {first, early}
    assert reader.artifacts.list(kind="feature_selection", complete_only=True) == [
        universe
    ]
    np.testing.assert_array_equal(
        reader.artifacts.load(early)["values"][:], [True, True, False, True, False]
    )
    # The datastore's own groups are read from their records too.
    assert read_fresh_group(reader.zw, "RNA").attrs.asdict() == summary
    assert "early" in reader.cells.columns


def test_every_zarr_group_open_skips_consolidated_metadata():
    package = Path(cytearc.__file__).parent
    unguarded = []
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open_group"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "zarr"
            ):
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                value = keywords.get("use_consolidated")
                if not (isinstance(value, ast.Constant) and value.value is False):
                    unguarded.append(f"{source.relative_to(package)}:{node.lineno}")
    assert unguarded == []
