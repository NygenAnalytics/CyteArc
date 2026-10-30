"""Joining the barcodes of imported files to the cells of a store."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.barcode_join_helpers import CASES, build_guide_store, import_case

_REFERENCE = Path(__file__).parent / "guide_calls_import_reference.json"
# Attributes that differ between two writes of one artifact.
_VOLATILE = {"artifact_id", "created_at_ns", "cytearc_version"}


def _stored(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.dtype.kind in "OUS":
        return {
            "dtype": "text",
            "shape": list(values.shape),
            "values": values.astype(str).tolist(),
        }
    flat = values.reshape(-1)
    if values.dtype.kind == "f":
        listed = [None if np.isnan(value) else float(value) for value in flat]
    else:
        listed = flat.tolist()
    return {"dtype": values.dtype.str, "shape": list(values.shape), "values": listed}


def test_imported_guide_calls_match_those_saved_before_the_shared_join(
    tmp_path,
) -> None:
    # The reference was written by import_guide_calls before it joined
    # barcodes through the shared helper; every array and attribute must
    # come out the same.
    reference = json.loads(_REFERENCE.read_text())
    store, selection = build_guide_store(tmp_path / "store.zarr")
    assert set(reference) == set(CASES)
    for name in CASES:
        # Each case's file names a barcode outside the selection.
        with pytest.warns(UserWarning, match="match no selected cell"):
            ref = import_case(store, selection, name, tmp_path)
        group = store.artifacts.load(ref)
        attributes = {
            key: group.attrs[key] for key in group.attrs if key not in _VOLATILE
        }
        attributes["provenance"]["inputs"]["cell_selection"].pop("artifact_id")
        expected = reference[name]
        assert store.artifacts.inspect(ref).parameters == expected["parameters"]
        assert attributes == expected["attributes"]
        assert {key: _stored(group[key][:]) for key in group.array_keys()} == expected[
            "arrays"
        ]
