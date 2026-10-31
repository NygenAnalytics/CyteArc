"""Recorder alleles of DNA Typewriter's mouse, downloaded at test time.

The 1-in-40 subsample of DNA Typewriter's single-cell consensus table
(shendurelab/dtt-mouse, GPL-3.0, 0.95 MB) is never checked in: the test
downloads it, checks the bytes against the repository's blob, and skips when
GitHub is unreachable or serves the file short. Its expectations are counted
from the raw file by the format's documented rules.
"""

import gzip
import hashlib
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.clone_helpers import open_clone_store
from tests.storage_helpers import write_count_store

_COMMIT = "a491be02509d5ae862181a076f9e9006600dc096"
_SUBSAMPLE = (
    f"https://raw.githubusercontent.com/shendurelab/dtt-mouse/{_COMMIT}/"
    "sc_tape/example_data/e3v8_subsample.cell_consensus.tsv.gz"
)
# The file's git blob at that commit.
_BLOB = "081bcbadf3997edd1402faaf3fbe5b1199ff0313"
_DOWNLOAD_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
)


def _download(folder: Path) -> Path:
    try:
        with urllib.request.urlopen(_SUBSAMPLE, timeout=60) as response:
            data = response.read()
    except _DOWNLOAD_ERRORS as error:
        pytest.skip(f"GitHub could not serve DNA Typewriter's subsample whole: {error}")
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324
    if blob != _BLOB:
        pytest.skip("GitHub served DNA Typewriter's subsample short or changed")
    path = folder / "e3v8_subsample.cell_consensus.tsv.gz"
    path.write_bytes(data)
    return path


@pytest.mark.integration
@pytest.mark.slow
def test_dna_typewriters_mouse_chains_read_as_their_slots(tmp_path) -> None:
    path = _download(tmp_path)
    with gzip.open(path, "rt") as handle:
        raw = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False)
    integrations = [
        name for name in raw.columns[1:] if not name.startswith(("n_", "mean_"))
    ]
    assert len(raw) == 38_732 and len(integrations) == 11
    expected = {"mark": 0, "unedited": 0, "missing": 0}
    for value in raw[integrations].to_numpy().reshape(-1):
        if value == "":
            expected["missing"] += 6
        elif value == "-":
            expected["unedited"] += 6
        else:
            edits = len(value.split("-"))
            expected["mark"] += edits
            expected["unedited"] += 6 - edits
    cells = raw["cell"].tolist()
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_clone_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection()
    )
    loaded = store.clones.load_alleles(alleles)
    states = loaded.records["state"].value_counts()
    assert {state: int(states[state]) for state in expected} == expected
    assert int(states["not_applicable"]) == int(states["unknown_mark"]) == 0
    assert sorted(set(loaded.records["integration"])) == sorted(integrations)
    # Each cell's edits are the symbols of its chains.
    edits = (
        raw[integrations]
        .apply(
            lambda column: column.map(
                lambda value: 0 if value in ("", "-") else len(value.split("-"))
            )
        )
        .sum(axis=1)
    )
    np.testing.assert_array_equal(loaded.cells["n_edits"].to_numpy(), edits.to_numpy())
