"""A guide screen and call files that pin what importing guide calls saves.

``tests/guide_calls_import_reference.json`` holds the arrays and attributes of
the imports below, written before ``import_guide_calls`` joined barcodes to
cells through the shared join helper, so the tests can check that the helper
changed nothing they save.
"""

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.perturbation_helpers import write_screen

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.refs import ArtifactRef

# Guides are named CRISPR0 to CRISPR3 and cells cell0 to cell7.
GUIDE_COUNTS = np.asarray(
    [
        [30, 0, 1, 0],
        [0, 25, 0, 2],
        [20, 20, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 2, 0],
        [0, 0, 40, 3],
        [5, 5, 5, 5],
        [0, 9, 0, 0],
    ],
    dtype=np.uint16,
)
HEADER = "cell_barcode,num_features,feature_call,num_umis\n"
CALL_FILE = (
    HEADER + "cell0,1,CRISPR0,30\n"
    "cell1,1,CRISPR1,25\n"
    "cell2,2,CRISPR1|CRISPR0,20|20\n"
    "cell5,2,CRISPR3|CRISPR2,3|40\n"
    "cell6,1,CRISPR3,5\n"
    "cell7,1,CRISPR1,9\n"
    "outside-1,1,CRISPR0,4\n"
)
# Each case: the call file's text, whether it is gzip-compressed, and the
# barcode prefix.
CASES: dict[str, tuple[str, bool, str | None]] = {
    "plain": (CALL_FILE, False, None),
    "prefix": (
        HEADER + "0,1,CRISPR0,30\n5,2,CRISPR2|CRISPR3,40|3\n9,1,CRISPR1,7\n",
        False,
        "cell",
    ),
    "gzip": (CALL_FILE.replace("outside-1,1,CRISPR0,4\n", ""), True, None),
}


def build_guide_store(path: Path) -> tuple["DataStore", "ArtifactRef"]:
    """Write the eight-cell screen and select every cell but cell6."""
    rna = np.random.default_rng(7).poisson(2.0, size=(len(GUIDE_COUNTS), 10))
    store = write_screen(path, rna, GUIDE_COUNTS)
    keep = np.ones(len(GUIDE_COUNTS), dtype=bool)
    keep[6] = False
    store.cells.insert("screened", keep)
    return store, store.snapshot_cell_selection("screened")


def write_case(name: str, folder: Path) -> Path:
    """Write one case's call file into ``folder``."""
    text, compressed, _prefix = CASES[name]
    if compressed:
        path = folder / f"{name}.csv.gz"
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path = folder / f"{name}.csv"
        path.write_text(text)
    return path


def import_case(
    store: "DataStore", selection: "ArtifactRef", name: str, folder: Path
) -> "ArtifactRef":
    """Import one case's call file with the case's prefix."""
    prefix = CASES[name][2]
    return store.perturbation.import_guide_calls(
        write_case(name, folder), cell_selection=selection, barcode_prefix=prefix
    )
