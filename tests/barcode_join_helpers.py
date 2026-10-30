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


# Cells of the join store: two samples merged with "<sample>__" prefixes and
# 10x's "-1" suffixes, so file barcodes need a prefix, a suffix or a pattern.
JOIN_CELLS = ("s1__AAAC-1", "s1__CCCG-1", "s1__TTTA-1", "s2__AAAC-1", "s2__GGGT-1")


def build_join_store(path: Path) -> tuple["DataStore", "ArtifactRef"]:
    """Write a store of ``JOIN_CELLS`` with RNA and guide counts, and select them."""
    from tests.perturbation_helpers import open_store
    from tests.storage_helpers import write_count_store

    n_cells = len(JOIN_CELLS)
    rng = np.random.default_rng(11)
    write_count_store(
        str(path),
        {
            "RNA": rng.poisson(2.0, size=(n_cells, 6)),
            "CRISPR": rng.poisson(3.0, size=(n_cells, 2)),
        },
        "uint16",
        cell_ids=JOIN_CELLS,
    )
    store = open_store(path)
    store.cells.insert("joined", np.ones(n_cells, dtype=bool))
    return store, store.snapshot_cell_selection("joined")


def write_guide_call_file(path: Path, barcodes: list[str]) -> Path:
    """A Cell Ranger call file that calls CRISPR0 in each barcode."""
    path.write_text(
        HEADER + "".join(f"{barcode},1,CRISPR0,5\n" for barcode in barcodes)
    )
    return path


def write_clone_file(path: Path, barcodes: list[str]) -> Path:
    """A TREX clones.txt that puts every barcode in clone 1."""
    path.write_text(
        "clone_nr\tcell_id\n" + "".join(f"1\t{barcode}\n" for barcode in barcodes)
    )
    return path
