"""Steps that the clone operations share, each charged to the operation's memory ledger.

Clone import, the clone summaries, lineage barcodes, recorder alleles,
phylogenies and receptors read the same kinds of stored arrays and name their
files and clones alike: integer codes read whole as int64, the IDs of a
cell-aligned artifact's cells, the labels of clones ranked by size, the
words for the files and the selected cells of an import, and why clones
called from lineage barcodes give no cell a clone. Operation modules may not
import each other, so each step lives here once.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

from ...clones.assignments import ranked_clone_label_bytes, ranked_clone_labels
from ...metadata.arguments import ClonotypeArguments
from ...metadata.rows import read_text_rows, read_whole
from ...metadata.selection import (
    CellSelectionRead,
    plan_cell_selection_read,
    read_cell_selection,
)
from ...storage.artifacts import ArtifactRef, inspect_artifact
from ...utils.memory_budget import MemoryLedger

__all__ = [
    "CODES_WITHIN_TYPE",
    "REPRESENTATIVES_WITHIN_TYPE",
    "artifact_cell_ids",
    "barcode_import_cause",
    "exported_cell_positions",
    "files_text",
    "hold_ranked_clone_labels",
    "holds_no_barcode",
    "measured_within_types",
    "no_barcode_cause",
    "plan_artifact_cells",
    "read_codes",
    "selected_cells_named",
    "selected_cells_text",
]

# Bytes of one int64 row position.
_INDEX_BYTES = 8

#: The array in which clonotypes defined across receptor types save each
#: cell's clonotype within its receptor type, as ``within_receptor_type=True``
#: codes it, -1 for none, which every clone summary measures.
CODES_WITHIN_TYPE = "codes_within_type"
#: The array in which clonotypes defined across receptor types save the
#: representative cell of each clonotype within a receptor type, by its code,
#: as ``within_receptor_type=True`` chooses it, which names it in figures.
REPRESENTATIVES_WITHIN_TYPE = "representatives_within_type"


def files_text(noun: str, n_files: int) -> str:
    """The words for the files of an import in its messages, such as "the clone files"."""
    return f"the {noun} file" if n_files == 1 else f"the {noun} files"


def selected_cells_named(diagnostics: Mapping[str, Any]) -> int:
    """The selected cells that an import's files name, from the join's saved counts: their barcodes less those of no cell of the store and those of cells outside the selection."""
    return (
        int(diagnostics["fileBarcodes"])
        - int(diagnostics["unmatchedBarcodes"])
        - int(diagnostics["unselectedBarcodes"])
    )


def selected_cells_text(n_cells: int) -> tuple[str, str]:
    """The words of a message for ``n_cells`` selected cells, and the pronoun that refers to them, such as ``("1 selected cell", "it")``."""
    if n_cells == 1:
        return "1 selected cell", "it"
    return f"{n_cells} selected cells", "them"


def holds_no_barcode(diagnostics: Mapping[str, Any]) -> bool:
    """Whether a barcode import's rows name selected cells but none of them holds a barcode.

    A row without a barcode and a pair counted 0 hold none, and ``emptyRows``
    counts the rows of selected cells that hold none, among the files' rows
    less ``unmatchedRows``, those of no selected cell. An import of an
    assay's counts holds a barcode in each of its rows.
    """
    selected = int(diagnostics["fileRows"]) - int(diagnostics["unmatchedRows"])
    return selected > 0 and int(diagnostics["emptyRows"]) == selected


def no_barcode_cause(n_files: int, n_cells: int) -> str:
    """How files name selected cells but hold no barcode for them."""
    cells, them = selected_cells_text(n_cells)
    if n_files == 1:
        return f"names {cells} but holds no barcode for {them}"
    return f"name {cells} but hold no barcode for {them}"


def barcode_import_cause(root: zarr.Group, barcodes: ArtifactRef) -> str | None:
    """Why clones called from ``barcodes`` give no cell a clone, in a refusal's words, when the barcodes' files hold no barcode of a selected cell; otherwise None.

    Files that hold no row, or whose rows of selected cells hold no barcode,
    leave every selected cell without a barcode, so every clone called from
    them is a clone of no cell, whatever the rules. The words name the
    remedy: other files, or a selection of the cells whose barcodes the
    files list, and clones called from the new barcodes. An import of an
    assay's counts reads no file, and barcodes that are not complete are not
    read, so neither gives a cause.
    """
    status = inspect_artifact(root, barcodes)
    if not status.complete:
        return None
    sources = (status.inputs or {})["sources"]
    if "assay" in sources[0]:
        return None
    diagnostics = status.diagnostics
    one = len(sources) == 1
    files = files_text("barcode", len(sources))
    remedy = (
        "import a file that lists the cells' lineage barcodes"
        if one
        else "import files that list the cells' lineage barcodes"
    )
    if int(diagnostics["fileRows"]) == 0:
        return (
            f"their barcodes' import read no row from {files}; {remedy} and call "
            "the clones from the new barcodes"
        )
    if not holds_no_barcode(diagnostics):
        return None
    if int(diagnostics["unselectedBarcodes"]):
        remedy += (
            f", or import {'it' if one else 'them'} on a cell_selection of the cells "
            f"whose barcodes {'it lists' if one else 'they list'},"
        )
    cause = no_barcode_cause(len(sources), selected_cells_named(diagnostics))
    return (
        f"{files} of their barcodes {cause}; {remedy} and call the clones from the "
        "new barcodes"
    )


def measured_within_types(clones: ArtifactRef, status: Any, group: zarr.Group) -> bool:
    """Whether the clone summaries measure a clone artifact by each cell's clonotype within its receptor type.

    Clonotypes defined across receptor types are measured so, by the codes
    that they save in ``CODES_WITHIN_TYPE``, so their summaries equal those
    of clonotypes defined within types.

    Args:
        clones: The clone artifact.
        status: Its record, as ``inspect_artifact`` gives it.
        group: Its group.

    Raises:
        ValueError: If clonotypes defined across receptor types were saved
            without each cell's clonotype within its type.
    """
    if clones.kind != ClonotypeArguments.artifact_kind or (status.parameters or {}).get(
        "within_receptor_type", True
    ):
        return False
    if CODES_WITHIN_TYPE not in group:
        raise ValueError(
            "clonotypes was defined across receptor types by an earlier version "
            "of CyteArc, without each cell's clonotype within its receptor type, "
            "which the clone summaries measure; define the clonotypes again with "
            "ds.repertoire.clonotypes, which saves it"
        )
    return True


def read_codes(
    ledger: MemoryLedger, group: zarr.Group, name: str, *, item: str
) -> np.ndarray:
    """Read the whole integer array ``name`` of ``group`` as int64, keeping it charged as ``item``.

    The array is read through :func:`~cytearc.metadata.rows.read_whole`; an
    array stored narrower is widened into an int64 array charged before it
    is made, and the stored values are let go.
    """
    stored = read_whole(ledger, group, name, item=item)
    if stored.dtype == np.int64:
        return stored
    if stored.dtype.kind not in "iub":
        raise TypeError(f"{item} must be stored as integers, not as {stored.dtype}")
    ledger.hold(item, 8 * stored.size)
    codes = stored.astype(np.int64)
    stored_bytes = stored.nbytes
    del stored
    ledger.release(stored_bytes)
    return codes


def hold_ranked_clone_labels(ledger: MemoryLedger, n_clones: int) -> np.ndarray:
    """The labels ``clone_1`` to ``clone_{n_clones}``, charged to ``ledger`` before they are made."""
    ledger.hold("the clones' labels", ranked_clone_label_bytes(n_clones))
    return ranked_clone_labels(n_clones)


def plan_artifact_cells(root: Any, artifact: ArtifactRef) -> CellSelectionRead:
    """Check a cell-aligned artifact's cell selection, for :func:`artifact_cell_ids`, before anything is charged.

    The artifact's record and the selection's identity check are read
    outside the budget, once in the operation's scope.
    """
    selection = inspect_artifact(root, artifact).input_ref("cell_selection")
    return plan_cell_selection_read(root, selection)


def exported_cell_positions(
    ledger: MemoryLedger, cells: Any, rows: np.ndarray, cell_order: object
) -> np.ndarray | None:
    """The order in which an export writes the cells at store rows ``rows``, kept charged while it is held.

    It is :func:`~cytearc.storage.cell_order.exported_cell_order` of the
    store's cell table ``cells``: None for store order, else positions in
    ``rows``. Finding it reads each cell's ``sourceRow`` and sorts them.

    Raises:
        TypeError: If ``cell_order`` is not a string.
        ValueError: If ``cell_order`` is neither ``"source"`` nor ``"stored"``.
        MemoryBudgetError: If the order would pass the ledger's budget.
    """
    from ...storage.cell_order import cell_order_argument, exported_cell_order

    cell_order = cell_order_argument(cell_order)
    item = "the cells' order in the imported file"
    if cell_order == "source":
        # Each cell's source row, its position in the sort, and the order.
        ledger.peak(item, 3 * _INDEX_BYTES * int(rows.size))
    order = exported_cell_order(cells._group, rows, cell_order)
    if order is not None:
        ledger.hold(item, order.nbytes)
    return order


def artifact_cell_ids(
    ledger: MemoryLedger,
    root: Any,
    cell_ids: Any,
    planned: CellSelectionRead,
    *,
    item: str = "the cells' IDs",
) -> np.ndarray:
    """The IDs of a cell-aligned artifact's cells, in the order of its arrays, kept charged as ``item``.

    The artifact's cell selection, as :func:`plan_artifact_cells` planned
    it, is read through
    :func:`~cytearc.metadata.selection.read_cell_selection` and the IDs of
    its cells from ``cell_ids``, the store's stored IDs, through
    :func:`~cytearc.metadata.rows.read_text_rows`, as Python strings.
    """
    cell_index = read_cell_selection(
        ledger, root, planned, item="the positions of the artifact's cells"
    )
    ids = read_text_rows(ledger, cell_ids, cell_index, item=item)
    index_bytes = cell_index.nbytes
    del cell_index
    ledger.release(index_bytes)
    return ids
