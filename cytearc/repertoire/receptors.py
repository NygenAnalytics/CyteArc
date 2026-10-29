"""The record of a set of cells' immune receptor chains.

An ``immune_receptors`` artifact holds every imported contig of the selected
cells and, for each cell, its ranked chains and its receptor type, subtype and
chain pairing (:mod:`cytearc.repertoire.chains`). ``ImmuneReceptors`` presents
them as two tables: one row per cell, with the main fields of each of its four
chain slots side by side, and one row per contig.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .chains import SLOTS
from .contigs import FIELDS, FLAG, INTEGER, RAGGED, TEXT

__all__ = [
    "SLOT_FIELDS",
    "ImmuneReceptors",
    "contig_frame",
    "receptor_cell_frame",
]

#: The fields of each chain slot in the table of cells, in column order:
#: text and counts; ``d_call`` belongs to the VDJ slots only.
SLOT_FIELDS = (
    "locus",
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    "junction",
    "junction_aa",
    "umi_count",
    "consensus_count",
)


@dataclass(frozen=True, slots=True)
class ImmuneReceptors:
    """The immune receptor chains of a set of cells, from ``immune_receptors``.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``: ``receptor_type``,
            ``receptor_subtype``, ``chain_pairing``, ``multichain``, ``n_vj``
            and ``n_vdj`` (the cell's chains of each arm), ``n_contigs`` (its
            imported contigs, chains or not), and for each slot, ``VJ_1``,
            ``VJ_2``, ``VDJ_1`` and ``VDJ_2``, the slot's chain fields, such as
            ``VDJ_1_junction_aa``, missing for an empty slot. Of
            ``SLOT_FIELDS``, the table holds those that were loaded, and
            ``d_call`` for the VDJ slots only.
        chains: One row per contig: the cell's ``ids``, its ``slot`` (``""``
            for a contig in no slot), ``chain`` (whether it counts as a chain)
            and each loaded field. Text is None where missing, counts and
            positions are ``Int64`` and flags ``boolean``, with ``<NA>`` where
            missing, and numbers are NaN where missing.
        fields: The stored contig fields, by AIRR name.
        productive_only: Whether only productive contigs count as chains.
    """

    cells: pd.DataFrame = field(repr=False)
    chains: pd.DataFrame = field(repr=False)
    fields: tuple[str, ...]
    productive_only: bool


def _field_column(kind: str, values: np.ndarray) -> Any:
    """Return stored values of one field kind in their table dtype."""
    if kind in (TEXT, RAGGED):
        # pandas stores the text with its default string dtype, NA where missing.
        return pd.array(np.asarray(values, dtype=object))
    if kind == INTEGER:
        integers = np.asarray(values, dtype=np.int64)
        missing = integers < 0
        return pd.arrays.IntegerArray(np.where(missing, 0, integers), missing)
    if kind == FLAG:
        flags = np.asarray(values, dtype=np.int8)
        return pd.arrays.BooleanArray(flags == 1, flags < 0)
    # Numbers keep NaN where missing.
    return np.asarray(values, dtype=np.float64)


def contig_frame(
    cell_ids: np.ndarray,
    *,
    contig_cells: np.ndarray,
    slots: np.ndarray,
    eligible: np.ndarray,
    columns: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Return the table of contigs.

    Args:
        cell_ids: The ``ids`` of the selected cells.
        contig_cells: Each contig's cell, as a position in ``cell_ids``.
        slots: The contig in each cell's slots, -1 for an empty slot.
        eligible: Whether each contig counts as a chain.
        columns: Each loaded field's decoded values, by AIRR name.
    """
    n_contigs = len(contig_cells)
    slot_names = np.full(n_contigs, "", dtype=object)
    slots = np.asarray(slots, dtype=np.int64)
    for column, name in enumerate(SLOTS):
        filled = slots[:, column] >= 0
        slot_names[slots[filled, column]] = name
    frame: dict[str, Any] = {
        "ids": np.asarray(cell_ids, dtype=object)[np.asarray(contig_cells)],
        "slot": slot_names,
        "chain": np.asarray(eligible, dtype=bool),
    }
    for name, values in columns.items():
        frame[name] = _field_column(FIELDS[name], values)
    return pd.DataFrame(frame)


def receptor_cell_frame(
    cell_ids: np.ndarray,
    *,
    receptor_type: np.ndarray,
    receptor_subtype: np.ndarray,
    chain_pairing: np.ndarray,
    n_vj: np.ndarray,
    n_vdj: np.ndarray,
    n_contigs: np.ndarray,
    slots: np.ndarray,
    columns: Mapping[str, np.ndarray],
    fields: Sequence[str],
) -> pd.DataFrame:
    """Return the table of cells with each chain slot's fields.

    Args:
        cell_ids: The ``ids`` of the selected cells.
        receptor_type: Each cell's receptor type.
        receptor_subtype: Each cell's receptor subtype.
        chain_pairing: Each cell's chain pairing.
        n_vj: Each cell's chains of the VJ arm.
        n_vdj: Each cell's chains of the VDJ arm.
        n_contigs: Each cell's imported contigs.
        slots: The contig in each cell's slots, -1 for an empty slot.
        columns: The decoded values of ``SLOT_FIELDS`` per contig, by name.
        fields: The loaded contig fields.
    """
    n_vj = np.asarray(n_vj, dtype=np.int32)
    n_vdj = np.asarray(n_vdj, dtype=np.int32)
    frame: dict[str, Any] = {
        "receptor_type": np.asarray(receptor_type, dtype=object),
        "receptor_subtype": np.asarray(receptor_subtype, dtype=object),
        "chain_pairing": np.asarray(chain_pairing, dtype=object),
        "multichain": (n_vj > 2) | (n_vdj > 2),
        "n_vj": n_vj,
        "n_vdj": n_vdj,
        "n_contigs": np.asarray(n_contigs, dtype=np.int32),
    }
    slots = np.asarray(slots, dtype=np.int64)
    available = [name for name in SLOT_FIELDS if name in fields]
    for column, slot in enumerate(SLOTS):
        positions = slots[:, column]
        filled = positions >= 0
        for name in available:
            if name == "d_call" and slot.startswith("VJ"):
                continue
            kind = FIELDS[name]
            values = np.asarray(columns[name])
            # Slot fields are text or counts.
            if kind == TEXT:
                taken = np.full(len(positions), None, dtype=object)
            else:
                taken = np.full(len(positions), -1, dtype=values.dtype)
            taken[filled] = values[positions[filled]]
            frame[f"{slot}_{name}"] = _field_column(kind, taken)
    return pd.DataFrame(frame, index=pd.Index(np.asarray(cell_ids), name="ids"))
