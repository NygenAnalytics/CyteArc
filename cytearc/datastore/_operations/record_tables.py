"""Store long tables of records that belong to cells in an artifact.

A :class:`~cytearc.metadata.records.RecordTable` holds records ordered by
cell, such as the contigs of a V(D)J library or the molecules of a lineage
barcode library. An artifact stores the records of one kind under a prefix
that names them, such as ``contig``:

- ``<prefix>_cells``: each record's cell, its position among the selected
  cells, as int32.
- ``<prefix>_sources``: each record's source, such as its file, as int32.
- ``<prefix>_<f>`` for each field ``f``: a text field's code per record, with
  the UTF-8 bytes of its sorted distinct values in ``<prefix>_<f>_values``
  and their offsets in ``<prefix>_<f>_offsets``; a ragged field's bytes,
  with ``<prefix>_<f>_offsets``; and the values of an integer, flag or number
  field.

Record arrays are chunked in about 2 MiB, since packed text and per-record
codes are read and written whole, which chunks of megabytes make fast.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import zarr

from ...metadata.records import (
    RAGGED,
    TEXT,
    EncodedText,
    FieldKind,
    RecordTable,
    decode_ragged,
    decode_text,
)
from ...storage.types import read_array

__all__ = [
    "record_arrays",
    "record_chunk_lengths",
    "stored_column",
    "stored_records",
    "stored_text",
]

# The bytes of each chunk of a record array.
_RECORD_CHUNK_BYTES = 2 * 2**20


def record_arrays(table: RecordTable, prefix: str) -> dict[str, np.ndarray]:
    """Return the arrays that store a table's records, keyed by array name."""
    arrays: dict[str, np.ndarray] = {
        f"{prefix}_cells": np.asarray(table.cells, dtype=np.int32),
        f"{prefix}_sources": np.asarray(table.sources, dtype=np.int32),
    }
    for name, kind in table.kinds.items():
        if kind == TEXT:
            encoded = table.text[name]
            arrays[f"{prefix}_{name}"] = np.asarray(encoded.codes, dtype=np.int32)
            arrays[f"{prefix}_{name}_values"] = encoded.data
            arrays[f"{prefix}_{name}_offsets"] = encoded.offsets
        elif kind == RAGGED:
            data, offsets = table.ragged[name]
            arrays[f"{prefix}_{name}"] = data
            arrays[f"{prefix}_{name}_offsets"] = offsets
        else:
            arrays[f"{prefix}_{name}"] = np.asarray(table.values[name])
    return arrays


def record_chunk_lengths(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return the rows of each chunk of record arrays, about 2 MiB of each."""
    return {
        name: max(1, _RECORD_CHUNK_BYTES // max(1, values.dtype.itemsize))
        for name, values in arrays.items()
    }


def stored_text(group: zarr.Group, prefix: str, name: str) -> EncodedText:
    """Read a stored text field's codes and distinct values."""
    return EncodedText(
        codes=read_array(group, f"{prefix}_{name}"),
        data=read_array(group, f"{prefix}_{name}_values"),
        offsets=read_array(group, f"{prefix}_{name}_offsets"),
    )


def stored_column(
    group: zarr.Group, prefix: str, name: str, kind: FieldKind
) -> np.ndarray:
    """Read and decode one stored field: text with None where missing, or values."""
    if kind == TEXT:
        encoded = stored_text(group, prefix, name)
        return decode_text(encoded.codes, encoded.categories)
    if kind == RAGGED:
        return decode_ragged(
            read_array(group, f"{prefix}_{name}"),
            read_array(group, f"{prefix}_{name}_offsets"),
        )
    return read_array(group, f"{prefix}_{name}")


def stored_records(
    group: zarr.Group, prefix: str, kinds: Mapping[str, FieldKind]
) -> RecordTable:
    """Read the stored records of the fields ``kinds`` names, encoded as stored."""
    text: dict[str, EncodedText] = {}
    ragged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    values: dict[str, np.ndarray] = {}
    for name, kind in kinds.items():
        if kind == TEXT:
            text[name] = stored_text(group, prefix, name)
        elif kind == RAGGED:
            ragged[name] = (
                read_array(group, f"{prefix}_{name}"),
                read_array(group, f"{prefix}_{name}_offsets"),
            )
        else:
            values[name] = read_array(group, f"{prefix}_{name}")
    return RecordTable(
        cells=read_array(group, f"{prefix}_cells").astype(np.int64),
        sources=read_array(group, f"{prefix}_sources"),
        kinds=MappingProxyType(dict(kinds)),
        text=MappingProxyType(text),
        ragged=MappingProxyType(ragged),
        values=MappingProxyType(values),
    )
