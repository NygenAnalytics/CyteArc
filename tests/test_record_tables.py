"""Record tables stored in an artifact's group read back as they were encoded.

The arrays are written with zarr directly, so the test checks the layout that
contigs and lineage barcodes share without an import in between.
"""

import numpy as np
import pandas as pd
import zarr

from cytearc.datastore._operations.record_tables import (
    record_arrays,
    record_chunk_lengths,
    stored_column,
    stored_records,
)
from cytearc.metadata.records import INTEGER, RAGGED, TEXT, RecordEncoder

_FIELDS = {"name": RAGGED, "barcode": TEXT, "reads": INTEGER}


def _table():
    encoder = RecordEncoder(_FIELDS)
    encoder.add(
        pd.DataFrame(
            {
                "name": pd.Series(["m3", None, "μόριο"], dtype=object),
                "barcode": pd.Series(["TT", "AC", None], dtype=object),
                "reads": np.asarray([3, -1, 7]),
            }
        ),
        np.asarray([2, 0, 0]),
        source=1,
    )
    return encoder.finish()


def test_records_round_trip_through_their_stored_arrays(tmp_path) -> None:
    table = _table()
    arrays = record_arrays(table, "molecule")
    group = zarr.open_group(str(tmp_path / "group.zarr"), mode="w")
    for name, values in arrays.items():
        group.create_array(name, data=values)

    assert sorted(arrays) == [
        "molecule_barcode",
        "molecule_barcode_offsets",
        "molecule_barcode_values",
        "molecule_cells",
        "molecule_name",
        "molecule_name_offsets",
        "molecule_reads",
        "molecule_sources",
    ]
    assert arrays["molecule_cells"].dtype == np.int32
    stored = stored_records(group, "molecule", _FIELDS)
    assert stored.cells.tolist() == [0, 0, 2]
    assert stored.sources.tolist() == [1, 1, 1]
    for name in _FIELDS:
        assert stored.column(name).tolist() == table.column(name).tolist()
        assert (
            stored_column(group, "molecule", name, _FIELDS[name]).tolist()
            == table.column(name).tolist()
        )
    assert stored.column("name").tolist() == [None, "μόριο", "m3"]
    assert stored.column("barcode").tolist() == ["AC", None, "TT"]


def test_record_arrays_are_chunked_in_about_two_mebibytes() -> None:
    lengths = record_chunk_lengths(
        {
            "codes": np.zeros(3, dtype=np.int32),
            "bytes": np.zeros(3, dtype=np.uint8),
            "offsets": np.zeros(3, dtype=np.int64),
        }
    )
    assert lengths == {"codes": 2**19, "bytes": 2**21, "offsets": 2**18}
