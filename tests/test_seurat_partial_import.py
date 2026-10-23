"""Seurat objects that import in part: metadata, assays, and reductions.

Payloads are serialized with the wire builder of ``test_seurat_reader``, from
the R serialization format, as the other Seurat tests do.
"""

import io
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc
import cytearc.readers.seurat as seurat_module
from cytearc.datastore.datastore import DataStore
from cytearc.readers._rds import R_INT_NA, RdsFormatError, RType
from cytearc.readers.seurat import SeuratImportError, SeuratReader
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.writers.seurat import SeuratToZarr
from tests.test_seurat_reader import _Wire


_CELLS = ["c1", "c2", "c3"]
_GENES = ["g1", "g2"]
# R's NA_real_: a NaN whose low word is 1954.
_NA_REAL = bytes.fromhex("7ff00000000007a2")


def _real_bytes(
    wire: _Wire, raw: list[bytes], attributes: list[tuple[str, bytes]]
) -> bytes:
    """A double vector of raw big-endian elements, such as NA_real_."""
    return (
        wire.integer(wire.flags(RType.REAL, attributes=bool(attributes)))
        + wire.integer(len(raw))
        + b"".join(raw)
        + (wire.attributes(attributes) if attributes else b"")
    )


def _classed(
    wire: _Wire,
    r_type: RType,
    values: list[int] | list[float],
    classes: list[str],
    extra: list[tuple[str, bytes]] | None = None,
) -> bytes:
    return wire.atomic_vector(
        r_type,
        values,
        attributes=[*(extra or []), ("class", wire.string_vector(classes))],
    )


def _wrapped_strings(wire: _Wire, values: list[str]) -> bytes:
    """A character vector serialized as R's ``wrap_string`` ALTREP."""
    return wire.altrep(
        "wrap_string",
        wire.untagged_pair(wire.string_vector(values), wire.integer_vector([0, 0])),
    )


def _deferred(wire: _Wire, argument: bytes, info: bytes | None = None) -> bytes:
    """``as.character`` of a numeric vector, as R's ``deferred_string``."""
    return wire.altrep(
        "deferred_string",
        wire.untagged_pair(
            argument, wire.integer_vector([0]) if info is None else info
        ),
    )


def _counts(
    wire: _Wire,
    *,
    features: list[str] = _GENES,
    cells: list[str] = _CELLS,
    values: list[int] | None = None,
) -> bytes:
    size = len(features) * len(cells)
    return wire.matrix(
        values if values is not None else [index % 4 for index in range(size)],
        (len(features), len(cells)),
        rows=features,
        columns=cells,
    )


def _legacy(
    wire: _Wire,
    *,
    counts: bytes | None = None,
    features: list[str] = _GENES,
    meta_features: bytes | None = None,
    source_class: str = "Assay",
    extra: list[tuple[str, bytes]] | None = None,
) -> bytes:
    return wire.s4(
        [
            ("counts", _counts(wire, features=features) if counts is None else counts),
            (
                "meta.features",
                wire.data_frame([], features)
                if meta_features is None
                else meta_features,
            ),
            *(extra or []),
            ("class", wire.string_vector([source_class])),
        ]
    )


def _empty_dgc(wire: _Wire) -> bytes:
    """``new("dgCMatrix")``: 0 by 0, without Dimnames."""
    return wire.s4(
        [
            ("i", wire.integer_vector([])),
            ("p", wire.integer_vector([0])),
            ("Dim", wire.integer_vector([0, 0])),
            ("Dimnames", wire.vector([wire.nil(), wire.nil()])),
            ("x", wire.real_vector([])),
            ("factors", wire.vector([])),
            ("class", wire.string_vector(["dgCMatrix"])),
        ]
    )


def _dimreduc(
    wire: _Wire,
    *,
    values: list[float] | None = None,
    rows: list[str] = _CELLS,
    columns: list[str] | None = None,
    assay_used: str = "RNA",
    key: str | None = "PC_",
    loadings: bytes | None = None,
    extra: list[tuple[str, bytes]] | None = None,
) -> bytes:
    columns = ["PC_1", "PC_2"] if columns is None else columns
    size = len(rows) * len(columns)
    slots = [
        (
            "cell.embeddings",
            wire.matrix(
                values if values is not None else [float(i + 1) for i in range(size)],
                (len(rows), len(columns)),
                rows=rows,
                columns=columns,
                real=True,
            ),
        ),
        (
            "feature.loadings",
            wire.matrix([], (0, 0), real=True) if loadings is None else loadings,
        ),
        ("assay.used", wire.string_vector([assay_used])),
        ("global", wire.logical_vector([0])),
        ("stdev", wire.real_vector([])),
    ]
    if key is not None:
        slots.append(("key", wire.string_vector([key])))
    slots.extend(extra or [])
    slots.append(("class", wire.string_vector(["DimReduc"])))
    return wire.s4(slots)


def _seurat(
    wire: _Wire,
    *,
    assays: dict[str, bytes] | None = None,
    metadata: bytes | None = None,
    active: str = "RNA",
    reductions: dict[str, bytes] | None = None,
    cells: list[str] = _CELLS,
) -> bytes:
    assays = {"RNA": _legacy(wire)} if assays is None else assays
    reductions = reductions or {}
    return wire.document(
        wire.s4(
            [
                ("assays", wire.vector(list(assays.values()), names=list(assays))),
                (
                    "meta.data",
                    wire.data_frame(
                        [
                            (
                                "group",
                                wire.string_vector(["a", "b", "c", "d"][: len(cells)]),
                            )
                        ],
                        cells,
                    )
                    if metadata is None
                    else metadata,
                ),
                ("active.assay", wire.string_vector([active])),
                (
                    "active.ident",
                    wire.factor([1] * len(cells), ["all"], names=cells),
                ),
                (
                    "reductions",
                    wire.vector(list(reductions.values()), names=list(reductions)),
                ),
                ("class", wire.string_vector(["Seurat"])),
            ]
        )
    )


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _writer(
    reader: SeuratReader,
    destination: MemoryStore,
    **options: object,
) -> SeuratToZarr:
    return SeuratToZarr(
        reader,
        destination,
        mem_budget="64M",
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
        **options,  # type: ignore[arg-type]
    )


def _destination_with_sentinel() -> MemoryStore:
    destination = MemoryStore()
    zarr.open_group(store=destination, mode="w").create_group("sentinel")
    return destination


def _untouched(destination: MemoryStore) -> bool:
    root = zarr.open_group(store=destination, mode="r")
    return set(root.group_keys()) == {"sentinel"}


def _notices(notices: tuple, code: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    return [
        (notice.objectPath, notice.context) for notice in notices if notice.code == code
    ]


def _mixed_metadata(wire: _Wire) -> bytes:
    """Cell metadata whose columns 1 to 5 cannot be imported."""
    return wire.data_frame(
        [
            ("kept", wire.string_vector(["a", "b", "c"])),
            ("l", wire.vector([wire.integer_vector([1])] * 3)),
            (
                "m",
                wire.real_vector(
                    [1.0] * 6, attributes=[("dim", wire.integer_vector([3, 2]))]
                ),
            ),
            ("", wire.real_vector([1.0, 2.0, 3.0])),
            (None, wire.real_vector([1.0, 2.0, 3.0])),  # type: ignore[list-item]
            ("kept", wire.string_vector(["x", "y", "z"])),
            ("n", wire.real_vector([1.5, float("nan"), 3.0])),
        ],
        _CELLS,
    )


_SKIPPED_COLUMNS = [
    (
        "meta.data/l",
        {
            "column": "l",
            "reason": "unsupported_metadata_type",
            "details": {"rType": "VECTOR", "classNames": ()},
        },
    ),
    (
        "meta.data/m",
        {
            "column": "m",
            "reason": "length_mismatch",
            "details": {"actual": 6, "expected": 3},
        },
    ),
    ("meta.data/names/3", {"column": None, "reason": "missing_id", "details": {}}),
    ("meta.data/names/4", {"column": None, "reason": "missing_id", "details": {}}),
    (
        "meta.data/names/5",
        {"column": "kept", "reason": "duplicate_id", "details": {"id": "kept"}},
    ),
]


def test_cell_metadata_columns_that_cannot_be_decoded_are_skipped(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "mixed.rds", _seurat(wire, metadata=_mixed_metadata(wire)))

    inspection = cytearc.inspect_seurat(path)
    assert inspection.cellMetadata.dimensions == (3, 2)
    assert _notices(inspection.cellMetadata.notices, "skipped_metadata_column") == (
        _SKIPPED_COLUMNS
    )
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        metadata = reader.cellMetadata
        assert metadata.columnNames == ("kept", "n")
        # R reads the first of two columns of one name.
        assert metadata.column("kept").read_block(0, 3).values == ("a", "b", "c")
        with pytest.raises(SeuratImportError) as error:
            metadata.column("l")
        assert error.value.code == "unsupported_metadata_type"
        with pytest.raises(KeyError, match="^'missing'$"):
            metadata.column("missing")
        result = _writer(reader, destination).dump(batch_size=1)

    assert _notices(result.notices, "skipped_metadata_column") == _SKIPPED_COLUMNS
    cells = zarr.open_group(store=destination, mode="r")["cellData"]
    assert cells["kept"][:].tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(cells["n"][:], [1.5, np.nan, 3.0])
    assert "l" not in cells and "m" not in cells


def test_cell_metadata_selection_includes_or_excludes_columns(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "mixed.rds", _seurat(wire, metadata=_mixed_metadata(wire)))

    for selection in (["n", "kept"], {"include": ["kept", "n"]}):
        with SeuratReader(path, cell_metadata=selection) as reader:
            # Columns keep the table's order.
            assert reader.cellMetadata.columnNames == ("kept", "n")
            # Only the repeated name is reported; other columns are not selected.
            assert (
                _notices(
                    reader.inspection.cellMetadata.notices, "skipped_metadata_column"
                )
                == _SKIPPED_COLUMNS[-1:]
            )
    with SeuratReader(path, cell_metadata={"include": ["n"]}) as reader:
        assert reader.cellMetadata.columnNames == ("n",)
        assert reader.inspection.cellMetadata.notices == ()
    excluded = cytearc.inspect_seurat(path, cell_metadata={"exclude": ["l", "m", "n"]})
    assert excluded.cellMetadata.dimensions == (3, 1)
    assert (
        _notices(excluded.cellMetadata.notices, "skipped_metadata_column")
        == _SKIPPED_COLUMNS[2:]
    )

    # A column selected by name must decode.
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path, cell_metadata=["kept", "m"])
    assert (error.value.code, error.value.objectPath) == (
        "length_mismatch",
        "meta.data/m",
    )


@pytest.mark.parametrize(
    ("selection", "error_type", "message"),
    [
        ("kept", TypeError, "^cell_metadata selection must be a sequence of names$"),
        (["n", "n"], ValueError, "^cell_metadata selection contains duplicate names$"),
        (
            {"include": ["n"], "exclude": ["l"]},
            ValueError,
            "^cell_metadata mapping must have exactly one key, 'include' or 'exclude'$",
        ),
        (
            {"only": ["n"]},
            ValueError,
            "^cell_metadata mapping must have exactly one key",
        ),
        (
            ["n", "zz"],
            ValueError,
            "^cell metadata selection names columns that meta.data does not hold: "
            "'zz'$",
        ),
        (
            {"exclude": ["yy", "zz"]},
            ValueError,
            "does not hold: 'yy', 'zz'$",
        ),
    ],
)
def test_cell_metadata_selections_are_validated(
    tmp_path: Path,
    selection: object,
    error_type: type[Exception],
    message: str,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "mixed.rds", _seurat(wire, metadata=_mixed_metadata(wire)))
    with pytest.raises(error_type, match=message):
        SeuratReader(path, cell_metadata=selection)  # type: ignore[arg-type]


def test_excluding_a_reserved_cell_metadata_column_lets_the_import_proceed(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    metadata = wire.data_frame(
        [
            ("ids", wire.string_vector(["x", "y", "z"])),
            ("group", wire.string_vector(["a", "b", "c"])),
        ],
        _CELLS,
    )
    path = _write(tmp_path / "reserved.rds", _seurat(wire, metadata=metadata))
    destination = _destination_with_sentinel()
    with SeuratReader(path) as reader:
        with pytest.raises(
            ValueError,
            match=r"^cell metadata column 'ids' is reserved; leave it out with "
            r"SeuratReader\(cell_metadata=\{'exclude': \['ids'\]\}\)$",
        ):
            _writer(reader, destination)
    assert _untouched(destination)
    destination = MemoryStore()
    with SeuratReader(path, cell_metadata={"exclude": ["ids"]}) as reader:
        _writer(reader, destination).dump()
    cells = zarr.open_group(store=destination, mode="r")["cellData"]
    assert cells["ids"][:].tolist() == _CELLS
    assert cells["group"][:].tolist() == ["a", "b", "c"]


def test_identifiers_serialized_as_altrep_are_read(tmp_path: Path) -> None:
    wire = _Wire()
    # Row names, column names, list names, and Dimnames, as R's wrap_string
    # and deferred_string ALTREP vectors.
    metadata = wire.vector(
        [wire.string_vector(["a", "b", "c"])],
        attributes=[
            ("names", _wrapped_strings(wire, ["group"])),
            ("row.names", _wrapped_strings(wire, _CELLS)),
            ("class", wire.string_vector(["data.frame"])),
        ],
    )
    counts = wire.integer_vector(
        [1, 0, 0, 2, 3, 0],
        attributes=[
            ("dim", wire.integer_vector([2, 3])),
            (
                "dimnames",
                wire.vector(
                    [_wrapped_strings(wire, _GENES), _wrapped_strings(wire, _CELLS)]
                ),
            ),
        ],
    )
    embeddings = wire.real_vector(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        attributes=[
            ("dim", wire.integer_vector([3, 2])),
            (
                "dimnames",
                wire.vector(
                    [
                        _wrapped_strings(wire, _CELLS),
                        # as.character(1:2)
                        _deferred(wire, wire.integer_vector([1, 2])),
                    ]
                ),
            ),
        ],
    )
    reduction = wire.s4(
        [
            ("cell.embeddings", embeddings),
            ("assay.used", wire.string_vector(["RNA"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    payload = wire.document(
        wire.s4(
            [
                (
                    "assays",
                    wire.vector(
                        [_legacy(wire, counts=counts)],
                        attributes=[("names", _wrapped_strings(wire, ["RNA"]))],
                    ),
                ),
                ("meta.data", metadata),
                ("active.assay", wire.string_vector(["RNA"])),
                ("active.ident", wire.factor([1, 1, 1], ["all"], names=_CELLS)),
                ("reductions", wire.vector([reduction], names=["pca"])),
                ("class", wire.string_vector(["Seurat"])),
            ]
        )
    )
    path = _write(tmp_path / "altrep-ids.rds", payload)
    with SeuratReader(path) as reader:
        assert tuple(reader.cellIds) == tuple(_CELLS)
        assert reader.cellMetadata.column("group").read_block(0, 3).values == (
            "a",
            "b",
            "c",
        )
        assay = reader.get_assay("RNA")
        assert tuple(assay.featureIds) == ("g1", "g2")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3), [[1, 0], [0, 2], [3, 0]]
        )
        pca = reader.get_reduction("pca")
        assert tuple(pca.cellEmbeddings.columnIds) == ("1", "2")
        assert pca.cellPositions is None

    unsupported = wire.data_frame(
        [("group", wire.string_vector(["a", "b", "c"]))], _CELLS
    ).replace(
        wire.string_vector(_CELLS),
        wire.altrep("mmap_string", wire.string_vector(_CELLS)),
    )
    path = _write(tmp_path / "unsupported-ids.rds", _seurat(wire, metadata=unsupported))
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)
    assert (error.value.code, error.value.objectPath) == (
        "unsupported_altrep",
        "meta.data/row.names",
    )


_DEFERRED_DOUBLES = [
    (struct.pack(">d", 1.5), "1.5"),
    (struct.pack(">d", 2.0), "2"),
    (struct.pack(">d", 1e5), "1e+05"),
    (struct.pack(">d", -1e5), "-1e+05"),
    (struct.pack(">d", 120_000.0), "120000"),
    (struct.pack(">d", 123_456.0), "123456"),
    (struct.pack(">d", 0.1 + 0.2), "0.3"),
    (struct.pack(">d", 1 / 3), "0.333333333333333"),
    (struct.pack(">d", 1e-4), "1e-04"),
    (struct.pack(">d", 0.00012), "0.00012"),
    (struct.pack(">d", 1e15), "1e+15"),
    (struct.pack(">d", 2.0**53), "9007199254740992"),
    (struct.pack(">d", 1e100), "1e+100"),
    (struct.pack(">d", 1_234_567.1), "1234567.1"),
    (struct.pack(">d", 5e-324), "4.94065645841247e-324"),
    (struct.pack(">d", -0.0), "0"),
    (struct.pack(">d", float("inf")), "Inf"),
    (struct.pack(">d", float("-inf")), "-Inf"),
    (struct.pack(">d", float("nan")), "NaN"),
    (_NA_REAL, None),
]


def test_deferred_strings_of_doubles_follow_r_formatting(tmp_path: Path) -> None:
    wire = _Wire()
    cells = [f"c{index}" for index in range(len(_DEFERRED_DOUBLES))]
    numbers = _real_bytes(wire, [raw for raw, _ in _DEFERRED_DOUBLES], [])
    metadata = wire.data_frame(
        [
            ("text", _deferred(wire, numbers)),
            # R's scipen option, kept with the vector, favors fixed notation.
            (
                "fixed",
                _deferred(
                    wire,
                    wire.real_vector([1e5, 1e-4, 1e15] * 6 + [123_456.0, 1.0]),
                    wire.integer_vector([100]),
                ),
            ),
            (
                "scientific",
                _deferred(
                    wire,
                    wire.real_vector([123_456.0] * 20),
                    wire.integer_vector([-6]),
                ),
            ),
            (
                "tied",
                _deferred(
                    wire,
                    wire.real_vector([123_456.0] * 20),
                    wire.integer_vector([-5]),
                ),
            ),
            # A state without the option uses R's default of 0.
            (
                "default",
                _deferred(wire, wire.real_vector([1e5] * 20), wire.nil()),
            ),
        ],
        cells,
    )
    assays = {"RNA": _legacy(wire, counts=_counts(wire, cells=cells), features=_GENES)}
    path = _write(
        tmp_path / "deferred.rds",
        _seurat(wire, assays=assays, metadata=metadata, cells=cells),
    )
    with SeuratReader(path) as reader:
        metadata_table = reader.cellMetadata
        text = metadata_table.column("text").read_block(0, len(cells))
        assert text.values == tuple(expected for _, expected in _DEFERRED_DOUBLES)
        assert text.missing.tolist() == [False] * 19 + [True]
        assert metadata_table.column("fixed").read_block(0, 5).values == (
            "100000",
            "0.0001",
            "1000000000000000",
            "100000",
            "0.0001",
        )
        assert metadata_table.column("scientific").read_block(0, 1).values == (
            "1.23456e+05",
        )
        assert metadata_table.column("tied").read_block(0, 1).values == ("123456",)
        assert metadata_table.column("default").read_block(0, 1).values == ("1e+05",)
        assert _notices(metadata_table.notices, "formatted_deferred_strings") == [
            (f"meta.data/{name}", {})
            for name in ("text", "fixed", "scientific", "tied", "default")
        ]


def test_metadata_classes_decide_how_values_import(tmp_path: Path) -> None:
    wire = _Wire()
    int64 = [1, -5, 2**40 + 3, -(2**63), 2**53 + 1]
    cells = [f"c{index}" for index in range(5)]
    metadata = wire.data_frame(
        [
            (
                "count64",
                _real_bytes(
                    wire,
                    [struct.pack(">q", value) for value in int64],
                    [("class", wire.string_vector(["integer64"]))],
                ),
            ),
            (
                "day",
                _classed(
                    wire,
                    RType.REAL,
                    [19_000.0, 19_000.75, float("nan"), -719_162.0, 2_932_896.0],
                    ["Date"],
                ),
            ),
            (
                "iday",
                _classed(
                    wire, RType.INTEGER, [0, R_INT_NA, 1, -1, 365], ["IDate", "Date"]
                ),
            ),
            (
                "stamp",
                _classed(
                    wire,
                    RType.REAL,
                    [0.0, 1.5, 86_399.9999996, -1.0, float("nan")],
                    ["POSIXct", "POSIXt"],
                    [("tzone", wire.string_vector(["UTC"]))],
                ),
            ),
            (
                "elapsed",
                _classed(
                    wire,
                    RType.REAL,
                    [1.0, 2.5, 0.0, 4.0, 5.0],
                    ["difftime"],
                    [("units", wire.string_vector(["hours"]))],
                ),
            ),
            (
                "clock",
                _classed(wire, RType.INTEGER, [1, 2, 3, 4, 5], ["ITime"]),
            ),
            ("plain", _classed(wire, RType.REAL, [1.0] * 5, ["AsIs", "numeric"])),
            # integer64 in an integer vector, and dates outside ISO 8601 years.
            (
                "bad64",
                _classed(wire, RType.INTEGER, [1] * 5, ["integer64"]),
            ),
            (
                "far",
                _classed(wire, RType.REAL, [1.0, 1e7, 1.0, 1.0, 1.0], ["Date"]),
            ),
            (
                "textstamp",
                wire.string_vector(
                    ["a"] * 5,
                    attributes=[("class", wire.string_vector(["POSIXct"]))],
                ),
            ),
        ],
        cells,
    )
    assays = {"RNA": _legacy(wire, counts=_counts(wire, cells=cells))}
    path = _write(
        tmp_path / "classes.rds",
        _seurat(wire, assays=assays, metadata=metadata, cells=cells),
    )
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        table = reader.cellMetadata
        assert table.columnNames == (
            "count64",
            "day",
            "iday",
            "stamp",
            "elapsed",
            "clock",
            "plain",
        )
        count64 = table.column("count64").read_block(0, 5)
        assert table.column("count64").kind == "integer"
        assert count64.values.dtype == np.int64
        assert count64.values.tolist()[:3] == [1, -5, 2**40 + 3]
        assert count64.values.tolist()[4] == 2**53 + 1
        assert count64.missing.tolist() == [False, False, False, True, False]
        assert table.column("day").read_block(0, 5).values == (
            "2022-01-08",
            "2022-01-08",
            None,
            "0001-01-01",
            "9999-12-31",
        )
        assert table.column("iday").read_block(0, 5).values == (
            "1970-01-01",
            None,
            "1970-01-02",
            "1969-12-31",
            "1971-01-01",
        )
        assert table.column("stamp").read_block(0, 5).values == (
            "1970-01-01T00:00:00Z",
            "1970-01-01T00:00:01.500000Z",
            "1970-01-02T00:00:00Z",
            "1969-12-31T23:59:59Z",
            None,
        )
        assert table.column("elapsed").kind == "real"
        assert table.column("clock").kind == "integer"
        assert [
            (notice.code, notice.objectPath, notice.context) for notice in table.notices
        ] == [
            (
                "converted_temporal_metadata",
                "meta.data/day",
                {"className": "Date"},
            ),
            (
                "converted_temporal_metadata",
                "meta.data/iday",
                {"className": "Date"},
            ),
            (
                "converted_temporal_metadata",
                "meta.data/stamp",
                {"className": "POSIXct"},
            ),
            (
                "classed_metadata_column",
                "meta.data/elapsed",
                {"classNames": ("difftime",), "units": "hours"},
            ),
            (
                "classed_metadata_column",
                "meta.data/clock",
                {"classNames": ("ITime",)},
            ),
            (
                "skipped_metadata_column",
                "meta.data/bad64",
                {
                    "column": "bad64",
                    "reason": "invalid_integer64",
                    "details": {"rType": "INTEGER"},
                },
            ),
            (
                "skipped_metadata_column",
                "meta.data/far/1",
                {
                    "column": "far",
                    "reason": "unsupported_metadata_value",
                    "details": {"value": 1e7},
                },
            ),
            (
                "skipped_metadata_column",
                "meta.data/textstamp",
                {
                    "column": "textstamp",
                    "reason": "unsupported_metadata_type",
                    "details": {"rType": "STRING", "classNames": ("POSIXct",)},
                },
            ),
        ]
        _writer(reader, destination).dump()

    columns = zarr.open_group(store=destination, mode="r")["cellData"]
    assert columns["count64"].dtype == np.int64
    assert columns["count64"][:].tolist()[4] == 2**53 + 1
    assert columns["day"][:].tolist()[0] == "2022-01-08"
    assert columns["stamp"][:].tolist()[1] == "1970-01-01T00:00:01.500000Z"


@pytest.mark.parametrize("empty_rows", ["character", "integer"])
@pytest.mark.parametrize("container", ["Assay", "Assay5"])
def test_an_empty_prototype_feature_table_is_empty_feature_metadata(
    tmp_path: Path,
    empty_rows: str,
    container: str,
) -> None:
    wire = _Wire()
    # data.frame(): no columns and no rows, which new("Assay") holds.
    prototype = wire.vector(
        [],
        attributes=[
            ("names", wire.string_vector([])),
            (
                "row.names",
                wire.string_vector([])
                if empty_rows == "character"
                else wire.integer_vector([]),
            ),
            ("class", wire.string_vector(["data.frame"])),
        ],
    )
    if container == "Assay":
        assay = _legacy(wire, meta_features=prototype)
    else:
        assay = wire.s4(
            [
                ("layers", wire.vector([_counts(wire)], names=["counts"])),
                ("cells", wire.logmap([1, 1, 1], _CELLS, ["counts"])),
                ("features", wire.logmap([1, 1], _GENES, ["counts"])),
                ("meta.data", prototype),
                ("class", wire.string_vector(["Assay5"])),
            ]
        )
    path = _write(tmp_path / "prototype.rds", _seurat(wire, assays={"RNA": assay}))
    with SeuratReader(path) as reader:
        assert reader.inspection.assay("RNA").importable
        metadata = reader.get_assay("RNA").featureMetadata
        assert metadata.columns == ()
        assert tuple(metadata.rowIds) == ("g1", "g2")
        assert reader.get_assay("RNA").notices == ()


def _integrated_object(wire: _Wire) -> bytes:
    """The v3/v4 IntegrateData layout: the active assay holds only ``data``."""
    integrated = _legacy(
        wire,
        counts=_empty_dgc(wire),
        extra=[
            (
                "data",
                wire.matrix(
                    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                    (2, 3),
                    rows=_GENES,
                    columns=_CELLS,
                    real=True,
                ),
            )
        ],
    )
    loadings = wire.matrix(
        [0.1, 0.2, 0.3, 0.4], (2, 2), rows=_GENES, columns=["PC_1", "PC_2"], real=True
    )
    return _seurat(
        wire,
        assays={
            "RNA": _legacy(wire, counts=_counts(wire, values=[1, 0, 0, 2, 3, 0])),
            "integrated": integrated,
        },
        active="integrated",
        reductions={
            "pca": _dimreduc(wire, assay_used="integrated", loadings=loadings),
            "umap": _dimreduc(
                wire,
                assay_used="integrated",
                key="UMAP_",
                columns=["UMAP_1", "UMAP_2"],
            ),
        },
    )


def test_an_assay_without_raw_counts_reports_counts_layer_missing(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "integrated.rds", _integrated_object(wire))

    inspection = cytearc.inspect_seurat(path)
    assert inspection.assay("RNA").importable
    blocked = inspection.assay("integrated")
    assert not blocked.importable
    diagnostic = blocked.blockingDiagnostic
    assert diagnostic is not None
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (
        "counts_layer_missing",
        "assays/integrated/counts",
        {"dimensions": (0, 0), "dataSlot": True},
    )
    # Reductions of that assay keep their coordinates and drop the loadings.
    pca = inspection.reduction("pca")
    assert (pca.importable, pca.assayUsed, pca.role) == (
        True,
        "integrated",
        "graphCoordinates",
    )
    assert [
        (notice.code, notice.objectPath, notice.context) for notice in pca.notices
    ] == [
        (
            "reduction_assay_unavailable",
            "reductions/pca/assay.used",
            {"assayUsed": "integrated", "reason": "counts_layer_missing"},
        ),
        (
            "dropped_reduction_loadings",
            "reductions/pca/feature.loadings",
            {"assayUsed": "integrated"},
        ),
    ]
    assert inspection.reduction("umap").role == "displayEmbedding"


def test_an_unimportable_active_assay_requires_an_explicit_default_assay(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "integrated.rds", _integrated_object(wire))
    destination = _destination_with_sentinel()

    with SeuratReader(path) as reader:
        with pytest.raises(
            ValueError,
            match=r"^Active assay 'integrated' cannot be imported \(counts_layer_missing: "
            r"legacy assay holds no raw counts: its counts slot is empty\); pass "
            r"default_assay naming one of the imported assays \['RNA'\]$",
        ):
            _writer(reader, destination)
        with pytest.raises(SeuratImportError) as error:
            _writer(reader, destination, default_assay="integrated")
        assert error.value.code == "counts_layer_missing"
        with pytest.raises(
            ValueError,
            match=r"^default_assay 'ADT' is not imported; pass one of the imported "
            r"assays \['RNA'\]$",
        ):
            _writer(reader, destination, default_assay="ADT")
        with pytest.raises(
            TypeError, match="^default_assay must be an assay name or None$"
        ):
            _writer(reader, destination, default_assay=1)
    with SeuratReader(path, assays=["RNA"]) as reader:
        with pytest.raises(
            ValueError,
            match=r"^Active assay 'integrated' is not selected for import; select it, "
            r"or pass default_assay naming one of the imported assays \['RNA'\]$",
        ):
            _writer(reader, destination)
    # An assay that the caller names must import.
    with SeuratReader(path, assays=["RNA", "integrated"]) as reader:
        with pytest.raises(SeuratImportError) as error:
            _writer(reader, destination, default_assay="RNA")
        assert error.value.code == "counts_layer_missing"
    assert _untouched(destination)


def test_default_assay_imports_the_rest_of_an_integrated_object(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "integrated.rds", _integrated_object(wire))
    destination = MemoryStore()

    with SeuratReader(path) as reader:
        result = _writer(reader, destination, default_assay="RNA").dump()

    assert (result.assayNames, result.defaultAssay) == (("RNA",), "RNA")
    assert _notices(result.notices, "skipped_assay") == [
        (
            "assays/integrated/counts",
            {
                "reason": "counts_layer_missing",
                "details": {"dimensions": (0, 0), "dataSlot": True},
            },
        )
    ]
    pca = result.reductionArtifacts["pca"]
    umap = result.reductionArtifacts["umap"]
    assert (pca.kind, pca.assay) == ("imported_coordinates", "RNA")
    assert (umap.kind, umap.assay) == ("embedding", "RNA")
    root = zarr.open_group(store=destination, mode="r")
    assert root.attrs["defaultAssay"] == "RNA"
    assert "loadings" not in artifact_group(root, pca)
    np.testing.assert_array_equal(
        artifact_group(root, umap)["values"][:], [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    )
    assert inspect_artifact(root, umap).parameters["role"] == "umap"
    ds = DataStore(destination, min_features_per_cell=0, nthreads=1, mem_budget="64M")
    assert ds.artifacts.find("embedding", imported="umap", from_assay="RNA") == umap

    # The selection that failed before: RNA and the UMAP, with the default.
    with SeuratReader(path, assays=["RNA"], reductions=["umap"]) as reader:
        result = _writer(reader, MemoryStore(), default_assay="RNA").dump()
    assert list(result.reductionArtifacts) == ["umap"]


def test_reductions_that_cannot_be_imported_are_skipped_by_default(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    broken = wire.s4(
        [
            (
                "cell.embeddings",
                wire.real_vector(
                    [1.0] * 6, attributes=[("dim", wire.integer_vector([3, 2]))]
                ),
            ),
            ("assay.used", wire.string_vector(["RNA"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    path = _write(
        tmp_path / "broken.rds",
        _seurat(wire, reductions={"pca": _dimreduc(wire), "broken": broken}),
    )
    with SeuratReader(path) as reader:
        result = _writer(reader, MemoryStore()).dump()
    assert list(result.reductionArtifacts) == ["pca"]
    assert _notices(result.notices, "skipped_reduction") == [
        (
            "reductions/broken/cell.embeddings/Dimnames",
            {"reason": "missing_dimnames", "details": {}},
        )
    ]
    destination = _destination_with_sentinel()
    with SeuratReader(path, reductions=["pca", "broken"]) as reader:
        with pytest.raises(
            ValueError,
            match=r"^Selected reductions cannot be imported: broken \(missing_dimnames\)$",
        ):
            _writer(reader, destination)
    assert _untouched(destination)


_ROLE_CASES = {
    # name: (key, columns, override, role, embedding role)
    "pca": ("PC_", 2, None, "graphCoordinates", None),
    "wnn.umap": ("wnnUMAP_", 2, None, "displayEmbedding", "umap"),
    "umap.rna": ("rnaUMAP_", 2, None, "displayEmbedding", "umap"),
    "ref.umap": ("refUMAP_", 2, None, "displayEmbedding", "umap"),
    "umap.harmony": ("UMAP_", 2, None, "displayEmbedding", "umap"),
    "umap_cca": ("umapcca_", 2, None, "displayEmbedding", "umap"),
    "tsne.rna": ("tSNE_", 2, None, "displayEmbedding", "tsne"),
    "UMAP.RNA": ("PC_", 2, None, "graphCoordinates", None),
    "umap": ("Embedding_", 2, None, "displayEmbedding", "umap"),
    "t-sne": ("Dim_", 2, None, "displayEmbedding", "tsne"),
    "TSNE": (None, 2, None, "displayEmbedding", "tsne"),
    "umap3d": ("UMAP_", 3, None, "displayEmbedding", "umap"),
    "umap10": ("UMAP_", 10, None, "graphCoordinates", None),
    "phate": ("PHATE_", 2, "umap", "displayEmbedding", "umap"),
    "umap.layout": ("UMAP_", 2, "coordinates", "graphCoordinates", None),
    "diffusion": ("DC_", 4, "tsne", "displayEmbedding", "tsne"),
}


def test_display_embeddings_are_recognized_by_key_name_and_columns(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    reductions = {
        name: _dimreduc(
            wire,
            key=key,
            columns=[f"D_{index}" for index in range(columns)],
        )
        for name, (key, columns, _override, _role, _embedding) in _ROLE_CASES.items()
    }
    path = _write(tmp_path / "roles.rds", _seurat(wire, reductions=reductions))
    overrides = {
        name: case[2] for name, case in _ROLE_CASES.items() if case[2] is not None
    }
    destination = MemoryStore()
    with SeuratReader(path, reduction_roles=overrides) as reader:  # type: ignore[arg-type]
        for name, (key, _columns, _override, role, embedding) in _ROLE_CASES.items():
            reduction = reader.get_reduction(name)
            assert (reduction.key, reduction.role, reduction.embeddingRole) == (
                key,
                role,
                embedding,
            ), name
            assert reader.inspection.reduction(name).role == role
        result = _writer(reader, destination).dump()

    root = zarr.open_group(store=destination, mode="r")
    for name, (_key, _columns, _override, role, embedding) in _ROLE_CASES.items():
        ref = result.reductionArtifacts[name]
        parameters = inspect_artifact(root, ref).parameters
        if embedding is None:
            assert (ref.kind, parameters["role"]) == (
                "imported_coordinates",
                "graphcoordinates",
            ), name
        else:
            assert (ref.kind, parameters["role"]) == ("embedding", embedding), name


@pytest.mark.parametrize(
    ("roles", "error_type", "message"),
    [
        (["umap"], TypeError, "^reduction_roles must map reduction names"),
        (
            {"pca": "pca"},
            ValueError,
            "^reduction role for 'pca' must be 'umap', 'tsne', or 'coordinates', "
            "not 'pca'$",
        ),
        (
            {"nope": "umap"},
            ValueError,
            "^reduction_roles names reductions that the object does not hold: 'nope'$",
        ),
    ],
)
def test_reduction_roles_are_validated(
    tmp_path: Path,
    roles: object,
    error_type: type[Exception],
    message: str,
) -> None:
    wire = _Wire()
    path = _write(
        tmp_path / "roles.rds", _seurat(wire, reductions={"pca": _dimreduc(wire)})
    )
    with pytest.raises(error_type, match=message):
        SeuratReader(path, reduction_roles=roles)  # type: ignore[arg-type]


def _cite_seq_object(wire: _Wire) -> bytes:
    """RNA and ADT with per-assay PCAs and a WNN UMAP."""
    proteins = ["p1", "p2", "p3"]
    rna_loadings = wire.matrix(
        [0.1, 0.2, 0.3, 0.4], (2, 2), rows=_GENES, columns=["PC_1", "PC_2"], real=True
    )
    adt_loadings = wire.matrix(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        (3, 2),
        rows=proteins,
        columns=["apca_1", "apca_2"],
        real=True,
    )
    return _seurat(
        wire,
        assays={
            "RNA": _legacy(wire),
            "ADT": _legacy(
                wire, counts=_counts(wire, features=proteins), features=proteins
            ),
        },
        reductions={
            "pca": _dimreduc(wire, loadings=rna_loadings),
            "apca": _dimreduc(
                wire,
                assay_used="ADT",
                key="apca_",
                columns=["apca_1", "apca_2"],
                loadings=adt_loadings,
            ),
            "wnn.umap": _dimreduc(
                wire, key="wnnUMAP_", columns=["wnnUMAP_1", "wnnUMAP_2"]
            ),
        },
    )


def _skill_recipe(path: Path, zarr_loc: Path) -> object:
    """The Seurat recipe of the skill's data-access reference."""
    si = cytearc.inspect_seurat(path)
    print(si.activeAssay, [(a.name, a.importable) for a in si.assays])
    print([(r.name, r.importable, r.role, r.assayUsed) for r in si.reductions])
    # Every importable assay, so each reduction keeps its assay and loadings.
    assays = [a.name for a in si.assays if a.importable]
    reductions = [r.name for r in si.reductions if r.importable]
    # An "integrated" active assay holds no raw counts: name another default.
    default = None if si.activeAssay in assays else "RNA"
    with cytearc.SeuratReader(path, assays=assays, reductions=reductions) as sr:
        out = cytearc.SeuratToZarr(
            sr, zarr_loc=str(zarr_loc), default_assay=default
        ).dump()
    print(out.defaultAssay, dict(out.reductionArtifacts), out.activeIdentity)
    print([notice.code for notice in out.notices])
    return out


def test_the_skill_recipe_imports_cite_seq_and_integrated_objects(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    cite_seq = _write(tmp_path / "cite.rds", _cite_seq_object(wire))
    result = _skill_recipe(cite_seq, tmp_path / "cite.zarr")
    assert result.assayNames == ("RNA", "ADT")  # type: ignore[attr-defined]
    assert result.defaultAssay == "RNA"  # type: ignore[attr-defined]
    artifacts = result.reductionArtifacts  # type: ignore[attr-defined]
    assert (artifacts["apca"].kind, artifacts["apca"].assay) == (
        "imported_coordinates",
        "ADT",
    )
    assert (artifacts["wnn.umap"].kind, artifacts["wnn.umap"].assay) == (
        "embedding",
        "RNA",
    )
    root = zarr.open_group(str(tmp_path / "cite.zarr"), mode="r")
    apca = artifact_group(root, artifacts["apca"])
    assert apca["feature_ids"][:].tolist() == ["p1", "p2", "p3"]
    # The loadings are the R doubles themselves.
    np.testing.assert_array_equal(
        apca["loadings"][:], [[0.5, 0.8], [0.6, 0.9], [0.7, 1.0]]
    )

    integrated = _write(tmp_path / "integrated.rds", _integrated_object(wire))
    result = _skill_recipe(integrated, tmp_path / "integrated.zarr")
    assert result.assayNames == ("RNA",)  # type: ignore[attr-defined]
    assert result.defaultAssay == "RNA"  # type: ignore[attr-defined]
    assert set(result.reductionArtifacts) == {"pca", "umap"}  # type: ignore[attr-defined]


def _sct_object(
    wire: _Wire, umi_assays: tuple[str | None, ...] = ("RNA", "RNA")
) -> bytes:
    def model(umi_assay: str | None) -> bytes:
        slots = [("model", wire.string_vector(["y ~ log_umi"]))]
        if umi_assay is not None:
            slots.append(("umi.assay", wire.string_vector([umi_assay])))
        slots.append(("class", wire.string_vector(["SCTModel"])))
        return wire.s4(slots)

    models = wire.vector(
        [model(umi) for umi in umi_assays],
        names=[f"model{index}" for index in range(len(umi_assays))],
    )
    extra = [("SCTModel.list", models)] if umi_assays else []
    return _seurat(
        wire,
        assays={
            "RNA": _legacy(wire),
            "SCT": _legacy(
                wire,
                counts=_counts(wire, values=[2, 2, 2, 2, 2, 2]),
                source_class="SCTAssay",
                extra=extra,
            ),
        },
        active="SCT",
    )


def test_sct_counts_are_flagged_and_the_raw_counts_become_the_default(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "sct.rds", _sct_object(wire))
    with SeuratReader(path) as reader:
        sct = reader.get_assay("SCT")
        assert sct.umiAssay == "RNA"
        assert (sct.notices[0].code, sct.notices[0].objectPath) == (
            "sct_corrected_counts",
            "assays/SCT/counts",
        )
        assert sct.notices[0].context == {"umiAssay": "RNA"}
        assert reader.get_assay("RNA").umiAssay is None
        destination = MemoryStore()
        result = _writer(reader, destination).dump()
    assert result.defaultAssay == "RNA"
    assert result.activeIdentity.assay == "RNA"
    assert _notices(result.notices, "sct_default_assay") == [
        ("active.assay", {"activeAssay": "SCT", "defaultAssay": "RNA"})
    ]
    assert zarr.open_group(store=destination, mode="r").attrs["defaultAssay"] == "RNA"

    # The SCT assay stays the default when asked for, or when RNA is not imported.
    with SeuratReader(path) as reader:
        result = _writer(reader, MemoryStore(), default_assay="SCT").dump()
    assert result.defaultAssay == "SCT"
    assert _notices(result.notices, "sct_default_assay") == []
    with SeuratReader(path, assays=["SCT"]) as reader:
        assert _writer(reader, MemoryStore()).defaultAssay == "SCT"


@pytest.mark.parametrize("umi_assays", [("RNA", "ADT"), ("RNA", None), ()])
def test_sct_without_one_recorded_umi_assay_keeps_the_active_default(
    tmp_path: Path,
    umi_assays: tuple[str | None, ...],
) -> None:
    wire = _Wire()
    path = _write(tmp_path / "sct.rds", _sct_object(wire, umi_assays))
    with SeuratReader(path) as reader:
        sct = reader.get_assay("SCT")
        assert sct.umiAssay is None
        assert sct.notices[0].context == {"umiAssay": None}
        assert _writer(reader, MemoryStore()).defaultAssay == "SCT"


def test_reductions_of_some_cells_or_another_order_align_to_the_cells(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    cells = ["c1", "c2", "c3", "c4"]
    path = _write(
        tmp_path / "sketch.rds",
        _seurat(
            wire,
            assays={"RNA": _legacy(wire, counts=_counts(wire, cells=cells))},
            cells=cells,
            reductions={
                # Reductions of a v5 sketch: some cells, in their own order.
                "pca.sketch": _dimreduc(
                    wire, rows=["c3", "c1"], values=[1.0, 2.0, 3.0, 4.0]
                ),
                "umap.sketch": _dimreduc(
                    wire,
                    rows=["c4", "c2"],
                    values=[5.0, 6.0, 7.0, 8.0],
                    key="UMAP_",
                ),
                "pca.full": _dimreduc(
                    wire,
                    rows=["c4", "c3", "c2", "c1"],
                    values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ),
            },
        ),
    )
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        positions = {
            name: reader.get_reduction(name).cellPositions.tolist()  # type: ignore[union-attr]
            for name in ("pca.sketch", "umap.sketch", "pca.full")
        }
        assert positions == {
            "pca.sketch": [2, 0],
            "umap.sketch": [3, 1],
            "pca.full": [3, 2, 1, 0],
        }
        assert _notices(
            reader.get_reduction("pca.sketch").notices, "aligned_reduction_cells"
        ) == [
            (
                "reductions/pca.sketch/cell.embeddings/Dimnames/0",
                {"cells": 2, "objectCells": 4},
            )
        ]
        result = _writer(reader, destination).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    sketch = result.reductionArtifacts["pca.sketch"]
    umap = result.reductionArtifacts["umap.sketch"]
    full = result.reductionArtifacts["pca.full"]
    # Rows follow the object's cells: c1 then c3, and c2 then c4.
    np.testing.assert_array_equal(
        artifact_group(root, sketch)["data"][:], [[2.0, 4.0], [1.0, 3.0]]
    )
    np.testing.assert_array_equal(
        artifact_group(root, umap)["values"][:], [[6.0, 8.0], [5.0, 7.0]]
    )
    np.testing.assert_array_equal(
        artifact_group(root, full)["data"][:],
        [[4.0, 8.0], [3.0, 7.0], [2.0, 6.0], [1.0, 5.0]],
    )
    for ref, selected in (
        (sketch, [True, False, True, False]),
        (umap, [False, True, False, True]),
    ):
        selection = inspect_artifact(root, ref).input_ref("cell_selection")
        assert inspect_artifact(root, selection).operation == "import_reduction_cells"
        assert artifact_group(root, selection)["values"][:].tolist() == selected
    assert (
        inspect_artifact(root, full).input_ref("cell_selection") == result.cellSelection
    )

    ds = DataStore(destination, min_features_per_cell=0, nthreads=1, mem_budget="64M")
    values = ds.artifacts.load_values(umap)
    assert values.cell_ids.tolist() == ["c2", "c4"]
    np.testing.assert_array_equal(values.values, [[6.0, 8.0], [5.0, 7.0]])


def test_assay5_slots_report_data_only_when_they_hold_it(tmp_path: Path) -> None:
    wire = _Wire()
    assay = wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [wire.matrix([1, 2, 3, 4, 5, 6], (2, 3))], names=["counts"]
                ),
            ),
            ("cells", wire.logmap([1, 1, 1], _CELLS, ["counts"])),
            ("features", wire.logmap([1, 1], _GENES, ["counts"])),
            ("meta.data", wire.data_frame([], 2)),
            ("key", wire.string_vector(["rna_"])),
            ("default", wire.integer_vector([1])),
            # R stores NULL in an S4 slot as a placeholder symbol.
            ("assay.orig", wire.symbol("\x01NULL\x01")),
            ("misc", wire.vector([wire.integer_vector([1])], names=["x"])),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    path = _write(tmp_path / "assay5-slots.rds", _seurat(wire, assays={"RNA": assay}))

    with SeuratReader(path) as reader:
        rna = reader.get_assay("RNA")
        assert [
            (notice.code, notice.objectPath, notice.context) for notice in rna.notices
        ] == [("ignored_assay_slot", "assays/RNA/misc", {"rType": "VECTOR"})]
        np.testing.assert_array_equal(
            rna.counts.read_cells(0, 3).toarray(), [[1, 2], [3, 4], [5, 6]]
        )


def test_assay5_cells_in_another_order_are_placed_by_name(tmp_path: Path) -> None:
    wire = _Wire()
    # Two layers over the assay's cells c3, c2, c1, which reverse the object's.
    assay = wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [
                        wire.matrix([1, 4, 2, 5], (2, 2)),
                        wire.matrix([3, 6], (2, 1)),
                    ],
                    names=["counts.1", "counts.2"],
                ),
            ),
            (
                "cells",
                wire.logmap(
                    [1, 1, 0, 0, 0, 1], ["c3", "c2", "c1"], ["counts.1", "counts.2"]
                ),
            ),
            (
                "features",
                wire.logmap([1, 1, 1, 1], _GENES, ["counts.1", "counts.2"]),
            ),
            ("meta.data", wire.data_frame([], 2)),
            ("key", wire.string_vector(["rna_"])),
            ("default", wire.integer_vector([2])),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    path = _write(tmp_path / "reordered.rds", _seurat(wire, assays={"RNA": assay}))
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        rna = reader.get_assay("RNA")
        # The structural key and default slots are not reported as data.
        assert rna.notices == ()
        np.testing.assert_array_equal(
            rna.counts.read_cells(0, 3).toarray(), [[3, 6], [2, 5], [1, 4]]
        )
        assert rna.cellMembership.allIncluded
        _writer(reader, destination).dump()
    np.testing.assert_array_equal(
        zarr.open_group(store=destination, mode="r")["RNA/counts"][:],
        [[3, 6], [2, 5], [1, 4]],
    )


def test_empty_s4_slots_and_metadata_slots_are_not_reported() -> None:
    fixtures = Path(__file__).resolve().parent / "datasets"
    inspection = cytearc.inspect_seurat(fixtures / "seurat_v4_1_3_pbmc_mye.rds")
    assert [
        (notice.code, notice.objectPath) for notice in inspection.assay("RNA").notices
    ] == [
        ("ignored_normalized_layer", "assays/RNA/data"),
        ("ignored_normalized_layer", "assays/RNA/scale.data"),
        ("ignored_assay_slot", "assays/RNA/var.features"),
    ]
    assert [
        (notice.code, notice.objectPath)
        for notice in inspection.reduction("pca").notices
    ] == [
        ("ignored_reduction_slot", "reductions/pca/misc"),
    ]
    assert inspection.reduction("umap").notices == ()


def test_s4_slots_report_data_only_when_some_slot_holds_it(tmp_path: Path) -> None:
    wire = _Wire()

    def jackstraw(scores: list[float]) -> bytes:
        return wire.s4(
            [
                ("empirical.p.values", wire.matrix([], (0, 0), real=True)),
                (
                    "fake.reduction.scores",
                    wire.matrix(scores, (len(scores), 1 if scores else 0), real=True),
                ),
                ("overall.p.values", wire.matrix([], (0, 0), real=True)),
                ("class", wire.string_vector(["JackStrawData"])),
            ]
        )

    assay = _legacy(
        wire,
        extra=[
            # R stores NULL in an S4 slot as a placeholder symbol.
            ("assay.orig", wire.symbol("\x01NULL\x01")),
            ("key", wire.string_vector(["rna_"])),
            ("misc", wire.vector([wire.integer_vector([1])], names=["x"])),
        ],
    )
    path = _write(
        tmp_path / "slots.rds",
        _seurat(
            wire,
            assays={"RNA": assay},
            reductions={
                "empty": _dimreduc(wire, extra=[("jackstraw", jackstraw([]))]),
                "full": _dimreduc(wire, extra=[("jackstraw", jackstraw([0.5, 0.25]))]),
            },
        ),
    )
    with SeuratReader(path) as reader:
        assert [
            (notice.code, notice.objectPath, notice.context)
            for notice in reader.get_assay("RNA").notices
        ] == [("ignored_assay_slot", "assays/RNA/misc", {"rType": "VECTOR"})]
        assert reader.get_reduction("empty").notices == ()
        assert [
            (notice.code, notice.objectPath)
            for notice in reader.get_reduction("full").notices
        ] == [("ignored_reduction_slot", "reductions/full/jackstraw")]


def test_reduction_rows_equal_to_the_cells_skip_their_uniqueness_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _Wire()
    path = _write(
        tmp_path / "rows.rds",
        _seurat(
            wire,
            reductions={
                "pca": _dimreduc(wire),
                "reordered": _dimreduc(wire, rows=["c3", "c2", "c1"]),
            },
        ),
    )
    validated: list[str] = []
    original = seurat_module._validate_unique_ids

    def recorded(values: object, *, object_path: str, **options: object) -> None:
        validated.append(object_path)
        original(values, object_path=object_path, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(seurat_module, "_validate_unique_ids", recorded)
    with SeuratReader(path) as reader:
        assert reader.inspection.reduction("pca").importable
    reductions = [path for path in validated if path.startswith("reductions/")]
    # The cell IDs were checked once, as meta.data row names; rows equal to them
    # need no second check, while reordered rows do.
    assert "meta.data/row.names" in validated
    assert reductions == [
        "reductions/pca/cell.embeddings/Dimnames/1",
        "reductions/reordered/cell.embeddings/Dimnames/0",
        "reductions/reordered/cell.embeddings/Dimnames/1",
    ]


def test_malformed_attributes_fail_as_malformed_input_not_a_type_error(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    column = wire.string_vector(["a", "b", "c"])
    payload = _seurat(wire, metadata=wire.data_frame([("group", column)], _CELLS))
    # The fuzzer's mutation: an attribute pairlist header becomes ALTREP (0xee).
    attributed = wire.string_vector(
        ["a", "b", "c"], attributes=[("note", wire.string_vector(["x"]))]
    )
    payload = payload.replace(column, attributed, 1)
    offset = payload.index(attributed) + len(column)
    mutated = payload[:offset] + struct.pack(">i", 0xEE) + payload[offset + 4 :]
    path = _write(tmp_path / "mutated.rds", mutated)
    with pytest.raises(
        RdsFormatError, match="attributes must be a pairlist, found ALTREP"
    ):
        SeuratReader(path)
    with pytest.raises(
        RdsFormatError, match="attributes must be a pairlist, found ALTREP"
    ):
        cytearc.inspect_seurat(io.BytesIO(mutated))


def test_converted_feature_metadata_follows_the_count_features(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    # meta.features lists g2 before g1, so each value is read by position.
    features = wire.data_frame(
        [
            ("score", _deferred(wire, wire.real_vector([2.5, 1e5]))),
            ("day", _classed(wire, RType.REAL, [1.0, 0.0], ["Date"])),
            ("stamp", _classed(wire, RType.REAL, [60.0, 0.5], ["POSIXct", "POSIXt"])),
        ],
        ["g2", "g1"],
    )
    path = _write(
        tmp_path / "features.rds",
        _seurat(wire, assays={"RNA": _legacy(wire, meta_features=features)}),
    )
    with SeuratReader(path) as reader:
        metadata = reader.get_assay("RNA").featureMetadata
        assert metadata.column("score").read_block(0, 2).values == ("1e+05", "2.5")
        assert metadata.column("day").read_block(0, 2).values == (
            "1970-01-01",
            "1970-01-02",
        )
        assert metadata.column("stamp").read_block(0, 2).values == (
            "1970-01-01T00:00:00.500000Z",
            "1970-01-01T00:01:00Z",
        )
        assert [notice.code for notice in reader.get_assay("RNA").notices] == [
            "formatted_deferred_strings",
            "converted_temporal_metadata",
            "converted_temporal_metadata",
        ]


def test_unreadable_keys_units_and_deferred_arguments_degrade_to_notices(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    metadata = wire.data_frame(
        [
            # Units that are not one string are left out of the notice.
            (
                "elapsed",
                _classed(
                    wire,
                    RType.REAL,
                    [1.0, 2.0, 3.0],
                    ["difftime"],
                    [("units", wire.integer_vector([1]))],
                ),
            ),
            # R defers as.character() only for integers and doubles.
            ("flags", _deferred(wire, wire.logical_vector([1, 0, 1]))),
        ],
        _CELLS,
    )
    # A DimReduc key that is not text leaves the name to decide the role.
    umap = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    [1.0] * 6, (3, 2), rows=_CELLS, columns=["A", "B"], real=True
                ),
            ),
            ("assay.used", wire.string_vector(["RNA"])),
            ("key", wire.integer_vector([1])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    path = _write(
        tmp_path / "degraded.rds",
        _seurat(wire, metadata=metadata, reductions={"umap": umap}),
    )
    with SeuratReader(path) as reader:
        assert reader.cellMetadata.columnNames == ("elapsed",)
        assert [
            (notice.code, notice.context) for notice in reader.cellMetadata.notices
        ] == [
            ("classed_metadata_column", {"classNames": ("difftime",)}),
            (
                "skipped_metadata_column",
                {
                    "column": "flags",
                    "reason": "unsupported_altrep",
                    "details": {"className": "deferred_string", "packageName": "base"},
                },
            ),
        ]
        reduction = reader.get_reduction("umap")
        assert (reduction.key, reduction.embeddingRole) == (None, "umap")


def test_a_reduction_of_an_assay_the_object_lacks_imports_its_coordinates(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write(
        tmp_path / "lacking.rds",
        _seurat(wire, reductions={"pca": _dimreduc(wire, assay_used="SCT")}),
    )
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        assert _notices(
            reader.get_reduction("pca").notices, "reduction_assay_unavailable"
        ) == [("reductions/pca/assay.used", {"assayUsed": "SCT", "reason": "missing"})]
        result = _writer(reader, destination).dump()
    assert result.reductionArtifacts["pca"].assay == "RNA"


@pytest.mark.parametrize(
    ("reductions", "cause"),
    [
        (
            lambda wire: wire.integer_vector([1]),
            "INTEGER is not a generic named vector",
        ),
        (
            lambda wire: wire.vector(
                [_dimreduc(wire)], attributes=[("names", wire.integer_vector([1]))]
            ),
            "named vector at $.attributes.cdr[4].car has no character names",
        ),
        (
            lambda wire: wire.vector([_dimreduc(wire), _dimreduc(wire)], names=["pca"]),
            "name count 1 does not match value count 2 at $.attributes.cdr[4].car",
        ),
    ],
)
def test_lists_without_one_name_per_value_are_rejected(
    tmp_path: Path,
    reductions: object,
    cause: str,
) -> None:
    wire = _Wire()
    payload = _seurat(wire).replace(
        wire.vector([], names=[]),
        reductions(wire),  # type: ignore[operator]
    )
    path = _write(tmp_path / "lists.rds", payload)
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)
    assert (error.value.code, error.value.objectPath, error.value.context) == (
        "invalid_named_list",
        "reductions",
        {"cause": cause},
    )


def test_cell_id_views_read_their_positions_in_runs() -> None:
    from cytearc.writers.seurat import _CellIdsAt, _consecutive_runs

    assert list(_consecutive_runs(np.array([2, 3, 4, 7, 9, 10]))) == [
        (0, 2, 5),
        (3, 7, 8),
        (4, 9, 11),
    ]
    ids = _CellIdsAt(("a", "b", "c", "d", "e"), np.array([1, 2, 4]))
    assert len(ids) == 3
    assert ids[2] == "e"
    assert ids[0:3] == ("b", "c", "e")
    assert ids[3:3] == ()
