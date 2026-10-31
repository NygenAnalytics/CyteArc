"""Long tables of records: parsing, text stored once per distinct value, ragged
values and the encoder that merges chunks into one table ordered by cell.

The contig encoding builds on these with its own fields; its tests are in
``tests/test_repertoire_contigs.py``. Here a schema of lineage barcode
molecules, which has nothing of a contig, checks that the encoder is generic.
"""

import numpy as np
import pandas as pd
import pytest

from cytearc.metadata import records as records_module
from cytearc.metadata.records import (
    FLAG,
    INTEGER,
    NUMBER,
    RAGGED,
    TEXT,
    EncodedText,
    RecordEncoder,
    RecordTable,
    chunk_rows_for,
    decode_ragged,
    decode_text,
    encode_ragged,
    encode_text,
    file_lines,
    pack_text,
    parse_columns,
    parse_flags,
    parse_integers,
    parse_numbers,
    parse_text,
    take_ragged,
    take_text,
    unpack_text,
)
from tests.perturbation_helpers import use_python_kernels

# Molecules of a lineage barcode library: a molecule ID, the library and the
# barcode it reads, its reads, whether it passed a filter, and a quality.
MOLECULE_FIELDS = {
    "molecule": RAGGED,
    "library": TEXT,
    "barcode": TEXT,
    "reads": INTEGER,
    "kept": FLAG,
    "quality": NUMBER,
}


def _molecules(rows: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.Series(values, dtype=object)
            if MOLECULE_FIELDS.get(name) in (TEXT, RAGGED, None)
            else np.asarray(values)
            for name, values in rows.items()
        }
    )


def test_text_parsing_turns_missing_tokens_into_none() -> None:
    parsed = parse_text([" TRBV1 ", "", "None", "NA", "nan", "null", None, "x"])
    assert parsed.tolist() == ["TRBV1", None, None, None, None, None, None, "x"]
    assert parsed.dtype == object


def test_flags_accept_every_spelling_of_true_and_false() -> None:
    flags = parse_flags(
        [
            "T",
            "F",
            "true",
            "false",
            "True",
            "False",
            "TRUE",
            "FALSE",
            "1",
            "0",
            "",
            "None",
        ],
        name="productive",
    )
    assert flags.tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1, -1]
    assert flags.dtype == np.int8


def test_unknown_flags_are_refused_with_their_lines() -> None:
    with pytest.raises(
        ValueError, match=r"productive must be true or false.*'maybe' on lines 4, 6"
    ):
        parse_flags(["T", "F", "maybe", "T", "maybe"], name="productive", first_line=2)


def test_integers_parse_whole_numbers_and_mark_missing_values() -> None:
    parsed = parse_integers(["3", "", "0", " 12 ", "None", "4.0"], name="umi_count")
    assert parsed.tolist() == [3, -1, 0, 12, -1, 4]
    assert parsed.dtype == np.int64


@pytest.mark.parametrize("value", ["-1", "2.5", "many", "inf"])
def test_integers_refuse_values_that_are_not_counts(value) -> None:
    with pytest.raises(ValueError, match="umi_count must be a whole number"):
        parse_integers(["1", value], name="umi_count", first_line=10)


def test_integers_refuse_values_beyond_int64() -> None:
    with pytest.raises(ValueError, match="too large"):
        parse_integers(["1", str(2**64)], name="umi_count")


def test_numbers_parse_reals_and_refuse_non_finite_values() -> None:
    parsed = parse_numbers(["0.95", "", "1"], name="v_identity")
    np.testing.assert_array_equal(parsed, [0.95, np.nan, 1.0])
    with pytest.raises(ValueError, match="v_identity must be a finite number.*line 3"):
        parse_numbers(["0.9", "inf"], name="v_identity")
    with pytest.raises(ValueError, match="'x'"):
        parse_numbers(["x"], name="v_identity")


def test_line_numbers_name_at_most_five_lines() -> None:
    with pytest.raises(ValueError, match=r"lines 2, 3, 4, 5, 6 and 2 more"):
        parse_flags(["?"] * 7, name="is_cell")
    assert file_lines(np.asarray([0]), 7) == "line 7"


def test_text_encoding_sorts_distinct_values_and_marks_missing() -> None:
    encoded = encode_text(["b", None, "a", "b", float("nan")])
    assert encoded.categories.tolist() == ["a", "b"]
    assert encoded.codes.tolist() == [1, -1, 0, 1, -1]
    assert encoded.codes.dtype == np.int32
    assert decode_text(encoded.codes, encoded.categories).tolist() == [
        "b",
        None,
        "a",
        "b",
        None,
    ]
    empty = encode_text([None, None])
    assert empty.codes.tolist() == [-1, -1] and len(empty.categories) == 0


def test_ragged_encoding_round_trips_and_tells_missing_from_values() -> None:
    values = ["ACGT", None, "", "TTGCA", None]
    data, offsets = encode_ragged(values)
    # The empty string is a value of length 0, as distinct from a missing one.
    assert offsets.tolist() == [0, 4, -1, 4, 9, -1]
    assert bytes(data) == b"ACGTTTGCA"
    assert decode_ragged(data, offsets).tolist() == ["ACGT", None, "", "TTGCA", None]
    data, offsets = encode_ragged(["αβ"])
    assert decode_ragged(data, offsets).tolist() == ["αβ"]


def test_chunk_rows_follow_the_budget_within_bounds() -> None:
    assert chunk_rows_for(10**3, columns=50) == 10_000
    assert chunk_rows_for(10**12, columns=50) == 50_000
    assert chunk_rows_for(20 * 1_000 * 10 * 20_000, columns=10) == 20_000


def test_packed_text_round_trips_ascii_and_other_text() -> None:
    for values in (["CASSF", "", "TRBV1*01"], ["αβ", "γ", "CAS"], []):
        data, offsets = pack_text(values)
        assert unpack_text(data, offsets).tolist() == values
    data, offsets = pack_text(["αβ", "x"])
    assert offsets.tolist() == [0, 4, 5]


def test_some_values_of_packed_text_are_decoded_alone() -> None:
    data, offsets = pack_text(["CASSF", "CAVF", "CARW", "αβ"])
    assert take_text(data, offsets, np.asarray([3, 0, 0])).tolist() == [
        "αβ",
        "CASSF",
        "CASSF",
    ]
    assert take_text(data, offsets, np.asarray([], dtype=np.int64)).tolist() == []


@pytest.mark.parametrize("python_kernels", [False, True])
def test_taking_ragged_values_copies_only_the_chosen_ones(
    monkeypatch: pytest.MonkeyPatch, python_kernels: bool
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, records_module)
    data, offsets = encode_ragged(["AAA", None, "", "CCGG", "κύτταρο"])

    chosen, bounds = take_ragged(data, offsets, np.asarray([4, 1, 3, 1, 2]))

    assert decode_ragged(chosen, bounds).tolist() == [
        "κύτταρο",
        None,
        "CCGG",
        None,
        "",
    ]
    # Only the chosen values' bytes are copied, so AAA is left behind.
    assert len(chosen) == len("κύτταρο".encode()) + 4
    nothing, bounds = take_ragged(data, offsets, np.asarray([], dtype=np.int64))
    assert (len(nothing), bounds.tolist()) == (0, [0])


def test_columns_parse_by_kind_in_field_order_and_name_bad_lines() -> None:
    frame = pd.DataFrame(
        {
            "quality": ["0.5", ""],
            "unlisted": ["x", "y"],
            "kept": ["T", "F"],
            "barcode": [" ACGT ", "NA"],
            "reads": ["3", "None"],
        }
    )

    parsed = parse_columns(frame, MOLECULE_FIELDS, first_line=40)

    assert list(parsed) == ["barcode", "reads", "kept", "quality"]
    assert parsed["barcode"].tolist() == ["ACGT", None]
    assert parsed["reads"].tolist() == [3, -1]
    assert parsed["kept"].tolist() == [1, 0]
    np.testing.assert_array_equal(parsed["quality"], [0.5, np.nan])
    frame.loc[1, "reads"] = "-2"
    with pytest.raises(ValueError, match="reads must be a whole number.*line 41"):
        parse_columns(frame, MOLECULE_FIELDS, first_line=40)
    with pytest.raises(ValueError, match="Field 'reads' has the kind 'count'"):
        parse_columns(frame, {"reads": "count"}, first_line=40)  # type: ignore[dict-item]


@pytest.mark.parametrize("python_kernels", [False, True])
def test_an_encoder_of_any_fields_orders_records_by_cell_source_and_row(
    monkeypatch: pytest.MonkeyPatch, python_kernels: bool
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, records_module)
    encoder = RecordEncoder(MOLECULE_FIELDS, id_field="molecule")
    encoder.add(
        _molecules(
            {
                "molecule": ["m1", "m2", "m3"],
                "library": ["lib_b", "lib_b", None],
                "barcode": ["TTGA", "ACCA", "TTGA"],
                "reads": [4, 1, -1],
            }
        ),
        np.asarray([3, 0, 3]),
        source=1,
    )
    # The second chunk lacks reads and adds the quality of its molecules.
    encoder.add(
        _molecules(
            {
                "molecule": ["m1", None],
                "library": ["lib_a", "lib_a"],
                "barcode": ["GGGG", "ACCA"],
                "quality": [0.25, 0.5],
            }
        ),
        np.asarray([3, 1]),
        source=0,
    )
    assert (encoder.n_records, encoder.missing_ids) == (5, 1)
    # m1 repeats only across sources, which is allowed.
    assert encoder.repeated_ids() == []

    table = encoder.finish()

    # Cell 0, cell 1, then cell 3 with source 0 before source 1, rows kept.
    assert table.cells.tolist() == [0, 1, 3, 3, 3]
    assert table.sources.tolist() == [1, 0, 0, 1, 1]
    assert table.fields == ("molecule", "library", "barcode", "reads", "quality")
    assert dict(table.kinds) == {name: MOLECULE_FIELDS[name] for name in table.fields}
    assert table.n_records == 5
    assert table.column("molecule").tolist() == ["m2", None, "m1", "m1", "m3"]
    assert table.text["barcode"].categories.tolist() == ["ACCA", "GGGG", "TTGA"]
    assert table.text["barcode"].codes.tolist() == [0, 0, 1, 2, 2]
    assert table.column("library").tolist() == [
        "lib_b",
        "lib_a",
        "lib_a",
        "lib_b",
        None,
    ]
    assert table.column("reads").tolist() == [1, -1, -1, 4, -1]
    np.testing.assert_array_equal(
        table.column("quality"), [np.nan, 0.5, 0.25, np.nan, np.nan]
    )
    assert table.column_at("molecule", np.asarray([4, 0])).tolist() == ["m3", "m2"]
    assert table.column_at("barcode", np.asarray([3])).tolist() == ["TTGA"]
    assert table.column_at("reads", np.asarray([3, 3])).tolist() == [4, 4]
    for read in (table.column, lambda name: table.column_at(name, np.asarray([0]))):
        with pytest.raises(KeyError, match="holds no field 'kept'"):
            read("kept")


def test_many_chunks_merge_to_the_order_a_stable_sort_gives() -> None:
    # An invariant independent of the encoder: the records sorted stably by
    # cell and source, with text equal to what each chunk held.
    rng = np.random.default_rng(11)
    encoder = RecordEncoder({"barcode": TEXT, "umi": RAGGED, "reads": INTEGER})
    pieces = []
    for chunk_index in range(12):
        n_rows = int(rng.integers(0, 30))
        barcodes = rng.choice(["AC", "GT", "κα", "TTTT", None], size=n_rows)
        frame = pd.DataFrame(
            {
                "barcode": pd.Series(barcodes, dtype=object),
                "umi": pd.Series(
                    [f"u{chunk_index}_{row}" for row in range(n_rows)], dtype=object
                ),
                "reads": rng.integers(0, 9, size=n_rows),
            }
        )
        cells = rng.integers(0, 6, size=n_rows)
        source = int(rng.integers(0, 3))
        encoder.add(frame, cells, source=source)
        pieces.append(frame.assign(cell=cells, source=source))
    expected = pd.concat(pieces, ignore_index=True).sort_values(
        ["cell", "source"], kind="stable"
    )

    table = encoder.finish()

    assert table.cells.tolist() == expected["cell"].tolist()
    assert table.sources.tolist() == expected["source"].tolist()
    assert table.column("umi").tolist() == expected["umi"].tolist()
    assert table.column("barcode").tolist() == [
        None if pd.isna(value) else value for value in expected["barcode"]
    ]
    assert table.column("reads").tolist() == expected["reads"].tolist()
    categories = table.text["barcode"].categories.tolist()
    assert categories == sorted(categories)


def test_a_chunk_without_the_id_field_holds_records_without_ids() -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS, id_field="molecule")
    encoder.add(_molecules({"barcode": ["AC", "GT"]}), np.asarray([0, 1]), source=0)
    encoder.add(_molecules({"molecule": ["m1"]}), np.asarray([2]), source=0)
    assert encoder.missing_ids == 2
    assert encoder.repeated_ids() == []


def test_empty_chunks_still_name_their_fields() -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS)
    encoder.add(
        _molecules({"barcode": [], "reads": []}), np.empty(0, np.int64), source=0
    )
    table = encoder.finish()
    assert table.fields == ("barcode", "reads")
    assert table.n_records == 0
    assert table.column("barcode").tolist() == []


def test_an_encoder_without_an_id_field_has_no_ids_to_check() -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS)
    encoder.add(_molecules({"molecule": ["m1", "m1"]}), np.asarray([0, 0]), source=0)
    with pytest.raises(ValueError, match="has no id_field"):
        encoder.repeated_ids()
    with pytest.raises(ValueError, match="has no id_field"):
        _ = encoder.missing_ids


@pytest.mark.parametrize(
    ("fields", "id_field", "error", "message"),
    [
        (
            {"barcode": "codes"},
            None,
            ValueError,
            "Field 'barcode' has the kind 'codes'",
        ),
        ([("barcode", TEXT)], None, TypeError, "fields must map each field"),
        (
            MOLECULE_FIELDS,
            "barcode",
            ValueError,
            "id_field 'barcode' must name a ragged",
        ),
        (
            MOLECULE_FIELDS,
            "cell_id",
            ValueError,
            "id_field 'cell_id' must name a ragged",
        ),
    ],
)
def test_an_encoder_refuses_fields_it_cannot_encode(
    fields, id_field, error, message
) -> None:
    with pytest.raises(error, match=message):
        RecordEncoder(fields, id_field=id_field)


def test_encoded_text_keeps_categories_in_the_order_given() -> None:
    encoded = EncodedText.from_categories([1, -1, 0], ["zeta", "alpha"])
    assert encoded.categories.tolist() == ["zeta", "alpha"]
    assert decode_text(encoded.codes, encoded.categories).tolist() == [
        "alpha",
        None,
        "zeta",
    ]


def test_text_parsing_reads_pandas_missing_tokens_as_missing() -> None:
    # The tokens that pandas reads as missing by default, which scirpy's
    # reader of Cell Ranger files sees as missing, and text that is not one.
    tokens = ["N/A", "n/a", "#N/A", "<NA>", "#NA", "-nan", "NULL", "none"]
    assert parse_text([*tokens, "NAN", "na", "N/A2"]).tolist() == [None] * 8 + [
        "NAN",
        "na",
        "N/A2",
    ]


@pytest.mark.parametrize(
    ("rows", "cells", "source", "error", "message"),
    [
        ({"reads": [1.0, np.nan]}, [0, 1], 0, TypeError, "'reads' holds integers"),
        ({"reads": [3, -7]}, [0, 1], 0, ValueError, "at least 0, with -1"),
        ({"kept": [2, 0]}, [0, 1], 0, ValueError, "'kept' holds flags"),
        ({"kept": [-3, 1]}, [0, 1], 0, ValueError, "'kept' holds flags"),
        ({"kept": [0.5, 1.0]}, [0, 1], 0, TypeError, "'kept' holds flags"),
        ({"quality": ["high", "low"]}, [0, 1], 0, TypeError, "real numbers"),
        ({"quality": [np.inf, 0.5]}, [0, 1], 0, ValueError, "must be finite"),
        ({"barcode": ["AC", 5]}, [0, 1], 0, TypeError, "'barcode' must be text"),
        ({"molecule": ["m1", 3]}, [0, 1], 0, TypeError, "'molecule' must be text"),
        ({"molecule": ["m1", ["m2"]]}, [0, 1], 0, TypeError, "holds \\['m2'\\]"),
        ({"barcode": ["AC", "GT"]}, [0, -5], 0, ValueError, "at least 0, but hold -5"),
        ({"barcode": ["AC", "GT"]}, [0.0, 1.0], 0, TypeError, "as integers"),
        ({"barcode": ["AC", "GT"]}, [0, 1], -1, ValueError, "at least 0, not -1"),
        ({"barcode": ["AC", "GT"]}, [0, 1], True, TypeError, "position of a file"),
    ],
)
def test_an_encoder_refuses_values_of_the_wrong_kind_and_stays_unchanged(
    rows, cells, source, error, message
) -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS, id_field="molecule")
    encoder.add(_molecules({"molecule": ["m0"], "barcode": ["TT"]}), [0], source=0)
    before = (encoder.n_records, encoder.nbytes, encoder.missing_ids)
    # Each bad chunk also holds valid fields, which a refusal must not keep.
    frame = _molecules({"library": ["lib_a", "lib_a"], **rows})
    with pytest.raises(error, match=message):
        encoder.add(frame, np.asarray(cells), source=source)
    assert (encoder.n_records, encoder.nbytes, encoder.missing_ids) == before
    assert encoder.finish().fields == ("molecule", "barcode")


def test_an_encoder_takes_integers_of_any_width_and_boolean_flags() -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS)
    encoder.add(
        pd.DataFrame(
            {
                "reads": np.asarray([7, -1], dtype=np.int32),
                "kept": np.asarray([True, False]),
                "quality": np.asarray([3, 4], dtype=np.int16),
            }
        ),
        np.asarray([1, 0], dtype=np.uint8),
        source=np.int64(2),
    )
    table = encoder.finish()
    assert table.column("reads").tolist() == [-1, 7]
    assert table.column("kept").tolist() == [0, 1]
    assert table.column("quality").tolist() == [4.0, 3.0]
    assert table.sources.tolist() == [2, 2]


def _hand_built_table(**changes: object) -> RecordTable:
    """Records built without an encoder, ordered by cell, barcode and UMI."""
    barcode = encode_text(["AAAA", "AAAA", "CCCC"])
    parts: dict[str, object] = {
        "cells": np.asarray([0, 2, 2], dtype=np.int64),
        "sources": np.asarray([1, 0, 0], dtype=np.int32),
        "kinds": {"barcode": TEXT, "umi": RAGGED, "umis": INTEGER},
        "text": {"barcode": barcode},
        "ragged": {"umi": encode_ragged(["u1", None, "u3"])},
        "values": {"umis": np.asarray([3, 1, -1], dtype=np.int64)},
    }
    parts.update(changes)
    return RecordTable(**parts)  # type: ignore[arg-type]


def test_a_table_built_by_hand_in_its_own_order_is_accepted() -> None:
    table = _hand_built_table()
    assert table.fields == ("barcode", "umi", "umis")
    assert table.column("barcode").tolist() == ["AAAA", "AAAA", "CCCC"]
    assert table.column("umi").tolist() == ["u1", None, "u3"]
    empty = RecordTable(
        cells=np.empty(0, np.int64),
        sources=np.empty(0, np.int32),
        kinds={},
        text={},
        ragged={},
        values={},
    )
    assert empty.n_records == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cells": np.asarray([2, 0, 2])}, "ascending order"),
        ({"cells": np.asarray([-1, 0, 2])}, "ascending order"),
        ({"sources": np.asarray([0, 0])}, "one cell and one source per record"),
        ({"cells": np.zeros((3, 1), np.int64)}, "one cell and one source"),
        ({"kinds": {"barcode": TEXT, "umi": RAGGED, "umis": "count"}}, "kind 'count'"),
        (
            {"kinds": {"barcode": TEXT, "umis": INTEGER}},
            "ragged must hold the fields none",
        ),
        ({"values": {}}, "values must hold the fields 'umis'"),
        ({"text": {"barcode": encode_text(["AAAA"])}}, "one code per record"),
        (
            {"text": {"barcode": EncodedText.from_categories([0, 1, 2], ["A", "C"])}},
            "one of its 2 values",
        ),
        ({"ragged": {"umi": encode_ragged(["u1"])}}, "one offset per record"),
        ({"values": {"umis": np.asarray([3, 1])}}, "'umis' needs one value per record"),
    ],
)
def test_a_table_refuses_arrays_that_do_not_fit_its_records(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _hand_built_table(**changes)


def test_ids_and_text_are_encoded_without_a_copy_padded_to_the_longest_value() -> None:
    import tracemalloc

    # A copy padded to the one long value would take 2,000 rows of 10,000
    # characters, 80 MB; the values themselves take about 10 MB.
    values = [f"m{row}" for row in range(1_999)] + ["x" * 10_000]
    frame = pd.DataFrame({"molecule": pd.Series(values, dtype=object)})
    tracemalloc.start()
    try:
        RecordEncoder(MOLECULE_FIELDS, id_field="molecule").add(
            frame, np.arange(len(values)), source=0
        )
        encode_text(values)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 20 * 2**20


def test_an_encoder_counts_the_bytes_of_every_array_it_holds() -> None:
    encoder = RecordEncoder(MOLECULE_FIELDS, id_field="molecule")
    encoder.add(
        _molecules(
            {
                "barcode": ["AAAA", "AAAA", "CC", None],
                "molecule": ["m1", "m22", None, "m4"],
                "reads": [1, 2, 3, 4],
            }
        ),
        np.arange(4),
        source=0,
    )
    # Derived by hand: four cells as int64; four barcode codes as int32 with
    # the distinct values AAAA and CC packed in 6 bytes and 3 int64 offsets;
    # the IDs' 7 bytes with 5 int64 offsets and 4 uint64 hashes; four reads.
    cells = 4 * 8
    barcode = 4 * 4 + 6 + 3 * 8
    molecule = 7 + 5 * 8 + 4 * 8
    reads = 4 * 8
    assert encoder.nbytes == cells + barcode + molecule + reads == 189
