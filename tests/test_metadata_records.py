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
