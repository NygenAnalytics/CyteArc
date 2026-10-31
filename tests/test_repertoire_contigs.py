"""The contig encoding: AIRR fields encoded as records, identified by contig ID."""

import numpy as np
import pandas as pd
import pytest

from cytearc.metadata import records as records_module
from cytearc.metadata.records import FLAG, INTEGER, NUMBER, RAGGED, TEXT
from cytearc.repertoire.contigs import FIELDS, REQUIRED_FIELDS, contig_encoder
from tests.perturbation_helpers import use_python_kernels


def _chunk(rows: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.Series(values, dtype=object)
            if FIELDS[name] in ("text", "ragged")
            else np.asarray(values)
            for name, values in rows.items()
        }
    )


def test_the_fields_are_airr_names_of_known_kinds() -> None:
    assert set(FIELDS.values()) == {TEXT, RAGGED, INTEGER, FLAG, NUMBER}
    assert set(REQUIRED_FIELDS) <= set(FIELDS)
    # A contig's ID tells it apart from every other contig of its file.
    assert FIELDS["sequence_id"] == RAGGED
    with pytest.raises(TypeError):
        FIELDS["cell_id"] = TEXT  # type: ignore[index]


@pytest.mark.parametrize("python_kernels", [False, True])
def test_encoder_orders_contigs_by_cell_source_and_row(
    monkeypatch: pytest.MonkeyPatch, python_kernels: bool
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, records_module)
    encoder = contig_encoder()
    encoder.add(
        _chunk(
            {
                "sequence_id": ["s1_c1", "s1_c2", "s1_c3"],
                "locus": ["TRB", "TRA", None],
                "junction_aa": ["CASSF", "CAVF", "CAVF"],
                "productive": np.asarray([1, 1, 0], dtype=np.int8),
                "umi_count": np.asarray([5, -1, 2], dtype=np.int64),
                "sequence": ["AAA", None, "GGGG"],
            }
        ),
        np.asarray([2, 0, 2]),
        source=1,
    )
    # The second chunk lacks umi_count and sequence and adds v_identity.
    encoder.add(
        _chunk(
            {
                "sequence_id": ["s0_c1"],
                "locus": ["IGH"],
                "junction_aa": ["CARW"],
                "productive": np.asarray([1], dtype=np.int8),
                "v_identity": np.asarray([0.97]),
            }
        ),
        np.asarray([2]),
        source=0,
    )
    assert encoder.n_records == 4
    assert encoder.nbytes > 0
    table = encoder.finish()
    # Cell 0 first, then cell 2 with source 0 before source 1, file rows kept.
    assert table.cells.tolist() == [0, 2, 2, 2]
    assert table.sources.tolist() == [1, 0, 1, 1]
    assert table.fields == (
        "sequence_id",
        "locus",
        "junction_aa",
        "productive",
        "umi_count",
        "v_identity",
        "sequence",
    )
    assert table.column("sequence_id").tolist() == ["s1_c2", "s0_c1", "s1_c1", "s1_c3"]
    assert table.column("locus").tolist() == ["TRA", "IGH", "TRB", None]
    # Codes follow the sorted distinct values.
    assert table.text["junction_aa"].categories.tolist() == ["CARW", "CASSF", "CAVF"]
    assert table.text["junction_aa"].codes.tolist() == [2, 0, 1, 2]
    assert table.column("umi_count").tolist() == [-1, -1, 5, 2]
    np.testing.assert_array_equal(
        table.column("v_identity"), [np.nan, 0.97, np.nan, np.nan]
    )
    assert table.column("sequence").tolist() == [None, None, "AAA", "GGGG"]
    assert table.column("productive").tolist() == [1, 1, 1, 0]
    assert table.n_records == 4
    with pytest.raises(KeyError, match="holds no field 'c_call'"):
        table.column("c_call")


def test_encoder_without_chunks_gives_an_empty_table() -> None:
    table = contig_encoder().finish()
    assert table.n_records == 0
    assert table.fields == ()


def test_encoder_needs_one_cell_per_contig() -> None:
    encoder = contig_encoder()
    with pytest.raises(ValueError, match="one cell per record"):
        encoder.add(_chunk({"sequence_id": ["a", "b"]}), np.asarray([0]), source=0)


def test_encoder_reuses_codes_of_text_seen_in_earlier_chunks() -> None:
    encoder = contig_encoder()
    for cells, values in (([0, 1], ["CAVF", "CASS"]), ([2, 3], ["CASS", "CAVF"])):
        encoder.add(
            _chunk({"junction_aa": values}),
            np.asarray(cells),
            source=0,
        )
    table = encoder.finish()
    assert table.text["junction_aa"].categories.tolist() == ["CASS", "CAVF"]
    assert table.text["junction_aa"].codes.tolist() == [1, 0, 0, 1]


def test_repeated_and_missing_ids_are_found_within_each_source() -> None:
    encoder = contig_encoder()
    assert encoder.repeated_ids() == []
    encoder.add(
        _chunk({"sequence_id": ["a", "b", None]}), np.asarray([0, 1, 2]), source=0
    )
    encoder.add(_chunk({"sequence_id": ["c", "a"]}), np.asarray([3, 4]), source=0)
    encoder.add(_chunk({"sequence_id": ["a", "d"]}), np.asarray([0, 1]), source=1)
    assert encoder.missing_ids == 1
    assert encoder.repeated_ids() == [(0, "a")]
    clean = contig_encoder()
    clean.add(_chunk({"sequence_id": ["a", "b"]}), np.asarray([0, 1]), source=0)
    clean.add(_chunk({"sequence_id": ["a"]}), np.asarray([2]), source=1)
    assert clean.repeated_ids() == []


def test_hash_collisions_are_not_reported_as_repeats(monkeypatch) -> None:
    # Every ID hashes to the same value, so only the text tells them apart.
    monkeypatch.setattr(
        records_module.pd.util,
        "hash_array",
        lambda values, **options: np.zeros(len(values), dtype=np.uint64),
    )
    encoder = contig_encoder()
    encoder.add(_chunk({"sequence_id": ["a", "b", "c", "b"]}), np.arange(4), source=0)
    assert encoder.repeated_ids() == [(0, "b")]
    other = contig_encoder()
    other.add(_chunk({"sequence_id": ["a", "b"]}), np.arange(2), source=0)
    assert other.repeated_ids() == []


def test_text_of_every_missing_value_merges_to_no_categories() -> None:
    encoder = contig_encoder()
    encoder.add(_chunk({"c_call": [None, None]}), np.asarray([0, 1]), source=0)
    encoder.add(_chunk({"c_call": ["IGHM"]}), np.asarray([2]), source=0)
    table = encoder.finish()
    assert table.text["c_call"].codes.tolist() == [-1, -1, 0]
    assert table.text["c_call"].categories.tolist() == ["IGHM"]


def test_a_text_field_missing_everywhere_has_no_categories() -> None:
    encoder = contig_encoder()
    encoder.add(_chunk({"d_call": [None, None]}), np.asarray([0, 1]), source=0)
    table = encoder.finish()
    assert table.fields == ("d_call",)
    assert table.text["d_call"].codes.tolist() == [-1, -1]
    assert len(table.text["d_call"].categories) == 0


def test_an_encoder_finishes_once() -> None:
    encoder = contig_encoder()
    encoder.add(_chunk({"junction_aa": ["CASS"]}), np.asarray([0]), source=0)
    encoder.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        encoder.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        encoder.add(_chunk({"junction_aa": ["CASS"]}), np.asarray([0]), source=0)


def test_fields_of_some_sources_are_missing_for_the_others() -> None:
    encoder = contig_encoder()
    encoder.add(
        _chunk(
            {
                "sequence_id": ["a"],
                "c_call": ["IGHM"],
                "sequence": ["ACGT"],
                "umi_count": np.asarray([3], dtype=np.int64),
            }
        ),
        np.asarray([1]),
        source=0,
    )
    other = _chunk({"sequence_id": ["b"]})
    other["note"] = ["a column outside the AIRR fields is ignored"]
    encoder.add(other, np.asarray([0]), source=1)
    assert encoder.nbytes > 0
    table = encoder.finish()
    assert table.column("sequence_id").tolist() == ["b", "a"]
    assert table.column("c_call").tolist() == [None, "IGHM"]
    assert table.column("sequence").tolist() == [None, "ACGT"]
    assert table.column("umi_count").tolist() == [-1, 3]
