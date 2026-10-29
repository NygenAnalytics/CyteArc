from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire.contigs import pack_text
from cytearc.repertoire.usage import (
    GENE_LEVELS,
    count_chains,
    count_units,
    gene_codes,
    gene_names,
    text_lengths,
)


def test_gene_calls_are_named_at_each_level() -> None:
    calls = np.asarray(
        [
            "TRBV12-3*01",
            "TRBV6-2*01,TRBV6-3*01",
            "TRAV29/DV5",
            "IGKV1D-39*01",
            "IGHV3OR16-9",
            "IGHG1",
            "TRBC2",
            "TRBD1",
            "IGHJ4*02",
            None,
            "",
            float("nan"),
        ],
        dtype=object,
    )
    assert gene_names(calls, "allele").tolist() == [
        "TRBV12-3*01",
        "TRBV6-2*01",
        "TRAV29/DV5",
        "IGKV1D-39*01",
        "IGHV3OR16-9",
        "IGHG1",
        "TRBC2",
        "TRBD1",
        "IGHJ4*02",
        None,
        None,
        None,
    ]
    assert gene_names(calls, "gene").tolist()[:9] == [
        "TRBV12-3",
        "TRBV6-2",
        "TRAV29/DV5",
        "IGKV1D-39",
        "IGHV3OR16-9",
        "IGHG1",
        "TRBC2",
        "TRBD1",
        "IGHJ4",
    ]
    assert gene_names(calls, "family").tolist()[:9] == [
        "TRBV12",
        "TRBV6",
        "TRAV29",
        "IGKV1",
        "IGHV3",
        "IGHG1",
        "TRBC2",
        "TRBD1",
        "IGHJ4",
    ]
    assert GENE_LEVELS == ("allele", "gene", "family")
    with pytest.raises(ValueError, match="level must be one of"):
        gene_names(calls, "segment")


def test_units_are_counted_with_shares_of_their_keys() -> None:
    units = pd.DataFrame(
        {
            "group": [0, 0, 0, 0, 1, 1],
            "locus": ["TRB", "TRB", "TRB", "TRA", "TRB", "TRB"],
            "gene": ["B", "A", "A", "C", "B", None],
        }
    )
    table = count_units(units, ["group", "locus"], ["gene"], "cells")
    assert table.to_dict("list") == {
        "group": [0, 0, 0, 1],
        "locus": ["TRA", "TRB", "TRB", "TRB"],
        "gene": ["C", "A", "B", "B"],
        "cells": [1, 2, 1, 1],
        "fraction": [1.0, 2 / 3, 1 / 3, 1.0],
    }
    empty = count_units(units.iloc[:0], ["group", "locus"], ["gene"], "cells")
    assert empty.columns.tolist() == ["group", "locus", "gene", "cells", "fraction"]
    assert empty.empty


def test_gene_codes_name_each_distinct_call_once() -> None:
    calls = np.asarray(
        ["TRBV12-3*01", "TRBV12-3*02", "TRBV6-2*01,TRBV6-3*01", ""], dtype=object
    )
    codes = np.asarray([1, 2, -1, 0, 3, 1])
    named, names = gene_codes(codes, calls, "gene")
    # Both alleles of TRBV12-3 name one gene; a blank call names nothing.
    assert names.tolist() == ["TRBV12-3", "TRBV6-2"]
    assert named.tolist() == [0, 1, -1, 0, -1, 0]
    alleles, allele_names = gene_codes(codes, calls, "allele")
    assert allele_names.tolist() == ["TRBV12-3*01", "TRBV12-3*02", "TRBV6-2*01"]
    assert alleles.tolist() == [1, 2, -1, 0, -1, 1]
    nothing, no_names = gene_codes(np.asarray([-1]), calls[:0], "family")
    assert nothing.tolist() == [-1] and no_names.tolist() == []


def test_text_lengths_count_characters() -> None:
    data, offsets = pack_text(np.asarray(["CASSF", "CAW", "CF"], dtype=object))
    assert text_lengths(data, offsets, np.asarray([2, 0, 0])).tolist() == [2, 5, 5]
    # Text beyond ASCII is measured in characters, not bytes.
    data, offsets = pack_text(np.asarray(["CAŚSF", "CAW"], dtype=object))
    assert len(data) == 9
    assert text_lengths(data, offsets, np.asarray([0, 1])).tolist() == [5, 3]
    empty, bounds = pack_text(np.asarray([], dtype=object))
    assert text_lengths(empty, bounds, np.asarray([], dtype=np.int64)).tolist() == []


def _units(rng: np.random.Generator, n_chains: int):
    groups = rng.integers(0, 3, size=n_chains)
    loci = rng.integers(0, 2, size=n_chains)
    genes = rng.integers(-1, 4, size=n_chains)
    pairs = rng.integers(-1, 2, size=n_chains)
    cells = rng.integers(0, n_chains // 2 + 1, size=n_chains)
    clonotypes = rng.integers(0, 6, size=n_chains)
    return groups, loci, genes, pairs, cells, clonotypes


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("count", ["cells", "clonotypes"])
def test_chains_are_counted_as_count_units_counts_their_labels(seed, count) -> None:
    rng = np.random.default_rng(seed)
    groups, loci, genes, pairs, cells, clonotypes = _units(rng, 300)
    keys = pd.DataFrame(
        {"group": ["s1, a", "s2, a", "s10, b"], "sample": ["s1", "s2", "s10"]}
    )
    locus_names = np.asarray(["TRA", "TRB"], dtype=object)
    gene_labels = np.asarray(["TRAV1", "TRAV10", "TRAV2", "TRBV1"], dtype=object)
    pair_labels = np.asarray(["TRAJ1", "TRAJ2"], dtype=object)
    table, n_cells = count_chains(
        keys,
        groups,
        cells,
        (loci, locus_names),
        {"v_call": (genes, gene_labels), "j_call": (pairs, pair_labels)},
        count,
        clonotypes=clonotypes,
    )
    # The same chains as labels, counted by count_units.
    units = pd.DataFrame(
        {
            "group": groups,
            "clonotype": clonotypes,
            "locus": locus_names[loci],
            "v_call": np.where(genes >= 0, gene_labels[genes], None),
            "j_call": np.where(pairs >= 0, pair_labels[pairs], None),
        }
    )
    if count == "clonotypes":
        units = units.drop_duplicates(
            ["group", "clonotype", "locus", "v_call", "j_call"]
        )
    expected = count_units(
        units.drop(columns="clonotype"), ["group", "locus"], ["v_call", "j_call"], count
    )
    positions = expected.pop("group").to_numpy()
    expected.insert(0, "sample", keys["sample"].to_numpy()[positions])
    expected.insert(0, "group", keys["group"].to_numpy()[positions])
    pd.testing.assert_frame_equal(table, expected, check_exact=True)
    known = (genes >= 0) & (pairs >= 0)
    assert n_cells == len(set(cells[known].tolist()))


def test_chains_are_ordered_by_value_or_by_count() -> None:
    keys = pd.DataFrame({"group": ["all cells"]})
    lengths = np.asarray([12, 14, 14, 13, 14, -1, 12])
    table, n_cells = count_chains(
        keys,
        np.zeros(7, dtype=np.int64),
        np.arange(7),
        (np.zeros(7, dtype=np.int64), np.asarray(["TRB"], dtype=object)),
        {"length": (lengths, np.arange(15))},
        "cells",
        by_count=False,
    )
    assert table[["length", "cells"]].values.tolist() == [[12, 2], [13, 1], [14, 3]]
    assert table["length"].dtype == np.int64 and n_cells == 6
    by_count = count_chains(
        keys,
        np.zeros(7, dtype=np.int64),
        np.arange(7),
        (np.zeros(7, dtype=np.int64), np.asarray(["TRB"], dtype=object)),
        {"length": (lengths, np.arange(15))},
        "cells",
    )[0]
    assert by_count["length"].tolist() == [14, 12, 13]
    empty, none = count_chains(
        keys,
        np.zeros(2, dtype=np.int64),
        np.arange(2),
        (np.zeros(2, dtype=np.int64), np.asarray(["TRB"], dtype=object)),
        {"d_call": (np.asarray([-1, -1]), np.asarray([], dtype=object))},
        "clonotypes",
        clonotypes=np.arange(2),
    )
    assert empty.columns.tolist() == [
        "group",
        "locus",
        "d_call",
        "clonotypes",
        "fraction",
    ]
    assert empty.empty and none == 0


def test_wide_keys_are_counted_row_by_row() -> None:
    from cytearc.repertoire.usage import _distinct

    rng = np.random.default_rng(2)
    columns = [rng.integers(0, 4, size=200), rng.integers(0, 3, size=200)]
    narrow = _distinct(columns, [4, 3])
    # Widths whose product reaches 2**62 cannot share one integer key.
    wide = _distinct(columns, [2**40, 2**30])
    for got in (narrow, wide):
        expected = sorted(
            Counter(zip(*(column.tolist() for column in columns))).items()
        )
        assert [tuple(row) for row in got[0].tolist()] == [key for key, _ in expected]
        assert got[1].tolist() == [count for _, count in expected]
