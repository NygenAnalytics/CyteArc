import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire.usage import GENE_LEVELS, count_units, gene_names


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
