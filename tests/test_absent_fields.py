"""A rule that tests a field the input lacks refuses by name rather than filter.

The case is the one that once filtered every call without a word: GoT's
calls, which carry no quality, under ``min_quality``. The other rules of this
kind are tested beside their methods.
"""

import warnings

import numpy as np
import pytest

from tests.clone_helpers import every_cell, write_lines, write_truth_store
from tests.clone_truth import clone_truth


def test_min_quality_refuses_44_calls_without_a_quality_that_it_would_filter(
    tmp_path,
) -> None:
    # A GoT table of one target: each cell's UMIs of the wild type and the
    # mutant, and GoT's call, MUT with a mutant UMI and WT with wild-type
    # UMIs alone, as GEO's tables write them, and no quality.
    truth = clone_truth()
    store = write_truth_store(tmp_path / "store.zarr", truth)
    rng = np.random.default_rng(1)
    rows, labels = [], []
    for cell in truth.cell_ids.tolist():
        wild, mutant = int(rng.integers(0, 5)), int(rng.integers(0, 3))
        label = "MUT" if mutant else ("WT" if wild else "NA")
        labels.append(label)
        rows.append(f"{cell}\t{wild}\t{mutant}\t{label}")
    path = write_lines(tmp_path, "TET2.tsv", "BC\twtUMIs\tmutUMIs\tGenotype", rows)
    assert (labels.count("WT"), labels.count("MUT")) == (11, 33)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(path, cell_selection=every_cell(store))
    assert store.artifacts.load(counts).attrs["call_fields"] == []
    assert "quality" not in store.variants.load_counts(counts).calls.columns
    with pytest.raises(
        ValueError,
        match="^min_quality tests call_quality, which these counts cannot give: "
        "their 'got' calls carry no quality",
    ):
        store.variants.genotypes(counts, method="source", min_quality=30.0)
    assert store.artifacts.list(kind="variant_genotypes", scope="datastore") == []
    # Without the rule, the 44 calls are adopted as GoT wrote them.
    calls = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="source")
    ).calls
    called = calls["call"].astype(str).value_counts().to_dict()
    assert (called["wild_type"], called["alternative_present"]) == (11, 33)
