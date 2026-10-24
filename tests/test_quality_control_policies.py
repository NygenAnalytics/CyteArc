"""MAD metric policies, warnings about missed populations, and bound types.

These cover the policy that ``qc.auto_filter`` gives each metric, the
warnings about bounds that a raw droplet matrix sets, the types of manual
bounds, the exclusive bounds of ``qc.filter``, and the reads of
``qc.feature_percentage``.
"""

import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from cytearc import DataStore
from cytearc.matrix import ChunkedArray
from cytearc.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from cytearc.metadata.selection import NamedCellArtifact
from cytearc.quality_control import filtering
from cytearc.quality_control.filtering import (
    CALLED_CELLS_REMEDY,
    RAW_DROPLET_REMEDY,
    MetricPolicy,
    _split_two_groups,
    filter_cell_metrics,
    resolve_metric_policies,
    sample_aware_mad_mask,
    validate_bound_types,
    validate_metric_policies,
)
from cytearc.storage.artifacts import ArtifactRef
from cytearc.storage.selections import read_stored_selection_mask
from tests.qc_helpers import (
    QC_VALUES,
    fresh_qc_store,
    open_qc_store,
    open_small_store,
    reference_gaussian_bounds,
    reference_mad_keep,
    write_small_store,
)

LOG_TWO_SIDED = {"transform": "log1p", "bound_direction": "two_sided"}
RAW_UPPER = {"transform": "identity", "bound_direction": "upper"}
RAW_TWO_SIDED = {"transform": "identity", "bound_direction": "two_sided"}


@pytest.fixture(scope="module")
def small_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("policies") / "store.zarr"
    write_small_store(path)
    return path


@pytest.fixture
def small(small_template, tmp_path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(small_template, target)
    return open_small_store(target)


@pytest.fixture
def prepared() -> DataStore:
    """Six cells whose preparation wrote both percentage columns."""
    store, _ = fresh_qc_store()
    return open_qc_store(store)


def _mask(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        store.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _quality_metric(
    store: DataStore, cells: ArtifactRef, name: str, values: np.ndarray
) -> NamedCellArtifact:
    """Store a ``quality_metric`` artifact that qc.feature_percentage did not write."""
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="imported_quality_metric",
        parameters={"name": name},
        inputs={},
        execution_options={},
        cell_selection=cells,
        arrays={"values": ((len(values),), None)},
    )
    write_cell_data_artifact(store.zw, planned, {"values": np.asarray(values)})
    return NamedCellArtifact(name, planned.ref)


def _mad_bounds(values: np.ndarray, *, log: bool, n_mads: float = 3.0):
    """Raw-scale MAD bounds of NumPy, before any clamping."""
    work = np.log1p(values) if log else np.asarray(values, dtype=float)
    center = np.median(work)
    spread = 1.4826 * np.median(np.abs(work - center))
    low, high = center - n_mads * spread, center + n_mads * spread
    return (np.expm1(low), np.expm1(high)) if log else (low, high)


# Explicit policies


def test_explicit_metric_policies_are_validated_before_any_read():
    assert validate_metric_policies(None, ["a"]) == {}
    assert validate_metric_policies(
        {"a": {"transform": "log1p", "bound_direction": "lower"}}, ["a", "b"]
    ) == {"a": MetricPolicy("log1p", "lower")}
    for policies, error, message in (
        ("a", TypeError, "metric_policies must map metric names to policies"),
        (
            {"c": RAW_UPPER},
            ValueError,
            "metric_policies names metrics that are not filtered: ['c']",
        ),
        (
            {"a": "upper"},
            TypeError,
            "The policy of 'a' must map 'transform' and 'bound_direction' to text",
        ),
        (
            {"a": {"transform": "log1p"}},
            ValueError,
            "The policy of 'a' must have exactly the keys 'transform' and "
            "'bound_direction'; got ['transform']",
        ),
        (
            {"a": {**RAW_UPPER, "log": True}},
            ValueError,
            "The policy of 'a' must have exactly the keys 'transform' and "
            "'bound_direction'; got ['bound_direction', 'log', 'transform']",
        ),
        (
            {"a": {"transform": "log", "bound_direction": "upper"}},
            ValueError,
            "The transform of 'a' must be one of ['log1p', 'identity']; got 'log'",
        ),
        (
            {"a": {"transform": "identity", "bound_direction": "both"}},
            ValueError,
            "The bound_direction of 'a' must be one of ['two_sided', 'lower', "
            "'upper']; got 'both'",
        ),
    ):
        with pytest.raises(error, match=f"^{re.escape(message)}$"):
            validate_metric_policies(policies, ["a", "b"])


def test_known_kinds_and_explicit_policies_win_over_names():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policies = resolve_metric_policies(
            ["RNA_nCounts", "n_genes", "pct_hb", "percentMito"],
            kinds={
                "RNA_nCounts": "counts",
                "n_genes": "features",
                "pct_hb": "percentage",
                "percentMito": "percentage",
            },
            explicit={"percentMito": MetricPolicy("identity", "two_sided")},
        )

    assert policies == {
        "RNA_nCounts": MetricPolicy("log1p", "two_sided"),
        "n_genes": MetricPolicy("log1p", "two_sided"),
        "pct_hb": MetricPolicy("identity", "upper", percentage=True),
        # An explicit policy keeps the percentage range of its kind.
        "percentMito": MetricPolicy("identity", "two_sided", percentage=True),
    }


def test_metrics_of_unknown_kind_take_their_name_policy_with_one_warning():
    with pytest.warns(UserWarning) as caught:
        policies = resolve_metric_policies(
            ["total_counts", "x_percentRibo", "custom_nFeatures", "done"],
            kinds={},
            explicit={"done": MetricPolicy("log1p", "lower")},
        )

    assert policies == {
        "total_counts": MetricPolicy("identity", "two_sided"),
        "x_percentRibo": MetricPolicy("identity", "upper", percentage=True),
        "custom_nFeatures": MetricPolicy("log1p", "two_sided"),
        # An explicit policy is never a guess.
        "done": MetricPolicy("log1p", "lower"),
    }
    assert [str(warning.message) for warning in caught] == [
        "QC metrics ['total_counts', 'x_percentRibo', 'custom_nFeatures'] were "
        "not written by CyteArc's preparation or by qc.feature_percentage, so "
        "their MAD policies come from their names: 'total_counts' (raw values "
        "with two-sided bounds), 'x_percentRibo' (raw values with an upper "
        "bound), 'custom_nFeatures' (log1p values with two-sided bounds). Pass "
        "metric_policies to set them, for example {'transform': 'log1p', "
        "'bound_direction': 'two_sided'} for counts or {'transform': "
        "'identity', 'bound_direction': 'upper'} for a percentage."
    ]


def test_a_lower_policy_bounds_only_from_below():
    values = np.r_[np.linspace(100.0, 200.0, 40), [5.0, 5000.0]]

    keep, provenance = sample_aware_mad_mask(
        values_by_attr={"total_counts": values},
        sample_labels=None,
        active=np.ones(len(values), dtype=bool),
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["total_counts"],
        metric_policies={"total_counts": MetricPolicy("log1p", "lower")},
    )

    low, _ = _mad_bounds(values, log=True)
    bounds = provenance["resolved_bounds"]["all"]["total_counts"]
    assert bounds["low"] == pytest.approx(low, rel=1e-12)
    assert bounds["high"] is None
    assert (bounds["transform"], bounds["bound_direction"]) == ("log1p", "lower")
    assert provenance["metric_policies"] == {
        "total_counts": {"transform": "log1p", "bound_direction": "lower"}
    }
    # The deep cell stays; only the shallow one falls under the lower bound.
    np.testing.assert_array_equal(keep, values > low)
    assert keep[-1] and not keep[-2]


def test_only_percentages_keep_their_upper_bound_within_100():
    counts = np.linspace(1000.0, 3000.0, 30)
    percent = np.r_[np.linspace(90.0, 99.0, 29), [100.0]]
    active = np.ones(30, dtype=bool)

    _, provenance = sample_aware_mad_mask(
        values_by_attr={"total_counts": counts, "pct": percent},
        sample_labels=None,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["total_counts", "pct"],
        metric_policies={
            # Raw values with an upper bound are not a percentage, so their
            # bound of more than 3000 is not cut to 100.
            "total_counts": MetricPolicy("identity", "upper"),
            "pct": MetricPolicy("identity", "two_sided", percentage=True),
        },
    )

    bounds = provenance["resolved_bounds"]["all"]
    count_high = _mad_bounds(counts, log=False)[1]
    assert bounds["total_counts"]["high"] == pytest.approx(count_high, rel=1e-12)
    assert bounds["total_counts"]["high"] > 3000
    low, high = _mad_bounds(percent, log=False)
    assert high > 100
    assert bounds["pct"]["high"] == 100.0
    # A lower bound below the values keeps them all, so it is not clamped.
    assert bounds["pct"]["low"] == pytest.approx(low, rel=1e-12)


def test_auto_filter_rejects_invalid_metric_policies_before_reading(small, monkeypatch):
    def unread(*args, **kwargs):
        raise AssertionError("metrics were read")

    monkeypatch.setattr(type(small.cells), "fetch_all_columns", unread)
    for options, error, message in (
        (
            {"metric_policies": {"RNA_nCounts": LOG_TWO_SIDED}, "method": "gaussian"},
            ValueError,
            "metric_policies apply only to method='mad'",
        ),
        (
            {"metric_policies": {"RNA_percentMito": RAW_UPPER}},
            ValueError,
            "metric_policies names metrics that are not filtered: ['RNA_percentMito']",
        ),
        (
            {"metric_policies": [("RNA_nCounts", LOG_TWO_SIDED)]},
            TypeError,
            "metric_policies must map metric names to policies",
        ),
    ):
        with pytest.raises(error, match=f"^{re.escape(message)}$"):
            small.qc.auto_filter(**options)
    assert small.artifacts.list(kind="cell_selection") == []


def test_default_auto_filter_keeps_the_records_of_earlier_releases(small):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = small.qc.auto_filter()

    status = small.artifacts.inspect(ref)
    parameters = status.parameters
    # The parameters and inputs that earlier releases recorded, with the
    # policies their name rule chose, so default calls keep their identities.
    assert set(parameters) == {
        "method",
        "attrs",
        "metric_sources",
        "n_mads",
        "min_cells_per_sample",
        "resolved_bounds",
        "sample_source",
        "mad_scale",
        "metric_policies",
        "sample_sizes",
        "skip_reasons",
    }
    assert set(status.inputs) == {
        "prior_cell_selection",
        "qc_metric_fingerprints",
        "artifact_metrics",
        "ordered_row_ids_fingerprint",
        "values_fingerprint",
    }
    assert parameters["metric_policies"] == {
        "RNA_nCounts": LOG_TWO_SIDED,
        "RNA_nFeatures": LOG_TWO_SIDED,
    }
    metrics = {
        attr: np.asarray(small.cells.fetch_all(attr), dtype=float)
        for attr in ("RNA_nCounts", "RNA_nFeatures")
    }
    np.testing.assert_array_equal(_mask(small, ref), reference_mad_keep(metrics))
    # Passing the default policies explicitly is the same request.
    assert (
        small.qc.auto_filter(
            metric_policies={
                "RNA_nCounts": LOG_TWO_SIDED,
                "RNA_nFeatures": LOG_TWO_SIDED,
            }
        )
        == ref
    )


def test_scanpy_named_columns_need_explicit_policies_to_match_cytearc_columns(small):
    # Imported copies of nCounts and nFeatures under scanpy names. Their names
    # do not say what they hold, so they get raw two-sided bounds with a
    # warning, as before; explicit policies give the bounds of the originals.
    small.cells.insert("total_counts", small.cells.fetch_all("RNA_nCounts"))
    small.cells.insert("n_genes_by_counts", small.cells.fetch_all("RNA_nFeatures"))
    reference = small.qc.auto_filter(attrs=["RNA_nCounts", "RNA_nFeatures"])

    with pytest.warns(UserWarning, match="come from their names") as caught:
        guessed = small.qc.auto_filter(attrs=["total_counts", "n_genes_by_counts"])
    explicit_policies = {
        "total_counts": LOG_TWO_SIDED,
        "n_genes_by_counts": LOG_TWO_SIDED,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = small.qc.auto_filter(
            attrs=["total_counts", "n_genes_by_counts"],
            metric_policies=explicit_policies,
        )

    assert len(caught) == 1
    assert small.artifacts.inspect(guessed).parameters["metric_policies"] == {
        "total_counts": RAW_TWO_SIDED,
        "n_genes_by_counts": RAW_TWO_SIDED,
    }
    np.testing.assert_array_equal(_mask(small, explicit), _mask(small, reference))
    reference_bounds = small.artifacts.inspect(reference).parameters["resolved_bounds"]
    explicit_parameters = small.artifacts.inspect(explicit).parameters
    assert explicit_parameters["metric_policies"] == explicit_policies
    assert explicit_parameters["resolved_bounds"]["all"] == {
        "total_counts": reference_bounds["all"]["RNA_nCounts"],
        "n_genes_by_counts": reference_bounds["all"]["RNA_nFeatures"],
    }


def test_metric_kinds_come_from_preparation_and_feature_percentages(prepared):
    has_counts = QC_VALUES.sum(axis=1) > 0
    prepared.cells.insert("has_counts", has_counts)
    cells = prepared.snapshot_cell_selection("has_counts")
    mito = prepared.qc.feature_percentage(
        cells, prepared.features.snapshot(feature_indexes=[0])
    )
    computed = NamedCellArtifact("percentHb", mito)
    imported = _quality_metric(prepared, cells, "percentHb2", np.arange(5.0))
    prepared.cells.insert("RNA_percentOther", np.arange(6.0))

    kinds = prepared._metric_kinds(
        [
            "RNA_nCounts",
            "RNA_nFeatures",
            "RNA_percentMito",
            "RNA_percentRibo",
            "RNA_percentOther",
        ],
        [computed, imported],
    )

    # A user column named like a percentage and an artifact of another
    # operation have no known kind.
    assert kinds == {
        "RNA_nCounts": "counts",
        "RNA_nFeatures": "features",
        "RNA_percentMito": "percentage",
        "RNA_percentRibo": "percentage",
        "percentHb": "percentage",
    }


def test_feature_percentage_artifacts_get_an_upper_bound(prepared):
    # A quality_metric artifact of qc.feature_percentage is a percentage, so a
    # cell with an unusually low percentage is no outlier. Its name used to
    # give it raw two-sided bounds.
    has_counts = QC_VALUES.sum(axis=1) > 0
    prepared.cells.insert("has_counts", has_counts)
    cells = prepared.snapshot_cell_selection("has_counts")
    mito = prepared.qc.feature_percentage(
        cells, prepared.features.snapshot(feature_indexes=[0])
    )
    values = np.asarray(prepared.artifacts.load(mito)["values"][:])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = prepared.qc.auto_filter(
            attrs=[],
            artifact_metrics=[NamedCellArtifact("percentHb", mito)],
            cell_selection=cells,
            min_cells_per_sample=2,
        )

    parameters = prepared.artifacts.inspect(ref).parameters
    assert parameters["metric_policies"] == {"percentHb": RAW_UPPER}
    bounds = parameters["resolved_bounds"]["all"]["percentHb"]
    high = min(100.0, _mad_bounds(values, log=False)[1])
    assert bounds["low"] is None
    assert bounds["high"] == pytest.approx(high, rel=1e-12)
    expected = np.zeros(prepared.cells.N, dtype=bool)
    expected[np.flatnonzero(has_counts)] = values < high
    np.testing.assert_array_equal(_mask(prepared, ref), expected)


def test_a_percentage_policy_keeps_cells_with_unusually_low_percentages():
    # Median 5% and scaled MAD 1.4826: raw two-sided bounds remove the
    # cleanest cell, at 0.3%, as an outlier.
    percent = np.r_[np.linspace(4.0, 6.0, 41), [0.3]]
    active = np.ones(len(percent), dtype=bool)
    two_sided, _ = sample_aware_mad_mask(
        values_by_attr={"percentHb": percent},
        sample_labels=None,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["percentHb"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policies = resolve_metric_policies(
            ["percentHb"], kinds={"percentHb": "percentage"}, explicit={}
        )
    upper, _ = sample_aware_mad_mask(
        values_by_attr={"percentHb": percent},
        sample_labels=None,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["percentHb"],
        metric_policies=policies,
    )

    low, high = _mad_bounds(percent, log=False)
    assert 0.3 < low < 4.0
    np.testing.assert_array_equal(two_sided, (percent > low) & (percent < high))
    assert not two_sided[-1]
    np.testing.assert_array_equal(upper, percent < high)
    assert upper.all()


# Missed populations and majority removals


def _empty_droplets_and_cells() -> tuple[np.ndarray, np.ndarray]:
    """200 near-empty droplets with 3 to 12 counts and 20 cells with about 1000."""
    empties = 3.0 + np.arange(200) % 10
    cells = 1000.0 + np.arange(20)
    values = np.r_[empties, cells]
    return values, np.r_[np.zeros(200, bool), np.ones(20, bool)]


def test_cells_above_bounds_set_by_empty_droplets_are_reported():
    values, is_cell = _empty_droplets_and_cells()

    with pytest.warns(UserWarning) as caught:
        result = filter_cell_metrics(
            {"RNA_nCounts": values}, {}, np.ones(len(values), dtype=bool), method="mad"
        )

    low, high = _mad_bounds(values, log=True)
    above = values[values >= high]
    np.testing.assert_array_equal(above, values[is_cell])
    # The empty droplets set the bounds, so every real cell is removed.
    np.testing.assert_array_equal(result.retained, ~is_cell)
    assert [str(warning.message) for warning in caught] == [
        f"Selected cells: 20 cells above the upper 'RNA_nCounts' bound of "
        f"{high:.6g} form a separate population (median {np.median(above):.6g}), "
        f"which the MAD bounds remove as outliers. {RAW_DROPLET_REMEDY} Otherwise "
        "check whether those cells are a real population that the bounds should "
        "keep."
    ]


def test_a_bimodal_metric_whose_low_population_is_kept_is_reported():
    empties = 3.0 + np.arange(100) % 10
    cells = 5000.0 + np.arange(100)
    values = np.r_[empties, cells]
    labels = np.array(["s1"] * 200)

    with pytest.warns(UserWarning) as caught:
        result = filter_cell_metrics(
            {"RNA_nFeatures": values},
            {},
            np.ones(200, dtype=bool),
            method="mad",
            sample_labels=labels,
        )

    # Half the values in each population give a scaled MAD so wide that the
    # bounds keep everything.
    assert result.retained.all()
    assert [str(warning.message) for warning in caught] == [
        "Sample 's1': 'RNA_nFeatures' is strongly bimodal: 100 cells form a low "
        f"population (median {np.median(empties):.6g}) and 100 a high population "
        f"(median {np.median(cells):.6g}), and the MAD bounds keep most of the low "
        f"population. MAD bounds assume one population of cells. {RAW_DROPLET_REMEDY}"
    ]


def test_called_cells_get_a_remedy_without_cell_calling():
    empties = 3.0 + np.arange(100) % 10
    cells = 5000.0 + np.arange(100)
    values = np.r_[empties, cells]

    with pytest.warns(UserWarning) as caught:
        filter_cell_metrics(
            {"RNA_nFeatures": values},
            {},
            np.ones(200, dtype=bool),
            method="mad",
            remedy=CALLED_CELLS_REMEDY,
        )

    # Cells that a run already called are not sent to call cells again.
    (message,) = [str(warning.message) for warning in caught]
    assert message.endswith(
        f"MAD bounds assume one population of cells. {CALLED_CELLS_REMEDY}"
    )
    assert "qc.call_cells" not in message


def test_a_bimodal_metric_without_mad_is_reported():
    # Most droplets hold exactly 2 counts, so the scaled MAD is zero and no
    # bound applies, while a quarter are cells.
    values = np.r_[np.full(150, 2.0), 4000.0 + np.arange(50)]

    with pytest.warns(UserWarning) as caught:
        filter_cell_metrics(
            {"RNA_nCounts": values}, {}, np.ones(200, dtype=bool), method="mad"
        )

    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 1
    assert messages[0].startswith(
        "Selected cells: 'RNA_nCounts' is strongly bimodal: 150 cells form a low "
        "population (median 2) and 50 a high population (median 4024.5)"
    )


def test_bounds_that_remove_low_debris_or_keep_one_population_raise_no_warning():
    rng = np.random.default_rng(5)
    cells = rng.lognormal(np.log(5000), 0.5, 3000).round()
    debris = rng.lognormal(np.log(60), 0.5, 150).round()
    doublets = rng.lognormal(np.log(10000), 0.5, 300).round()
    few_outliers = np.r_[cells, 10 * cells[:9]]
    single_value = np.full(50, 7.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for counts in (
            cells,
            np.r_[cells, debris],
            np.r_[cells, doublets],
            few_outliers,
        ):
            filter_cell_metrics(
                {"RNA_nCounts": counts},
                {},
                np.ones(len(counts), dtype=bool),
                method="mad",
            )
        # Raw metrics are not checked for populations, and a constant metric
        # has no split.
        filter_cell_metrics(
            {"score": np.r_[cells, debris]},
            {},
            np.ones(len(cells) + len(debris), dtype=bool),
            method="mad",
        )
        filter_cell_metrics(
            {"RNA_nCounts": single_value}, {}, np.ones(50, dtype=bool), method="mad"
        )


def test_automatic_bounds_that_remove_most_cells_are_reported():
    values = np.arange(1.0, 101.0)
    active = np.ones(100, dtype=bool)

    with pytest.warns(UserWarning) as caught:
        result = filter_cell_metrics(
            {"score": values}, {}, active, method="mad", n_mads=0.2
        )

    low, high = _mad_bounds(values, log=False, n_mads=0.2)
    removed = int(np.sum(~((values > low) & (values < high))))
    assert removed > 50
    assert int(result.retained.sum()) == 100 - removed
    assert [str(warning.message) for warning in caught] == [
        f"Automatic QC bounds removed {removed} of 100 selected cells "
        f"({removed / 100:.0%}), so most selected cells lie outside bounds meant to "
        "remove outliers. Check the resolved bounds and the metric distributions. "
        f"{RAW_DROPLET_REMEDY}"
    ]
    gaussian_low, gaussian_high = reference_gaussian_bounds(values, 0.45, 0.55)
    gaussian_removed = int(np.sum((values <= gaussian_low) | (values >= gaussian_high)))
    assert gaussian_removed == 92
    with pytest.warns(UserWarning, match="^Automatic QC bounds removed 92 of 100"):
        filter_cell_metrics(
            {"score": values}, {}, active, method="gaussian", min_p=0.45, max_p=0.55
        )
    # Half of the cells is not a majority, and manual bounds are a choice.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filter_cell_metrics(
            {"score": values}, {}, active, method="manual", lows=[90.0], highs=[None]
        )
        half = filter_cell_metrics(
            {"score": np.r_[np.zeros(50), np.ones(50)]},
            {},
            active,
            method="gaussian",
            min_p=0.01,
            max_p=0.5,
        )
    assert int(half.retained.sum()) == 50


def _brute_force_split(values: np.ndarray) -> tuple[float, float]:
    """Try every split between distinct sorted values and keep the first best."""
    ordered = np.sort(values)
    best = None
    for n_low in range(1, len(ordered)):
        if ordered[n_low - 1] == ordered[n_low]:
            continue
        low, high = ordered[:n_low], ordered[n_low:]
        between = n_low * (len(ordered) - n_low) * (high.mean() - low.mean()) ** 2
        if best is None or between > best[0] * (1 + 1e-12):
            separation = (high.mean() - low.mean()) / np.sqrt(
                (low.var() + high.var()) / 2
            )
            best = (between, separation, ordered[n_low - 1])
    assert best is not None
    return best[1], best[2]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_group_split_is_the_exact_two_means_split(seed):
    rng = np.random.default_rng(seed)
    values = np.round(np.r_[rng.normal(0, 1, 25), rng.normal(4, 1.5, 15)], 1)
    rng.shuffle(values)

    separation, threshold = _split_two_groups(values)

    expected_separation, expected_threshold = _brute_force_split(values)
    assert separation == pytest.approx(expected_separation, rel=1e-12)
    assert threshold == expected_threshold


def test_two_group_split_handles_constant_values_and_samples_long_inputs(monkeypatch):
    assert _split_two_groups(np.full(5, 3.0)) is None
    # Two values per group without spread are perfectly separated.
    assert _split_two_groups(np.array([1.0, 1.0, 9.0, 9.0])) == (np.inf, 1.0)

    values = np.random.default_rng(3).normal(size=20)
    monkeypatch.setattr(filtering, "_SPLIT_SAMPLE_SIZE", 7)
    sampled = _split_two_groups(values)
    monkeypatch.undo()

    # Twenty values over a cap of seven take every third value.
    assert sampled == _split_two_groups(values[::3])


# Bound types and exclusive bounds


@pytest.mark.parametrize(
    ("dtype", "low", "high", "message"),
    [
        (
            np.float64,
            "100",
            None,
            "Column 'x' holds numeric values, so its bounds must be numbers or "
            "None; got ['100', None]",
        ),
        (
            np.uint32,
            None,
            "a",
            "Column 'x' holds numeric values, so its bounds must be numbers or "
            "None; got [None, 'a']",
        ),
        (
            np.dtype("<U3"),
            1,
            None,
            "Column 'x' holds text, so its bounds must be text or None; got [1, None]",
        ),
        (
            np.dtype(object),
            "a",
            2.5,
            "Column 'x' holds text, so its bounds must be text or None; got ['a', 2.5]",
        ),
        (
            np.dtype("datetime64[D]"),
            1,
            None,
            "Column 'x' holds datetime64[D] values, which bounds cannot filter",
        ),
    ],
)
def test_bounds_must_match_their_column(dtype, low, high, message):
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        validate_bound_types("x", np.dtype(dtype), low, high)


def test_matching_bounds_and_open_sides_pass_for_every_column_kind():
    validate_bound_types("x", np.dtype(bool), 0, None)
    validate_bound_types("x", np.dtype(np.int64), -1, 5.5)
    validate_bound_types("x", np.dtype("<U4"), "a", "z")
    validate_bound_types("x", np.dtype("S4"), None, "z")
    validate_bound_types("x", np.dtype("datetime64[D]"), None, None)


def test_text_bounds_on_numeric_columns_fail_before_any_read(prepared, monkeypatch):
    prepared.cells.insert("label", np.array(list("abcabc")))

    def unread(*args, **kwargs):
        raise AssertionError("metrics were read")

    monkeypatch.setattr(type(prepared.cells), "fetch_all_columns", unread)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Column 'RNA_nCounts' holds numeric values, so its bounds must be "
            "numbers or None; got ['100', None]"
        ),
    ):
        prepared.qc.filter(["RNA_nCounts"], lows=["100"], highs=[None])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Column 'label' holds text, so its bounds must be text or None; "
            "got [None, 5]"
        ),
    ):
        prepared.qc.filter(["label"], lows=[None], highs=[5])
    assert prepared.artifacts.list(kind="cell_selection") == []


def test_manual_filters_check_bound_types_against_their_values():
    # The pipeline's manual filter reaches this check with snapshot values.
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Column 'label' holds text, so its bounds must be text or None; "
            "got [1.0, None]"
        ),
    ):
        filter_cell_metrics(
            {"label": np.array(["a", "b"])},
            {},
            np.ones(2, dtype=bool),
            method="manual",
            lows=[1.0],
            highs=[None],
        )


def test_filter_bounds_are_exclusive_unless_kept(small):
    # scanpy's filter_cells(min_genes=k) keeps cells with at least k genes,
    # as keep_bounds=True does; the default removes a cell with exactly k.
    features = np.asarray(small.cells.fetch_all("RNA_nFeatures"))
    k = float(np.sort(features)[len(features) // 2])
    assert np.sum(features == k) > 0

    exclusive = small.qc.filter(["RNA_nFeatures"], lows=[k], highs=[None])
    inclusive = small.qc.filter(
        ["RNA_nFeatures"], lows=[k], highs=[None], keep_bounds=True
    )

    np.testing.assert_array_equal(_mask(small, exclusive), features > k)
    np.testing.assert_array_equal(_mask(small, inclusive), features >= k)


# Feature percentages


def test_feature_percentage_reads_only_the_selected_features(prepared, monkeypatch):
    # Every cell, including the one without counts, whose percentage is NaN.
    prepared.cells.insert("every_cell", np.ones(len(QC_VALUES), dtype=bool))
    cells = prepared.snapshot_cell_selection("every_cell")
    features = prepared.features.snapshot(feature_indexes=[0, 3])
    streamed: list[tuple[int, int]] = []
    original = ChunkedArray._stream_blocks

    def recorded(self, *args, **kwargs):
        streamed.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ChunkedArray, "_stream_blocks", recorded)

    ref = prepared.qc.feature_percentage(cells, features)

    # The denominators are the prepared totals, so no read spans every
    # feature; the numerators read the two selected features.
    assert streamed == [(len(QC_VALUES), 2)]
    totals = QC_VALUES.sum(axis=1).astype(np.float64)
    expected = np.full(len(QC_VALUES), np.nan)
    np.divide(
        100.0 * QC_VALUES[:, [0, 3]].sum(axis=1),
        totals,
        out=expected,
        where=totals != 0,
    )
    np.testing.assert_array_equal(prepared.artifacts.load(ref)["values"][:], expected)


def test_feature_percentage_divides_by_the_prepared_totals(prepared, monkeypatch):
    cells = np.arange(len(QC_VALUES))
    rna = type(prepared.RNA)
    original = rna._cell_count_totals
    halved = prepared.RNA._compute_feature_percentage(cells, np.array([0, 1]))
    monkeypatch.setattr(
        rna, "_cell_count_totals", lambda self, index: 2 * original(self, index)
    )

    doubled_totals = prepared.RNA._compute_feature_percentage(cells, np.array([0, 1]))

    np.testing.assert_array_equal(doubled_totals, halved / 2)
