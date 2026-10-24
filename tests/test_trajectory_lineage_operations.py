"""Lineage-resolved trajectory operations store what their inputs define.

The producers run on the session PBMC store, whose fate map has three sinks.
Their payloads are compared with dense recomputations from the normalized
expression, coordinates, and fate probabilities, and their loaders refuse
records whose lineage, parameters, or payload were changed. Records are
changed on copies under new identities, so the session results stay intact.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

from cytearc.assay.normalization import resolve_normalization_params
from cytearc.graph.feature_projection import resolve_native_graph_inputs
from cytearc.matrix import ChunkedArray
from cytearc.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    fingerprint_stored_arrays,
    inspect_artifact,
    new_artifact_id,
)
from cytearc.storage.feature_selection import read_feature_selection_indices
from cytearc.storage.selections import (
    read_stored_selection_indices,
    resolve_metadata_snapshot,
)
from cytearc.trajectory import (
    LineageAggregationResult,
    LineageCurvesResult,
    LineageDriverResult,
    LineageMarkerResult,
    LineageTreeResult,
)
from cytearc.trajectory.artifacts import (
    LINEAGE_AGGREGATION_PAYLOAD,
    LINEAGE_CURVE_PAYLOAD,
    LINEAGE_DRIVER_PAYLOAD,
    LINEAGE_MARKER_PAYLOAD,
    LINEAGE_TREE_PAYLOAD,
)
from cytearc.trajectory.curves import (
    fit_lineage_curves,
    lineage_ranks,
    project_onto_curve,
)
from cytearc.trajectory.lineage_tree import (
    cluster_distances,
    cluster_moments,
    spanning_tree,
)
from cytearc.trajectory.lineages import (
    bin_smoothing_matrix,
    equal_mass_bin_operator,
    finalize_lineage_profiles,
    lineage_profiles,
)

from .fixtures_lineages import LINEAGE_SINKS

_OPERATIONS = "cytearc.datastore._operations.trajectory"


def _selected_indices(datastore, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        datastore.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64)


def _feature_indices(datastore, features: ArtifactRef) -> np.ndarray:
    return read_feature_selection_indices(
        datastore.zw, "RNA", datastore.features.resolve("RNA", features)
    ).astype(np.int64)


def _normalized(datastore, cells: np.ndarray, features: np.ndarray) -> np.ndarray:
    assay = datastore.get_assay("RNA")
    parameters = resolve_normalization_params(
        assay, {}, caller="run_lineage_marker_search"
    )
    return np.asarray(
        assay.normed(cell_idx=cells, feat_idx=features, **parameters).compute(),
        dtype=np.float64,
    )


def _scored(datastore, fate_ref: ArtifactRef) -> dict[str, Any]:
    fate = datastore.trajectory.load_fate(fate_ref)
    pseudotime = datastore.trajectory.load_pseudotime(fate.pseudotime)
    cells = _selected_indices(datastore, fate.cell_selection)[fate.valid]
    return {
        "fate": fate,
        "cells": cells,
        "ordering": pseudotime.values[fate.valid],
        "weights": np.asarray(fate.values[fate.valid], dtype=np.float64),
    }


def _sink_mask(datastore, leiden_clustering, fate) -> np.ndarray:
    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    return np.isin(labels[fate.valid], LINEAGE_SINKS)


def _weighted_pearson(x: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    total = w.sum()
    centered_x = x - (w[:, None] * x).sum(axis=0) / total
    centered_z = z - (w * z).sum() / total
    covariance = (w[:, None] * centered_x * centered_z[:, None]).sum(axis=0)
    variance_x = (w[:, None] * centered_x**2).sum(axis=0)
    variance_z = (w * centered_z**2).sum()
    # Constant features, which the operations leave untested, divide 0 by 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        return covariance / np.sqrt(variance_x * variance_z)


def _forged(
    datastore,
    ref: ArtifactRef,
    *,
    payload: tuple[str, ...],
    provenance: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> ArtifactRef:
    """Copy an artifact to a new identity with changed records or arrays.

    The copy's payload fingerprint is recomputed after any array change, so
    only the changed fact can fail validation.
    """
    source = artifact_group(datastore.zw, ref)
    forged = ArtifactRef(
        scope=ref.scope,
        kind=ref.kind,
        artifact_id=new_artifact_id(),
        assay=ref.assay,
    )
    group = datastore.zw.create_group(artifact_path(forged))
    for name in source.array_keys():
        values = (arrays or {}).get(name)
        group.create_array(
            name,
            data=np.asarray(source[name][...]) if values is None else values,
        )
    copied = dict(source.attrs)
    copied["artifact_id"] = forged.artifact_id
    if provenance is not None:
        copied["provenance"] = {**copied["provenance"], **provenance}
    copied["payload_fingerprint"] = fingerprint_stored_arrays(group, payload)
    if attrs is not None:
        copied.update(attrs)
    group.attrs.update(copied)
    return forged


def _record(datastore, ref: ArtifactRef) -> dict[str, Any]:
    return dict(artifact_group(datastore.zw, ref).attrs["provenance"])


def _with_parameters(datastore, ref: ArtifactRef, **parameters: Any) -> dict[str, Any]:
    record = _record(datastore, ref)
    return {"parameters": {**record["parameters"], **parameters}}


def _with_inputs(datastore, ref: ArtifactRef, **inputs: Any) -> dict[str, Any]:
    record = _record(datastore, ref)
    changed = {
        name: value.to_dict() if isinstance(value, ArtifactRef) else value
        for name, value in inputs.items()
    }
    return {"inputs": {**record["inputs"], **changed}}


def _without_input(datastore, ref: ArtifactRef, name: str) -> dict[str, Any]:
    record = _record(datastore, ref)
    return {
        "inputs": {key: value for key, value in record["inputs"].items() if key != name}
    }


def _subset_selection(datastore, name: str, keep: np.ndarray) -> ArtifactRef:
    datastore.cells.insert(name, keep, key="I", overwrite=True)
    return datastore.snapshot_cell_selection(name)


# Lineage markers


def test_lineage_markers_match_fate_weighted_correlations(
    datastore, lineage_fate, lineage_markers, mark_hvgs
):
    result = datastore.trajectory.load_lineage_markers(lineage_markers)
    scored = _scored(datastore, lineage_fate)
    features = _feature_indices(datastore, mark_hvgs)
    values = _normalized(datastore, scored["cells"], features)
    weights = scored["weights"]

    assert isinstance(result, LineageMarkerResult)
    assert result.lineages == tuple(LINEAGE_SINKS)
    assert result.fate == lineage_fate
    np.testing.assert_allclose(
        result.effective_cells,
        weights.sum(axis=0) ** 2 / (weights**2).sum(axis=0),
        rtol=1e-12,
    )
    assert len(result.table) == len(features) * len(LINEAGE_SINKS)
    for column, lineage in enumerate(LINEAGE_SINKS):
        rows = result.table[result.table["lineage"] == lineage]
        np.testing.assert_array_equal(rows["feature_index"], features)
        w = weights[:, column]
        expected = _weighted_pearson(values, scored["ordering"], w)
        mass = (w[:, None] * (values > 0)).sum(axis=0)
        tested = mass >= 10.0
        np.testing.assert_allclose(
            rows["r_value"].to_numpy()[tested], expected[tested], atol=1e-10
        )
        assert (
            rows["r_value"].to_numpy()[~tested].size == 0
            or np.isnan(rows["r_value"].to_numpy()[~tested]).all()
        )
        degrees = result.effective_cells[column] - 2.0
        r = expected[tested]
        statistic = r * np.sqrt(degrees / ((1.0 - r) * (1.0 + r)))
        p_values = 2.0 * student_t.sf(np.abs(statistic), degrees)
        np.testing.assert_allclose(
            rows["p_value"].to_numpy()[tested], p_values, rtol=1e-6, atol=1e-300
        )
        np.testing.assert_allclose(
            rows["p_value_adjusted"].to_numpy()[tested],
            multipletests(p_values, method="fdr_bh")[1],
            rtol=1e-6,
            atol=1e-300,
        )


def test_lineage_markers_reuse_and_recompute_identically(
    datastore, lineage_fate, lineage_markers, mark_hvgs
):
    assert (
        datastore.trajectory.lineage_markers(lineage_fate, features=mark_hvgs)
        == lineage_markers
    )
    fresh = datastore.trajectory.lineage_markers(
        lineage_fate, features=mark_hvgs, invalidate_cache=True
    )
    first = datastore.trajectory.load_lineage_markers(lineage_markers).table
    second = datastore.trajectory.load_lineage_markers(fresh).table
    pd.testing.assert_frame_equal(first, second)


def test_lineage_marker_subset_follows_the_requested_lineage_order(
    datastore, lineage_fate, lineage_markers, mark_hvgs
):
    ref = datastore.trajectory.lineage_markers(
        lineage_fate, features=mark_hvgs, lineages=[4, 2], gene_batch_size=7
    )
    subset = datastore.trajectory.load_lineage_markers(ref)
    full = datastore.trajectory.load_lineage_markers(lineage_markers)
    assert subset.lineages == (4, 2)
    for lineage in (4, 2):
        np.testing.assert_allclose(
            subset.table.loc[subset.table["lineage"] == lineage, "r_value"],
            full.table.loc[full.table["lineage"] == lineage, "r_value"],
            atol=1e-12,
        )


def test_lineage_marker_arguments_are_checked(datastore, lineage_fate, mark_hvgs):
    markers = datastore.trajectory.lineage_markers
    with pytest.raises(TypeError, match="fate must be an ArtifactRef"):
        markers("fate", features=mark_hvgs)
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        markers(lineage_fate, features="features")
    with pytest.raises(ValueError, match="assay-scoped feature selection"):
        markers(lineage_fate, features=datastore.snapshot_cell_selection("I"))
    with pytest.raises(TypeError, match="lineages must be a list"):
        markers(lineage_fate, features=mark_hvgs, lineages=(2,))
    with pytest.raises(ValueError, match="not a sink"):
        markers(lineage_fate, features=mark_hvgs, lineages=[99])
    with pytest.raises(ValueError, match="min_cells"):
        markers(lineage_fate, features=mark_hvgs, min_cells=0)
    with pytest.raises(ValueError, match="gene_batch_size"):
        markers(lineage_fate, features=mark_hvgs, gene_batch_size=0)
    with pytest.raises(TypeError, match="invalidate_cache"):
        markers(lineage_fate, features=mark_hvgs, invalidate_cache=1)
    with pytest.raises(TypeError):
        markers(lineage_fate, features=mark_hvgs, unknown_option=True)


# Lineage drivers


def test_lineage_drivers_match_fisher_statistics(
    datastore, lineage_fate, lineage_drivers, mark_hvgs, leiden_clustering
):
    result = datastore.trajectory.load_drivers(lineage_drivers)
    scored = _scored(datastore, lineage_fate)
    driving = ~_sink_mask(datastore, leiden_clustering, scored["fate"])
    features = _feature_indices(datastore, mark_hvgs)
    values = _normalized(datastore, scored["cells"][driving], features)
    probabilities = scored["weights"][driving]

    assert isinstance(result, LineageDriverResult)
    assert result.n_cells == int(driving.sum())
    assert result.driver_cells == scored["fate"].cell_selection
    n_cells = result.n_cells
    for column, lineage in enumerate(LINEAGE_SINKS):
        rows = result.table[result.table["lineage"] == lineage]
        expected = _weighted_pearson(values, probabilities[:, column], np.ones(n_cells))
        tested = (values > 0).sum(axis=0) >= 10
        r = rows["r_value"].to_numpy()
        np.testing.assert_allclose(r[tested], expected[tested], atol=1e-10)
        z = np.arctanh(expected[tested])
        error = 1.0 / np.sqrt(n_cells - 3)
        np.testing.assert_allclose(
            rows["p_value"].to_numpy()[tested],
            2.0 * norm.sf(np.abs(z) / error),
            rtol=1e-6,
            atol=1e-300,
        )
        quantile = norm.ppf(0.975)
        np.testing.assert_allclose(
            rows["ci_low"].to_numpy()[tested], np.tanh(z - quantile * error), atol=1e-9
        )
        np.testing.assert_allclose(
            rows["ci_high"].to_numpy()[tested], np.tanh(z + quantile * error), atol=1e-9
        )


def test_lineage_drivers_can_include_sinks_and_restrict_cells(
    datastore, lineage_fate, lineage_drivers, mark_hvgs, leiden_clustering
):
    assert (
        datastore.trajectory.drivers(lineage_fate, features=mark_hvgs)
        == lineage_drivers
    )
    scored = _scored(datastore, lineage_fate)
    with_sinks = datastore.trajectory.load_drivers(
        datastore.trajectory.drivers(
            lineage_fate, features=mark_hvgs, include_sinks=True, confidence_level=0.9
        )
    )
    assert with_sinks.n_cells == len(scored["cells"])

    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    keep = np.zeros(datastore.cells.N, dtype=bool)
    chosen = scored["cells"][np.isin(labels[scored["fate"].valid], [5, 6])]
    keep[chosen] = True
    subset = _subset_selection(datastore, "lineage_driver_test_cells", keep)
    restricted = datastore.trajectory.load_drivers(
        datastore.trajectory.drivers(lineage_fate, features=mark_hvgs, cells=subset)
    )
    assert restricted.n_cells == len(chosen)
    assert restricted.driver_cells == subset


def test_lineage_driver_arguments_are_checked(datastore, lineage_fate, mark_hvgs):
    drivers = datastore.trajectory.drivers
    with pytest.raises(TypeError, match="cells must be a cell-selection"):
        drivers(lineage_fate, features=mark_hvgs, cells="cells")
    with pytest.raises(TypeError, match="lineages must be a list"):
        drivers(lineage_fate, features=mark_hvgs, lineages=(2,))
    with pytest.raises(ValueError, match="confidence_level"):
        drivers(lineage_fate, features=mark_hvgs, confidence_level=1.0)
    with pytest.raises(TypeError, match="include_sinks"):
        drivers(lineage_fate, features=mark_hvgs, include_sinks=1)
    with pytest.raises(ValueError, match="gene_batch_size"):
        drivers(lineage_fate, features=mark_hvgs, gene_batch_size=0)
    everyone = _subset_selection(
        datastore, "lineage_driver_all_cells", np.ones(datastore.cells.N, dtype=bool)
    )
    if _selected_indices(datastore, everyone).size > len(
        _scored(datastore, lineage_fate)["cells"]
    ):
        with pytest.raises(ValueError, match="subset of the fate map"):
            drivers(lineage_fate, features=mark_hvgs, cells=everyone)
    scored = _scored(datastore, lineage_fate)
    few = np.zeros(datastore.cells.N, dtype=bool)
    few[scored["cells"][:2]] = True
    with pytest.raises(ValueError, match="at least four correlated cells"):
        drivers(
            lineage_fate,
            features=mark_hvgs,
            cells=_subset_selection(datastore, "lineage_driver_few_cells", few),
        )


# Lineage aggregation


def test_lineage_aggregation_matches_dense_binned_profiles(
    datastore, lineage_fate, lineage_aggregation, mark_hvgs
):
    result = datastore.trajectory.load_lineage_aggregation(lineage_aggregation)
    scored = _scored(datastore, lineage_fate)
    features = _feature_indices(datastore, mark_hvgs)
    values = _normalized(datastore, scored["cells"], features)
    operator, bin_ordering, totals = equal_mass_bin_operator(
        scored["ordering"], scored["weights"], 12
    )
    profiles = lineage_profiles(
        values, operator, bin_smoothing_matrix(12, 2.0), len(LINEAGE_SINKS)
    )
    expected, valid = finalize_lineage_profiles(
        profiles, values.mean(axis=0), min_expression=1e-3, z_scale=True
    )

    assert isinstance(result, LineageAggregationResult)
    np.testing.assert_array_equal(result.feature_indices, features[valid])
    np.testing.assert_allclose(result.data, expected[valid], atol=1e-10)
    np.testing.assert_allclose(result.bin_pseudotime, bin_ordering, atol=1e-12)
    np.testing.assert_allclose(result.lineage_weight, totals, rtol=1e-12)
    np.testing.assert_allclose(result.lineage_weight, scored["weights"].sum(axis=0))
    assert result.data.shape == (int(valid.sum()), len(LINEAGE_SINKS), 12)
    assert set(np.unique(result.feature_clusters)) <= set(range(1, 6))
    assert (np.diff(result.bin_pseudotime, axis=1) >= -1e-12).all()
    assert result.feature_names.shape == result.feature_ids.shape


def test_lineage_aggregation_without_scaling_keeps_weighted_means(
    datastore, lineage_fate, mark_hvgs
):
    ref = datastore.trajectory.lineage_aggregation(
        lineage_fate,
        features=mark_hvgs,
        lineages=[3],
        n_bins=8,
        smoothing=0.0,
        z_scale=False,
        n_clusters=3,
        batch_size=13,
        ann_params={"ef": 40},
    )
    result = datastore.trajectory.load_lineage_aggregation(ref)
    scored = _scored(datastore, lineage_fate)
    features = _feature_indices(datastore, mark_hvgs)
    values = _normalized(datastore, scored["cells"], features)
    weights = scored["weights"][:, [1]]
    operator, _, _ = equal_mass_bin_operator(scored["ordering"], weights, 8)
    means = (operator @ values).T
    position = np.searchsorted(features, result.feature_indices)
    np.testing.assert_allclose(result.data[:, 0, :], means[position], atol=1e-10)
    assert result.lineages == (3,)
    assert (
        datastore.trajectory.lineage_aggregation(
            lineage_fate,
            features=mark_hvgs,
            lineages=[3],
            n_bins=8,
            smoothing=0.0,
            z_scale=False,
            n_clusters=3,
            batch_size=13,
            ann_params={"ef": 40},
        )
        == ref
    )


def test_lineage_aggregation_arguments_are_checked(datastore, lineage_fate, mark_hvgs):
    aggregate = datastore.trajectory.lineage_aggregation
    two = datastore.features.snapshot(
        from_assay="RNA", feature_indexes=_feature_indices(datastore, mark_hvgs)[:2]
    )
    one = datastore.features.snapshot(
        from_assay="RNA", feature_indexes=_feature_indices(datastore, mark_hvgs)[:1]
    )
    with pytest.raises(TypeError, match="lineages must be a list"):
        aggregate(lineage_fate, features=mark_hvgs, lineages=(2,))
    with pytest.raises(ValueError, match="At least two selected features"):
        aggregate(lineage_fate, features=one)
    with pytest.raises(ValueError, match="n_neighbours must be smaller"):
        aggregate(lineage_fate, features=two, n_neighbours=2, n_clusters=1)
    with pytest.raises(ValueError, match="n_clusters cannot exceed"):
        aggregate(lineage_fate, features=two, n_neighbours=1, n_clusters=3)
    with pytest.raises(ValueError, match="below n_bins"):
        aggregate(lineage_fate, features=mark_hvgs, n_bins=100_000)
    # The error names the light lineage by its sink label.
    with pytest.raises(ValueError, match=r"Lineage 4 carries a total fate weight of"):
        aggregate(lineage_fate, features=mark_hvgs, lineages=[4, 3], n_bins=100_000)
    with pytest.raises(ValueError, match="Only 0 of"):
        aggregate(lineage_fate, features=mark_hvgs, min_exp=1e9)
    with pytest.raises(ValueError, match="nan_cluster_value"):
        aggregate(lineage_fate, features=mark_hvgs, nan_cluster_value=1)
    with pytest.raises(ValueError, match="batch_size"):
        aggregate(lineage_fate, features=mark_hvgs, batch_size=0)


def test_lineage_aggregation_reuse_builds_no_bin_operator(
    datastore, lineage_fate, mark_hvgs, lineage_aggregation, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("a reused aggregation must not bin the cells")

    monkeypatch.setattr(f"{_OPERATIONS}._equal_mass_bin_operator_impl", refuse)

    assert (
        datastore.trajectory.lineage_aggregation(
            lineage_fate, features=mark_hvgs, n_bins=12, n_clusters=5
        )
        == lineage_aggregation
    )


def test_lineage_streams_reserve_the_arrays_their_callers_hold(
    datastore, lineage_fate, mark_hvgs, monkeypatch
):
    """Feature batches are sized after what stays allocated beside them."""
    assay_type = type(datastore._get_assay("RNA"))
    original = assay_type.iter_normed_feature_wise
    reserved: list[int] = []

    def recording(self, *args, **kwargs):
        reserved.append(kwargs["resident_bytes"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(assay_type, "iter_normed_feature_wise", recording)
    # Parameters of their own make new results without touching the shared ones.
    datastore.trajectory.lineage_markers(lineage_fate, features=mark_hvgs, min_cells=9)
    datastore.trajectory.drivers(lineage_fate, features=mark_hvgs, min_cells=9)
    datastore.trajectory.lineage_aggregation(
        lineage_fate, features=mark_hvgs, n_bins=11, n_clusters=5
    )

    scored = _scored(datastore, lineage_fate)
    probabilities = scored["fate"].values.nbytes
    weights = scored["weights"].nbytes
    markers, drivers, aggregation = reserved
    # The fate probabilities and the cells' weights stay allocated, and the
    # correlations also hold their regressors and a centred copy of them.
    assert markers >= probabilities + 3 * weights
    assert drivers >= probabilities + weights
    assert aggregation >= probabilities + weights


def test_lineage_aggregation_checks_the_bin_operator_against_the_budget(
    datastore, lineage_fate, mark_hvgs, monkeypatch
):
    sizes = []

    def huge(n_cells, n_lineages, n_bins):
        sizes.append((n_cells, n_lineages, n_bins))
        return datastore.memoryBytes

    monkeypatch.setattr(f"{_OPERATIONS}._bin_operator_bytes_impl", huge)

    with pytest.raises(
        MemoryError, match=rf"Binning \d+ cells along {len(LINEAGE_SINKS)} lineages"
    ):
        datastore.trajectory.lineage_aggregation(
            lineage_fate,
            features=mark_hvgs,
            n_bins=12,
            n_clusters=5,
            invalidate_cache=True,
        )
    scored = _scored(datastore, lineage_fate)
    assert sizes == [(len(scored["cells"]), len(LINEAGE_SINKS), 12)]


# Lineage curves


def _graph_coordinates(datastore, fate) -> tuple[ArtifactRef, np.ndarray]:
    coordinates = resolve_native_graph_inputs(datastore.zw, fate.graph).coordinates
    values = np.asarray(
        artifact_group(datastore.zw, coordinates)["data"][...], dtype=np.float64
    )
    return coordinates, values


def test_curves_match_the_domain_fit_and_projection(
    datastore, lineage_fate, lineage_curves
):
    result = datastore.trajectory.load_curves(lineage_curves)
    scored = _scored(datastore, lineage_fate)
    fate = scored["fate"]
    coordinates, values = _graph_coordinates(datastore, fate)
    weights = np.zeros((len(fate.valid), len(LINEAGE_SINKS)))
    weights[fate.valid] = scored["weights"]
    ordering = np.zeros(len(fate.valid))
    ordering[fate.valid] = scored["ordering"]
    ranks = np.full_like(weights, np.nan)
    ranks[fate.valid] = lineage_ranks(scored["ordering"], scored["weights"])
    fit = fit_lineage_curves(
        [(0, values)],
        ranks,
        weights,
        ordering,
        n_dims=values.shape[1],
        n_points=40,
        degrees_of_freedom=10.0,
    )

    assert isinstance(result, LineageCurvesResult)
    assert result.coordinates == coordinates
    assert result.lineages == tuple(LINEAGE_SINKS)
    np.testing.assert_allclose(result.points, fit.points, atol=1e-9)
    np.testing.assert_allclose(result.curve_pseudotime, fit.ordering, atol=1e-9)
    np.testing.assert_array_equal(result.valid, fate.valid)
    # The weights are the fate probabilities each cell carried in the fit.
    np.testing.assert_array_equal(result.weights[fate.valid], scored["weights"])
    assert np.isnan(result.weights[~fate.valid]).all()
    for lineage in range(len(LINEAGE_SINKS)):
        position, distance = project_onto_curve(
            values[fate.valid], fit.points[lineage], fit.progress
        )
        np.testing.assert_allclose(
            result.progress[fate.valid, lineage], position, atol=1e-9
        )
        np.testing.assert_allclose(
            result.distance[fate.valid, lineage], distance, atol=1e-9
        )
    projected = result.project(values[:5])
    np.testing.assert_allclose(projected.progress, result.progress[:5], atol=1e-9)
    # Cell positions read like any cell-aligned result.
    progress = datastore.artifacts.load_values(lineage_curves)
    np.testing.assert_allclose(progress.values, result.progress, atol=0.0)
    distance = datastore.artifacts.load_values(lineage_curves, value="distance")
    np.testing.assert_allclose(distance.values, result.distance, atol=0.0)


def test_curves_accept_explicit_coordinates_and_lineage_subsets(
    datastore, lineage_fate, lineage_curves
):
    fate = datastore.trajectory.load_fate(lineage_fate)
    coordinates, _ = _graph_coordinates(datastore, fate)
    ref = datastore.trajectory.curves(
        lineage_fate,
        lineages=[3],
        coordinates=coordinates,
        n_points=40,
        batch_size=97,
    )
    subset = datastore.trajectory.load_curves(ref)
    full = datastore.trajectory.load_curves(lineage_curves)
    np.testing.assert_allclose(subset.points[0], full.points[1], atol=1e-9)
    assert datastore.trajectory.curves(lineage_fate, n_points=40) == lineage_curves


def test_curve_arguments_are_checked(datastore, lineage_fate, umap, monkeypatch):
    curves = datastore.trajectory.curves
    with pytest.raises(TypeError, match="fate must be an ArtifactRef"):
        curves("fate")
    with pytest.raises(TypeError, match="invalidate_cache"):
        curves(lineage_fate, invalidate_cache="yes")
    with pytest.raises(ValueError, match="batch_size"):
        curves(lineage_fate, batch_size=0)
    with pytest.raises(ValueError, match="n_points"):
        curves(lineage_fate, n_points=1)
    with pytest.raises(ValueError, match="df"):
        curves(lineage_fate, df=2.0)
    with pytest.raises(ValueError, match="not a sink"):
        curves(lineage_fate, lineages=[99])
    # Like the other lineage operations, curves take lineages as a list.
    with pytest.raises(TypeError, match="lineages must be a list of sink labels"):
        curves(lineage_fate, lineages=(2,))
    with pytest.raises(TypeError, match="coordinates must be an ArtifactRef"):
        curves(lineage_fate, coordinates="pca")
    with pytest.raises(ValueError, match="coordinates must be a reduction"):
        curves(lineage_fate, coordinates=umap)
    monkeypatch.setattr(
        datastore,
        "_coordinate_data",
        lambda coordinates, *, batch_size: ChunkedArray.from_numpy(np.zeros((3, 2))),
    )
    with pytest.raises(ValueError, match="do not align with the fate map"):
        curves(lineage_fate, n_points=31)


def test_curves_refuse_explicit_coordinates_of_other_cells(
    datastore, lineage_fate, monkeypatch
):
    fate = datastore.trajectory.load_fate(lineage_fate)
    coordinates, _ = _graph_coordinates(datastore, fate)
    other = datastore.snapshot_cell_selection("I")

    class _Lineage:
        cell_selection = other

    monkeypatch.setattr(
        f"{_OPERATIONS}.resolve_coordinate_inputs", lambda root, ref: _Lineage()
    )
    if other != fate.cell_selection:
        with pytest.raises(ValueError, match="cells of the trajectory"):
            datastore.trajectory.curves(lineage_fate, coordinates=coordinates)


def test_integrated_graphs_need_explicit_coordinates(datastore, lineage_fate):
    fate = datastore.trajectory.load_fate(lineage_fate)
    integrated = ArtifactRef(
        scope="datastore", kind="integrated_graph", artifact_id=new_artifact_id()
    )
    with pytest.raises(ValueError, match="no coordinates of its own"):
        datastore._lineage_coordinates(integrated, None, fate.cell_selection)


def test_curves_reuse_ranks_no_cells(
    datastore, lineage_fate, lineage_curves, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("reused curves must not rank the cells")

    monkeypatch.setattr(f"{_OPERATIONS}._lineage_ranks_impl", refuse)

    assert datastore.trajectory.curves(lineage_fate, n_points=40) == lineage_curves


def test_curves_validate_the_pseudotime_like_other_lineage_operations(
    datastore, lineage_fate, monkeypatch
):
    fate = datastore.trajectory.load_fate(lineage_fate)
    pseudotime = datastore.trajectory.load_pseudotime(fate.pseudotime)
    flat = dataclasses.replace(
        pseudotime, values=np.where(pseudotime.valid, 0.5, np.nan)
    )
    monkeypatch.setattr(datastore, "_trajectory_load_pseudotime", lambda ref: flat)

    with pytest.raises(ValueError, match="at least two distinct values"):
        datastore.trajectory.curves(lineage_fate, n_points=35)


def test_curves_check_the_memory_budget_before_reading(
    datastore, lineage_fate, monkeypatch
):
    fate = datastore.trajectory.load_fate(lineage_fate)
    pseudotime = datastore.trajectory.load_pseudotime(fate.pseudotime)
    monkeypatch.setattr(datastore, "_trajectory_load_fate", lambda ref: fate)
    monkeypatch.setattr(
        datastore, "_trajectory_load_pseudotime", lambda ref: pseudotime
    )
    monkeypatch.setattr(datastore, "memoryBytes", 4096)
    with pytest.raises(MemoryError, match="Lineage curves need about"):
        datastore.trajectory.curves(lineage_fate, n_points=33)


# Lineage tree


def test_lineage_tree_matches_the_domain_tree(
    datastore, connectivity_graph, leiden_clustering, lineage_tree
):
    result = datastore.trajectory.load_lineage_tree(lineage_tree)
    native = resolve_native_graph_inputs(datastore.zw, connectivity_graph)
    values = np.asarray(
        artifact_group(datastore.zw, native.coordinates)["data"][...], dtype=np.float64
    )
    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    clusters = sorted(set(labels.tolist()))
    codes = np.searchsorted(clusters, labels)
    counts, means, covariances = cluster_moments(
        [(0, values)], codes, n_clusters=len(clusters), n_dims=values.shape[1]
    )
    tree = spanning_tree(cluster_distances(means, covariances), root=0)

    assert isinstance(result, LineageTreeResult)
    assert result.clusters == tuple(clusters)
    np.testing.assert_array_equal(result.sizes, counts)
    np.testing.assert_allclose(result.means, means, atol=1e-10)
    np.testing.assert_array_equal(
        result.edges[["source", "target"]].to_numpy(),
        np.asarray(clusters)[tree.edges],
    )
    np.testing.assert_allclose(result.edges["length"], tree.lengths, rtol=1e-10)
    assert result.root == 1
    assert result.lineages == tuple(
        tuple(clusters[index] for index in path) for path in tree.lineages
    )
    assert result.candidate_sinks == tuple(
        clusters[leaf] for leaf in tree.leaves if leaf != 0
    )


def test_lineage_tree_ends_become_leaves(
    datastore, connectivity_graph, leiden_clustering
):
    ref = datastore.trajectory.lineage_tree(
        connectivity_graph, leiden_clustering, root=1, ends=[2, 3], batch_size=101
    )
    result = datastore.trajectory.load_lineage_tree(ref)
    assert {2, 3} <= set(result.leaves)
    assert result.ends == (2, 3)
    unrooted = datastore.trajectory.load_lineage_tree(
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering)
    )
    assert unrooted.root is None
    assert unrooted.lineages == ()
    assert unrooted.candidate_sinks == unrooted.leaves
    assert (
        datastore.trajectory.lineage_tree(
            connectivity_graph, leiden_clustering, root=1, ends=[2, 3], batch_size=101
        )
        == ref
    )
    # The tree does not depend on the order of the ends, so any order reuses
    # the tree, whose record lists the ends in cluster order.
    assert (
        datastore.trajectory.lineage_tree(
            connectivity_graph, leiden_clustering, root=1, ends=[3, 2], batch_size=101
        )
        == ref
    )
    assert inspect_artifact(datastore.zw, ref).parameters["ends"] == [2, 3]


def _labels_artifact(datastore, selection: ArtifactRef, values: np.ndarray):
    ids = np.asarray(datastore.cells.fetch_all("ids"))[
        _selected_indices(datastore, selection)
    ]
    return resolve_metadata_snapshot(
        datastore.zw,
        values=values,
        row_ids=ids,
        operation="test_lineage_tree_labels",
        parameters={},
        inputs={"cell_selection": selection},
        source_columns=["labels"],
    )


def test_lineage_tree_arguments_are_checked(
    datastore, connectivity_graph, leiden_clustering, umap, monkeypatch
):
    tree = datastore.trajectory.lineage_tree
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        tree("graph", leiden_clustering)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        tree(connectivity_graph, "labels")
    with pytest.raises(ValueError):
        tree(umap, leiden_clustering)
    with pytest.raises(TypeError, match="ends must be a list"):
        tree(connectivity_graph, leiden_clustering, ends=(2,))
    with pytest.raises(ValueError, match="root cannot also be an end"):
        tree(connectivity_graph, leiden_clustering, root=2, ends=[2])
    with pytest.raises(ValueError, match="root 99 is not a cluster label"):
        tree(connectivity_graph, leiden_clustering, root=99)
    with pytest.raises(ValueError, match="End 99 is not a cluster label"):
        tree(connectivity_graph, leiden_clustering, ends=[99])
    with pytest.raises(ValueError, match="batch_size"):
        tree(connectivity_graph, leiden_clustering, batch_size=0)
    with pytest.raises(TypeError, match="invalidate_cache"):
        tree(connectivity_graph, leiden_clustering, invalidate_cache=None)

    selection = datastore.trajectory.load_lineage_tree(
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering)
    ).cell_selection
    n_cells = _selected_indices(datastore, selection).size
    constant = _labels_artifact(datastore, selection, np.full(n_cells, "same"))
    with pytest.raises(ValueError, match="at least two clusters"):
        tree(connectivity_graph, constant)
    other = _labels_artifact(
        datastore,
        datastore.snapshot_cell_selection("I"),
        np.full(datastore.cells.fetch("I").size, "x"),
    )
    if datastore.snapshot_cell_selection("I") != selection:
        with pytest.raises(ValueError, match="cells of the graph"):
            tree(connectivity_graph, other)


def test_lineage_tree_refuses_unlabelled_cells_and_bad_shapes(
    datastore, connectivity_graph, leiden_clustering, monkeypatch
):
    selection = datastore.trajectory.load_lineage_tree(
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering)
    ).cell_selection
    labels = np.asarray(datastore.artifacts.load_values(leiden_clustering).values)
    missing = np.zeros(labels.size, dtype=bool)
    missing[0] = True
    monkeypatch.setattr(
        f"{_OPERATIONS}._load_cell_artifact_values",
        lambda root, ref: (labels, selection, missing),
    )
    with pytest.raises(ValueError, match="Every cell needs a label"):
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering, root=2)
    monkeypatch.setattr(
        f"{_OPERATIONS}._load_cell_artifact_values",
        lambda root, ref: (labels.reshape(-1, 1), selection, None),
    )
    with pytest.raises(ValueError, match="one label per cell"):
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering, root=2)


def test_lineage_tree_checks_alignment_and_memory(
    datastore, connectivity_graph, leiden_clustering, monkeypatch
):
    monkeypatch.setattr(
        datastore,
        "_coordinate_data",
        lambda coordinates, *, batch_size: ChunkedArray.from_numpy(np.zeros((3, 2))),
    )
    with pytest.raises(ValueError, match="do not align with the graph"):
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering, root=3)
    monkeypatch.undo()
    monkeypatch.setattr(datastore, "memoryBytes", 1024)
    with pytest.raises(MemoryError, match="lineage tree needs about"):
        datastore.trajectory.lineage_tree(connectivity_graph, leiden_clustering, root=4)


# Read-only stores


def test_lineage_producers_refuse_read_only_stores(
    datastore, lineage_fate, mark_hvgs, connectivity_graph, leiden_clustering
):
    from cytearc.datastore.datastore import DataStore

    read_only = DataStore(datastore.zarr_loc, default_assay="RNA", zarr_mode="r")
    trajectory = read_only.trajectory
    with pytest.raises(PermissionError, match="run_lineage_marker_search"):
        trajectory.lineage_markers(lineage_fate, features=mark_hvgs, min_cells=3)
    with pytest.raises(PermissionError, match="run_lineage_driver_search"):
        trajectory.drivers(lineage_fate, features=mark_hvgs, min_cells=3)
    with pytest.raises(PermissionError, match="run_lineage_aggregation"):
        trajectory.lineage_aggregation(lineage_fate, features=mark_hvgs, n_bins=7)
    with pytest.raises(PermissionError, match="fit_lineage_curves"):
        trajectory.curves(lineage_fate, n_points=17)
    with pytest.raises(PermissionError, match="build_lineage_tree"):
        trajectory.lineage_tree(connectivity_graph, leiden_clustering, root=5)


# Loaders refuse changed records


def _missing_ref(kind: str, scope: str = "assay") -> ArtifactRef:
    return ArtifactRef(
        scope=scope,
        kind=kind,
        artifact_id=new_artifact_id(),
        assay="RNA" if scope == "assay" else None,
    )


@pytest.mark.parametrize(
    ("fixture", "loader", "kind"),
    [
        ("lineage_markers", "load_lineage_markers", "lineage_markers"),
        ("lineage_drivers", "load_drivers", "lineage_drivers"),
        ("lineage_aggregation", "load_lineage_aggregation", "lineage_aggregation"),
        ("lineage_curves", "load_curves", "lineage_curves"),
        ("lineage_tree", "load_lineage_tree", "lineage_tree"),
    ],
)
def test_loaders_refuse_wrong_types_kinds_and_missing_artifacts(
    request, datastore, fixture, loader, kind
):
    load = getattr(datastore.trajectory, loader)
    ref = request.getfixturevalue(fixture)
    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        load("ref")
    other = "lineage_tree" if kind != "lineage_tree" else "lineage_curves"
    with pytest.raises(ValueError):
        load(_missing_ref(other))
    with pytest.raises(ValueError, match="unavailable or invalid"):
        load(_missing_ref(kind, ref.scope))


_FEATURE_LOADERS = {
    "lineage_markers": ("load_lineage_markers", LINEAGE_MARKER_PAYLOAD),
    "lineage_drivers": ("load_drivers", LINEAGE_DRIVER_PAYLOAD),
    "lineage_aggregation": ("load_lineage_aggregation", LINEAGE_AGGREGATION_PAYLOAD),
}


@pytest.mark.parametrize("fixture", sorted(_FEATURE_LOADERS))
def test_feature_loaders_refuse_changed_records(request, datastore, fixture):
    loader, payload = _FEATURE_LOADERS[fixture]
    load = getattr(datastore.trajectory, loader)
    ref = request.getfixturevalue(fixture)
    assert load(_forged(datastore, ref, payload=payload)).ref != ref

    tampered = [
        ({"operation": "other_operation"}, "unavailable or invalid"),
        (_without_input(datastore, ref, "fate"), "inputs"),
        (_with_parameters(datastore, ref, lineages=[]), "parameters do not match"),
        (
            _with_inputs(datastore, ref, ordered_feature_ids_fingerprint=7),
            "feature identities are malformed",
        ),
        (
            _with_inputs(datastore, ref, ordered_feature_ids_fingerprint="0" * 64),
            "feature ID identity has changed",
        ),
        (
            _with_inputs(datastore, ref, dataset_fingerprint="0" * 64),
            "dataset identity has changed",
        ),
        (_with_parameters(datastore, ref, lineages=[2, 99]), "lineages are not sinks"),
    ]
    for provenance, message in tampered:
        with pytest.raises(ValueError, match=message):
            load(_forged(datastore, ref, payload=payload, provenance=provenance))
    fate = datastore.trajectory.load_fate(
        ArtifactRef.from_dict(_record(datastore, ref)["inputs"]["fate"])
    )
    changed_selection = datastore.snapshot_cell_selection("I")
    if changed_selection != fate.cell_selection:
        with pytest.raises(ValueError, match="does not match its fate map"):
            load(
                _forged(
                    datastore,
                    ref,
                    payload=payload,
                    provenance=_with_inputs(
                        datastore, ref, cell_selection=changed_selection
                    ),
                )
            )
    with pytest.raises(ValueError, match="ref must be an assay-scoped"):
        load(
            ArtifactRef(scope="datastore", kind=ref.kind, artifact_id=new_artifact_id())
        )


@pytest.mark.parametrize("fixture", sorted(_FEATURE_LOADERS))
def test_feature_loaders_report_other_failures_as_they_are(
    request, datastore, fixture, monkeypatch
):
    """Only a lineage mismatch is reported as one; other errors keep their cause."""
    load = getattr(datastore.trajectory, _FEATURE_LOADERS[fixture][0])
    ref = request.getfixturevalue(fixture)

    def broken_axes(fate, lineages):
        raise ValueError("Pseudotime column 'x' must contain at least two values")

    monkeypatch.setattr(datastore, "_lineage_axes", broken_axes)

    with pytest.raises(ValueError, match="must contain at least two values") as caught:
        load(ref)
    assert "lineages are not sinks" not in str(caught.value)


@pytest.mark.parametrize(
    ("fixture", "name"),
    [
        ("lineage_markers", "r_value"),
        ("lineage_drivers", "ci_low"),
        ("lineage_aggregation", "lineage_weight"),
    ],
)
def test_feature_loaders_refuse_changed_payloads(request, datastore, fixture, name):
    loader, payload = _FEATURE_LOADERS[fixture]
    ref = request.getfixturevalue(fixture)
    values = np.asarray(artifact_group(datastore.zw, ref)[name][...]).copy()
    values[...] = np.inf
    with pytest.raises(ValueError, match="payload is invalid"):
        getattr(datastore.trajectory, loader)(
            _forged(datastore, ref, payload=payload, arrays={name: values})
        )


def test_driver_loader_recounts_its_cells(datastore, lineage_drivers):
    forged = _forged(
        datastore, lineage_drivers, payload=LINEAGE_DRIVER_PAYLOAD, attrs={"n_cells": 3}
    )
    with pytest.raises(ValueError, match="payload is invalid"):
        datastore.trajectory.load_drivers(forged)


def test_curve_loader_refuses_changed_records(datastore, lineage_curves, lineage_fate):
    load = datastore.trajectory.load_curves
    assert load(_forged(datastore, lineage_curves, payload=LINEAGE_CURVE_PAYLOAD))
    tampered = [
        ({"operation": "other_operation"}, "unavailable or invalid"),
        (_without_input(datastore, lineage_curves, "coordinates"), "inputs"),
        (_with_parameters(datastore, lineage_curves, n_points=1), "malformed"),
        (
            _with_parameters(datastore, lineage_curves, lineages=[2, 99]),
            "not sinks of the fate map",
        ),
    ]
    for provenance, message in tampered:
        with pytest.raises(ValueError, match=message):
            load(
                _forged(
                    datastore,
                    lineage_curves,
                    payload=LINEAGE_CURVE_PAYLOAD,
                    provenance=provenance,
                )
            )
    fate = datastore.trajectory.load_fate(lineage_fate)
    changed_selection = datastore.snapshot_cell_selection("I")
    if changed_selection != fate.cell_selection:
        with pytest.raises(ValueError, match="do not use their fate map's cells"):
            load(
                _forged(
                    datastore,
                    lineage_curves,
                    payload=LINEAGE_CURVE_PAYLOAD,
                    provenance=_with_inputs(
                        datastore, lineage_curves, cell_selection=changed_selection
                    ),
                )
            )
    progress = np.asarray(
        artifact_group(datastore.zw, lineage_curves)["progress"][...]
    ).copy()
    progress[0, 0] = 2.0
    with pytest.raises(ValueError, match="payload is invalid"):
        load(
            _forged(
                datastore,
                lineage_curves,
                payload=LINEAGE_CURVE_PAYLOAD,
                arrays={"progress": progress},
            )
        )


def test_curve_loader_requires_the_fate_scope(datastore, lineage_curves):
    source = artifact_group(datastore.zw, lineage_curves)
    forged = ArtifactRef(
        scope="datastore", kind="lineage_curves", artifact_id=new_artifact_id()
    )
    group = datastore.zw.create_group(artifact_path(forged))
    for name in source.array_keys():
        group.create_array(name, data=np.asarray(source[name][...]))
    group.attrs.update({**dict(source.attrs), "artifact_id": forged.artifact_id})
    with pytest.raises(ValueError, match="does not share its fate scope"):
        datastore.trajectory.load_curves(forged)


def test_tree_loader_refuses_changed_records(
    datastore, lineage_tree, leiden_clustering
):
    load = datastore.trajectory.load_lineage_tree
    assert load(_forged(datastore, lineage_tree, payload=LINEAGE_TREE_PAYLOAD))
    tampered = [
        ({"operation": "other_operation"}, "unavailable or invalid"),
        (_without_input(datastore, lineage_tree, "labels"), "inputs"),
        (_with_parameters(datastore, lineage_tree, ends="x"), "malformed"),
        (
            _with_parameters(datastore, lineage_tree, root=99),
            "root or ends are not cluster labels",
        ),
    ]
    for provenance, message in tampered:
        with pytest.raises(ValueError, match=message):
            load(
                _forged(
                    datastore,
                    lineage_tree,
                    payload=LINEAGE_TREE_PAYLOAD,
                    provenance=provenance,
                )
            )
    with pytest.raises(ValueError, match="payload is invalid"):
        load(
            _forged(
                datastore,
                lineage_tree,
                payload=LINEAGE_TREE_PAYLOAD,
                attrs={"cluster_labels": ["a"]},
            )
        )


def test_tree_loader_checks_its_coordinates_cells(datastore, lineage_tree, monkeypatch):
    other = datastore.snapshot_cell_selection("I")
    recorded = ArtifactRef.from_dict(
        _record(datastore, lineage_tree)["inputs"]["cell_selection"]
    )
    if other == recorded:
        pytest.skip("the session graph uses every cell")

    class _Lineage:
        cell_selection = other

    monkeypatch.setattr(
        f"{_OPERATIONS}.resolve_coordinate_inputs", lambda root, ref: _Lineage()
    )
    with pytest.raises(ValueError, match="coordinates do not hold its cells"):
        datastore.trajectory.load_lineage_tree(lineage_tree)
