"""Native Milo against miloR 2.6.0 (edgeR 4.8.2, limma 3.66.0).

The stage tests run the storage-free steps on miloR's inputs. The pipeline
test imports the same reduced dimensions into a store and runs
``DataStore.abundance`` end to end: the ANN graph must reproduce the exact
kNN graph, and the seed must reproduce the sampled vertices.

Sampling, membership, counts, k-distances, and annotation match miloR
exactly. The neighbourhood test runs edgeR's steps in edgePython and matches
edgeR to near its convergence tolerance; logFC differs by up to about 5e-4.
"""

import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cytearc.abundance import milo
from cytearc.datastore.datastore import DataStore
from cytearc.embeddings.imported import write_imported_coordinates
from cytearc.metadata.selection import CellField
from cytearc.stats.design import treatment_design
from cytearc.storage.artifacts import fingerprint_array
from cytearc.writers import SparseToZarr

FIXTURE = Path(__file__).with_name("milor_2_6_0_golden.json")

pytest.importorskip("edgepython")

# Largest differences from miloR on these cases, with headroom: p-values,
# F, and spatial FDR within 1.5e-6, 4.2e-5, and 1.5e-6 relative, logFC within
# 5.3e-4 absolute (batch_and_condition, condition_refined).
_LOG_FC_ATOL = 1e-3
_P_VALUE_RTOL = 1e-5
_F_RTOL = 1e-4


@cache
def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _names() -> list[str]:
    return [case["name"] for case in _fixture()["inputs"]["cases"]]


def _case(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    index = _names().index(name)
    return _fixture()["inputs"]["cases"][index], _fixture()["milor"][index]


def _close(actual: Any, expected: Any, *, rtol: float = 1e-7) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=1e-12,
    )


def _design(case: dict[str, Any], samples: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table = case["designDf"]
    row = {sample: idx for idx, sample in enumerate(table["sample"])}
    variables = {}
    if "batch" in case["formula"]:
        variables["batch"] = [table["batch"][row[sample]] for sample in samples]
    variables["condition"] = [table["condition"][row[sample]] for sample in samples]
    design = treatment_design(variables)
    contrast = np.zeros(design.matrix.shape[1])
    contrast[-1] = 1
    if case["modelContrasts"]:
        # ~0 + condition with "b - c" spans the same model as the
        # treatment-coded ~condition with the contrast b - c.
        contrast = np.array([0.0, 1.0, -1.0])
    return design.matrix, contrast


def _assert_near_milor(
    expected: dict[str, Any],
    *,
    norm_factors: Any,
    log_cpm: Any,
    log_fc: Any,
    p_value: Any,
    spatial_fdr: Any,
) -> None:
    _close(norm_factors, expected["normFactors"], rtol=1e-12)
    tested = np.isfinite(np.asarray(expected["pValue"], dtype=np.float64))
    _close(np.asarray(log_cpm)[tested], np.asarray(expected["logCpm"])[tested])
    np.testing.assert_allclose(
        np.asarray(log_fc, dtype=np.float64)[tested],
        np.asarray(expected["logFc"], dtype=np.float64)[tested],
        rtol=0,
        atol=_LOG_FC_ATOL,
    )
    _close(p_value, expected["pValue"], rtol=_P_VALUE_RTOL)
    _close(spatial_fdr, expected["spatialFdr"], rtol=_P_VALUE_RTOL)
    np.testing.assert_array_equal(
        np.asarray(spatial_fdr, dtype=np.float64) < 0.1,
        np.asarray(expected["spatialFdr"], dtype=np.float64) < 0.1,
    )


def test_fixture_pins_milor_and_its_inputs():
    fixture = _fixture()
    provenance = fixture["provenance"]
    assert provenance["packageVersion"] == "2.6.0"
    assert provenance["edgeRVersion"] == "4.8.2"
    cases = fixture["inputs"]["cases"]
    encoded = json.dumps(cases, separators=(",", ":"), sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == fixture["inputs"]["sha256"]


@pytest.mark.parametrize("name", _names())
def test_milo_stages_match_milor(name):
    case, expected = _case(name)
    coordinates = np.asarray(case["coordinates"], dtype=np.float64)
    knn = np.asarray(case["knn"], dtype=np.int64) - 1
    n_cells = coordinates.shape[0]
    sampled = milo.sample_index_cells(n_cells, case["prop"], seed=case["samplingSeed"])
    np.testing.assert_array_equal(sampled, np.asarray(case["randomVertices"]) - 1)

    def nearest(points: np.ndarray) -> np.ndarray:
        squared = ((points[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(squared, axis=1)

    if case["refined"]:
        index = milo.refine_index_cells(
            sampled, knn[sampled], lambda cells: coordinates[cells], nearest
        )
    else:
        index = milo.unique_in_order(sampled)
    np.testing.assert_array_equal(index, expected["refinedVertices"])

    position = np.full(n_cells, -1, dtype=np.int32)
    position[index] = np.arange(index.size)
    reverse = milo.reverse_neighbors(index, knn[index])
    blocks = [
        milo.block_membership(
            start, knn[start : start + 50], position, reverse, index.size
        )
        for start in range(0, n_cells, 50)
    ]
    row_counts = np.concatenate([block[0] for block in blocks])
    ids = np.concatenate([block[1] for block in blocks])
    cells = np.repeat(np.arange(n_cells), row_counts)
    for nhood, members in enumerate(expected["nhoodMembers"]):
        np.testing.assert_array_equal(np.sort(cells[ids == nhood]), members)

    samples = np.asarray(case["sample"])
    order = [str(value) for value in milo.unique_in_order(samples)]
    assert order == expected["sampleOrder"]
    codes = np.asarray([order.index(value) for value in samples])
    counts = np.zeros((index.size, len(order)), dtype=np.int64)
    milo.count_block(counts, row_counts, ids, codes)
    np.testing.assert_array_equal(counts, expected["nhoodCounts"])

    kth = milo.kth_distances(coordinates[index], coordinates[knn[index]])
    _close(kth, expected["kDistance"], rtol=1e-12)

    design, contrast = _design(case, order)
    fit = milo.differential_abundance(
        counts,
        design,
        contrast,
        kth_distance=kth,
        norm_method=case["normMethod"],
        min_mean=case["minMean"],
    )
    np.testing.assert_array_equal(
        fit.tested, np.isfinite(np.asarray(expected["pValue"], dtype=np.float64))
    )
    _assert_near_milor(
        expected,
        norm_factors=fit.norm_factors,
        log_cpm=fit.log_cpm,
        log_fc=fit.log_fc,
        p_value=fit.p_value,
        spatial_fdr=fit.spatial_fdr,
    )

    labels = np.asarray(case["label"])
    levels = tuple(str(value) for value in milo.unique_in_order(labels))
    label_codes = np.asarray([levels.index(value) for value in labels])
    label_counts = np.zeros((index.size, len(levels)), dtype=np.int64)
    milo.count_block(label_counts, row_counts, ids, label_codes)
    annotation = milo.annotate_neighborhoods(label_counts, levels)
    assert list(annotation.label) == expected["labelMajority"]
    _close(annotation.fraction, expected["labelFraction"], rtol=1e-14)


@pytest.mark.parametrize("name", _names())
def test_neighborhood_test_matches_edger(name):
    case, expected = _case(name)
    design, contrast = _design(case, expected["sampleOrder"])
    fit = milo.differential_abundance(
        np.asarray(expected["nhoodCounts"]),
        design,
        contrast,
        kth_distance=np.asarray(expected["kDistance"]),
        norm_method=case["normMethod"],
        min_mean=case["minMean"],
    )
    _close(fit.trended_dispersion, expected["trendedDispersion"], rtol=1e-8)
    _close(fit.f_statistic, expected["f"], rtol=_F_RTOL)
    _assert_near_milor(
        expected,
        norm_factors=fit.norm_factors,
        log_cpm=fit.log_cpm,
        log_fc=fit.log_fc,
        p_value=fit.p_value,
        spatial_fdr=fit.spatial_fdr,
    )


def test_neighborhood_test_names_the_extra_without_edgepython(monkeypatch):
    monkeypatch.setitem(sys.modules, "edgepython", None)
    with pytest.raises(ImportError, match=r"cytearc\[milo\]"):
        milo.differential_abundance(
            np.array([[5.0, 6.0, 7.0, 8.0]] * 4),
            np.column_stack([np.ones(4), [0.0, 0.0, 1.0, 1.0]]),
            np.array([0.0, 1.0]),
            kth_distance=None,
        )


def _build_store(directory: Path, case: dict[str, Any]) -> tuple[DataStore, Any]:
    coordinates = np.asarray(case["coordinates"], dtype=np.float32)
    n_cells = coordinates.shape[0]
    cell_ids = np.asarray([f"cell{idx:04d}" for idx in range(n_cells)])
    counts = np.random.default_rng(3).integers(
        1, 20, size=(n_cells, 5), dtype=np.uint32
    )
    path = directory / "cells.zarr"
    SparseToZarr(
        csr_matrix(counts),
        str(path),
        cell_ids=cell_ids,
        feature_ids=[f"gene{idx}" for idx in range(5)],
        nthreads=1,
    ).dump()
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    table = case["designDf"]
    row = {sample: idx for idx, sample in enumerate(table["sample"])}
    store.cells.insert("sample", np.asarray(case["sample"]))
    store.cells.insert("label", np.asarray(case["label"]))
    for column in ("condition", "batch"):
        store.cells.insert(
            column, np.asarray([table[column][row[value]] for value in case["sample"]])
        )
    imported = write_imported_coordinates(
        store.zw,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=hashlib.sha256(coordinates.tobytes()).digest(),
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=cell_ids,
        cell_selection=store.snapshot_cell_selection("I"),
    )
    index = store.graph.ann_index(imported, ann_efc=400, ann_ef=400, ann_m=64)
    neighbors = store.graph.neighbors(index, k=len(case["knn"][0]))
    return store, neighbors


@pytest.mark.slow
@pytest.mark.parametrize("name", ["condition_refined", "batch_and_condition"])
def test_datastore_pipeline_matches_milor(tmp_path, name):
    case, expected = _case(name)
    store, neighbors = _build_store(tmp_path, case)
    stored = store.artifacts.load(neighbors)["indices"][:]
    exact = np.asarray(case["knn"]) - 1
    for row in range(exact.shape[0]):
        assert set(stored[row]) == set(exact[row])

    nhoods = store.abundance.make_neighborhoods(
        neighbors, prop=case["prop"], seed=case["samplingSeed"]
    )
    loaded = store.abundance.load_neighborhoods(nhoods)
    np.testing.assert_array_equal(loaded.index_rows, expected["refinedVertices"])
    np.testing.assert_array_equal(
        loaded.sizes, [len(members) for members in expected["nhoodMembers"]]
    )
    _close(loaded.kth_distance, expected["kDistance"], rtol=1e-6)

    counts = store.abundance.count_neighborhoods(nhoods, sample_by="sample")
    table = store.abundance.load_neighborhood_counts(counts)
    assert list(table.samples) == expected["sampleOrder"]
    np.testing.assert_array_equal(table.counts, expected["nhoodCounts"])

    covariates = ["batch"] if "batch" in case["formula"] else []
    result = store.abundance.test_neighborhoods(
        counts, condition_by="condition", covariates=covariates
    )
    assert result.contrast == {"conditiontreated": 1.0}
    frame = result.table
    _assert_near_milor(
        expected,
        norm_factors=result.norm_factors,
        log_cpm=frame["log_cpm"],
        log_fc=frame["log_fc"],
        p_value=frame["p_value"],
        spatial_fdr=frame["spatial_fdr"],
    )
    reloaded = store.abundance.load_neighborhood_tests(result.artifact)
    np.testing.assert_array_equal(reloaded.table.to_numpy(), frame.to_numpy())
    again = store.abundance.test_neighborhoods(
        counts, condition_by="condition", covariates=covariates
    )
    assert again.artifact == result.artifact

    annotation = store.abundance.annotate_neighborhoods(
        result.artifact, CellField("label", kind="categorical")
    )
    assert list(annotation["label"]) == expected["labelMajority"]
    _close(annotation["fraction"], expected["labelFraction"], rtol=1e-14)
