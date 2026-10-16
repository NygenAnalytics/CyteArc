"""DataStore.abundance: persisted propeller tests over a synthetic store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc import DataStore
from cytearc.abundance.propeller import count_categories, propeller
from cytearc.metadata.selection import CellField, StudyDesign
from cytearc.storage.artifacts import ArtifactRef

N_SAMPLES = 8
CELLS_PER_SAMPLE = 60
N_CELLS = N_SAMPLES * CELLS_PER_SAMPLE
N_GENES = 25


def _design() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    sample = np.repeat([f"s{idx}" for idx in range(N_SAMPLES)], CELLS_PER_SAMPLE)
    condition = np.where(
        np.repeat(np.arange(N_SAMPLES), CELLS_PER_SAMPLE) < N_SAMPLES // 2,
        "ctrl",
        "treated",
    )
    cell_type = np.empty(N_CELLS, dtype=object)
    for idx in range(N_SAMPLES):
        rows = slice(idx * CELLS_PER_SAMPLE, (idx + 1) * CELLS_PER_SAMPLE)
        weights = (
            [0.5, 0.3, 0.2, 0.0] if idx < N_SAMPLES // 2 else [0.25, 0.3, 0.2, 0.25]
        )
        weights = np.asarray(weights) + 0.02
        cell_type[rows] = rng.choice(
            ["B", "Mono", "NK", "T"], size=CELLS_PER_SAMPLE, p=weights / weights.sum()
        )
    mixed = condition.copy()
    mixed[0] = "treated"
    leiden = pd.Series(cell_type).map({"B": 0, "Mono": 1, "NK": 2, "T": 3}).to_numpy()
    gappy_sample = pd.array(sample, dtype="string")
    gappy_sample[3] = pd.NA
    return {
        "sample": sample,
        "condition": condition,
        "cell_type": cell_type.astype(str),
        "mixed_condition": mixed,
        "leiden": leiden,
        "gappy_sample": np.asarray(gappy_sample, dtype=object),
    }


def _write_h5ad(path: Path) -> None:
    anndata = pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(0)
    design = _design()
    counts = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    gappy = pd.array(
        [None if value is pd.NA else value for value in design["gappy_sample"]],
        dtype="string",
    )
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(design["sample"]),
            "condition": pd.Categorical(design["condition"]),
            "cell_type": pd.Categorical(design["cell_type"]),
            "mixed_condition": pd.Categorical(design["mixed_condition"]),
            "leiden": pd.array(design["leiden"], dtype="Int64"),
            "gappy_sample": pd.Categorical(gappy),
        },
        index=[f"cell{idx}" for idx in range(N_CELLS)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"G{idx}" for idx in range(N_GENES)]},
        index=[f"ENSG{idx:05d}" for idx in range(N_GENES)],
    )
    anndata.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


@pytest.fixture(scope="module")
def abundance_store(tmp_path_factory) -> tuple[Path, ArtifactRef]:
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    directory = tmp_path_factory.mktemp("abundance")
    h5ad_path = directory / "cells.h5ad"
    _write_h5ad(h5ad_path)
    store = directory / "cells.zarr"
    reader = H5adReader(str(h5ad_path), cluster_keys=("leiden",))
    try:
        result = H5adToZarr(reader, zarr_loc=str(store), nthreads=1).dump()
    finally:
        reader.h5.close()
    DataStore(str(store), default_assay="RNA", min_features_per_cell=0)
    return store, result.clusterArtifacts["leiden"]


def _open(store: Path, mode: str = "r+") -> DataStore:
    return DataStore(
        str(store), default_assay="RNA", min_features_per_cell=0, zarr_mode=mode
    )


def _expected_table() -> pd.DataFrame:
    design = _design()
    counts, categories, samples = count_categories(
        design["cell_type"], design["sample"]
    )
    groups = dict(zip(design["sample"], design["condition"], strict=True))
    return propeller(
        counts,
        categories=categories,
        samples=samples,
        sample_groups=[groups[sample] for sample in samples],
    ).table


def test_test_categories_persists_and_reloads(abundance_store):
    store, _ = abundance_store
    ds = _open(store)
    result = ds.abundance.test_categories(
        CellField("cell_type", kind="categorical"),
        sample_by="sample",
        condition_by="condition",
        invalidate_cache=True,
    )
    assert result.artifact is not None
    assert result.artifact.kind == "category_tests"
    assert result.method == "t_test"
    assert result.groups == ("ctrl", "treated")
    pd.testing.assert_frame_equal(result.table, _expected_table())
    assert result.table.loc["T", "t_statistic"] < 0
    assert result.table.loc["T", "fdr"] < 0.05
    loaded = ds.abundance.load_category_tests(result.artifact)
    pd.testing.assert_frame_equal(loaded.table, result.table)
    pd.testing.assert_frame_equal(loaded.counts(), result.counts())
    pd.testing.assert_frame_equal(loaded.proportions(), result.proportions())
    assert loaded.sample_groups().to_dict() == result.sample_groups().to_dict()
    assert loaded.category_field == CellField("cell_type", kind="categorical")

    again = ds.abundance.test_categories(
        CellField("cell_type", kind="categorical"),
        study_design=StudyDesign(sample_by="sample", condition_by="condition"),
    )
    assert again.artifact == result.artifact
    lineage = ds.artifacts.list(kind="category_tests", scope="datastore")
    assert result.artifact in set(lineage)


def test_test_categories_reads_cluster_artifacts_and_orders_groups(abundance_store):
    store, leiden = abundance_store
    ds = _open(store)
    result = ds.abundance.test_categories(
        leiden,
        sample_by="sample",
        condition_by="condition",
        groups=["treated", "ctrl"],
        skip_save=True,
    )
    assert result.artifact is None
    assert result.categories == leiden
    assert list(result.table.index) == ["0", "1", "2", "3"]
    expected = _expected_table()
    np.testing.assert_allclose(
        result.table["t_statistic"].to_numpy(), -expected["t_statistic"].to_numpy()
    )


def test_test_categories_rejects_invalid_designs(abundance_store):
    store, _ = abundance_store
    ds = _open(store)
    field = CellField("cell_type", kind="categorical")
    with pytest.raises(ValueError, match="belong to one condition"):
        ds.abundance.test_categories(
            field, sample_by="sample", condition_by="mixed_condition", skip_save=True
        )
    with pytest.raises(ValueError, match="missing values"):
        ds.abundance.test_categories(
            field, sample_by="gappy_sample", condition_by="condition", skip_save=True
        )
    with pytest.raises(ValueError, match="required"):
        ds.abundance.test_categories(field, sample_by="sample", skip_save=True)
    with pytest.raises(ValueError, match="at least two groups"):
        ds.abundance.test_categories(
            field,
            sample_by="sample",
            condition_by="condition",
            groups=["ctrl"],
            skip_save=True,
        )
    with pytest.raises(ValueError, match="must reference category_tests"):
        ds.abundance.load_category_tests(
            ArtifactRef(
                scope="datastore", kind="statistical_tests", artifact_id="0" * 64
            )
        )


def test_test_categories_requires_a_writable_store_to_save(abundance_store):
    store, _ = abundance_store
    ds = _open(store, mode="r")
    with pytest.raises(PermissionError, match="test_categories"):
        ds.abundance.test_categories(
            CellField("cell_type", kind="categorical"),
            sample_by="sample",
            condition_by="condition",
            transform="asin",
        )
