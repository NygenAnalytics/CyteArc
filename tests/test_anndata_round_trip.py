"""AnnData round trips: artifacts in exports and external coordinates in imports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import cytearc
from cytearc.readers import H5adReader, inspect_h5ad
from cytearc.storage.artifacts import (
    artifact_group,
    fingerprint_array,
    fingerprint_strings,
)
from cytearc.storage.refs import ArtifactRef
from cytearc.writers import H5adToZarr

anndata = pytest.importorskip("anndata")

_N_CELLS = 48
_N_GENES = 30


def _source_counts() -> np.ndarray:
    """Counts of two groups of cells that express different genes."""
    rng = np.random.default_rng(17)
    rates = np.full((_N_CELLS, _N_GENES), 0.5)
    rates[: _N_CELLS // 2, :10] = 6.0
    rates[_N_CELLS // 2 :, 10:20] = 6.0
    return rng.poisson(rates).astype(np.float32)


def _import_store(directory: Path) -> tuple[cytearc.DataStore, object]:
    """Import an AnnData file with a UMAP into a store."""
    counts = _source_counts()
    adata = anndata.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame(index=[f"cell{index}" for index in range(_N_CELLS)]),
        var=pd.DataFrame(index=[f"gene{index}" for index in range(_N_GENES)]),
    )
    adata.obsm["X_umap"] = np.column_stack(
        [np.arange(_N_CELLS, dtype=np.float32), np.ones(_N_CELLS, dtype=np.float32)]
    )
    path = directory / "source.h5ad"
    adata.write_h5ad(path)
    reader = H5adReader.from_inspect(
        inspect_h5ad(str(path)), embedding_roles={"X_umap": "umap"}
    )
    try:
        result = H5adToZarr(
            reader, str(directory / "store.zarr"), nthreads=1, analysis_assay="RNA"
        ).dump()
    finally:
        reader.close()
    store = cytearc.DataStore(
        str(directory / "store.zarr"), min_features_per_cell=0, nthreads=1
    )
    return store, result


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    """A store with a PCA, a KNN graph, a connectivity map, and Leiden labels."""
    store, result = _import_store(tmp_path_factory.mktemp("round_trip"))
    cells = store.snapshot_cell_selection("I")
    hvgs = store.features.hvgs(
        cells, top_n=20, min_cells=0, max_cells=np.inf, show_plot=False
    )
    normalized = store.features.normalize(cells, hvgs)
    pca = store.reduction.pca(normalized, dims=4)
    ann = store.graph.ann_index(pca, ann_efc=40, ann_ef=40, ann_m=8)
    neighbors = store.graph.neighbors(ann, k=5)
    graph = store.graph.connectivity(neighbors)
    leiden = store.clusters.leiden(graph, resolution=0.5)
    return {
        "store": store,
        "import": result,
        "cells": cells,
        "hvgs": hvgs,
        "pca": pca,
        "neighbors": neighbors,
        "graph": graph,
        "leiden": leiden,
    }


def _export(analysis, **kwargs):
    return analysis["store"].to_anndata(
        cell_selection=analysis["cells"],
        obsm={
            "X_pca": analysis["pca"],
            "X_umap": analysis["import"].embeddingArtifacts["X_umap"],
        },
        obs={
            "leiden": analysis["leiden"],
            "leiden_values": (analysis["leiden"], "values"),
        },
        graph=analysis["graph"],
        **kwargs,
    )


def _expected_graph(analysis) -> tuple[np.ndarray, np.ndarray]:
    """Return the fuzzy-union connectivities and the KNN distances, densely."""
    store = analysis["store"]
    directed = store.graph.load(analysis["graph"]).toarray()
    union = directed + directed.T - directed * directed.T
    knn = store.artifacts.load(analysis["neighbors"])
    indices = np.asarray(knn["indices"][:]).astype(np.int64)
    distances = np.zeros((_N_CELLS, _N_CELLS), dtype=np.float32)
    rows = np.repeat(np.arange(_N_CELLS), indices.shape[1])
    distances[rows, indices.ravel()] = np.asarray(knn["distances"][:]).ravel()
    return union, distances


@pytest.mark.slow
def test_live_export_holds_artifacts_in_scanpy_layout(analysis, tmp_path) -> None:
    store = analysis["store"]
    exported = _export(analysis)

    pca = store.artifacts.load(analysis["pca"])
    np.testing.assert_array_equal(exported.obsm["X_pca"], pca["data"][:])
    np.testing.assert_array_equal(
        exported.obsm["X_umap"],
        store.artifacts.load(analysis["import"].embeddingArtifacts["X_umap"])["values"][
            :
        ],
    )
    # varm["PCs"] holds the loadings at the highly variable genes, and zeros
    # at the genes the PCA did not use, as Scanpy stores them.
    hvg_mask = np.asarray(
        store.artifacts.load(analysis["hvgs"])["values"][:], dtype=bool
    )
    assert hvg_mask.sum() == 20
    assert exported.varm["PCs"].shape == (_N_GENES, 4)
    np.testing.assert_array_equal(exported.varm["PCs"][hvg_mask], pca["loadings"][:])
    assert not exported.varm["PCs"][~hvg_mask].any()
    # uns["pca"] holds the variance and variance ratio that the PCA stored.
    assert sorted(exported.uns["pca"]) == ["variance", "variance_ratio"]
    np.testing.assert_array_equal(exported.uns["pca"]["variance"], pca["variance"][:])
    np.testing.assert_array_equal(
        exported.uns["pca"]["variance_ratio"], pca["variance_ratio"][:]
    )
    assert exported.uns["pca"]["variance"].shape == (4,)
    # Labels are categoricals of text labels in natural order.
    labels = store.artifacts.load_values(analysis["leiden"]).values
    leiden = exported.obs["leiden"]
    assert isinstance(leiden.dtype, pd.CategoricalDtype)
    assert leiden.cat.categories.tolist() == [
        str(label) for label in sorted(set(labels.tolist()))
    ]
    assert leiden.tolist() == [str(label) for label in labels]
    assert exported.obs["leiden_values"].tolist() == leiden.tolist()
    # The graph in Scanpy's layout: symmetric connectivities, KNN distances
    # without the cell itself, and the neighbors entry that points to them.
    union, distances = _expected_graph(analysis)
    connectivities = exported.obsp["connectivities"]
    assert connectivities.format == "csr"
    np.testing.assert_array_equal(connectivities.toarray(), union)
    np.testing.assert_array_equal(connectivities.toarray(), connectivities.T.toarray())
    np.testing.assert_array_equal(exported.obsp["distances"].toarray(), distances)
    np.testing.assert_array_equal(
        exported.obsp["distances"].getnnz(axis=1), np.full(_N_CELLS, 5)
    )
    assert exported.uns["neighbors"] == {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": 6,
            "method": "umap",
            "metric": "euclidean",
            "use_rep": "X_pca",
        },
    }
    assert exported.uns["cytearc"] == {
        "artifacts": {
            "obsm": {
                "X_pca": analysis["pca"].to_dict(),
                "X_umap": analysis["import"].embeddingArtifacts["X_umap"].to_dict(),
            },
            "obs": {
                "leiden": analysis["leiden"].to_dict(),
                "leiden_values": {**analysis["leiden"].to_dict(), "value": "values"},
            },
            "obsp": {"connectivities": analysis["graph"].to_dict()},
        }
    }
    # AnnData writes the object, and reads it back unchanged.
    exported.write_h5ad(tmp_path / "written.h5ad")
    reread = anndata.read_h5ad(tmp_path / "written.h5ad")
    np.testing.assert_array_equal(reread.obsp["connectivities"].toarray(), union)
    assert reread.obs["leiden"].cat.categories.tolist() == (
        leiden.cat.categories.tolist()
    )


@pytest.mark.slow
def test_streamed_export_holds_what_the_in_memory_export_holds(
    analysis, tmp_path, monkeypatch
) -> None:
    import sys

    store = analysis["store"]
    expected = _export(analysis)
    path = tmp_path / "streamed.h5ad"
    with monkeypatch.context() as patched:
        # The streamed export needs no AnnData.
        patched.setitem(sys.modules, "anndata", None)
        cytearc.to_h5ad(
            store.RNA,
            str(path),
            cell_selection=analysis["cells"],
            obsm={
                "X_pca": analysis["pca"],
                "X_umap": analysis["import"].embeddingArtifacts["X_umap"],
            },
            obs={
                "leiden": analysis["leiden"],
                "leiden_values": (analysis["leiden"], "values"),
            },
            graph=analysis["graph"],
        )
    written = anndata.read_h5ad(path)

    assert written.obs_names.tolist() == expected.obs_names.tolist()
    for name in ("X_pca", "X_umap"):
        np.testing.assert_array_equal(written.obsm[name], expected.obsm[name])
    np.testing.assert_array_equal(written.varm["PCs"], expected.varm["PCs"])
    for name in ("connectivities", "distances"):
        np.testing.assert_array_equal(
            written.obsp[name].toarray(), expected.obsp[name].toarray()
        )
    pd.testing.assert_series_equal(
        written.obs["leiden"], expected.obs["leiden"], check_index=False
    )
    assert written.uns["neighbors"] == expected.uns["neighbors"]
    assert written.uns["cytearc"] == expected.uns["cytearc"]
    np.testing.assert_array_equal(written.X.toarray(), expected.X.toarray())


@pytest.mark.slow
def test_empty_artifact_mappings_export_nothing_beside_the_cells(
    analysis, tmp_path
) -> None:
    store = analysis["store"]
    cells = analysis["cells"]
    plain = store.to_anndata(cell_selection=cells)
    exported = store.to_anndata(cell_selection=cells, obsm={}, obs={})
    cytearc.to_h5ad(store.RNA, str(tmp_path / "plain.h5ad"), cell_selection=cells)
    cytearc.to_h5ad(
        store.RNA, str(tmp_path / "empty.h5ad"), cell_selection=cells, obsm={}, obs={}
    )
    plain_written = anndata.read_h5ad(tmp_path / "plain.h5ad")
    written = anndata.read_h5ad(tmp_path / "empty.h5ad")

    for adata, reference in ((exported, plain), (written, plain_written)):
        # Empty mappings add no arrays, columns, or record of artifacts.
        assert not adata.obsm and not adata.obsp and not adata.varm
        assert "artifacts" not in adata.uns.get("cytearc", {})
        assert adata.obs.columns.tolist() == reference.obs.columns.tolist()
        assert dict(adata.uns) == dict(reference.uns)
        assert adata.obs_names.tolist() == reference.obs_names.tolist()
        np.testing.assert_array_equal(adata.X.toarray(), reference.X.toarray())


@pytest.mark.slow
def test_exports_reject_misaligned_and_malformed_artifacts(analysis, tmp_path):
    store = analysis["store"]
    cells = analysis["cells"]
    graph = analysis["graph"]
    pca = analysis["pca"]
    leiden = analysis["leiden"]
    # Labels and coordinates of some cells, which cannot cover every cell.
    subset = store.select_cells(leiden, include=[1])
    n_subset = int(np.count_nonzero(store.artifacts.load(subset)["values"][:]))
    assert 0 < n_subset < _N_CELLS
    part = store.snapshot_cluster_labels(leiden, cell_selection=subset)
    partial = store.reduction.import_coordinates(
        np.ones((n_subset, 2)), cell_selection=subset, name="partial"
    )
    with pytest.raises(
        ValueError,
        match=rf"obs 'part': the cluster_labels artifact holds values for "
        rf"{n_subset} of the 48 exported cells\. Export the cells of its cell "
        "selection, with cell_selection=<that selection>",
    ):
        store.to_anndata(cell_selection=cells, obs={"part": part})
    with pytest.raises(
        ValueError, match=r"obsm 'partial': the imported_coordinates artifact holds"
    ):
        store.to_anndata(cell_selection=cells, obsm={"partial": partial})
    # A graph is exported only over exactly its cells, never over a subset.
    with pytest.raises(
        ValueError,
        match=rf"graph: the connectivity_map artifact covers 48 cells, and a graph "
        rf"is exported only over exactly its cells; the export holds {n_subset} "
        "other or more cells",
    ):
        store.to_anndata(cell_selection=subset, graph=graph)
    # A subset of an artifact's cells is exported.
    assert store.to_anndata(cell_selection=subset, obsm={"X_pca": pca}).obsm[
        "X_pca"
    ].shape == (n_subset, 4)
    over_cells = [
        ({"obsm": {"bad/name": pca}}, ValueError, "contains '/'"),
        ({"obsm": {"": pca}}, ValueError, "names must be non-empty strings"),
        ({"obsm": [pca]}, TypeError, "obsm must map names to artifact references"),
        ({"obs": {"x": (leiden,)}}, TypeError, r"obs\['x'\] must be an ArtifactRef"),
        ({"obs": {"x": pca}}, ValueError, "reduction coordinates belong in obsm"),
        ({"obsm": {"x": leiden}}, ValueError, "is not two-dimensional; export it"),
        (
            {"obs": {"x": analysis["import"].embeddingArtifacts["X_umap"]}},
            ValueError,
            "is not one-dimensional; export it in obsm",
        ),
        ({"obs": {"x": (leiden, "nope")}}, ValueError, r"obs 'x': 'cluster_labels'"),
        ({"obsm": {"x": (pca, "loadings")}}, ValueError, "export only their 'data'"),
        ({"obs": {"I": leiden}}, ValueError, "obs 'I' names a column that the export"),
        ({"graph": pca}, ValueError, "graph must be a connectivity_map"),
        ({"graph": "graph"}, TypeError, "graph must be an ArtifactRef"),
        ({"obsm": {"x": analysis["hvgs"]}}, ValueError, "'feature_selection'"),
    ]
    for kwargs, error, message in over_cells:
        with pytest.raises(error, match=message):
            store.to_anndata(cell_selection=cells, **kwargs)
    with pytest.raises(ValueError, match="cell_key and cell_selection are mutually"):
        store.to_anndata(cell_key="I", cell_selection=cells)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.to_anndata(cell_selection="I")
    with pytest.raises(ValueError, match="cell_selection must be a cell_selection"):
        store.to_anndata(cell_selection=pca)
    with pytest.raises(MemoryError, match="Exporting the graph's 240 edges needs"):
        cytearc.to_h5ad(
            _small_budget(store).RNA,
            str(tmp_path / "graph.h5ad"),
            cell_selection=cells,
            graph=graph,
        )
    assert not (tmp_path / "graph.h5ad").exists()


def _small_budget(store):
    """Open ``store`` again with a budget that cannot hold its graph."""
    return cytearc.DataStore(
        store.zw.store, min_features_per_cell=0, nthreads=1, mem_budget=1000
    )


@pytest.mark.slow
def test_imported_coordinates_feed_graph_construction(analysis) -> None:
    store = analysis["store"]
    cells = analysis["cells"]
    exported = store.to_anndata(cell_selection=cells)
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(_N_CELLS, 3)).astype(np.float32)
    # Rows in another order, and a row of a cell outside the selection.
    frame = pd.concat(
        [
            pd.DataFrame(latent, index=exported.obs_names).iloc[::-1],
            pd.DataFrame(np.zeros((1, 3), dtype=np.float32), index=["elsewhere"]),
        ]
    )
    assert frame.dtypes.tolist() == [np.dtype(np.float32)] * 3
    source = {"tool": "scvi-tools", "version": "1.2.0", "parameters": {"n": [3]}}

    ref = store.reduction.import_coordinates(
        frame, cell_selection=cells, name="X_scVI", source=source
    )

    assert ref.kind == "imported_coordinates" and ref.assay == "RNA"
    stored = store.artifacts.load(ref)["data"]
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored[:], latent)
    status = store.artifacts.inspect(ref)
    assert status.parameters["source"] == source
    assert status.parameters["dimreduc_key"] == "X_scVI"
    assert status.parameters["role"] == "coordinates"
    assert status.inputs["payload_fingerprints"] == {"data": fingerprint_array(latent)}
    assert status.inputs["source_digest"] == {"bytes_hex": fingerprint_array(latent)}
    # A DataFrame of exactly the selected cells imports the same values.
    assert (
        store.reduction.import_coordinates(
            pd.DataFrame(latent, index=exported.obs_names),
            cell_selection=cells,
            name="X_scVI",
            source=source,
        )
        == ref
    )
    # The same values in store order reuse the artifact.
    assert (
        store.reduction.import_coordinates(
            latent, cell_selection=cells, name="X_scVI", source=source
        )
        == ref
    )
    # Integer coordinates are stored as float64; another source is another
    # artifact.
    whole = store.reduction.import_coordinates(
        np.arange(_N_CELLS * 2).reshape(_N_CELLS, 2), cell_selection=cells, name="X"
    )
    assert store.artifacts.load(whole)["data"].dtype == np.float64
    assert "source" not in store.artifacts.inspect(whole).parameters
    ann = store.graph.ann_index(ref, ann_efc=40, ann_ef=40, ann_m=8)
    graph = store.graph.connectivity(store.graph.neighbors(ann, k=5))
    clusters = store.clusters.leiden(graph, resolution=0.5)
    assert store.artifacts.load_values(clusters).values.shape == (_N_CELLS,)
    # The graph is exported over the selection, with the imported
    # coordinates as its representation.
    exported = store.to_anndata(cell_selection=cells, obsm={"X_scVI": ref}, graph=graph)
    assert exported.uns["neighbors"]["params"]["use_rep"] == "X_scVI"


@pytest.mark.slow
def test_import_coordinates_rejects_misaligned_values(analysis, tmp_path) -> None:
    store = analysis["store"]
    cells = analysis["cells"]
    ids = store.cells.fetch_all("ids")
    values = np.ones((_N_CELLS, 2))
    cases = [
        (values[:-1], ValueError, "have 47 rows; the cell selection holds 48 cells"),
        (np.ones(_N_CELLS), ValueError, "must be two-dimensional; found 1"),
        (np.ones((_N_CELLS, 0)), ValueError, "need at least one dimension"),
        (np.full((_N_CELLS, 2), np.nan), ValueError, "only finite values"),
        (np.full((_N_CELLS, 2), "a"), TypeError, "uses unsupported dtype"),
        (
            pd.DataFrame(values[1:], index=ids[1:]),
            ValueError,
            "have no row for 1 of the 48 selected cells, such as 'cell0'",
        ),
        (
            pd.DataFrame(np.ones((2, 2)), index=["cell0", "cell0"]),
            ValueError,
            "repeat cell IDs, such as 'cell0'",
        ),
    ]
    for coordinates, error, message in cases:
        with pytest.raises(error, match=message):
            store.reduction.import_coordinates(
                coordinates, cell_selection=cells, name="bad"
            )
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        store.reduction.import_coordinates(values, cell_selection=cells, name="")
    with pytest.raises(TypeError, match=r"source\['tool'\] holds a object"):
        store.reduction.import_coordinates(
            values, cell_selection=cells, name="x", source={"tool": object()}
        )
    with pytest.raises(ValueError, match=r"source\['n'\]\[0\] must be a finite"):
        store.reduction.import_coordinates(
            values, cell_selection=cells, name="x", source={"n": [float("nan")]}
        )
    with pytest.raises(TypeError, match=r"source\['p'\] must have text keys"):
        store.reduction.import_coordinates(
            values, cell_selection=cells, name="x", source={"p": {1: 2}}
        )
    with pytest.raises(TypeError, match="source must be a mapping"):
        store.reduction.import_coordinates(
            values, cell_selection=cells, name="x", source=["tool"]
        )
    read_only = cytearc.DataStore(
        store.zw.store, min_features_per_cell=0, nthreads=1, zarr_mode="r"
    )
    with pytest.raises(PermissionError, match="import_coordinates"):
        read_only.reduction.import_coordinates(
            values * 7, cell_selection=cells, name="new"
        )


@pytest.mark.slow
def test_count_dtype_option_and_default(analysis, tmp_path) -> None:
    store = analysis["store"]
    assert store.RNA.rawData.dtype == np.uint8
    counts = _source_counts()

    default = store.to_anndata(layers={"raw": "RNA"})
    narrow = store.to_anndata(dtype="uint8", layers={"raw": "RNA"})

    # Narrow unsigned counts export as float32, whose arithmetic cannot wrap.
    assert default.X.dtype == np.float32
    assert default.layers["raw"].dtype == np.float32
    assert narrow.X.dtype == np.uint8 and narrow.layers["raw"].dtype == np.uint8
    np.testing.assert_array_equal(default.X.toarray(), counts)
    np.testing.assert_array_equal(narrow.X.toarray(), counts)
    assert (default.X * 300).max() == counts.max() * 300
    # Normalized values keep their dtype; dtype applies to raw counts only.
    assert store.to_anndata(matrix="normed", dtype="uint8").X.dtype == np.float64
    # A dtype that cannot hold a count raises instead of wrapping.
    assert counts.max() > 1
    with pytest.raises(OverflowError, match="Sparse values exceed"):
        store.to_anndata(dtype=bool)
    with pytest.raises(OverflowError, match="Sparse values exceed"):
        cytearc.to_h5ad(store.RNA, str(tmp_path / "bool.h5ad"), dtype=bool)
    assert not (tmp_path / "bool.h5ad").exists()
    with pytest.raises(TypeError, match="dtype must be a boolean, integer"):
        store.to_anndata(dtype="U4")
    path = tmp_path / "counts.h5ad"
    cytearc.to_h5ad(store.RNA, str(path))
    assert anndata.read_h5ad(path).X.dtype == np.float32
    cytearc.to_h5ad(store.RNA, str(path), dtype=np.uint16)
    assert anndata.read_h5ad(path).X.dtype == np.uint16
    with pytest.raises(TypeError, match="dtype must be a boolean, integer"):
        cytearc.to_h5ad(store.RNA, str(path), dtype="U4")


def test_export_count_dtype_policy() -> None:
    from cytearc.writers.export import cast_counts, export_count_dtype

    assert export_count_dtype(np.uint8) == np.float32
    assert export_count_dtype(np.uint16) == np.float32
    assert export_count_dtype(np.uint32) == np.uint32
    assert export_count_dtype(np.float64) == np.float64
    assert export_count_dtype(np.uint16, "int32") == np.int32
    with pytest.raises(TypeError, match="not complex128"):
        export_count_dtype(np.uint8, np.complex128)
    matrix = csr_matrix(np.asarray([[0, 70_000]], dtype=np.uint32))
    assert cast_counts(matrix, np.dtype(np.uint32)) is matrix
    assert cast_counts(matrix, np.dtype(np.float32)).toarray().tolist() == [
        [0.0, 70_000.0]
    ]
    with pytest.raises(OverflowError, match="exceed the range of the export dtype"):
        cast_counts(matrix, np.dtype(np.float16))
    with pytest.raises(OverflowError, match="exceed the destination dtype"):
        cast_counts(matrix, np.dtype(np.uint16))


@pytest.fixture(scope="module")
def multimodal(tmp_path_factory):
    """A store with RNA and ADT graphs over one selection, and their SNN graph."""
    from tests.storage_helpers import write_count_store

    rng = np.random.default_rng(23)
    adt = rng.poisson(5.0, size=(_N_CELLS, 6))
    location = str(tmp_path_factory.mktemp("multimodal") / "store.zarr")
    write_count_store(location, {"RNA": _source_counts(), "ADT": adt}, np.uint16)
    store = cytearc.DataStore(
        location, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = store.snapshot_cell_selection("I")
    # The export reads only the graphs, so each assay's graph uses all of its
    # features.
    features = {
        assay: store.features.universe(from_assay=assay) for assay in ("RNA", "ADT")
    }
    graphs = {}
    for assay, selected in features.items():
        normalized = store.features.normalize(cells, selected)
        pca = store.reduction.pca(normalized, dims=3)
        ann = store.graph.ann_index(pca, ann_efc=40, ann_ef=40, ann_m=8)
        graphs[assay] = store.graph.connectivity(store.graph.neighbors(ann, k=5))
    integrated = store.integration.modalities(
        [graphs["RNA"], graphs["ADT"]], method="snn"
    )
    return store, cells, integrated


@pytest.mark.slow
def test_integrated_graph_exports_connectivities_without_distances(multimodal):
    store, cells, integrated = multimodal

    exported = store.to_anndata(cell_selection=cells, graph=integrated)

    directed = store.graph.load(integrated).toarray()
    union = directed + directed.T - directed * directed.T
    assert set(exported.obsp) == {"connectivities"}
    np.testing.assert_array_equal(exported.obsp["connectivities"].toarray(), union)
    k = int(store.artifacts.load(integrated).attrs["n_neighbors"])
    assert exported.uns["neighbors"] == {
        "connectivities_key": "connectivities",
        "params": {"n_neighbors": k + 1, "method": "snn"},
    }


@pytest.mark.slow
def test_pca_entries_follow_the_coordinates_that_hold_them(analysis) -> None:
    from cytearc.embeddings.imported import write_imported_coordinates

    store = analysis["store"]
    cells = analysis["cells"]
    coordinates = np.arange(_N_CELLS * 2, dtype=np.float32).reshape(_N_CELLS, 2)
    plain = store.reduction.import_coordinates(
        coordinates, cell_selection=cells, name="plain"
    )
    # Imported coordinates with loadings over two genes and a deviation per
    # dimension, as a Seurat import stores them.
    loadings = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    feature_ids = np.asarray(["gene3", "gene0"])
    stdev = np.asarray([2.0, 0.5])
    with_loadings = write_imported_coordinates(
        store.zw,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates + 1,
        source_digest=b"\x01" * 32,
        payload_fingerprints={
            "data": fingerprint_array(coordinates + 1),
            "loadings": fingerprint_array(loadings),
            "feature_ids": fingerprint_strings(feature_ids),
            "stdev": fingerprint_array(stdev),
        },
        source_cell_ids=store.cells.fetch_all("ids"),
        cell_selection=cells,
        loadings=loadings,
        feature_ids=feature_ids,
        stdev=stdev,
    )

    exported = store.to_anndata(cell_selection=cells, obsm={"X_pca": with_loadings})

    pcs = np.zeros((_N_GENES, 2))
    pcs[3], pcs[0] = loadings
    np.testing.assert_array_equal(exported.varm["PCs"], pcs)
    np.testing.assert_array_equal(exported.uns["pca"]["variance"], [4.0, 0.25])
    without = store.to_anndata(cell_selection=cells, obsm={"X_pca": plain})
    assert "PCs" not in without.varm and "pca" not in without.uns
    # A PCA from before 1.3.0 stored no variance, so it exports none.
    earlier = store.reduction.pca(
        store.features.normalize(cells, analysis["hvgs"]), dims=3
    )
    group = artifact_group(store.zw, earlier)
    del group["variance"]
    del group["variance_ratio"]
    old_pca = store.to_anndata(cell_selection=cells, obsm={"X_pca": earlier})
    assert old_pca.varm["PCs"].shape == (_N_GENES, 3) and "pca" not in old_pca.uns
    # A layout as X_pca has no loadings either.
    layout = store.to_anndata(
        cell_selection=cells,
        obsm={"X_pca": analysis["import"].embeddingArtifacts["X_umap"]},
    )
    assert "PCs" not in layout.varm and "pca" not in layout.uns


@pytest.mark.slow
def test_exports_refuse_unavailable_and_damaged_artifacts(analysis) -> None:
    from cytearc.storage.artifacts import artifact_group

    store = analysis["store"]
    cells = analysis["cells"]
    missing = ArtifactRef(
        scope="assay", kind="cluster_labels", artifact_id="0" * 64, assay="RNA"
    )
    with pytest.raises(
        ValueError, match="The cluster_labels artifact is unavailable or incomplete"
    ):
        store.to_anndata(cell_selection=cells, obs={"x": missing})
    labels = store.snapshot_cluster_labels(analysis["leiden"], cell_selection=cells)
    # A damaged artifact whose label array was deleted.
    del artifact_group(store.zw, labels)["values"]
    with pytest.raises(
        ValueError, match="obs 'x': the cluster_labels artifact has no 'values' array"
    ):
        store.to_anndata(cell_selection=cells, obs={"x": labels})


def test_plans_check_obsp_and_varm_shapes(tmp_path) -> None:
    import dataclasses

    from cytearc.writers.export import (
        H5adColumn,
        H5adExportPlan,
        H5adMatrix,
        write_h5ad_plan,
    )

    names = np.asarray(["a", "b"])
    plan = H5adExportPlan(
        x=H5adMatrix(
            shape=(2, 2),
            dtype=np.dtype(np.int32),
            blocks=lambda: iter([np.eye(2, dtype=np.int32)]),
        ),
        obs_index=H5adColumn("_index", lambda: (names, None)),
        obs=(),
        var_index=H5adColumn("_index", lambda: (names, None)),
        var=(),
    )
    path = tmp_path / "shapes.h5ad"
    for changes, message in (
        (
            {"obsp": {"g": lambda: csr_matrix((2, 3))}},
            r"obsp 'g' has shape \(2, 3\); the export declares \(2, 2\)",
        ),
        (
            {"varm": {"PCs": lambda: np.zeros((3, 2))}},
            r"varm 'PCs' has shape \(3, 2\); the export declares 2 features",
        ),
        (
            {"varm": {"PCs": lambda: np.zeros(2)}},
            r"varm 'PCs' has shape \(2,\); the export declares 2 features",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            write_h5ad_plan(dataclasses.replace(plan, **changes), path)
        assert not path.exists()
    write_h5ad_plan(
        dataclasses.replace(
            plan,
            obsp={"g": lambda: csr_matrix(np.eye(2))},
            varm={"PCs": lambda: np.ones((2, 1))},
            uns={"flag": True, "name": "x", "nested": {"values": [1, 2]}},
        ),
        path,
    )
    exported = anndata.read_h5ad(path)
    np.testing.assert_array_equal(exported.obsp["g"].toarray(), np.eye(2))
    np.testing.assert_array_equal(exported.varm["PCs"], np.ones((2, 1)))
    assert np.asarray(exported.uns["flag"]).dtype == bool
    assert bool(exported.uns["flag"]) is True
    assert exported.uns["name"] == "x"
    assert exported.uns["nested"]["values"].tolist() == [1, 2]
