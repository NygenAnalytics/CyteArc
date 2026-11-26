"""Stores that CyteArc 1.4.0 wrote open, read, replay, export and merge as the compatibility policy states.

``tests/data/cytearc_1_4_0`` holds stores that the released CyteArc 1.4.0
wrote, the script that wrote them, and ``manifest.json``, the refs and values
that 1.4.0 read back from them. Each test works on a copy of a store, or opens
the store in the folder read-only, and checks this release's reads against the
manifest, the files that the stores imported, or the arrays that Zarr reads
from the stores directly. The policy is in "Stores written by other releases"
of ``docs/source/developers/zarr_internals.md``.
"""

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
import zarr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from scipy.io import mmread  # noqa: E402

import cytearc  # noqa: E402
from cytearc import DataStore  # noqa: E402
from cytearc.merge import DataStoreMerge  # noqa: E402
from cytearc.metadata.membership import (  # noqa: E402
    features_measured_by_every_source,
    measured_features_by_source,
)
from cytearc.readers import H5adReader  # noqa: E402
from cytearc.storage.artifacts import ArtifactRef  # noqa: E402
from cytearc.storage.identity import nonzero_value_count  # noqa: E402
from cytearc.storage.operation_revisions import OPERATION_REVISIONS  # noqa: E402
from cytearc.writers import H5adToZarr  # noqa: E402
from tests.storage_helpers import recomputation, recomputations  # noqa: E402

anndata = pytest.importorskip("anndata")

_FIXTURE = Path(__file__).parent / "data" / "cytearc_1_4_0"
_MANIFEST: dict[str, Any] = json.loads((_FIXTURE / "manifest.json").read_text())
_STORES = ("rna.zarr", "atac.zarr", "merged.zarr")
_RUNS = {"rna.zarr": "rna", "atac.zarr": "atac"}
# The categories and colors of cell_type in the H5AD file that rna.zarr imported.
_CELL_TYPES = ["T", "B", "NK", "Mono", "Other"]
_CELL_TYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"]


def _ref(record: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        scope=record["scope"],
        kind=record["kind"],
        artifact_id=record["artifactId"],
        assay=record["assay"],
    )


def _outputs(store: str, label: str) -> dict[str, Any]:
    """The outputs that 1.4.0 recorded for a run, as refs and tuples of refs."""
    recorded = _MANIFEST[store.removesuffix(".zarr")]["runs"][label]["outputs"]
    return {
        key: tuple(map(_ref, value)) if isinstance(value, list) else _ref(value)
        for key, value in recorded.items()
    }


_RNA = _outputs("rna.zarr", "rna")
_ARTIFACTS = {
    name: _ref(record) for name, record in _MANIFEST["rna"]["artifacts"].items()
}


def _contents(root: Path) -> dict[str, str]:
    """Every file below ``root`` with the SHA-256 of its bytes."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy(folder: Path, name: str) -> Path:
    """Copy a store or an input file of the fixture into ``folder``."""
    target = folder / name
    if (_FIXTURE / name).is_dir():
        shutil.copytree(_FIXTURE / name, target)
    else:
        shutil.copy(_FIXTURE / name, target)
    return target


def _writable(folder: Path, name: str = "rna.zarr") -> DataStore:
    return DataStore(str(_copy(folder, name)), nthreads=1)


def _read_only(name: str = "rna.zarr") -> DataStore:
    return DataStore(str(_FIXTURE / name), zarr_mode="r", nthreads=1)


def _change(operation: str, revisions: tuple[int, ...]) -> str:
    """The changes of some revisions of ``operation``, as the recomputation line joins them."""
    entries = {entry.revision: entry for entry in OPERATION_REVISIONS[operation]}
    return "; ".join(entries[revision].change for revision in revisions)


def _legend(result: Any) -> list[tuple[str, str]]:
    """The labels and colors of a plot's first legend, in their order."""
    legend = result.figure.legends[0]
    colors = [
        handle.get_markerfacecolor()
        if hasattr(handle, "get_markerfacecolor")
        else handle.get_facecolor()
        for handle in legend.legend_handles
    ]
    return [
        (text.get_text(), to_hex(color))
        for text, color in zip(legend.get_texts(), colors, strict=True)
    ]


def _feature_name(store: DataStore, feature_id: str) -> str:
    ids = store.RNA.feats.fetch_all("ids").astype(str).tolist()
    return str(store.RNA.feats.fetch_all("names")[ids.index(feature_id)])


def test_the_stores_were_written_by_cytearc_1_4_0() -> None:
    # Every artifact and run records the release that wrote it, and no store
    # holds what later releases add to every new store.
    assert _MANIFEST["packages"]["cytearc"] == "1.4.0"
    for store in _STORES:
        artifacts: list[str] = []
        runs: list[str] = []
        for document in (_FIXTURE / store).rglob("zarr.json"):
            attributes = json.loads(document.read_text()).get("attributes", {})
            if "artifact_id" in attributes:
                artifacts.append(attributes["cytearc_version"])
            if "runId" in attributes and "recipe" in attributes:
                runs.append(attributes["cytearcVersion"])
        assert set(artifacts) == ({"1.4.0"} if store in _RUNS else set())
        recorded = _MANIFEST[store.removesuffix(".zarr")].get("runs", {})
        assert runs == ["1.4.0"] * len(recorded)
        root = zarr.open_group(str(_FIXTURE / store), mode="r")
        assay = "ATAC" if store == "atac.zarr" else "RNA"
        assert "rowNonzero" not in root[f"{assay}/countSummaries"]
        assert "sourceRow" not in root["cellData"]


@pytest.mark.parametrize("store", _STORES)
def test_opening_a_store_writes_nothing(store, tmp_path) -> None:
    copy = _copy(tmp_path, store)
    for path, mode in ((copy, "r+"), (_FIXTURE / store, "r")):
        before = _contents(path)
        opened = DataStore(str(path), zarr_mode=mode, nthreads=1)
        opened.summary()
        for run in opened.pipeline.list_runs(limit=100):
            run.report()
        assert _contents(path) == before, mode


@pytest.mark.parametrize("store", sorted(_RUNS))
def test_runs_list_open_and_report_as_1_4_0_recorded_them(store) -> None:
    opened = _read_only(store)
    recorded = _MANIFEST[store.removesuffix(".zarr")]["runs"]
    listed = opened.pipeline.list_runs(limit=100)
    assert {run.label: run.run_id for run in listed} == {
        label: run["runId"] for label, run in recorded.items()
    }
    for label in recorded:
        run = opened.pipeline.open(label=label)
        assert run.status == "completed"
        assert dict(run.items()) == _outputs(store, label)
        report = run.report()
        assert report["run"]["runId"] == recorded[label]["runId"]
        assert report["run"]["cytearcVersion"] == "1.4.0"
        assert [stage["stage"] for stage in report["stages"]] == report["run"][
            "stageOrder"
        ]


def test_a_run_reads_the_labels_that_it_froze() -> None:
    opened = _read_only()
    run = opened.pipeline.open(label="rna")
    by_id = dict(
        zip(
            opened.cells.fetch_all("ids").tolist(),
            _MANIFEST["rna"]["read"]["cellType"],
            strict=True,
        )
    )
    frozen = run.cells.fetch("cell_type").tolist()
    assert frozen == [by_id[cell] for cell in run.cells.fetch("ids").tolist()]
    assert len(frozen) < opened.cells.N


@pytest.mark.parametrize(
    ("store", "label", "method"),
    [("rna.zarr", "rna", "lib_size"), ("atac.zarr", "atac", "tf_idf")],
)
def test_a_run_takes_the_normalizer_that_its_saved_values_record(
    store, label, method
) -> None:
    # 1.4.0 recorded no normalization method, so the settings that replay the
    # run take the normalizer of the normalized values that it saved.
    run = _read_only(store).pipeline.open(label=label)
    assert "method" not in run.report()["run"]["config"]["stages"]["normalization"]
    assert run.settings()["normalization"] == {"method": method}


def test_an_atac_run_replays_without_doublet_scoring_and_reuses_every_result(
    tmp_path,
) -> None:
    opened = _writable(tmp_path, "atac.zarr")
    run = opened.pipeline.open(label="atac")
    assert "doublets" not in run.settings()
    planned = opened.pipeline.plan(run)
    assert {stage.status for stage in planned.stages} == {"reused", "skipped"}

    def saved() -> set[ArtifactRef]:
        return {
            *opened.artifacts.list(scope="datastore"),
            *opened.artifacts.list(from_assay="ATAC"),
        }

    before = saved()
    again = opened.pipeline.rerun(run, label="replayed")
    assert saved() == before
    assert dict(again.items()) == dict(run.items())
    assert again.settings()["doublets"] is False


def test_a_replay_selects_the_same_genes_under_the_new_revision_and_says_so(
    tmp_path,
) -> None:
    opened = _writable(tmp_path)
    run = opened.pipeline.open(label="rna")
    stages = {stage.stage: stage for stage in opened.pipeline.plan(run).stages}
    assert (stages["cell_cycle"].status, stages["filtering"].status) == (
        "reused",
        "reused",
    )
    assert (
        stages["highly_variable_features"].status,
        stages["highly_variable_features"].reason,
    ) == ("computed", "superseded")
    assert {stages[name].reason for name in ("pca", "doublets", "markers")} == {
        "upstream"
    }
    saved = _RNA["highly_variable_features"]
    with recomputations() as lines:
        replayed = opened.pipeline.rerun(run, label="replayed")
    assert lines == [
        recomputation("select_hvgs", saved, 1, 3, _change("select_hvgs", (3,)))
    ]
    selected = np.asarray(
        opened.artifacts.load(replayed["highly_variable_features"])["values"][:],
        dtype=bool,
    )
    names = opened.RNA.feats.fetch_all("names").astype(str)
    assert sorted(names[selected]) == sorted(_MANIFEST["rna"]["read"]["hvgNames"])


def test_a_selection_of_1_4_0_reads_its_ratio_as_unmeasured_and_serves_explicit_calls() -> (
    None
):
    opened = _read_only()
    saved = _RNA["highly_variable_features"]
    status = opened.artifacts.inspect(saved)
    assert (status.revision, status.current_revision, status.is_current) == (
        1,
        3,
        False,
    )
    assert status.diagnostics["selectedDetectionRatio"] is None
    assert status.diagnostics["selectedFeatures"] == 150
    # Normalizing over it reuses the values that the run saved, read-only.
    assert (
        opened.features.normalize(_RNA["analysis_cell_selection"], saved)
        == _RNA["normalized"]
    )


# Each result that 1.4.0 saved at a revision that this release supersedes: its
# operation, its ref, the revision it records, the revisions of this release
# that apply to it, and a call with the arguments that saved it.
_SUPERSEDED: dict[
    str,
    tuple[str, ArtifactRef, int, tuple[int, ...], Callable[[DataStore, Path], Any]],
] = {
    "hvgs": (
        "select_hvgs",
        _RNA["highly_variable_features"],
        1,
        (3,),
        lambda store, _folder: store.features.hvgs(
            _RNA["analysis_cell_selection"], top_n=150, min_cells=5
        ),
    ),
    # Revision 4 applies to scores whose threshold is automatic, as these are.
    "doublets": (
        "run_doublet_detection",
        _ARTIFACTS["doublets"],
        2,
        (3, 4),
        lambda store, _folder: store.qc.doublets(
            _RNA["clusters"], _RNA["connectivity_map"]
        ),
    ),
    "receptors": (
        "import_immune_receptors",
        _ARTIFACTS["receptors"],
        1,
        (2,),
        lambda store, folder: store.repertoire.import_receptors(
            folder / "receptors.tsv", cell_selection=_RNA["analysis_cell_selection"]
        ),
    ),
    "clones": (
        "import_clones",
        _ARTIFACTS["clones"],
        1,
        (2,),
        lambda store, folder: store.clones.import_clones(
            folder / "clones.csv",
            cell_selection=_RNA["analysis_cell_selection"],
            columns={"cell": "cell", "clone": "clone"},
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(_SUPERSEDED))
def test_a_superseded_result_is_computed_anew_with_a_log_line_and_refused_read_only(
    name, tmp_path
) -> None:
    operation, saved, revision, applying, call = _SUPERSEDED[name]
    for item in ("receptors.tsv", "clones.csv"):
        _copy(tmp_path, item)
    read_only = _read_only()
    status = read_only.artifacts.inspect(saved)
    current = applying[-1]
    assert (status.revision, status.current_revision) == (revision, current)
    # A read-only store cannot save the result again, and the refusal names
    # the operation that computes it.
    with pytest.raises(PermissionError, match=f"^{operation} requires"):
        call(read_only, tmp_path)
    opened = _writable(tmp_path)
    with recomputations() as lines:
        again = call(opened, tmp_path)
    assert again != saved
    assert lines == [
        recomputation(operation, saved, revision, current, _change(operation, applying))
    ]
    assert opened.artifacts.inspect(again).revision == current
    # The saved result is still there, and still reads.
    assert opened.artifacts.inspect(saved).complete


def test_receptor_and_clone_imports_of_1_4_0_read_their_files_counts_as_unmeasured() -> (
    None
):
    opened = _read_only()
    receptors = opened.artifacts.inspect(_ARTIFACTS["receptors"]).diagnostics
    clones = opened.artifacts.inspect(_ARTIFACTS["clones"]).diagnostics
    assert receptors["contigsPerFile"] is None
    assert clones["rowsPerFile"] is None
    with (_FIXTURE / "receptors.tsv").open() as handle:
        contigs = list(csv.DictReader(handle, delimiter="\t"))
    with (_FIXTURE / "clones.csv").open() as handle:
        listed = list(csv.DictReader(handle))
    assert receptors["fileContigs"] == len(contigs)
    assert clones["fileRows"] == len(listed)


def test_receptors_and_clones_of_1_4_0_export_their_cells(tmp_path) -> None:
    opened = _read_only()
    selected = set(
        opened.cells.fetch_all("ids")[
            opened.artifacts.load_values(_RNA["clusters"]).cell_idx
        ].tolist()
    )
    with (_FIXTURE / "receptors.tsv").open() as handle:
        contigs = [row for row in csv.DictReader(handle, delimiter="\t")]
    with (_FIXTURE / "clones.csv").open() as handle:
        clone_of = {row["cell"]: row["clone"] for row in csv.DictReader(handle)}
    exported = opened.repertoire.export_airr(
        _ARTIFACTS["receptors"], tmp_path / "receptors.tsv"
    )
    with exported.path.open() as handle:
        written = [row for row in csv.DictReader(handle, delimiter="\t")]
    assert sorted(row["sequence_id"] for row in written) == sorted(
        row["sequence_id"] for row in contigs if row["cell_id"] in selected
    )
    exported = opened.clones.export_clones(
        _ARTIFACTS["clones"], tmp_path / "clones.csv"
    )
    with exported.path.open() as handle:
        rows = list(csv.reader(handle))
    assert {cell: clone for cell, clone in rows[1:]} == {
        cell: clone for cell, clone in clone_of.items() if cell in selected
    }
    # The clonotypes that 1.4.0 saved are reused.
    assert (
        opened.repertoire.clonotypes(_ARTIFACTS["receptors"])
        == (_ARTIFACTS["clonotypes"])
    )


def test_an_aggregation_with_hnsw_settings_loads_plots_and_is_computed_anew(
    tmp_path,
) -> None:
    opened = _read_only()
    saved = _ARTIFACTS["aggregation"]
    status = opened.artifacts.inspect(saved)
    assert {"ef", "M", "ef_construction"} <= set(status.parameters["ann_params"])
    loaded = opened.trajectory.load_aggregation(saved)
    group = opened.artifacts.load(saved)
    assert np.array_equal(loaded.feature_clusters, group["feature_clusters"][:])
    assert len(set(loaded.feature_clusters.tolist()) - {-1}) == 3
    try:
        opened.plots.pseudotime_heatmap(aggregation=saved, show=False)
    finally:
        plt.close("all")

    def call(store: DataStore) -> ArtifactRef:
        return store.trajectory.aggregation(
            _ARTIFACTS["pseudotime"],
            features=_RNA["highly_variable_features"],
            window_size=30,
            chunk_size=10,
            n_neighbors=5,
            n_clusters=3,
        )

    with pytest.raises(PermissionError, match="^run_pseudotime_aggregation requires"):
        call(opened)
    writable = _writable(tmp_path)
    again = call(writable)
    assert again != saved
    assert writable.artifacts.inspect(again).revision == 3


def test_label_snapshots_without_categories_are_reused_and_plot_in_natural_order(
    tmp_path,
) -> None:
    saved = _ARTIFACTS["cellTypeLabels"]
    # 1.4.0 froze the labels alone, without the column's categories.
    values = zarr.open_group(str(_FIXTURE / "rna.zarr"), mode="r")[
        f"artifacts/cluster_labels/{saved.artifact_id}/values"
    ]
    assert dict(values.attrs) == {}
    for opened in (_read_only(), _writable(tmp_path)):
        assert (
            opened.snapshot_cluster_labels(
                "cell_type", cell_selection=_RNA["analysis_cell_selection"]
            )
            == saved
        )
    opened = _read_only()
    try:
        snapshot = _legend(
            opened.plots.embedding(
                layout=_RNA["umap"], color_by=saved, legend_loc="right", show=False
            )
        )
        column = _legend(
            opened.plots.embedding(
                layout=_RNA["umap"],
                color_by="cell_type",
                legend_loc="right",
                show=False,
            )
        )
    finally:
        plt.close("all")
    assert [label for label, _color in snapshot] == sorted(_CELL_TYPES)
    assert column == list(zip(_CELL_TYPES, _CELL_TYPE_COLORS, strict=True))


def test_run_exports_write_frozen_columns_in_their_display_order_and_colors(
    tmp_path,
) -> None:
    opened = _read_only()
    run = opened.pipeline.open(label="rna")
    exported = opened.to_anndata(run=run)
    assert list(exported.obs["cell_type"].cat.categories) == _CELL_TYPES
    assert list(exported.uns["cell_type_colors"]) == _CELL_TYPE_COLORS
    cytearc.to_h5ad(opened.RNA, str(tmp_path / "run.h5ad"), run=run)
    written = anndata.read_h5ad(tmp_path / "run.h5ad")
    assert list(written.obs["cell_type"].cat.categories) == _CELL_TYPES
    assert list(written.uns["cell_type_colors"]) == _CELL_TYPE_COLORS
    assert (
        written.obs["cell_type"].astype(str).tolist()
        == run.cells.fetch("cell_type").tolist()
    )


def test_categories_that_read_alike_keep_one_level_per_label(tmp_path) -> None:
    stored = zarr.open_group(str(_FIXTURE / "rna.zarr"), mode="r")["cellData/batch"]
    assert stored.attrs["levels"] == ["b1", "b2", "b1"]
    opened = _read_only()
    labels = _MANIFEST["rna"]["read"]["batch"]
    assert opened.cells.fetch_all("batch").tolist() == labels
    exported = opened.to_anndata().obs["batch"]
    assert list(exported.cat.categories) == ["b1", "b2"]
    assert exported.astype(str).tolist() == labels
    try:
        drawn = _legend(
            opened.plots.embedding(
                layout=_RNA["umap"], color_by="batch", legend_loc="right", show=False
            )
        )
    finally:
        plt.close("all")
    assert [label for label, _color in drawn] == ["b1", "b2"]
    # A merge takes each label of the 1.4.0 store once, then the new store's.
    joined = _merged_with_a_new_store(tmp_path)
    assert joined.cells._get_array("orig_batch").attrs["levels"] == ["b1", "b2", "b3"]


def test_settings_that_this_release_refuses_stop_a_replay_until_overridden(
    tmp_path,
) -> None:
    opened = _read_only()
    run = opened.pipeline.open(label="rna_text_sampling")
    stages = run.report()["run"]["config"]["stages"]
    assert stages["embedding_initialization"] == {"kmeans_sampling": "0.2"}
    refusal = "^embedding_initialization: kmeans_sampling must be a real number"
    with pytest.raises(TypeError, match=refusal):
        opened.pipeline.plan(run)
    copy = _copy(tmp_path, "rna.zarr")
    writable = DataStore(str(copy), nthreads=1)
    before = _contents(copy)
    with pytest.raises(TypeError, match=refusal):
        writable.pipeline.rerun(
            writable.pipeline.open(label="rna_text_sampling"), label="replayed"
        )
    assert _contents(copy) == before
    # With the setting overridden, the run replays, and the min_dist above
    # the spread that 1.4.0 accepted stays accepted.
    fixed = {"embedding_initialization": {"kmeans_sampling": 0.2}}
    assert stages["umap"]["min_dist"] > stages["umap"]["spread"]
    planned = opened.pipeline.plan(run, **fixed)
    assert planned.config["stages"]["umap"]["min_dist"] == 1.5
    replayed = writable.pipeline.rerun(
        writable.pipeline.open(label="rna_text_sampling"), label="replayed", **fixed
    )
    assert replayed.status == "completed"


def test_the_columns_of_a_1_4_0_merge_read_back_as_stored() -> None:
    opened = _read_only("merged.zarr")
    read = _MANIFEST["merged"]["read"]
    cells = opened.cells
    assert cells.fetch_all("ids").tolist() == read["ids"]
    # NaN and NaT that 1.4.0 merged into text stay the text "nan" and "NaT".
    assert cells.fetch_all("orig_note").tolist() == read["note"]
    assert cells.fetch_all("orig_when").tolist() == read["when"]
    assert "nan" in read["note"] and "NaT" in read["when"]
    for column in ("orig_note", "orig_when"):
        assert "missing_mask" not in cells._get_array(column).attrs
    # Dates of two units stay floats.
    collected = cells.fetch_all("orig_collected")
    assert collected.dtype == np.float64
    assert collected.tolist() == read["collected"]


def test_a_1_4_0_merge_keeps_the_size_factor_that_it_recorded() -> None:
    # The left study recorded 2500.5, but its session set sf=None, so the
    # merged assay took 1000 when 1.4.0 first opened it.
    opened = _read_only("merged.zarr")
    stored = zarr.open_group(str(_FIXTURE / "merged.zarr"), mode="r")["RNA"]
    assert stored.attrs["size_factor"] == 1000
    assert opened.RNA.sf == 1000 == _MANIFEST["merged"]["read"]["sizeFactor"]


def test_the_zero_fills_of_a_1_4_0_union_merge_read_as_measured_counts() -> None:
    opened = _read_only("merged.zarr")
    assert measured_features_by_source(opened.RNA.feats) is None
    assert features_measured_by_every_source(opened.RNA.feats).all()
    counts = zarr.open_group(str(_FIXTURE / "merged.zarr"), mode="r")["RNA/counts"]
    ids = opened.RNA.feats.fetch_all("ids").astype(str).tolist()
    study = np.asarray(_MANIFEST["merged"]["read"]["study"])
    for own, filled in _MANIFEST["merged"]["read"]["zeroFills"].items():
        stored = np.asarray(counts[:, ids.index(filled["gene"])])
        assert stored.tolist() == filled["counts"]
        other = study != own
        assert not stored[other].any() and stored[~other].any()
        values = opened.get_cell_vals("RNA", "I", _feature_name(opened, filled["gene"]))
        assert np.isfinite(values).all()
        assert not np.asarray(values)[other].any()


def test_a_merged_display_that_lists_some_labels_draws_the_others_after_them() -> None:
    opened = _read_only("merged.zarr")
    attributes = dict(opened.cells._get_array("orig_cell_type").attrs)
    assert [entry["value"] for entry in attributes["display"]["categories"]] == [
        "T",
        "B",
    ]
    assert set(opened.cells.fetch_all("orig_cell_type").tolist()) == {"T", "B", "NK"}
    try:
        drawn = _legend(
            opened.plots.composition(category_by="orig_cell_type", show=False)
        )
    finally:
        plt.close("all")
    # NK follows the stored labels in the third generated color.
    assert drawn == [("T", "#1f77b4"), ("B", "#ff7f0e"), ("NK", "#279e68")]


@pytest.mark.parametrize(
    ("store", "assay"), [("rna.zarr", "RNA"), ("merged.zarr", "RNA")]
)
def test_exports_count_the_values_that_1_4_0_summaries_leave_out(
    store, assay, tmp_path
) -> None:
    opened = _read_only(store)
    matrix = zarr.open_group(str(_FIXTURE / store), mode="r")[assay]
    counts = np.asarray(matrix["counts"][:])
    # 1.4.0 stored unsigned counts, whose positive values are the nonzero ones.
    assert counts.dtype.kind == "u"
    assert nonzero_value_count(matrix, matrix["counts"]) == np.count_nonzero(counts)
    cytearc.to_mtx(opened.get_assay(assay), str(tmp_path / "mtx"))
    written = mmread(next((tmp_path / "mtx").glob("matrix.mtx*"))).toarray()
    # Matrix Market files hold features by cells.
    assert np.array_equal(written.T, counts)
    cytearc.to_h5ad(opened.get_assay(assay), str(tmp_path / "counts.h5ad"))
    exported = anndata.read_h5ad(tmp_path / "counts.h5ad")
    assert np.array_equal(np.asarray(exported.X.todense()), counts)


def _new_store(folder: Path, genes: list[str], names: list[str]) -> DataStore:
    """Write a store of 30 cells of ``genes`` with this release."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(1.0, size=(30, len(genes))).astype(np.float32)
    source = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(["T", "DC"] * 15),
                "batch": pd.Categorical(["b2", "b3"] * 15),
            },
            index=[f"new{index}" for index in range(30)],
        ),
        var=pd.DataFrame({"gene_short_name": names}, index=genes),
    )
    source.write_h5ad(folder / "new.h5ad")
    H5adToZarr(
        H5adReader(str(folder / "new.h5ad")),
        zarr_loc=str(folder / "new.zarr"),
        assay_name="RNA",
        nthreads=1,
    ).dump()
    return DataStore(str(folder / "new.zarr"), nthreads=1)


def _merged_with_a_new_store(folder: Path) -> DataStore:
    """Merge a copy of rna.zarr with a store that this release writes of the same genes."""
    old = _writable(folder)
    genes = old.RNA.feats.fetch_all("ids").astype(str).tolist()
    names = old.RNA.feats.fetch_all("names").astype(str).tolist()
    new = _new_store(folder, genes, names)
    DataStoreMerge(
        [old, new], str(folder / "joined.zarr"), ["old", "new"], nthreads=1
    ).dump()
    return DataStore(str(folder / "joined.zarr"), nthreads=1)


def test_a_merge_with_a_store_this_release_writes_keeps_the_1_4_0_cells(
    tmp_path,
) -> None:
    joined = _merged_with_a_new_store(tmp_path)
    ids = joined.cells.fetch_all("ids").astype(str)
    old = np.char.startswith(ids, "old__")
    assert int(old.sum()) == 200 and int((~old).sum()) == 30
    source = _read_only()
    source_ids = source.cells.fetch_all("ids").astype(str).tolist()
    rows = [source_ids.index(cell.removeprefix("old__")) for cell in ids[old]]
    stored = np.asarray(
        zarr.open_group(str(_FIXTURE / "rna.zarr"), mode="r")["RNA/counts"][:]
    )
    merged = np.asarray(
        zarr.open_group(str(tmp_path / "joined.zarr"), mode="r")["RNA/counts"][:]
    )
    assert np.array_equal(merged[old], stored[rows])
    cell_types = joined.cells.fetch_all("orig_cell_type").astype(str)
    assert cell_types[old].tolist() == [
        _MANIFEST["rna"]["read"]["cellType"][row] for row in rows
    ]
    # The 1.4.0 store's categories come first, in its order and colors.
    display = joined.cells._get_array("orig_cell_type").attrs["display"]
    assert [entry["value"] for entry in display["categories"]] == [*_CELL_TYPES, "DC"]
    assert [entry["color"] for entry in display["categories"]][:5] == _CELL_TYPE_COLORS
    phases = joined.cells.fetch_all("orig_phase").astype(str)
    assert phases[old].tolist() == [
        _MANIFEST["rna"]["read"]["phase"][row] for row in rows
    ]


def test_a_merge_of_a_1_4_0_union_merge_keeps_its_columns_and_counts_its_zero_fills(
    tmp_path,
) -> None:
    merged = DataStore(str(_copy(tmp_path, "merged.zarr")), nthreads=1)
    genes = merged.RNA.feats.fetch_all("ids").astype(str).tolist()
    names = merged.RNA.feats.fetch_all("names").astype(str).tolist()
    new = _new_store(tmp_path, genes[:100], names[:100])
    DataStoreMerge(
        [merged, new], str(tmp_path / "again.zarr"), ["earlier", "new"], nthreads=1
    ).dump()
    again = DataStore(str(tmp_path / "again.zarr"), nthreads=1)
    ids = again.cells.fetch_all("ids").astype(str)
    earlier = np.char.startswith(ids, "earlier__")
    read = _MANIFEST["merged"]["read"]
    rows = [read["ids"].index(cell.removeprefix("earlier__")) for cell in ids[earlier]]
    note = again.cells.fetch_all("orig_orig_note").astype(str)
    assert note[earlier].tolist() == [read["note"][row] for row in rows]
    # The earlier merge records no measured features, so its zero fills
    # count as measured in the record of the new merge.
    record = measured_features_by_source(again.RNA.feats)
    assert record is not None
    earlier_source = record.sources.index("earlier")
    assert record.measured[earlier_source].sum() == len(genes)
