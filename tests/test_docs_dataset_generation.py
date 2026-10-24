import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from scripts import regenerate_docs_datasets as generator


def test_labelled_cluster_mask_rejects_blank_and_nan_values():
    mask = generator._labelled_cluster_mask(
        ["B", "nan", "NaN", "", "  ", "CD4 T"],
    )

    assert mask.tolist() == [True, False, False, False, False, True]


def test_prepared_run_columns_are_explicit_full_axis_copies():
    class Cells:
        N = 3

        def __init__(self) -> None:
            self.columns = {"I": np.ones(self.N, dtype=bool)}

        def insert(
            self,
            name: str,
            values: np.ndarray,
            *,
            fill_value: object,
            overwrite: bool,
        ) -> None:
            assert overwrite is True
            if np.issubdtype(values.dtype, np.integer):
                assert fill_value == 0
            else:
                assert np.isnan(fill_value)
            self.columns[name] = np.asarray(values)

        def reset_key(self, key: str) -> None:
            self.columns[key] = np.ones(self.N, dtype=bool)

        def update_key(self, values: np.ndarray, key: str) -> None:
            self.columns[key] &= np.asarray(values, dtype=bool)

    class RunCells:
        columns = ("I", "umap_1", "clusters")

        @staticmethod
        def fetch_all(name: str) -> np.ndarray:
            return {
                "I": np.asarray([True, False, True]),
                "umap_1": np.asarray([1.0, np.nan, 3.0]),
                "clusters": np.asarray([4, -1, 7]),
            }[name]

    store = SimpleNamespace(cells=Cells())
    run = SimpleNamespace(cells=RunCells())

    generator._materialize_run_cell_columns(
        store,
        run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_clusters": "clusters",
        },
        set_selection=True,
    )

    np.testing.assert_allclose(
        store.cells.columns["RNA_UMAP1"],
        [1.0, np.nan, 3.0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        store.cells.columns["RNA_clusters"],
        [4, -1, 7],
    )
    np.testing.assert_array_equal(store.cells.columns["I"], [True, False, True])


class _PreparedCells:
    """Record the teaching columns that a prepared store's cells receive."""

    def __init__(self, n_cells: int) -> None:
        self.N = n_cells
        # A selection that the run's own must replace, not narrow.
        self.columns = {"I": np.arange(n_cells) != 0}

    def insert(
        self,
        name: str,
        values: np.ndarray,
        *,
        fill_value: object,
        overwrite: bool,
    ) -> None:
        # Full-axis values leave no row to the fill.
        assert len(values) == self.N
        assert overwrite is True
        self.columns[name] = np.asarray(values)

    def reset_key(self, key: str) -> None:
        self.columns[key] = np.ones(self.N, dtype=bool)

    def update_key(self, values: np.ndarray, key: str) -> None:
        self.columns[key] &= np.asarray(values, dtype=bool)


def test_prepared_run_columns_hold_the_fills_of_the_run_fields():
    from tests.test_pipeline_run_foundation import _completed_run, _root

    # A completed run of four cells that analyzed cells 0 and 2, with UMAP
    # coordinates and cluster labels.
    run = _completed_run(_root())
    store = SimpleNamespace(cells=_PreparedCells(4))

    generator._materialize_run_cell_columns(
        store,
        run,
        {"RNA_UMAP1": "umap_1", "RNA_clusters": "clusters"},
        set_selection=False,
    )

    # Unselected cells hold each field's fill: NaN for coordinates and -1
    # for labels, which no cluster takes.
    np.testing.assert_allclose(
        store.cells.columns["RNA_UMAP1"], [1.0, np.nan, 3.0, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(store.cells.columns["RNA_clusters"], [0, -1, 2, -1])
    # Without set_selection the prepared store keeps its own selection; with
    # it, the run's selection replaces that selection.
    np.testing.assert_array_equal(store.cells.columns["I"], [False, True, True, True])
    generator._materialize_run_cell_columns(store, run, {}, set_selection=True)
    np.testing.assert_array_equal(store.cells.columns["I"], [True, False, True, False])

    # A field that the run did not record is refused, and the selection is
    # left as it was.
    store.cells.columns["I"] = np.ones(4, dtype=bool)
    with pytest.raises(KeyError, match="Pipeline run has no prepared field 'paris'"):
        generator._materialize_run_cell_columns(
            store, run, {"RNA_paris_cluster": "paris"}, set_selection=True
        )
    assert "RNA_paris_cluster" not in store.cells.columns
    np.testing.assert_array_equal(store.cells.columns["I"], np.ones(4, dtype=bool))
    # The run's fields must cover every cell of the prepared store.
    other = SimpleNamespace(cells=_PreparedCells(5))
    with pytest.raises(
        ValueError, match="Pipeline field 'clusters' is not a full-axis vector"
    ):
        generator._materialize_run_cell_columns(
            other, run, {"RNA_clusters": "clusters"}, set_selection=False
        )
    assert "RNA_clusters" not in other.cells.columns


def test_drop_retired_assay_state_removes_leftover_groups(tmp_path):
    import zarr

    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    rna = root.create_group("RNA")
    rna.attrs["is_assay"] = True
    rna.create_group("state")
    rna.create_group("artifacts")
    cell_data = root.create_group("cellData")
    cell_data.create_group("ids")

    removed = generator._drop_retired_assay_state(root)

    assert removed == ("RNA/state",)
    assert "state" not in root["RNA"]
    assert "artifacts" in root["RNA"]


class _RecordingPipeline:
    """Record the recipe calls of a dataset analysis and return named runs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _run(self, recipe: str, *args: object, **settings: object) -> object:
        self.calls.append((recipe, args, settings))
        return SimpleNamespace(recipe=recipe, label=settings["label"])

    def rna(self, **settings: object) -> object:
        return self._run("rna", **settings)

    def atac(self, **settings: object) -> object:
        return self._run("atac", **settings)

    def adt(self, **settings: object) -> object:
        return self._run("adt", **settings)

    def multimodal(self, runs: list[object], **settings: object) -> object:
        return self._run("multimodal", runs, **settings)


def _record_teaching_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, str], bool]]:
    copied: list[tuple[str, dict[str, str], bool]] = []

    def copy_columns(
        store: object,
        run: SimpleNamespace,
        columns: dict[str, str],
        *,
        set_selection: bool,
    ) -> None:
        copied.append((run.label, dict(columns), set_selection))

    monkeypatch.setattr(generator, "_materialize_run_cell_columns", copy_columns)
    return copied


def test_embedding_columns_name_the_umap_and_leiden_fields_of_a_run():
    assert generator._embedding_columns("ADT", "1.0") == {
        "ADT_UMAP1": "umap_1",
        "ADT_UMAP2": "umap_2",
        "ADT_leiden_cluster": "leiden_1.0",
    }


def test_citeseq_analysis_integrates_modality_runs_of_the_same_cells(monkeypatch):
    copied = _record_teaching_columns(monkeypatch)
    pipeline = _RecordingPipeline()

    generator._analyze_citeseq(SimpleNamespace(pipeline=pipeline))

    recipes = [
        (recipe, settings["label"]) for recipe, _args, settings in pipeline.calls
    ]
    assert recipes == [
        ("rna", "rna"),
        ("adt", "adt"),
        ("multimodal", "snn"),
        ("multimodal", "wnn"),
    ]
    rna = pipeline.calls[0]
    adt = pipeline.calls[1]
    # The ADT run analyzes the RNA run's cells, and both multimodal runs
    # integrate the two runs.
    assert adt[2]["cells"].label == "rna"
    for _recipe, (runs,), settings in pipeline.calls[2:]:
        assert [run.label for run in runs] == ["rna", "adt"]
        assert settings["method"] == settings["label"]
    assert rna[2]["markers"] is False
    # The teaching columns keep the names that the tutorials read.
    assert copied == [
        ("adt", generator._embedding_columns("ADT", "1.0"), False),
        ("snn", generator._embedding_columns("RNA+ADT", "1.75"), False),
        ("wnn", generator._embedding_columns("RNA+ADT_wnn", "1.75"), False),
        (
            "rna",
            {**generator._embedding_columns("RNA", "1.0"), "RNA_clusters": "clusters"},
            True,
        ),
    ]


def test_teaseq_analysis_integrates_three_modality_runs(monkeypatch):
    copied = _record_teaching_columns(monkeypatch)
    pipeline = _RecordingPipeline()

    generator._analyze_teaseq(SimpleNamespace(pipeline=pipeline))

    recipes = [
        (recipe, settings["label"]) for recipe, _args, settings in pipeline.calls
    ]
    assert recipes == [
        ("rna", "rna"),
        ("atac", "atac"),
        ("adt", "adt"),
        ("multimodal", "snn"),
        ("multimodal", "wnn"),
    ]
    for recipe, _args, settings in pipeline.calls[1:3]:
        assert settings["cells"].label == "rna", recipe
    for _recipe, (runs,), _settings in pipeline.calls[3:]:
        assert [run.label for run in runs] == ["rna", "atac", "adt"]
    assert [label for label, _columns, _selection in copied] == [
        "rna",
        "atac",
        "adt",
        "snn",
        "wnn",
    ]
    assert not any(selection for _label, _columns, selection in copied)


def test_kang_recipes_subset_current_sources_by_cluster_labels():
    for dataset in (
        generator.KANG_CONTROL_DATASET,
        generator.KANG_STIMULATED_DATASET,
    ):
        recipe = generator.RECIPES[dataset]

        assert isinstance(recipe, generator.DerivedDatasetRecipe)
        assert recipe.source_datasets == (dataset,)
        assert recipe.derive is generator._derive_labelled_kang_store


def test_derived_sources_prefer_stores_built_in_same_run(tmp_path):
    local = tmp_path / "local"
    (local / generator.STORE_NAME).mkdir(parents=True)
    remote = tmp_path / "remote"
    (remote / generator.STORE_NAME).mkdir(parents=True)

    class Repository:
        def __init__(self) -> None:
            self.downloads: list[str] = []

        def download_dataset(
            self,
            name: str,
            destination: str | Path,
            *,
            zarr: bool,
        ) -> Path:
            assert Path(destination) == tmp_path / "work"
            assert zarr is True
            self.downloads.append(name)
            return remote

    repository = Repository()
    resolved = generator._resolve_derived_sources(
        repository=repository,
        source_datasets=["local_source", "remote_source"],
        local_sources={"local_source": local},
        work=tmp_path / "work",
    )

    assert resolved == {
        "local_source": local,
        "remote_source": remote,
    }
    assert repository.downloads == ["remote_source"]


def test_derived_sources_reuse_store_from_an_earlier_invocation(tmp_path):
    source = tmp_path / "tabula_sapiens"
    store = source / generator.STORE_NAME
    store.mkdir(parents=True)
    payload = store / "counts"
    payload.write_bytes(b"prepared source counts")

    class Repository:
        def download_dataset(self, *args, **kwargs):
            raise AssertionError("An existing local source must not be downloaded")

    resolved = generator._resolve_derived_sources(
        repository=Repository(),
        source_datasets=["tabula_sapiens"],
        local_sources={},
        work=tmp_path / "_source",
    )

    assert resolved == {"tabula_sapiens": source}
    assert payload.read_bytes() == b"prepared source counts"


def test_external_source_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"checksum-pinned source"
    source = generator.ExternalSource(
        filename="source.bin",
        url="https://example.test/source.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    requested: list[tuple[str, int]] = []

    def open_source(url: str, *, timeout: int) -> BytesIO:
        requested.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(generator, "urlopen", open_source)
    downloaded = generator._download_external_source(source, tmp_path)

    assert downloaded.read_bytes() == payload
    assert requested == [
        (source.url, generator.EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS),
    ]

    requested.clear()
    assert generator._download_external_source(source, tmp_path) == downloaded
    assert requested == []


def test_external_source_checksum_failure_removes_partial_file(
    tmp_path,
    monkeypatch,
):
    source = generator.ExternalSource(
        filename="source.bin",
        url="https://example.test/source.bin",
        sha256="0" * 64,
    )
    monkeypatch.setattr(
        generator,
        "urlopen",
        lambda _url, **_kwargs: BytesIO(b"wrong"),
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        generator._download_external_source(source, tmp_path)

    assert not (tmp_path / "source.bin").exists()
    assert not (tmp_path / "source.bin.partial").exists()


def test_teaseq_recipe_is_explicit_but_excluded_from_all():
    recipe = generator.RECIPES[generator.TEASEQ_DATASET]

    assert isinstance(recipe, generator.ExternalDatasetRecipe)
    assert generator.TEASEQ_DATASET not in generator._datasets_for_all()
    assert {source.sha256 for source in recipe.sources} == {
        "501a1716a370a3958a71a1aec8e8620f1496d115329d6943ed2bfa450eefac9f",
        "012e6a61de2a79bd96302353536d0a8e44f527007df8f32a4c44417e2bfc1197",
    }
    parameters = {stage: dict(values) for stage, values in recipe.analysis_parameters}
    assert parameters["integration"]["assayOrder"] == ("RNA", "ATAC", "ADT")
    assert parameters["neighborhood"]["selfFreeNeighbors"] == 20


def test_dataset_publication_requires_matching_archive_and_manifest(
    tmp_path, monkeypatch
):
    from scripts import publish_docs_datasets as publisher

    dataset = generator.TEASEQ_DATASET
    directory = tmp_path / dataset
    directory.mkdir()
    archive = directory / publisher.ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        entry = tarfile.TarInfo("data.zarr/zarr.json")
        payload = b'{"zarr_format": 3, "node_type": "group"}'
        entry.size = len(payload)
        handle.addfile(entry, BytesIO(payload))
    manifest_path = tmp_path / f"{dataset}_manifest.json"
    manifest_path.write_text(
        json.dumps({"archiveSha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    )
    monkeypatch.setattr(publisher, "MANIFEST_DIR", tmp_path)

    plan = publisher._plan_dataset(dataset, tmp_path)

    assert dict(plan.upload) == {
        archive: f"cytearc_docs/{dataset}/data.zarr.tar.gz",
        manifest_path: f"cytearc_docs/{dataset}/manifest.json",
    }
    assert plan.remove == []

    archive.write_bytes(b"changed after manifest creation")
    with pytest.raises(RuntimeError, match="does not match its manifest checksum"):
        publisher._plan_dataset(dataset, tmp_path)


def test_all_cannot_silently_ignore_named_external_dataset(capsys):
    with pytest.raises(SystemExit) as exited:
        generator.main(["--all", generator.TEASEQ_DATASET])

    assert exited.value.code == 2
    assert (
        capsys.readouterr()
        .err.rstrip()
        .endswith("error: --all cannot be combined with named datasets")
    )


def test_a_run_needs_named_datasets_or_all(capsys):
    with pytest.raises(SystemExit) as exited:
        generator.main([])

    assert exited.value.code == 2
    assert (
        capsys.readouterr()
        .err.rstrip()
        .endswith("error: name at least one dataset or pass --all")
    )


def test_teaseq_annotations_map_original_barcodes_through_well_suffix(tmp_path):
    n_cells = generator.TEASEQ_TOTAL_CELLS
    n_publication_cells = generator.TEASEQ_MATCHED_PUBLICATION_CELLS
    original = np.asarray([f"BC{index:05d}-1" for index in range(n_cells)])
    publication = pd.DataFrame(
        {
            "barcode": [f"BC{index:05d}-3" for index in range(n_publication_cells)],
            "seurat_pbmc_cell_type": ["T cell"] * n_publication_cells,
            "seurat_pbmc_type_color": ["#123456"] * n_publication_cells,
            "predicted.celltype.l2": ["CD4 T"] * n_publication_cells,
            "predicted.celltype.l2.score": [0.9] * n_publication_cells,
        }
    )
    archive_path = tmp_path / generator.TEASEQ_ANNOTATIONS_NAME
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Figure4_SourceData2_TypeLabelsUMAP.csv",
            publication.to_csv(index=False),
        )

    class Cells:
        def __init__(self) -> None:
            self.columns = {
                "original_barcodes": original,
                "well_id": np.full(n_cells, "X066-MP0C1W3"),
            }

        def fetch_all(self, name: str) -> np.ndarray:
            return self.columns[name]

        def insert(
            self,
            name: str,
            values: np.ndarray,
            *,
            overwrite: bool,
        ) -> None:
            assert overwrite is True
            self.columns[name] = np.asarray(values)

        def update_key(self, values: np.ndarray, key: str) -> None:
            self.columns[key] = np.asarray(values) & self.columns[key]

    class Store:
        cells = Cells()

    store = Store()
    store.cells.columns["I"] = np.ones(n_cells, dtype=bool)
    generator._add_teaseq_annotations(store, tmp_path)

    assert int(store.cells.columns["I"].sum()) == n_publication_cells
    assert store.cells.columns["publication_barcode"][0] == "BC00000-3"
    assert store.cells.columns["tea_cell_type"][0] == "T cell"
    assert store.cells.columns["tea_cell_type"][-1] == ""
    assert np.isnan(store.cells.columns["tea_prediction_score"][-1])
