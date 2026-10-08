import json
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import cast

import pytest


_EXPECTED_EXPORTS = {
    "ArtifactLineage": "cytearc.storage.lineage",
    "ArtifactRef": "cytearc.storage.refs",
    "ArtifactResolutionError": "cytearc.storage.errors",
    "ArtifactStatus": "cytearc.storage.artifacts",
    "CSVReader": "cytearc.readers",
    "CSVtoZarr": "cytearc.writers",
    "CrDirReader": "cytearc.readers",
    "CrH5Reader": "cytearc.readers",
    "CrReader": "cytearc.readers",
    "CrToZarr": "cytearc.writers",
    "DataStore": "cytearc.datastore.datastore",
    "DataStoreSummary": "cytearc.datastore.summary",
    "DataStoreMerge": "cytearc.merge",
    "EnrichmentResult": "cytearc.features.enrichment.results",
    "FateMappingResult": "cytearc.trajectory.results",
    "H5adInspectResult": "cytearc.readers",
    "H5adImportResult": "cytearc.writers",
    "H5adReader": "cytearc.readers",
    "H5adToZarr": "cytearc.writers",
    "MtxReader": "cytearc.readers",
    "MtxToZarr": "cytearc.writers",
    "SeuratImportResult": "cytearc.writers",
    "SeuratInspectResult": "cytearc.readers",
    "SeuratReader": "cytearc.readers",
    "SeuratToZarr": "cytearc.writers",
    "LabelTransferResult": "cytearc.mapping.models",
    "MappingReference": "cytearc.mapping.reference",
    "MappingResult": "cytearc.mapping.models",
    "mount_datastore": "cytearc.datastore.datastore",
    "PseudotimeAggregationResult": "cytearc.trajectory.results",
    "PseudotimeMarkerResult": "cytearc.trajectory.results",
    "PseudotimeScoreResult": "cytearc.trajectory.results",
    "PipelineExecutionError": "cytearc.datastore.pipeline_run",
    "PipelineRun": "cytearc.datastore.pipeline_run",
    "SparseToZarr": "cytearc.writers",
    "SubsetZarr": "cytearc.writers",
    "clean_array": "cytearc.utils",
    "configure_output": "cytearc.utils",
    "controlled_compute": "cytearc.utils",
    "create_zarr_count_assay": "cytearc.writers",
    "create_zarr_dataset": "cytearc.writers",
    "create_zarr_obj_array": "cytearc.writers",
    "chunked_to_zarr": "cytearc.writers",
    "inspect_h5ad": "cytearc.readers",
    "inspect_mtx": "cytearc.readers",
    "inspect_seurat": "cytearc.readers",
    "load_zarr": "cytearc.storage.stores",
    "logger": "cytearc.utils",
    "permute_into_chunks": "cytearc.utils",
    "read_gmt": "cytearc.features.enrichment.net",
    "rescale_array": "cytearc.utils",
    "rolling_window": "cytearc.utils",
    "set_verbosity": "cytearc.utils",
    "compute_with_progress": "cytearc.utils",
    "subset_assay_zarr": "cytearc.writers",
    "to_h5ad": "cytearc.writers",
    "to_mtx": "cytearc.writers",
    "tqdmbar": "cytearc.utils",
    "tqdm_params": "cytearc.utils",
    "write_renorm_subset_to_zarr": "cytearc.writers",
}

_EXPECTED_MODULE_ATTRIBUTES = {
    "assay": "cytearc.assay",
    "cytebase": "cytearc.cytebase",
    "datastore": "cytearc.datastore",
    "embeddings": "cytearc.embeddings",
    "features": "cytearc.features",
    "mapping": "cytearc.mapping",
    "matrix": "cytearc.matrix",
    "merge": "cytearc.merge",
    "metadata": "cytearc.metadata",
    "metrics": "cytearc.metrics",
    "quality_control": "cytearc.quality_control",
    "readers": "cytearc.readers",
    "storage": "cytearc.storage",
    "utils": "cytearc.utils",
    "writers": "cytearc.writers",
}

_EXPECTED_UTILS_EXPORTS = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "configure_output",
    "set_verbosity",
    "rescale_array",
    "clean_array",
    "permute_into_chunks",
    "compute_with_progress",
    "controlled_compute",
    "process_rss_mb",
    "array_digest",
    "rolling_window",
]

_EXPECTED_PLOTTING_EXPORTS = (
    "CategoricalScale",
    "CellField",
    "ColorScale",
    "DensityOverlay",
    "FeatureRef",
    "Highlight",
    "LegendSpec",
    "NormalizationSpec",
    "PlotProvenance",
    "PlotOutput",
    "PlotOutputSettings",
    "PlotPanelTarget",
    "PlotRecipe",
    "PlotRecipeResult",
    "PlotResult",
    "PlotStep",
    "SizeScale",
    "StudyDesign",
    "THEMES",
    "cluster_tree",
    "cluster_connectivity",
    "compose_results",
    "composition",
    "distribution",
    "dotplot",
    "elbow",
    "embedding",
    "embedding_raster",
    "graph_qc",
    "highly_variable_features",
    "label_panels",
    "marker_heatmap",
    "mapping_calibration",
    "mapping_confusion",
    "mapping_evidence",
    "mapping_score",
    "matrixplot",
    "modality_weights",
    "pseudotime_heatmap",
    "qc",
    "run_recipe",
    "theme_context",
)


def _run_probe(source: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


def _missing_distribution(_name: str) -> str:
    raise PackageNotFoundError


def test_version_resolution_prefers_distribution(tmp_path: Path):
    import cytearc

    assert (
        cytearc._resolve_version(lambda _name: "9.8.7", tmp_path / "missing.py")
        == "9.8.7"
    )


def test_version_resolution_reads_generated_file(tmp_path: Path):
    import cytearc

    version_path = tmp_path / "_version.py"
    version_path.write_text("__version__ = version = '9.8.7'\n", encoding="utf-8")

    assert cytearc._resolve_version(_missing_distribution, version_path) == "9.8.7"


def test_version_resolution_rejects_missing_or_invalid_file(tmp_path: Path):
    import cytearc

    version_path = tmp_path / "_version.py"
    assert (
        cytearc._resolve_version(_missing_distribution, version_path) == "unavailable"
    )

    version_path.write_text("version is missing\n", encoding="utf-8")
    assert (
        cytearc._resolve_version(_missing_distribution, version_path) == "unavailable"
    )


@pytest.fixture(scope="module")
def cold_import_probe() -> dict[str, object]:
    """Observe one fresh interpreter through the lazy import contract.

    Each step records its observation before the next step imports more: the
    bare import, the zarr warning filter, module attributes, then a star
    import of the exports.
    """
    lazy_names = [*_EXPECTED_EXPORTS, *_EXPECTED_MODULE_ATTRIBUTES]
    return _run_probe(
        f"""
import importlib
import json
import sys
import warnings

import cytearc

heavy_modules = {{
    "dask",
    "h5py",
    "hnswlib",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "zarr",
}}
lazy_names = {lazy_names!r}
bare = {{
    "boundLazyNames": sorted(name for name in lazy_names if name in vars(cytearc)),
    "heavyModules": sorted(heavy_modules.intersection(sys.modules)),
    "plotsInDir": "plots" in dir(cytearc),
    "plottingInDir": "plotting" in dir(cytearc),
    "cytearcModules": sorted(
        name for name in sys.modules
        if name == "cytearc" or name.startswith("cytearc.")
    ),
    "versionIsSet": isinstance(cytearc.__version__, str) and bool(cytearc.__version__),
}}

zarr_was_loaded = "zarr" in sys.modules

import numpy as np
import zarr
from zarr.storage import MemoryStore

with warnings.catch_warnings(record=True) as caught:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_array("text", data=np.asarray(["a", "bb"]))
zarr_warning = {{
    "unstableWarnings": [
        type(item.message).__name__
        for item in caught
        if type(item.message).__name__ == "UnstableSpecificationWarning"
    ],
    "zarrWasLoaded": zarr_was_loaded,
}}

module_names = {_EXPECTED_MODULE_ATTRIBUTES!r}
before = sorted(name for name in module_names if name in vars(cytearc))
advertised = sorted(name for name in module_names if name in dir(cytearc))
identical = {{
    name: getattr(cytearc, name) is importlib.import_module(module_name)
    for name, module_name in module_names.items()
}}
cached = sorted(name for name in module_names if name in vars(cytearc))
module_attributes = {{
    "advertised": advertised,
    "before": before,
    "cached": cached,
    "identical": identical,
}}

namespace = {{}}
exec("from cytearc import *", namespace)
bound = sorted(name for name in namespace if not name.startswith("__"))
print(json.dumps({{
    "bare": bare,
    "moduleAttributes": module_attributes,
    "starImport": {{"bound": bound}},
    "zarrWarning": zarr_warning,
}}))
"""
    )


def test_bare_import_is_lazy(cold_import_probe: dict[str, object]):
    assert cold_import_probe["bare"] == {
        "boundLazyNames": [],
        "heavyModules": [],
        "plotsInDir": False,
        "plottingInDir": False,
        "cytearcModules": ["cytearc", "cytearc._facade"],
        "versionIsSet": True,
    }


def test_utils_package_is_lazy():
    result = _run_probe(
        f"""
import json
import sys
import cytearc.utils as utils

exports = {_EXPECTED_UTILS_EXPORTS!r}
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(utils)),
    "bound": sorted(name for name in exports if name in vars(utils)),
    "exports": utils.__all__,
    "heavyModules": sorted({{"numba", "numpy", "scipy", "zarr"}}.intersection(sys.modules)),
}}))
"""
    )

    assert result == {
        "advertised": sorted(_EXPECTED_UTILS_EXPORTS),
        "bound": [],
        "exports": _EXPECTED_UTILS_EXPORTS,
        "heavyModules": [],
    }


def test_clustering_package_is_lazy():
    exports = [
        "CoalesceTree",
        "ParisClusterDiagnostic",
        "ParisClusteringResult",
        "adaptive_cut",
        "leiden_membership",
        "make_digraph",
        "straight_cut",
    ]
    result = _run_probe(
        f"""
import json
import sys
import cytearc.clustering as clustering

exports = {exports!r}
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(clustering)),
    "bound": sorted(name for name in exports if name in vars(clustering)),
    "exports": clustering.__all__,
    "heavyModules": sorted({{"numba", "numpy", "scipy"}}.intersection(sys.modules)),
}}))
"""
    )

    assert result == {
        "advertised": sorted(exports),
        "bound": [],
        "exports": exports,
        "heavyModules": [],
    }


def test_plotting_package_is_lazy():
    result = _run_probe(
        f"""
import json
import sys
import cytearc.plotting as plotting

exports = {_EXPECTED_PLOTTING_EXPORTS!r}
concrete = (
        "cytearc.plotting.cluster_connectivity",
    "cytearc.plotting.composition",
    "cytearc.plotting.diagnostics",
    "cytearc.plotting.distribution",
    "cytearc.plotting.embedding",
    "cytearc.plotting.embedding_raster",
    "cytearc.plotting.heatmaps",
    "cytearc.plotting.mapping",
    "cytearc.plotting.modality_weights",
    "cytearc.plotting.recipes",
    "cytearc.plotting.summary",
)
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(plotting)),
    "bound": sorted(name for name in exports if name in vars(plotting)),
    "concreteModules": sorted(set(concrete).intersection(sys.modules)),
    "exports": plotting.__all__,
}}))
"""
    )

    assert result == {
        "advertised": sorted(_EXPECTED_PLOTTING_EXPORTS),
        "bound": [],
        "concreteModules": [],
        "exports": list(_EXPECTED_PLOTTING_EXPORTS),
    }


def test_utils_surface_preserves_module_metadata():
    utils = import_module("cytearc.utils")

    assert utils.__all__ == _EXPECTED_UTILS_EXPORTS
    for name in _EXPECTED_UTILS_EXPORTS:
        value = getattr(utils, name)
        if callable(value):
            assert value.__module__ == "cytearc.utils"


def test_public_exports_match_canonical_objects():
    import cytearc

    assert cytearc.__all__ == list(_EXPECTED_EXPORTS)
    assert isinstance(cytearc.__version__, str)
    assert cytearc.__version__
    assert set(_EXPECTED_EXPORTS).issubset(dir(cytearc))

    for name, module_name in _EXPECTED_EXPORTS.items():
        canonical = getattr(import_module(module_name), name)
        exported = getattr(cytearc, name)
        assert exported is canonical
        assert vars(cytearc)[name] is canonical
        if callable(exported):
            assert exported.__module__ == module_name


def test_marker_facade_does_not_export_layout_internals():
    markers = import_module("cytearc.features.markers")
    internal_names = {
        "MARKER_STAT_COLUMNS",
        "load_marker_table",
    }

    assert internal_names.isdisjoint(markers.__all__)
    assert internal_names.isdisjoint(dir(markers))


def test_graph_package_does_not_export_artifact_references():
    graph = import_module("cytearc.graph")

    assert "ArtifactRef" not in graph.__all__
    assert not hasattr(graph, "ArtifactRef")


def test_domain_packages_export_canonical_objects():
    exports = {
        (
            "cytearc.clustering",
            "ParisClusterDiagnostic",
        ): "cytearc.clustering.paris_multiscale",
        (
            "cytearc.clustering",
            "ParisClusteringResult",
        ): "cytearc.clustering.paris_multiscale",
        ("cytearc.clustering", "adaptive_cut"): "cytearc.clustering.paris_multiscale",
        ("cytearc.embeddings", "fit_harmony"): "cytearc.embeddings.harmony",
        ("cytearc.features", "binned_sampling"): "cytearc.features.scoring",
        ("cytearc.features", "fit_lowess"): "cytearc.features.variability",
        (
            "cytearc.features",
            "EnrichmentResult",
        ): "cytearc.features.enrichment.results",
        ("cytearc.features", "read_gmt"): "cytearc.features.enrichment.net",
        (
            "cytearc.features",
            "select_highly_variable_features",
        ): "cytearc.features.variability",
        ("cytearc.mapping", "LabelTransferResult"): "cytearc.mapping.models",
        ("cytearc.mapping", "MappingReference"): "cytearc.mapping.reference",
        ("cytearc.mapping", "MappingResult"): "cytearc.mapping.models",
        (
            "cytearc.features",
            "find_markers_by_rank",
        ): "cytearc.features.markers.search",
        (
            "cytearc.features",
            "compare_group_distributions",
        ): "cytearc.features.statistical",
        (
            "cytearc.features",
            "StatisticalTestResult",
        ): "cytearc.features.statistical",
        (
            "cytearc.features",
            "GroupComparisonResult",
        ): "cytearc.features.statistical",
        (
            "cytearc.features",
            "resolve_group_order",
        ): "cytearc.features.statistical",
        ("cytearc.matrix", "ChunkedArray"): "cytearc.matrix.chunked",
        (
            "cytearc.metrics",
            "ClusterSeparabilityResult",
        ): "cytearc.metrics.cluster_separability",
        ("cytearc.metrics", "clisi_knn"): "cytearc.metrics.lisi",
        ("cytearc.metrics", "compute_lisi"): "cytearc.metrics.lisi",
        (
            "cytearc.metrics",
            "evaluate_cluster_separability",
        ): "cytearc.metrics.cluster_separability",
        ("cytearc.metrics", "graph_connectivity"): "cytearc.metrics.connectivity",
        ("cytearc.metrics", "ilisi_knn"): "cytearc.metrics.lisi",
        ("cytearc.metrics", "silhouette_scoring"): "cytearc.metrics.silhouette",
        (
            "cytearc.quality_control",
            "assign_cell_cycle_phase",
        ): "cytearc.quality_control.cell_cycle",
        (
            "cytearc.quality_control",
            "simulate_doublet_pairs",
        ): "cytearc.quality_control.doublets",
        (
            "cytearc.quality_control",
            "s_phase_genes",
        ): "cytearc.quality_control.cell_cycle_genes",
    }

    for (package_name, symbol), module_name in exports.items():
        package = import_module(package_name)
        canonical = import_module(module_name)
        assert getattr(package, symbol) is getattr(canonical, symbol)


def test_star_import_matches_all(cold_import_probe: dict[str, object]):
    result = cast(dict[str, object], cold_import_probe["starImport"])

    assert result["bound"] == sorted(_EXPECTED_EXPORTS)


def test_de_facto_module_attributes_resolve_lazily(
    cold_import_probe: dict[str, object],
):
    result = cast(dict[str, object], cold_import_probe["moduleAttributes"])

    expected_names = sorted(_EXPECTED_MODULE_ATTRIBUTES)
    assert result["advertised"] == expected_names
    assert result["before"] == []
    assert result["cached"] == expected_names
    assert result["identical"] == dict.fromkeys(_EXPECTED_MODULE_ATTRIBUTES, True)


def test_legacy_dataset_download_api_is_absent():
    from importlib.util import find_spec

    import cytearc

    assert not hasattr(cytearc, "fetch_dataset")
    assert not hasattr(cytearc, "show_available_datasets")
    assert find_spec("cytearc.readers.datasets") is None


def test_retired_merge_names_are_absent():
    import cytearc
    import cytearc.merge as merge_module

    for name in ("DatasetMerge", "AssayMerge"):
        assert not hasattr(cytearc, name)
        assert not hasattr(merge_module, name)


def test_retired_loom_names_are_absent():
    from importlib import import_module

    import cytearc
    import cytearc.readers as readers_module
    import cytearc.writers as writers_module

    for module in (cytearc, readers_module, writers_module):
        for name in ("LoomReader", "LoomToZarr"):
            assert not hasattr(module, name)
    for module_name in (
        "cytearc.readers.loom",
        "cytearc.writers.loom",
        "cytearc.agent.ingest.loom",
    ):
        with pytest.raises(ModuleNotFoundError) as missing:
            import_module(module_name)
        assert missing.value.name == module_name or module_name.startswith(
            f"{missing.value.name}."
        )


def test_retired_dask_names_are_absent():
    import cytearc
    import cytearc.storage.materialize as storage_materialize
    import cytearc.utils as utils_module
    import cytearc.utils.compute as compute_module
    import cytearc.writers as writers_module
    import cytearc.writers._materialize as writers_materialize

    for name in ("show_dask_progress", "dask_to_zarr"):
        assert not hasattr(cytearc, name)
        assert not hasattr(utils_module, name)
        assert not hasattr(writers_module, name)
    assert not hasattr(compute_module, "show_dask_progress")
    assert not hasattr(storage_materialize, "dask_to_zarr")
    assert not hasattr(writers_materialize, "dask_to_zarr")
    assert "compute_with_progress" in dir(utils_module)


def test_unbounded_diffusion_operator_is_absent():
    import cytearc.neighbors as neighbors
    import cytearc.neighbors.diffusion as diffusion

    assert "diffusion_operator" not in neighbors.__all__
    assert "diffusion_operator" not in dir(neighbors)
    assert not hasattr(neighbors, "diffusion_operator")
    assert not hasattr(diffusion, "diffusion_operator")


def test_lazy_facades_clear_cached_exports_on_reload():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib

cases = (
    ("cytearc", "MappingResult"),
    ("cytearc.features", "fit_lowess"),
    ("cytearc.readers", "CSVReader"),
    ("cytearc.writers", "CSVtoZarr"),
    ("cytearc.merge", "DataStoreMerge"),
    ("cytearc.utils", "clean_array"),
    ("cytearc.neighbors", "calc_snn"),
    ("cytearc.clustering", "straight_cut"),
    ("cytearc.embeddings", "initial_embedding"),
    ("cytearc.trajectory", "PseudotimeScoreResult"),
    ("cytearc.plotting", "embedding"),
)

for module_name, export_name in cases:
    module = importlib.import_module(module_name)
    original = getattr(module, export_name)
    setattr(module, export_name, object())
    importlib.reload(module)
    assert export_name not in vars(module), (module_name, export_name)
    assert getattr(module, export_name) is original, (module_name, export_name)
""",
        ],
        check=True,
    )


def test_lazy_facade_imports_submodules_and_exports_on_first_access(
    tmp_path, monkeypatch
):
    import importlib

    from cytearc._facade import lazy_facade

    package = tmp_path / "cytearc_lazy_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "child.py").write_text("VALUE = 7\n")
    (package / "source.py").write_text("def exported():\n    return 'exported'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    names = (
        "cytearc_lazy_probe",
        "cytearc_lazy_probe.child",
        "cytearc_lazy_probe.source",
    )
    try:
        module = importlib.import_module("cytearc_lazy_probe")
        module.__getattr__, module.__dir__ = lazy_facade(
            "cytearc_lazy_probe",
            {"exported": ".source"},
            modules=("child",),
        )
        assert "cytearc_lazy_probe.child" not in sys.modules
        assert module.child.VALUE == 7
        assert vars(module)["child"] is sys.modules["cytearc_lazy_probe.child"]
        assert module.exported() == "exported"
        assert {"child", "exported"} <= set(dir(module))
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            module.missing  # noqa: B018
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_lazy_facade_binding_keeps_patched_exports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import cytearc.readers as readers

patched = object()
readers.H5adReader = patched
assert readers.inspect_h5ad.__module__ == "cytearc.readers"
assert readers.H5adReader is patched
""",
        ],
        check=True,
    )


def test_lazy_facades_rename_only_objects_they_own():
    result = _run_probe(
        """
import json

import loguru

import cytearc
import cytearc.merge as merge
import cytearc.utils as utils
import cytearc.writers as writers

shared = cytearc.ArtifactRef.__getstate__
exports = (merge.MergePlan, writers.H5adImportResult, utils.logger)
print(json.dumps({
    "exportModules": [value.__module__ for value in exports[:2]],
    "loggerPatched": "__module__" in vars(loguru.logger),
    "sharedModule": shared.__module__,
}))
"""
    )

    assert result == {
        "exportModules": ["cytearc.merge", "cytearc.writers"],
        "loggerPatched": False,
        "sharedModule": "dataclasses",
    }


def test_zarr_warning_filter_does_not_make_import_eager(
    cold_import_probe: dict[str, object],
):
    assert cold_import_probe["zarrWarning"] == {
        "unstableWarnings": [],
        "zarrWasLoaded": False,
    }


def test_modern_plotting_surface_matches_baseline():
    plotting = import_module("cytearc.plotting")

    assert tuple(plotting.__all__) == _EXPECTED_PLOTTING_EXPORTS
    for name in _EXPECTED_PLOTTING_EXPORTS:
        assert hasattr(plotting, name)


def test_plotting_submodule_import_does_not_clobber_function_export():
    plotting = import_module("cytearc.plotting")
    canonical_module = import_module("cytearc.plotting.embedding")

    assert plotting.embedding is canonical_module.embedding


def test_legacy_plotting_surface_remains_absent():
    from importlib.util import find_spec

    from cytearc import DataStore
    from cytearc.datastore.plot_accessor import DataStorePlotAccessor

    plotting = import_module("cytearc.plotting")

    assert hasattr(DataStore, "plots")
    assert DataStorePlotAccessor.__module__ == "cytearc.datastore.plot_accessor"
    assert not hasattr(import_module("cytearc"), "DataStorePlotAccessor")
    assert not [name for name in dir(DataStore) if name.startswith("plot_")]
    assert "unified_embedding" not in plotting.__all__
    assert not hasattr(plotting, "unified_embedding")
    assert find_spec("cytearc.plots") is None
    assert find_spec("cytearc.plotting._legacy") is None
    assert find_spec("cytearc.plotting.unified") is None


def test_pipeline_accessor_has_a_public_import_path():
    from cytearc import DataStore, PipelineExecutionError, PipelineRun
    from cytearc.datastore.pipeline_accessor import PipelineAccessor

    assert hasattr(DataStore, "pipeline")
    assert PipelineAccessor.__module__ == "cytearc.datastore.pipeline_accessor"
    assert PipelineRun.__module__ == "cytearc.datastore.pipeline_run"
    assert PipelineExecutionError.__module__ == "cytearc.datastore.pipeline_run"
    assert {"open", "list_runs", "run"} <= set(vars(PipelineAccessor))
