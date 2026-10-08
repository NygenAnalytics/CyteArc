import argparse
import os
import platform
import subprocess
import sys
import tempfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


_RETIRED_MODULES = {
    "cytearc/_types.py",
    "cytearc/agent/ingest/loom.py",
    "cytearc/ann.py",
    "cytearc/assay.py",
    "cytearc/bio_data.py",
    "cytearc/chunked.py",
    "cytearc/clustering/feature_graph.py",
    "cytearc/cytebase.py",
    "cytearc/dendrogram.py",
    "cytearc/downloader.py",
    "cytearc/doublet_utils.py",
    "cytearc/feat_utils.py",
    "cytearc/features/lowess.py",
    "cytearc/features/markers/batching.py",
    "cytearc/harmony.py",
    "cytearc/harmony/__init__.py",
    "cytearc/harmony/api.py",
    "cytearc/harmony/models.py",
    "cytearc/harmony/optimizer.py",
    "cytearc/genomics/__init__.py",
    "cytearc/genomics/gff.py",
    "cytearc/genomics/intervals.py",
    "cytearc/genomics/melding.py",
    "cytearc/genomics/reference.py",
    "cytearc/graph/build.py",
    "cytearc/knn_utils.py",
    "cytearc/lineage.py",
    "cytearc/mapping/coral.py",
    "cytearc/mapping_reference.py",
    "cytearc/mapping_utils.py",
    "cytearc/markers.py",
    "cytearc/markers/__init__.py",
    "cytearc/markers/batching.py",
    "cytearc/markers/rank.py",
    "cytearc/markers/regression.py",
    "cytearc/markers/search.py",
    "cytearc/meld_assay.py",
    "cytearc/merge.py",
    "cytearc/merge/assays.py",
    "cytearc/metadata.py",
    "cytearc/metrics.py",
    "cytearc/neighbors/graph_store.py",
    "cytearc/neighbors/persistence.py",
    "cytearc/neighbors/query.py",
    "cytearc/neighbors/stream.py",
    "cytearc/parallel.py",
    "cytearc/plots.py",
    "cytearc/plots/__init__.py",
    "cytearc/plotting/_legacy.py",
    "cytearc/plotting/_legacy/__init__.py",
    "cytearc/plotting/unified.py",
    "cytearc/readers.py",
    "cytearc/readers/datasets.py",
    "cytearc/readers/loom.py",
    "cytearc/results.py",
    "cytearc/storage/zarr_store.py",
    "cytearc/trajectory/aggregation.py",
    "cytearc/symphony.py",
    "cytearc/umap.py",
    "cytearc/utils.py",
    "cytearc/utils/blocks.py",
    "cytearc/utils/memory.py",
    "cytearc/utils/storage.py",
    "cytearc/utils/system.py",
    "cytearc/utils/windows.py",
    "cytearc/writers/loom.py",
    "cytearc/writers.py",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "cytearc"
_WORKFLOW_SCRIPT = Path(__file__).resolve().with_name("smoke_workflow.py")
_REQUIRED_MODULES = {
    f"cytearc/{path.relative_to(_SOURCE_ROOT).as_posix()}"
    for path in _SOURCE_ROOT.rglob("*.py")
}
_SMOKE_CODE = """
import importlib.util
from pathlib import Path

import cytearc
import cytearc.plotting as plotting
import cytearc.cytebase
import cytearc.embeddings.harmony
import cytearc.features.genomic
import cytearc.features.markers
import cytearc.features.variability
import cytearc.matrix
import cytearc.merge
import cytearc.metadata
import cytearc.readers
import cytearc.writers
from cytearc.datastore.datastore import DataStore
from cytearc.datastore.graph_datastore import GraphDataStore
from cytearc.datastore.mapping_datastore import MappingDatastore
from cytearc.datastore.namespaces import (
    FeaturesAccessor,
    MappingAccessor,
    MarkersAccessor,
    TrajectoryAccessor,
)
from cytearc.cytebase import Repository, connect, list_repositories
from cytearc.embeddings.harmony import Harmony, HarmonyResult, fit_harmony
from cytearc.features import (
    RankMarkerResult,
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
)
from cytearc.matrix import ChunkedArray
from cytearc.merge import DataStoreMerge
from cytearc.metadata import MetaData, MetaDataRowBlock
from cytearc.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adReader,
    SeuratReader,
    inspect_seurat,
)
from cytearc.storage.lineage import ArtifactLineage
from cytearc.trajectory.feature_dynamics import knn_clustering
from cytearc.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adImportResult,
    H5adToZarr,
    SeuratImportResult,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    create_zarr_count_assay,
    create_zarr_dataset,
    create_zarr_obj_array,
    chunked_to_zarr,
    subset_assay_zarr,
    to_h5ad,
    to_mtx,
    write_renorm_subset_to_zarr,
)

assert "site-packages" in Path(cytearc.__file__).as_posix()
assert issubclass(DataStore, MappingDatastore)
assert issubclass(MappingDatastore, GraphDataStore)
for harmony_object in (Harmony, HarmonyResult, fit_harmony):
    assert harmony_object.__module__ == "cytearc.embeddings.harmony"
assert ChunkedArray.__module__ == "cytearc.matrix"
for metadata_class in (MetaData, MetaDataRowBlock):
    assert metadata_class.__module__ == "cytearc.metadata"
assert cytearc.DataStoreMerge is cytearc.merge.DataStoreMerge is DataStoreMerge
assert not hasattr(cytearc, "AssayMerge")
assert not hasattr(cytearc.merge, "AssayMerge")
assert not hasattr(cytearc, "DatasetMerge")
assert not hasattr(cytearc.merge, "DatasetMerge")
assert not hasattr(cytearc, "ZarrMerge")
assert not hasattr(cytearc.merge, "ZarrMerge")
assert not hasattr(cytearc, "LoomReader")
assert not hasattr(cytearc, "LoomToZarr")
assert DataStoreMerge.__module__ == "cytearc.merge"
assert cytearc.CrH5Reader is cytearc.readers.CrH5Reader
assert cytearc.CrToZarr is cytearc.writers.CrToZarr
assert cytearc.cytebase.Repository is Repository
assert cytearc.cytebase.connect is connect
assert cytearc.cytebase.list_repositories is list_repositories
assert cytearc.ArtifactLineage is ArtifactLineage
assert ArtifactLineage.__module__ == "cytearc.storage.lineage"
for feature_function in (
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
):
    assert callable(feature_function)
assert RankMarkerResult.__module__ == "cytearc.features.markers.table"
assert cytearc.features.markers.RankMarkerResult is RankMarkerResult
assert callable(knn_clustering)
for reader_class in (
    CrH5Reader,
    CrDirReader,
    CrReader,
    H5adReader,
    SeuratReader,
    CSVReader,
):
    assert reader_class.__module__ == "cytearc.readers"
for writer_class in (
    CrToZarr,
    H5adToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    CSVtoZarr,
):
    assert writer_class.__module__ == "cytearc.writers"
assert H5adImportResult.__module__ == "cytearc.writers"
assert SeuratImportResult.__module__ == "cytearc.writers"
assert inspect_seurat.__module__ == "cytearc.readers"
for writer_function in (
    create_zarr_dataset,
    create_zarr_obj_array,
    create_zarr_count_assay,
    subset_assay_zarr,
    chunked_to_zarr,
    write_renorm_subset_to_zarr,
    to_h5ad,
    to_mtx,
):
    assert callable(writer_function)
    assert writer_function.__module__ == "cytearc.writers"
for method in (
    MappingAccessor.run,
    MarkersAccessor.search,
    FeaturesAccessor.hvgs,
    TrajectoryAccessor.aggregation,
    TrajectoryAccessor.markers,
):
    assert callable(method)
for method in (
    "_load_unified_layout_data",
    "load_metric_lisi",
    "load_unified_graph",
    "metric_lisi",
    "run_unified_tsne",
    "run_unified_umap",
):
    assert not hasattr(DataStore, method)
assert not hasattr(plotting, "unified_embedding")
for name in (
    "cytearc._types",
    "cytearc.bio_data",
    "cytearc.chunked",
    "cytearc.downloader",
    "cytearc.doublet_utils",
    "cytearc.feat_utils",
    "cytearc.harmony",
    "cytearc.genomics",
    "cytearc.knn_utils",
    "cytearc.lineage",
    "cytearc.mapping.coral",
    "cytearc.mapping_reference",
    "cytearc.mapping_utils",
    "cytearc.markers",
    "cytearc.meld_assay",
    "cytearc.plotting.unified",
    "cytearc.readers.loom",
    "cytearc.symphony",
    "cytearc.writers.loom",
):
    assert importlib.util.find_spec(name) is None, name
for name in (
    "cytearc.embeddings.harmony",
    "cytearc.features.genomic",
    "cytearc.features.markers",
    "cytearc.matrix",
    "cytearc.metadata",
    "cytearc.metrics",
):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.submodule_search_locations is not None, name
"""


# CyteArc is pure Python. Its one wheel installs on every platform, so it must
# carry no native executable or library and nothing outside the import root.
_PURE_TAG = "py3-none-any"
_MACH_O_MAGIC = frozenset(
    {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        # Universal binaries, in both byte orders and both offset widths.
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }
)


def _native_format(data: bytes) -> str | None:
    """Name the executable format whose magic bytes start ``data``, if any."""
    if data.startswith(b"\x7fELF"):
        return "ELF"
    if data[:4] in _MACH_O_MAGIC:
        return "Mach-O"
    # A DOS header names the offset of the PE signature at byte 0x3C, so text
    # that merely starts with "MZ" is not mistaken for a Windows binary.
    if data.startswith(b"MZ") and len(data) >= 0x40:
        offset = int.from_bytes(data[0x3C:0x40], "little")
        if data[offset : offset + 4] == b"PE\0\0":
            return "PE"
    return None


def _wheel_tag_problems(wheel: Path, archive: ZipFile) -> list[str]:
    problems: list[str] = []
    file_tag = "-".join(wheel.name.removesuffix(".whl").split("-")[-3:])
    if file_tag != _PURE_TAG:
        problems.append(f"file name tag {file_tag!r}, expected {_PURE_TAG!r}")
    wheel_files = [
        name
        for name in archive.namelist()
        if len(PurePosixPath(name).parts) == 2
        and PurePosixPath(name).parts[0].endswith(".dist-info")
        and PurePosixPath(name).name == "WHEEL"
    ]
    if len(wheel_files) != 1:
        problems.append("the wheel must contain exactly one .dist-info/WHEEL file")
        return problems
    metadata = Parser().parsestr(archive.read(wheel_files[0]).decode("utf-8"))
    tags = [str(tag).strip() for tag in metadata.get_all("Tag") or []]
    if tags != [_PURE_TAG]:
        problems.append(f"WHEEL tags {tags}, expected [{_PURE_TAG!r}]")
    purelib = str(metadata.get("Root-Is-Purelib", "")).strip().lower()
    if purelib != "true":
        problems.append("WHEEL must declare Root-Is-Purelib: true")
    return problems


def validate_wheel_contents(wheel: Path) -> None:
    """Check that ``wheel`` is the complete pure-Python CyteArc wheel."""
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        problems = _wheel_tag_problems(wheel, archive)
        for info in archive.infolist():
            if info.is_dir():
                continue
            if PurePosixPath(info.filename).parts[0].endswith(".data"):
                problems.append(
                    f"{info.filename} is install-time data outside the import root"
                )
            with archive.open(info) as member:
                native = _native_format(member.read())
            if native is not None:
                problems.append(f"{info.filename} is a native {native} binary")
    retired = sorted(_RETIRED_MODULES.intersection(names))
    missing = sorted(_REQUIRED_MODULES.difference(names))
    if retired:
        problems.append(f"retired modules are present: {retired}")
    if missing:
        problems.append(f"required modules are missing: {missing}")
    if problems:
        raise RuntimeError(
            "\n".join(
                ["Wheel contents violate the pure-Python contract:"]
                + [f"- {problem}" for problem in problems]
            )
        )


def installs_tsne_extra() -> bool:
    """Whether the smoke installs the tsne extra on this platform.

    sgtsnepi publishes Linux x86_64 wheels for every supported Python, and the
    test and docs extras install it there, so the smoke expects t-SNE on Linux
    x86_64 and the installation guidance everywhere else.
    """
    return sys.platform == "linux" and platform.machine() == "x86_64"


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def smoke_installed_wheel(wheel: Path) -> None:
    """Install ``wheel`` into a clean environment, then import and use it.

    The environment holds only the wheel, its dependencies, and on Linux
    x86_64 the ``tsne`` extra. Its interpreter runs in isolated mode from a
    scratch directory, so neither ``PYTHONPATH`` nor a source checkout can
    stand in for the installed package.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    with_tsne = installs_tsne_extra()
    requirement = f"cytearc[tsne] @ {wheel.as_uri()}" if with_tsne else str(wheel)
    env = os.environ.copy()
    env["HNSWLIB_NO_NATIVE"] = "1"
    with tempfile.TemporaryDirectory(
        prefix="cytearc-wheel-smoke-", ignore_cleanup_errors=True
    ) as temp_dir:
        root = Path(temp_dir)
        environment = root / "environment"
        subprocess.run(
            ["uv", "venv", "--python", python_version, str(environment)],
            cwd=root,
            env=env,
            check=True,
        )
        python = _environment_python(environment)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), requirement],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", _SMOKE_CODE],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                str(_WORKFLOW_SCRIPT),
                "with-tsne" if with_tsne else "without-tsne",
            ],
            cwd=root,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    validate_wheel_contents(wheel)
    smoke_installed_wheel(wheel)
    print(f"Wheel smoke passed: {wheel.name}")


if __name__ == "__main__":
    main()
