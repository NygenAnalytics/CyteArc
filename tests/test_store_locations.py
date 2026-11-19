"""Local store locations: drive paths, home directories, and relative paths."""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import LocalStore, WrapperStore

from cytearc import DataStore, mount_datastore
from cytearc.composite import create_composite
from cytearc.storage.refs import ArtifactRef
from cytearc.storage.stores import (
    MATRIX_SOURCE_ATTR,
    _location_identity,
    locations_overlap,
    make_store,
)
from cytearc.tools.repack_zarr import repack_store
from cytearc.writers import SparseToZarr
from tests.storage_helpers import object_store_copy, serve_uris


def test_a_windows_drive_path_is_a_file_path_not_a_uri():
    # urlsplit reads "C:" as the scheme "c" of a URI.
    for location in (r"C:\data\a.zarr", "C:/data/a.zarr", "d:relative.zarr"):
        kind, identity = _location_identity(location)
        assert kind == "file"
        assert identity == str(Path(location).resolve())
    # A drive path and the same path below it overlap as file paths do.
    assert locations_overlap("C:/data/a.zarr", "C:/data/a.zarr/sub")
    assert not locations_overlap("C:/data/a.zarr", "s3://data/a.zarr")


def test_uris_keep_their_scheme_and_file_uris_name_paths(tmp_path):
    assert _location_identity("S3://bucket/a/../b.zarr") == (
        "uri",
        "s3://bucket/b.zarr",
    )
    assert _location_identity(f"file://{tmp_path}/x.zarr") == (
        "file",
        str((tmp_path / "x.zarr").resolve()),
    )


def test_make_store_expands_home_and_absolutizes_local_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    assert make_store("~/x.zarr") == str(home / "x.zarr")
    assert make_store("file://~/y.zarr") == str(home / "y.zarr")
    assert make_store("rel.zarr") == str(work / "rel.zarr")
    assert make_store(f"file://{work}/../abs.zarr") == str(tmp_path / "abs.zarr")


def _write_store(location: str) -> None:
    counts = np.array([[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]], dtype=np.uint32)
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=location,
        cell_ids=[f"c{index}" for index in range(4)],
        feature_ids=["g0", "g1", "g2"],
        assay_name="RNA",
        nthreads=1,
        shuffle_cells=False,
    ).dump(batch_size=4)


def test_a_store_written_to_a_home_path_lands_in_the_home_directory(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    _write_store("~/tilde.zarr")

    assert (home / "tilde.zarr" / "zarr.json").is_file()
    # No directory literally named "~" appears in the working directory.
    assert not (work / "~").exists()
    ds = DataStore("~/tilde.zarr", default_assay="RNA", nthreads=1)
    assert ds.cells.fetch_all("ids").tolist() == ["c0", "c1", "c2", "c3"]


def test_a_store_opened_by_a_relative_path_survives_a_directory_change(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_store("relative.zarr")
    ds = DataStore("relative.zarr", default_assay="RNA", nthreads=1)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.chdir(elsewhere)

    assert ds.cells.fetch_all("ids").tolist() == ["c0", "c1", "c2", "c3"]
    np.testing.assert_array_equal(
        ds.RNA.rawData.compute(),
        [[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]],
    )
    reopened = DataStore(str(tmp_path / "relative.zarr"), nthreads=1)
    assert reopened.cells.N == 4


def test_a_composite_of_home_paths_names_the_home_directory(tmp_path, monkeypatch):
    from cytearc.composite import create_composite

    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    _write_store("~/source.zarr")
    source = DataStore("~/source.zarr", default_assay="RNA", nthreads=1)

    create_composite({"a": source}, at="~/joint.zarr", features="union")

    assert (home / "joint.zarr" / "zarr.json").is_file()
    assert not (work / "~").exists()
    # Opened again, the composite reads its counts from the source location
    # that it recorded, in the home directory.
    joint = DataStore("~/joint.zarr", nthreads=1)
    rows = [
        int(str(identity).removeprefix("a__c"))
        for identity in joint.cells.fetch_all("ids")
    ]
    np.testing.assert_array_equal(
        joint.RNA.rawData.compute(),
        np.array([[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]])[rows],
    )


def test_a_mount_of_a_home_path_records_the_home_directory(tmp_path, monkeypatch):
    from cytearc import mount_datastore

    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    _write_store("~/source.zarr")
    DataStore("~/source.zarr", default_assay="RNA", nthreads=1)

    mount_datastore("~/source.zarr", at="~/mount.zarr", nthreads=1)

    assert (home / "mount.zarr" / "zarr.json").is_file()
    assert not (work / "~").exists()
    # Opened again, the mount reads its counts from the source location that
    # it recorded, in the home directory.
    mounted = DataStore("~/mount.zarr", default_assay="RNA", nthreads=1)
    np.testing.assert_array_equal(
        mounted.RNA.rawData.compute(),
        [[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]],
    )


_COUNTS = np.array([[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]])


def _composite_counts(joint: DataStore) -> np.ndarray:
    """Return the source counts in the order of the composite's cells."""
    rows = [
        int(str(identity).removeprefix("a__c"))
        for identity in joint.cells.fetch_all("ids")
    ]
    return _COUNTS[rows]


def _build_folder(
    sources: Path, derived: Path, **options: bool
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    """Write a source in ``sources`` and a mount and a composite of it in ``derived``.

    ``options``, such as ``relative_sources``, go to both the mount and the
    composite. Returns a cell selection saved in the source, which the mount
    reuses, a feature selection saved in the mount, and a cell selection saved
    in the composite.
    """
    sources.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    _write_store(str(sources / "source.zarr"))
    source = DataStore(
        str(sources / "source.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    in_source = source.snapshot_cell_selection()
    mounted = mount_datastore(
        str(sources / "source.zarr"),
        at=str(derived / "mount.zarr"),
        min_features_per_cell=-1,
        nthreads=1,
        **options,
    )
    assert mounted.snapshot_cell_selection() == in_source
    in_mount = mounted.features.detected(in_source, min_cells=1)
    joint = create_composite(
        {"a": source},
        at=derived / "joint.zarr",
        features="union",
        nthreads=1,
        **options,
    )
    in_joint = joint.snapshot_cell_selection()
    for store in (source, mounted, joint):
        store.z.store.close()
    return in_source, in_mount, in_joint


def _recorded(folder: Path) -> tuple[str, str]:
    """Return the source locations that the mount and the composite record."""
    mount = zarr.open_group(str(folder / "mount.zarr"), mode="r")
    joint = zarr.open_group(str(folder / "joint.zarr"), mode="r")
    return (
        mount.attrs[MATRIX_SOURCE_ATTR]["location"],
        joint.attrs["composite"]["sources"][0]["location"],
    )


def test_sources_are_recorded_by_absolute_paths_with_relative_sources_false(
    tmp_path,
):
    folder = tmp_path / "project"
    _build_folder(folder, folder, relative_sources=False)
    # As earlier releases record them by default, which every release can
    # read.
    assert _recorded(folder) == (
        str(folder / "source.zarr"),
        str((folder / "source.zarr").resolve()),
    )
    np.testing.assert_array_equal(
        DataStore(str(folder / "mount.zarr"), nthreads=1).RNA.rawData.compute(),
        _COUNTS,
    )
    reopened = DataStore(str(folder / "joint.zarr"), nthreads=1)
    np.testing.assert_array_equal(
        reopened.RNA.rawData.compute(), _composite_counts(reopened)
    )
    reopened.z.store.close()

    # An absolute record keeps naming the folder the stores were written in.
    moved = tmp_path / "moved"
    folder.rename(moved)
    looked_for = re.escape(str((folder / "source.zarr").resolve()))
    for name in ("mount.zarr", "joint.zarr"):
        with pytest.raises(FileNotFoundError, match=looked_for):
            DataStore(str(moved / name), nthreads=1)


def test_every_entry_point_records_nearby_sources_by_relative_paths_by_default(
    tmp_path,
):
    import inspect

    from cytearc.storage.stores import create_matrix_source

    for function in (mount_datastore, create_composite, create_matrix_source):
        parameter = inspect.signature(function).parameters["relative_sources"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function
        assert parameter.default is True, function
    # mount_datastore passes its own setting on, and the folder tests below
    # record its default and the composite's; called alone, the matrix
    # source records a source beside it by a relative path too.
    _write_store(str(tmp_path / "source.zarr"))
    # A first writable open prepares the store to be mounted.
    DataStore(str(tmp_path / "source.zarr"), min_features_per_cell=-1, nthreads=1)
    target = create_matrix_source(
        str(tmp_path / "source.zarr"),
        str(tmp_path / "mount.zarr"),
        required_transposes=frozenset(),
    )
    assert target.attrs[MATRIX_SOURCE_ATTR]["location"] == "source.zarr"


def test_relative_sources_must_be_a_boolean(tmp_path):
    _write_store(str(tmp_path / "source.zarr"))
    source = DataStore(
        str(tmp_path / "source.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    with pytest.raises(TypeError, match="^relative_sources must be a boolean$"):
        mount_datastore(
            str(tmp_path / "source.zarr"),
            at=str(tmp_path / "mount.zarr"),
            relative_sources="yes",
        )
    with pytest.raises(TypeError, match="^relative_sources must be a boolean$"):
        create_composite(
            {"a": source},
            at=tmp_path / "joint.zarr",
            features="union",
            relative_sources=1,
        )
    assert not (tmp_path / "mount.zarr").exists()
    assert not (tmp_path / "joint.zarr").exists()


def test_a_moved_folder_of_a_source_a_mount_and_a_composite_still_opens(tmp_path):
    folder = tmp_path / "project"
    in_source, in_mount, in_joint = _build_folder(folder, folder)
    # By default, a source beside the stores built on it is recorded relative
    # to them.
    assert _recorded(folder) == ("source.zarr", "source.zarr")

    moved = tmp_path / "archive" / "2026" / "project"
    moved.parent.mkdir(parents=True)
    folder.rename(moved)

    mounted = DataStore(str(moved / "mount.zarr"), nthreads=1)
    np.testing.assert_array_equal(mounted.RNA.rawData.compute(), _COUNTS)
    # The mount reads its own artifacts and, through its source, the source's.
    assert mounted.artifacts.inspect(in_mount).complete
    assert mounted.artifacts.inspect(in_source).complete
    joint = DataStore(str(moved / "joint.zarr"), nthreads=1)
    np.testing.assert_array_equal(joint.RNA.rawData.compute(), _composite_counts(joint))
    assert joint.artifacts.inspect(in_joint).complete
    for store in (mounted, joint):
        store.z.store.close()

    # A repack reads the mount's moved source and checks its own destination
    # against where that source now is.
    with pytest.raises(ValueError, match="overlaps the mounted count owner"):
        repack_store(str(moved / "mount.zarr"), str(moved / "source.zarr" / "copy"))
    repack_store(str(moved / "mount.zarr"), str(tmp_path / "repacked.zarr"), nthreads=1)
    repacked = DataStore(str(tmp_path / "repacked.zarr"), nthreads=1)
    np.testing.assert_array_equal(repacked.RNA.rawData.compute(), _COUNTS)


@pytest.mark.parametrize(
    "options", [{"relative_sources": False}, {}], ids=["absolute", "default"]
)
def test_a_source_moved_alone_is_reported_where_it_was_looked_for(tmp_path, options):
    folder = tmp_path / "project"
    _build_folder(folder, folder, **options)
    (folder / "source.zarr").rename(tmp_path / "source.zarr")

    looked_for = re.escape(str((folder / "source.zarr").resolve()))
    with pytest.raises(FileNotFoundError, match=looked_for):
        DataStore(str(folder / "mount.zarr"), nthreads=1)
    with pytest.raises(FileNotFoundError, match=looked_for):
        DataStore(str(folder / "joint.zarr"), nthreads=1)


def test_a_source_kept_elsewhere_stays_absolute_when_its_stores_move(tmp_path):
    _build_folder(tmp_path / "data", tmp_path / "analyses")
    absolute = str((tmp_path / "data" / "source.zarr").resolve())
    assert _recorded(tmp_path / "analyses") == (absolute, absolute)

    moved = tmp_path / "archive" / "analyses"
    moved.parent.mkdir()
    (tmp_path / "analyses").rename(moved)
    np.testing.assert_array_equal(
        DataStore(str(moved / "mount.zarr"), nthreads=1).RNA.rawData.compute(),
        _COUNTS,
    )
    joint = DataStore(str(moved / "joint.zarr"), nthreads=1)
    np.testing.assert_array_equal(joint.RNA.rawData.compute(), _composite_counts(joint))


def test_a_folder_served_from_a_bucket_finds_its_sources_there(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    in_source, _, in_joint = _build_folder(folder, folder)
    bucket = "s3://bucket/published/project"
    serve_uris(
        monkeypatch,
        {
            f"{bucket}/{name}": (object_store_copy(folder / name), None)
            for name in ("source.zarr", "mount.zarr", "joint.zarr")
        },
    )
    # Nothing is left at the paths where the stores were written.
    folder.rename(tmp_path / "removed")

    mounted = DataStore(f"{bucket}/mount.zarr", zarr_mode="r", nthreads=1)
    np.testing.assert_array_equal(mounted.RNA.rawData.compute(), _COUNTS)
    assert mounted.artifacts.inspect(in_source).complete
    joint = DataStore(f"{bucket}/joint.zarr", zarr_mode="r", nthreads=1)
    np.testing.assert_array_equal(joint.RNA.rawData.compute(), _composite_counts(joint))
    assert joint.artifacts.inspect(in_joint).complete


def test_a_relative_source_needs_the_location_of_its_store(tmp_path):
    folder = tmp_path / "project"
    _build_folder(folder, folder)

    # An object store does not say where it is, so a relative source cannot be found.
    for name, subject in (
        ("mount.zarr", "The source of this mount"),
        ("joint.zarr", "Composite source 'a'"),
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(f"{subject} is recorded at 'source.zarr', a path relative"),
        ):
            DataStore(object_store_copy(folder / name), zarr_mode="r", nthreads=1)
    # A local store says where it is, also inside a store that wraps it.
    mounted = DataStore(LocalStore(folder / "mount.zarr"), nthreads=1)
    np.testing.assert_array_equal(mounted.RNA.rawData.compute(), _COUNTS)
    wrapped = DataStore(WrapperStore(LocalStore(folder / "mount.zarr")), nthreads=1)
    np.testing.assert_array_equal(wrapped.RNA.rawData.compute(), _COUNTS)
    joint = DataStore(WrapperStore(LocalStore(folder / "joint.zarr")), nthreads=1)
    np.testing.assert_array_equal(joint.RNA.rawData.compute(), _composite_counts(joint))


@pytest.mark.parametrize(
    ("source", "derived", "recorded"),
    [
        ("/data/project/a.zarr", "/data/project/joint.zarr", "a.zarr"),
        ("/data/project/in/a.zarr", "/data/project/joint.zarr/", "in/a.zarr"),
        ("/data/a.zarr", "/data/project/joint.zarr", "/data/a.zarr"),
        ("/data/projects/a.zarr", "/data/project/joint.zarr", "/data/projects/a.zarr"),
        ("/data/project/a.zarr", None, "/data/project/a.zarr"),
        ("s3://bucket/project/a.zarr", "s3://bucket/project/m.zarr", "a.zarr"),
        ("S3://bucket/project/a.zarr", "s3://bucket/project/m.zarr", "a.zarr"),
        ("s3://bucket/a.zarr", "s3://bucket/project/m.zarr", "s3://bucket/a.zarr"),
        ("s3://other/project/a.zarr", "s3://bucket/project/m.zarr", None),
        ("gs://bucket/project/a.zarr", "s3://bucket/project/m.zarr", None),
        ("s3://bucket/project/a.zarr?v=1", "s3://bucket/project/m.zarr", None),
        ("hf://buckets/o/n/project/a.zarr", "/data/project/joint.zarr", None),
        ("/data/project/a.zarr", "s3://bucket/data/project/m.zarr", None),
    ],
)
def test_a_store_records_only_sources_beside_or_below_it_by_relative_paths(
    source, derived, recorded
):
    from cytearc.storage.stores import record_source_location, resolve_source_location

    written = record_source_location(source, derived)
    assert written == (source if recorded is None else recorded)
    if derived is not None:
        # What is recorded names the source again from where the store is.
        assert resolve_source_location(written, derived, subject="source") == (
            source if "://" not in source else source.replace("S3://", "s3://")
        )


def test_a_relative_source_is_found_from_where_its_store_is_opened(tmp_path):
    from cytearc.storage.stores import resolve_source_location

    def resolve(recorded, derived):
        return resolve_source_location(recorded, derived, subject="Source 'a'")

    assert resolve("a.zarr", str(tmp_path / "joint.zarr")) == str(
        tmp_path.resolve() / "a.zarr"
    )
    assert resolve("in/./a.zarr", "s3://bucket/p/m.zarr/") == "s3://bucket/p/in/a.zarr"
    assert resolve("../a.zarr", "hf://buckets/o/n/p/m.zarr") == (
        "hf://buckets/o/n/a.zarr"
    )
    # Absolute paths and URLs, as earlier releases recorded, name the source.
    for location in ("/data/a.zarr", r"C:\data\a.zarr", "s3://bucket/a.zarr"):
        assert resolve(location, "s3://bucket/p/m.zarr") == location
    with pytest.raises(ValueError, match="climbs above the root of s3://bucket/m"):
        resolve("../a.zarr", "s3://bucket/m.zarr")
    with pytest.raises(ValueError, match="store object whose location is unknown"):
        resolve("a.zarr", None)


def test_a_mount_in_the_bucket_of_its_source_records_it_relative_by_default(
    tmp_path, monkeypatch
):
    from obstore.store import MemoryStore as ObjectMemoryStore
    from zarr.storage import ObjectStore

    _write_store(str(tmp_path / "source.zarr"))
    DataStore(
        str(tmp_path / "source.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    targets = {
        f"s3://bucket/project/{name}.zarr": ObjectStore(ObjectMemoryStore())
        for name in ("relative", "absolute", "other")
    }
    serve_uris(
        monkeypatch,
        {
            "s3://bucket/project/source.zarr": (
                object_store_copy(tmp_path / "source.zarr"),
                None,
            ),
            "s3://other/source.zarr": (
                object_store_copy(tmp_path / "source.zarr"),
                None,
            ),
            **{uri: (store, None) for uri, store in targets.items()},
        },
    )

    for source, target, options, recorded in (
        (
            "s3://bucket/project/source.zarr",
            "s3://bucket/project/relative.zarr",
            {},
            "source.zarr",
        ),
        (
            "s3://bucket/project/source.zarr",
            "s3://bucket/project/absolute.zarr",
            {"relative_sources": False},
            None,
        ),
        # A source in another bucket keeps its URL.
        ("s3://other/source.zarr", "s3://bucket/project/other.zarr", {}, None),
    ):
        mount_datastore(
            source, at=target, min_features_per_cell=-1, nthreads=1, **options
        )
        root = zarr.open_group(store=targets[target], mode="r")
        assert root.attrs[MATRIX_SOURCE_ATTR]["location"] == (
            source if recorded is None else recorded
        )
        reopened = DataStore(target, nthreads=1)
        np.testing.assert_array_equal(reopened.RNA.rawData.compute(), _COUNTS)
