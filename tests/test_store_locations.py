"""Local store locations: drive paths, home directories, and relative paths."""

import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from cytearc import DataStore
from cytearc.storage.stores import _location_identity, locations_overlap, make_store
from cytearc.writers import SparseToZarr


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
