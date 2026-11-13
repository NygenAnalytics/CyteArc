import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from huggingface_hub import BucketFile

from tests.fixtures_hub import bucket_file, bucket_folder
from tests.network_helpers import hub_or_skip


@pytest.fixture(autouse=True)
def _offline_unless_live(request):
    """Keep every test except the explicit live checks off the network."""
    if request.node.get_closest_marker("integration") is None:
        request.getfixturevalue("hub_offline")


def _write_tar(path: Path, members: list[tuple[str, str, bytes | str | None]]) -> Path:
    """Write a gzip tar archive from ``(name, kind, payload)`` member specs.

    ``kind`` is ``file`` (payload is the content), ``symlink`` or ``hardlink``
    (payload is the link name), ``fifo``, or ``chardev``.
    """
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, payload in members:
            member = tarfile.TarInfo(name)
            if kind == "file":
                assert isinstance(payload, bytes)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                continue
            member.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
                "chardev": tarfile.CHRTYPE,
            }[kind]
            if payload is not None:
                member.linkname = str(payload)
            archive.addfile(member)
    return path


def _tree(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


@pytest.mark.parametrize("member_name", ["notes.txt", "other.zarr/zarr.json"])
def test_archive_cannot_replace_unrelated_siblings(tmp_path, member_name):
    from cytearc.datasets import _extract_and_replace

    notes = tmp_path / "notes.txt"
    notes.write_text("keep this")
    output = tmp_path / "data.zarr"
    output.mkdir()
    (output / "zarr.json").write_text("original")
    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        for name in ["data.zarr/zarr.json", member_name]:
            member = tarfile.TarInfo(name)
            member.size = 7
            archive.addfile(member, io.BytesIO(b"replace"))
    destination = tmp_path / "data.zarr.tar.gz"
    with pytest.raises(ValueError, match="must contain only the directory"):
        _extract_and_replace(staged, destination)
    assert notes.read_text() == "keep this"
    assert (output / "zarr.json").read_text() == "original"
    assert not destination.exists()


def test_archive_replaces_only_its_named_output_directory(tmp_path):
    from cytearc.datasets import _extract_and_replace

    output = tmp_path / "data.zarr"
    output.mkdir()
    (output / "old").write_text("old data")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep this")
    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        member = tarfile.TarInfo("data.zarr/zarr.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    destination = tmp_path / "data.zarr.tar.gz"
    _extract_and_replace(staged, destination)
    assert (output / "zarr.json").read_text() == "{}"
    assert not (output / "old").exists()
    assert notes.read_text() == "keep this"
    assert destination.is_file()


def test_archive_output_directory_cannot_be_a_symlink(tmp_path):
    from cytearc.datasets import _extract_and_replace

    staged = tmp_path / "download"
    with tarfile.open(staged, "w:gz") as archive:
        member = tarfile.TarInfo("data.zarr")
        member.type = tarfile.SYMTYPE
        member.linkname = "."
        archive.addfile(member)
    with pytest.raises(ValueError, match="must contain only the directory"):
        _extract_and_replace(staged, tmp_path / "data.zarr.tar.gz")
    assert not (tmp_path / "data.zarr").exists()


def test_list_repositories_is_sorted_and_anonymous(monkeypatch):
    from cytearc import datasets

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("cytearc_docs"),
            bucket_file("README.md", 3),
            bucket_folder("cellxgene"),
        ]

    monkeypatch.setattr(datasets, "list_bucket_tree", fake_list_bucket_tree)

    assert datasets.list_repositories() == ["cellxgene", "cytearc_docs"]
    assert calls == [("Nygen/cytebase", None, False, False)]


def test_connect_returns_repository(monkeypatch):
    from cytearc import datasets

    monkeypatch.setattr(
        datasets,
        "list_repositories",
        lambda: ["cellxgene", "cytearc_docs"],
    )

    repository = datasets.connect("cytearc_docs")

    assert repository == datasets.Repository("cytearc_docs")


def test_connect_rejects_unknown_repository(monkeypatch):
    from cytearc import datasets

    monkeypatch.setattr(datasets, "list_repositories", lambda: ["cellxgene", "docs"])

    with pytest.raises(KeyError) as raised:
        datasets.connect("missing")
    assert raised.value.args[0] == (
        "'missing' is not a CyteBase repository. Available repositories:\n"
        "cellxgene\ndocs"
    )


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "/outside", r"..\outside", "a/b"],
)
def test_connect_rejects_invalid_repository_name(monkeypatch, name):
    from cytearc import datasets

    def unexpected_listing():
        raise AssertionError("an invalid name must be rejected before listing")

    monkeypatch.setattr(datasets, "list_repositories", unexpected_listing)
    with pytest.raises(ValueError, match="Invalid repository name"):
        datasets.connect(name)
    with pytest.raises(ValueError, match="Invalid repository name"):
        datasets.Repository(name)


def test_repository_lists_datasets(monkeypatch):
    from cytearc import datasets

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("cytearc_docs/zeta"),
            bucket_file("cytearc_docs/index.json", 2),
            bucket_folder("cytearc_docs/alpha"),
            # Object stores match prefixes as strings, so a sibling repository
            # and deeper folders can appear in a non-recursive listing.
            bucket_folder("cytearc_docs_v2"),
            bucket_folder("cytearc_docs_v2/beta"),
            bucket_folder("cytearc_docs/alpha/raw"),
        ]

    monkeypatch.setattr(datasets, "list_bucket_tree", fake_list_bucket_tree)

    assert datasets.Repository("cytearc_docs").list_datasets() == ["alpha", "zeta"]
    assert calls == [("Nygen/cytebase", "cytearc_docs", False, False)]


def test_repository_file_listing_is_scoped_and_sorted(monkeypatch):
    from cytearc import datasets

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_file("cytearc_docs/alpha/matrix.mtx.gz", 4),
            bucket_file("cytearc_docs/alphabet/other.bin", 3),
            bucket_file("cytearc_docs/alpha/barcodes.tsv.gz", 2),
        ]

    monkeypatch.setattr(datasets, "list_bucket_tree", fake_list_bucket_tree)

    files = datasets._bucket_files("cytearc_docs", "alpha", recursive=False)

    assert [file.path for file in files] == [
        "cytearc_docs/alpha/barcodes.tsv.gz",
        "cytearc_docs/alpha/matrix.mtx.gz",
    ]
    assert calls == [("Nygen/cytebase", "cytearc_docs/alpha", False, False)]
    files = datasets._bucket_files("cytearc_docs", recursive=True)
    assert [file.path for file in files] == [
        "cytearc_docs/alpha/barcodes.tsv.gz",
        "cytearc_docs/alpha/matrix.mtx.gz",
        "cytearc_docs/alphabet/other.bin",
    ]
    assert calls[-1] == ("Nygen/cytebase", "cytearc_docs", True, False)


def test_repository_downloads_file_anonymously(monkeypatch, tmp_path):
    from cytearc import datasets

    file = bucket_file("cytearc_docs/alpha/data.bin", 7)
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    calls = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        calls.append((bucket_id, files, raise_on_missing_files, token))
        files[0][1].write_bytes(b"payload")

    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)

    downloads = tmp_path / "downloads"
    downloaded = datasets.Repository("cytearc_docs").download(
        "alpha/data.bin",
        downloads,
    )

    destination = downloads / "alpha" / "data.bin"
    assert downloaded == [destination]
    assert destination.read_bytes() == b"payload"
    assert calls[0][0] == "Nygen/cytebase"
    assert calls[0][1][0][0] is file
    assert calls[0][2:] == (False, False)
    # The staging directory is gone and nothing else was written.
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_repository_download_respects_progress_setting(
    monkeypatch, tmp_path, progress, fails
):
    import importlib

    from cytearc import configure_output, datasets
    from cytearc.utils.logging import progress_enabled

    hf_progress = importlib.import_module("huggingface_hub.utils.tqdm")
    monkeypatch.setattr(hf_progress, "HF_HUB_DISABLE_PROGRESS_BARS", None)
    monkeypatch.setattr(hf_progress, "progress_bar_states", {})
    file = bucket_file("cytearc_docs/alpha/data.bin", 7)
    monkeypatch.setattr(
        datasets, "_bucket_files", lambda repository, path, recursive: [file]
    )
    output = io.StringIO()
    group = "huggingface_hub.download_bucket_files"

    def download(bucket_id, files, *, token):
        for name in (group, f"{group}.transfer"):
            with hf_progress.tqdm(
                total=7, file=output, disable=False, name=name
            ) as bar:
                assert bar.disable is (not progress)
                bar.update(7)
        if fails:
            raise RuntimeError("Transfer failed")
        files[0][1].write_bytes(b"payload")

    monkeypatch.setattr(datasets, "download_bucket_files", download)
    original_progress = progress_enabled()
    configure_output(progress=progress)
    try:
        repository = datasets.Repository("cytearc_docs")
        if fails:
            with pytest.raises(RuntimeError, match="Transfer failed"):
                repository.download("alpha/data.bin", tmp_path / "downloads")
        else:
            paths = repository.download("alpha/data.bin", tmp_path / "downloads")
            assert paths[0].read_bytes() == b"payload"
        assert bool(output.getvalue()) is progress
        with hf_progress.tqdm(
            total=1, file=io.StringIO(), disable=False, name=group
        ) as bar:
            assert not bar.disable
    finally:
        configure_output(progress=original_progress)


def test_download_dataset_excludes_zarr_archive_by_default(monkeypatch, tmp_path):
    from cytearc import datasets

    source = bucket_file("cytearc_docs/alpha/data.h5", 6)
    archive = bucket_file("cytearc_docs/alpha/data.zarr.tar.gz", 12)
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [archive, source],
    )
    requested = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        requested.extend(file.path for file, _ in files)
        for _, destination in files:
            destination.write_bytes(b"source")

    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)

    dataset_path = datasets.Repository("cytearc_docs").download_dataset(
        "alpha",
        tmp_path,
    )

    assert dataset_path == tmp_path / "alpha"
    assert requested == ["cytearc_docs/alpha/data.h5"]
    assert (dataset_path / "data.h5").read_bytes() == b"source"
    assert not (dataset_path / "data.zarr.tar.gz").exists()


def test_download_dataset_copies_local_catalog_zarr(monkeypatch, tmp_path):
    from cytearc import datasets

    local_root = tmp_path / "catalog"
    store = local_root / "alpha" / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(datasets._LOCAL_CATALOG_ENV, str(local_root))

    def fail_bucket(*_args, **_kwargs):
        raise AssertionError("local catalog should not list the remote bucket")

    monkeypatch.setattr(datasets, "_bucket_files", fail_bucket)

    destination = tmp_path / "downloads"
    dataset_path = datasets.Repository("cytearc_docs").download_dataset(
        "alpha",
        destination,
        zarr=True,
    )

    copied = dataset_path / "data.zarr" / "zarr.json"
    assert dataset_path == destination / "alpha"
    assert copied.read_text(encoding="utf-8") == "{}"
    assert (store / "zarr.json").read_text(encoding="utf-8") == "{}"


def test_local_catalog_copy_replaces_only_the_zarr_store(monkeypatch, tmp_path):
    from cytearc import datasets

    store = tmp_path / "catalog" / "alpha" / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(datasets._LOCAL_CATALOG_ENV, str(tmp_path / "catalog"))

    def fail_bucket(*_args, **_kwargs):
        raise AssertionError("local catalog should not list the remote bucket")

    monkeypatch.setattr(datasets, "_bucket_files", fail_bucket)
    dataset = tmp_path / "downloads" / "alpha"
    (dataset / "data.zarr").mkdir(parents=True)
    (dataset / "data.zarr" / "stale.json").write_text("old", encoding="utf-8")
    (dataset / "data.h5").write_bytes(b"raw counts")

    dataset_path = datasets.Repository("cytearc_docs").download_dataset(
        "alpha",
        tmp_path / "downloads",
        zarr=True,
        overwrite=True,
    )

    assert dataset_path == dataset
    assert sorted(path.name for path in dataset.iterdir()) == [
        ".cytebase-sources.json",
        "data.h5",
        "data.zarr",
    ]
    assert (dataset / "data.h5").read_bytes() == b"raw counts"
    assert sorted(path.name for path in (dataset / "data.zarr").iterdir()) == [
        "zarr.json"
    ]


def test_download_dataset_selects_and_extracts_zarr_archive(monkeypatch, tmp_path):
    from cytearc import datasets

    archive_buffer = io.BytesIO()
    payload = b'{"zarr_format": 3, "node_type": "group"}'
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("data.zarr/zarr.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_bytes = archive_buffer.getvalue()
    source = bucket_file("cytearc_docs/alpha/data.h5", 6)
    zarr_archive = bucket_file(
        "cytearc_docs/alpha/data.zarr.tar.gz",
        len(archive_bytes),
    )
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [source, zarr_archive],
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        assert [file.path for file, _ in files] == [
            "cytearc_docs/alpha/data.zarr.tar.gz"
        ]
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)

    dataset_path = datasets.Repository("cytearc_docs").download_dataset(
        "alpha",
        tmp_path,
        zarr=True,
    )

    assert (dataset_path / "data.zarr.tar.gz").read_bytes() == archive_bytes
    assert (dataset_path / "data.zarr" / "zarr.json").read_bytes() == payload


def test_download_dataset_reports_missing_zarr(monkeypatch, tmp_path):
    from cytearc import datasets

    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [
            bucket_file("cytearc_docs/alpha/data.h5", 6)
        ],
    )

    with pytest.raises(
        FileNotFoundError, match="^No Zarr archive is available for 'alpha'$"
    ):
        datasets.Repository("cytearc_docs").download_dataset(
            "alpha",
            tmp_path / "downloads",
            zarr=True,
        )
    assert not (tmp_path / "downloads").exists()


def test_download_dataset_reports_unknown_name(monkeypatch, tmp_path):
    from cytearc import datasets

    monkeypatch.setattr(datasets, "_bucket_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        datasets.Repository,
        "list_datasets",
        lambda self: ["alpha", "beta"],
    )

    with pytest.raises(KeyError) as raised:
        datasets.Repository("cytearc_docs").download_dataset("missing", tmp_path)
    assert raised.value.args[0] == (
        "'missing' is not in repository 'cytearc_docs'. Available datasets:\nalpha\nbeta"
    )


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "../outside", "/outside", r"..\outside", "a//b"],
)
def test_repository_rejects_invalid_remote_paths(path, tmp_path):
    from cytearc import datasets

    with pytest.raises(ValueError, match="Invalid download path"):
        datasets.Repository("cytearc_docs").download(path, tmp_path)


def test_download_preserves_existing_file_when_transfer_fails(monkeypatch, tmp_path):
    from cytearc import datasets

    file = bucket_file("cytearc_docs/alpha/data.bin", 3)
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def fail_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(b"new")
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(datasets, "download_bucket_files", fail_download)

    with pytest.raises(RuntimeError, match="transfer failed"):
        datasets.Repository("cytearc_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"existing"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


def test_download_rejects_size_mismatch_and_preserves_existing(
    monkeypatch,
    tmp_path,
):
    from cytearc import datasets

    file = bucket_file("cytearc_docs/alpha/data.bin", 4)
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def incomplete_download(
        bucket_id,
        files,
        *,
        raise_on_missing_files=False,
        token=None,
    ):
        files[0][1].write_bytes(b"new")

    monkeypatch.setattr(datasets, "download_bucket_files", incomplete_download)

    with pytest.raises(
        OSError,
        match="^Downloaded 3 bytes for 'cytearc_docs/alpha/data.bin', expected 4$",
    ):
        datasets.Repository("cytearc_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"existing"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


def test_download_rejects_unsafe_tar_member(monkeypatch, tmp_path):
    from cytearc import datasets

    archive_buffer = io.BytesIO()
    payload = b"unsafe"
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_bytes = archive_buffer.getvalue()
    file = bucket_file(
        "cytearc_docs/alpha/data.tar.gz",
        len(archive_bytes),
    )
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)

    downloads = tmp_path / "downloads"
    with pytest.raises(ValueError, match="^Unsafe tar archive member: ../outside.txt$"):
        datasets.Repository("cytearc_docs").download("alpha/data.tar.gz", downloads)

    assert not (tmp_path / "outside.txt").exists()
    assert _tree(downloads) == ["alpha"]


def test_download_of_a_directory_skips_folders_and_sibling_prefixes(
    monkeypatch, tmp_path
):
    from cytearc import datasets

    calls = []

    def fake_list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None):
        calls.append((bucket_id, prefix, recursive, token))
        return [
            bucket_folder("cytearc_docs/alpha/raw"),
            bucket_file("cytearc_docs/alpha/raw/matrix.mtx", 3),
            bucket_file("cytearc_docs/alpha/barcodes.tsv", 2),
            # A string prefix match, not a file under alpha/.
            bucket_file("cytearc_docs/alphabet/other.bin", 1),
        ]

    requested = []

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        for file, destination in files:
            requested.append(file.path)
            destination.write_bytes(b"x" * file.size)

    monkeypatch.setattr(datasets, "list_bucket_tree", fake_list_bucket_tree)
    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)
    downloads = tmp_path / "downloads"

    downloaded = datasets.Repository("cytearc_docs").download("alpha", downloads)

    assert calls == [("Nygen/cytebase", "cytearc_docs/alpha", True, False)]
    assert requested == [
        "cytearc_docs/alpha/barcodes.tsv",
        "cytearc_docs/alpha/raw/matrix.mtx",
    ]
    assert downloaded == [
        downloads / "alpha" / "barcodes.tsv",
        downloads / "alpha" / "raw" / "matrix.mtx",
    ]
    assert (downloads / "alpha" / "raw" / "matrix.mtx").read_bytes() == b"xxx"
    assert _tree(downloads) == [
        "alpha",
        "alpha/barcodes.tsv",
        "alpha/raw",
        "alpha/raw/matrix.mtx",
    ]


def test_download_reports_a_path_without_files(monkeypatch, tmp_path):
    from cytearc import datasets

    monkeypatch.setattr(
        datasets,
        "list_bucket_tree",
        lambda *args, **kwargs: [bucket_folder("cytearc_docs/alpha/empty")],
    )

    with pytest.raises(
        FileNotFoundError, match="^No CyteBase files found at cytearc_docs/alpha/empty$"
    ):
        datasets.Repository("cytearc_docs").download("alpha/empty", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_download_rejects_a_listing_that_escapes_the_destination(monkeypatch, tmp_path):
    from cytearc import datasets

    monkeypatch.setattr(
        datasets,
        "list_bucket_tree",
        lambda *args, **kwargs: [bucket_file("cytearc_docs/alpha/../../escape", 1)],
    )
    monkeypatch.setattr(datasets, "download_bucket_files", _unexpected_download)

    with pytest.raises(
        ValueError, match="^Invalid bucket file path: 'alpha/../../escape'$"
    ):
        datasets.Repository("cytearc_docs").download("alpha", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _unexpected_download(*_args, **_kwargs):
    raise AssertionError("nothing may be downloaded")


@pytest.mark.parametrize("path", ["other/alpha/data.bin", "cytearc_docs_v2/data.bin"])
def test_bucket_file_paths_must_stay_inside_their_repository(path):
    from cytearc import datasets

    # Listings are filtered by the requested prefix first, so this guards the
    # helper rather than a path that Repository methods reach today.
    with pytest.raises(
        ValueError, match=f"^File is outside repository 'cytearc_docs': '{path}'$"
    ):
        datasets._relative_file_path("cytearc_docs", bucket_file(path, 1))


def test_download_rejects_files_the_transfer_skipped(monkeypatch, tmp_path):
    from cytearc import datasets

    file = bucket_file("cytearc_docs/alpha/data.bin", 3)
    monkeypatch.setattr(
        datasets,
        "_bucket_files",
        lambda repository, path, recursive: [file],
    )
    downloads = tmp_path / "downloads"
    destination = downloads / "alpha" / "data.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    # The transfer returns without writing the file it was asked for.
    monkeypatch.setattr(
        datasets, "download_bucket_files", lambda bucket_id, files, **kwargs: None
    )

    with pytest.raises(
        FileNotFoundError,
        match="^CyteBase did not download 'cytearc_docs/alpha/data.bin'$",
    ):
        datasets.Repository("cytearc_docs").download("alpha/data.bin", downloads)

    assert destination.read_bytes() == b"old"
    assert _tree(downloads) == ["alpha", "alpha/data.bin"]


@pytest.mark.parametrize("previous", [True, False], ids=["replace", "first"])
def test_failed_archive_install_restores_the_previous_download(
    monkeypatch, tmp_path, previous
):
    from cytearc import datasets

    dataset = tmp_path / "downloads" / "alpha"
    dataset.mkdir(parents=True)
    if previous:
        (dataset / "data.zarr").mkdir()
        (dataset / "data.zarr" / "old.json").write_text("old store")
        (dataset / "data.zarr.tar.gz").write_bytes(b"old archive")
    before = _tree(dataset)
    archive_bytes = _write_tar(
        tmp_path / "new.tar.gz", [("data.zarr/zarr.json", "file", b"{}")]
    ).read_bytes()
    remote = bucket_file("cytearc_docs/alpha/data.zarr.tar.gz", len(archive_bytes))
    monkeypatch.setattr(
        datasets, "_bucket_files", lambda repository, path, recursive: [remote]
    )

    def fake_download(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(archive_bytes)

    monkeypatch.setattr(datasets, "download_bucket_files", fake_download)
    real_replace = os.replace

    def replace(source, destination):
        # The extracted store is already in place when the archive move fails.
        if ".cytebase-download-" in str(source):
            raise OSError("No space left on device")
        real_replace(source, destination)

    monkeypatch.setattr(datasets.os, "replace", replace)

    with pytest.raises(OSError, match="^No space left on device$"):
        datasets.Repository("cytearc_docs").download_dataset(
            "alpha", tmp_path / "downloads", zarr=True, overwrite=previous
        )

    assert _tree(tmp_path / "downloads") == ["alpha", *(f"alpha/{p}" for p in before)]
    if previous:
        assert (dataset / "data.zarr" / "old.json").read_text() == "old store"
        assert (dataset / "data.zarr.tar.gz").read_bytes() == b"old archive"


@pytest.mark.parametrize(
    ("member", "message"),
    [
        pytest.param(
            ("data.zarr/link", "symlink", "/etc/passwd"),
            "Unsafe tar archive link: data.zarr/link",
            id="absolute-symlink",
        ),
        pytest.param(
            ("data.zarr/link", "symlink", "../../outside"),
            "Unsafe tar archive link: data.zarr/link",
            id="parent-symlink",
        ),
        pytest.param(
            ("data.zarr/hard", "hardlink", "../outside"),
            "Unsafe tar archive link: data.zarr/hard",
            id="parent-hardlink",
        ),
        pytest.param(
            ("data.zarr/pipe", "fifo", None),
            "Unsupported tar archive member: data.zarr/pipe",
            id="fifo",
        ),
        pytest.param(
            ("data.zarr/tty", "chardev", None),
            "Unsupported tar archive member: data.zarr/tty",
            id="device",
        ),
    ],
)
def test_archive_rejects_unsafe_links_and_special_members(tmp_path, member, message):
    from cytearc.datasets import _extract_and_replace

    work = tmp_path / "work"
    output = work / "data.zarr"
    output.mkdir(parents=True)
    (output / "zarr.json").write_text("original")
    staged = _write_tar(
        work / "download", [("data.zarr/zarr.json", "file", b"{}"), member]
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        _extract_and_replace(staged, work / "data.zarr.tar.gz")

    assert (output / "zarr.json").read_text() == "original"
    assert _tree(work) == ["data.zarr", "data.zarr/zarr.json", "download"]


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (("escape/file", "file", b"x"), "Unsafe tar archive member: escape/file"),
        (
            ("data/hard", "hardlink", "escape/file"),
            "Unsafe tar archive link: data/hard",
        ),
    ],
)
def test_tar_members_must_resolve_inside_the_staging_directory(
    tmp_path, member, message
):
    from cytearc.datasets import _safe_tar_members

    # Downloads extract into a fresh directory, where a relative name without
    # ".." stays inside; an existing symlink is what could carry it outside.
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging / "escape").symlink_to(outside, target_is_directory=True)
    path = _write_tar(tmp_path / "archive.tar.gz", [member])

    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(ValueError, match=f"^{message}$"):
            _safe_tar_members(archive, staging)
    assert list(outside.iterdir()) == []


def test_archive_cannot_replace_its_own_download_path(tmp_path):
    from cytearc.datasets import _extract_and_replace

    work = tmp_path / "work"
    work.mkdir()
    staged = _write_tar(work / "download", [("data.zarr/zarr.json", "file", b"{}")])

    # Without a .tar.gz suffix, the archive's directory would take the
    # archive's own destination.
    with pytest.raises(
        ValueError,
        match="^Tar archive member conflicts with an internal download path$",
    ):
        _extract_and_replace(staged, work / "data.zarr")
    assert _tree(work) == ["download"]


def test_remove_path_unlinks_symlinks_without_following_them(tmp_path):
    from cytearc.datasets import _remove_path

    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    nested = tmp_path / "nested" / "inner"
    nested.mkdir(parents=True)
    (nested / "data.bin").write_bytes(b"x")

    _remove_path(link)
    _remove_path(tmp_path / "nested")
    _remove_path(tmp_path / "missing")

    assert not link.is_symlink()
    assert (target / "keep.txt").read_text() == "keep"
    assert not (tmp_path / "nested").exists()


class _Bucket:
    """Serve bytes as the two hub functions that ``cytearc.datasets`` calls.

    Like the bucket, the listing names each file by a hash of its bytes, so
    publishing other bytes, even as many, changes the listing. ``fetched``
    records the path of each file downloaded.
    """

    def __init__(self, monkeypatch):
        from cytearc import datasets

        self.files: dict[str, bytes] = {}
        self.fetched: list[str] = []
        monkeypatch.setattr(datasets, "list_bucket_tree", self.list_bucket_tree)
        monkeypatch.setattr(
            datasets, "download_bucket_files", self.download_bucket_files
        )

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        return [
            BucketFile(
                type="file",
                path=path,
                size=len(data),
                xetHash=hashlib.sha256(data).hexdigest(),
                mtime=None,
                uploadedAt=None,
            )
            for path, data in sorted(self.files.items())
            if path.startswith(f"{prefix}/")
        ]

    def download_bucket_files(
        self, bucket_id, files, *, token=None, raise_on_missing_files=False
    ):
        for file, local in files:
            self.fetched.append(file.path)
            Path(local).write_bytes(self.files[file.path])


ARCHIVE = "cytearc_docs/alpha/data.zarr.tar.gz"


def _store_archive(tmp_path: Path, payload: bytes = b"{}") -> bytes:
    """Return a gzip tar archive that holds ``data.zarr/zarr.json``."""
    path = tmp_path / f"archive-{hashlib.sha256(payload).hexdigest()}.tar.gz"
    return _write_tar(path, [("data.zarr/zarr.json", "file", payload)]).read_bytes()


def _same_size_other_bytes(data: bytes) -> bytes:
    """Return ``data`` with another gzip modification time: as long, and other bytes."""
    changed = bytearray(data)
    changed[4] ^= 0xFF
    return bytes(changed)


def _download(*, zarr=True, overwrite=False, destination: Path):
    from cytearc import datasets

    return datasets.Repository("cytearc_docs").download_dataset(
        "alpha", destination, zarr=zarr, overwrite=overwrite
    )


def test_second_download_returns_the_store_untouched_with_saved_results(
    monkeypatch, tmp_path
):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"

    dataset = _download(destination=downloads)
    # A tutorial saves results into the store after the first call.
    (dataset / "data.zarr" / "results").write_text("saved analysis")
    (dataset / "data.zarr" / "zarr.json").write_text('{"attributes": "updated"}')
    again = _download(destination=downloads)

    assert again == dataset == downloads / "alpha"
    assert bucket.fetched == [ARCHIVE]
    assert (dataset / "data.zarr" / "results").read_text() == "saved analysis"
    assert (dataset / "data.zarr" / "zarr.json").read_text() == (
        '{"attributes": "updated"}'
    )
    assert (dataset / "data.zarr.tar.gz").read_bytes() == bucket.files[ARCHIVE]
    assert _tree(downloads) == [
        "alpha",
        "alpha/.cytebase-sources.json",
        "alpha/data.zarr",
        "alpha/data.zarr.tar.gz",
        "alpha/data.zarr/results",
        "alpha/data.zarr/zarr.json",
    ]


def test_second_download_of_source_files_fetches_nothing(monkeypatch, tmp_path):
    bucket = _Bucket(monkeypatch)
    bucket.files["cytearc_docs/alpha/data.h5"] = b"counts"
    bucket.files["cytearc_docs/alpha/raw/matrix.mtx"] = b"matrix"
    bucket.files[ARCHIVE] = b"not selected without zarr"
    downloads = tmp_path / "downloads"

    first = _download(zarr=False, destination=downloads)
    second = _download(zarr=False, destination=downloads)

    assert first == second == downloads / "alpha"
    assert bucket.fetched == [
        "cytearc_docs/alpha/data.h5",
        "cytearc_docs/alpha/raw/matrix.mtx",
    ]
    assert (first / "raw" / "matrix.mtx").read_bytes() == b"matrix"


@pytest.mark.parametrize(
    "change", ["truncated", "grown", "deleted", "republished", "same-size-republished"]
)
def test_download_fetches_only_the_file_that_no_longer_matches(
    monkeypatch, tmp_path, change
):
    bucket = _Bucket(monkeypatch)
    bucket.files["cytearc_docs/alpha/a.bin"] = b"aaaa"
    bucket.files["cytearc_docs/alpha/b.bin"] = b"bbbb"
    downloads = tmp_path / "downloads"
    dataset = _download(zarr=False, destination=downloads)
    local = dataset / "a.bin"
    if change == "truncated":
        local.write_bytes(b"aa")
    elif change == "grown":
        local.write_bytes(b"aaaaaaaa")
    elif change == "deleted":
        local.unlink()
    elif change == "republished":
        bucket.files["cytearc_docs/alpha/a.bin"] = b"new a"
    else:
        bucket.files["cytearc_docs/alpha/a.bin"] = b"AAAA"
    bucket.fetched.clear()

    _download(zarr=False, destination=downloads)

    assert bucket.fetched == ["cytearc_docs/alpha/a.bin"]
    assert local.read_bytes() == bucket.files["cytearc_docs/alpha/a.bin"]
    assert (dataset / "b.bin").read_bytes() == b"bbbb"
    # The next call finds both files current.
    _download(zarr=False, destination=downloads)
    assert bucket.fetched == ["cytearc_docs/alpha/a.bin"]


@pytest.mark.parametrize("same_size", [False, True], ids=["resized", "same-size"])
def test_republished_archive_raises_and_keeps_the_store(
    monkeypatch, tmp_path, same_size
):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = old = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    (dataset / "data.zarr" / "results").write_text("saved analysis")
    before = _tree(downloads)
    bucket.files[ARCHIVE] = (
        _same_size_other_bytes(old)
        if same_size
        else _store_archive(tmp_path, b'{"version": 2}')
    )

    with pytest.raises(FileExistsError) as raised:
        _download(destination=downloads)

    store = dataset / "data.zarr"
    assert str(raised.value) == (
        f"{store} is from an older version of the dataset: {ARCHIVE} has changed "
        "since data.zarr was made from it. Pass overwrite=True to replace data.zarr "
        "with the current version, deleting anything saved in it, or move it "
        "elsewhere to keep it."
    )
    assert bucket.fetched == [ARCHIVE]
    assert _tree(downloads) == before
    assert (store / "results").read_text() == "saved analysis"
    assert (dataset / "data.zarr.tar.gz").read_bytes() == old


def test_overwrite_replaces_the_store_with_the_current_version(monkeypatch, tmp_path):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    (dataset / "data.zarr" / "results").write_text("saved analysis")
    bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 2}')

    assert _download(overwrite=True, destination=downloads) == dataset
    _download(destination=downloads)

    assert bucket.fetched == [ARCHIVE, ARCHIVE]
    assert not (dataset / "data.zarr" / "results").exists()
    assert (dataset / "data.zarr" / "zarr.json").read_bytes() == b'{"version": 2}'
    assert (dataset / "data.zarr.tar.gz").read_bytes() == bucket.files[ARCHIVE]


def test_overwrite_fetches_current_files_again(monkeypatch, tmp_path):
    bucket = _Bucket(monkeypatch)
    bucket.files["cytearc_docs/alpha/data.h5"] = b"counts"
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"
    _download(zarr=False, destination=downloads)
    _download(destination=downloads)

    _download(zarr=False, overwrite=True, destination=downloads)
    _download(overwrite=True, destination=downloads)

    assert bucket.fetched == ["cytearc_docs/alpha/data.h5", ARCHIVE] * 2


@pytest.mark.parametrize("archive", ["current", "missing", "other-size"])
def test_store_extracted_before_the_record_existed(monkeypatch, tmp_path, archive):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"
    dataset = downloads / "alpha"
    # An earlier CyteArc left the store and its archive, and no record.
    (dataset / "data.zarr").mkdir(parents=True)
    (dataset / "data.zarr" / "results").write_text("saved analysis")
    if archive == "current":
        (dataset / "data.zarr.tar.gz").write_bytes(bucket.files[ARCHIVE])
    elif archive == "other-size":
        (dataset / "data.zarr.tar.gz").write_bytes(b"an older archive")
    before = _tree(downloads)

    if archive == "current":
        assert _download(destination=downloads) == dataset
        assert _tree(downloads) == sorted([*before, "alpha/.cytebase-sources.json"])
        # The record now names the store's archive, so the bucket's next
        # version is reported rather than taken for the same one.
        bucket.files[ARCHIVE] = _same_size_other_bytes(bucket.files[ARCHIVE])
        with pytest.raises(FileExistsError, match="is from an older version"):
            _download(destination=downloads)
    else:
        with pytest.raises(FileExistsError) as raised:
            _download(destination=downloads)
        assert str(raised.value) == (
            f"{dataset / 'data.zarr'} exists, but nothing records that it was made "
            f"from the current {ARCHIVE}, so it may be from an older version of the "
            "dataset. Pass overwrite=True to replace data.zarr with the current "
            "version, deleting anything saved in it, or move it elsewhere to keep it."
        )
        assert _tree(downloads) == before
    assert bucket.fetched == []
    assert (dataset / "data.zarr" / "results").read_text() == "saved analysis"


def test_store_is_reused_after_its_archive_is_deleted(monkeypatch, tmp_path):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    # Keeping only the store halves the disk that the dataset takes.
    (dataset / "data.zarr.tar.gz").unlink()
    (dataset / "data.zarr" / "results").write_text("saved analysis")

    _download(destination=downloads)

    assert bucket.fetched == [ARCHIVE]
    assert not (dataset / "data.zarr.tar.gz").exists()
    assert (dataset / "data.zarr" / "results").read_text() == "saved analysis"


def test_deleted_store_is_extracted_again_from_the_archive_in_place(
    monkeypatch, tmp_path
):
    import shutil

    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 1}')
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    before = _tree(downloads)
    shutil.rmtree(dataset / "data.zarr")

    _download(destination=downloads)

    assert bucket.fetched == [ARCHIVE]
    assert _tree(downloads) == before
    assert (dataset / "data.zarr" / "zarr.json").read_bytes() == b'{"version": 1}'
    assert (dataset / "data.zarr.tar.gz").read_bytes() == bucket.files[ARCHIVE]


@pytest.mark.parametrize("archive", ["truncated", "republished"])
def test_deleted_store_is_downloaded_again_when_its_archive_does_not_match(
    monkeypatch, tmp_path, archive
):
    import shutil

    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 1}')
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    shutil.rmtree(dataset / "data.zarr")
    if archive == "truncated":
        (dataset / "data.zarr.tar.gz").write_bytes(bucket.files[ARCHIVE][:20])
    else:
        bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 2}')

    _download(destination=downloads)
    _download(destination=downloads)

    assert bucket.fetched == [ARCHIVE, ARCHIVE]
    expected = b'{"version": 1}' if archive == "truncated" else b'{"version": 2}'
    assert (dataset / "data.zarr" / "zarr.json").read_bytes() == expected
    assert (dataset / "data.zarr.tar.gz").read_bytes() == bucket.files[ARCHIVE]


def test_staging_folder_of_a_killed_download_is_neither_used_nor_removed(
    monkeypatch, tmp_path
):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 1}')
    downloads = tmp_path / "downloads"
    # A killed download leaves part of the archive in its hidden staging
    # folder, which could as well belong to a download still running.
    partial = downloads / ".cytebase-download-killed" / "alpha" / "data.zarr.tar.gz"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bucket.files[ARCHIVE][:20])

    dataset = _download(destination=downloads)

    assert bucket.fetched == [ARCHIVE]
    assert (dataset / "data.zarr" / "zarr.json").read_bytes() == b'{"version": 1}'
    assert partial.read_bytes() == bucket.files[ARCHIVE][:20]
    assert sorted(path.name for path in downloads.iterdir()) == [
        ".cytebase-download-killed",
        "alpha",
    ]


@pytest.mark.parametrize("value", [1, 0, "yes", None, 1.0])
def test_overwrite_must_be_a_boolean(monkeypatch, tmp_path, value):
    from cytearc import datasets

    monkeypatch.setattr(datasets, "list_bucket_tree", _unexpected_download)
    monkeypatch.setattr(datasets, "download_bucket_files", _unexpected_download)

    with pytest.raises(TypeError, match="^overwrite must be a boolean$"):
        datasets.Repository("cytearc_docs").download_dataset(
            "alpha", tmp_path / "downloads", zarr=True, overwrite=value
        )
    assert not (tmp_path / "downloads").exists()


def test_store_files_listed_beside_the_archive_keep_saved_results(
    monkeypatch, tmp_path
):
    bucket = _Bucket(monkeypatch)
    # Some datasets also list their store's files, for remote reads.
    bucket.files["cytearc_docs/alpha/data.h5"] = b"counts"
    bucket.files["cytearc_docs/alpha/data.zarr/zarr.json"] = b'{"version": 1}'
    downloads = tmp_path / "downloads"
    dataset = _download(zarr=False, destination=downloads)
    store = dataset / "data.zarr"
    # Saving results rewrites the store's metadata and adds files.
    (store / "zarr.json").write_bytes(b'{"version": 1, "results": ["saved"]}')
    (store / "results").write_text("saved analysis")

    _download(zarr=False, destination=downloads)
    assert bucket.fetched == [
        "cytearc_docs/alpha/data.h5",
        "cytearc_docs/alpha/data.zarr/zarr.json",
    ]

    bucket.files["cytearc_docs/alpha/data.zarr/zarr.json"] = b'{"version": 2}'
    with pytest.raises(FileExistsError) as raised:
        _download(zarr=False, destination=downloads)
    assert str(raised.value) == (
        f"{store} is from an older version of the dataset: "
        "cytearc_docs/alpha/data.zarr/zarr.json has changed since data.zarr was "
        "made from it. Pass overwrite=True to download the files of data.zarr "
        "again over their local copies, or move it elsewhere to keep it."
    )
    assert len(bucket.fetched) == 2
    assert (store / "zarr.json").read_bytes() == b'{"version": 1, "results": ["saved"]}'

    _download(zarr=False, overwrite=True, destination=downloads)
    assert (store / "zarr.json").read_bytes() == b'{"version": 2}'


def test_extracted_store_is_not_overwritten_by_listed_store_files(
    monkeypatch, tmp_path
):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path, b'{"version": 1}')
    bucket.files["cytearc_docs/alpha/data.zarr/zarr.json"] = b'{"version": 1}'
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    (dataset / "data.zarr" / "zarr.json").write_bytes(b'{"results": ["saved"]}')

    with pytest.raises(FileExistsError, match="nothing records that it was made"):
        _download(zarr=False, destination=downloads)

    assert bucket.fetched == [ARCHIVE]
    assert (dataset / "data.zarr" / "zarr.json").read_bytes() == (
        b'{"results": ["saved"]}'
    )


def _local_catalog(monkeypatch, tmp_path) -> Path:
    from cytearc import datasets

    store = tmp_path / "catalog" / "alpha" / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text('{"version": 1}')
    monkeypatch.setenv(datasets._LOCAL_CATALOG_ENV, str(tmp_path / "catalog"))
    monkeypatch.setattr(datasets, "list_bucket_tree", _unexpected_download)
    monkeypatch.setattr(datasets, "download_bucket_files", _unexpected_download)
    return store


def test_local_catalog_copy_is_reused_until_its_source_changes(monkeypatch, tmp_path):
    source = _local_catalog(monkeypatch, tmp_path)
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    copy = dataset / "data.zarr"
    (copy / "results").write_text("saved analysis")

    assert _download(destination=downloads) == dataset
    assert (copy / "results").read_text() == "saved analysis"

    # Rebuilding the catalog's store rewrites its files.
    (source / "zarr.json").write_text('{"version": 2}')
    with pytest.raises(FileExistsError) as raised:
        _download(destination=downloads)
    assert str(raised.value) == (
        f"{copy} is from an older version of the dataset: {source} has changed "
        "since data.zarr was made from it. Pass overwrite=True to replace data.zarr "
        "with the current version, deleting anything saved in it, or move it "
        "elsewhere to keep it."
    )
    assert (copy / "results").read_text() == "saved analysis"

    _download(overwrite=True, destination=downloads)
    assert sorted(path.name for path in copy.iterdir()) == ["zarr.json"]
    assert (copy / "zarr.json").read_text() == '{"version": 2}'
    assert _download(destination=downloads) == dataset


def test_local_catalog_copy_keeps_a_store_it_has_no_record_of(monkeypatch, tmp_path):
    source = _local_catalog(monkeypatch, tmp_path)
    copy = tmp_path / "downloads" / "alpha" / "data.zarr"
    copy.mkdir(parents=True)
    (copy / "results").write_text("saved analysis")

    with pytest.raises(
        FileExistsError,
        match=f"nothing records that it was made from the current {source}",
    ):
        _download(destination=tmp_path / "downloads")

    assert _tree(copy.parent) == ["data.zarr", "data.zarr/results"]


def test_store_made_from_another_source_is_not_replaced(monkeypatch, tmp_path):
    from cytearc import datasets

    source = _local_catalog(monkeypatch, tmp_path)
    downloads = tmp_path / "downloads"
    dataset = _download(destination=downloads)
    monkeypatch.delenv(datasets._LOCAL_CATALOG_ENV)
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)

    with pytest.raises(FileExistsError) as raised:
        _download(destination=downloads)

    assert str(raised.value).startswith(
        f"{dataset / 'data.zarr'} was made from {source}, not from {ARCHIVE}. "
        "Pass overwrite=True"
    )
    assert bucket.fetched == []


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '{"files": {}}', '{"files": [], "stores": {}}', "\udcff"],
    ids=["invalid", "list", "no-stores", "files-list", "undecodable"],
)
def test_a_record_that_cytearc_did_not_write_raises(monkeypatch, tmp_path, content):
    bucket = _Bucket(monkeypatch)
    bucket.files["cytearc_docs/alpha/data.h5"] = b"counts"
    record = tmp_path / "alpha" / ".cytebase-sources.json"
    record.parent.mkdir()
    record.write_bytes(content.encode("utf-8", "surrogateescape"))

    with pytest.raises(ValueError) as raised:
        _download(zarr=False, destination=tmp_path)

    assert str(raised.value) == (
        f"{record} is not a record that CyteArc wrote. Move it elsewhere to start "
        "a new record; the next call then downloads the dataset's files again."
    )
    assert bucket.fetched == []


def test_a_failed_record_write_keeps_the_earlier_record_whole(monkeypatch, tmp_path):
    from cytearc import datasets

    bucket = _Bucket(monkeypatch)
    bucket.files["cytearc_docs/alpha/a.bin"] = b"aaaa"
    downloads = tmp_path / "downloads"
    dataset = _download(zarr=False, destination=downloads)
    record = (dataset / ".cytebase-sources.json").read_bytes()
    bucket.files["cytearc_docs/alpha/b.bin"] = b"bbbb"
    real_replace = os.replace

    def replace(source, destination):
        if Path(destination).name == ".cytebase-sources.json":
            raise OSError("No space left on device")
        real_replace(source, destination)

    monkeypatch.setattr(datasets.os, "replace", replace)
    with pytest.raises(OSError, match="^No space left on device$"):
        _download(zarr=False, destination=downloads)
    monkeypatch.setattr(datasets.os, "replace", real_replace)

    # The files are placed before the record, which only a whole file replaces.
    assert (dataset / ".cytebase-sources.json").read_bytes() == record
    assert _tree(downloads) == [
        "alpha",
        "alpha/.cytebase-sources.json",
        "alpha/a.bin",
        "alpha/b.bin",
    ]
    _download(zarr=False, destination=downloads)
    assert bucket.fetched == [
        "cytearc_docs/alpha/a.bin",
        "cytearc_docs/alpha/b.bin",
        "cytearc_docs/alpha/b.bin",
    ]


def test_record_keeps_what_another_call_recorded_meanwhile(monkeypatch, tmp_path):
    bucket = _Bucket(monkeypatch)
    bucket.files[ARCHIVE] = _store_archive(tmp_path)
    bucket.files["cytearc_docs/alpha/data.h5"] = b"counts"
    downloads = tmp_path / "downloads"
    fetch = bucket.download_bucket_files

    def fetch_while_another_call_finishes(bucket_id, files, **kwargs):
        # Another process downloads the store while this call fetches data.h5.
        if files[0][0].path.endswith("data.h5"):
            _download(destination=downloads)
        fetch(bucket_id, files, **kwargs)

    monkeypatch.setattr(
        "cytearc.datasets.download_bucket_files", fetch_while_another_call_finishes
    )
    _download(zarr=False, destination=downloads)

    record = json.loads((downloads / "alpha" / ".cytebase-sources.json").read_text())
    assert sorted(record["files"]) == ["data.h5", "data.zarr.tar.gz"]
    assert sorted(record["stores"]) == ["data.zarr"]
    _download(zarr=False, destination=downloads)
    _download(destination=downloads)
    assert bucket.fetched == [ARCHIVE, "cytearc_docs/alpha/data.h5"]


@pytest.mark.integration
def test_live_bucket_catalog_is_public():
    from cytearc import datasets

    with hub_or_skip("the cytearc_docs catalog of the Hugging Face bucket"):
        repositories = datasets.list_repositories()
        repository = datasets.connect("cytearc_docs")
        names = set(repository.list_datasets())
        files = [
            datasets._relative_file_path("cytearc_docs", file).as_posix()
            for file in datasets._bucket_files(
                "cytearc_docs", "tenx_5K_pbmc_rnaseq", recursive=True
            )
        ]
    assert "cytearc_docs" in repositories
    assert {
        "annotations",
        "bastidas-ponce_4K_pancreas-d15_rnaseq",
        "kang_14K_ifnb-pbmc_rnaseq",
        "kang_15K_pbmc_rnaseq",
        "kang_29K_ctrl-ifnb_pbmc_rnaseq",
        "swanson_7K_pbmc_teaseq",
        "tenx_10K_pbmc-v1_atacseq",
        "tenx_5K_pbmc_rnaseq",
        "tenx_8K_pbmc_citeseq",
    } <= names
    assert {
        "tenx_5K_pbmc_rnaseq/data.h5",
        "tenx_5K_pbmc_rnaseq/data.zarr.tar.gz",
        "tenx_5K_pbmc_rnaseq/manifest.json",
    } <= set(files)
    assert any(name.startswith("tenx_5K_pbmc_rnaseq/data.zarr/") for name in files)


@pytest.mark.integration
def test_live_zarr_archive_download(tmp_path, monkeypatch):
    from cytearc import datasets

    monkeypatch.delenv("CYTEARC_CYTEBASE_LOCAL", raising=False)
    listed = []
    list_files = datasets._bucket_files

    def recorded(*args, **kwargs):
        files = list_files(*args, **kwargs)
        listed.extend(files)
        return files

    monkeypatch.setattr(datasets, "_bucket_files", recorded)
    with hub_or_skip("tenx_5K_pbmc_rnaseq of the Hugging Face bucket"):
        dataset_path = datasets.connect("cytearc_docs").download_dataset(
            "tenx_5K_pbmc_rnaseq",
            tmp_path,
            zarr=True,
        )

    # Only the archive is downloaded, and no staging directory is left.
    assert dataset_path == tmp_path / "tenx_5K_pbmc_rnaseq"
    assert sorted(path.name for path in tmp_path.iterdir()) == [dataset_path.name]
    assert sorted(path.name for path in dataset_path.iterdir()) == [
        ".cytebase-sources.json",
        "data.zarr",
        "data.zarr.tar.gz",
    ]
    (remote,) = [file for file in listed if file.path.endswith("/data.zarr.tar.gz")]
    assert (dataset_path / "data.zarr.tar.gz").stat().st_size == remote.size
    # The extracted directory is the root group of a Zarr store.
    metadata = json.loads((dataset_path / "data.zarr" / "zarr.json").read_text())
    assert (metadata["zarr_format"], metadata["node_type"]) == (3, "group")

    # A second call lists the bucket again and fetches nothing.
    monkeypatch.setattr(datasets, "download_bucket_files", _unexpected_download)
    with hub_or_skip("tenx_5K_pbmc_rnaseq of the Hugging Face bucket"):
        again = datasets.connect("cytearc_docs").download_dataset(
            "tenx_5K_pbmc_rnaseq", tmp_path, zarr=True
        )
    assert again == dataset_path


def test_datasets_exports_the_example_repository_api():
    from cytearc import datasets

    assert datasets.__all__ == ["Repository", "connect", "list_repositories"]
    assert set(datasets.__all__) <= set(dir(datasets))


def test_cytebase_keeps_the_earlier_names_of_the_example_downloads():
    from cytearc import cytebase, datasets

    assert cytebase.__all__ == datasets.__all__
    for name in datasets.__all__:
        assert getattr(cytebase, name) is getattr(datasets, name)
    assert set(cytebase.__all__) <= set(dir(cytebase))


@pytest.mark.parametrize(
    "name",
    [
        "Catalog",
        "DatasetEntry",
        "Release",
        "embedding",
        "embedding_coordinates",
        "embeddings",
        "releases",
    ],
)
def test_cytebase_sdk_names_point_to_the_separate_package(name):
    from cytearc import cytebase

    with pytest.raises(AttributeError, match=r"pip install cytebase"):
        getattr(cytebase, name)
    assert not hasattr(cytebase, name)


def test_cytebase_rejects_unknown_names():
    from cytearc import cytebase

    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        cytebase.missing
