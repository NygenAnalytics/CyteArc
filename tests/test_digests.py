"""The digests that identify the files an import reads, and the one SplitMix64 finalizer.

Importers identify a file by the SHA-256 of its bytes and a folder, such as a
Zarr store, by the paths and bytes of its files. A saved import is reused only
while these digests stay the same, so the digests of checked-in fixtures are
pinned here: the SHA-256 of a file's bytes as ``sha256sum`` gives it, and the
digests of a folder and of a variant source as the importers record them.
"""

import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from cytearc.utils.digest import directory_sha256, file_sha256, path_sha256

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parent.parent
# The most bytes that a digest reads at once.
READ_BYTES = 256 * 1024


def _store(path: Path, files: dict[str, bytes]) -> Path:
    """A directory of files, written in the order given, such as a Zarr store's."""
    for relative, data in files.items():
        (path / relative).parent.mkdir(parents=True, exist_ok=True)
        (path / relative).write_bytes(data)
    return path


@pytest.mark.parametrize(
    "size",
    [0, 1, 1000, READ_BYTES - 1, READ_BYTES, READ_BYTES + 1, 3 * READ_BYTES + 7],
)
def test_a_file_is_identified_by_the_sha256_of_its_bytes(tmp_path, size) -> None:
    data = np.random.default_rng(size).bytes(size)
    path = tmp_path / "file.bin"
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
    assert file_sha256(str(path)) == file_sha256(path)
    assert path_sha256(path) == file_sha256(path)


def test_the_fixtures_keep_the_digests_that_identify_their_imports() -> None:
    # sha256sum gives the files' digests.
    assert (
        file_sha256(DATA / "clones" / "trex_clones.txt")
        == "b53cad6121ee0cb240660fdcaf67419cb36615c20dea7feb73bfa510d77506b8"
    )
    assert (
        file_sha256(DATA / "guide_calls" / "protospacer_calls_per_cell.csv.gz")
        == "ea4285c64f4f47a31e5603744db9ca1dfd0f8bbe48ef93d50b47a9ea0b88ed24"
    )
    # The Zarr store of treedata 0.4.0, as tree and allele imports record it.
    store = DATA / "phylogenies" / "truth_tree_treedata_0_4_0.zarr"
    digest = "48f4e63113da315569a9f0ac78af849ed9eec2646873eda4355aab570576faeb"
    assert directory_sha256(store) == path_sha256(store) == digest
    # An mgatk folder, as a variant import records it: the SHA-256 of each
    # file that the reader takes, by its role, and of the roles with their
    # digests.
    from cytearc.readers._variant_files import source_sha256
    from cytearc.readers.mgatk import mgatk_source

    folder = DATA / "variants" / "signac_test_mgatk"
    reader = mgatk_source(folder, format="mgatk", swap_strands=False)
    digest, digests = source_sha256(reader.files)
    assert digest == "3fb404f880b3afd0f9b3fae77e1f34a82dc813bcd8f676bef1f4c6c177731e34"
    assert digests == {
        role: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for role, path in reader.files
    }
    assert sorted(digests) == ["A", "C", "G", "T", "depth_table", "reference"]


def test_a_directory_is_identified_by_the_paths_and_bytes_of_its_files(
    tmp_path,
) -> None:
    files = {"zarr.json": b'{"node_type": "group"}', "obsm/characters/c/0": b"abc"}
    first = _store(tmp_path / "first.zarr", files)
    digest = directory_sha256(first)
    assert len(digest) == 64 and int(digest, 16) >= 0
    # Another location, files written in the other order and an empty folder
    # give the same identity.
    again = _store(tmp_path / "again.zarr", dict(reversed(files.items())))
    (again / "empty").mkdir()
    assert directory_sha256(again) == digest
    copy = tmp_path / "copy.zarr"
    shutil.copytree(first, copy)
    assert directory_sha256(copy) == digest
    # Other bytes, another path for the same bytes, and the same bytes split
    # between the files at another place each give another identity.
    edited = _store(tmp_path / "edited.zarr", {**files, "obsm/characters/c/0": b"abd"})
    renamed = _store(
        tmp_path / "renamed.zarr",
        {"zarr.json": files["zarr.json"], "obsm/characters/c/1": b"abc"},
    )
    moved = _store(
        tmp_path / "moved.zarr",
        {"zarr.json": files["zarr.json"] + b"a", "obsm/characters/c/0": b"bc"},
    )
    # One file whose bytes spell the next file's length and path, and two
    # files: their paths and bytes run together alike, but not their sizes.
    record = len(b"b").to_bytes(8, "little") + b"b"
    joined = _store(tmp_path / "joined.zarr", {"a": b"X" + record + b"Y"})
    apart = _store(tmp_path / "apart.zarr", {"a": b"X", "b": b"Y"})
    digests = {
        directory_sha256(path) for path in (edited, renamed, moved, joined, apart)
    }
    assert len(digests) == 5 and digest not in digests


def test_a_directory_of_files_longer_than_one_read_keeps_its_digest(tmp_path) -> None:
    # A file of more than one read, an empty file and an empty folder, with
    # the digest that tree and allele imports record for such a store.
    folder = tmp_path / "folder"
    (folder / "a" / "b").mkdir(parents=True)
    (folder / "a" / "x").write_bytes(b"x" * 300_000)
    (folder / "y").write_bytes(b"")
    assert (
        directory_sha256(folder)
        == "7d41a858ee988a3fcf40df5bc68a09b31ecf623ed8028089fee7095e0736e32f"
    )


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="permissions do not stop the superuser from listing a folder",
)
def test_a_directory_whose_folder_cannot_be_listed_is_refused(tmp_path) -> None:
    store = _store(tmp_path / "store.zarr", {"zarr.json": b"{}", "obsm/c/0": b"x"})
    hidden = store / "obsm"
    hidden.chmod(0)
    try:
        # Without the refusal, the folder would hash as if it held no file.
        with pytest.raises(PermissionError):
            directory_sha256(store)
    finally:
        hidden.chmod(0o755)


def test_each_digest_refuses_a_path_it_cannot_read(tmp_path) -> None:
    folder = _store(tmp_path / "folder", {"file": b"x"})
    with pytest.raises(NotADirectoryError):
        directory_sha256(folder / "file")
    with pytest.raises(FileNotFoundError):
        directory_sha256(tmp_path / "absent")
    with pytest.raises(IsADirectoryError):
        file_sha256(folder)
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        path_sha256(tmp_path / "absent")
    assert (
        path_sha256(folder) == directory_sha256(folder) != file_sha256(folder / "file")
    )


def test_a_directory_refuses_a_file_name_that_is_not_text(tmp_path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    try:
        # A name of bytes that UTF-8 cannot decode, which Linux accepts.
        with open(os.path.join(os.fsencode(folder), b"\xff.bin"), "wb") as handle:
            handle.write(b"x")
    except OSError:
        pytest.skip("the file system refuses names that are not UTF-8")
    with pytest.raises(UnicodeEncodeError):
        directory_sha256(folder)


@pytest.mark.skipif(
    not os.path.exists("/proc/version"), reason="needs Linux's /proc/version"
)
def test_a_file_whose_size_reads_as_zero_is_read_to_its_end() -> None:
    # The kernel reports a size of 0 for its own files, which hold text.
    path = Path("/proc/version")
    assert path.stat().st_size == 0
    data = path.read_bytes()
    assert data and file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_splitmix64s_multipliers_live_only_in_the_one_finalizer() -> None:
    # Every module of the package and of the profiling tools mixes with
    # utils.digest's mix64 or mix64_array; another copy of the finalizer
    # would be another place for a seeded result to change.
    multipliers = re.compile(
        r"0x(?:BF58476D1CE4E5B9|94D049BB133111EB)|13787848793156543929"
        r"|10723151780598845931",
        re.IGNORECASE,
    )
    copies = sorted(
        path.relative_to(ROOT).as_posix()
        for package in ("cytearc", "profiling")
        for path in (ROOT / package).rglob("*.py")
        if multipliers.search(path.read_text(encoding="utf-8"))
    )
    assert copies == ["cytearc/utils/digest.py"]
