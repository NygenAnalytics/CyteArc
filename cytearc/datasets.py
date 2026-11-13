"""Download CyteArc's public example datasets and the networks its operations read.

The examples live in repositories of the public ``Nygen/cytebase`` Hugging Face
bucket, such as ``cytearc_docs`` for the documentation datasets, and the STRING
networks in its ``cytearc_networks`` repository. This module only lists and
downloads files. The CyteBase catalog of cloud-hosted datasets is the separate
``cytebase`` package.
"""

import errno
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zlib
from collections.abc import Collection, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import httpx2
from huggingface_hub import (
    BucketFile,
    BucketFolder,
    download_bucket_files,
    list_bucket_tree,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    XetDownloadError,
)
from huggingface_hub.utils import disable_progress_bars

from .utils.arguments import bool_argument
from .utils.cache import cache_directory
from .utils.digest import file_sha256
from .utils.logging import progress_enabled

__all__ = ["Repository", "connect", "list_repositories"]

_BUCKET_ID = "Nygen/cytebase"
_ZARR_ARCHIVE_SUFFIX = ".zarr.tar.gz"
_ARCHIVE_SUFFIX = ".tar.gz"
_LOCAL_CATALOG_ENV = "CYTEARC_CYTEBASE_LOCAL"
# The record, in each dataset's directory, of what its files came from.
_SOURCES_NAME = ".cytebase-sources.json"


def _safe_name(name: str, *, kind: str) -> str:
    posix_name = PurePosixPath(name)
    windows_name = PureWindowsPath(name)
    if (
        not name
        or name in {".", ".."}
        or posix_name.name != name
        or windows_name.name != name
    ):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


def _safe_relative_path(path: str, *, kind: str) -> PurePosixPath:
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"Invalid {kind}: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid {kind}: {path!r}")
    return PurePosixPath(*parts)


def _remote_path(repository: str, path: str | None = None) -> str:
    if path is None:
        return repository
    relative_path = _safe_relative_path(path, kind="remote path")
    return f"{repository}/{relative_path.as_posix()}"


def _relative_file_path(repository: str, file: BucketFile) -> PurePosixPath:
    repository_prefix = f"{repository}/"
    if not file.path.startswith(repository_prefix):
        raise ValueError(f"File is outside repository {repository!r}: {file.path!r}")
    return _safe_relative_path(
        file.path.removeprefix(repository_prefix),
        kind="bucket file path",
    )


def _bucket_files(
    repository: str,
    path: str | None = None,
    *,
    recursive: bool,
) -> list[BucketFile]:
    prefix = _remote_path(repository, path)
    files = []
    for item in list_bucket_tree(
        _BUCKET_ID,
        prefix=prefix,
        recursive=recursive,
        token=False,
    ):
        if not isinstance(item, BucketFile):
            continue
        if item.path != prefix and not item.path.startswith(f"{prefix}/"):
            continue
        _relative_file_path(repository, item)
        files.append(item)
    return sorted(files, key=lambda file: file.path)


def list_repositories() -> list[str]:
    """Return the top-level example repositories in the public bucket."""
    repositories = []
    for item in list_bucket_tree(_BUCKET_ID, recursive=False, token=False):
        if isinstance(item, BucketFolder) and "/" not in item.path:
            repositories.append(_safe_name(item.path, kind="repository name"))
    return sorted(repositories)


def connect(repository: str) -> "Repository":
    """Connect to a public example repository, such as ``cytearc_docs``."""
    repository = _safe_name(repository, kind="repository name")
    available = list_repositories()
    if repository not in available:
        choices = "\n".join(available)
        raise KeyError(
            f"{repository!r} is not a CyteBase repository. "
            f"Available repositories:\n{choices}"
        )
    return Repository(repository)


@dataclass(frozen=True, slots=True)
class Repository:
    """Read files from one top-level example repository."""

    name: str

    def __post_init__(self) -> None:
        _safe_name(self.name, kind="repository name")

    def list_datasets(self) -> list[str]:
        """Return immediate dataset directories in this repository."""
        prefix = f"{self.name}/"
        datasets = []
        for item in list_bucket_tree(
            _BUCKET_ID,
            prefix=self.name,
            recursive=False,
            token=False,
        ):
            if not isinstance(item, BucketFolder):
                continue
            if not item.path.startswith(prefix):
                continue
            relative_path = item.path.removeprefix(prefix)
            if "/" not in relative_path:
                datasets.append(_safe_name(relative_path, kind="dataset name"))
        return sorted(datasets)

    def download(
        self,
        path: str,
        destination: str | Path = ".",
    ) -> list[Path]:
        """Download a file or directory while preserving its repository path."""
        _safe_relative_path(path, kind="download path")
        files = _bucket_files(self.name, path, recursive=True)
        if not files:
            raise FileNotFoundError(f"No CyteBase files found at {self.name}/{path}")
        return _download_files(self.name, files, Path(destination))

    def download_dataset(
        self,
        name: str,
        destination: str | Path = ".",
        *,
        zarr: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Download one dataset into ``destination/name`` and return that directory.

        Without ``zarr``, the call downloads the dataset's files other than its
        Zarr store, such as its source count files. It skips the store's
        archive and the files under the store's directory, which some datasets
        also list unpacked for remote reads. With ``zarr=True``, it downloads
        the dataset's ``data.zarr.tar.gz`` archive and extracts it to
        ``data.zarr``, keeping the archive beside it.

        A later call reuses what an earlier call placed in ``destination``.
        Each call records the bucket's listing of each file it placed, its
        path, size and Xet hash, in ``.cytebase-sources.json`` in the returned
        directory, together with the archive that each extracted directory
        came from. A file is not downloaded again while the bucket's listing
        of it equals the recorded one and the local copy has the listed size.
        An existing ``data.zarr`` is returned untouched, with everything saved
        in it, while the record says that it was extracted from the archive
        that the bucket lists now; its archive is not needed for that. A
        ``data.zarr`` extracted before CyteArc kept this record is reused when
        the archive beside it has the size of the bucket's archive; no bytes
        are compared.

        Args:
            name: The dataset's directory in the repository.
            destination: The directory that receives the dataset's directory.
            zarr: Download and extract the dataset's Zarr archive instead of
                its other files.
            overwrite: Download the selected files again and replace their
                local copies. With ``zarr=True``, the extracted ``data.zarr``
                replaces the existing one, deleting anything saved in it.
                Without ``overwrite``, an existing ``data.zarr`` is never
                replaced, even one that another call places while this one
                runs.

        Raises:
            TypeError: If ``zarr`` or ``overwrite`` is not a boolean.
            KeyError: If the repository holds no such dataset.
            FileNotFoundError: If ``zarr`` is set and the dataset has no Zarr
                archive, or if it is not set and the dataset lists no files
                besides its Zarr store.
            FileExistsError: If ``overwrite`` is not set and ``data.zarr``
                exists but the record does not say that it was extracted from
                the archive that the bucket lists now, or if another call
                places ``data.zarr`` while this one runs and has not recorded
                it as current.
            ValueError: If ``.cytebase-sources.json`` is not a record that
                CyteArc wrote, or if the archive kept beside a deleted
                ``data.zarr`` cannot be extracted.
        """
        zarr = bool_argument(zarr, "zarr")
        overwrite = bool_argument(overwrite, "overwrite")
        name = _safe_name(name, kind="dataset name")
        destination = Path(destination).absolute()
        if zarr:
            local_dataset = _local_dataset_store(name)
            if local_dataset is not None:
                return _copy_local_dataset(
                    local_dataset, destination / name, overwrite=overwrite
                )
        files = _bucket_files(self.name, name, recursive=True)
        if not files:
            choices = "\n".join(self.list_datasets())
            raise KeyError(
                f"{name!r} is not in repository {self.name!r}. "
                f"Available datasets:\n{choices}"
            )

        if zarr:
            selected = [
                file for file in files if file.path.endswith(_ZARR_ARCHIVE_SUFFIX)
            ]
            if not selected:
                raise FileNotFoundError(f"No Zarr archive is available for {name!r}")
        else:
            selected = [
                file
                for file in files
                if not file.path.endswith(_ZARR_ARCHIVE_SUFFIX)
                and not _inside_zarr_store(_dataset_key(self.name, file))
            ]
            if not selected:
                raise FileNotFoundError(
                    f"{name!r} lists no files besides its Zarr store. Pass "
                    "zarr=True to download the store's archive, or open the store "
                    "remotely by its hf:// URI."
                )

        dataset_path = destination / name
        plan = _plan_dataset_download(
            self.name, selected, dataset_path, overwrite=overwrite
        )
        if plan.forget:
            # Only a call that places these stores again may record them.
            _record_sources(dataset_path, files={}, stores={}, forget=plan.forget)
        action = "downloading"
        try:
            if plan.fetch:
                _download_files(self.name, plan.fetch, destination, overwrite=overwrite)
            action = "extracting its archive again"
            for file in plan.extract:
                _extract_archive_in_place(dataset_path / _dataset_key(self.name, file))
        except _StoreAppeared as appeared:
            # Another call placed the store meanwhile: reuse it once recorded.
            if _stores_recorded(dataset_path, plan.stores):
                return dataset_path
            raise FileExistsError(
                _appeared_message(Path(appeared.filename), action)
            ) from None
        if plan.files or plan.stores:
            _record_sources(dataset_path, files=plan.files, stores=plan.stores)
        return dataset_path


def _local_catalog_root() -> Path | None:
    raw = os.environ.get(_LOCAL_CATALOG_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def _local_dataset_store(name: str) -> Path | None:
    root = _local_catalog_root()
    if root is None:
        return None
    store = root / name / "data.zarr"
    return store if store.is_dir() else None


def _copy_local_dataset(store: Path, target: Path, *, overwrite: bool) -> Path:
    """Copy ``store`` to ``target/data.zarr``, keeping any other files in ``target``.

    A ``data.zarr`` already in ``target`` is kept when the record says that it
    was copied from ``store`` as ``store`` is now, replaced when ``overwrite``
    is set, and raises otherwise. ``store`` counts as changed when any of its
    files changes path, size or modification time, so touching a file of the
    catalog's store makes the copy read as older.
    """
    copy = target / "data.zarr"
    current = {"source": str(store.absolute()), "digest": _listing_digest(store)}
    made_from = _read_sources(target)["stores"].get(copy.name)
    if _path_exists(copy) and not overwrite:
        _check_existing_store(copy, made_from, current)
        return target
    if made_from is not None and not overwrite:
        # The recorded copy is gone; only a call that copies it again may record it.
        _record_sources(target, files={}, stores={}, forget=[copy.name])
    target.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(dir=target, prefix=".data.zarr.copy-"))
    try:
        staged = staging_path / "data.zarr"
        shutil.copytree(store, staged)
        if overwrite:
            backup_path = staging_path / "backup"
            backup_path.mkdir()
            _replace_staged_paths([(staged, copy)], backup_path)
        else:
            _rename_new(staged, copy)
    except _StoreAppeared:
        # Another call copied the store meanwhile: reuse it once recorded.
        if _stores_recorded(target, {copy.name: current}):
            return target
        raise FileExistsError(
            _appeared_message(copy, "copying it from the local catalog")
        ) from None
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
    _record_sources(target, files={}, stores={copy.name: current})
    return target


def _listing_digest(directory: Path) -> str:
    """Return a digest of the paths, sizes and modification times below ``directory``.

    Rewriting, adding, removing or touching any file changes the digest, which
    reads no file's bytes.
    """
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            relative = path.relative_to(directory).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _dataset_key(repository: str, file: BucketFile) -> str:
    """Return the path of ``file`` inside its dataset's directory."""
    relative = _relative_file_path(repository, file)
    return PurePosixPath(*relative.parts[1:]).as_posix()


def _inside_zarr_store(key: str) -> bool:
    """Return whether a file at ``key`` in a dataset lies in a Zarr store's directory."""
    top = key.split("/", 1)[0]
    return top != key and top.endswith(".zarr")


def _bucket_source(file: BucketFile) -> dict[str, Any]:
    """Return the bucket's listing of ``file``, which changes when it is published again."""
    return {"source": file.path, "size": file.size, "xetHash": file.xet_hash}


# The fields of a record entry that identify its source; any other field is
# ignored when entries are compared.
_SOURCE_FIELDS = ("source", "size", "xetHash", "digest")


def _same_source(
    recorded: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> bool:
    """Return whether the record entry ``recorded`` names the source ``current``."""
    return recorded is not None and all(
        recorded.get(field) == current.get(field) for field in _SOURCE_FIELDS
    )


def _read_sources(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Return the record of what each file and directory in ``dataset_path`` came from.

    ``files`` maps each downloaded file's path in the directory to the bucket's
    listing of it. ``stores`` maps each directory extracted or copied there,
    such as ``data.zarr``, to the archive or local store it was made from.
    """
    path = dataset_path / _SOURCES_NAME
    try:
        sources = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"files": {}, "stores": {}}
    except (ValueError, IsADirectoryError):
        sources = None
    if not (
        isinstance(sources, dict)
        and all(isinstance(sources.get(part), dict) for part in ("files", "stores"))
        and all(
            isinstance(entry, dict) and isinstance(entry.get("source"), str)
            for part in ("files", "stores")
            for entry in sources[part].values()
        )
    ):
        raise ValueError(
            f"{path} is not a record that CyteArc wrote. Move it elsewhere to "
            "start a new record. The next call then downloads the dataset's "
            "files again, and keeps an existing data.zarr only when the archive "
            "beside it has the size of the bucket's."
        )
    return sources


def _record_sources(
    dataset_path: Path,
    *,
    files: Mapping[str, dict[str, Any]],
    stores: Mapping[str, dict[str, Any]],
    forget: Collection[str] = (),
) -> None:
    """Add what a call placed in ``dataset_path`` to its record, replacing the file whole.

    ``forget`` names stores to drop from the record, ones that are gone.

    The record is read again first, which keeps most of what another call
    recorded meanwhile; a call that writes between that read and the
    replacement loses its entries, and the next call then judges those files
    as unrecorded.
    """
    sources = _read_sources(dataset_path)
    sources["files"].update(files)
    sources["stores"].update(stores)
    for key in forget:
        sources["stores"].pop(key, None)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=dataset_path,
        prefix=f"{_SOURCES_NAME}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(sources, handle, indent=2, sort_keys=True)
        os.replace(handle.name, dataset_path / _SOURCES_NAME)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _stores_recorded(
    dataset_path: Path, stores: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Return whether the record now names each of ``stores`` as made from its source."""
    recorded = _read_sources(dataset_path)["stores"]
    return all(
        _same_source(recorded.get(key), source) for key, source in stores.items()
    )


@dataclass(slots=True)
class _DatasetPlan:
    """What one ``download_dataset`` call fetches, extracts and records."""

    fetch: list[BucketFile] = field(default_factory=list)
    # Archives whose directory is gone, to extract again from their local copy.
    extract: list[BucketFile] = field(default_factory=list)
    # Stores that the record names but that are gone, to drop before placing.
    forget: list[str] = field(default_factory=list)
    # The record's new entries, written once everything is in place.
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    stores: dict[str, dict[str, Any]] = field(default_factory=dict)


def _plan_dataset_download(
    repository: str,
    selected: list[BucketFile],
    dataset_path: Path,
    *,
    overwrite: bool,
) -> _DatasetPlan:
    """Plan how a call delivers the ``selected`` files of a dataset.

    Without ``overwrite``, a file is fetched unless the record holds the
    bucket's listing of it and its local copy has the listed size. A
    directory extracted from an archive is never fetched while it exists. It
    is kept when the record says that it was extracted from the archive the
    bucket lists now, and adopted as that archive's when nothing is recorded
    and it is a directory with the archive beside it at the listed size, as
    CyteArc left it before it kept a record. Only the directory is recorded
    then, since the archive's bytes were not compared. Any other existing
    directory raises. A directory that is gone is extracted again from its
    archive when the local copy is recorded as current, and fetched otherwise.

    Raises:
        FileExistsError: If an existing directory would have to be replaced.
        ValueError: If the record is not one that CyteArc wrote.
    """
    sources = _read_sources(dataset_path)
    plan = _DatasetPlan()
    for file in selected:
        key = _dataset_key(repository, file)
        source = _bucket_source(file)
        local = dataset_path / key
        complete = local.is_file() and local.stat().st_size == file.size
        current = complete and _same_source(sources["files"].get(key), source)
        if not key.endswith(_ARCHIVE_SUFFIX):
            if overwrite or not current:
                plan.fetch.append(file)
                plan.files[key] = source
            continue
        directory_key = key.removesuffix(_ARCHIVE_SUFFIX)
        directory = dataset_path / directory_key
        if overwrite or not _path_exists(directory):
            if overwrite or not current:
                plan.fetch.append(file)
                plan.files[key] = source
            else:
                plan.extract.append(file)
            plan.stores[directory_key] = source
            if not overwrite and directory_key in sources["stores"]:
                plan.forget.append(directory_key)
            continue
        made_from = sources["stores"].get(directory_key)
        if made_from is None and complete and directory.is_dir():
            plan.stores[directory_key] = source
            continue
        _check_existing_store(directory, made_from, source)
    return plan


def _check_existing_store(
    directory: Path,
    made_from: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> None:
    """Return when the record says ``directory`` was made from ``current``, and raise otherwise.

    Raises:
        FileExistsError: Saying what ``directory`` was made from.
    """
    if _same_source(made_from, current):
        return
    if made_from is None and not directory.is_dir():
        reason = f"{directory} exists but is not a directory"
    elif made_from is None:
        reason = (
            f"{directory} exists, but nothing records what it was made from, so "
            "it may be from an older version of the dataset"
        )
    elif made_from["source"] == current["source"]:
        reason = (
            f"{directory} is from an older version of the dataset: it was made "
            f"from an earlier version of {_describe_source(made_from)}"
        )
    else:
        reason = (
            f"{directory} was made from {_describe_source(made_from)}, not from "
            f"{_describe_source(current)}"
        )
    raise FileExistsError(
        f"{reason}. Pass overwrite=True to replace {directory.name} with the "
        "current version, deleting anything saved in it, or move it elsewhere "
        "to keep it."
    )


def _describe_source(source: Mapping[str, Any]) -> str:
    """Name what a record entry says a directory was made from."""
    if "digest" in source:
        return f"the local store {source['source']}"
    return f"the archive {source['source']}"


def _appeared_message(directory: Path, action: str) -> str:
    return (
        f"{directory} appeared while this call was {action}, so it is kept and "
        "nothing replaced it. Call again to reuse it, or pass overwrite=True to "
        "replace it, deleting anything saved in it."
    )


def _extract_archive_in_place(archive: Path) -> None:
    """Extract ``archive`` again into its directory, which must not exist."""
    try:
        _extract_and_replace(archive, archive, overwrite=False)
    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as error:
        raise ValueError(
            f"{archive} cannot be extracted: {error}. Pass overwrite=True to "
            "download it again."
        ) from error


def _download_files(
    repository: str,
    files: list[BucketFile],
    destination: Path,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """Download ``files`` below ``destination``, keeping their repository paths.

    Every file is downloaded into a hidden staging directory and checked
    before anything is placed. An archive is extracted beside itself. Without
    ``overwrite``, an extracted directory never replaces one that exists by
    then, and ``_StoreAppeared`` is raised instead.
    """
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            dir=destination,
            prefix=".cytebase-download-",
        )
    )
    downloads = []

    try:
        for file in files:
            relative_path = _relative_file_path(repository, file)
            staged_file = staging_path.joinpath(*relative_path.parts)
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            final_file = destination.joinpath(*relative_path.parts)
            downloads.append((file, staged_file, final_file))

        with (
            nullcontext()
            if progress_enabled()
            else disable_progress_bars("huggingface_hub.download_bucket_files")
        ):
            download_bucket_files(
                _BUCKET_ID,
                files=[(file, staged_file) for file, staged_file, _ in downloads],
                token=False,
            )

        for file, staged_file, _ in downloads:
            if not staged_file.is_file():
                raise FileNotFoundError(f"CyteBase did not download {file.path!r}")
            if staged_file.stat().st_size != file.size:
                raise OSError(
                    f"Downloaded {staged_file.stat().st_size} bytes for {file.path!r}, "
                    f"expected {file.size}"
                )

        for _, staged_file, final_file in downloads:
            final_file.parent.mkdir(parents=True, exist_ok=True)
            if final_file.name.endswith(_ARCHIVE_SUFFIX):
                _extract_and_replace(staged_file, final_file, overwrite=overwrite)
            else:
                os.replace(staged_file, final_file)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)

    return [final_file for _, _, final_file in downloads]


def _safe_tar_members(
    archive: tarfile.TarFile,
    staging_path: Path,
) -> list[tarfile.TarInfo]:
    staging_root = staging_path.resolve()
    members = archive.getmembers()

    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe tar archive member: {member.name}")

        target = (staging_root / Path(*member_path.parts)).resolve()
        if not target.is_relative_to(staging_root):
            raise ValueError(f"Unsafe tar archive member: {member.name}")

        if member.issym() or member.islnk():
            link_path = PurePosixPath(member.linkname)
            if link_path.is_absolute() or ".." in link_path.parts:
                raise ValueError(f"Unsafe tar archive link: {member.name}")

            link_root = target.parent if member.issym() else staging_root
            link_target = (link_root / Path(*link_path.parts)).resolve()
            if not link_target.is_relative_to(staging_root):
                raise ValueError(f"Unsafe tar archive link: {member.name}")
        elif not member.isfile() and not member.isdir():
            raise ValueError(f"Unsupported tar archive member: {member.name}")

    return members


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


class _StoreAppeared(FileExistsError):
    """A directory appeared where a call was about to place one.

    Its ``filename`` names the directory.
    """


def _rename_new(source: Path, destination: Path) -> None:
    """Rename the directory ``source`` to ``destination``, which must not exist.

    POSIX renames a directory onto a missing or an empty directory only, and
    Windows onto a missing path only, so a directory that another process
    filled at ``destination`` is kept.

    Raises:
        _StoreAppeared: If ``destination`` exists.
    """
    try:
        os.rename(source, destination)
    except OSError as error:
        if _path_exists(destination):
            raise _StoreAppeared(
                errno.EEXIST, "Directory appeared meanwhile", str(destination)
            ) from error
        raise


def _replace_staged_paths(
    staged_paths: list[tuple[Path, Path]],
    backup_path: Path,
    *,
    keep: Collection[Path] = frozenset(),
) -> None:
    """Move each staged path to its destination, or none of them.

    A destination that exists is moved into ``backup_path`` and restored if a
    later move fails, except one in ``keep``, a directory that must not exist,
    which is never moved aside, so that its move raises ``_StoreAppeared``.
    """
    replacements: list[tuple[Path, Path | None]] = []

    try:
        for index, (source, destination) in enumerate(staged_paths):
            if destination in keep:
                _rename_new(source, destination)
                replacements.append((destination, None))
                continue
            backup = backup_path / str(index)
            previous = backup if _path_exists(destination) else None
            if previous is not None:
                os.replace(destination, previous)

            try:
                os.replace(source, destination)
            except BaseException:
                if previous is not None:
                    os.replace(previous, destination)
                raise
            replacements.append((destination, previous))
    except BaseException:
        for destination, previous in reversed(replacements):
            _remove_path(destination)
            if previous is not None:
                os.replace(previous, destination)
        raise


def _extract_and_replace(
    archive_path: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Extract ``archive_path`` beside ``destination`` and move the archive there.

    The archive's one directory replaces an existing one only with
    ``overwrite``, and raises otherwise.
    """
    staging_path = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.extract-",
        )
    )
    backup_path = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.backup-",
        )
    )

    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _safe_tar_members(archive, staging_path)
            archive.extractall(staging_path, members=members, filter="data")
            # Read the stream to its end, so gzip checks its CRC and length.
            stream = archive.fileobj
            while stream is not None and stream.read(1 << 20):
                pass

        extracted_paths = sorted(staging_path.iterdir(), key=lambda path: path.name)
        output_name = destination.name.removesuffix(_ARCHIVE_SUFFIX)
        if (
            len(extracted_paths) != 1
            or extracted_paths[0].name != output_name
            or extracted_paths[0].is_symlink()
            or not extracted_paths[0].is_dir()
        ):
            raise ValueError(
                f"Tar archive must contain only the directory {output_name!r}"
            )
        extracted_replacements = [
            (path, destination.parent / path.name) for path in extracted_paths
        ]
        reserved_paths = {destination, archive_path, staging_path, backup_path}
        if any(
            extracted_destination in reserved_paths
            for _, extracted_destination in extracted_replacements
        ):
            raise ValueError(
                "Tar archive member conflicts with an internal download path"
            )

        # An archive that is already at its destination stays where it is.
        moves = [] if archive_path == destination else [(archive_path, destination)]
        _replace_staged_paths(
            [*extracted_replacements, *moves],
            backup_path,
            keep=frozenset()
            if overwrite
            else {path for _, path in extracted_replacements},
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(backup_path, ignore_errors=True)


# The repository and folder of the public bucket that hold the STRING networks,
# named here and nowhere else.
_NETWORK_REPOSITORY = "cytearc_networks"
_NETWORK_FOLDER = "string_v12.5"

# What the hub raises when it cannot serve a file: an unreachable host, offline
# mode, a refused status, a failed transfer or a file that is not there.
_UNSERVED = (
    OSError,
    EntryNotFoundError,
    HfHubHTTPError,
    XetDownloadError,
    httpx2.TransportError,
)


@dataclass(frozen=True, slots=True)
class NetworkFile:
    """A network file that CyteArc downloads, as ``scripts/build_string_networks.py`` wrote it."""

    species: str
    taxid: int
    string_version: str
    name: str
    size: int
    sha256: str

    @property
    def remote_path(self) -> str:
        """The file's path in the ``cytearc_networks`` repository."""
        return f"{_NETWORK_FOLDER}/{self.name}"

    @property
    def uri(self) -> str:
        """Where the public bucket holds the file."""
        return f"hf://buckets/{_BUCKET_ID}/{_NETWORK_REPOSITORY}/{self.remote_path}"

    @property
    def label(self) -> str:
        return f"STRING v{self.string_version} {self.species} network"


#: The STRING v12.5 networks by species: protein links scored without STRING's
#: ``database`` channel, kept at a recombined score of at least 0.700.
STRING_NETWORKS: Mapping[str, NetworkFile] = MappingProxyType(
    {
        "human": NetworkFile(
            species="human",
            taxid=9606,
            string_version="12.5",
            name="human_9606.npz",
            size=781_609,
            sha256="8f038b97f824170679dff4aa4f8a3ad23fbf0f6709a4b9ae1b4966132388e89f",
        ),
        "mouse": NetworkFile(
            species="mouse",
            taxid=10090,
            string_version="12.5",
            name="mouse_10090.npz",
            size=526_992,
            sha256="ee2bad7fc335e23e7eb5fa51e6f40c08a126b7c919f839c8503b75ce28ff7e21",
        ),
    }
)


def string_network_file(
    species: str, *, network: str | os.PathLike[str] | None = None
) -> Path:
    """Return a STRING network file whose size and SHA-256 digest match its record.

    ``network`` names a local copy of the file, which is checked and returned.
    Without it, the copy cached at ``$XDG_CACHE_HOME/cytearc/networks`` is
    checked and returned, and the first call downloads it from the
    ``cytearc_networks`` repository of the public ``Nygen/cytebase`` bucket. A
    download is checked before it replaces anything, so a failed one leaves
    nothing in the cache. A cached file that no longer matches stays where it is
    and raises, since a file that changed in the cache is a fault to report, not
    one to repair unseen.

    Args:
        species: A key of ``STRING_NETWORKS``, ``"human"`` or ``"mouse"``.
        network: A local copy of the species' network file.

    Raises:
        TypeError: If ``species`` is not text or ``network`` is not a path.
        ValueError: If ``species`` has no network, or if a file's size or
            digest differs from its record.
        FileNotFoundError: If ``network`` is not a file, or if the network is
            not cached and could not be downloaded. The message names the file,
            its digest, the command that downloads it and ``network=``.
    """
    if not isinstance(species, str):
        raise TypeError(f"species must be text, got {type(species).__name__}")
    entry = STRING_NETWORKS.get(species)
    if entry is None:
        raise ValueError(
            f"No STRING network for species {species!r}. "
            f"Networks exist for: {', '.join(sorted(STRING_NETWORKS))}"
        )
    if network is not None:
        if not isinstance(network, str | os.PathLike):
            raise TypeError(f"network must be a path, got {type(network).__name__}")
        local = Path(network)
        if not local.is_file():
            raise FileNotFoundError(
                f"network={str(local)!r} is not a file. It must be a copy of "
                f"{entry.name}, the {entry.label} ({entry.size:,} bytes, SHA-256 "
                f"{entry.sha256})"
            )
        _check_network_file(local, entry, f"The file passed as network=, {local},")
        return local
    cached = cache_directory("networks") / entry.remote_path
    if cached.is_file():
        _check_network_file(
            cached,
            entry,
            f"The cached file {cached}",
            "Delete it to download the network again, or pass network= a copy "
            "that matches.",
        )
        return cached
    _download_network(entry, cached)
    return cached


def _check_network_file(
    path: Path, entry: NetworkFile, what: str, remedy: str = ""
) -> None:
    size = path.stat().st_size
    digest = file_sha256(path)
    if size == entry.size and digest == entry.sha256:
        return
    message = (
        f"{what} does not match the {entry.label}: expected {entry.size:,} bytes "
        f"with SHA-256 {entry.sha256}, found {size:,} bytes with SHA-256 {digest}."
    )
    raise ValueError(f"{message} {remedy}" if remedy else message)


def _download_network(entry: NetworkFile, destination: Path) -> None:
    """Download ``entry`` to ``destination`` once its size and digest match."""
    created = [
        directory
        for directory in (destination.parent, *destination.parent.parents)
        if not directory.exists()
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".cytearc-network-"))
    saved = False
    try:
        try:
            downloaded = connect(_NETWORK_REPOSITORY).download(
                entry.remote_path, destination=staging
            )
        except (KeyError, *_UNSERVED) as error:
            raise FileNotFoundError(
                _unavailable_message(entry, destination, error)
            ) from error
        if len(downloaded) != 1:
            raise FileNotFoundError(
                _unavailable_message(
                    entry,
                    destination,
                    f"{entry.uri} names {len(downloaded)} files rather than one",
                )
            )
        _check_network_file(
            downloaded[0],
            entry,
            f"The file downloaded from {entry.uri}",
            "Nothing was saved.",
        )
        os.replace(downloaded[0], destination)
        saved = True
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not saved:
            # Leave no directory behind that the failed download created.
            for directory in created:
                try:
                    directory.rmdir()
                except OSError:
                    break


def _unavailable_message(
    entry: NetworkFile, destination: Path, reason: BaseException | str
) -> str:
    if isinstance(reason, KeyError) and reason.args:
        reason = str(reason.args[0])
    return (
        f"The {entry.label} ({entry.name}, {entry.size:,} bytes, SHA-256 "
        f"{entry.sha256}) is not cached at {destination} and could not be "
        f"downloaded from {entry.uri}: {reason}. On a machine with network "
        "access, download it with "
        f'cytearc.datasets.connect("{_NETWORK_REPOSITORY}").download('
        f'"{entry.remote_path}", destination="<directory>"), then pass '
        f'network="<directory>/{entry.remote_path}".'
    )
