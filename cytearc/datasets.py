"""Download CyteArc's public example datasets and the networks its operations read.

The examples live in repositories of the public ``Nygen/cytebase`` Hugging Face
bucket, such as ``cytearc_docs`` for the documentation datasets, and the STRING
networks in its ``cytearc_networks`` repository. This module only lists and
downloads files. The CyteBase catalog of cloud-hosted datasets is the separate
``cytebase`` package.
"""

import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

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
_LOCAL_CATALOG_ENV = "CYTEARC_CYTEBASE_LOCAL"


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
    ) -> Path:
        """Download one dataset, optionally selecting its Zarr archive.

        Raises:
            TypeError: If ``zarr`` is not a boolean.
            KeyError: If the repository holds no such dataset.
            FileNotFoundError: If ``zarr`` is set and the dataset has no Zarr
                archive.
        """
        zarr = bool_argument(zarr, "zarr")
        name = _safe_name(name, kind="dataset name")
        destination = Path(destination).absolute()
        if zarr:
            local_dataset = _local_dataset_store(name)
            if local_dataset is not None:
                return _copy_local_dataset(local_dataset, destination / name)
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
                file for file in files if not file.path.endswith(_ZARR_ARCHIVE_SUFFIX)
            ]

        _download_files(self.name, selected, destination)
        return destination / name


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


def _copy_local_dataset(store: Path, target: Path) -> Path:
    """Replace only ``data.zarr`` in ``target``, keeping any other files."""
    target.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(dir=target, prefix=".data.zarr.copy-"))
    backup_path = Path(tempfile.mkdtemp(dir=target, prefix=".data.zarr.backup-"))
    try:
        staged = staging_path / "data.zarr"
        shutil.copytree(store, staged)
        _replace_staged_paths([(staged, target / "data.zarr")], backup_path)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(backup_path, ignore_errors=True)
    return target


def _download_files(
    repository: str,
    files: list[BucketFile],
    destination: Path,
) -> list[Path]:
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
            if final_file.name.endswith(".tar.gz"):
                _extract_and_replace(staged_file, final_file)
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


def _replace_staged_paths(
    staged_paths: list[tuple[Path, Path]],
    backup_path: Path,
) -> None:
    replacements: list[tuple[Path, Path | None]] = []

    try:
        for index, (source, destination) in enumerate(staged_paths):
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
) -> None:
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

        extracted_paths = sorted(staging_path.iterdir(), key=lambda path: path.name)
        output_name = destination.name.removesuffix(".tar.gz")
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

        _replace_staged_paths(
            [
                *extracted_replacements,
                (archive_path, destination),
            ],
            backup_path,
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
