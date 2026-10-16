"""Offline Hugging Face fakes for CyteArc's bucket and remote-store tests.

Nothing here is autouse. Test modules opt in with
``pytestmark = pytest.mark.usefixtures("hub_offline")``, which blocks
non-loopback network access and isolates credentials and the home directory.
"""

import ipaddress
import json
import os
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import BucketFile, BucketFolder
from huggingface_hub.errors import EntryNotFoundError

BUCKET_ID = "test/cytearc"


class NetworkAccessBlocked(BaseException):
    """Escapes ``retry()`` and ``except Exception`` so network use fails loudly."""


def _loopback(host: object) -> bool:
    if host in (None, "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture
def hub_offline(monkeypatch, tmp_path):
    """Block remote traffic and isolate Hugging Face credentials and home."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(
            address[0]
        ):
            raise NetworkAccessBlocked(f"Offline test tried to connect to {address!r}")

    def connect(sock: socket.socket, address: Any) -> None:
        check(sock, address)
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        check(sock, address)
        return real_connect_ex(sock, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _loopback(host):
            raise NetworkAccessBlocked(f"Offline test tried to resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CYTEARC_CYTEBASE_LOCAL",
    ):
        monkeypatch.delenv(name, raising=False)
    # Keep matplotlib's existing font cache while HOME points elsewhere.
    real_home = Path.home()
    for name, default in (
        ("XDG_CACHE_HOME", real_home / ".cache"),
        ("XDG_CONFIG_HOME", real_home / ".config"),
    ):
        if not os.environ.get(name):
            monkeypatch.setenv(name, str(default))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def bucket_file(path: str, size: int) -> BucketFile:
    return BucketFile(
        type="file",
        path=path,
        size=size,
        xetHash="test-hash",
        mtime=None,
        uploadedAt=None,
    )


def bucket_folder(path: str) -> BucketFolder:
    return BucketFolder(type="directory", path=path, uploadedAt=None)


class FakeHub:
    """Directory-backed stand-in for the Hugging Face bucket functions.

    Each bucket is a directory under ``root``. ``hooks[name]`` callbacks run
    before the named operation with its arguments.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[Any, ...]] = []
        self.hooks: dict[str, list[Callable[..., None]]] = {}

    def directory(self, bucket_id: str = BUCKET_ID) -> Path:
        path = self.root / bucket_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, remote: str, bucket_id: str = BUCKET_ID) -> Path:
        return self.directory(bucket_id) / remote

    def put(
        self, remote: str, data: bytes | str | dict, bucket_id: str = BUCKET_ID
    ) -> None:
        if isinstance(data, dict):
            data = json.dumps(data)
        target = self.path(remote, bucket_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data.encode() if isinstance(data, str) else data)
        os.replace(temporary, target)

    def read(self, remote: str, bucket_id: str = BUCKET_ID) -> bytes:
        return self.path(remote, bucket_id).read_bytes()

    def read_json(self, remote: str, bucket_id: str = BUCKET_ID) -> Any:
        return json.loads(self.read(remote, bucket_id))

    def files(self, bucket_id: str = BUCKET_ID) -> list[str]:
        base = self.directory(bucket_id)
        return sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )

    def _run_hooks(self, name: str, *args: Any) -> None:
        for hook in self.hooks.get(name, []):
            hook(*args)

    def list_bucket_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        *,
        recursive: bool | None = None,
        token: str | bool | None = None,
    ) -> list[BucketFile | BucketFolder]:
        self.calls.append(("list_bucket_tree", bucket_id, prefix, recursive, token))
        self._run_hooks("list_bucket_tree", bucket_id, prefix)
        base = self.directory(bucket_id)
        if recursive:
            # Object stores match prefixes as strings, including sibling names.
            entries = [
                path
                for path in sorted(base.rglob("*"))
                if path.relative_to(base).as_posix().startswith(prefix or "")
            ]
        else:
            start = base / prefix.rstrip("/") if prefix else base
            entries = sorted(start.iterdir()) if start.is_dir() else []
        if not entries:
            raise EntryNotFoundError(f"No bucket entries under {prefix}")
        return [
            bucket_file(path.relative_to(base).as_posix(), path.stat().st_size)
            if path.is_file()
            else bucket_folder(path.relative_to(base).as_posix())
            for path in entries
        ]

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool = False,
        token: str | bool | None = None,
    ) -> None:
        remotes = [remote for remote, _ in files]
        self.calls.append(("download_bucket_files", bucket_id, remotes, token))
        for remote in remotes:
            self._run_hooks("download_bucket_files", bucket_id, remote)
        missing = [
            remote for remote in remotes if not self.path(remote, bucket_id).is_file()
        ]
        if missing and raise_on_missing_files:
            raise EntryNotFoundError(f"Missing bucket files: {', '.join(missing)}")
        for remote, destination in files:
            source = self.path(remote, bucket_id)
            if source.is_file():
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes | Path, str]] | None = None,
        copy: list | None = None,
        delete: list[str] | None = None,
        token: str | bool | None = None,
    ) -> None:
        added = [remote for _, remote in add or []]
        self.calls.append(
            ("batch_bucket_files", bucket_id, added, list(delete or []), token)
        )
        self._run_hooks("batch_bucket_files", bucket_id, added, list(delete or []))
        for source, remote in add or []:
            data = source if isinstance(source, bytes) else Path(source).read_bytes()
            self.put(remote, data, bucket_id)
        for remote in delete or []:
            self.path(remote, bucket_id).unlink(missing_ok=True)

    def sync_bucket(
        self,
        source: str,
        dest: str,
        *,
        delete: bool = False,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(("sync_bucket", source, dest, delete, token))
        self._run_hooks("sync_bucket", source, dest)
        if dest.startswith("hf://buckets/"):
            namespace, name, *rest = dest.removeprefix("hf://buckets/").split("/")
            target = self.directory(f"{namespace}/{name}").joinpath(*rest)
        else:
            target = Path(dest)
        shutil.copytree(source, target, dirs_exist_ok=True)
