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
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from huggingface_hub import BucketFile, BucketFolder
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import EntryNotFoundError

BUCKET_ID = "test/cytearc"
# The dataset repository that ``LoopbackHub`` serves at revision ``main``.
REPOSITORY_ID = "test/cytearc-data"


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
    # huggingface_hub adds the detected AI agent to its user agent, and detects
    # it from a registry that it fetches from the Hub when its cache file under
    # HF_HOME is missing or older than a day. The library reads its settings
    # when imported, so the constant is patched rather than the variable.
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_TELEMETRY", True)

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
    # huggingface_hub fixed its token files when imported, so a machine's login
    # would still be read, and an OAuth token near expiry refreshed over the
    # network and written back.
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(home / "hf_token"))
    monkeypatch.setattr(
        hf_constants, "HF_STORED_TOKENS_PATH", str(home / "hf_stored_tokens")
    )
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
        files: list[tuple[str | BucketFile, Path]],
        *,
        raise_on_missing_files: bool = False,
        token: str | bool | None = None,
    ) -> None:
        # Like Hugging Face's function, take each file as its path or its listing.
        files = [
            (remote.path if isinstance(remote, BucketFile) else remote, destination)
            for remote, destination in files
        ]
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


class LoopbackHub(ThreadingHTTPServer):
    """Serve a directory as the Hugging Face bucket ``BUCKET_ID`` on a loopback port.

    The same directory is also the dataset repository ``REPOSITORY_ID`` at
    revision ``main``, whose recursive listings name folders as well as files,
    as the Hub lists a repository.

    ``plan[kind]`` lists what the next requests of a kind meet, in order:
    ``"stall"`` holds a request unanswered until the hub closes, which ends it
    with status 410, ``"whole"`` answers a file request with the whole file
    and status 200 whatever range it asked for, as a server that ignores
    ranges does, and a status, or a status with headers, answers with that
    status. The kinds are
    ``"bucket"``, the bucket's lookup, ``"repository"``, the repository's,
    ``"tree"``, a listing, ``"paths-info"``, a lookup of named files of the
    bucket or the repository, and ``"file"``, a file's bytes. A request that
    its plan does not cover is served, and ``requests`` records the kind and
    path of each, the path of a lookup being the paths it names, joined by
    ``", "``. ``received`` records them with what else the request sent:
    ``"recursive"`` or ``"directory"`` for a listing, the ``Range`` header, or
    None, for a file, and the ``Authorization`` header, or None, for a lookup.
    ``authorizations`` records the kind and the ``Authorization`` header, or
    None, of every request.
    """

    daemon_threads = True

    def __init__(self, root: Path) -> None:
        super().__init__(("127.0.0.1", 0), _LoopbackHandler)
        self.root = root
        self.plan: dict[str, list[str | int | tuple[int, dict[str, str]]]] = {}
        self.requests: list[tuple[str, str]] = []
        self.received: list[tuple[str, str, str | None]] = []
        self.authorizations: list[tuple[str, str | None]] = []
        self.closing = threading.Event()
        self._lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take(
        self, kind: str, path: str, detail: str | None = None
    ) -> str | int | tuple[int, dict[str, str]]:
        with self._lock:
            self.requests.append((kind, path))
            self.received.append((kind, path, detail))
            planned = self.plan.get(kind)
            return planned.pop(0) if planned else "serve"


class _LoopbackHandler(BaseHTTPRequestHandler):
    server: LoopbackHub
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _answer(
        self,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _json(
        self, status: int, payload: Any, headers: dict[str, str] | None = None
    ) -> None:
        self._answer(status, json.dumps(payload).encode(), headers)

    def _planned(self, kind: str, path: str, detail: str | None) -> str | None:
        """Take the plan of a request: answer a planned status, or return the plan."""
        self.server.authorizations.append((kind, self.headers.get("Authorization")))
        planned = self.server.take(kind, path, detail)
        if planned == "stall":
            # Once the hub closes, end the request with a status that
            # huggingface_hub never retries, so that a client which gave up on
            # it sends nothing more.
            self.server.closing.wait(timeout=60)
            planned = 410
        if isinstance(planned, str):
            return planned
        status, headers = planned if isinstance(planned, tuple) else (planned, {})
        self._json(status, {"error": f"Planned status {status}"}, headers)
        return None

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        repository = url.path == f"/api/datasets/{REPOSITORY_ID}/paths-info/main"
        if not repository and url.path != f"/api/buckets/{BUCKET_ID}/paths-info":
            self._json(404, {"error": f"No route for {url.path}"})
            return
        try:
            # huggingface_hub sends a bucket's lookup as JSON and a
            # repository's form-encoded.
            paths = (
                parse_qs(body.decode(), strict_parsing=True)["paths"]
                if repository
                else json.loads(body)["paths"]
            )
        except (KeyError, ValueError):
            self._json(400, {"error": "The lookup names no paths"})
            return
        if (
            self._planned(
                "paths-info", ", ".join(paths), self.headers.get("Authorization")
            )
            is None
        ):
            return
        # The Hub's schemas for the answer: an entry for each path that names a
        # file or a folder, and none for a missing path. A repository names
        # each entry's git object.
        found: list[dict[str, Any]] = []
        for path in paths:
            target = self.server.root / path
            entry: dict[str, Any] = (
                {"path": path, "oid": "0" * 40}
                if repository
                else {"path": path, "uploadedAt": "2026-01-01T00:00:00.000Z"}
            )
            if target.is_file():
                entry |= {"type": "file", "size": target.stat().st_size}
                found.append(entry if repository else entry | {"xetHash": "test-hash"})
            elif target.is_dir():
                found.append(
                    entry
                    | (
                        {"type": "directory", "size": 0}
                        if repository
                        else {"type": "directory"}
                    )
                )
        self._json(200, found)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        api = f"/api/buckets/{BUCKET_ID}"
        files = f"/buckets/{BUCKET_ID}/resolve/"
        repository = f"/api/datasets/{REPOSITORY_ID}"
        repository_files = f"/datasets/{REPOSITORY_ID}/resolve/main/"
        in_repository = url.path.startswith((repository, repository_files))
        if url.path == api:
            kind, path = "bucket", ""
        elif url.path.startswith(f"{api}/tree"):
            kind, path = "tree", unquote(url.path.removeprefix(f"{api}/tree"))
        elif url.path.startswith(files):
            kind, path = "file", unquote(url.path.removeprefix(files))
        elif url.path in (repository, f"{repository}/revision/main"):
            kind, path = "repository", ""
        elif url.path.startswith(f"{repository}/tree/main"):
            kind = "tree"
            path = unquote(url.path.removeprefix(f"{repository}/tree/main"))
        elif url.path.startswith(repository_files):
            kind, path = "file", unquote(url.path.removeprefix(repository_files))
        else:
            self._json(404, {"error": f"No route for {url.path}"})
            return
        path = path.lstrip("/")
        recursive = parse_qs(url.query).get("recursive") == ["true"]
        detail = (
            ("recursive" if recursive else "directory")
            if kind == "tree"
            else self.headers.get("Range")
            if kind == "file"
            else None
        )
        planned = self._planned(kind, path, detail)
        if planned is None:
            return
        if planned == "whole" and kind == "file":
            self._file(path, whole=True)
            return
        if kind == "repository":
            self._json(200, {"id": REPOSITORY_ID, "sha": "0" * 40})
        elif kind == "bucket":
            self._json(
                200,
                {
                    "id": BUCKET_ID,
                    "private": False,
                    "createdAt": "2026-01-01T00:00:00.000Z",
                    "size": 0,
                    "totalFiles": 0,
                },
            )
        elif kind == "tree":
            self._tree(path, recursive, repository=in_repository)
        else:
            self._file(path)

    def _tree(self, prefix: str, recursive: bool, *, repository: bool) -> None:
        base = self.server.root
        if recursive:
            # The Hub matches a recursive listing's prefix as text, and lists
            # only the files of a bucket.
            entries = [
                path
                for path in sorted(base.rglob("*"))
                if (repository or path.is_file())
                and path.relative_to(base).as_posix().startswith(prefix)
            ]
        else:
            start = base / prefix
            entries = sorted(start.iterdir()) if start.is_dir() else []
        if not entries:
            self._json(404, {"error": "No entries"}, {"X-Error-Code": "EntryNotFound"})
            return
        # A repository names each entry's git object.
        identity = {"oid": "0" * 40} if repository else {}
        self._json(
            200,
            [
                {
                    "type": "file",
                    "path": path.relative_to(base).as_posix(),
                    "size": path.stat().st_size,
                    "xetHash": "test-hash",
                    **identity,
                }
                if path.is_file()
                else {
                    "type": "directory",
                    "path": path.relative_to(base).as_posix(),
                    **identity,
                }
                for path in entries
            ],
        )

    def _file(self, path: str, *, whole: bool = False) -> None:
        source = self.server.root / path
        if not source.is_file():
            self._json(404, {"error": "No file"}, {"X-Error-Code": "EntryNotFound"})
            return
        data = source.read_bytes()
        requested = self.headers.get("Range")
        if requested is None or whole:
            self._answer(200, data, content_type="application/octet-stream")
            return
        # As RFC 9110 answers a range from a start: a range past the end is
        # unsatisfiable, and one that runs past it ends at the end.
        first, _, last = requested.removeprefix("bytes=").partition("-")
        start = int(first)
        if start >= len(data):
            self._answer(
                416,
                b"",
                {"Content-Range": f"bytes */{len(data)}"},
                content_type="application/octet-stream",
            )
            return
        stop = min(int(last) + 1, len(data)) if last else len(data)
        self._answer(
            206,
            data[start:stop],
            {"Content-Range": f"bytes {start}-{stop - 1}/{len(data)}"},
            content_type="application/octet-stream",
        )


@pytest.fixture
def loopback_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[LoopbackHub]:
    """A running ``LoopbackHub`` that serves ``tmp_path / "bucket"``."""
    from huggingface_hub import constants

    # Requests to the hub must not depend on the environment's offline mode.
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    root = tmp_path / "bucket"
    root.mkdir()
    hub = LoopbackHub(root)
    # A short poll lets closing the hub return at once.
    serving = threading.Thread(
        target=hub.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    serving.start()
    try:
        yield hub
    finally:
        hub.closing.set()
        hub.shutdown()
        hub.server_close()
        serving.join()
