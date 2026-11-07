"""Downloads skip only when a host cannot serve a file whole, and a run names those skips."""

import gzip
import http.client
import os
import re
import socket
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.network_helpers import (
    NETWORK_SKIP,
    download_or_skip,
    git_blob_sha1,
    hub_or_skip,
    network_skips,
    pytest_terminal_summary,
)

_WHOLE = gzip.compress(b"BC\twtUMIs\tmutUMIs\n", mtime=0)


class _Response:
    """A host's answer that serves ``data`` under a stated ``length``, or whose read raises ``error``."""

    def __init__(
        self, data: bytes, length: int | None = None, error: Exception | None = None
    ) -> None:
        self._data, self._error = data, error
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        if self._error is not None:
            raise self._error
        return self._data


def _status(code: int) -> Exception:
    return urllib.error.HTTPError("https://host/file", code, "status", {}, None)  # type: ignore[arg-type]


def _cut(data: bytes) -> Exception:
    """What a read raises when the server closes the connection before the length it announced."""
    return http.client.IncompleteRead(data[:10], len(data))


@pytest.mark.parametrize(
    ("name", "answers", "outcome"),
    [
        ("table.tsv.gz", [_Response(_WHOLE, len(_WHOLE))], "written"),
        # A failed download is tried once more.
        ("table.tsv.gz", [_status(503), _Response(_WHOLE)], "written"),
        (
            "table.tsv.gz",
            [_Response(b"", error=_cut(_WHOLE)), _Response(_WHOLE)],
            "written",
        ),
        ("table.tsv.gz", [_status(503), _status(429)], "skipped"),
        ("table.tsv.gz", [_status(403), _status(408)], "skipped"),
        ("table.tsv.gz", [_status(500), _status(599)], "skipped"),
        ("table.tsv.gz", [_status(501), _status(502)], "skipped"),
        # A connection closed before its stated length, a body shorter than
        # its stated length, and a gzip stream that ends early without one.
        (
            "table.tsv.gz",
            [_Response(b"", error=_cut(_WHOLE)), _Response(b"", error=_cut(_WHOLE))],
            "skipped",
        ),
        ("table.tsv", [_Response(b"BC\t", 20), _Response(b"BC\t", 20)], "skipped"),
        (
            "table.tsv.gz",
            [_Response(_WHOLE[:10], len(_WHOLE)), _Response(_WHOLE[:10], len(_WHOLE))],
            "skipped",
        ),
        ("table.tsv.gz", [_Response(_WHOLE[:-8]), _Response(_WHOLE[:-8])], "skipped"),
        (
            "table.tsv.gz",
            [ConnectionResetError("reset"), TimeoutError("slow")],
            "skipped",
        ),
        (
            "table.tsv.gz",
            [
                urllib.error.URLError("unreachable"),
                urllib.error.URLError("unreachable"),
            ],
            "skipped",
        ),
        # A status that names a missing file fails at once, since skipping
        # would hide a file that moved.
        ("table.tsv.gz", [_status(404)], "failed"),
        ("table.tsv.gz", [_status(410)], "failed"),
        ("table.tsv.gz", [_status(503), _status(404)], "failed"),
        ("table.tsv.gz", [_status(401)], "failed"),
    ],
)
def test_downloads_skip_only_when_the_host_cannot_serve_a_file_whole(
    monkeypatch, tmp_path, name, answers, outcome
) -> None:
    pending = list(answers)

    def urlopen(url: str, timeout: float) -> _Response:
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    url = f"https://host/{name}"
    path = tmp_path / name
    if outcome == "written":
        assert download_or_skip(url, path) == path
        assert path.read_bytes() == _WHOLE
    elif outcome == "skipped":
        with pytest.raises(
            pytest.skip.Exception, match=f"^{re.escape(f'{NETWORK_SKIP} {url}: ')}"
        ):
            download_or_skip(url, path)
    else:
        with pytest.raises(urllib.error.HTTPError):
            download_or_skip(url, path)
    # Each answer was asked for, and only a whole file is left behind.
    assert not pending
    assert list(tmp_path.iterdir()) == ([path] if outcome == "written" else [])


@pytest.mark.parametrize(
    ("bodies", "outcome"),
    [
        ([_WHOLE], "written"),
        # A body of another blob is cut short: tried again, then skipped.
        ([_WHOLE[:-1] + b"\x01", _WHOLE], "written"),
        ([_WHOLE + b"\x00", _WHOLE[:-1] + b"\x01"], "skipped"),
    ],
)
def test_a_pinned_blob_is_written_only_when_its_bytes_are_the_blobs(
    monkeypatch, tmp_path, bodies, outcome
) -> None:
    pending = [_Response(body) for body in bodies]
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: pending.pop(0))
    # Not gzip by its name, so only the blob tells a changed body.
    path = tmp_path / "subsample.tsv"
    blob = git_blob_sha1(_WHOLE)
    if outcome == "written":
        assert (
            download_or_skip("https://host/subsample.tsv", path, git_blob=blob) == path
        )
        assert path.read_bytes() == _WHOLE
    else:
        with pytest.raises(pytest.skip.Exception, match="git blob"):
            download_or_skip("https://host/subsample.tsv", path, git_blob=blob)
    assert not pending
    assert list(tmp_path.iterdir()) == ([path] if outcome == "written" else [])


def test_git_blob_sha1_is_the_hash_git_gives_a_file(tmp_path) -> None:
    # The empty blob's hash, which every git repository knows, and one that
    # git hash-object gives these bytes.
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


class _Report:
    def __init__(self, nodeid: str, longrepr: object) -> None:
        self.nodeid, self.longrepr = nodeid, longrepr


class _Reporter:
    """A terminal reporter that keeps what a summary writes."""

    def __init__(self, skipped: list[_Report]) -> None:
        self.stats = {"skipped": skipped}
        self.lines: list[str] = []

    def section(self, title: str, sep: str = "=", **markup: bool) -> None:
        self.lines.append(f"[{title}]")

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def test_the_summary_names_only_the_tests_that_skipped_for_the_network() -> None:
    network = f"{NETWORK_SKIP} https://host/a.gz: <urlopen error unreachable>"
    reporter = _Reporter(
        [
            _Report(
                "tests/test_a.py::test_one",
                ("tests/test_a.py", 3, f"Skipped: {network}"),
            ),
            _Report(
                "tests/test_a.py::test_two",
                ("tests/test_a.py", 9, "Skipped: needs a GPU"),
            ),
            # A skip that pytest reports without a location.
            _Report("tests/test_a.py::test_three", None),
        ]
    )

    pytest_terminal_summary(reporter)

    assert reporter.lines == [
        "[tests skipped for the network]",
        f"tests/test_a.py::test_one: {network}",
    ]
    assert network_skips(reporter.stats["skipped"]) == [
        ("tests/test_a.py::test_one", network)
    ]
    # A run with no such skip writes nothing.
    quiet = _Reporter(reporter.stats["skipped"][1:])
    pytest_terminal_summary(quiet)
    assert quiet.lines == []


def test_a_run_ends_naming_each_test_that_skipped_for_the_network(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "test_downloads.py").write_text(
        textwrap.dedent(
            """
            import urllib.error
            import urllib.request

            import pytest

            from tests.network_helpers import download_or_skip


            def unreachable(url, timeout):
                raise urllib.error.URLError("unreachable")


            def test_unreachable(monkeypatch, tmp_path):
                monkeypatch.setattr(urllib.request, "urlopen", unreachable)
                download_or_skip("https://host/a.gz", tmp_path / "a.gz")


            def test_skipped_otherwise():
                pytest.skip("needs a GPU")
            """
        )
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.network_helpers",
            "-p",
            "no:cacheprovider",
            "test_downloads.py",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    summary = run.stdout[run.stdout.index("tests skipped for the network") :]
    assert "test_downloads.py::test_unreachable: " + NETWORK_SKIP in summary
    assert "test_skipped_otherwise" not in summary


def _hub_status(status: int) -> Exception:
    """The error that the Hugging Face client raises for an HTTP ``status``."""
    import httpx2
    from huggingface_hub.errors import HfHubHTTPError

    request = httpx2.Request("GET", "https://huggingface.co/api/buckets/Nygen/cytebase")
    return HfHubHTTPError(
        f"{status} for the bucket", response=httpx2.Response(status, request=request)
    )


def _unserved() -> list[Exception]:
    import httpx2
    from huggingface_hub.errors import OfflineModeIsEnabled, XetDownloadError

    return [
        httpx2.ConnectError("[Errno 111] Connection refused"),
        httpx2.ConnectError("[Errno -3] Temporary failure in name resolution"),
        httpx2.ReadTimeout("The read operation timed out"),
        OfflineModeIsEnabled("Cannot reach the bucket: offline mode is enabled"),
        XetDownloadError("The transfer of data.zarr.tar.gz failed"),
        _hub_status(429),
        _hub_status(503),
    ]


@pytest.mark.parametrize("index", range(7))
def test_the_hub_skips_a_test_when_it_cannot_serve_it(index: int) -> None:
    error = _unserved()[index]
    with pytest.raises(pytest.skip.Exception) as skipped:
        with hub_or_skip("the catalog of the bucket"):
            raise error
    assert str(skipped.value) == f"{NETWORK_SKIP} the catalog of the bucket: {error}"


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("the catalog lacks tenx_5K_pbmc_rnaseq"),
        OSError("Downloaded 3 bytes for 'a/data.h5', expected 4"),
        ValueError("Unsafe tar archive member: ../a"),
    ],
    ids=["assertion", "short download", "unsafe archive"],
)
def test_the_hub_fails_a_test_for_what_it_served(error: Exception) -> None:
    with pytest.raises(type(error), match=re.escape(str(error))):
        with hub_or_skip("the catalog of the bucket"):
            raise error


def test_the_hub_fails_a_test_for_a_bucket_or_file_that_moved() -> None:
    from huggingface_hub.errors import HfHubHTTPError

    with pytest.raises(HfHubHTTPError, match="404"):
        with hub_or_skip("the catalog of the bucket"):
            raise _hub_status(404)


def test_the_live_dataset_tests_skip_when_the_hub_is_unreachable() -> None:
    # A port that nothing listens on refuses the connection at once, so the
    # live tests meet an unreachable host without leaving the machine.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "HF_ENDPOINT": f"http://127.0.0.1:{port}"}
    env.pop("HF_HUB_OFFLINE", None)
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "integration",
            "tests/test_datasets.py",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert re.search(r"\b2 skipped\b", run.stdout), run.stdout
    summary = run.stdout[run.stdout.index("tests skipped for the network") :]
    for name in (
        "test_live_bucket_catalog_is_public",
        "test_live_zarr_archive_download",
    ):
        assert f"tests/test_datasets.py::{name}: {NETWORK_SKIP}" in summary
