"""Download the files of tests marked integration, or skip a test when its host cannot serve them.

A download that a host cannot serve whole, whether unreachable, slow, cut
short or refused for the moment, skips its test, and the run ends naming every
test that skipped so. A status that names a missing file fails the test,
since skipping it would hide a file that moved.
"""

import gzip
import hashlib
import http.client
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

#: The start of the reason of every skip for the network, which the summary at
#: the end of a run looks for.
NETWORK_SKIP = "The network could not serve a file whole:"


class _OtherBlob(Exception):
    """A body whose git blob is not the one its pinned commit names."""


# What a download raises when its host is unreachable, slow or cuts the file
# short: IncompleteRead is an HTTPException rather than an OSError, and a gzip
# stream that ends early raises EOFError.
_NOT_SERVED_WHOLE = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
    EOFError,
    _OtherBlob,
)


def _transient(status: int) -> bool:
    """Whether an HTTP status is a host's refusal or failure for the moment, as GEO's under load."""
    return status in (403, 408, 429) or 500 <= status <= 599


def git_blob_sha1(data: bytes) -> str:
    """The SHA-1 of a file's git blob, as a repository's tree at a commit names it."""
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324


def _fetched(url: str, git_blob: str | None) -> bytes:
    """The bytes at ``url``, refusing a body shorter than its stated length, a gzip file that ends early and a body of another git blob."""
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
        length = response.headers.get("Content-Length")
    if length is not None and len(data) != int(length):
        raise http.client.IncompleteRead(data, int(length) - len(data))
    if url.endswith(".gz"):
        gzip.decompress(data)
    if git_blob is not None and git_blob_sha1(data) != git_blob:
        raise _OtherBlob(
            f"the body's git blob is {git_blob_sha1(data)}, not {git_blob}"
        )
    return data


def download_or_skip(url: str, path: Path, *, git_blob: str | None = None) -> Path:
    """Download a file for a test marked integration, or skip the test when its host cannot serve the file whole.

    A failed download is tried once more, and the file is written only once
    whole, under a name of its own and then renamed. Unreachable hosts,
    timeouts, bodies cut short, statuses 403, 408 and 429 and every 5xx
    status skip the test after the second try; any other error status, such
    as 404 for a file that moved, fails it. ``git_blob``, the git blob SHA-1
    of a file at a pinned commit, makes a body of another blob one cut short.
    """
    failure: Exception | None = None
    for _attempt in range(2):
        try:
            data = _fetched(url, git_blob)
        except urllib.error.HTTPError as error:
            if not _transient(error.code):
                raise
            failure = error
            continue
        except _NOT_SERVED_WHOLE as error:
            failure = error
            continue
        partial = path.with_name(f"{path.name}.partial")
        partial.write_bytes(data)
        partial.replace(path)
        return path
    pytest.skip(f"{NETWORK_SKIP} {url}: {failure}")


def network_skips(reports: Iterable[Any]) -> list[tuple[str, str]]:
    """The test IDs and reasons of the skipped reports that skipped for the network."""
    found = []
    for report in reports:
        longrepr = getattr(report, "longrepr", None)
        if not (isinstance(longrepr, tuple) and len(longrepr) == 3):
            continue
        reason = str(longrepr[2]).removeprefix("Skipped: ")
        if reason.startswith(NETWORK_SKIP):
            found.append((report.nodeid, reason))
    return found


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """End a run with a warning that names each test skipped for the network."""
    skipped = network_skips(terminalreporter.stats.get("skipped", []))
    if not skipped:
        return
    terminalreporter.section("tests skipped for the network", sep="!", yellow=True)
    for nodeid, reason in skipped:
        terminalreporter.write_line(f"{nodeid}: {reason}")
