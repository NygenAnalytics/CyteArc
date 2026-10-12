"""Split the test files into balanced shards for parallel CI jobs.

``--shard=I/N`` collects, from the test directories, the files that
longest-first packing assigns to shard I of N. Each file weighs its recorded
seconds in ``tests/shard_durations.json``, or the mean when it has no record,
plus a fixed import cost. Files outside the shard are never imported, so a
shard also collects only its own files. The shard runs its heaviest files
first, so that a long file does not start late and finish after the rest.

``--store-shard-durations`` records the seconds of every file that ran into the
same file, keeping the records of files that did not run.
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

DURATIONS_PATH = Path(__file__).with_name("shard_durations.json")

# Importing and collecting a file costs time even when all its tests skip.
_FILE_IMPORT_SECONDS = 0.5

_SHARD_FILES = pytest.StashKey[frozenset[str]]()
_FILE_WEIGHTS = pytest.StashKey[dict[str, float]]()


def parse_shard(value: str) -> tuple[int, int]:
    index, sep, count = value.partition("/")
    if not sep or not index.isdigit() or not count.isdigit():
        raise ValueError(f"--shard expects I/N, got {value!r}")
    shard, shards = int(index), int(count)
    if not 1 <= shard <= shards:
        raise ValueError(f"--shard {value!r} needs 1 <= I <= N")
    return shard, shards


def file_weights(files: list[str], seconds: dict[str, float]) -> dict[str, float]:
    recorded = [seconds[name] for name in files if name in seconds]
    default = sum(recorded) / len(recorded) if recorded else 1.0
    return {name: seconds.get(name, default) + _FILE_IMPORT_SECONDS for name in files}


def assign_shards(weights: dict[str, float], shards: int) -> list[frozenset[str]]:
    """Pack files into shards, heaviest first, onto the lightest shard."""
    loads = [0.0] * shards
    members: list[set[str]] = [set() for _ in range(shards)]
    for name in sorted(weights, key=lambda name: (-weights[name], name)):
        target = min(range(shards), key=lambda index: (loads[index], index))
        members[target].add(name)
        loads[target] += weights[name]
    return [frozenset(names) for names in members]


def _test_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests").rglob("test_*.py")
    )


def _read_durations() -> dict[str, float]:
    if not DURATIONS_PATH.is_file():
        return {}
    return json.loads(DURATIONS_PATH.read_text())


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shards")
    group.addoption(
        "--shard",
        metavar="I/N",
        help="Run shard I of N test files, balanced by tests/shard_durations.json.",
    )
    group.addoption(
        "--store-shard-durations",
        action="store_true",
        help="Record the seconds of each test file in tests/shard_durations.json.",
    )


def pytest_configure(config: pytest.Config) -> None:
    value = config.getoption("shard")
    if value is not None:
        try:
            shard, shards = parse_shard(value)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from None
        weights = file_weights(_test_files(config.rootpath), _read_durations())
        config.stash[_FILE_WEIGHTS] = weights
        config.stash[_SHARD_FILES] = assign_shards(weights, shards)[shard - 1]
        # xdist hands out files in collection order unless it reorders them by
        # test count; keep the duration order that the collection hook sets.
        if hasattr(config.option, "loadscopereorder"):
            config.option.loadscopereorder = False
    # xdist workers send their reports to the controller, which records them.
    if config.getoption("store_shard_durations") and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_DurationRecorder(config.rootpath))


def pytest_report_header(config: pytest.Config) -> str | None:
    selected = config.stash.get(_SHARD_FILES, None)
    if selected is None:
        return None
    return f"shard {config.getoption('shard')}: {len(selected)} test files"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    selected = config.stash.get(_SHARD_FILES, None)
    if selected is None or not collection_path.name.startswith("test_"):
        return None
    if collection_path.suffix != ".py":
        return None
    if not collection_path.is_relative_to(config.rootpath):
        return None
    name = collection_path.relative_to(config.rootpath).as_posix()
    return True if name not in selected else None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    weights = config.stash.get(_FILE_WEIGHTS, None)
    if weights is None:
        return
    # A stable sort keeps the tests of each file together and in their order.
    items.sort(key=lambda item: -weights.get(item.nodeid.split("::", 1)[0], 0.0))


class _DurationRecorder:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._seconds: defaultdict[str, float] = defaultdict(float)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._seconds[report.nodeid.split("::", 1)[0]] += report.duration

    def pytest_sessionfinish(self) -> None:
        if not self._seconds:
            return
        merged = _read_durations() | {
            name: round(seconds, 1) for name, seconds in self._seconds.items()
        }
        existing = {
            name: seconds
            for name, seconds in sorted(merged.items())
            if (self._root / name).is_file()
        }
        DURATIONS_PATH.write_text(json.dumps(existing, indent=2) + "\n")
