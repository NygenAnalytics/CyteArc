import json
from types import SimpleNamespace

import pytest

from tests import shards


def test_every_file_lands_in_exactly_one_shard() -> None:
    files = [f"tests/test_{index}.py" for index in range(17)]
    seconds = {name: float(index % 5) for index, name in enumerate(files)}

    assigned = shards.assign_shards(shards.file_weights(files, seconds), 4)
    reversed_weights = shards.file_weights(list(reversed(files)), seconds)

    assert sorted(name for shard in assigned for name in shard) == sorted(files)
    assert assigned == shards.assign_shards(reversed_weights, 4)


def test_heaviest_files_are_packed_onto_the_lightest_shard() -> None:
    weights = {"a": 10.0, "b": 6.0, "c": 5.0, "d": 4.0}

    assigned = shards.assign_shards(weights, 2)

    assert assigned == [frozenset({"a", "d"}), frozenset({"b", "c"})]


def test_an_unrecorded_file_weighs_the_mean_plus_its_import() -> None:
    weights = shards.file_weights(["a", "b", "new"], {"a": 9.0, "b": 3.0})

    assert weights == {"a": 9.5, "b": 3.5, "new": 6.5}


def test_files_that_record_no_time_still_spread_across_shards() -> None:
    files = [f"tests/test_{index}.py" for index in range(6)]

    weights = shards.file_weights(files, dict.fromkeys(files, 0.0))

    assert [len(shard) for shard in shards.assign_shards(weights, 3)] == [2, 2, 2]


def test_a_shard_runs_its_heaviest_files_first_keeping_each_file_in_order() -> None:
    config = SimpleNamespace(stash=pytest.Stash())
    config.stash[shards._FILE_WEIGHTS] = {
        "tests/test_light.py": 1.0,
        "tests/test_heavy.py": 9.0,
    }
    items = [
        SimpleNamespace(nodeid=nodeid)
        for nodeid in (
            "tests/test_light.py::test_a",
            "tests/test_light.py::test_b",
            "tests/test_heavy.py::test_c",
            "tests/test_heavy.py::test_d",
        )
    ]

    shards.pytest_collection_modifyitems(config, items)

    assert [item.nodeid for item in items] == [
        "tests/test_heavy.py::test_c",
        "tests/test_heavy.py::test_d",
        "tests/test_light.py::test_a",
        "tests/test_light.py::test_b",
    ]


@pytest.mark.parametrize("value", ["2", "0/3", "4/3", "a/3", "1/", "-1/3"])
def test_malformed_shards_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="--shard"):
        shards.parse_shard(value)


def test_recorded_durations_update_files_that_ran_and_drop_deleted_files(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "tests").mkdir()
    for name in ("test_ran.py", "test_kept.py"):
        (tmp_path / "tests" / name).write_text("")
    durations = tmp_path / "shard_durations.json"
    durations.write_text(
        json.dumps(
            {
                "tests/test_deleted.py": 4.0,
                "tests/test_kept.py": 2.0,
                "tests/test_ran.py": 1.0,
            }
        )
    )
    monkeypatch.setattr(shards, "DURATIONS_PATH", durations)
    recorder = shards._DurationRecorder(tmp_path)

    for phase_seconds in (0.25, 3.0, 0.5):
        recorder.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="tests/test_ran.py::test_case[1]", duration=phase_seconds
            )
        )
    recorder.pytest_sessionfinish()

    assert json.loads(durations.read_text()) == {
        "tests/test_kept.py": 2.0,
        "tests/test_ran.py": 3.8,
    }
