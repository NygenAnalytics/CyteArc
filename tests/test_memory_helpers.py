"""The memory test helpers find the budgets that calls accept and fail the checks a wrong estimate breaks."""

import json
import subprocess
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as pool_thread
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from tests.memory_helpers import (
    ZARR_READS,
    budget_bounds,
    budget_refusals,
    budget_that_runs,
    check_budget,
    cost_bound,
    hold_finished_reads,
    named_budget,
    smallest_budget,
    stated_bytes,
    traced_peak,
)
from tests.storage_helpers import open_store, write_count_store

MIB = 1 << 20


def test_named_budget_reads_the_budget_that_a_refusal_closes_with() -> None:
    assert (
        named_budget(
            MemoryError(
                "Counting guides holds about 5000 bytes, more than the memory budget "
                "of 100 bytes. Raise mem_budget to at least 5000 bytes."
            )
        )
        == 5000
    )
    assert (
        named_budget(
            MemoryError(
                "Labelling holds about 812 bytes. Raise mem_budget to at least 812 "
                "bytes, or pass a smaller cell_selection."
            )
        )
        == 812
    )
    # A refusal that names no budget asks for the bytes it holds.
    assert (
        named_budget(MemoryError("A batch holds about 77 bytes. Raise mem_budget."))
        == 77
    )


@pytest.mark.parametrize(
    "message",
    [
        # A budget other than the bytes the call holds.
        "Labelling holds about 812 bytes. Raise mem_budget to at least 900 bytes.",
        # No closing budget.
        "Labelling holds about 812 bytes, more than the budget.",
        # A closing request that says nothing of what the call holds.
        "Labelling does not fit. Raise mem_budget.",
    ],
)
def test_named_budget_refuses_a_refusal_that_names_no_budget_it_holds(
    message: str,
) -> None:
    with pytest.raises(AssertionError):
        named_budget(MemoryError(message))


def test_stated_bytes_reads_the_bytes_that_a_refusal_states() -> None:
    assert stated_bytes(MemoryError("Grouping holds about 4096 bytes, more")) == 4096
    assert stated_bytes(MemoryError("The search needs about 12 bytes, more")) == 12
    # The first bytes stated, before any that follow.
    assert (
        stated_bytes(MemoryError("A holds about 30 bytes after about 20 bytes")) == 30
    )
    with pytest.raises(AssertionError, match="states no bytes"):
        stated_bytes(MemoryError("Grouping does not fit the memory budget"))


def test_traced_peak_counts_what_the_call_holds_beside_what_was_held() -> None:
    held = np.ones(4 * MIB, dtype=np.uint8)
    _, peak = traced_peak(lambda: np.ones(MIB, dtype=np.uint8).size, 2 * MIB)
    assert MIB <= peak < MIB + 64 * 1024
    del held


def test_traced_peak_traces_again_a_peak_above_the_budget() -> None:
    runs: list[int] = []

    def first_run_grows_a_table() -> int:
        # The first run holds 8 MiB more, once, as Python's table of interned
        # strings grows in one allocation.
        runs.append(1)
        once = np.ones(8 * MIB, dtype=np.uint8) if len(runs) == 1 else None
        block = np.ones(MIB, dtype=np.uint8)
        del once
        return len(runs) + int(block[0]) - 1

    result, peak = traced_peak(first_run_grows_a_table, 2 * MIB)
    assert len(runs) == 2 and result == 2
    assert MIB <= peak < 2 * MIB
    # A call within its budget runs once.
    runs.clear()
    traced_peak(first_run_grows_a_table, 16 * MIB)
    assert len(runs) == 1


def test_traced_peak_keeps_a_peak_that_passes_the_budget_in_both_runs() -> None:
    runs: list[int] = []

    def always_over() -> None:
        runs.append(1)
        np.ones(4 * MIB, dtype=np.uint8).size

    _, peak = traced_peak(always_over, MIB)
    assert len(runs) == 2
    assert peak >= 4 * MIB


class _ScriptedStore:
    """A store whose operation checks each step's bytes against ``memoryBytes``, as an operation does.

    ``needed(budget)`` gives the bytes of each step at a budget. A step that
    does not fit refuses in master's form, naming the bytes it holds, and the
    call that fits holds ``holds`` bytes and saves an artifact.
    """

    def __init__(self, needed, holds: int, *, writes_when_refused: bool = False):
        self.memoryBytes = 1 << 40
        self.needed, self.holds = needed, holds
        self.writes_when_refused = writes_when_refused
        self.saved: list[str] = []
        self.budgets: list[int] = []
        self.pipeline = SimpleNamespace(list_runs=lambda limit: [])
        self.artifacts = SimpleNamespace(list=lambda **options: list(self.saved))
        self.assay_names: list[str] = []

    def run(self, store: "_ScriptedStore") -> int:
        assert store is self
        self.budgets.append(self.memoryBytes)
        for step, needed in enumerate(self.needed(self.memoryBytes)):
            if needed > self.memoryBytes:
                if self.writes_when_refused:
                    self.saved.append(f"partial {len(self.saved)}")
                raise MemoryError(
                    f"Step {step} holds about {needed} bytes, more than the memory "
                    f"budget of {self.memoryBytes} bytes. Raise mem_budget to at least "
                    f"{needed} bytes."
                )
        block = np.ones(self.holds, dtype=np.uint8)
        self.saved.append(f"result {len(self.saved)}")
        return int(block.size)


def test_budget_that_runs_follows_each_refusal_to_the_budget_that_runs() -> None:
    store = _ScriptedStore(lambda budget: [3 * MIB, 5 * MIB, 4 * MIB], holds=2 * MIB)

    budget, refusals, result, peak = budget_that_runs(store, store.run)

    assert budget == 5 * MIB
    assert [refusal.split(" holds")[0] for refusal in refusals] == ["Step 0", "Step 1"]
    assert store.budgets == [1, 3 * MIB, 5 * MIB]
    assert result == 2 * MIB and 2 * MIB <= peak < 3 * MIB
    # The store gets its own budget back.
    assert store.memoryBytes == 1 << 40


def test_budget_that_runs_reads_the_bytes_that_stack_refusals_state() -> None:
    class Stated(_ScriptedStore):
        def run(self, store: "_ScriptedStore") -> int:
            try:
                return super().run(store)
            except MemoryError as refusal:
                # The words of the clone, repertoire and variant refusals.
                raise MemoryError(str(refusal).split(". Raise")[0] + ".") from None

    store = Stated(lambda budget: [3 * MIB], holds=MIB)
    with pytest.raises(AssertionError, match="does not end naming a budget"):
        budget_that_runs(store, store.run)
    budget, refusals, _, _ = budget_that_runs(
        store, store.run, refusal_budget=stated_bytes
    )
    assert budget == 3 * MIB and len(refusals) == 1


def test_budget_that_runs_fails_a_refusal_that_asks_for_no_more() -> None:
    store = _ScriptedStore(lambda budget: [], holds=MIB)

    def refuse(store: _ScriptedStore) -> None:
        # A refusal of the budget it names.
        needed = max(store.memoryBytes, 100)
        raise MemoryError(
            f"Step 0 holds about {needed} bytes. Raise mem_budget to at least "
            f"{needed} bytes."
        )

    with pytest.raises(AssertionError, match="names no budget above 100"):
        budget_that_runs(store, refuse)
    assert store.memoryBytes == 1 << 40


def test_budget_that_runs_gives_up_after_ten_refusals_or_the_number_given() -> None:
    # Each budget makes the next step need one byte more.
    store = _ScriptedStore(lambda budget: [budget + 1], holds=MIB)
    with pytest.raises(AssertionError, match="^10 refusals named no budget"):
        budget_that_runs(store, store.run)
    assert len(store.budgets) == 10
    store.budgets.clear()
    with pytest.raises(AssertionError, match="^25 refusals named no budget"):
        budget_that_runs(store, store.run, max_refusals=25)
    assert len(store.budgets) == 25
    # A step whose estimate grows with the budget settles after 15 refusals.
    store = _ScriptedStore(
        lambda budget: [min(budget + 1, 16) if budget < 16 else 0], holds=1000
    )
    with pytest.raises(AssertionError):
        budget_that_runs(store, store.run)
    budget, refusals, _, _ = budget_that_runs(store, store.run, max_refusals=16)
    assert (budget, len(refusals)) == (16, 15)


def test_budget_refusals_name_each_step_and_its_bytes() -> None:
    def needed(budget: int) -> list[int]:
        # The second step's estimate grows with the budget until 3,000 bytes.
        return [500, budget + 100 if budget < 3000 else 0]

    store = _ScriptedStore(needed, holds=1000)
    refusals = budget_refusals(store, lambda: store.run(store))
    assert refusals == [("Step 0", 500)] + [
        ("Step 1", stated) for stated in range(600, 3100, 100)
    ]
    assert store.memoryBytes == 1 << 40


def test_a_path_is_opened_at_each_budget_with_the_options_given(tmp_path) -> None:
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((3, 2), dtype=np.uint16)},
        "uint16",
    )
    opened: list[tuple[int, int]] = []

    def call(store) -> int:
        opened.append((store.memoryBytes, store.nthreads))
        if store.memoryBytes < 5000:
            raise MemoryError(
                "Reading holds about 5000 bytes. Raise mem_budget to at least 5000 "
                "bytes."
            )
        return 1

    budget, refusals, result, _ = budget_that_runs(
        tmp_path / "store.zarr", call, nthreads=2
    )
    assert (budget, len(refusals), result) == (5000, 1, 1)
    assert opened == [(1, 2), (5000, 2)]
    with pytest.raises(AssertionError, match="apply only to a store's path"):
        budget_that_runs(open_store(tmp_path / "store.zarr"), call, nthreads=2)


def test_a_call_that_takes_its_budget_follows_its_refusals_too() -> None:
    def read(budget: int) -> int:
        # A domain function that takes its memory budget as an argument.
        if budget < 7000:
            raise MemoryError(
                f"Reading holds about {min(budget * 5, 7000)} bytes, more"
            )
        return budget

    budget, refusals, result, _ = budget_that_runs(
        lambda budget: budget, read, refusal_budget=stated_bytes
    )
    # Budgets of 1, 5, 25, 125, 625 and 3,125 bytes are refused.
    assert (budget, len(refusals), result) == (7000, 6, 7000)
    with pytest.raises(AssertionError, match="no path, open store or function"):
        budget_that_runs(5, read)


def test_check_budget_passes_a_call_that_holds_its_budget_and_refuses_below_it() -> (
    None
):
    store = _ScriptedStore(lambda budget: [3 * MIB], holds=2 * MIB)
    budget, refusals, result = check_budget(store, store.run)
    assert (budget, len(refusals), result) == (3 * MIB, 1, 2 * MIB)
    # One byte below the budget was refused.
    assert store.budgets[-1] == 3 * MIB - 1
    assert store.memoryBytes == 1 << 40


def test_check_budget_fails_a_call_that_holds_more_than_its_budget() -> None:
    store = _ScriptedStore(lambda budget: [MIB], holds=2 * MIB)
    with pytest.raises(AssertionError, match="held .* bytes within a budget of"):
        check_budget(store, store.run)


def test_check_budget_fails_a_budget_that_one_byte_less_does_not_name() -> None:
    # Below half a MiB the step names 1 MiB, which runs, and just below 1 MiB
    # it names more, as a step whose chunks grow with the budget could.
    store = _ScriptedStore(
        lambda budget: [MIB if budget < MIB // 2 else 2 * MIB if budget < MIB else 0],
        holds=1000,
    )
    with pytest.raises(AssertionError, match="was refused naming 2097152"):
        check_budget(store, store.run)


def test_check_budget_fails_a_refusal_that_writes() -> None:
    store = _ScriptedStore(
        lambda budget: [3 * MIB], holds=MIB, writes_when_refused=True
    )
    with pytest.raises(AssertionError, match="changed what the store holds"):
        check_budget(store, store.run)
    # Without the check of writes, the same call passes.
    store = _ScriptedStore(
        lambda budget: [3 * MIB], holds=MIB, writes_when_refused=True
    )
    check_budget(store, store.run, check_writes=False)


def test_at_most_fails_an_estimate_inflated_beyond_the_traced_peak() -> None:
    # The refusal asks for 8 MiB to hold 1 MiB.
    store = _ScriptedStore(lambda budget: [8 * MIB], holds=MIB)
    check_budget(store, store.run)
    check_budget(store, store.run, at_most=8.5)
    with pytest.raises(AssertionError, match="more than 2 times its traced peak"):
        check_budget(store, store.run, at_most=2)
    # A fixed allowance beside the multiple covers what no trace shows.
    check_budget(store, store.run, at_most=2, allowance=7 * MIB)
    with pytest.raises(AssertionError, match="and 5242880 bytes besides"):
        check_budget(store, store.run, at_most=2, allowance=5 * MIB)


def test_smallest_budget_finds_the_first_budget_that_runs() -> None:
    tried: list[int] = []

    def call(budget: int) -> None:
        tried.append(budget)
        if budget < 12_345:
            raise MemoryError("too small")

    assert smallest_budget(call, 1 << 20) == 12_345
    assert len(tried) <= 21
    # A call that fits from the first byte needs no budget.
    assert smallest_budget(lambda budget: None, 1 << 20) == 0


def test_budget_bounds_bracket_the_smallest_budget_within_the_share_given() -> None:
    def call(budget: int) -> None:
        # The bytes stated grow with the budget, as chunks that grow with it do.
        if budget < 3_000_000:
            raise MemoryError(f"Importing holds about {budget + 1000} bytes, more")

    refused, accepted = budget_bounds(call, 1 << 20, parts=100)
    assert refused < 3_000_000 <= accepted
    assert accepted - refused <= accepted // 100
    # A start that fits measures nothing.
    with pytest.raises(AssertionError, match="too small a budget to measure"):
        budget_bounds(call, 4_000_000, parts=100)


def test_budget_bounds_fail_a_refusal_that_states_no_more_than_its_budget() -> None:
    def call(budget: int) -> None:
        raise MemoryError(f"Importing holds about {budget} bytes, more")

    with pytest.raises(AssertionError, match="states no more than 1048576 bytes"):
        budget_bounds(call, 1 << 20, parts=100)


def test_cost_bound_passes_an_estimate_close_above_the_traced_peak() -> None:
    result, peak = cost_bound(
        MIB + 32 * 1024, lambda: int(np.ones(MIB, dtype=np.uint8).size)
    )
    assert result == MIB and MIB <= peak


@pytest.mark.parametrize(
    ("estimate", "message"),
    [
        # A term left out.
        (MIB // 2, "held .* bytes, more than"),
        # A term counted twice.
        (2 * MIB, "more than 1.3 times the traced peak"),
    ],
)
def test_cost_bound_fails_an_estimate_from_either_side(estimate, message) -> None:
    with pytest.raises(AssertionError, match=message):
        cost_bound(estimate, lambda: int(np.ones(MIB, dtype=np.uint8).size))


def test_cost_bound_takes_an_allowance_beside_its_slack() -> None:
    cost_bound(2 * MIB, lambda: np.ones(MIB, dtype=np.uint8).size, allowance=MIB)


def _gone(reference: weakref.ref, seconds: float = 10.0) -> bool:
    """Whether the referenced object is freed within ``seconds``, as a pool thread lets go of it."""
    deadline = time.monotonic() + seconds
    while reference() is not None:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _finished(pool: ThreadPoolExecutor) -> weakref.ref:
    """A weak reference to the result of one work item that ``pool`` has finished."""
    future = pool.submit(np.ones, 1000)
    return weakref.ref(future.result())


def test_each_read_is_held_until_the_next_read_ends() -> None:
    run = pool_thread._WorkItem.run
    with ThreadPoolExecutor(1) as pool:
        with hold_finished_reads() as end_read:
            first = _finished(pool)
            end_read()
            # The read after it has not ended, so the first read is held.
            assert first() is not None
            second = _finished(pool)
            assert first() is not None
            end_read()
            assert _gone(first)
            assert second() is not None
        # Leaving the context lets go of every read it held.
        assert _gone(second)
        assert pool_thread._WorkItem.run is run
        # Outside it, a pool thread lets go of its finished work.
        assert _gone(_finished(pool))


def test_a_function_named_as_a_read_ends_a_read_each_time_it_returns() -> None:
    owner = SimpleNamespace(read=lambda value: value + 1)
    read = owner.read
    with ThreadPoolExecutor(1) as pool:
        with hold_finished_reads((owner, "read")):
            first = _finished(pool)
            assert owner.read(1) == 2
            assert first() is not None
            assert owner.read(2) == 3
            assert _gone(first)
        assert owner.read is read


# Chunks of 12,000 rows of 40 int32 values, 1.92 MB decoded, read one after
# another in a fresh process whose pool runs no earlier test's reads. The
# reads are traced alone, then with the work of the read before held, ending
# each read by hand and then at each read of a Zarr array.
_HELD_READ_SCRIPT = """
import contextlib, json, sys, tracemalloc
import zarr
from tests.memory_helpers import ZARR_READS, hold_finished_reads

array = zarr.open_array(sys.argv[1], mode="r")
rows = array.chunks[0]

def read_each_chunk(end_read):
    for start in range(0, array.shape[0], rows):
        block = array[start : start + rows]
        del block
        end_read()

def held(name):
    if name == "free":
        return contextlib.nullcontext(lambda: None)
    return hold_finished_reads(*(ZARR_READS if name == "at each Zarr read" else ()))

read_each_chunk(lambda: None)
peaks = {}
for name in ("free", "by hand", "at each Zarr read"):
    with held(name) as end_read:
        tracemalloc.start()
        start = tracemalloc.get_traced_memory()[0]
        read_each_chunk(end_read if name == "by hand" else lambda: None)
        peaks[name] = tracemalloc.get_traced_memory()[1] - start
        tracemalloc.stop()
print(json.dumps(peaks))
"""


def test_a_read_that_a_pool_thread_still_holds_is_missed_unless_reads_are_held(
    tmp_path,
) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 1000, size=(60_000, 40)).astype(np.int32)
    array = zarr.open_group(str(tmp_path / "codes.zarr"), mode="w").create_array(
        "codes", shape=codes.shape, chunks=(12_000, 40), dtype="int32"
    )
    array[...] = codes
    traced = subprocess.run(
        [
            sys.executable,
            "-c",
            _HELD_READ_SCRIPT,
            str(tmp_path / "codes.zarr" / "codes"),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    peaks = json.loads(traced.stdout.strip().splitlines()[-1])

    chunk = 12_000 * 40 * 4
    free = peaks.pop("free")
    for name, held in peaks.items():
        # Without the hold the trace sees one read at a time; with it, the
        # decoded chunk of the read before beside each read.
        assert held - free > chunk // 2, (name, free, held)
        # A read holds its chunk as stored and decoded and the rows it
        # returns, and the read before it adds no more than that again.
        assert held <= 2 * free + chunk, (name, free, held)


def test_reads_of_zarr_arrays_end_only_while_reads_are_held() -> None:
    methods = {name: getattr(zarr.Array, name) for _owner, name in ZARR_READS}
    with hold_finished_reads(*ZARR_READS):
        assert all(
            getattr(zarr.Array, name) is not method for name, method in methods.items()
        )
    assert all(getattr(zarr.Array, name) is method for name, method in methods.items())
