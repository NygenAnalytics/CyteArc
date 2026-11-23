"""Read the budget that a memory refusal names, follow refusals to a budget that runs, give a domain function a ledger of a budget, wait until the arrays handed to writes are freed, hold a read's decodes together while it is traced, and skip tests that need the resident memory of the test process."""

import asyncio
import re
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

# CyteArc reads the resident memory of a process from Linux's /proc filesystem
# and reports none where /proc is missing, as on macOS.
needs_proc_resident_memory = pytest.mark.skipif(
    not Path("/proc/self/status").is_file(),
    reason=(
        "needs the resident memory of the test process, which CyteArc reads "
        "from Linux's /proc filesystem"
    ),
)


def named_budget(refusal: BaseException) -> int:
    """Return the memory budget that a refusal names in its closing ``raise mem_budget to at least N bytes``.

    A refusal that says the call holds about M bytes must name M. A refusal
    that closes with ``Raise mem_budget.`` names no budget, as Mixscape's
    refusal of a classification batch does, and asks for the M bytes the call
    holds.
    """
    message = str(refusal)
    held = re.search(r"holds? about (\d+) bytes", message)
    if message.endswith(" Raise mem_budget."):
        assert held is not None, f"The refusal names no budget: {message}"
        return int(held.group(1))
    named = re.search(
        r"[Rr]aise mem_budget to at least (\d+) bytes(?:, or [^.]*)?\.$", message
    )
    assert named is not None, f"The refusal does not end naming a budget: {message}"
    assert held is None or held.group(1) == named.group(1), (
        f"The refusal names a budget other than what it holds: {message}"
    )
    return int(named.group(1))


def climb_refusals(
    run: Callable[[int], Any], *, start: int = 1
) -> tuple[list[Any], Any]:
    """Run ``run`` at a budget of ``start`` bytes, then at each budget that its refusal names, until it runs.

    Each refusal must name a budget above the one it refused, so each step
    that refuses is passed at the next budget. The budgets come from the
    refusals alone, so the climb holds whatever the steps charge.

    Returns:
        Each refusal, a ``MemoryBudgetError``, in order, and what the run at
        the last budget returned.
    """
    from cytearc.utils.memory_budget import MemoryBudgetError

    refusals: list[MemoryBudgetError] = []
    budget = start
    for _ in range(1_000):
        try:
            return refusals, run(budget)
        except MemoryBudgetError as error:
            named = named_budget(error)
            assert named > budget, f"A refusal at {budget} bytes names {named}: {error}"
            refusals.append(error)
            budget = named
    raise AssertionError(f"1,000 refusals named no budget that runs: {refusals[-1]}")


def budgeted(budget: int | None, doing: str = "Calling clones") -> Any:
    """A memory ledger of ``budget`` bytes that holds nothing yet, or None for no budget.

    Domain functions that a test calls alone charge it as they would an
    operation's ledger that held nothing beside them.
    """
    from cytearc.utils.memory_budget import MemoryLedger

    return None if budget is None else MemoryLedger(budget, doing)


def recording_ledger(doing: str = "Recording charges") -> Any:
    """A memory ledger without a budget whose ``most`` records the most bytes held at once, peaks included."""
    from cytearc.utils.memory_budget import MemoryLedger

    class RecordingLedger(MemoryLedger):
        def __init__(self) -> None:
            super().__init__(None, doing)
            self.most = 0

        def hold(self, item, nbytes, *, remedy=None):
            held = super().hold(item, nbytes, remedy=remedy)
            self.most = max(self.most, self.held_bytes)
            return held

        def peak(self, item, nbytes, *, remedy=None):
            super().peak(item, nbytes, remedy=remedy)
            self.most = max(self.most, self.held_bytes + int(nbytes))

    return RecordingLedger()


class HandedArrays:
    """Arrays handed to writes, and whether each wait saw them all freed.

    A write's values can outlive the call that writes them by a moment: Zarr
    encodes and stores chunks on pool threads, which let go of their views of
    the values after they have handed back their results, and a write that
    ends runs its weak-reference callbacks before it lets go of its
    attributes. A count of the arrays alive at one instant can catch one of
    them on its way out, so ``wait_until_freed`` waits for their release
    instead, which comes as soon as those threads go on. An array that the
    code under test keeps stays alive while the test waits, and the wait ends
    without it after ``seconds``.
    """

    def __init__(self, seconds: float = 10.0) -> None:
        self._arrays: list[weakref.ref] = []
        self._released = threading.Condition()
        self._seconds = seconds
        self.freed: list[bool] = []

    def __len__(self) -> int:
        return len(self._arrays)

    def hand(self, values: Any) -> None:
        """Watch ``values``, which a write was handed, until it is freed."""
        self._arrays.append(weakref.ref(values, self._on_release))

    def _on_release(self, _array: weakref.ref) -> None:
        with self._released:
            self._released.notify_all()

    def wait_until_freed(self) -> None:
        """Record in ``freed`` whether every array handed so far is freed.

        The first wait that ends with an array alive records False, and the
        waits after it record without waiting, so a test that keeps an array
        fails after one wait.
        """
        with self._released:
            self.freed.append(
                self._released.wait_for(
                    lambda: all(array() is None for array in self._arrays),
                    self._seconds if all(self.freed) else 0,
                )
            )


@contextmanager
def decodes_held_together(
    codec: type, parties: int, timeout: float = 10.0
) -> Iterator[None]:
    """Make each decode of Zarr's ``codec`` hold its chunk, as stored and decoded, until ``parties`` decodes hold theirs.

    How many of a read's decodes overlap depends on the threads that decode
    them and on how the machine schedules them, so a traced read at an I/O
    concurrency of ``parties`` holds that many decoded chunks at once only
    where they are made to wait for each other. Each decode that ends waits
    on the event loop that runs its read, so the read then holds ``parties``
    decoded chunks at once on any machine. A read that never has that many
    decodes in flight raises ``TimeoutError`` after ``timeout`` seconds
    instead of being traced at less.
    """
    decode = codec._decode_single
    barriers: weakref.WeakKeyDictionary[Any, asyncio.Barrier] = (
        weakref.WeakKeyDictionary()
    )

    async def held(self: Any, chunk_bytes: Any, chunk_spec: Any) -> Any:
        decoded = await decode(self, chunk_bytes, chunk_spec)
        # Each event loop that runs a read has a barrier of its own.
        loop = asyncio.get_running_loop()
        barrier = barriers.get(loop)
        if barrier is None:
            barrier = barriers[loop] = asyncio.Barrier(parties)
        async with asyncio.timeout(timeout):
            await barrier.wait()
        return decoded

    codec._decode_single = held
    try:
        yield
    finally:
        codec._decode_single = decode
