"""Read the budget that a memory refusal names, follow refusals to a budget that runs, give a domain function a ledger of a budget, trace a call after an untraced call, trace what a call holds against what its ledger charges, trace a read of selected cells, wait until the arrays handed to writes are freed, hold a read's decodes together while it is traced, record and trace what an import's process sends through its pipe, and skip tests that need the resident memory of the test process."""

import asyncio
import re
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

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


class TracedCharge(NamedTuple):
    """What a traced call held against what its ledger charged.

    Attributes:
        held: The most the call held at once.
        charged: The most its ledger charged at once, peaks included.
        kept: What the call still held reachable when it returned, its result
            among it.
        kept_charged: What its ledger still held charged when it returned.
    """

    held: int
    charged: int
    kept: int
    kept_charged: int


def traced_reruns[T](call: Callable[[], T], runs: int = 1) -> list[T]:
    """Call ``call`` once untraced, then ``runs`` times while tracemalloc traces, each after a full collection, and return what the traced calls returned.

    ``call`` reads tracemalloc's counts itself, which read zero in the
    untraced call; ``tracemalloc.is_tracing()`` tells the calls apart, as a
    call that writes a new destination each time needs. The untraced call
    makes the first calls of every function that the traced calls run, so
    that what a first call allocates once stays out of the traces: imports,
    Zarr's objects and coverage's records. Coverage, measuring through
    ``sys.monitoring`` as CI's Python 3.14 job does, records each function
    at its first call in a dict whose table doubles when it fills; a traced
    dense reader process whose pipe end's first call filled it held the new
    table, 1.31 MB, beside its own 0.45 MB.
    """
    import gc
    import tracemalloc

    call()
    traced = []
    for _ in range(max(1, int(runs))):
        gc.collect()
        tracemalloc.start()
        try:
            traced.append(call())
        finally:
            tracemalloc.stop()
    return traced


def traced_charge(call: Callable[[Any], Any], runs: int = 1) -> TracedCharge:
    """Run ``call(ledger)`` against a :func:`recording_ledger` through :func:`traced_reruns`, and return the traced run that held the least.

    A traced run can also hold a block that the call does not make for
    itself: the interpreter's table of interned strings resizes when it
    fills, as pathlib's interning of the paths that Zarr's local store reads
    fills it, which one of five traced reads of 1,000 bands held as a block
    of 3.84 MB. The least of a few runs leaves such a block out.
    """
    import gc
    import tracemalloc

    def measured() -> TracedCharge:
        ledger = recording_ledger()
        before, _ = tracemalloc.get_traced_memory()
        result = call(ledger)
        _, peak = tracemalloc.get_traced_memory()
        # What the call keeps is what stays reachable, its result among it.
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
        del result
        return TracedCharge(
            peak - before, ledger.most, current - before, ledger.held_bytes
        )

    return min(traced_reruns(measured, runs), key=lambda run: run.held)


def traced_selection_read(
    array: Any, rows: Any, columns: Any, runs: int = 2
) -> tuple[Any, int]:
    """Read ``rows`` by ``columns`` of ``array`` with ``read_selection`` at one decode at a time, once untraced and ``runs`` times traced after a full collection each, and return the values and the least that a traced run held at its peak.

    The least of a few runs leaves out a resize of the interpreter's table of
    interned strings, as :func:`traced_charge` does.
    """
    import gc
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.virtual_arrays import read_selection

    peaks = []
    with zarr_io_concurrency(1):
        values = read_selection(array, rows, columns)
        for _ in range(max(1, int(runs))):
            del values
            gc.collect()
            tracemalloc.start()
            try:
                before, _ = tracemalloc.get_traced_memory()
                values = read_selection(array, rows, columns)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            peaks.append(peak - before)
    return values, min(peaks)


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


@contextmanager
def decodes_on_threads_held_together(
    codec: type, parties: int, timeout: float = 10.0
) -> Iterator[None]:
    """Make each synchronous decode of Zarr's ``codec`` hold its chunk, as stored and decoded, until ``parties`` decodes hold theirs.

    Zarr's ``FusedCodecPipeline`` decodes on the threads of a pool of its
    own, outside any event loop, so the decodes that overlap there depend on
    the machine as well. Each decode that ends waits on its thread until
    ``parties`` decodes hold their chunks, so a read that runs that many at
    once holds that many decoded chunks on any machine. A read with fewer
    decodes at once raises ``threading.BrokenBarrierError`` after
    ``timeout`` seconds instead of being traced at less.
    """
    decode = codec._decode_sync
    barrier = threading.Barrier(parties, timeout=timeout)

    def held(self: Any, chunk_bytes: Any, chunk_spec: Any) -> Any:
        decoded = decode(self, chunk_bytes, chunk_spec)
        barrier.wait()
        return decoded

    codec._decode_sync = held
    try:
        yield
    finally:
        codec._decode_sync = decode


def received_messages(sent: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Read back what a process sent through one end of a pipe, as an H5AD import reads it.

    ``sent`` holds, in order, ``("message", value)`` or ``("pickled", bytes)``
    for each message and ``("bytes", buffer)`` for each raw buffer. Each
    message comes back with the arrays whose buffers follow it.
    """
    import pickle

    from cytearc.writers.h5ad import _received

    stream = iter(sent)

    class _Receiving:
        def recv(self) -> Any:
            form, value = next(stream)
            assert form in ("message", "pickled"), f"{form} where a message was due"
            return pickle.loads(value) if form == "pickled" else value

        def recv_bytes(self) -> bytes:
            entry = next(stream, None)
            if entry is None:
                raise EOFError
            assert entry[0] == "bytes", f"{entry[0]} where a buffer was due"
            return bytes(entry[1])

    messages: list[tuple[str, Any]] = []
    end = _Receiving()
    while True:
        try:
            messages.append(_received(end, "the process"))
        except StopIteration:
            return messages


class TracedPipeEnd:
    """One end of a pipe that sends as multiprocessing's connections do, tracing each send.

    ``send`` pickles a message with ``ForkingPickler``, as
    ``multiprocessing.connection.Connection.send`` does, and ``send_bytes``
    keeps a view of the buffer, which ``Connection.send_bytes`` writes to the
    pipe from the buffer itself. ``held`` records for each send the most that
    tracemalloc saw allocated beside what was allocated before it; tracing
    must be on. Make it before tracing starts: it sends a message and a
    buffer untraced as it is made, and drops them, so that no traced send
    makes a first call, which under coverage allocates records inside the
    trace, as :func:`traced_reruns` describes.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.held: list[int] = []
        self.closed = False
        self.send(None)
        self.send_bytes(b"")
        self.sent.clear()
        self.held.clear()

    def _traced(self, form: str, send: Callable[[], Any]) -> None:
        import tracemalloc

        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        self.sent.append((form, send()))
        _, peak = tracemalloc.get_traced_memory()
        self.held.append(peak - before)

    def send(self, message: Any) -> None:
        from multiprocessing.reduction import ForkingPickler

        self._traced("pickled", lambda: bytes(ForkingPickler.dumps(message)))

    def send_bytes(self, buffer: Any) -> None:
        self._traced("bytes", lambda: memoryview(buffer))

    def close(self) -> None:
        self.closed = True

    @property
    def messages(self) -> list[tuple[str, Any]]:
        """The messages as the import reads them, with their arrays."""
        return received_messages(self.sent)
