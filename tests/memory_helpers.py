"""Check that calls hold no more than the memory budgets they accept.

A memory budget bounds the bytes that tracemalloc sees a call allocate beside
what the process held before it. These helpers trace that peak, find the
smallest budget a call accepts, from its refusals or by bisection, bound
estimates from both sides, and hold the reads that Zarr's pool threads may
still hold on a busy machine, so that every memory test checks budgets one way.
"""

import re
import threading
import tracemalloc
from collections.abc import Callable, Iterator
from concurrent.futures import thread as pool_thread
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import zarr

from tests.storage_helpers import open_store, written

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore


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


def stated_bytes(refusal: BaseException, pattern: str = r"about (\d+) bytes") -> int:
    """Return the bytes that a refusal states, as in ``holds about N bytes``.

    The refusals of the clone, repertoire and variant operations state what
    the call would hold, or need, without naming a budget to raise to, so the
    bytes they state are the budget to try next. ``pattern``, a regular
    expression whose last group is the number, finds them: by default the
    first ``about N bytes`` of the message. A test that requires the words
    before the number, such as ``holds about``, passes a pattern of its own.
    """
    found = re.search(pattern, str(refusal))
    assert found is not None, f"The refusal states no bytes as {pattern!r}: {refusal}"
    return int(found.group(found.lastindex or 0))


def traced_peak(call: Callable[[], Any], budget: int) -> tuple[Any, int]:
    """Return the result of ``call()`` and the most bytes it held beside what was held before it.

    tracemalloc counts every allocation in the process while the call runs,
    other threads' included, so a peak above ``budget`` is traced again on a
    second run, and the smaller peak is returned: a call that holds more than
    its budget does so in both runs.
    """

    def trace() -> tuple[Any, int]:
        tracemalloc.start()
        try:
            start = tracemalloc.get_traced_memory()[0]
            result = call()
            return result, tracemalloc.get_traced_memory()[1] - start
        finally:
            tracemalloc.stop()

    result, peak = trace()
    if peak > budget:
        result, again = trace()
        peak = min(peak, again)
    return result, peak


def _store_at_budget(store: Any, options: dict[str, Any]) -> Callable[[int], Any]:
    """Return a function that gives what the call takes at a budget, as :func:`budget_that_runs` describes."""
    if isinstance(store, str | Path):
        return lambda budget: open_store(store, mem_budget=budget, **options)
    assert not options, f"Options {sorted(options)} apply only to a store's path"
    if not _is_open_store(store):
        assert callable(store), f"{store!r} is no path, open store or function"
        return store

    def at(budget: int) -> "DataStore":
        store.memoryBytes = budget
        return store

    return at


def _is_open_store(store: Any) -> bool:
    """Whether ``store`` is an open store, whose budget is its ``memoryBytes``."""
    return not isinstance(store, str | Path) and hasattr(store, "memoryBytes")


def budget_that_runs(
    store: "str | Path | DataStore | Callable[[int], Any]",
    call: Callable[[Any], Any],
    *,
    refusal_budget: Callable[[BaseException], int] = named_budget,
    max_refusals: int = 10,
    traced: bool = True,
    **options: Any,
) -> tuple[int, list[str], Any, int | None]:
    """Follow the refusals of ``call`` from a budget of one byte to the budget that runs it.

    ``call`` runs an operation on what ``store`` gives at each budget. That is
    the store at the path ``store``, opened with the budget and ``options``;
    the open store ``store``, whose ``memoryBytes`` each budget sets and which
    gets its own budget back afterwards; or what the function ``store`` gives
    for the budget, such as the budget itself, ``lambda budget: budget``, for
    a call that takes its memory budget as an argument. Each refusal must name
    a budget above the one it refused, as ``refusal_budget`` reads it, which
    the next attempt takes, and a check that a later step makes may refuse
    that budget and name more. Each attempt is traced as :func:`traced_peak`
    traces it, or runs untraced without ``traced``, and ``max_refusals``
    refusals fail the search: ten, unless a step's estimate grows with the
    budget, as those of the clone, repertoire and variant steps that size
    their chunks from it do.

    Returns:
        The budget that runs the call, the messages of the refusals on the
        way, and the result and traced peak of the call at that budget, the
        peak None when untraced.
    """
    at_budget = _store_at_budget(store, options)
    restored = store.memoryBytes if _is_open_store(store) else None
    budget, refusals = 1, []
    try:
        while len(refusals) < max_refusals:
            opened = at_budget(budget)
            try:
                if traced:
                    result, peak = traced_peak(partial(call, opened), budget)
                else:
                    result, peak = call(opened), None
            except MemoryError as refusal:
                named = refusal_budget(refusal)
                assert named > budget, f"{refusal} names no budget above {budget}"
                refusals.append(str(refusal))
                budget = named
            else:
                return budget, refusals, result, peak
    finally:
        if restored is not None:
            store.memoryBytes = restored  # type: ignore[union-attr]
    raise AssertionError(
        f"{max_refusals} refusals named no budget that runs: {refusals}"
    )


#: A refusal's step and the bytes it states: what comes before its last
#: ``holds about N bytes``, and N.
_REFUSAL_STEP = r"^(.*) holds about (\d+) bytes"


def budget_refusals(
    store: "DataStore", run: Callable[[], object]
) -> list[tuple[str, int]]:
    """Each refusal of ``run`` on the way from a budget of one byte to one that runs, as the step it names and the bytes it states.

    :func:`budget_that_runs` follows the refusals on the open ``store``,
    untraced, by the bytes of each refusal's last ``holds about N bytes``, so
    each later step's check shows once the earlier ones pass, for up to 100
    refusals.
    """
    _budget, refusals, _result, _peak = budget_that_runs(
        store,
        lambda _store: run(),
        refusal_budget=partial(stated_bytes, pattern=_REFUSAL_STEP),
        max_refusals=100,
        traced=False,
    )
    found: list[tuple[str, int]] = []
    for refusal in refusals:
        matched = re.search(_REFUSAL_STEP, refusal)
        assert matched is not None, refusal
        found.append((matched.group(1), int(matched.group(2))))
    return found


def check_budget(
    store: "str | Path | DataStore",
    call: Callable[["DataStore"], Any],
    *,
    check_writes: bool = True,
    at_most: float | None = None,
    allowance: int = 0,
    refusal_budget: Callable[[BaseException], int] = named_budget,
    **options: Any,
) -> tuple[int, list[str], Any]:
    """Check that ``call`` runs within the budget that its refusals name.

    :func:`budget_that_runs` finds that budget and traces the call at it. The
    traced peak must be within the budget, and one byte less must be refused
    naming the same budget. With ``check_writes``, the refused call must leave
    what the store holds unchanged. With ``at_most``, the budget must also be
    at most that many times the traced peak, plus ``allowance`` bytes, so an
    estimate that counts far more than the call holds fails too. ``call``
    should compute again rather than reuse.

    Returns:
        The budget, the messages of the refusals on the way, and the result
        of the call at that budget.
    """
    budget, refusals, result, peak = budget_that_runs(
        store, call, refusal_budget=refusal_budget, **options
    )
    assert peak is not None and peak <= budget, (
        f"The call held {peak} bytes within a budget of {budget}"
    )
    if at_most is not None:
        assert budget <= at_most * peak + allowance, (
            f"The call accepted a budget of {budget} bytes, more than {at_most} "
            f"times its traced peak of {peak} bytes and {allowance} bytes besides"
        )
    if isinstance(store, str | Path):
        path = store
        saved = written(open_store(path)) if check_writes else None
        with pytest.raises(MemoryError) as refused:
            call(open_store(path, mem_budget=budget - 1, **options))
        after = written(open_store(path)) if check_writes else None
    else:
        restored = store.memoryBytes
        saved = written(store) if check_writes else None
        store.memoryBytes = budget - 1
        try:
            with pytest.raises(MemoryError) as refused:
                call(store)
        finally:
            store.memoryBytes = restored
        after = written(store) if check_writes else None
    named = refusal_budget(refused.value)
    assert named == budget, (
        f"One byte below the budget of {budget} bytes was refused naming {named}"
    )
    assert after == saved, "The refused call changed what the store holds"
    return budget, refusals, result


def smallest_budget(call: Callable[[int], object], high: int) -> int:
    """The smallest budget below ``high`` at which ``call`` raises no MemoryError.

    Found by bisection, so ``call`` must succeed at every larger budget.
    """
    low = 0
    while low < high:
        middle = (low + high) // 2
        try:
            call(middle)
        except MemoryError:
            low = middle + 1
        else:
            high = middle
    return low


def at_budget(store: "DataStore", run: Callable[[], Any]) -> Callable[[int], Any]:
    """A call of ``run`` on ``store`` at the budget it is given, for :func:`smallest_budget` and :func:`budget_bounds`.

    The store keeps the last budget given; a test that goes on with the store
    sets the budget it needs.
    """

    def call(budget: int) -> Any:
        store.memoryBytes = budget
        return run()

    return call


def budget_bounds(
    call: Callable[[int], object],
    start: int,
    *,
    parts: int,
    refusal_bytes: Callable[[BaseException], int] = stated_bytes,
) -> tuple[int, int]:
    """The largest budget that ``call`` refuses and the smallest it accepts, at most one part in ``parts`` of the larger apart.

    ``call(budget)`` raises MemoryError below some budget, stating the bytes
    it would hold, and runs at every budget above it. The search climbs from
    a refused ``start`` to the bytes that each refusal states, or a quarter
    more where that is more, since a call whose chunks grow with its budget
    states more at a larger one, and then bisects; ``refusal_bytes`` reads the
    bytes a refusal states, and 100 refusals in the climb fail the search. A
    call that holds no more than the largest budget refused, plus one byte, at
    the budget accepted holds no more than any budget it accepts.
    """

    def refused(budget: int) -> int | None:
        try:
            call(budget)
        except MemoryError as refusal:
            stated = refusal_bytes(refusal)
            assert stated > budget, f"{refusal} states no more than {budget} bytes"
            return stated
        return None

    low, high = 0, start
    for _climb in range(100):
        stated = refused(high)
        if stated is None:
            break
        low, high = high, max(stated, high + high // 4)
    else:
        raise AssertionError(
            f"100 refusals from {start} bytes named no budget that runs"
        )
    assert low, f"The call fits {start} bytes, too small a budget to measure"
    while high - low > high // parts:
        middle = (low + high) // 2
        if refused(middle) is None:
            high = middle
        else:
            low = middle
    return low, high


def cost_bound(
    estimate: int, call: Callable[[], Any], *, slack: float = 1.3, allowance: int = 0
) -> tuple[Any, int]:
    """Check that ``estimate`` bounds what ``call`` holds, and by no more than ``slack`` times it.

    The traced peak must be at most ``estimate``, and ``estimate`` at most
    ``slack`` times the peak plus ``allowance`` bytes, so a cost that leaves
    out a term fails, and so does one that counts a term twice.

    Returns:
        The result of the call and its traced peak.
    """
    result, peak = traced_peak(call, estimate)
    assert peak <= estimate, f"The call held {peak} bytes, more than {estimate}"
    assert estimate <= slack * peak + allowance, (
        f"The estimate of {estimate} bytes is more than {slack} times the "
        f"traced peak of {peak} bytes and {allowance} bytes besides"
    )
    return result, peak


#: The methods in which every read of a stored Zarr array's values ends, by
#: index or by selection, for :func:`hold_finished_reads` to end a read at each.
ZARR_READS: tuple[tuple[Any, str], ...] = tuple(
    (zarr.Array, name)
    for name in (
        "get_basic_selection",
        "get_orthogonal_selection",
        "get_coordinate_selection",
        "get_mask_selection",
        "get_block_selection",
    )
)


@contextmanager
def hold_finished_reads(*reads: tuple[Any, str]) -> Iterator[Callable[[], None]]:
    """Hold the work of each read of Zarr's pool until the next read ends.

    The thread of Zarr's pool that decoded a chunk holds the chunk as stored,
    the work it was given, and as decoded, its result, until that thread next
    runs, which on a busy machine came after the next read had begun. Within
    this context every work item that a pool thread runs for a read is kept
    with that read, and let go when the read after it ends, so a trace sees
    each read beside the one before it, without timing. Work items of earlier
    reads are let go at once.

    A read is a call of a function named in ``reads``, as (owner, attribute)
    pairs, and ends when the call returns. Work that a pool thread starts while
    no such call runs, such as the writes after an operation's last read,
    belongs to no read and is let go as it would be without the hold. With no
    function named, the caller ends each read by calling the function that
    this context gives, and all work belongs to the read under way.

    The hold applies to every thread pool of the process, so a test that must
    see only its own reads traces them in a fresh process, which imports this
    helper as an in-process test does.
    """
    lock = threading.Lock()
    reading = [0]
    # The calls of named reads under way, in any thread.
    running = [0]
    held: dict[int, list[object]] = {}
    run = pool_thread._WorkItem.run

    def run_and_hold(item: object, *args: Any) -> None:
        with lock:
            read = reading[0]
            for_a_read = running[0] > 0 or not reads
        run(item, *args)  # type: ignore[arg-type]
        with lock:
            if for_a_read and read >= reading[0] - 1:
                held.setdefault(read, []).append(item)

    def end_read() -> None:
        with lock:
            held.pop(reading[0] - 1, None)
            reading[0] += 1

    def ending(function: Callable[..., Any]) -> Callable[..., Any]:
        def read(*args: Any, **kwargs: Any) -> Any:
            with lock:
                running[0] += 1
            try:
                value = function(*args, **kwargs)
            finally:
                with lock:
                    running[0] -= 1
            end_read()
            return value

        return read

    replaced = [(owner, name, getattr(owner, name)) for owner, name in reads]
    for owner, name, function in replaced:
        setattr(owner, name, ending(function))
    pool_thread._WorkItem.run = run_and_hold  # type: ignore[method-assign]
    try:
        yield end_read
    finally:
        pool_thread._WorkItem.run = run  # type: ignore[method-assign]
        for owner, name, function in reversed(replaced):
            setattr(owner, name, function)
        with lock:
            held.clear()
            reading[0] = 0
