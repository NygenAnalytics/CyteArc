"""Check that calls hold no more than the memory budgets they accept.

A memory budget bounds the bytes that tracemalloc sees a call allocate beside
what the process held before it. These helpers trace that peak, find the
smallest budget a call accepts, from its refusals or by bisection, bound
estimates from both sides, and hold the reads and writes that Zarr's pool
threads may still hold on a busy machine, so that every memory test checks
budgets one way.
"""

import inspect
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
    that closes with ``Raise mem_budget.`` names no budget, as some refusals
    of the clone stack do, and asks for the M bytes the call holds.
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


def index_copies_references() -> bool:
    """Whether this pandas copies an object array's references into an index built on it, as pandas 3 does and pandas 2 does not."""
    import numpy as np
    import pandas as pd

    values = np.asarray(["x", "y"], dtype=object)
    return not np.shares_memory(pd.Index(values, dtype=object).to_numpy(), values)


def budgeted(budget: int | None, doing: str = "Calling clones") -> Any:
    """A memory ledger of ``budget`` bytes that holds nothing yet, or None for no budget.

    Domain functions that a test calls alone charge it as they would an
    operation's ledger that held nothing beside them.
    """
    from cytearc.utils.memory_budget import MemoryLedger

    return None if budget is None else MemoryLedger(budget, doing)


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
    max_refusals: int = 10,
    **options: Any,
) -> tuple[int, list[str], Any]:
    """Check that ``call`` runs within the budget that its refusals name.

    :func:`budget_that_runs` finds that budget, following up to
    ``max_refusals`` refusals, and traces the call at it. The
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
        store,
        call,
        refusal_budget=refusal_budget,
        max_refusals=max_refusals,
        **options,
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


def follow_refusals(
    at_budget: Callable[[int], object], *, start: int = 1, limit: int = 60
) -> tuple[int, list[Any]]:
    """Follow the refusals of ``at_budget`` from ``start`` bytes to the budget that runs it.

    Every refusal on the way must be a ``MemoryBudgetError`` of the budget
    tried, ``budget_bytes``, that names a larger budget, ``needed_bytes``,
    and a step, ``item``; the next attempt takes the budget it names, so a
    refusal stated in other terms than the call's whole budget fails. One
    byte below the budget that runs is refused naming it, as plans that
    depend only on their budget do.

    Returns:
        The budget that runs the call, and the refusals on the way.
    """
    from cytearc.utils.memory_budget import MemoryBudgetError

    budget, refusals = start, []
    for _attempt in range(limit):
        try:
            at_budget(budget)
        except MemoryError as error:
            assert isinstance(error, MemoryBudgetError), error
            assert error.budget_bytes == budget, error
            assert error.needed_bytes > budget, error
            assert error.item, error
            refusals.append(error)
            budget = error.needed_bytes
        else:
            break
    else:
        raise AssertionError(f"{limit} refusals named no budget that runs")
    assert refusals, f"{start} bytes ran the call"
    with pytest.raises(MemoryError) as below:
        at_budget(budget - 1)
    assert isinstance(below.value, MemoryBudgetError), below.value
    assert below.value.needed_bytes == budget, below.value
    return budget, refusals


def refusal_naming(
    at_budget: Callable[[int], object], words: str, *, start: int = 1, limit: int = 100
) -> Any:
    """Follow the refusals of ``at_budget`` from ``start`` bytes to the first whose message holds ``words``, and return it.

    Every refusal on the way must be a ``MemoryBudgetError`` of the budget
    tried that names a larger budget, which the next attempt takes, as
    :func:`follow_refusals` requires; a call that runs before a refusal holds
    ``words`` fails, so the steps before the one sought refuse without
    writing anything.
    """
    from cytearc.utils.memory_budget import MemoryBudgetError

    budget = start
    for _attempt in range(limit):
        try:
            at_budget(budget)
        except MemoryError as error:
            assert isinstance(error, MemoryBudgetError), error
            assert error.budget_bytes == budget, error
            assert error.needed_bytes > budget, error
            if words in str(error):
                return error
            budget = error.needed_bytes
        else:
            raise AssertionError(
                f"The call ran at {budget} bytes before a refusal named {words!r}"
            )
    raise AssertionError(f"{limit} refusals named no {words!r}")


def named_budget_passes(
    at_budget: Callable[[int], object], budget: int, *, item: str
) -> Any:
    """Check the refusal of ``at_budget(budget)`` and that the budget it names passes its step, which a later step may refuse.

    The refusal is a ``MemoryBudgetError`` that names ``budget``, ``item``
    and more bytes than the budget; at the named budget the call runs or
    refuses another step, and one byte less is refused naming the same
    budget and step.

    Returns:
        The refusal.
    """
    from cytearc.utils.memory_budget import MemoryBudgetError

    def refusal_at(memory: int) -> MemoryBudgetError:
        with pytest.raises(MemoryError) as raised:
            at_budget(memory)
        assert isinstance(raised.value, MemoryBudgetError), raised.value
        return raised.value

    refusal = refusal_at(budget)
    assert refusal.budget_bytes == budget, refusal
    assert refusal.needed_bytes > budget, refusal
    assert refusal.item == item, refusal
    try:
        at_budget(refusal.needed_bytes)
    except MemoryBudgetError as later:
        assert later.item != item, later
    below = refusal_at(refusal.needed_bytes - 1)
    assert (below.needed_bytes, below.item) == (refusal.needed_bytes, item), below
    return refusal


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


def charges(call: Callable[[], Any]) -> tuple[Any, list[tuple[str, int, int]]]:
    """Run ``call`` and return its result with each charge that its memory ledgers took, in order.

    Each charge is its item, its bytes and what its ledger then planned: what
    it held, beside what its last read may still hold, and a transient
    peak's bytes. A release is ``("a release", bytes, planned)``.
    """
    from cytearc.utils.memory_budget import observed_charges

    taken: list[tuple[str, int, int]] = []
    with observed_charges(
        lambda item, nbytes, planned: taken.append((item, nbytes, planned))
    ):
        result = call()
    return result, taken


#: The Python objects of a call's frames and of its ledgers and of the
#: observer that follows their charges, which no ledger plans: what an
#: interval may hold beyond its plan in an audit.
AUDIT_ALLOWANCE = 64 * 1024


def step_charge(call: Callable[[], Any], item: str) -> tuple[int, int]:
    """The bytes of the first charge of ``item`` that ``call`` takes, and what its ledger then plans.

    A test pins a step's own cost by its bytes, and the smallest budget at
    which the step runs by the plan, as :func:`charges` reports them.
    """
    _result, taken = charges(call)
    for charged, nbytes, planned in taken:
        if charged == item:
            return nbytes, planned
    raise AssertionError(f"No charge of {item!r} among {[entry[0] for entry in taken]}")


def audited(
    call: Callable[[], Any], *, allowance: int = AUDIT_ALLOWANCE
) -> tuple[Any, list[tuple[str, int, int]]]:
    """Run ``call`` and return its result with each charge whose traced interval passed what its ledger planned.

    tracemalloc traces the call. From each charge or release of a memory
    ledger to the next, the most bytes traced beside what the process held
    when the call began must stay within what the ledger planned at the
    first of them: what it held then, beside what its last read may still
    hold, and that charge's transient peak, as ``observed_charges`` reports
    it. The last interval runs to the end of the call; what runs before the
    first charge belongs to no ledger. ``allowance`` covers the Python
    objects of the call's frames and of the ledger itself: each interval
    that passed its plan by more is returned as the item that planned it,
    the bytes planned and the bytes traced. Run ``call`` once before, so
    that one-off costs such as lazy imports and the interpreter's string
    table do not count.
    """
    from cytearc.utils.memory_budget import observed_charges

    overruns: list[tuple[str, int, int]] = []
    pending: list[tuple[str, int]] = []

    def observe(item: str, _nbytes: int, planned: int) -> None:
        traced = tracemalloc.get_traced_memory()[1] - start
        if pending and traced > pending[0][1] + allowance:
            overruns.append((pending[0][0], pending[0][1], traced))
        tracemalloc.reset_peak()
        pending[:] = [(item, planned)]

    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        with observed_charges(observe):
            result = call()
        observe("the end of the call", 0, 0)
    finally:
        tracemalloc.stop()
    return result, overruns


def interval_peaks(
    call: Callable[[], Any],
) -> tuple[Any, list[tuple[str, int, int]]]:
    """Run ``call`` and return its result with every charge's item, plan and traced interval, in order.

    Each entry is a charge or release of a memory ledger, ``"a release"`` for
    a release, what its ledger planned from then until the next charge, and
    the most bytes traced in that interval beside what the process held when
    the call began, as :func:`audited` traces them. A test bounds a step's
    charge from below by comparing the plan of its interval with what was
    traced there, beside the bound from above that :func:`audited` checks.
    Run ``call`` once before, as for :func:`audited`.
    """
    from cytearc.utils.memory_budget import observed_charges

    intervals: list[tuple[str, int, int]] = []
    pending: list[tuple[str, int]] = []

    def observe(item: str, _nbytes: int, planned: int) -> None:
        traced = tracemalloc.get_traced_memory()[1] - start
        if pending:
            intervals.append((pending[0][0], pending[0][1], traced))
        tracemalloc.reset_peak()
        pending[:] = [(item, planned)]

    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        with observed_charges(observe):
            result = call()
        observe("the end of the call", 0, 0)
    finally:
        tracemalloc.stop()
    return result, intervals


def audit_at(
    path: str | Path, call: Callable[[Any], Any], budget: int, **options: Any
) -> Any:
    """Check that ``call`` holds no more than its ledgers plan from each charge to the next, on the store at ``path`` opened at ``budget``.

    The store is opened with ``options``, and ``call`` runs once untraced and
    then under :func:`audited`, with every read's work held until the next
    read ends, in one validation scope, so that the second run's checks of its
    inputs are those of a store already open. Every interval must stay within
    its plan and the audit's allowance.

    Returns:
        The audited call's result.
    """
    from cytearc.storage.validation_scope import validation_scope

    store = open_store(path, mem_budget=budget, **options)
    with validation_scope():
        call(store)
        with hold_finished_reads(*ZARR_READS):
            result, overruns = audited(lambda: call(store))
    assert overruns == [], f"Intervals passed their plans: {overruns}"
    return result


def _needed_bytes(refusal: BaseException) -> int:
    """Return the ``needed_bytes`` of a ``MemoryBudgetError``; any other refusal fails."""
    from cytearc.utils.memory_budget import MemoryBudgetError

    assert isinstance(refusal, MemoryBudgetError), refusal
    return refusal.needed_bytes


def audited_least(path: str | Path, call: Callable[[Any], Any], **options: Any) -> int:
    """Find the smallest budget that runs ``call`` on the store at ``path`` from its refusals, and audit ``call`` there as :func:`audit_at` does.

    The refusals are followed untraced, up to 100 of them, on stores opened
    with ``options``, each by the ``needed_bytes`` of its ``MemoryBudgetError``,
    whatever its words, as the reads of cell-aligned artifacts keep their own.

    Returns:
        The budget.
    """
    budget, _refusals, _result, _peak = budget_that_runs(
        path,
        call,
        refusal_budget=_needed_bytes,
        max_refusals=100,
        traced=False,
        **options,
    )
    audit_at(path, call, budget, **options)
    return budget


#: The methods through which a read of a stored Zarr array's values goes, by
#: index or by selection, for :func:`hold_finished_reads` to end a read at
#: each: those of the synchronous array, and those of its ``async_array``,
#: through which the feature stream and the count layout read. A synchronous
#: read runs its selection without calling these asynchronous methods, so
#: no read is counted twice.
ZARR_READS: tuple[tuple[Any, str], ...] = tuple(
    (zarr.Array, name)
    for name in (
        "get_basic_selection",
        "get_orthogonal_selection",
        "get_coordinate_selection",
        "get_mask_selection",
        "get_block_selection",
    )
) + tuple(
    (zarr.AsyncArray, name)
    for name in (
        "getitem",
        "get_orthogonal_selection",
        "get_coordinate_selection",
        "get_mask_selection",
    )
)


#: The methods through which a write of a Zarr array's values goes, by index
#: or by selection. Each write of a text chunk is one call, so holding them
#: with :func:`hold_finished_reads` keeps the work of each write, the block
#: that Zarr encoded and the encoded chunk, until the next write ends, as a
#: busy machine can; a whole array of numbers written in one call keeps every
#: chunk until the next write, as no pool thread does.
ZARR_WRITES: tuple[tuple[Any, str], ...] = tuple(
    (zarr.Array, name)
    for name in (
        "set_basic_selection",
        "set_orthogonal_selection",
        "set_coordinate_selection",
        "set_mask_selection",
        "set_block_selection",
    )
)


#: An attribute that an owner does not hold itself.
_ABSENT = object()


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
    pairs, and ends when the call returns or raises; a read through a
    coroutine function, such as ``zarr.AsyncArray.getitem``, ends when its
    coroutine finishes. Work belongs to the read under way when a pool thread
    starts it, so work that finishes after its read returned stays held, and
    work that starts while no named read runs, such as the writes after an
    operation's last read, belongs to no read and is let go as it would be
    without the hold. Synchronous named reads run one at a time: one that
    begins while another runs, in any thread, fails. Reads through
    coroutines may overlap, as a stream's reads ahead do; work that a pool
    thread starts while several run belongs to the one that began last, and
    each read's work is held until the read that ends after it ends. With no
    function named, the
    caller ends each read by calling the function that this context gives,
    and all work belongs to the read under way. The last read stays held until
    the context closes, so a test holds the reads of the step whose estimate
    counts the read before, not of a whole operation with the writes after its
    last read.

    The hold applies to every thread pool of the process, so a test that must
    see only its own reads traces them in a fresh process, which imports this
    helper as an in-process test does.
    """
    lock = threading.Lock()
    # Each read has an id: a named read the next when it begins, and without
    # named reads the read under way the count of reads ended. A read's work
    # is held by its id until the read that ends after it ends.
    next_id = [0]
    under_way = [0]
    # The ids of the named reads in flight, of the synchronous ones, and of the
    # read that ended last; the reads whose work was let go; and whether the
    # context has closed.
    running: list[int] = []
    running_sync = [0]
    last_ended: list[int | None] = [None]
    let_go: set[int] = set()
    closed = [False]
    held: dict[int, list[object]] = {}
    run = pool_thread._WorkItem.run

    def run_and_hold(item: object, *args: Any) -> None:
        with lock:
            if not reads:
                owner: int | None = under_way[0]
            else:
                # Work belongs to the newest named read in flight.
                owner = max(running) if running else None
        run(item, *args)  # type: ignore[arg-type]
        with lock:
            if owner is not None and not closed[0] and owner not in let_go:
                held.setdefault(owner, []).append(item)

    def ended(read_id: int) -> None:
        """Let go of the work of the read that ended before ``read_id``."""
        with lock:
            before = last_ended[0]
            if before is not None:
                held.pop(before, None)
                let_go.add(before)
            last_ended[0] = read_id

    def end_read() -> None:
        with lock:
            read_id = under_way[0]
            under_way[0] += 1
        ended(read_id)

    def begin(synchronous: bool, function: Callable[..., Any]) -> int:
        with lock:
            if synchronous:
                assert not running_sync[0], (
                    f"{getattr(function, '__qualname__', function)} began while "
                    "another named read ran; hold_finished_reads holds one read "
                    "at a time"
                )
                running_sync[0] += 1
            read_id = next_id[0]
            next_id[0] += 1
            running.append(read_id)
            return read_id

    def finish(read_id: int, synchronous: bool) -> None:
        with lock:
            running.remove(read_id)
            if synchronous:
                running_sync[0] -= 1
        ended(read_id)

    def ending(function: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(function):

            async def read_async(*args: Any, **kwargs: Any) -> Any:
                read_id = begin(False, function)
                try:
                    return await function(*args, **kwargs)
                finally:
                    finish(read_id, False)

            return read_async

        def read(*args: Any, **kwargs: Any) -> Any:
            read_id = begin(True, function)
            try:
                return function(*args, **kwargs)
            finally:
                finish(read_id, True)

        return read

    # Each owner's own attribute, or none where it inherits the function, comes
    # back when the context closes, and a static or class method stays one.
    replaced = []
    for owner, name in reads:
        static = inspect.getattr_static(owner, name)
        if isinstance(static, staticmethod | classmethod):
            wrapper: object = type(static)(ending(static.__func__))
        else:
            wrapper = ending(getattr(owner, name))
        replaced.append((owner, name, vars(owner).get(name, _ABSENT)))
        setattr(owner, name, wrapper)
    pool_thread._WorkItem.run = run_and_hold  # type: ignore[method-assign]
    try:
        yield end_read
    finally:
        pool_thread._WorkItem.run = run  # type: ignore[method-assign]
        for owner, name, own in reversed(replaced):
            if own is _ABSENT:
                delattr(owner, name)
            else:
                setattr(owner, name, own)
        with lock:
            closed[0] = True
            held.clear()
