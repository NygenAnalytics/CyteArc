"""Charge what an operation holds against its memory budget before it allocates.

An operation creates one :class:`MemoryLedger` with its budget and charges it
before each step allocates: what the step keeps, and the transient peak of
what it builds and lets go. The ledger refuses a charge that would pass the
budget with a :class:`MemoryBudgetError` that names the step, the bytes that
it needs and the budget, so a call is refused before it reads or writes what
would not fit. Planners that size work outside any ledger refuse through
:func:`memory_refusal`, so every refusal reads one way.

The shared costs below give the bytes of the structures that many steps
build: Python strings, fixed-width text, object arrays, sorts and hash
tables. ``tests/test_memory_costs.py`` bounds each one from both sides.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

__all__ = [
    "MemoryBudgetError",
    "MemoryLedger",
    "fixed_text_bytes",
    "hash_table_bytes",
    "memory_refusal",
    "object_array_bytes",
    "observed_charges",
    "python_text_bytes",
    "sort_bytes",
]


class MemoryBudgetError(MemoryError):
    """A step of a call would hold more memory than the call's budget.

    Attributes:
        needed_bytes: The bytes that the call holds at the step that was
            refused: what it held before, what its last read may still hold,
            and the step's own charge. A budget of at least this many bytes
            lets the call pass that step.
        budget_bytes: The memory budget that the step would pass.
        item: What the step's charge was for, such as ``"the cells' IDs"``.
    """

    def __init__(
        self, message: str, *, needed_bytes: int, budget_bytes: int, item: str
    ) -> None:
        super().__init__(message)
        self.needed_bytes = int(needed_bytes)
        self.budget_bytes = int(budget_bytes)
        self.item = str(item)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_memory_budget_error,
            (
                type(self),
                str(self.args[0]),
                self.needed_bytes,
                self.budget_bytes,
                self.item,
            ),
            self.__dict__,
        )


def _restore_memory_budget_error(
    error_type: type[MemoryBudgetError],
    message: str,
    needed_bytes: int,
    budget_bytes: int,
    item: str,
) -> MemoryBudgetError:
    """Rebuild a pickled MemoryBudgetError, whose fields are keywords."""
    error = MemoryBudgetError.__new__(error_type)
    MemoryBudgetError.__init__(
        error,
        message,
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
    )
    return error


def memory_refusal(
    doing: str,
    item: str,
    needed_bytes: int,
    budget_bytes: int,
    *,
    remedy: str | None = None,
) -> MemoryBudgetError:
    """Return the refusal of a step that needs more bytes than the memory budget.

    The message reads "{doing} holds about {needed_bytes} bytes for {item},
    more than the memory budget of {budget_bytes} bytes. Raise mem_budget to
    at least {needed_bytes} bytes.", with ", or {remedy}" before the closing
    period when a step can also be made smaller.

    Args:
        doing: What the call does, such as ``"Importing contigs for 8 cells"``.
        item: What the refused charge is for.
        needed_bytes: The bytes that the call needs at that step.
        budget_bytes: The memory budget.
        remedy: Another way to fit the budget, such as ``"pass a smaller
            cell_selection"``, without a closing period.

    Raises:
        TypeError: If a number of bytes is not an integer.
        ValueError: If a number of bytes is negative, or ``needed_bytes``
            does not pass ``budget_bytes``.
    """
    for value, name in ((needed_bytes, "needed_bytes"), (budget_bytes, "budget_bytes")):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(f"{name} must be a number of bytes, not {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be at least 0, not {value}")
    if needed_bytes <= budget_bytes:
        raise ValueError(
            f"A refusal needs more than the budget of {budget_bytes} bytes, not {needed_bytes}"
        )
    needed_bytes, budget_bytes = int(needed_bytes), int(budget_bytes)
    closing = f"Raise mem_budget to at least {needed_bytes} bytes"
    if remedy:
        closing += f", or {remedy}"
    return MemoryBudgetError(
        f"{doing} holds about {needed_bytes} bytes for {item}, more than the "
        f"memory budget of {budget_bytes} bytes. {closing}.",
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
    )


#: Called at each charge and release of any ledger while a test observes
#: them, as :func:`observed_charges` describes; None otherwise.
_observer: Callable[[str, int, int], None] | None = None


@contextmanager
def observed_charges(observer: Callable[[str, int, int], None]) -> Iterator[None]:
    """Call ``observer`` at each charge and release of every ledger in this context.

    ``observer(item, nbytes, planned)`` gets the item of the charge, or ``"a
    release"``, its bytes, and ``planned``, the bytes that the ledger allows
    from then until its next charge: what it holds and, for a transient peak,
    that peak. An audit compares the traced peak of each interval with what
    was planned for it. Production code runs no observer.
    """
    global _observer
    previous, _observer = _observer, observer
    try:
        yield
    finally:
        _observer = previous


class MemoryLedger:
    """What one call holds, charged against its memory budget before each step.

    A step charges what it will keep with :meth:`hold`, which it lets go with
    :meth:`release`, and what it builds and lets go with :meth:`peak`. A read
    of a stored array records with :meth:`leave` what Zarr's codec threads
    may still hold after it returns, which every later charge counts until
    the next read replaces it. A budget of None never refuses.

    Attributes:
        budget: The memory budget in bytes, or None for none.
        doing: What the call does, as its refusals name it; a step can set
            its own.
        remedy: Another way to fit the budget that refusals name, or None;
            a charge can pass its own.
        codec_threads: The threads of Zarr's codec pool that the call's reads
            decode on: the store's ``nthreads``, whatever the machine. The
            reads that the ledger charges decode on a pool of exactly this
            many threads, as ``storage.async_execution.zarr_decoding_threads``
            gives them. The pool spreads a read's chunks over its threads
            even one chunk at a time, and each thread can still hold the last
            chunk it decoded after the read returns.
        charged: The bytes held by charges not released.
        left: What the last read or write may still hold.
    """

    def __init__(
        self,
        budget: int | None,
        doing: str,
        *,
        remedy: str | None = None,
        codec_threads: int = 1,
    ) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise ValueError(
                f"A memory budget must be a non-negative number of bytes or None, not {budget!r}"
            )
        if (
            isinstance(codec_threads, bool)
            or not isinstance(codec_threads, int)
            or codec_threads < 1
        ):
            raise ValueError(
                f"codec_threads must be a number of threads of at least 1, not {codec_threads!r}"
            )
        self.budget = budget
        self.doing = doing
        self.remedy = remedy
        self.codec_threads = codec_threads
        self.charged = 0
        self.left = 0

    def holding_threads(self, chunks: int) -> int:
        """The codec threads that can each still hold a chunk after a read that decoded ``chunks`` chunks.

        Each of the call's codec threads can hold the last chunk it decoded,
        and no read leaves more held chunks than it decoded.
        """
        return max(1, min(self.codec_threads, int(chunks)))

    @property
    def held_bytes(self) -> int:
        """The bytes charged and not released, beside what the last read may still hold."""
        return self.charged + self.left

    @property
    def room(self) -> int | None:
        """The bytes that a step may still take: the budget less ``held_bytes``, None without a budget."""
        if self.budget is None:
            return None
        return max(0, self.budget - self.held_bytes)

    def refuse_beyond(self, item: str, needed: int, remedy: str | None = None) -> None:
        """Raise :class:`MemoryBudgetError` if holding ``needed`` bytes in all would pass the budget."""
        if self.budget is not None and needed > self.budget:
            raise memory_refusal(
                self.doing,
                item,
                needed,
                self.budget,
                remedy=self.remedy if remedy is None else remedy,
            )

    def hold(self, item: str, nbytes: int, *, remedy: str | None = None) -> int:
        """Charge ``nbytes`` that the call keeps for ``item``, refusing them beyond the budget.

        Returns:
            ``nbytes``, which :meth:`release` lets go.
        """
        nbytes = _bytes(nbytes)
        self.refuse_beyond(item, self.held_bytes + nbytes, remedy)
        self.charged += nbytes
        self._observe(item, nbytes, self.held_bytes)
        return nbytes

    def peak(self, item: str, nbytes: int, *, remedy: str | None = None) -> None:
        """Check that ``nbytes`` more, which a step builds and lets go, fit beside what is held."""
        nbytes = _bytes(nbytes)
        needed = self.held_bytes + nbytes
        self.refuse_beyond(item, needed, remedy)
        self._observe(item, nbytes, needed)

    def release(self, nbytes: int) -> None:
        """Let go of ``nbytes`` that :meth:`hold` charged, once they are freed."""
        nbytes = _bytes(nbytes)
        if nbytes > self.charged:
            raise ValueError(
                f"Releasing {nbytes} bytes, more than the {self.charged} bytes charged"
            )
        self.charged -= nbytes
        self._observe("a release", nbytes, self.held_bytes)

    def leave(self, nbytes: int) -> None:
        """Record what the read or write that just ended may still hold, in place of the last one's.

        The read's own charge planned for it, so the next charge counts it.
        """
        self.left = _bytes(nbytes)

    def _observe(self, item: str, nbytes: int, planned: int) -> None:
        if _observer is not None:
            _observer(item, nbytes, planned)


def _bytes(nbytes: int) -> int:
    resolved = int(nbytes)
    if resolved < 0:
        raise ValueError(
            f"A charge must be a non-negative number of bytes, not {nbytes}"
        )
    return resolved


# CPython's compact strings: the bytes of a string beside one character
# width for each character, by the widest code point the string holds. An
# ASCII string keeps its trailing NUL in its header.
_STRING_KINDS = ((0x7F, 41, 1), (0xFF, 57, 1), (0xFFFF, 58, 2), (0x10FFFF, 60, 4))


def python_text_bytes(n_values: int, characters: int, *, widest: int = 0x7F) -> int:
    """Return the bytes of ``n_values`` Python strings that hold ``characters`` characters in all.

    Each string takes a header and one, two or four bytes per character,
    by the widest code point among its characters: ASCII strings 41 bytes
    and one per character, strings up to U+00FF 57 and one, up to U+FFFF 58
    and two, and beyond 60 and four. Every string is counted at the kind of
    ``widest``, the widest code point of any of them.
    """
    for limit, header, width in _STRING_KINDS:
        if widest <= limit:
            return int(n_values) * header + width * int(characters)
    raise ValueError(f"A code point cannot exceed U+10FFFF, not {widest:#x}")


def fixed_text_bytes(n_values: int, width: int) -> int:
    """Return the bytes of ``n_values`` values of NumPy or Zarr text ``width`` characters wide.

    Fixed-width text stores every value at the width of the widest, four
    bytes per character.
    """
    return 4 * int(width) * int(n_values)


def object_array_bytes(n_values: int) -> int:
    """Return the bytes of an object array's references to ``n_values`` values, beside the values."""
    return 8 * int(n_values)


def sort_bytes(
    n_values: int, itemsize: int, *, stable: bool = False, order: bool = False
) -> int:
    """Return the bytes that sorting ``n_values`` values of ``itemsize`` bytes holds, beside them.

    ``np.sort`` returns a copy, and its stable sort holds half as much again
    while it merges, or, on values of one or two bytes, which it sorts by
    radix, as much again; with ``order``, ``np.argsort`` returns the order as
    int64, and its stable sort holds half of that again, or by radix as much
    again.
    """
    n_values, itemsize = int(n_values), int(itemsize)
    radix = itemsize <= 2
    if order:
        if not stable:
            return 8 * n_values
        return (16 if radix else 12) * n_values
    copy = n_values * itemsize
    if not stable:
        return copy
    return 2 * copy if radix else copy + copy // 2


def _power_of_two(n: int) -> int:
    """The smallest power of two of at least ``n``, and at least 4."""
    return max(4, 1 << max(0, int(n) - 1).bit_length())


def _buckets(n_keys: int) -> int:
    """The buckets of a khash table that holds ``n_keys`` keys at most 77% full: a power of two."""
    candidate = _power_of_two(n_keys)
    return 2 * candidate if candidate * 0.77 + 0.5 < n_keys else candidate


#: pandas sizes a hash table for at most this many keys before it hashes
#: them, and grows it from there as distinct keys arrive.
_SIZE_HINT_LIMIT = (1 << 20) + 7


def hash_table_bytes(n_values: int, *, n_distinct: int | None = None) -> int:
    """Return the bytes of the hash table that pandas builds to hash ``n_values`` values.

    ``pd.factorize``, ``pd.unique`` and ``Index.get_indexer`` hash their
    values into a khash table of 16 bytes per bucket, a key and a position,
    and a bit per bucket for whether it is empty, with a power of two of
    buckets that are at most 77% full. pandas sizes it
    for every value given, up to about a million, whether or not the values
    repeat, and doubles it in place while its distinct keys outgrow it.
    ``n_distinct`` bounds the distinct values, all of them when None. Beside the table, pandas
    returns codes, and the distinct values, which it gathers in a buffer
    that grows four times over when it fills, and an object array hashes the
    values' strings, which fixed-width text becomes first.
    """
    n_values = int(n_values)
    distinct = n_values if n_distinct is None else min(int(n_distinct), n_values)
    initial = _power_of_two(min(_buckets(n_values), _SIZE_HINT_LIMIT))
    buckets = max(initial, _buckets(distinct))
    return 16 * buckets + buckets // 8
