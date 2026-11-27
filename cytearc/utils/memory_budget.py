"""Charge an estimate of what an operation holds against its memory budget before it allocates.

An operation creates one :class:`MemoryLedger` with its budget and charges it
before each step allocates: an estimate of what the step keeps, and of the
transient peak of what it builds and lets go. The estimates count the arrays
that grow with the data, such as blocks, coordinates, graphs and per-cell IDs
and labels, and ``READ_ALLOWANCE_BYTES`` for each read of stored arrays in
flight. They leave out the interpreter, the other objects and buffers of the
libraries, and other allocations that do not grow with the data, so a step can
hold a little more than its charge. The ledger refuses a charge that would pass
the budget with a :class:`MemoryBudgetError` that names the step, the bytes
that it needs and the budget, so a call is refused before it reads or writes
what would not fit. Every refusal closes naming the budget of
:func:`suggested_budget`, a quarter more than the bytes it needs rounded up to
a whole MiB, at which a rerun of the call passes the refused step. Planners
that size work outside any ledger raise the same error with the same fields in
their own words, and :func:`memory_refusal` words the refusal of a step that
has none.

The shared costs below estimate the structures that many steps build, at a
cost for each item that bounds what the structure holds at any size, with the
growth of pandas' and NumPy's buffers: Python strings, fixed-width text,
object arrays, sorts, hash tables and index lookups.
"""

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "READ_ALLOWANCE_BYTES",
    "MemoryBudgetError",
    "MemoryLedger",
    "fixed_text_bytes",
    "factorize_bytes",
    "hash_table_bytes",
    "held_array_bytes",
    "index_lookup_bytes",
    "admitted_need",
    "memory_refusal",
    "object_array_bytes",
    "python_text_bytes",
    "refusal_words",
    "sort_bytes",
    "suggested_budget",
]


#: What one read of stored arrays in flight holds beside the arrays that grow
#: with the data: the objects of Zarr's read and of the chunks it decodes,
#: NumPy's buffers of a read that takes values by index arrays, and the
#: buffer of a file read in part, as a read of part of a shard is, which
#: Python sizes by the block of the file system. On a local file system, a
#: read that decodes up to 10 chunks at once, Zarr's default async
#: concurrency, held at most 50 KB of these on Python 3.13 and 97 KB on
#: Python 3.14, whose buffer of a file read in part is larger; a file system
#: with larger blocks gives each read a larger buffer. Each further chunk
#: that a read decodes at once holds up to 12.1 KB more, which the read
#: geometry charges as ``READ_DECODE_BYTES``.
READ_ALLOWANCE_BYTES = 128 * 1024


#: A refusal names a budget of a whole number of these bytes, a mebibyte, so
#: that it reads as a budget that a user would type, such as ``"3M"``.
_NAMED_BUDGET_UNIT = 1024**2


def suggested_budget(needed_bytes: int) -> int:
    """Return the memory budget that the refusal of a step that needs ``needed_bytes`` names.

    The budget is a quarter more than the bytes that the step needs, rounded
    up to a whole MiB. The quarter leaves room for a step whose blocks take a
    share of the budget, and so grow with it, and for the steps after it,
    which can need a little more than it, such as a read beside the
    allowance of its read; at least 1 MiB.
    """
    needed = max(1, int(needed_bytes))
    with_headroom = needed + needed // 4
    return -(-with_headroom // _NAMED_BUDGET_UNIT) * _NAMED_BUDGET_UNIT


class MemoryBudgetError(MemoryError):
    """A step of a call would hold more memory than the call's budget, by the estimate that the step charges.

    The estimate counts the arrays that grow with the data and leaves out the
    memory of the interpreter and its libraries, so the budget plans the work
    rather than capping the process. The message closes with "Raise
    mem_budget to at least {suggested} MiB.", with ", or {remedy}" before
    the period when the step can also be made smaller.

    Attributes:
        needed_bytes: The estimated bytes that the call holds at the step
            that was refused: what it held before and the step's own charge.
            A step whose blocks take a share of the budget can need more at
            a larger budget; one that checks its plan at the budget that it
            names gives what it needs at the largest budget that the check
            refused. A count write whose producer hands it a batch beyond the
            producer's reservation names instead what the write would
            reserve with the batch.
        budget_bytes: The memory budget that the step would pass.
        item: What the step's charge was for, such as ``"the cells' IDs"``.
        suggested_budget_bytes: The budget that the refusal names: a
            quarter more than ``needed_bytes``, rounded up to a whole MiB. A
            rerun of the call with this budget passes the refused step, and
            a later step can need more.
        remedy: Another way to fit the budget that the message names, such
            as ``"pass a smaller cell_selection"``, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        needed_bytes: int,
        budget_bytes: int,
        item: str,
        remedy: str | None = None,
    ) -> None:
        super().__init__(message)
        self.needed_bytes = int(needed_bytes)
        self.budget_bytes = int(budget_bytes)
        self.item = str(item)
        self.remedy = remedy or None
        self.suggested_budget_bytes = suggested_budget(self.needed_bytes)

    def __str__(self) -> str:
        words = str(self.args[0]) if self.args else ""
        if words and not words.endswith("."):
            words += "."
        closing = (
            "Raise mem_budget to at least "
            f"{self.suggested_budget_bytes // _NAMED_BUDGET_UNIT} MiB"
        )
        if self.remedy:
            closing += f", or {self.remedy}"
        return f"{words} {closing}.".lstrip()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_memory_budget_error,
            (
                type(self),
                str(self.args[0]),
                self.needed_bytes,
                self.budget_bytes,
                self.item,
                self.remedy,
            ),
            self.__dict__,
        )


def _restore_memory_budget_error(
    error_type: type[MemoryBudgetError],
    message: str,
    needed_bytes: int,
    budget_bytes: int,
    item: str,
    remedy: str | None,
) -> MemoryBudgetError:
    """Rebuild a pickled MemoryBudgetError, whose fields are keywords."""
    error = MemoryBudgetError.__new__(error_type)
    MemoryBudgetError.__init__(
        error,
        message,
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
        remedy=remedy,
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
    at least {suggested} MiB.", with ", or {remedy}" before the closing
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
    return MemoryBudgetError(
        f"{doing} holds about {needed_bytes} bytes for {item}, more than the "
        f"memory budget of {budget_bytes} bytes.",
        needed_bytes=needed_bytes,
        budget_bytes=budget_bytes,
        item=item,
        remedy=remedy,
    )


def refusal_words(error: MemoryBudgetError) -> str:
    """Return the message of ``error`` without the sentences that close it with its bytes and its budget.

    A step that refuses inside a larger one, such as a conversion beside the
    data it converts, can close its own words with "It needs at least
    {needed_bytes} bytes." The larger step keeps the words before it and
    states the bytes of its whole budget after them, and its own refusal
    names the budget that they need.
    """
    return str(error.args[0]).removesuffix(
        f" It needs at least {error.needed_bytes} bytes."
    )


class MemoryLedger:
    """What one call holds, charged against its memory budget before each step.

    A step charges what it will keep with :meth:`hold`, which it lets go with
    :meth:`release`, and what it builds and lets go with :meth:`peak`. A
    budget of None never refuses.

    Attributes:
        budget: The memory budget in bytes, or None for none.
        doing: What the call does, as its refusals name it; a step can set
            its own.
        remedy: Another way to fit the budget that refusals name, or None;
            a charge can pass its own.
        codec_threads: The threads of Zarr's codec pool that the call's reads
            decode on: the store's ``nthreads``, whatever the machine, as
            ``storage.async_execution.zarr_decoding_threads`` gives them.
        charged: The bytes held by charges not released.
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

    @property
    def held_bytes(self) -> int:
        """The bytes charged and not released."""
        return self.charged

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
        return nbytes

    def peak(self, item: str, nbytes: int, *, remedy: str | None = None) -> None:
        """Check that ``nbytes`` more, which a step builds and lets go, fit beside what is held."""
        nbytes = _bytes(nbytes)
        self.refuse_beyond(item, self.held_bytes + nbytes, remedy)

    def release(self, nbytes: int) -> None:
        """Let go of ``nbytes`` that :meth:`hold` charged, once they are freed."""
        nbytes = _bytes(nbytes)
        if nbytes > self.charged:
            raise ValueError(
                f"Releasing {nbytes} bytes, more than the {self.charged} bytes charged"
            )
        self.charged -= nbytes


def _bytes(nbytes: int) -> int:
    resolved = int(nbytes)
    if resolved < 0:
        raise ValueError(
            f"A charge must be a non-negative number of bytes, not {nbytes}"
        )
    return resolved


#: The bytes of a Python string beside its characters: 41 for ASCII text, and
#: at most 60 for text beyond ASCII.
_ASCII_STRING_BYTES = 41
_STRING_BYTES = 60


def python_text_bytes(n_values: int, characters: int, *, widest: int = 0x7F) -> int:
    """Return the bytes of ``n_values`` Python strings that hold ``characters`` characters in all.

    An ASCII string takes 41 bytes beside its characters, a byte each. Where
    ``widest``, the widest code point of any string, is beyond ASCII, a
    string takes at most 60 bytes beside its characters, which take a byte
    each, two when ``widest`` is beyond U+00FF and four when it is beyond
    U+FFFF.
    """
    if widest > 0x10FFFF:
        raise ValueError(f"A code point cannot exceed U+10FFFF, not {widest:#x}")
    if widest <= 0x7F:
        return int(n_values) * _ASCII_STRING_BYTES + int(characters)
    width = 1 if widest <= 0xFF else 2 if widest <= 0xFFFF else 4
    return int(n_values) * _STRING_BYTES + width * int(characters)


def fixed_text_bytes(n_values: int, width: int) -> int:
    """Return the bytes of ``n_values`` values of NumPy or Zarr text ``width`` characters wide.

    Fixed-width text stores every value at the width of the widest, four
    bytes per character.
    """
    return 4 * int(width) * int(n_values)


def object_array_bytes(n_values: int) -> int:
    """Return the bytes of an object array's references to ``n_values`` values, beside the values."""
    return 8 * int(n_values)


# The most budgets at which a step whose need grows with its budget is checked
# before its refusal is passed on: each is at least a quarter larger than the
# one before, so sixteen span a need 35 times the first.
_ADMISSION_CHECKS = 16


def admitted_need(admit: Callable[[int], object], refused: MemoryBudgetError) -> int:
    """Return a need of the step that ``admit`` refused, whose suggested budget it passes.

    ``admit(budget)`` checks the step at ``budget``, planning it but running
    nothing, and raises :class:`MemoryBudgetError` for a budget that it
    refuses. A step whose blocks take a share of the budget needs more at a
    larger budget, which can pass even the headroom of
    :func:`suggested_budget`: the step is checked at the suggested budget of
    the need that ``refused`` names, then at that of each need that a check
    names in turn, and the need whose suggested budget passes is returned,
    for the step's refusal to name.

    A step that refuses 16 suggested budgets in a row keeps the need that
    ``refused`` named, as one that passes at no budget would.
    """
    needed = int(refused.needed_bytes)
    for _ in range(_ADMISSION_CHECKS):
        try:
            admit(suggested_budget(needed))
        except MemoryBudgetError as error:
            needed = int(error.needed_bytes)
        else:
            return needed
    return int(refused.needed_bytes)


def held_array_bytes(values: np.ndarray) -> int:
    """Return the bytes of an array, with those of the objects that an object array holds.

    Each object counts once, as ``sys.getsizeof`` measures it, however many
    elements refer to it, as the rows of a column share the strings that
    pandas reads once. A reader measures the text it keeps once, rather
    than at every step that charges it.
    """
    held = int(values.nbytes)
    if values.dtype.kind == "O":
        flat = values.ravel()
        _ids, first = np.unique(
            np.fromiter(map(id, flat), dtype=np.intp, count=flat.size),
            return_index=True,
        )
        held += sum(map(sys.getsizeof, flat[first]))
    return held


def sort_bytes(
    n_values: int, itemsize: int, *, stable: bool = False, order: bool = False
) -> int:
    """Return the bytes that sorting ``n_values`` values of ``itemsize`` bytes holds, beside them.

    ``np.sort`` returns a copy, and with ``order`` ``np.argsort`` returns the
    order as int64; a stable sort holds as much again while it merges.
    """
    n_values, itemsize = int(n_values), int(itemsize)
    held = 8 * n_values if order else n_values * itemsize
    return 2 * held if stable else held


#: pandas' hash table of strings takes 24 bytes and a bit of flags a bucket,
#: and a table of other values 16 bytes and the bit, as each table's
#: ``sizeof`` gives them.
_HASH_BUCKET_BYTES = 24
#: The most values that pandas sizes a hash table for before it hashes them.
_HASH_SIZE_HINT_LIMIT = (1 << 20) + 7
#: pandas infers the type of an object array that it makes an index of, or
#: looks up, from arrays of each type that the values could take, 42 bytes a
#: value.
_INDEX_INFERENCE_BYTES = 42


def hash_table_bytes(n_values: int, *, n_distinct: int | None = None) -> int:
    """Return the bytes of the hash table that pandas builds to hash ``n_values`` values, ``n_distinct`` of them distinct.

    pandas sizes the table for the values given, at most about a million,
    and the table grows as the distinct values fill it, as all of the values
    do unless ``n_distinct`` says fewer. khash keeps a power of two of
    buckets, at most 77% full, as pandas' ``kh_needed_n_buckets`` gives them:
    1.3 to 2.6 buckets a value. Each bucket is charged the 24 bytes that a
    table of strings takes; a table of other values takes 16.
    """
    n_values = max(0, int(n_values))
    distinct = n_values if n_distinct is None else min(int(n_distinct), n_values)
    filled = max(min(n_values, _HASH_SIZE_HINT_LIMIT), distinct)
    buckets = next(
        b for b in (1 << k for k in range(2, 64)) if int(b * 0.77 + 0.5) >= filled
    )
    return _HASH_BUCKET_BYTES * buckets + buckets // 8


def factorize_bytes(n_values: int, *, n_distinct: int | None = None) -> int:
    """Return the bytes that ``pd.factorize`` or ``pd.unique`` holds for ``n_values`` values, beside the codes and distinct values that it returns.

    Beside its hash table, as :func:`hash_table_bytes` counts it, pandas
    works with two arrays of 8 bytes a value, and gathers the distinct
    values in a buffer that doubles for objects and grows four times over
    for numbers before pandas trims it: up to 24 bytes for each distinct
    value beside the 8 of each that it returns. The values are all distinct
    unless ``n_distinct`` says fewer.
    """
    n_values = max(0, int(n_values))
    distinct = n_values if n_distinct is None else min(int(n_distinct), n_values)
    return (
        hash_table_bytes(n_values, n_distinct=n_distinct)
        + 16 * n_values
        + 24 * distinct
    )


def index_lookup_bytes(n_index: int, n_lookups: int, *, objects: bool = True) -> int:
    """Return the bytes that finding ``n_lookups`` values in a pandas index of ``n_index`` distinct values holds, beside the values given.

    pandas copies the values that it makes the index of, 8 bytes each, and
    hashes them into a table, as :func:`hash_table_bytes` counts it. It
    copies the lookups and returns each one's position, 8 bytes each. Before
    it hashes ``objects``, values or lookups in object arrays, it infers
    their type, holding 42 bytes a value.
    """
    n_index, n_lookups = max(0, int(n_index)), max(0, int(n_lookups))
    inferring = _INDEX_INFERENCE_BYTES * max(n_index, n_lookups) if objects else 0
    return 8 * n_index + max(inferring, hash_table_bytes(n_index)) + 16 * n_lookups
