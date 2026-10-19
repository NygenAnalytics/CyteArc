"""Deterministic building blocks of CyteArc's native Leiden backend.

The native backend implements the Leiden algorithm of Traag, Waltman and van
Eck (2019). This module holds the pieces that all of its phases share:

- a counter-based random number generator, in which every draw is a pure
  function of a 64-bit key and a counter (the splitmix64 finaliser used as a
  mixing function), so no generator state is shared between threads;
- a small open-addressing hash accumulator that one worker uses to collect
  edge weights per neighbouring community;
- parallel primitives (prefix sums, stable bucket sorts, conflict-free
  scatter-adds, fixed-block floating-point sums) whose results do not depend on
  the number of worker threads.

Determinism rules followed here and in the other ``_leiden_*`` modules:

- Integer sums are exact in any order, so the parallel primitives only split
  the work between workers.
- A floating-point total is summed over fixed blocks of ``FLOAT_SUM_BLOCK``
  consecutive elements, each block in order, and the block sums are then added
  in block order. The grouping never depends on the worker count.
- Every parallel loop is an explicit worker loop: ``prange`` runs over worker
  indices, each worker visits blocks in a fixed cyclic order, and per-worker
  scratch rows are indexed by that worker index, never by a runtime thread id.
  Kernels check that every scratch array has one row per worker.
- A kernel may switch between a single-thread loop and its parallel loop on
  input size (``parallel_min_items``). Both loops compute the same values, so
  the switch only affects speed.

The constants below are part of the backend's definition. Changing the random
number constants or ``FLOAT_SUM_BLOCK`` changes the labels of every native
Leiden run, so they are frozen. The other block sizes only affect speed.

Reference:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z
"""

import numpy as np
from numba import njit, prange

# Counter-based random numbers. Fixed forever (see the module docstring).
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
MIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)
SEED_SALT = np.uint64(0x6C656964656E3031)
_SHIFT_11 = np.uint64(11)
_SHIFT_27 = np.uint64(27)
_SHIFT_30 = np.uint64(30)
_SHIFT_31 = np.uint64(31)
_UNIT_INTERVAL_SCALE = 1.0 / 9007199254740992.0  # 2**-53

# Stream identifiers of the random phases.
PHASE_MOVE_ORDER = 1
PHASE_GROUPS = 2
PHASE_REFINE = 3

# Inputs smaller than this run the single-thread loop of a kernel (group
# steps below this many nodes or movers; scans and fixed-block sums below 64
# times, bucket sorts below 16 times and scatter-adds below 4 times this
# value). Results do not depend on it; tests lower it to exercise the
# multi-worker loops. 256 was measured faster than 1024 at 4 threads.
PARALLEL_MIN_ITEMS = 256

# Block sizes. Only FLOAT_SUM_BLOCK changes results (the last bits of floating
# totals), so it is frozen; the others only change how work is split.
WORKER_BLOCK = 64
COMPACT_BLOCK = 4096
BUCKET_CHUNK = 16384
BUCKET_DIRECT_KEYS = 4096
SCATTER_RANGES = 256
SCAN_BLOCK = 65536
FLOAT_SUM_BLOCK = 65536

# Marker of an empty hash slot.
EMPTY_SLOT = -1

# Id lists up to this length are sorted by insertion.
INSERTION_SORT_LIMIT = 32


@njit(cache=True, nogil=True)
def mix64(value: np.uint64) -> np.uint64:
    """Return the splitmix64 finaliser of a 64-bit unsigned integer."""
    mixed = (value ^ (value >> _SHIFT_30)) * MIX_MULTIPLIER_1
    mixed = (mixed ^ (mixed >> _SHIFT_27)) * MIX_MULTIPLIER_2
    return mixed ^ (mixed >> _SHIFT_31)


@njit(cache=True, nogil=True)
def derive_key(key: np.uint64, value: int) -> np.uint64:
    """Derive a child key from a key and a non-negative integer."""
    return mix64(key ^ mix64(np.uint64(value) + GOLDEN))


@njit(cache=True, nogil=True)
def stream_key(seed: np.uint64, iteration: int, level: int, phase: int) -> np.uint64:
    """Return the key of one random stream of a run."""
    key = mix64(seed ^ SEED_SALT)
    return derive_key(derive_key(derive_key(key, iteration), level), phase)


@njit(cache=True, nogil=True)
def uniform01(key: np.uint64, counter: int) -> float:
    """Return draw number ``counter`` of stream ``key``, uniform on [0, 1)."""
    mixed = mix64(key + np.uint64(counter) * GOLDEN)
    return np.float64(mixed >> _SHIFT_11) * _UNIT_INTERVAL_SCALE


def hash_capacity(max_distinct: int) -> tuple[int, np.uint64]:
    """Return a hash table capacity and its slot shift for ``max_distinct`` keys.

    The capacity is a power of two, at least 16 and at least twice the number
    of keys plus one, so linear probing always finds an empty slot quickly.
    """
    capacity = 16
    while capacity < 2 * (max_distinct + 1):
        capacity *= 2
    bits = capacity.bit_length() - 1
    return capacity, np.uint64(64 - bits)


@njit(cache=True, nogil=True)
def acc_add(
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    n_used: int,
    key: int,
    value: float,
    shift: np.uint64,
) -> int:
    """Add ``value`` to the entry of ``key`` and return the new used count.

    ``keys`` holds ``EMPTY_SLOT`` in free slots. ``used`` lists the occupied
    slots in insertion order, so a reset only touches those slots.
    """
    mask = keys.shape[0] - 1
    slot = np.int64((np.uint64(key) * GOLDEN) >> shift)
    while True:
        stored = keys[slot]
        if stored == key:
            vals[slot] += value
            return n_used
        if stored == EMPTY_SLOT:
            keys[slot] = key
            vals[slot] = value
            used[n_used] = slot
            return n_used + 1
        slot = (slot + 1) & mask


@njit(cache=True, nogil=True)
def acc_insert(
    keys: np.ndarray,
    used: np.ndarray,
    n_used: int,
    key: int,
    shift: np.uint64,
) -> int:
    """Insert ``key`` into a hash set and return the new used count."""
    mask = keys.shape[0] - 1
    slot = np.int64((np.uint64(key) * GOLDEN) >> shift)
    while True:
        stored = keys[slot]
        if stored == key:
            return n_used
        if stored == EMPTY_SLOT:
            keys[slot] = key
            used[n_used] = slot
            return n_used + 1
        slot = (slot + 1) & mask


@njit(cache=True, nogil=True)
def acc_find(keys: np.ndarray, key: int, shift: np.uint64) -> int:
    """Return the slot of ``key``, or -1 when it is absent."""
    mask = keys.shape[0] - 1
    slot = np.int64((np.uint64(key) * GOLDEN) >> shift)
    while True:
        stored = keys[slot]
        if stored == key:
            return int(slot)
        if stored == EMPTY_SLOT:
            return -1
        slot = (slot + 1) & mask


@njit(cache=True, nogil=True)
def acc_reset(keys: np.ndarray, used: np.ndarray, n_used: int) -> None:
    """Free the occupied slots of a hash table."""
    for index in range(n_used):
        keys[used[index]] = EMPTY_SLOT


@njit(cache=True, nogil=True)
def _sift_down(ids: np.ndarray, root: int, end: int) -> None:
    """Restore the max-heap order of ``ids[:end]`` below ``root``."""
    while True:
        child = 2 * root + 1
        if child >= end:
            return
        if child + 1 < end and ids[child] < ids[child + 1]:
            child += 1
        if ids[root] >= ids[child]:
            return
        held = ids[root]
        ids[root] = ids[child]
        ids[child] = held
        root = child


@njit(cache=True, nogil=True)
def sort_ids(ids: np.ndarray, count: int) -> None:
    """Sort ``ids[:count]`` ascending in place.

    Short lists use insertion sort, longer ones heapsort. The ids are
    distinct, so the result does not depend on the algorithm.
    """
    if count <= INSERTION_SORT_LIMIT:
        for index in range(1, count):
            held = ids[index]
            other = index - 1
            while other >= 0 and ids[other] > held:
                ids[other + 1] = ids[other]
                other -= 1
            ids[other + 1] = held
        return
    for start in range(count // 2 - 1, -1, -1):
        _sift_down(ids, start, count)
    top = int(np.int64(0))
    for end in range(count - 1, 0, -1):
        held = ids[top]
        ids[top] = ids[end]
        ids[end] = held
        _sift_down(ids, top, end)


@njit(cache=True, nogil=True)
def bit_length(value: int) -> int:
    """Return the number of bits needed to represent a non-negative integer."""
    bits = 0
    while value > 0:
        value >>= 1
        bits += 1
    return bits


@njit(cache=True, nogil=True)
def check_worker_rows(n_rows: int, n_workers: int) -> None:
    """Raise when a scratch array has fewer rows than there are workers."""
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    if n_rows < n_workers:
        raise ValueError("every scratch array needs one row per worker")


@njit(cache=True, nogil=True, parallel=True)
def exclusive_scan(
    counts: np.ndarray, n_workers: int, parallel_min_items: int
) -> np.ndarray:
    """Return ``out`` with ``out[0] = 0`` and ``out[i + 1] = out[i] + counts[i]``.

    ``counts`` is an int64 array. Large inputs use blocks of ``SCAN_BLOCK``:
    block sums in parallel, a sequential scan of the block sums, then local
    scans in parallel. Integer sums are exact, so every path is identical.
    """
    n_items = counts.shape[0]
    out = np.empty(n_items + 1, dtype=np.int64)
    out[0] = 0
    if n_workers <= 1 or n_items < 64 * parallel_min_items:
        running = np.int64(0)
        for index in range(n_items):
            running += counts[index]
            out[index + 1] = running
        return out
    n_blocks = (n_items + SCAN_BLOCK - 1) // SCAN_BLOCK
    block_sums = np.zeros(n_blocks, dtype=np.int64)
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            block_total = np.int64(0)
            for index in range(
                block * SCAN_BLOCK, min(n_items, (block + 1) * SCAN_BLOCK)
            ):
                block_total += counts[index]
            block_sums[block] = block_total
    offset = np.int64(0)
    for scanned in range(n_blocks):
        this_block = block_sums[scanned]
        block_sums[scanned] = offset
        offset += this_block
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            local = block_sums[block]
            for index in range(
                block * SCAN_BLOCK, min(n_items, (block + 1) * SCAN_BLOCK)
            ):
                local += counts[index]
                out[index + 1] = local
    return out


@njit(cache=True, nogil=True, parallel=True)
def bucket_by_shifted_key(
    keys: np.ndarray,
    shift: int,
    n_buckets: int,
    n_workers: int,
    parallel_min_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stably order item indices by bucket ``keys[i] >> shift``.

    Returns ``start`` (int64, ``n_buckets + 1`` offsets) and ``order`` (int32
    item indices sorted by bucket, ties by item index). Large inputs count
    per chunk of ``BUCKET_CHUNK`` items, scan the counts in bucket-major
    order and scatter each chunk in parallel.
    """
    n_items = keys.shape[0]
    start = np.zeros(n_buckets + 1, dtype=np.int64)
    order = np.empty(n_items, dtype=np.int32)
    if n_workers <= 1 or n_items < 16 * parallel_min_items:
        for index in range(n_items):
            start[(keys[index] >> shift) + 1] += 1
        for bucket in range(n_buckets):
            start[bucket + 1] += start[bucket]
        cursor = start[:-1].copy()
        for index in range(n_items):
            bucket = keys[index] >> shift
            order[cursor[bucket]] = index
            cursor[bucket] += 1
        return start, order
    n_chunks = (n_items + BUCKET_CHUNK - 1) // BUCKET_CHUNK
    counts = np.zeros((n_chunks, n_buckets), dtype=np.int64)
    for worker in prange(n_workers):
        for chunk in range(worker, n_chunks, n_workers):
            row = counts[chunk]
            for index in range(
                chunk * BUCKET_CHUNK, min(n_items, (chunk + 1) * BUCKET_CHUNK)
            ):
                row[keys[index] >> shift] += 1
    running = np.int64(0)
    for scan_bucket in range(n_buckets):
        start[scan_bucket] = running
        for scan_chunk in range(n_chunks):
            count = counts[scan_chunk, scan_bucket]
            counts[scan_chunk, scan_bucket] = running
            running += count
    start[n_buckets] = running
    for worker in prange(n_workers):
        for chunk in range(worker, n_chunks, n_workers):
            row = counts[chunk]
            for index in range(
                chunk * BUCKET_CHUNK, min(n_items, (chunk + 1) * BUCKET_CHUNK)
            ):
                bucket = keys[index] >> shift
                order[row[bucket]] = index
                row[bucket] += 1
    return start, order


@njit(cache=True, nogil=True, parallel=True)
def stable_bucket_order(
    keys: np.ndarray, n_keys: int, n_workers: int, parallel_min_items: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stably order item indices by integer key in ``[0, n_keys)``.

    Returns ``start`` (int64, ``n_keys + 1`` offsets) and ``order`` (int32
    item indices sorted by key, ties by item index). Up to
    ``BUCKET_DIRECT_KEYS`` keys are bucketed directly. More keys are first
    bucketed by their high bits into at most ``BUCKET_DIRECT_KEYS`` contiguous
    key ranges; each range is then counting-sorted by exact key, the ranges in
    parallel and each one sequentially, which keeps the order stable.
    """
    n_items = keys.shape[0]
    if n_keys <= BUCKET_DIRECT_KEYS:
        return bucket_by_shifted_key(
            keys, np.int64(0), n_keys, n_workers, parallel_min_items
        )
    shift = bit_length(n_keys - 1) - bit_length(BUCKET_DIRECT_KEYS - 1)
    n_ranges = ((n_keys - 1) >> shift) + 1
    range_start, range_order = bucket_by_shifted_key(
        keys, shift, n_ranges, n_workers, parallel_min_items
    )
    width = 1 << shift
    start = np.empty(n_keys + 1, dtype=np.int64)
    start[n_keys] = n_items
    order = np.empty(n_items, dtype=np.int32)
    n_active = n_workers if n_items >= 16 * parallel_min_items else 1
    cursors = np.zeros((n_active, width), dtype=np.int64)
    for worker in prange(n_active):
        cursor = cursors[worker]
        for key_range in range(worker, n_ranges, n_active):
            low = range_start[key_range]
            high = range_start[key_range + 1]
            first_key = key_range << shift
            n_range_keys = min(width, n_keys - first_key)
            for offset in range(n_range_keys):
                cursor[offset] = 0
            for position in range(low, high):
                cursor[keys[range_order[position]] - first_key] += 1
            running = low
            for offset in range(n_range_keys):
                count = cursor[offset]
                start[first_key + offset] = running
                cursor[offset] = running
                running += count
            for position in range(low, high):
                item = range_order[position]
                offset = keys[item] - first_key
                order[cursor[offset]] = item
                cursor[offset] += 1
    return start, order


@njit(cache=True, nogil=True, parallel=True)
def scatter_add(
    keys: np.ndarray,
    dq: np.ndarray,
    ds: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    n_workers: int,
    parallel_min_items: int,
) -> None:
    """Apply ``totals[keys[i]] += dq[i]`` and ``sizes[keys[i]] += ds[i]``.

    ``totals`` is int64 and ``sizes`` int32, so the order of additions does not
    matter. Without atomics, large inputs are bucketed into ``SCATTER_RANGES``
    contiguous key ranges and each range is updated by one worker, so no two
    workers write the same entry.
    """
    n_items = keys.shape[0]
    if n_workers <= 1 or n_items < 4 * parallel_min_items:
        for index in range(n_items):
            key = keys[index]
            totals[key] += dq[index]
            sizes[key] += ds[index]
        return
    capacity = totals.shape[0]
    shift = max(0, bit_length(capacity - 1) - bit_length(SCATTER_RANGES - 1))
    n_ranges = ((capacity - 1) >> shift) + 1
    start, order = bucket_by_shifted_key(
        keys, shift, n_ranges, n_workers, parallel_min_items
    )
    for worker in prange(n_workers):
        for key_range in range(worker, n_ranges, n_workers):
            for position in range(start[key_range], start[key_range + 1]):
                item = order[position]
                key = keys[item]
                totals[key] += dq[item]
                sizes[key] += ds[item]


@njit(cache=True, nogil=True, parallel=True)
def fixed_block_sum(
    values: np.ndarray, n_workers: int, parallel_min_items: int
) -> float:
    """Sum a float64 array in a fixed order that ignores the worker count.

    Blocks of ``FLOAT_SUM_BLOCK`` consecutive elements are summed in order,
    then the block sums are added in block order.
    """
    n_items = values.shape[0]
    n_blocks = (n_items + FLOAT_SUM_BLOCK - 1) // FLOAT_SUM_BLOCK
    block_sums = np.zeros(n_blocks, dtype=np.float64)
    if n_workers <= 1 or n_items < 64 * parallel_min_items:
        for serial_block in range(n_blocks):
            serial_total = 0.0
            for index in range(
                serial_block * FLOAT_SUM_BLOCK,
                min(n_items, (serial_block + 1) * FLOAT_SUM_BLOCK),
            ):
                serial_total += values[index]
            block_sums[serial_block] = serial_total
    else:
        for worker in prange(n_workers):
            for block in range(worker, n_blocks, n_workers):
                block_total = 0.0
                for index in range(
                    block * FLOAT_SUM_BLOCK, min(n_items, (block + 1) * FLOAT_SUM_BLOCK)
                ):
                    block_total += values[index]
                block_sums[block] = block_total
    total = 0.0
    for summed in range(n_blocks):
        total += block_sums[summed]
    return total


@njit(cache=True, nogil=True)
def compact_ids(
    sizes: np.ndarray, n_workers: int, parallel_min_items: int
) -> tuple[np.ndarray, int]:
    """Number the used ids (``sizes > 0``) ``0, 1, ...`` in ascending order.

    Returns the new id of every id (meaningful for used ids only) as int32 and
    the number of used ids. The prefix sum runs in parallel; the two linear
    passes around it are simple enough to stay sequential.
    """
    n_ids = sizes.shape[0]
    flags = np.empty(n_ids, dtype=np.int64)
    for index in range(n_ids):
        flags[index] = 1 if sizes[index] > 0 else 0
    offsets = exclusive_scan(flags, n_workers, parallel_min_items)
    new_ids = np.empty(n_ids, dtype=np.int32)
    for index in range(n_ids):
        new_ids[index] = offsets[index]
    return new_ids, int(offsets[n_ids])


@njit(cache=True, nogil=True, parallel=True)
def gather(mapping: np.ndarray, indices: np.ndarray, n_workers: int) -> np.ndarray:
    """Return ``mapping[indices]`` as int32, computed in parallel blocks."""
    n_items = indices.shape[0]
    out = np.empty(n_items, dtype=np.int32)
    n_blocks = (n_items + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for index in range(
                block * WORKER_BLOCK, min(n_items, (block + 1) * WORKER_BLOCK)
            ):
                out[index] = mapping[indices[index]]
    return out
