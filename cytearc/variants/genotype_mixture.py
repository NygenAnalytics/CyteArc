"""SCClone's mixture of clone genotypes over bit-packed calls, fitted by EM.

Yu, Du and Song model the cells of a sample as a mixture of clones
("SCClone: Accurate Clustering of Tumor Single-Cell DNA Sequencing Data",
Frontiers in Genetics 13:823941, 2022, doi:10.3389/fgene.2022.823941,
equations 1 to 9). Each clone ``k`` is a binary genotype over the sites. A
cell of clone ``k`` reads wild type at a site the clone carries with the
dropout probability, the paper's false-negative rate, and mutant at a site
it lacks with the error probability, its false-positive rate; a site the
cell was not called at adds nothing. EM gives each cell's posterior of each
clone and sets each genotype entry, each clone's weight and both rates in
closed form; a rate the caller gives is held. This module was written from
the paper, which is CC BY 4.0, without reading SCClone's code.

Two departures, which :mod:`cytearc.variants.genotype_clones` relies on:

- The all-zero genotype, the wild type, is always component 0.
- Each site has its own dropout and error rate, each shrunk toward the
  pooled rate by a prior worth ``DROPOUT_PRIOR_CALLS`` and
  ``ERROR_PRIOR_CALLS`` calls, and a site's error rate stays at most
  ``MAX_SITE_ERROR_RATIO`` times the pooled error, so that a clone's private
  mutation is not read as that site's error. The pooled rates are the
  expected shares of wild-type calls at carried sites and of mutant calls at
  absent sites over every component, and an estimated pooled error stays at
  most ``MAX_POOLED_ERROR``. A site that every cell carries fits its calls
  as well with error ``1 - dropout`` and no clone carrying it, so without
  that bound a population of one clone, or a mutation that every cell
  carries, would be read as errors whenever no wild-type cell pinned the
  error down.

Calls are packed 64 sites to a word, mutant and wild-type calls apart, site
``j`` at bit ``j % 64`` of word ``j // 64``. The E-step walks a minimum
spanning tree over the genotypes, so each genotype's log-likelihood is its
tree parent's plus the sites where the two differ; the M-step walks only
each cell's unread bits and the bits where it reads other than its most
probable component's genotype. Sums over cells run in ``N_CHUNKS`` fixed
chunks, so the results do not depend on the thread count.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import prange

from ..utils.memory_budget import MemoryLedger
from ..utils.numba import njit

#: Fixed chunks of cells whose sums are added in order, so that results do
#: not depend on the thread count.
N_CHUNKS = 16
#: The calls that each site's dropout and error rate are shrunk toward the
#: pooled rates by, as a prior worth that many calls at the pooled rate.
DROPOUT_PRIOR_CALLS = 20.0
ERROR_PRIOR_CALLS = 200.0
#: The most that a site's error rate may be, as a multiple of the pooled
#: error rate.
MAX_SITE_ERROR_RATIO = 5.0
#: The most that an estimated pooled error rate may be: the largest
#: false-positive rate of SCClone's simulations, whose real data gave 0.6%
#: to 2.3%. With it, a site's error stays at most 0.5.
MAX_POOLED_ERROR = 0.1
#: Each component keeps at least this posterior mass, so that a component
#: whose cells all prefer another, such as the wild type where every cell
#: carries hundreds of mutations, keeps a finite log weight.
_LEAST_MASS = 1e-300
#: The dropout and error rates that EM starts from where they are estimated.
INITIAL_DROPOUT = 0.1
INITIAL_ERROR = 0.01
#: The relative rise in log-likelihood below which EM has converged.
EM_TOLERANCE = 1e-8
#: Rates are kept this far inside (0, 1) before their logarithms are taken.
_TINY = 1e-12
#: Posteriors below this add nothing to the M-step's sums.
SUPPORT = 1e-8

_ONE = np.uint64(1)
_FIVES = np.uint64(0x5555555555555555)
_THREES = np.uint64(0x3333333333333333)
_NIBBLES = np.uint64(0x0F0F0F0F0F0F0F0F)
_ALL_BITS = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True, nogil=True, inline="always")
def _popcount(word: np.uint64) -> np.uint64:
    """The set bits of a 64-bit word, by sums of ever wider fields that never carry past the word."""
    word = word - ((word >> _ONE) & _FIVES)
    word = (word & _THREES) + ((word >> np.uint64(2)) & _THREES)
    word = (word + (word >> np.uint64(4))) & _NIBBLES
    word = word + (word >> np.uint64(8))
    word = word + (word >> np.uint64(16))
    word = word + (word >> np.uint64(32))
    return word & np.uint64(0x7F)


@njit(cache=True, nogil=True, inline="always")
def _lowest_bit(word: np.uint64) -> np.int64:
    """The position of the lowest set bit of a nonzero word: the bits below it, counted."""
    return np.int64(_popcount((word & (~word + _ONE)) - _ONE))


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a ``(rows, sites)`` boolean mask into ``(rows, words)`` uint64, site j at bit j % 64 of word j // 64."""
    mask = np.asarray(mask, dtype=bool)
    rows, sites = mask.shape
    words = max(1, -(-sites // 64))
    packed = np.packbits(mask, axis=1, bitorder="little")
    out = np.zeros((rows, words * 8), dtype=np.uint8)
    out[:, : packed.shape[1]] = packed
    return out.view(np.uint64)


def n_words(n_sites: int) -> int:
    """The 64-bit words that hold one bit per site, at least one."""
    return max(1, -(-int(n_sites) // 64))


@njit(cache=True, nogil=True)
def pack_calls(
    cells: np.ndarray,
    sites: np.ndarray,
    calls: np.ndarray,
    site_rank: np.ndarray,
    mutant: np.ndarray,
    mutant_bits: np.ndarray,
    wild_bits: np.ndarray,
) -> None:
    """Set each call's bit: ``mutant[call]`` marks mutant calls, a call of 0 is wild type, others unknown.

    ``cells`` gives each call's row of the bit matrices, ``sites`` its stored
    site and ``site_rank`` each stored site's column, -1 for a site left out.
    """
    for row in range(cells.shape[0]):
        column = site_rank[sites[row]]
        if column < 0:
            continue
        call = calls[row]
        bit = _ONE << np.uint64(column % 64)
        if mutant[call]:
            mutant_bits[cells[row], column // 64] |= bit
        elif call == 0:
            wild_bits[cells[row], column // 64] |= bit


@njit(cache=True, nogil=True, parallel=True)
def any_bits(bits: np.ndarray, out: np.ndarray) -> None:
    """Whether each row holds a set bit."""
    for row in prange(bits.shape[0]):
        found = False
        for word in range(bits.shape[1]):
            if bits[row, word]:
                found = True
                break
        out[row] = found


@njit(cache=True, nogil=True)
def site_counts(bits: np.ndarray, rows: np.ndarray, n_sites: int) -> np.ndarray:
    """Each site's set bits over ``rows`` of ``bits``."""
    out = np.zeros(n_sites, dtype=np.int64)
    for index in range(rows.shape[0]):
        row = rows[index]
        for word in range(bits.shape[1]):
            hit = bits[row, word]
            while hit:
                out[word * 64 + _lowest_bit(hit)] += 1
                hit &= hit - _ONE
    return out


@njit(cache=True, nogil=True)
def column_counts(bits: np.ndarray, n_sites: int) -> np.ndarray:
    """Each site's set bits over every row of ``bits``."""
    out = np.zeros(n_sites, dtype=np.int64)
    for row in range(bits.shape[0]):
        for word in range(bits.shape[1]):
            hit = bits[row, word]
            while hit:
                out[word * 64 + _lowest_bit(hit)] += 1
                hit &= hit - _ONE
    return out


@njit(cache=True, nogil=True)
def _pair_word(
    mbits: np.ndarray,
    wbits: np.ndarray,
    row: int,
    word: int,
    last_mask: np.uint64,
    unknown: bool,
) -> np.uint64:
    """A row's word of mutant calls, or of sites it was not called at."""
    bits: np.uint64 = mbits[row, word]
    if not unknown:
        return bits
    bits = ~(bits | wbits[row, word])
    if word == mbits.shape[1] - 1:
        bits &= last_mask
    return bits


@njit(cache=True, nogil=True)
def _count_bit_pairs(
    mbits: np.ndarray,
    wbits: np.ndarray,
    rows: np.ndarray,
    last_mask: np.uint64,
    unknown: bool,
) -> int:
    total = 0
    for index in range(rows.shape[0]):
        for word in range(mbits.shape[1]):
            total += int(
                _popcount(
                    _pair_word(mbits, wbits, rows[index], word, last_mask, unknown)
                )
            )
    return total


@njit(cache=True, nogil=True)
def _fill_bit_pairs(
    mbits: np.ndarray,
    wbits: np.ndarray,
    rows: np.ndarray,
    last_mask: np.uint64,
    unknown: bool,
    cells: np.ndarray,
    sites: np.ndarray,
) -> None:
    at = 0
    for index in range(rows.shape[0]):
        row = rows[index]
        for word in range(mbits.shape[1]):
            hit = _pair_word(mbits, wbits, row, word, last_mask, unknown)
            while hit:
                cells[at] = row
                sites[at] = word * 64 + _lowest_bit(hit)
                at += 1
                hit &= hit - _ONE


def bit_pairs(
    ledger: MemoryLedger,
    mbits: np.ndarray,
    wbits: np.ndarray,
    rows: np.ndarray,
    n_sites: int,
    *,
    unknown: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Each of ``rows``' mutant calls, or with ``unknown`` each site among ``n_sites`` it was not called at, as a pair of the row and the site.

    The pairs come in order of the rows and, within a row, of the sites, and
    stay charged to ``ledger`` as the bytes returned, which the caller
    releases once it lets them go.
    """
    tail = int(n_sites) - (mbits.shape[1] - 1) * 64
    last_mask = _ALL_BITS if tail == 64 else (_ONE << np.uint64(tail)) - _ONE
    rows = np.asarray(rows, dtype=np.int64)
    n_pairs = _count_bit_pairs(mbits, wbits, rows, last_mask, unknown)
    held = ledger.hold(
        "each cell's unread sites" if unknown else "each cell's mutant sites",
        16 * n_pairs,
    )
    cells = np.empty(n_pairs, dtype=np.int64)
    sites = np.empty(n_pairs, dtype=np.int64)
    _fill_bit_pairs(mbits, wbits, rows, last_mask, unknown, cells, sites)
    return cells, sites, held


@njit(cache=True, nogil=True)
def _distance(gbits: np.ndarray, first: int, second: int) -> int:
    total = 0
    for word in range(gbits.shape[1]):
        total += int(_popcount(gbits[first, word] ^ gbits[second, word]))
    return total


@njit(cache=True, nogil=True)
def distances(gbits: np.ndarray) -> np.ndarray:
    """The sites at which each pair of packed genotypes differ."""
    n = gbits.shape[0]
    out = np.zeros((n, n), dtype=np.int64)
    for first in range(n):
        for second in range(first + 1, n):
            value = _distance(gbits, first, second)
            out[first, second] = value
            out[second, first] = value
    return out


@njit(cache=True, nogil=True)
def spanning_tree(gbits: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Prim's minimum spanning tree over packed genotypes from node 0: its length, each node's parent and the order the nodes joined.

    Of the nodes nearest the tree, the one of lowest index joins first.
    """
    n = gbits.shape[0]
    parent = np.full(n, -1, dtype=np.int64)
    order = np.empty(n, dtype=np.int64)
    best = np.empty(n, dtype=np.int64)
    source = np.zeros(n, dtype=np.int64)
    in_tree = np.zeros(n, dtype=np.bool_)
    in_tree[0] = True
    order[0] = 0
    for node in range(n):
        best[node] = _distance(gbits, 0, node)
    total = 0
    for step in range(1, n):
        node = -1
        for candidate in range(n):
            if not in_tree[candidate] and (node < 0 or best[candidate] < best[node]):
                node = candidate
        in_tree[node] = True
        order[step] = node
        parent[node] = source[node]
        total += best[node]
        for candidate in range(n):
            if not in_tree[candidate]:
                value = _distance(gbits, node, candidate)
                if value < best[candidate]:
                    best[candidate] = value
                    source[candidate] = node
    return total, parent, order


@njit(cache=True, nogil=True)
def _tree_edits(
    gbits: np.ndarray, parent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each node, the words where it differs from its tree parent, with the bits it gains and loses."""
    n, words_per_node = gbits.shape
    count = 0
    for node in range(1, n):
        for word in range(words_per_node):
            if gbits[node, word] != gbits[parent[node], word]:
                count += 1
    pointers = np.zeros(n + 1, dtype=np.int64)
    words = np.empty(count, dtype=np.int64)
    plus = np.empty(count, dtype=np.uint64)
    minus = np.empty(count, dtype=np.uint64)
    edit = 0
    for node in range(n):
        pointers[node] = edit
        if node > 0:
            above = parent[node]
            for word in range(words_per_node):
                own = gbits[node, word]
                theirs = gbits[above, word]
                if own != theirs:
                    words[edit] = word
                    plus[edit] = own & ~theirs
                    minus[edit] = theirs & ~own
                    edit += 1
    pointers[n] = edit
    return pointers, words, plus, minus


@njit(cache=True, nogil=True, parallel=True)
def _relative_tree(
    mbits: np.ndarray,
    wbits: np.ndarray,
    order: np.ndarray,
    parent: np.ndarray,
    pointers: np.ndarray,
    words: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    carried_mutant: np.ndarray,
    carried_wild: np.ndarray,
    out: np.ndarray,
) -> None:
    """Each cell's log-likelihood under each node less the wild type's, accumulated down the spanning tree."""
    n_cells = mbits.shape[0]
    n_nodes = order.shape[0]
    for cell in prange(n_cells):
        out[cell, 0] = 0.0
        for index in range(1, n_nodes):
            node = order[index]
            total = out[cell, parent[node]]
            for edit in range(pointers[node], pointers[node + 1]):
                word = words[edit]
                mutant_word = mbits[cell, word]
                wild_word = wbits[cell, word]
                hit = mutant_word & plus[edit]
                while hit:
                    total += carried_mutant[word * 64 + _lowest_bit(hit)]
                    hit &= hit - _ONE
                hit = wild_word & plus[edit]
                while hit:
                    total += carried_wild[word * 64 + _lowest_bit(hit)]
                    hit &= hit - _ONE
                hit = mutant_word & minus[edit]
                while hit:
                    total -= carried_mutant[word * 64 + _lowest_bit(hit)]
                    hit &= hit - _ONE
                hit = wild_word & minus[edit]
                while hit:
                    total -= carried_wild[word * 64 + _lowest_bit(hit)]
                    hit &= hit - _ONE
            out[cell, node] = total


@njit(cache=True, nogil=True, parallel=True)
def _posteriors(
    relative: np.ndarray, log_weights: np.ndarray, out: np.ndarray, sums: np.ndarray
) -> None:
    """Normalise each cell's relative log-likelihoods plus log weights into ``out``, with each cell's log sum in ``sums``."""
    n_cells, n_components = relative.shape
    for cell in prange(n_cells):
        top = -np.inf
        for component in range(n_components):
            value = relative[cell, component] + log_weights[component]
            out[cell, component] = value
            if value > top:
                top = value
        total = 0.0
        for component in range(n_components):
            out[cell, component] = math.exp(out[cell, component] - top)
            total += out[cell, component]
        for component in range(n_components):
            out[cell, component] /= total
        sums[cell] = top + math.log(total)


@njit(cache=True, nogil=True, parallel=True)
def _call_sums(
    mbits: np.ndarray,
    wbits: np.ndarray,
    gbits: np.ndarray,
    posteriors: np.ndarray,
    n_sites: int,
    mass: np.ndarray,
    scratch: np.ndarray,
    bits: np.ndarray,
    mutant: np.ndarray,
    wild: np.ndarray,
) -> None:
    """Each component's expected mutant and wild-type calls at each site, from each cell's unread and other-state bits.

    A cell's calls are its most probable component's genotype at the sites
    it was read at, except where it reads the other state. With ``h`` that
    component of cell i and ``u``, ``d`` its unread and other-state bits, the
    expected mutant calls of component k at site j are
    ``sum_h g[h, j] R[h, k] - sum_i r[i, k] u[i, j] g[h, j] + sum_i r[i, k] d[i, j] (1 - 2 g[h, j])``
    with ``R[h, k]`` the posterior mass of k among h's cells, and its
    wild-type calls are ``N[k] - sum_i r[i, k] u[i, j]`` less the mutant
    calls. ``mass`` (chunks, components, components), ``scratch`` (3,
    chunks, components, sites) and ``bits`` (chunks, 3, words) are the
    chunks' sums and words, overwritten.
    """
    n_cells, n_word = mbits.shape
    n_components = posteriors.shape[1]
    n_chunks = mass.shape[0]
    tail = n_sites - (n_word - 1) * 64
    last_mask = _ALL_BITS if tail == 64 else (_ONE << np.uint64(tail)) - _ONE
    mass[:] = 0.0
    scratch[:] = 0.0
    step = (n_cells + n_chunks - 1) // n_chunks
    for index in prange(n_chunks):
        # prange's index is unsigned; np.int64 keeps the arithmetic signed.
        chunk = np.int64(index)
        other = scratch[0, chunk]
        unread_carried = scratch[1, chunk]
        unread = scratch[2, chunk]
        dbits = bits[chunk, 0]
        ubits = bits[chunk, 1]
        cbits = bits[chunk, 2]
        for cell in range(chunk * step, min(n_cells, (chunk + 1) * step)):
            hard = 0
            top = posteriors[cell, 0]
            for component in range(1, n_components):
                if posteriors[cell, component] > top:
                    top = posteriors[cell, component]
                    hard = component
            for word in range(n_word):
                genotype = gbits[hard, word]
                mutant_word = mbits[cell, word]
                wild_word = wbits[cell, word]
                dbits[word] = (mutant_word & ~genotype) | (wild_word & genotype)
                unread_word = ~(mutant_word | wild_word)
                if word == n_word - 1:
                    unread_word &= last_mask
                ubits[word] = unread_word
                cbits[word] = unread_word & genotype
            for component in range(n_components):
                weight = posteriors[cell, component]
                if weight < SUPPORT:
                    continue
                mass[chunk, hard, component] += weight
                for word in range(n_word):
                    genotype = gbits[hard, word]
                    hit = dbits[word]
                    while hit:
                        bit = _lowest_bit(hit)
                        site = word * 64 + bit
                        if (genotype >> np.uint64(bit)) & _ONE:
                            other[component, site] -= weight
                        else:
                            other[component, site] += weight
                        hit &= hit - _ONE
                    hit = ubits[word]
                    while hit:
                        unread[component, word * 64 + _lowest_bit(hit)] += weight
                        hit &= hit - _ONE
                    hit = cbits[word]
                    while hit:
                        unread_carried[component, word * 64 + _lowest_bit(hit)] += (
                            weight
                        )
                        hit &= hit - _ONE
    # The chunks' sums, added in chunk order.
    for first in range(1, n_chunks):
        for component in range(n_components):
            for other_component in range(n_components):
                mass[0, component, other_component] += mass[
                    first, component, other_component
                ]
            for column in range(n_sites):
                scratch[0, 0, component, column] += scratch[0, first, component, column]
                scratch[1, 0, component, column] += scratch[1, first, component, column]
                scratch[2, 0, component, column] += scratch[2, first, component, column]
    mutant[:] = 0.0
    for component in range(n_components):
        for hard in range(n_components):
            share = mass[0, hard, component]
            if share == 0.0:
                continue
            for word in range(n_word):
                hit = gbits[hard, word]
                while hit:
                    site = word * 64 + _lowest_bit(hit)
                    if site < n_sites:
                        mutant[component, site] += share
                    hit &= hit - _ONE
    for component in range(n_components):
        for column in range(n_sites):
            mutant[component, column] += (
                scratch[0, 0, component, column] - scratch[1, 0, component, column]
            )
    for component in range(n_components):
        counted = 0.0
        for source in range(n_components):
            counted += mass[0, source, component]
        for column in range(n_sites):
            wild[component, column] = (
                counted - scratch[2, 0, component, column] - mutant[component, column]
            )


@njit(cache=True, nogil=True, parallel=True)
def hard_call_sums(
    mbits: np.ndarray,
    wbits: np.ndarray,
    hard: np.ndarray,
    n_sites: int,
    scratch: np.ndarray,
    mutant: np.ndarray,
    wild: np.ndarray,
) -> None:
    """Each component's mutant and wild-type calls at each site over the cells whose most probable component it is.

    ``scratch`` (2, chunks, components, sites) holds the chunks' sums, added
    in chunk order.
    """
    n_cells, n_word = mbits.shape
    n_chunks = scratch.shape[1]
    n_components = scratch.shape[2]
    scratch[:] = 0.0
    step = (n_cells + n_chunks - 1) // n_chunks
    for index in prange(n_chunks):
        chunk = np.int64(index)
        for cell in range(chunk * step, min(n_cells, (chunk + 1) * step)):
            component = hard[cell]
            for word in range(n_word):
                hit = mbits[cell, word]
                while hit:
                    scratch[0, chunk, component, word * 64 + _lowest_bit(hit)] += 1.0
                    hit &= hit - _ONE
                hit = wbits[cell, word]
                while hit:
                    scratch[1, chunk, component, word * 64 + _lowest_bit(hit)] += 1.0
                    hit &= hit - _ONE
    for component in range(n_components):
        for column in range(n_sites):
            mutant_total = 0.0
            wild_total = 0.0
            for part in range(n_chunks):
                mutant_total += scratch[0, part, component, column]
                wild_total += scratch[1, part, component, column]
            mutant[component, column] = mutant_total
            wild[component, column] = wild_total


@njit(cache=True, nogil=True, parallel=True)
def other_state_bits(
    mbits: np.ndarray, wbits: np.ndarray, rows: np.ndarray, carried: np.ndarray
) -> np.ndarray:
    """Each of ``rows``' calls of the state that the packed genotype ``carried`` lacks: wild type where it carries, mutant where not."""
    out = np.empty((rows.shape[0], mbits.shape[1]), dtype=np.uint64)
    for index in prange(rows.shape[0]):
        row = rows[index]
        for word in range(mbits.shape[1]):
            out[index, word] = (wbits[row, word] & carried[word]) | (
                mbits[row, word] & ~carried[word]
            )
    return out


@njit(cache=True, nogil=True, parallel=True)
def local_bits(
    mbits: np.ndarray, wbits: np.ndarray, rows: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The calls of ``rows`` at ``sites``, packed with site ``sites[c]`` at bit c."""
    n_rows = rows.shape[0]
    n_local = sites.shape[0]
    n_local_words = max(1, (n_local + 63) // 64)
    local_mutant = np.zeros((n_rows, n_local_words), dtype=np.uint64)
    local_wild = np.zeros((n_rows, n_local_words), dtype=np.uint64)
    for index in prange(n_rows):
        row = rows[index]
        for column in range(n_local):
            site = sites[column]
            bit = _ONE << np.uint64(site % 64)
            out = _ONE << np.uint64(column % 64)
            if mbits[row, site // 64] & bit:
                local_mutant[index, column // 64] |= out
            elif wbits[row, site // 64] & bit:
                local_wild[index, column // 64] |= out
    return local_mutant, local_wild


@njit(cache=True, nogil=True, parallel=True)
def weighted_site_counts(
    mbits: np.ndarray,
    wbits: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    n_sites: int,
    n_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected mutant and wild-type calls at every site of ``rows`` weighted by ``weights``."""
    n_rows = rows.shape[0]
    n_word = mbits.shape[1]
    mutant = np.zeros((n_chunks, n_sites))
    wild = np.zeros((n_chunks, n_sites))
    step = (n_rows + n_chunks - 1) // n_chunks
    for position in prange(n_chunks):
        chunk = np.int64(position)
        for index in range(chunk * step, min(n_rows, (chunk + 1) * step)):
            weight = weights[index]
            if weight < _TINY:
                continue
            row = rows[index]
            for word in range(n_word):
                hit = mbits[row, word]
                while hit:
                    mutant[chunk, word * 64 + _lowest_bit(hit)] += weight
                    hit &= hit - _ONE
                hit = wbits[row, word]
                while hit:
                    wild[chunk, word * 64 + _lowest_bit(hit)] += weight
                    hit &= hit - _ONE
    mutant_total = np.zeros(n_sites)
    wild_total = np.zeros(n_sites)
    for part in range(n_chunks):
        for site in range(n_sites):
            mutant_total[site] += mutant[part, site]
            wild_total[site] += wild[part, site]
    return mutant_total, wild_total


#: Up to this many sites, a split's call patterns are grouped by a base-3
#: code into a dense table instead of by sorting.
DENSE_PATTERN_SITES = 10


@njit(cache=True, nogil=True)
def _group_patterns_dense(
    local_mutant: np.ndarray, local_wild: np.ndarray, weights: np.ndarray, n_local: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`call_patterns` for at most ``DENSE_PATTERN_SITES`` sites: each row's base-3 code indexes a dense table."""
    n_rows = local_mutant.shape[0]
    size = 1
    for _site in range(n_local):
        size *= 3
    codes = np.empty(n_rows, dtype=np.int64)
    summed = np.zeros(size)
    for row in range(n_rows):
        mutant_word = local_mutant[row, 0]
        wild_word = local_wild[row, 0]
        code = 0
        for column in range(n_local - 1, -1, -1):
            bit = _ONE << np.uint64(column)
            code *= 3
            if mutant_word & bit:
                code += 1
            elif wild_word & bit:
                code += 2
        codes[row] = code
        summed[code] += weights[row]
    index = np.full(size, -1, dtype=np.int64)
    count = 0
    for code in range(size):
        if summed[code] > 0.0:
            index[code] = count
            count += 1
    # Rows of zero weight still need a pattern for the inverse map.
    for row in range(n_rows):
        if index[codes[row]] < 0:
            index[codes[row]] = count
            count += 1
    pattern_mutant = np.zeros((count, 1), dtype=np.uint64)
    pattern_wild = np.zeros((count, 1), dtype=np.uint64)
    pattern_weights = np.zeros(count)
    inverse = np.empty(n_rows, dtype=np.int64)
    for row in range(n_rows):
        position = index[codes[row]]
        inverse[row] = position
        pattern_mutant[position, 0] = local_mutant[row, 0]
        pattern_wild[position, 0] = local_wild[row, 0]
    for code in range(size):
        if index[code] >= 0 and summed[code] > 0.0:
            pattern_weights[index[code]] = summed[code]
    return pattern_mutant, pattern_wild, pattern_weights, inverse


def call_patterns(
    local_mutant: np.ndarray, local_wild: np.ndarray, weights: np.ndarray, n_local: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The distinct local call patterns of the rows, each weighted by its rows' weights, and each row's pattern.

    A split's E-step gives every row of one pattern the same posterior, so a
    fit over the patterns weighted by their rows' summed weights has the
    same bound as a fit over the rows.
    """
    if n_local <= DENSE_PATTERN_SITES:
        return _group_patterns_dense(local_mutant, local_wild, weights, n_local)
    if local_mutant.shape[1] == 1 and local_mutant.shape[0]:
        combined = local_mutant[:, 0] | local_wild[:, 0]
        narrow = int(np.max(combined, initial=0)) < 2**32
        del combined
        if narrow:
            keys = local_mutant[:, 0] | (local_wild[:, 0] << np.uint64(32))
            unique, first, inverse = np.unique(
                keys, return_index=True, return_inverse=True
            )
            del keys
            summed = np.bincount(inverse, weights=weights, minlength=len(unique))
            return (
                np.ascontiguousarray(local_mutant[first]),
                np.ascontiguousarray(local_wild[first]),
                summed,
                inverse,
            )
    stacked = np.ascontiguousarray(np.hstack([local_mutant, local_wild]))
    view = stacked.view(
        np.dtype((np.void, stacked.dtype.itemsize * stacked.shape[1]))
    ).ravel()
    _unique, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    del stacked, view, _unique
    summed = np.bincount(inverse, weights=weights, minlength=len(first))
    return (
        np.ascontiguousarray(local_mutant[first]),
        np.ascontiguousarray(local_wild[first]),
        summed,
        inverse.ravel(),
    )


@njit(cache=True, nogil=True)
def _clip(value: float) -> float:
    return min(max(value, _TINY), 1.0 - _TINY)


@njit(cache=True, nogil=True)
def site_term(
    carried_wild: float,
    carried_all: float,
    absent_mutant: float,
    absent_all: float,
    dropout: float,
    error: float,
) -> float:
    """One site's expected complete-data log-likelihood from its expected calls and rates."""
    dropout = _clip(dropout)
    error = _clip(error)
    return (
        carried_wild * math.log(dropout)
        + (carried_all - carried_wild) * math.log1p(-dropout)
        + absent_mutant * math.log(error)
        + (absent_all - absent_mutant) * math.log1p(-error)
    )


@njit(cache=True, nogil=True)
def rate_update(
    carried_wild: float,
    carried_all: float,
    absent_mutant: float,
    absent_all: float,
    pooled_dropout: float,
    pooled_error: float,
    shrink_dropout: float,
    shrink_error: float,
    cap: float,
    fixed_dropout: float,
    fixed_error: float,
) -> tuple[float, float]:
    """A site's rates from its expected calls, as the M-step sets them: shrunk toward the pooled rates, the error capped.

    A fixed rate of 0 or more is held; ``cap`` below 0 caps nothing.
    """
    if fixed_dropout >= 0:
        dropout = fixed_dropout
    else:
        dropout = (carried_wild + shrink_dropout * pooled_dropout) / (
            carried_all + shrink_dropout
        )
    if fixed_error >= 0:
        error = fixed_error
    else:
        error = (absent_mutant + shrink_error * pooled_error) / (
            absent_all + shrink_error
        )
        if cap > 0:
            error = min(error, cap * pooled_error)
    return dropout, error


@njit(cache=True, nogil=True)
def _split_counts(
    local_mutant: np.ndarray,
    local_wild: np.ndarray,
    weights: np.ndarray,
    shares: np.ndarray,
    mutant: np.ndarray,
    wild: np.ndarray,
) -> None:
    n_rows, n_local_words = local_mutant.shape
    mutant[:] = 0.0
    wild[:] = 0.0
    for row in range(n_rows):
        weight = weights[row] * shares[row]
        if weight < _TINY:
            continue
        for word in range(n_local_words):
            hit = local_mutant[row, word]
            while hit:
                mutant[word * 64 + _lowest_bit(hit)] += weight
                hit &= hit - _ONE
            hit = local_wild[row, word]
            while hit:
                wild[word * 64 + _lowest_bit(hit)] += weight
                hit &= hit - _ONE


@njit(cache=True, nogil=True)
def _split_e_step(
    local_mutant: np.ndarray,
    local_wild: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    mutant_weight: np.ndarray,
    wild_weight: np.ndarray,
    share: float,
    shares: np.ndarray,
) -> None:
    """Each row's posterior of the first child against the second, from the sites where their genotypes differ."""
    n_rows, n_local_words = local_mutant.shape
    n_local = first.shape[0]
    prior = math.log(share) - math.log1p(-share)
    plus = np.zeros(n_local_words, dtype=np.uint64)
    minus = np.zeros(n_local_words, dtype=np.uint64)
    for column in range(n_local):
        bit = _ONE << np.uint64(column % 64)
        if first[column] and not second[column]:
            plus[column // 64] |= bit
        elif second[column] and not first[column]:
            minus[column // 64] |= bit
    for row in range(n_rows):
        delta = prior
        for word in range(n_local_words):
            hit = local_mutant[row, word] & plus[word]
            while hit:
                delta += mutant_weight[word * 64 + _lowest_bit(hit)]
                hit &= hit - _ONE
            hit = local_wild[row, word] & plus[word]
            while hit:
                delta += wild_weight[word * 64 + _lowest_bit(hit)]
                hit &= hit - _ONE
            hit = local_mutant[row, word] & minus[word]
            while hit:
                delta -= mutant_weight[word * 64 + _lowest_bit(hit)]
                hit &= hit - _ONE
            hit = local_wild[row, word] & minus[word]
            while hit:
                delta -= wild_weight[word * 64 + _lowest_bit(hit)]
                hit &= hit - _ONE
        if delta > 0:
            shares[row] = 1.0 / (1.0 + math.exp(-delta))
        else:
            odds = math.exp(delta)
            shares[row] = odds / (1.0 + odds)


@njit(cache=True, nogil=True)
def split_fit(
    local_mutant: np.ndarray,
    local_wild: np.ndarray,
    weights: np.ndarray,
    shares: np.ndarray,
    parent: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    clone_mutant: np.ndarray,
    clone_wild: np.ndarray,
    carried_wild: np.ndarray,
    carried_all: np.ndarray,
    absent_mutant: np.ndarray,
    absent_all: np.ndarray,
    dropout: np.ndarray,
    error: np.ndarray,
    pooled_dropout: float,
    pooled_error: float,
    shrink_dropout: float,
    shrink_error: float,
    cap: float,
    fixed_dropout: float,
    fixed_error: float,
    max_iter: int,
    tol: float,
    fix_second: bool,
    first_share: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit a split of one clone into two children at its candidate sites, and return the lower bound on its gain.

    ``local_mutant`` and ``local_wild`` are the clone's cells' calls at the
    candidate sites, or their distinct patterns, ``weights`` each row's
    posterior of the clone and ``shares`` its share in the first child,
    written in place, as are the children's genotypes ``first`` and
    ``second``, which start from the seed's majority calls; the first
    E-step uses them with the share ``first_share``. ``clone_mutant`` and
    ``clone_wild`` are the clone's expected mutant and wild-type calls at
    the sites, ``carried_wild``, ``carried_all``, ``absent_mutant`` and
    ``absent_all`` every component's expected calls there (wild type at
    carried sites, all calls at carried sites, mutant at absent sites, all
    calls at absent sites), and ``dropout`` and ``error`` the current rates.
    With ``fix_second`` the second child keeps its genotype, as the wild
    type does when it splits off a new clone. Coordinate ascent on the
    evidence lower bound: genotypes, then the sites' rates in closed form,
    then the children's shares, then each row's posterior between the
    children.

    Returns:
        The change in the sites' expected log-likelihood, the change from
        sharing the clone's weight and the posteriors' entropy, the first
        child's share, the new rates, the first child's expected mutant and
        wild-type calls, and the iterations run.
    """
    n_rows = local_mutant.shape[0]
    n_local = parent.shape[0]
    old = 0.0
    for column in range(n_local):
        old += site_term(
            carried_wild[column],
            carried_all[column],
            absent_mutant[column],
            absent_all[column],
            dropout[column],
            error[column],
        )
    # Every component's calls without the clone's.
    rest_carried_wild = np.empty(n_local)
    rest_carried_all = np.empty(n_local)
    rest_absent_mutant = np.empty(n_local)
    rest_absent_all = np.empty(n_local)
    for column in range(n_local):
        if parent[column]:
            rest_carried_wild[column] = carried_wild[column] - clone_wild[column]
            rest_carried_all[column] = (
                carried_all[column] - clone_mutant[column] - clone_wild[column]
            )
            rest_absent_mutant[column] = absent_mutant[column]
            rest_absent_all[column] = absent_all[column]
        else:
            rest_carried_wild[column] = carried_wild[column]
            rest_carried_all[column] = carried_all[column]
            rest_absent_mutant[column] = absent_mutant[column] - clone_mutant[column]
            rest_absent_all[column] = (
                absent_all[column] - clone_mutant[column] - clone_wild[column]
            )
    total = 0.0
    for row in range(n_rows):
        total += weights[row]
    new_dropout = dropout.copy()
    new_error = error.copy()
    mutant_weight = np.empty(n_local)
    wild_weight = np.empty(n_local)
    first_mutant = np.zeros(n_local)
    first_wild = np.zeros(n_local)
    for column in range(n_local):
        rate_dropout = _clip(new_dropout[column])
        rate_error = _clip(new_error[column])
        mutant_weight[column] = math.log((1.0 - rate_dropout) / rate_error)
        wild_weight[column] = math.log(rate_dropout / (1.0 - rate_error))
    _split_e_step(
        local_mutant,
        local_wild,
        first,
        second,
        mutant_weight,
        wild_weight,
        first_share,
        shares,
    )
    _split_counts(local_mutant, local_wild, weights, shares, first_mutant, first_wild)
    previous = -np.inf
    gain = -np.inf
    site_gain = 0.0
    share = 0.5
    iteration = 0
    for iteration in range(max_iter):
        # Genotypes from the counts with the rates so far.
        for column in range(n_local):
            rate_dropout = _clip(new_dropout[column])
            rate_error = _clip(new_error[column])
            mutant_weight[column] = math.log((1.0 - rate_dropout) / rate_error)
            wild_weight[column] = math.log(rate_dropout / (1.0 - rate_error))
            second_mutant = clone_mutant[column] - first_mutant[column]
            second_wild = clone_wild[column] - first_wild[column]
            if first_mutant[column] + first_wild[column] > SUPPORT:
                first[column] = (
                    first_mutant[column] * mutant_weight[column]
                    + first_wild[column] * wild_weight[column]
                    > 0
                )
            if second_mutant + second_wild > SUPPORT and not fix_second:
                second[column] = (
                    second_mutant * mutant_weight[column]
                    + second_wild * wild_weight[column]
                    > 0
                )
        # The sites' rates from every component's counts with the clone split.
        new = 0.0
        for column in range(n_local):
            second_mutant = clone_mutant[column] - first_mutant[column]
            second_wild = clone_wild[column] - first_wild[column]
            site_carried_wild = rest_carried_wild[column]
            site_carried_all = rest_carried_all[column]
            site_absent_mutant = rest_absent_mutant[column]
            site_absent_all = rest_absent_all[column]
            if first[column]:
                site_carried_wild += first_wild[column]
                site_carried_all += first_mutant[column] + first_wild[column]
            else:
                site_absent_mutant += first_mutant[column]
                site_absent_all += first_mutant[column] + first_wild[column]
            if second[column]:
                site_carried_wild += second_wild
                site_carried_all += second_mutant + second_wild
            else:
                site_absent_mutant += second_mutant
                site_absent_all += second_mutant + second_wild
            rate_dropout, rate_error = rate_update(
                site_carried_wild,
                site_carried_all,
                site_absent_mutant,
                site_absent_all,
                pooled_dropout,
                pooled_error,
                shrink_dropout,
                shrink_error,
                cap,
                fixed_dropout,
                fixed_error,
            )
            new_dropout[column] = rate_dropout
            new_error[column] = rate_error
            new += site_term(
                site_carried_wild,
                site_carried_all,
                site_absent_mutant,
                site_absent_all,
                rate_dropout,
                rate_error,
            )
        first_total = 0.0
        entropy = 0.0
        for row in range(n_rows):
            first_total += weights[row] * shares[row]
            row_share = shares[row]
            if 0.0 < row_share < 1.0:
                entropy -= weights[row] * (
                    row_share * math.log(row_share)
                    + (1.0 - row_share) * math.log1p(-row_share)
                )
        share = min(max(first_total / max(total, 1e-300), _TINY), 1.0 - _TINY)
        mixing = total * (share * math.log(share) + (1.0 - share) * math.log1p(-share))
        site_gain = new - old
        gain = site_gain + mixing + entropy
        if gain - previous <= tol or iteration == max_iter - 1:
            break
        previous = gain
        # Each row's posterior between the children under the new rates.
        for column in range(n_local):
            rate_dropout = _clip(new_dropout[column])
            rate_error = _clip(new_error[column])
            mutant_weight[column] = math.log((1.0 - rate_dropout) / rate_error)
            wild_weight[column] = math.log(rate_dropout / (1.0 - rate_error))
        _split_e_step(
            local_mutant,
            local_wild,
            first,
            second,
            mutant_weight,
            wild_weight,
            share,
            shares,
        )
        _split_counts(
            local_mutant, local_wild, weights, shares, first_mutant, first_wild
        )
    return (
        site_gain,
        gain - site_gain,
        share,
        new_dropout,
        new_error,
        first_mutant,
        first_wild,
        iteration + 1,
    )


@njit(cache=True, nogil=True, parallel=True)
def removal_logliks(
    relative: np.ndarray, log_weights: np.ndarray, posteriors: np.ndarray, n_chunks: int
) -> np.ndarray:
    """For each component, the sum over cells of log(1 - posterior), from the log-likelihoods where the posterior is large."""
    n_cells, n_components = relative.shape
    out = np.zeros((n_chunks, n_components))
    step = (n_cells + n_chunks - 1) // n_chunks
    for index in prange(n_chunks):
        chunk = np.int64(index)
        values = np.empty(n_components)
        for cell in range(chunk * step, min(n_cells, (chunk + 1) * step)):
            top = -np.inf
            for component in range(n_components):
                values[component] = relative[cell, component] + log_weights[component]
                if values[component] > top:
                    top = values[component]
            full = 0.0
            for component in range(n_components):
                full += math.exp(values[component] - top)
            log_full = top + math.log(full)
            for component in range(n_components):
                posterior = posteriors[cell, component]
                if posterior < 0.5:
                    out[chunk, component] += math.log1p(-posterior)
                    continue
                rest_top = -np.inf
                for other in range(n_components):
                    if other != component and values[other] > rest_top:
                        rest_top = values[other]
                if rest_top == -np.inf:
                    out[chunk, component] += -np.inf
                    continue
                rest = 0.0
                for other in range(n_components):
                    if other != component:
                        rest += math.exp(values[other] - rest_top)
                out[chunk, component] += rest_top + math.log(rest) - log_full
    total = np.zeros(n_components)
    for part in range(n_chunks):
        for component in range(n_components):
            total[component] += out[part, component]
    return total


@dataclass(frozen=True, slots=True)
class SiteRates:
    """Each site's dropout and error rate, with the pooled rates that they are shrunk toward.

    Attributes:
        dropout: Each site's dropout rate, the chance that a cell carrying its
            mutation reads wild type.
        error: Each site's error rate, the chance that a cell without its
            mutation reads mutant.
        pooled_dropout: The share of wild-type calls at carried sites over
            every component.
        pooled_error: The share of mutant calls at absent sites over every
            component.
    """

    dropout: np.ndarray
    error: np.ndarray
    pooled_dropout: float
    pooled_error: float

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        """What a carried site adds to a cell's log-likelihood over the wild type's: for a mutant call, and for a wild-type call."""
        dropout = np.clip(self.dropout, _TINY, 1 - _TINY)
        error = np.clip(self.error, _TINY, 1 - _TINY)
        return np.log((1 - dropout) / error), np.log(dropout / (1 - error))


@dataclass(slots=True)
class MixtureFit:
    """A fit of the mixture: the clones' genotypes, every component's log weight and the sites' rates.

    Component 0 is the wild type; component ``k`` above it has the genotype
    ``genotypes[k - 1]``. A fit that EM returned also holds what it computed
    at these parameters, charged to the mixture's ledger as ``held`` until
    :meth:`GenotypeMixture.release` lets it go: each cell's relative
    log-likelihoods and posteriors, and each component's expected mutant and
    wild-type calls per site.

    Attributes:
        genotypes: The clones' genotypes, (clones, sites).
        log_weights: Each component's log weight, the wild type first.
        rates: The sites' rates.
        loglik: The log-likelihood of the calls.
        first_loglik: The log-likelihood at the parameters EM started from.
        posteriors: Each cell's posterior of each component, or None.
        relative: Each cell's log-likelihood under each component less the
            wild type's, or None.
        mutant: Each component's expected mutant calls at each site, or None.
        wild: Each component's expected wild-type calls at each site, or None.
        held: The bytes charged for the arrays.
    """

    genotypes: np.ndarray
    log_weights: np.ndarray
    rates: SiteRates
    loglik: float = -np.inf
    first_loglik: float = -np.inf
    posteriors: np.ndarray | None = None
    relative: np.ndarray | None = None
    mutant: np.ndarray | None = None
    wild: np.ndarray | None = None
    held: int = 0

    @property
    def n_components(self) -> int:
        """The components: the clones and the wild type."""
        return len(self.genotypes) + 1


@dataclass(frozen=True, slots=True)
class FitArrays:
    """The arrays that EM computed for a fit."""

    posteriors: np.ndarray
    relative: np.ndarray
    mutant: np.ndarray
    wild: np.ndarray


def fit_arrays(fit: MixtureFit) -> FitArrays:
    """The arrays that EM computed for ``fit``, which it must hold."""
    if (
        fit.posteriors is None
        or fit.relative is None
        or fit.mutant is None
        or fit.wild is None
    ):
        raise ValueError("The fit holds no arrays of EM; run EM on it first")
    return FitArrays(fit.posteriors, fit.relative, fit.mutant, fit.wild)


def fit_array_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What a fit of ``n_components`` components holds.

    Each cell's relative log-likelihoods and posteriors, each component's
    expected mutant and wild-type calls per site, and three copies of the
    parameters, the genotypes, the log weights and each site's two rates:
    the fit's, those EM started from and those of the fit before it.
    """
    return (
        16 * n_cells * n_components
        + 19 * n_components * n_sites
        + 24 * n_components
        + 48 * n_sites
    )


def em_scratch_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What EM holds beside its fit's arrays while it runs.

    Each cell's log sum, and the M-step's chunks of each component's mass of
    the others and of its other-state, unread and unread carried calls per
    site, with each chunk's words of a cell.
    """
    return (
        8 * n_cells
        + 8 * N_CHUNKS * n_components * n_components
        + 24 * N_CHUNKS * n_components * n_sites
        + 24 * N_CHUNKS * n_words(n_sites)
    )


def em_iteration_bytes(n_sites: int, n_components: int) -> int:
    """What an iteration of EM builds and lets go, whatever the cells.

    The genotypes packed with their spanning tree's edits, the sites'
    weights, at most three arrays of components by sites in float64 at once
    while the genotypes are updated, the genotypes they become, and the
    sites' rates as each is derived.
    """
    words = n_words(n_sites)
    return (
        28 * n_components * n_sites
        + 40 * n_components * words
        + 64 * n_components
        + 128 * n_sites
    )


class GenotypeMixture:
    """The packed calls of one group of cells and EM of the mixture on them.

    Args:
        mutant_bits: Each cell's mutant calls, packed.
        wild_bits: Each cell's wild-type calls, packed.
        n_sites: The sites.
        dropout_probability: A dropout rate to hold at every site, or None to
            estimate each site's.
        error_probability: An error rate to hold at every site, or None to
            estimate each site's.
        ledger: The memory ledger that each fit's arrays and each step are
            charged to.
    """

    def __init__(
        self,
        mutant_bits: np.ndarray,
        wild_bits: np.ndarray,
        n_sites: int,
        *,
        dropout_probability: float | None,
        error_probability: float | None,
        ledger: MemoryLedger,
    ) -> None:
        self.mbits = mutant_bits
        self.wbits = wild_bits
        self.n_sites = int(n_sites)
        self.n_cells = int(mutant_bits.shape[0])
        self.fixed_dropout = dropout_probability
        self.fixed_error = error_probability
        self.ledger = ledger
        self.column_mutant = column_counts(mutant_bits, self.n_sites).astype(np.float64)
        self.column_wild = column_counts(wild_bits, self.n_sites).astype(np.float64)

    def initial_rates(self) -> SiteRates:
        """The rates that EM starts from: those given, or 0.1 and 0.01."""
        dropout = INITIAL_DROPOUT if self.fixed_dropout is None else self.fixed_dropout
        error = INITIAL_ERROR if self.fixed_error is None else self.fixed_error
        return SiteRates(
            np.full(self.n_sites, float(dropout)),
            np.full(self.n_sites, float(error)),
            float(dropout),
            float(error),
        )

    def rate_settings(
        self, rates: SiteRates
    ) -> tuple[float, float, float, float, float, float, float]:
        """The pooled rates, priors, error cap and held rates as :func:`rate_update` takes them."""
        return (
            float(rates.pooled_dropout),
            float(rates.pooled_error),
            DROPOUT_PRIOR_CALLS,
            ERROR_PRIOR_CALLS,
            -1.0 if self.fixed_error is not None else MAX_SITE_ERROR_RATIO,
            -1.0 if self.fixed_dropout is None else float(self.fixed_dropout),
            -1.0 if self.fixed_error is None else float(self.fixed_error),
        )

    def baseline(self, rates: SiteRates) -> float:
        """The summed log-likelihood of every cell's calls under the wild type alone, from each site's totals."""
        error = np.clip(rates.error, _TINY, 1 - _TINY)
        return float(
            (self.column_mutant * np.log(error)).sum()
            + (self.column_wild * np.log1p(-error)).sum()
        )

    def relative(
        self, genotypes: np.ndarray, rates: SiteRates, out: np.ndarray
    ) -> None:
        """Each cell's log-likelihood under each component less the wild type's, walked down the genotypes' spanning tree, into ``out``."""
        gbits = pack_mask(
            np.vstack([np.zeros((1, self.n_sites), dtype=bool), genotypes])
        )
        _total, parent, order = spanning_tree(gbits)
        pointers, words, plus, minus = _tree_edits(gbits, parent)
        carried_mutant, carried_wild = rates.weights()
        _relative_tree(
            self.mbits,
            self.wbits,
            order,
            parent,
            pointers,
            words,
            plus,
            minus,
            carried_mutant,
            carried_wild,
            out,
        )

    def update_genotypes(
        self,
        genotypes: np.ndarray,
        mutant: np.ndarray,
        wild: np.ndarray,
        rates: SiteRates,
    ) -> np.ndarray:
        """Each clone's genotype entry as the state of the larger expected log-likelihood; a site none of its cells were read at keeps its state."""
        carried_mutant, carried_wild = rates.weights()
        gain = mutant[1:] * carried_mutant + wild[1:] * carried_wild
        updated: np.ndarray = gain > 0
        del gain
        unread = (mutant[1:] + wild[1:]) <= SUPPORT
        updated[unread] = genotypes[unread]
        return updated

    def update_rates(
        self, genotypes: np.ndarray, mutant: np.ndarray, wild: np.ndarray
    ) -> SiteRates:
        """Each site's rates from every component's expected calls, shrunk toward the pooled rates and the error capped."""
        carried = np.vstack([np.zeros((1, self.n_sites), dtype=bool), genotypes])
        carried_wild = (wild * carried).sum(axis=0)
        carried_all = ((mutant + wild) * carried).sum(axis=0)
        absent = ~carried
        absent_mutant = (mutant * absent).sum(axis=0)
        absent_all = ((mutant + wild) * absent).sum(axis=0)
        del carried, absent
        # A pseudo-count of one call each way keeps pooled rates inside (0, 1).
        pooled_dropout = (carried_wild.sum() + 1.0) / (carried_all.sum() + 2.0)
        pooled_error = (absent_mutant.sum() + 1.0) / (absent_all.sum() + 2.0)
        if self.fixed_dropout is not None:
            pooled_dropout = self.fixed_dropout
        if self.fixed_error is not None:
            pooled_error = self.fixed_error
        else:
            pooled_error = min(pooled_error, MAX_POOLED_ERROR)
        dropout = (
            np.full(self.n_sites, float(pooled_dropout))
            if self.fixed_dropout is not None
            else (carried_wild + DROPOUT_PRIOR_CALLS * pooled_dropout)
            / (carried_all + DROPOUT_PRIOR_CALLS)
        )
        error = (
            np.full(self.n_sites, float(pooled_error))
            if self.fixed_error is not None
            else (absent_mutant + ERROR_PRIOR_CALLS * pooled_error)
            / (absent_all + ERROR_PRIOR_CALLS)
        )
        if self.fixed_error is None:
            # A site's error rate may not exceed a multiple of the pooled
            # one, so a clone's private mutations cannot be read as errors.
            error = np.minimum(error, MAX_SITE_ERROR_RATIO * pooled_error)
        return SiteRates(dropout, error, float(pooled_dropout), float(pooled_error))

    def em(
        self, start: MixtureFit, *, max_iter: int, tol: float = EM_TOLERANCE
    ) -> MixtureFit:
        """Run EM from ``start`` for at most ``max_iter`` iterations, until the log-likelihood rises by at most ``tol`` of itself.

        The fit returned holds the arrays of its own parameters, charged to
        the ledger until :meth:`release` lets them go; ``start`` should hold
        none.
        """
        genotypes = start.genotypes.copy()
        log_weights = start.log_weights.copy()
        rates = start.rates
        n_components = len(genotypes) + 1
        ledger = self.ledger
        held = ledger.hold(
            f"the posteriors of {self.n_cells} cells for {n_components} components",
            fit_array_bytes(self.n_cells, self.n_sites, n_components),
        )
        scratch_bytes = ledger.hold(
            f"EM's sums over {self.n_cells} cells",
            em_scratch_bytes(self.n_cells, self.n_sites, n_components),
        )
        words = self.mbits.shape[1]
        relative = np.empty((self.n_cells, n_components))
        posteriors = np.empty((self.n_cells, n_components))
        mutant = np.empty((n_components, self.n_sites))
        wild = np.empty((n_components, self.n_sites))
        sums = np.empty(self.n_cells)
        mass = np.empty((N_CHUNKS, n_components, n_components))
        scratch = np.empty((3, N_CHUNKS, n_components, self.n_sites))
        bits = np.empty((N_CHUNKS, 3, words), dtype=np.uint64)
        ledger.peak(
            f"each iteration of EM of {n_components} components",
            em_iteration_bytes(self.n_sites, n_components),
        )
        previous = -np.inf
        loglik = -np.inf
        first_loglik = -np.inf
        for iteration in range(1, max_iter + 1):
            self.relative(genotypes, rates, relative)
            _posteriors(relative, log_weights, posteriors, sums)
            loglik = float(sums.sum()) + self.baseline(rates)
            if iteration == 1:
                first_loglik = loglik
            gbits = pack_mask(
                np.vstack([np.zeros((1, self.n_sites), dtype=bool), genotypes])
            )
            _call_sums(
                self.mbits,
                self.wbits,
                gbits,
                posteriors,
                self.n_sites,
                mass,
                scratch,
                bits,
                mutant,
                wild,
            )
            del gbits
            if (
                iteration > 1 and loglik - previous <= tol * max(1.0, abs(loglik))
            ) or iteration == max_iter:
                break
            previous = loglik
            genotypes = self.update_genotypes(genotypes, mutant, wild, rates)
            totals = posteriors.sum(axis=0)
            log_weights = np.log(np.maximum(totals, _LEAST_MASS) / self.n_cells)
            del totals
            rates = self.update_rates(genotypes, mutant, wild)
        del sums, mass, scratch, bits
        ledger.release(scratch_bytes)
        return MixtureFit(
            genotypes,
            log_weights,
            rates,
            loglik,
            first_loglik,
            posteriors,
            relative,
            mutant,
            wild,
            held,
        )

    def release(self, fit: MixtureFit) -> MixtureFit:
        """The fit's parameters without its arrays, whose charge the ledger lets go."""
        held = fit.held
        bare = MixtureFit(
            fit.genotypes, fit.log_weights, fit.rates, fit.loglik, fit.first_loglik
        )
        fit.posteriors = fit.relative = fit.mutant = fit.wild = None
        fit.held = 0
        self.ledger.release(held)
        return bare
