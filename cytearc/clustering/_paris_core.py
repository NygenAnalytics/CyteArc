import time
from dataclasses import dataclass

import numpy as np
from numba import (
    config,
    get_num_threads,
    njit,
    prange,
    set_num_threads,
)
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ParisRoundDiagnostics:
    """Measurements for one reciprocal-neighbor contraction round."""

    component: int
    round_index: int
    active_vertices: int
    active_edges: int
    merges: int
    scan_seconds: float
    sort_seconds: float
    contraction_seconds: float
    contraction_remap_seconds: float
    contraction_filter_seconds: float
    contraction_build_seconds: float
    contraction_cleanup_seconds: float


@dataclass(frozen=True)
class ParisFitDiagnostics:
    """Measurements from a Paris hierarchy fit."""

    preprocessing_seconds: float
    component_seconds: float
    fit_seconds: float
    rounds: tuple[ParisRoundDiagnostics, ...]


@dataclass(frozen=True)
class ParisHierarchy:
    """Flat-array representation of a Paris hierarchy."""

    children: np.ndarray
    heights: np.ndarray
    sizes: np.ndarray
    component_roots: np.ndarray
    synthetic_joins: np.ndarray
    n_leaves: int
    total_weight: float
    diagnostics: ParisFitDiagnostics | None = None

    def __post_init__(self) -> None:
        n_merges = self.n_leaves - 1
        if self.children.shape != (n_merges, 2):
            raise ValueError("children must have shape (n_leaves - 1, 2)")
        for name, values in (
            ("heights", self.heights),
            ("sizes", self.sizes),
            ("synthetic_joins", self.synthetic_joins),
        ):
            if values.shape != (n_merges,):
                raise ValueError(f"{name} must have length n_leaves - 1")
        for values in (
            self.children,
            self.heights,
            self.sizes,
            self.component_roots,
            self.synthetic_joins,
        ):
            values.setflags(write=False)


@njit(cache=True, nogil=True, parallel=True)
def _nearest_neighbors(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    volumes: np.ndarray,
    logical_ids: np.ndarray,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_vertices = volumes.shape[0]
    nearest = np.full(n_vertices, -1, dtype=np.int64)
    between_weights = np.zeros(n_vertices, dtype=np.float64)
    for worker in prange(n_workers):
        for row in range(worker, n_vertices, n_workers):
            best_neighbor = -1
            best_logical_id = np.iinfo(np.int64).max
            best_similarity = -np.inf
            row_volume = volumes[row]
            for offset in range(indptr[row], indptr[row + 1]):
                neighbor = indices[offset]
                weight = data[offset]
                if neighbor == row or weight <= 0:
                    continue
                similarity = weight / (row_volume * volumes[neighbor])
                logical_id = logical_ids[neighbor]
                if similarity > best_similarity or (
                    similarity == best_similarity and logical_id < best_logical_id
                ):
                    best_neighbor = neighbor
                    best_logical_id = logical_id
                    best_similarity = similarity
                    between_weights[row] = weight
            nearest[row] = best_neighbor
    return nearest, between_weights


@njit(cache=True, nogil=True)
def _contraction_layout(
    indptr: np.ndarray,
    mapping: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_old_vertices = mapping.size
    member_counts = np.zeros(n_vertices, dtype=mapping.dtype)
    for old_vertex in range(n_old_vertices):
        member_counts[mapping[old_vertex]] += 1

    member_offsets = np.empty(n_vertices + 1, dtype=mapping.dtype)
    member_offsets[0] = 0
    for row in range(n_vertices):
        member_offsets[row + 1] = member_offsets[row] + member_counts[row]

    cursors = member_offsets[:-1].copy()
    members = np.empty(n_old_vertices, dtype=mapping.dtype)
    for old_vertex in range(n_old_vertices):
        group = mapping[old_vertex]
        members[cursors[group]] = old_vertex
        cursors[group] += 1

    workspace_offsets = np.empty(n_vertices + 1, dtype=indptr.dtype)
    workspace_offsets[0] = 0
    for row in range(n_vertices):
        degree = 0
        for position in range(member_offsets[row], member_offsets[row + 1]):
            old_vertex = members[position]
            degree += indptr[old_vertex + 1] - indptr[old_vertex]
        workspace_offsets[row + 1] = workspace_offsets[row] + degree

    return member_offsets, members, workspace_offsets


@njit(cache=True, nogil=True, parallel=True)
def _aggregate_contracted_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    mapping: np.ndarray,
    member_offsets: np.ndarray,
    members: np.ndarray,
    workspace_offsets: np.ndarray,
    n_vertices: int,
    n_workers: int,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workspace_size = workspace_offsets[-1]
    workspace_indices = np.empty(workspace_size, dtype=indices.dtype)
    workspace_data = np.empty(workspace_size, dtype=data.dtype)
    unique_counts = np.zeros(n_vertices, dtype=indices.dtype)

    for worker in prange(n_workers):
        for row in range(worker, n_vertices, n_workers):
            output_offset = workspace_offsets[row]
            unique_count = 0
            for position in range(member_offsets[row], member_offsets[row + 1]):
                old_vertex = members[position]
                for edge in range(indptr[old_vertex], indptr[old_vertex + 1]):
                    destination = mapping[indices[edge]]
                    if destination == row:
                        continue
                    value = values[worker, destination]
                    if np.isnan(value):
                        values[worker, destination] = data[edge]
                        workspace_indices[output_offset + unique_count] = destination
                        unique_count += 1
                    else:
                        values[worker, destination] = value + data[edge]

            if unique_count:
                workspace_indices[output_offset : output_offset + unique_count].sort()
                for position in range(unique_count):
                    destination = workspace_indices[output_offset + position]
                    workspace_data[output_offset + position] = values[
                        worker,
                        destination,
                    ]
                    values[worker, destination] = np.nan
            unique_counts[row] = unique_count

    return workspace_indices, workspace_data, unique_counts


@njit(cache=True, nogil=True)
def _compact_contracted_rows(
    workspace_offsets: np.ndarray,
    workspace_indices: np.ndarray,
    workspace_data: np.ndarray,
    unique_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vertices = unique_counts.size
    indptr = np.empty(n_vertices + 1, dtype=workspace_offsets.dtype)
    indptr[0] = 0
    for row in range(n_vertices):
        indptr[row + 1] = indptr[row] + unique_counts[row]

    for row in range(n_vertices):
        source_offset = workspace_offsets[row]
        target_offset = indptr[row]
        count = unique_counts[row]
        workspace_indices[target_offset : target_offset + count] = workspace_indices[
            source_offset : source_offset + count
        ]
        workspace_data[target_offset : target_offset + count] = workspace_data[
            source_offset : source_offset + count
        ]
    return (
        indptr,
        workspace_indices[: indptr[-1]],
        workspace_data[: indptr[-1]],
    )


@njit(cache=True, nogil=True)
def _raise_to_child_heights(
    children: np.ndarray,
    heights: np.ndarray,
    n_leaves: int,
) -> None:
    """Raise each merge height in place to at least its child merge heights."""
    for merge_index in range(heights.shape[0]):
        height = heights[merge_index]
        for side in range(2):
            child = children[merge_index, side]
            if child >= n_leaves and heights[child - n_leaves] > height:
                height = heights[child - n_leaves]
        heights[merge_index] = height


def canonicalize_paris_graph(graph: spmatrix) -> csr_matrix:
    """Build the canonical additive, loop-free Paris graph."""
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Paris requires a square adjacency matrix")
    if graph.shape[0] < 2:
        raise ValueError("Paris requires at least two vertices")

    directed = csr_matrix(graph, dtype=np.float64, copy=True)
    directed.sum_duplicates()
    if not np.isfinite(directed.data).all():
        raise ValueError("Paris graph weights must be finite")
    if np.any(directed.data < 0):
        raise ValueError("Paris graph weights must be non-negative")

    additive = csr_matrix(directed + directed.T, dtype=np.float64)
    additive.sum_duplicates()
    additive.setdiag(0)
    additive.eliminate_zeros()
    additive.sort_indices()
    if not np.isfinite(additive.data).all():
        raise ValueError("Paris graph weights overflowed during symmetrization")
    return additive


def _reciprocal_pairs(
    nearest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.arange(nearest.size, dtype=np.int64)
    valid = nearest >= 0
    reciprocal = np.zeros(nearest.size, dtype=bool)
    reciprocal[valid] = nearest[nearest[valid]] == vertices[valid]
    first = np.flatnonzero(reciprocal & (vertices < nearest))
    second = nearest[first]
    return first, second


@njit(cache=True, nogil=True)
def _prepare_contraction_groups(
    logical_ids: np.ndarray,
    volumes: np.ndarray,
    sizes: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    new_ids: np.ndarray,
    mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vertices = logical_ids.size
    merge_count = first.size
    unmerged_count = n_vertices - 2 * merge_count
    new_count = n_vertices - merge_count
    pair_by_vertex = np.full(n_vertices, -1, dtype=mapping.dtype)
    for pair in range(merge_count):
        pair_by_vertex[first[pair]] = pair
        pair_by_vertex[second[pair]] = pair

    new_logical_ids = np.empty(new_count, dtype=logical_ids.dtype)
    new_volumes = np.empty(new_count, dtype=volumes.dtype)
    new_sizes = np.empty(new_count, dtype=sizes.dtype)
    output = 0
    for vertex in range(n_vertices):
        if pair_by_vertex[vertex] < 0:
            mapping[vertex] = output
            new_logical_ids[output] = logical_ids[vertex]
            new_volumes[output] = volumes[vertex]
            new_sizes[output] = sizes[vertex]
            output += 1

    if output != unmerged_count:
        raise RuntimeError("Paris contraction group assembly failed")
    for pair in range(merge_count):
        output = unmerged_count + pair
        left = first[pair]
        right = second[pair]
        mapping[left] = output
        mapping[right] = output
        new_logical_ids[output] = new_ids[pair]
        new_volumes[output] = volumes[left] + volumes[right]
        new_sizes[output] = sizes[left] + sizes[right]

    return new_logical_ids, new_volumes, new_sizes


def _contract_graph(
    graph: csr_matrix,
    mapping: np.ndarray,
    n_vertices: int,
    values: np.ndarray,
) -> tuple[csr_matrix, float, float, float, float]:
    remap_start = time.perf_counter()
    member_offsets, members, workspace_offsets = _contraction_layout(
        graph.indptr,
        mapping,
        n_vertices,
    )
    remap_seconds = time.perf_counter() - remap_start

    filter_start = time.perf_counter()
    n_workers = min(values.shape[0], n_vertices)
    workspace_indices, workspace_data, unique_counts = _aggregate_contracted_rows(
        graph.indptr,
        graph.indices,
        graph.data,
        mapping,
        member_offsets,
        members,
        workspace_offsets,
        n_vertices,
        n_workers,
        values,
    )
    filter_seconds = time.perf_counter() - filter_start

    build_start = time.perf_counter()
    indptr, indices, data = _compact_contracted_rows(
        workspace_offsets,
        workspace_indices,
        workspace_data,
        unique_counts,
    )
    build_seconds = time.perf_counter() - build_start

    cleanup_start = time.perf_counter()
    contracted = csr_matrix(
        (data, indices, indptr),
        shape=(n_vertices, n_vertices),
        copy=False,
    )
    cleanup_seconds = time.perf_counter() - cleanup_start
    return (
        contracted,
        remap_seconds,
        filter_seconds,
        build_seconds,
        cleanup_seconds,
    )


def _fit_component(
    graph: csr_matrix,
    leaf_ids: np.ndarray,
    *,
    component: int,
    next_node: int,
    total_weight: float,
    round_diagnostics: list[ParisRoundDiagnostics],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int, int]:
    logical_ids = np.asarray(leaf_ids, dtype=np.int64)
    volumes = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64, copy=False)
    sizes = np.ones(logical_ids.size, dtype=np.int64)
    contraction_values = np.full(
        (min(get_num_threads(), logical_ids.size), logical_ids.size),
        np.nan,
        dtype=np.float64,
    )
    child_blocks: list[np.ndarray] = []
    height_blocks: list[np.ndarray] = []
    size_blocks: list[np.ndarray] = []
    round_index = 0

    while logical_ids.size > 1:
        active_edges = graph.nnz
        scan_start = time.perf_counter()
        n_workers = min(contraction_values.shape[0], logical_ids.size)
        nearest, between_weights = _nearest_neighbors(
            graph.indptr,
            graph.indices,
            graph.data,
            volumes,
            logical_ids,
            n_workers,
        )
        first, second = _reciprocal_pairs(nearest)
        scan_seconds = time.perf_counter() - scan_start
        if first.size == 0:
            raise RuntimeError(
                "Paris reciprocal-neighbor scan made no progress in a connected component"
            )

        merge_count = first.size
        new_ids = np.arange(next_node, next_node + merge_count, dtype=np.int64)
        first_ids = logical_ids[first]
        second_ids = logical_ids[second]
        children = np.column_stack(
            (np.minimum(first_ids, second_ids), np.maximum(first_ids, second_ids))
        )
        heights = (volumes[first] / total_weight) * (
            volumes[second] / between_weights[first]
        )
        if not np.isfinite(heights).all():
            raise ValueError("Paris merge distances must be finite")
        merged_sizes = sizes[first] + sizes[second]
        child_blocks.append(children)
        height_blocks.append(heights)
        size_blocks.append(merged_sizes)

        sort_start = time.perf_counter()
        index_dtype = graph.indices.dtype
        mapping = np.empty(logical_ids.size, dtype=index_dtype)
        new_logical_ids, new_volumes, new_sizes = _prepare_contraction_groups(
            logical_ids,
            volumes,
            sizes,
            first,
            second,
            new_ids,
            mapping,
        )
        new_count = new_logical_ids.size
        sort_seconds = time.perf_counter() - sort_start
        contract_start = time.perf_counter()
        (
            graph,
            contraction_remap_seconds,
            contraction_filter_seconds,
            contraction_build_seconds,
            contraction_cleanup_seconds,
        ) = _contract_graph(
            graph,
            mapping,
            new_count,
            contraction_values,
        )
        contract_seconds = time.perf_counter() - contract_start

        round_diagnostics.append(
            ParisRoundDiagnostics(
                component=component,
                round_index=round_index,
                active_vertices=logical_ids.size,
                active_edges=active_edges,
                merges=merge_count,
                scan_seconds=scan_seconds,
                sort_seconds=sort_seconds,
                contraction_seconds=contract_seconds,
                contraction_remap_seconds=contraction_remap_seconds,
                contraction_filter_seconds=contraction_filter_seconds,
                contraction_build_seconds=contraction_build_seconds,
                contraction_cleanup_seconds=contraction_cleanup_seconds,
            )
        )
        logical_ids = new_logical_ids
        volumes = new_volumes
        sizes = new_sizes
        next_node += merge_count
        round_index += 1

    return (
        child_blocks,
        height_blocks,
        size_blocks,
        int(logical_ids[0]),
        next_node,
    )


def _fit_components(
    graph: csr_matrix,
    leaf_ids: np.ndarray,
    components: np.ndarray,
    *,
    first_component: int,
    next_node: int,
    total_weight: float,
    round_diagnostics: list[ParisRoundDiagnostics],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Fit the connected components of a block-diagonal graph together.

    ``components`` numbers each vertex's component from zero, in fit order.
    Every component makes the merges, with the heights, that fitting it alone
    makes: a reciprocal-neighbor round reads only a component's own edges,
    and contraction keeps the relative order of its vertices and the order of
    its sums. Merges are returned component by component and numbered from
    ``next_node``, as fitting the components in turn numbers them, and so are
    the roots and the round diagnostics, whose timings are split evenly
    between the components that a round fits. Returns None, and records
    nothing, when a round fits no pair of a component or makes a non-finite
    height; fitting the components in turn then raises as it would.
    """
    n_components = int(components[-1]) + 1
    logical_ids = np.asarray(leaf_ids, dtype=np.int64)
    volumes = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64, copy=False)
    sizes = np.ones(logical_ids.size, dtype=np.int64)
    contraction_values = np.full(
        (min(get_num_threads(), logical_ids.size), logical_ids.size),
        np.nan,
        dtype=np.float64,
    )
    active = np.bincount(components, minlength=n_components)
    rounds = np.zeros(n_components, dtype=np.int64)
    merge_components: list[np.ndarray] = []
    child_blocks: list[np.ndarray] = []
    height_blocks: list[np.ndarray] = []
    size_blocks: list[np.ndarray] = []
    diagnostics: list[ParisRoundDiagnostics] = []
    node = next_node

    while np.any(active > 1):
        fitting = np.flatnonzero(active > 1)
        active_edges = np.bincount(
            components, weights=np.diff(graph.indptr), minlength=n_components
        ).astype(np.int64)
        scan_start = time.perf_counter()
        n_workers = min(contraction_values.shape[0], logical_ids.size)
        nearest, between_weights = _nearest_neighbors(
            graph.indptr,
            graph.indices,
            graph.data,
            volumes,
            logical_ids,
            n_workers,
        )
        first, second = _reciprocal_pairs(nearest)
        scan_seconds = time.perf_counter() - scan_start
        merges = np.bincount(components[first], minlength=n_components)
        if np.any(merges[fitting] == 0):
            return None

        merge_count = first.size
        new_ids = np.arange(node, node + merge_count, dtype=np.int64)
        first_ids = logical_ids[first]
        second_ids = logical_ids[second]
        children = np.column_stack(
            (np.minimum(first_ids, second_ids), np.maximum(first_ids, second_ids))
        )
        # Fitting the components in turn warns about any overflow once.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            heights = (volumes[first] / total_weight) * (
                volumes[second] / between_weights[first]
            )
        if not np.isfinite(heights).all():
            return None
        merge_components.append(components[first])
        child_blocks.append(children)
        height_blocks.append(heights)
        size_blocks.append(sizes[first] + sizes[second])

        sort_start = time.perf_counter()
        mapping = np.empty(logical_ids.size, dtype=graph.indices.dtype)
        new_logical_ids, new_volumes, new_sizes = _prepare_contraction_groups(
            logical_ids,
            volumes,
            sizes,
            first,
            second,
            new_ids,
            mapping,
        )
        # Contraction keeps the unmerged vertices in order, then one vertex
        # for each pair.
        unmerged = np.ones(logical_ids.size, dtype=bool)
        unmerged[first] = False
        unmerged[second] = False
        new_components = np.concatenate((components[unmerged], components[first]))
        sort_seconds = time.perf_counter() - sort_start
        contract_start = time.perf_counter()
        (
            graph,
            contraction_remap_seconds,
            contraction_filter_seconds,
            contraction_build_seconds,
            contraction_cleanup_seconds,
        ) = _contract_graph(
            graph,
            mapping,
            new_logical_ids.size,
            contraction_values,
        )
        contract_seconds = time.perf_counter() - contract_start

        share = 1.0 / fitting.size
        for component in fitting.tolist():
            diagnostics.append(
                ParisRoundDiagnostics(
                    component=first_component + component,
                    round_index=int(rounds[component]),
                    active_vertices=int(active[component]),
                    active_edges=int(active_edges[component]),
                    merges=int(merges[component]),
                    scan_seconds=scan_seconds * share,
                    sort_seconds=sort_seconds * share,
                    contraction_seconds=contract_seconds * share,
                    contraction_remap_seconds=contraction_remap_seconds * share,
                    contraction_filter_seconds=contraction_filter_seconds * share,
                    contraction_build_seconds=contraction_build_seconds * share,
                    contraction_cleanup_seconds=contraction_cleanup_seconds * share,
                )
            )
        rounds[fitting] += 1
        active -= merges
        node += merge_count
        logical_ids = new_logical_ids
        volumes = new_volumes
        sizes = new_sizes
        components = new_components

    # One vertex is left in each component; renumber the merges component by
    # component, keeping each component's round order.
    roots = np.empty(n_components, dtype=np.int64)
    roots[components] = logical_ids
    order = np.argsort(np.concatenate(merge_components), kind="stable")
    renumbered = np.empty(order.size, dtype=np.int64)
    renumbered[order] = np.arange(next_node, next_node + order.size, dtype=np.int64)
    children = np.concatenate(child_blocks)[order]
    merged_children = children >= next_node
    children[merged_children] = renumbered[children[merged_children] - next_node]
    merged_roots = roots >= next_node
    roots[merged_roots] = renumbered[roots[merged_roots] - next_node]
    round_diagnostics.extend(
        sorted(diagnostics, key=lambda item: (item.component, item.round_index))
    )
    return (
        children,
        np.concatenate(height_blocks)[order],
        np.concatenate(size_blocks)[order],
        roots,
        next_node + order.size,
    )


def _component_block(
    canonical: csr_matrix,
    leaf_ids: np.ndarray,
    positions: np.ndarray,
) -> csr_matrix:
    """Return the graph of whole components, in the order of ``leaf_ids``.

    Each component's vertices are ascending in ``leaf_ids``, so every row keeps
    its sorted indices and its values, as indexing rows and columns with the
    component's vertices keeps them. ``positions`` is a scratch array with one
    entry per vertex.
    """
    rows = canonical[leaf_ids]
    positions[leaf_ids] = np.arange(leaf_ids.size)
    return csr_matrix(
        (
            rows.data,
            positions[rows.indices].astype(rows.indices.dtype, copy=False),
            rows.indptr,
        ),
        shape=(leaf_ids.size, leaf_ids.size),
        copy=False,
    )


def _component_batches(
    component_sizes: np.ndarray,
    max_vertices: int,
) -> list[tuple[int, int]]:
    """Group consecutive components into runs of at most ``max_vertices``.

    A component larger than that is a run of its own.
    """
    batches: list[tuple[int, int]] = []
    first = 0
    vertices = 0
    for index, size in enumerate(component_sizes.tolist()):
        if vertices and vertices + size > max_vertices:
            batches.append((first, index))
            first = index
            vertices = 0
        vertices += size
    batches.append((first, component_sizes.size))
    return batches


# Components are fitted together until they hold this many vertices, so that
# graphs of many small components pay the per-round overhead once per batch.
_COMPONENT_BATCH_VERTICES = 4096


def fit_paris_hierarchy(
    graph: spmatrix,
    *,
    nthreads: int | None = None,
) -> ParisHierarchy:
    """Fit a deterministic Paris hierarchy with reciprocal-neighbor rounds."""
    start = time.perf_counter()
    canonical = canonicalize_paris_graph(graph)
    preprocessing_seconds = time.perf_counter() - start
    n_leaves = canonical.shape[0]
    total_weight = float(canonical.data.sum(dtype=np.float64))
    if not np.isfinite(total_weight):
        raise ValueError("Paris graph total weight must be finite")

    component_start = time.perf_counter()
    n_components, labels = connected_components(
        canonical,
        directed=False,
        return_labels=True,
    )
    component_mins = np.full(n_components, n_leaves, dtype=np.int64)
    np.minimum.at(component_mins, labels, np.arange(n_leaves, dtype=np.int64))
    component_order = np.argsort(component_mins, kind="stable")
    # Vertices in fit order: by component, ascending within each, so every
    # component and every run of consecutive components is one slice.
    component_rank = np.empty(n_components, dtype=np.int64)
    component_rank[component_order] = np.arange(n_components, dtype=np.int64)
    vertices_by_component = np.argsort(component_rank[labels], kind="stable")
    component_sizes = np.bincount(labels, minlength=n_components)[component_order]
    component_offsets = np.empty(n_components + 1, dtype=np.int64)
    component_offsets[0] = 0
    np.cumsum(component_sizes, out=component_offsets[1:])
    component_seconds = time.perf_counter() - component_start

    requested_threads = get_num_threads() if nthreads is None else max(1, int(nthreads))
    previous_threads = get_num_threads()
    set_num_threads(min(requested_threads, config.NUMBA_NUM_THREADS))
    fit_start = time.perf_counter()
    child_blocks: list[np.ndarray] = []
    height_blocks: list[np.ndarray] = []
    size_blocks: list[np.ndarray] = []
    component_roots = np.empty(n_components, dtype=np.int64)
    round_diagnostics: list[ParisRoundDiagnostics] = []
    positions: np.ndarray | None = None
    next_node = n_leaves

    def component_block(leaf_ids: np.ndarray) -> csr_matrix:
        # Only a graph of several components needs the scratch positions.
        nonlocal positions
        if positions is None:
            positions = np.empty(n_leaves, dtype=np.int64)
        return _component_block(canonical, leaf_ids, positions)

    def fit_one(component_index: int) -> None:
        nonlocal next_node
        start_offset = component_offsets[component_index]
        end_offset = component_offsets[component_index + 1]
        leaf_ids = vertices_by_component[start_offset:end_offset]
        if leaf_ids.size == 1:
            component_roots[component_index] = leaf_ids[0]
            return
        if leaf_ids.size == n_leaves:
            component_graph = canonical
        else:
            component_graph = component_block(leaf_ids)
        (
            component_children,
            component_heights,
            component_merge_sizes,
            component_root,
            next_node,
        ) = _fit_component(
            component_graph,
            leaf_ids,
            component=component_index,
            next_node=next_node,
            total_weight=total_weight,
            round_diagnostics=round_diagnostics,
        )
        child_blocks.extend(component_children)
        height_blocks.extend(component_heights)
        size_blocks.extend(component_merge_sizes)
        component_roots[component_index] = component_root

    try:
        for first, stop in _component_batches(
            component_sizes,
            _COMPONENT_BATCH_VERTICES,
        ):
            batch_sizes = component_sizes[first:stop]
            if stop - first > 1 and np.any(batch_sizes > 1):
                leaf_ids = vertices_by_component[
                    component_offsets[first] : component_offsets[stop]
                ]
                fitted = _fit_components(
                    component_block(leaf_ids),
                    leaf_ids,
                    np.repeat(np.arange(stop - first, dtype=np.int64), batch_sizes),
                    first_component=first,
                    next_node=next_node,
                    total_weight=total_weight,
                    round_diagnostics=round_diagnostics,
                )
                if fitted is not None:
                    (
                        batch_children,
                        batch_heights,
                        batch_merge_sizes,
                        batch_roots,
                        next_node,
                    ) = fitted
                    child_blocks.append(batch_children)
                    height_blocks.append(batch_heights)
                    size_blocks.append(batch_merge_sizes)
                    component_roots[first:stop] = batch_roots
                    continue
            for component_index in range(first, stop):
                fit_one(component_index)
    finally:
        set_num_threads(previous_threads)

    synthetic_count = n_components - 1
    index_dtype = np.int32 if 2 * n_leaves - 1 <= np.iinfo(np.int32).max else np.int64
    children = np.empty((n_leaves - 1, 2), dtype=index_dtype)
    heights = np.empty(n_leaves - 1, dtype=np.float64)
    size_dtype = np.int32 if n_leaves <= np.iinfo(np.int32).max else np.int64
    sizes = np.empty(n_leaves - 1, dtype=size_dtype)
    synthetic_joins = np.zeros(n_leaves - 1, dtype=bool)

    offset = 0
    for child_block, height_block, size_block in zip(
        child_blocks,
        height_blocks,
        size_blocks,
        strict=True,
    ):
        block_size = child_block.shape[0]
        children[offset : offset + block_size] = child_block
        heights[offset : offset + block_size] = height_block
        sizes[offset : offset + block_size] = size_block
        offset += block_size
    # Paris distances are reducible, so exact heights never decrease toward the
    # root. Rounding can leave a merge a few ulps below a tied child merge.
    _raise_to_child_heights(children[:offset], heights[:offset], n_leaves)

    if synthetic_count:
        # Each join adds the next component's root to the join before it.
        joins = slice(offset, offset + synthetic_count)
        children[joins, 0] = np.concatenate(
            (
                component_roots[:1],
                np.arange(next_node, next_node + synthetic_count - 1),
            )
        )
        children[joins, 1] = component_roots[1:]
        heights[joins] = np.inf
        sizes[joins] = np.cumsum(component_sizes)[1:]
        synthetic_joins[joins] = True
        next_node += synthetic_count
        offset += synthetic_count

    if offset != n_leaves - 1 or next_node != 2 * n_leaves - 1:
        raise RuntimeError(
            "Paris hierarchy does not contain exactly n_leaves - 1 merges"
        )

    fit_seconds = time.perf_counter() - fit_start
    diagnostics = ParisFitDiagnostics(
        preprocessing_seconds=preprocessing_seconds,
        component_seconds=component_seconds,
        fit_seconds=fit_seconds,
        rounds=tuple(round_diagnostics),
    )
    return ParisHierarchy(
        children=children,
        heights=heights,
        sizes=sizes,
        component_roots=component_roots.astype(index_dtype),
        synthetic_joins=synthetic_joins,
        n_leaves=n_leaves,
        total_weight=total_weight,
        diagnostics=diagnostics,
    )
