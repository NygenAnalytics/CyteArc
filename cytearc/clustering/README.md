# Clustering

`cytearc.clustering` holds CyteArc's graph clustering: the Leiden algorithm and the Paris hierarchy.
`leiden_membership` is the Leiden entry point; `DataStore.clusters.leiden`, the pipeline's Leiden
stages and the regulon gene programs call it. Its default backend, `"cytearc"`, is CyteArc's own
multi-threaded and deterministic implementation of Leiden (Traag et al. 2019). This page describes
that implementation. The Paris hierarchy is described in the methods documentation.

## Modules

| Module | Responsibility |
| --- | --- |
| `leiden.py` | `leiden_membership`: argument checks and the choice of backend. The `"igraph"` and `"leidenalg"` backends call those GPL-licensed packages, which CyteArc does not install |
| `_leiden_native.py` | The `"cytearc"` driver: iterations, levels, refinement retries, shutdown checkpoints, the thread count, and the constants that fix the labels |
| `_leiden_graph.py` | Input checks, the symmetric graph `A = W + W^T`, weight normalisation and fixed-point node strengths |
| `_leiden_primitives.py` | Deterministic parallel building blocks: counter-based random numbers, per-worker hash accumulators, prefix sums, stable bucketing, conflict-free scatter and fixed-order sums |
| `_leiden_moving.py` | Local moving: the paper's sequential queue and the grouped parallel sweeps |
| `_leiden_refine.py` | Refinement of each community into well-connected parts |
| `_leiden_aggregate.py` | Aggregation by the refined partition, the connected-component fallback and the final label numbering |
| `_paris_core.py`, `_paris_modularity.py`, `paris.py`, `paris_multiscale.py` | The Paris hierarchy, its fixed and adaptive cuts, and their diagnostics |
| `cluster_tree.py` | The cluster tree drawn by the plotting functions |

## Method

The objective is modularity with a resolution `gamma` (the configuration null model). Every stored
entry `(i, j, w)` of the input matrix is one undirected edge, so the adjacency is `A = W + W^T`.
Each iteration starts from the previous partition, the first from singletons, and repeats three
phases level by level:

1. local moving of nodes to their best neighbouring community;
2. refinement of each community: only singletons that are well connected to the community join,
   and only refined communities that are themselves well connected, chosen at random with a
   probability that grows with the quality gain;
3. aggregation of the graph by the refined partition, with the unrefined partition as the
   aggregate graph's starting point.

The backend runs three iterations.

The backend parallelises every phase, and its labels do not depend on the thread count:

- **Local moving.** On large levels each sweep splits the active nodes into 64 random groups. The
  nodes of a group decide from the state at the start of the group, and their moves are applied
  together. A minimum-label rule (Lu et al. 2015) stops two singletons from swapping. Small levels,
  and the last few active nodes, use the paper's sequential queue.
- **Refinement.** Communities are refined in parallel, each one sequentially with its own random
  stream.
- **Aggregation.** The aggregate graph is built in two parallel passes, count then fill, with
  per-worker hash tables (as in Sahu et al. 2024).
- **Determinism.** Node strengths and community totals are exact integers in fixed point. Every
  floating-point sum has a fixed order. Random numbers come from counter-based streams keyed by the
  seed, the iteration, the level, the phase and the sweep or community. Every switch between code
  paths depends only on sizes.
- **GIL.** Every Numba kernel releases the GIL. The Python driver runs between phases, where
  shutdown checkpoints run.

Where the paper is silent or ambiguous, the implementation makes these choices. The docstrings in
`_leiden_native.py` give the details.

- **Randomness of refinement.** `theta = 0.01`, measured in units of the mean edge weight, so the
  choice does not depend on the units of the weights or on the size of the graph. Larger values,
  from 0.05 to 1, never improved modularity beyond seed noise on the benchmark graphs, and 1 clearly
  lowered it.
- **Iterations.** Three. A third iteration raised modularity on every benchmark graph and costs
  15 to 30% more time at 1M cells.
- **Strict moves.** A node moves only if it gains more than `1e-10` times its strength. Staying
  wins every tie, then the smallest community id.
- **Refinement that merges nothing.** It is retried up to 8 times with fresh random streams. If it
  still merges nothing, the communities are split into connected components and the level loop
  ends. Retries are rare (0.9% of runs on small random graphs) and were never seen on kNN graphs.
- **Hang guard.** The sequential queue stops after `1000` evaluations per node of a level. It has
  never been reached.
- **Labels.** Clusters are numbered from 1 by decreasing size, ties broken by the smallest node
  index.

These constants define the labels. After a release, changing one needs an operation revision
scoped to `backend == "cytearc"`.

## How it was written

This is a clean-room implementation.

- **Specification.** A specification was written from published papers alone: the Leiden paper
  (Traag et al. 2019), Louvain (Blondel et al. 2008), and papers on parallel Louvain and Leiden
  (Lu et al. 2015; Staudt and Meyerhenke 2016; Sahu et al. 2024).
- **Separate roles.** Separate agents held the specification, implementation, documentation and
  review roles.
- **Isolation.** None had access to the source code, documentation or tests of any Leiden or
  Louvain implementation, including igraph, leidenalg, NetworkX, scikit-network and NetworKit. None
  could reach the web. Their tool use was audited call by call.
- **Measurements.** igraph and leidenalg were run only as black boxes, to measure the numbers below.

The same work removed igraph and leidenalg from CyteArc's dependencies.

## How it was verified

- **Formulas.** Every gain, well-connectedness condition and aggregation identity was checked
  against brute-force modularity differences on 400 random graphs (6,022 checks, no failures).
- **Tests.** `tests/test_leiden_native.py` checks:
  - the formulas and the parallel primitives, against brute-force references;
  - identical labels at 1, 2, 3 and 4 threads, on the default and the forced parallel paths;
  - the paper's guarantees after a run: every community is connected, no two communities can be
    merged for a gain, and every refinement join is valid;
  - the refinement's random choice, against its closed form;
  - the release of the GIL, validation, edge cases and adversarial inputs.
- **Independent reference.** A reviewer wrote a slow sequential Leiden directly from the paper,
  before reading this code. The two agree within seed noise. On the 808-cell test graph over 60
  seeds, mean modularity is 0.76716 here and 0.76726 there.
- **Determinism at scale.** Labels were bitwise identical at 1, 4 and 8 threads on graphs of
  1 million cells.

## How it compares with other implementations

**Graphs.** Directed k-nearest-neighbour graphs with k = 11 and CyteArc's UMAP weights, which is
what `clusters.leiden` receives by default. The 808-cell graph is the test fixture; the 3,696-cell
graph is the Bastidas-Ponce pancreas. The others are synthetic: "hard" graphs have overlapping
cell types.

**Setup.**
- Modularity is at resolution 1, as a mean over seeds 0 to 4 on the 808- and 3,696-cell graphs,
  0 to 2 at 100,000 cells, 0 and 1 at 1,000,000 cells, and seed 0 alone at 10,000,000 cells. The
  808-cell entry therefore differs slightly from the 60-seed mean above.
- Times cover the whole `leiden_membership` call after an untimed warm-up.
- Memory is the peak resident memory of the process.
- The machine was an Intel Core Ultra 7 265U laptop with 8 vCPUs under WSL2. Its cores share one
  power budget, so thread scaling is lower than on a server.

| Graph | Modularity: igraph | leidenalg | cytearc |
| --- | --- | --- | --- |
| 808 cells | 0.76678 | 0.76395 | 0.76736 |
| 3,696 cells | 0.80898 | 0.80404 | 0.80954 |
| 100,000 cells, hard | 0.93673 | 0.93433 | 0.93652 |
| 1,000,000 cells | 0.93107 | 0.92887 | 0.93235 |
| 1,000,000 cells, hard | 0.92349 | 0.90193 | 0.92357 |
| 10,000,000 cells, hard | 0.95254 | not run | 0.95212 |

| Graph | igraph, 1 thread | cytearc, 1 / 4 / 8 threads | Peak memory: igraph / cytearc |
| --- | --- | --- | --- |
| 1,000,000 cells | 35 s | 26 / 13 / 10 s | 1.5 / 0.85 GB |
| 1,000,000 cells, hard | 37 s | 25 / 12 / 9.4 s | 1.5 / 0.86 GB |
| 10,000,000 cells, hard | 718 s | 164 s at 8 threads | 14.5 / 7.0 GB |

- **Agreement with igraph.** The adjusted Rand index between the native and igraph labels is
  0.99 at 1M cells and 0.97 at 10M.
- **Connectivity.** No implementation produced a disconnected community.
- **leidenalg.** Its timings were taken on a loaded machine. In those conditions it ran several
  times slower than igraph. Its modularity was the lowest of the three on every graph in the table.
- **Compilation.** The first call in a fresh environment compiles the kernels once, in about 26 s.
  Later processes load them from Numba's cache in about 0.4 s.

Labels are not expected to match other Leiden implementations label for label, because the random
choices differ.

## References

- Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected
  communities. Sci Rep 9, 5233 (2019). doi:10.1038/s41598-019-41695-z
- Blondel VD, Guillaume J-L, Lambiotte R, Lefebvre E. Fast unfolding of communities in large
  networks. J Stat Mech P10008 (2008). doi:10.1088/1742-5468/2008/10/P10008
- Lu H, Halappanavar M, Kalyanaraman A. Parallel heuristics for scalable community detection.
  Parallel Computing 47, 19-37 (2015). doi:10.1016/j.parco.2015.03.003
- Staudt CL, Meyerhenke H. Engineering parallel algorithms for community detection in massive
  networks. IEEE TPDS 27(1), 171-184 (2016). doi:10.1109/TPDS.2015.2390633
- Sahu S, Kothapalli K, Banerjee DS. Fast Leiden algorithm for community detection in shared memory
  setting. Proceedings of the 53rd International Conference on Parallel Processing (ICPP 2024),
  11-20. doi:10.1145/3673038.3673146
