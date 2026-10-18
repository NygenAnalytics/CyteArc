# Harmony2 batch correction

`cytearc.embeddings.harmony` is CyteArc's implementation of Harmony2 (Patikas et al. 2026), built on
the base algorithm of Harmony (Korsunsky et al. 2019). `fit_harmony` corrects reduced coordinates,
usually PCA, for one or more batch columns. `DataStore.reduction.harmony` and the pipeline's
`harmony` stage call it, store the corrected coordinates, and keep the fitted state that
Symphony-style reference mapping reads.

## Modules

| Module | Responsibility |
| --- | --- |
| `fit.py` | `fit_harmony`: validation, the rounds of clustering and correction, stopping, the result |
| `parameters.py` | Option checks, defaults, expansion to batch levels and clusters, `harmony_cluster_count` |
| `design.py` | Batch columns as integer level codes, with cells grouped by level combination; no one-hot matrix |
| `seeding.py` | k-means++ seeding of the cluster centroids |
| `clustering.py` | Maximum-diversity soft clustering, centroids and the objective |
| `kernels.py` | Numba kernels for the soft assignments and the per-level assignment mass |
| `correction.py` | Per-cluster ridge regressions and the corrected coordinates |
| `memory.py` | `harmony_peak_bytes`, the peak-memory estimate that the DataStore checks before a fit |
| `models.py` | `HarmonyResult` |

## Method

Each round clusters the cells softly, then corrects them:

- **Clustering.** Cells are scaled to unit length and assigned softly to clusters.
  - The diversity penalty takes Harmony2's stabilised form, which stays bounded when a batch is
    missing from a cluster. With several batch columns the penalties add, so the assignment factor
    is a product over the columns.
  - Cells are updated in blocks against batch counts that leave the block out.
- **Correction.** Each cluster runs a ridge regression of the original coordinates on an
  unpenalised intercept and one term per batch level.
  - The penalty is dynamic by default: 0.2 times the level's expected mass in the cluster.
  - Levels with almost no mass in a cluster are pruned from its regression.
  - Cells subtract the batch terms, weighted by their assignments, and keep the intercepts.
  - With one batch column the normal equations are solved in closed form, as an arrowhead system.
    With several, the largest column is eliminated and a small Schur complement is factored by
    Cholesky.

Where the papers are silent or ambiguous, the implementation makes these choices. Docstrings give
the details.

- **k-means++ seeding.** Each new seed is drawn with probability proportional to its squared
  distance from the nearest seed already chosen, by an exponential race. The key printed in the
  preprint has the opposite sign.
- **Objective.** It is divided by sigma per cluster, so a sigma per cluster is exact.
- **Warm start.** Each round starts from the previous round's assignments.
- **Stopping.** On real data the clustering and the correction can settle into two states that
  alternate from round to round. The fit therefore stops when either:
  - the objective changes by at most `epsilon_harmony` twice in a row; or
  - from round five on, a least-squares slope with the alternation removed is at most
    `epsilon_harmony`.
- **Numerical safeguards.** The solves avoid cancellation. With several batch columns, a fixed
  penalty too small for an accurate solve raises an error. A cluster whose cells cancel exactly keeps
  a zero centroid.

## How it was written

This is a clean-room implementation. A specification was written from the two papers alone, and
the code was then written from that specification. Separate agents held the specification,
implementation and review roles. None had access to the source code, documentation or tests of any
other Harmony or Symphony implementation, and their tool use was audited.

## How it was verified

- **Dense references.** `tests/test_harmony_algorithm.py` checks each step against a calculation
  written directly from the equations:
  - the ridge solves, pruning and the dynamic penalty;
  - the assignment update, as the exact minimiser of each cell's objective;
  - a whole clustering pass, against a dense pass of the 2019 algorithm;
  - the objective and the seeding distribution;
  - a synthetic set whose two groups of batches share no cell types. Over seeds 0, 1 and 2,
    batch mixing rises from between 0.07 and 0.20 before correction to about 0.65 after it,
    and cell-type purity stays at 1.0.
- **Contracts.** `tests/test_harmony_api.py` covers the API, validation, the stopping rule and failure
  handling.
- **Integration.** The DataStore, pipeline, Symphony mapping and memory-admission tests cover how the
  rest of CyteArc uses the fit.
- **Independent review.** A separate dense implementation, also written from the papers, matched
  every step and 22 whole fits to about 1e-14. Mutation testing confirmed that the tests catch
  deliberate errors in each step.
- **Scale.** On one thread, with 30 dimensions and 20 batches: 200,000 cells take 30 s and 1,000,000
  cells 271 s. The peak memory is 1.57 GB at 1M cells, within `harmony_peak_bytes`.
- **Real data.** On the 9,000 RA PBMC cells of the batch-correction tutorial, iLISI rises from 0.062
  to 0.213, while cLISI moves from 0.997 to 0.995 and graph connectivity from 0.949 to 0.947. The
  fit converges after 9 rounds.
- **Clone detection.** A token-fingerprint scan against public Harmony and Symphony implementations
  found no shared code.

Results are not expected to match other Harmony implementations number for number.

## References

- Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony.
  Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
- Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2.
  bioRxiv (2026). doi:10.64898/2026.03.16.711825
