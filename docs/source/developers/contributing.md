# Contributing

## Contributions through pull requests

If you would like to add a new feature, fix a bug or make some improvements, please follow this [guideline].
When planning a new feature, introduce the proposal and discuss it on the [discussion page].
Automated contributors must also follow the repository `AGENTS.md` and any instructions scoped to the directory being changed.

The project uses [Ruff] for formatting and linting.
Before opening a pull request, run the same static checks as CI:

```bash
uv run ruff check cytearc profiling tests
uv run ruff format --check cytearc profiling tests
uv run mypy cytearc profiling
```

## Testing locally

You can run the tests locally on your branch with [pytest].
Configurations are in `pyproject.toml`.
Install the development, profiling, and test dependencies with:

```bash
uv sync --group dev --group profiling --extra test --extra extra
uv run python -m tests.download_fixtures --with-h5ad
```

Python 3.12 or newer is required (`requires-python >=3.12`).

Two markers select the expensive parts of the suite: `slow` for tests that build neighbourhood graphs or run iterative numerical workflows, and `integration` for tests that need live network access.
While iterating, skip both:

    uv run pytest -m "not slow and not integration"

CI runs the whole suite, so run `uv run pytest` before opening a pull request.
For full CI parity, also run the visual regression step (or see `AGENTS.md`):

    MPLBACKEND=Agg CYTEARC_RUN_VISUAL_REGRESSION=1 \
      uv run pytest -n 0 -m visual tests/test_plotting_showcase.py

### Performance benchmarks

`tests/benchmarks/` times the operations that dominate large runs and projects each timing onto production sizes, such as one and ten million cells.
`test_kernels.py` covers compute kernels, `test_stages.py` the `DataStore` stages, `test_cold_start.py` the import and compilation cost of a new process, and the `test_scaling_*.py` files readers, graphs, features, plotting, merge and export, and the agent.
Every benchmark also checks its result against an independent reference.
A normal test run executes each benchmark once at a small size, without timing it.
Timed runs need a quiet machine and their own process:

    CYTEARC_RUN_BENCHMARKS=1 uv run pytest -n 0 tests/benchmarks

Each benchmark times a ladder of input sizes and fits one of three models: a power law for kernels, a fixed overhead plus a per-unit rate for calls whose overhead hides their per-unit work at small sizes, and a constant for per-call overheads and the cold start of a new process.
Repeats are interleaved across sizes, so a change in machine load during the run slows every size alike instead of bending the fitted scaling.
The run prints each benchmark's largest measured time, its fitted scaling, and its projected single-thread time at the production sizes. When a baseline is available, it also reports the change from that baseline.
With a comparison baseline, a benchmark fails only when its slowdown exceeds timing noise (30% by default, after calibrating for machine speed) and projects to a meaningful delay at some production size: 2 s per million cells by default, so 2 s at one million and 20 s at ten million, or 0.5 s of fixed overhead per call.
Judging every size catches a superlinear cost that is still small at one million cells.
A growing power-law exponent also fails when it projects to a meaningful delay at ten million cells, even if small inputs look unchanged.
Scaling is judged only when both fits are tight (r-squared at least 0.97); a noisier fit is judged at its largest measured size alone.
Before failing, a benchmark measures its ladder again and keeps the fastest time at each size, so a regression must survive two measurements and transient noise cannot fail it.

When `tests/benchmarks/baseline.json` is absent, a timed run records measurements in `build/benchmarks/latest.json` without comparing performance or declaring a regression. An explicitly configured `CYTEARC_BENCHMARK_BASELINE` must exist.
For a comparison, measure the base and changed code on the same machine:

    CYTEARC_RUN_BENCHMARKS=1 CYTEARC_BENCHMARK_OUTPUT=/tmp/base.json uv run pytest -n 0 tests/benchmarks
    # then, in the changed checkout:
    CYTEARC_RUN_BENCHMARKS=1 CYTEARC_BENCHMARK_BASELINE=/tmp/base.json uv run pytest -n 0 tests/benchmarks

To create or replace the default baseline with measurements from the current code:

    CYTEARC_RUN_BENCHMARKS=1 CYTEARC_BENCHMARK_UPDATE=1 uv run pytest -n 0 tests/benchmarks

`CYTEARC_BENCHMARK_SLOWDOWN` sets the tolerated slowdown ratio, for example `1.5` when comparing across machines.
`CYTEARC_BENCHMARK_THREADS` sets the Numba thread count, one by default.
`CYTEARC_BENCHMARK_OUTPUT` sets where results are written, `build/benchmarks/latest.json` by default.

### Operation revisions

A fix that changes what an analysis operation computes for unchanged parameters and inputs must also stop CyteArc from reusing the results that earlier code stored ([](operation_revisions.md)).
To change an operation's outputs:

1. Decide with the ladder in [](operation_revisions.md) whether the change needs no revision, a scoped revision, or a revision of every artifact of the operation.
2. Append the `OperationRevision` entry to the operation's tuple in `cytearc/storage/operation_revisions.py`.
3. Add a test that a new artifact of the operation records the revision and that an artifact of the earlier revision is recomputed, not reused.
4. A change that needs no revision but still moves results is listed under {ref}`stable_identity_result_changes` on the operation revisions page.

## Contributions to the documentation

Author executable tutorials and the quick start as `.ipynb` files under `docs/source/`.
Each tracked notebook is the page's sole source, containing prose, code, outputs, and execution
metadata. Edit it directly. Prose guides and API references remain Markdown; excluded agent pages
retain their source-only MyST Markdown. MyST renders the main site; Sphinx builds the API
reference from `docs/source/reference/api/`.

Use Node.js 24 and install the Python tools:

```bash
uv sync --group docs-modal --extra agent --extra docs --extra test --extra extra --extra cytebase
```

### Execute affected notebooks

Choose the pages that need fresh results after changing their code, CyteArc, dependencies, or
input datasets. Run them concurrently in separate ephemeral Modal containers:

```bash
make -C docs execute-docs-modal PAGES="tea_seq cell_cycle" MODAL_ENVIRONMENT=your_environment
```

The runner reads the selected source notebooks and atomically replaces each successful one
at the same path in `docs/source/`. It records `metadata.executed_at` and the executed code's
`metadata.executed_code_sha256` fingerprint. Do not edit the fingerprint by hand. Commit the
updated notebook alongside any package changes that required re-execution. A failed page
leaves its previous notebook intact and reports the error; rerun it after fixing the cause.
Submit affected pages in one invocation, with one container per page. Do not execute the same
notebook concurrently or render while the runner is replacing it.
The runner consumes published datasets and does not rebuild or publish them.
Provider-backed tutorials require separate credentials and authorization.

Each code cell's output ends with its UTC start time and elapsed wall time, recorded
automatically by the runner. CPU time is not measured. Keep `%%time` out of notebook cells.

### Build and inspect

```bash
make -C docs html
```

Open `docs/build/site/_build/html/index.html` through a local static web server. This command
builds the Sphinx API, checks its public-symbol coverage, and renders MyST without executing
notebooks. Each source notebook supplies both the rendered page and its notebook download.
Prose edits can reuse outputs; changed code requires re-execution to update its fingerprint.
Every executable page needs a complete, successful execution with a matching fingerprint.

Use `make -C docs preview` while editing unexecuted notebooks. Unexecuted pages display a
source-preview warning explaining that their figures and results are unavailable. This command
still rejects changed code and incomplete or failed executions. Pages excluded through
`project.exclude` in `docs/source/myst.yml` remain source-only. Both commands rebuild the generated
site and API output so removed pages do not leave old routes behind.

Run the documentation checks with:

```bash
uv run pytest -n 0 tests/test_docs_structure.py tests/test_docs_render.py \
    tests/test_docs_execution.py
make -C docs check-reference
```

### Publish to GitHub Pages

The `GitHub Pages` workflow builds and publishes the complete site on pushes to `master`.
Pull requests, PyPI release checks, and Pages validate and render the notebooks committed in
`docs/source/` with `make -C docs html`. CI does not execute notebooks. Missing, failed, or stale
executions fail the build. After changing executable code, rerun the affected pages on Modal,
inspect the complete HTML build, and commit the updated notebooks. Prose-only changes can reuse
outputs. Rendered site files under `docs/build/` remain ignored by Git.

In the repository's Pages settings, select **GitHub Actions** as the publishing source.
Set the custom domain to `docs.nygen.io`; the documentation is served at
`https://docs.nygen.io/CyteArc/`.
The workflow checks each notebook's execution metadata and code fingerprint, then builds with
`BASE_URL=/CyteArc` and places the site inside the Pages artifact's `CyteArc/` directory.
The deployment job alone receives Pages write permissions.
A manual run of `GitHub Pages` on `master` can publish
the current commit again.

### Add a tutorial

1. Add an `.ipynb` file under `docs/source/tutorials/` with Markdown cells, code cells, and kernel metadata.
2. Register it in `docs/source/myst.yml`.
3. Execute it on Modal and inspect its figures and downloaded notebook.
4. Commit the notebook with its outputs and execution metadata. Keep rendered site files outside Git.

Fact-check method names against `cytearc/`, dataset identifiers against the `cytearc_docs`
Cytebase repository, and numerical claims against actual outputs. Keep credentials and private
connection values out of sources and notebook outputs.

### Republishing the example stores

Pages that are not about building an artifact lineage open a pre-analyzed store with `download_dataset(..., zarr=True)`.
Source stores are rebuilt from raw counts, while declared derived stores are rebuilt from their published inputs.
`scripts/regenerate_docs_datasets.py` writes both kinds and creates a manifest under
`docs/source/developers/dataset_manifests/` recording the recipe, cell counts, artifact and
pipeline-run inventories, and archive checksum:

    uv run python scripts/regenerate_docs_datasets.py --all

`--all` excludes checksum-pinned external recipes and their direct derivatives so routine regeneration does not trigger large third-party downloads.
Rebuild one of those recipes by name:

    uv run python scripts/regenerate_docs_datasets.py swanson_7K_pbmc_teaseq

Prepare Tabula Sapiens once, then derive the teaching sample from its local store:

    uv run python scripts/regenerate_docs_datasets.py tabula_sapiens tabula_sapiens_100k

The full atlas retains the authors' annotations and UMAPs. The 100,000-cell sample adds a
prepared CyteArc analysis shared by the quick start and result-reuse tutorial. Rebuilding only
`tabula_sapiens_100k` reuses `build/cytebase/tabula_sapiens/data.zarr` when it is present.
Documentation rendering reads saved notebooks and does not rebuild either dataset.

Rebuild a store whenever its recipe changes, or whenever the stored layout changes in a way that would stop the published artifacts from being reused.
Nothing leaves `build/cytebase` until you publish:

    uv run python scripts/publish_docs_datasets.py            # print the plan
    uv run python scripts/publish_docs_datasets.py --apply    # needs a write token

Publishing replaces `<dataset>/data.zarr.tar.gz` and `<dataset>/manifest.json`.
For stores used in remote examples, it also synchronizes their unpacked files.
A dataset needs a matching generated manifest and archive before it can be published.

## Releasing

Publishing a GitHub release runs `.github/workflows/publish.yml`.
Every job checks out the exact commit that the release tag named when the release was published (`github.sha`), and the upload to PyPI waits for four gates on that commit:

- `verify` runs the test workflow, `pytest.yml`: the static checks, the visual regression comparison, and the complete suite on Python 3.12, 3.13, and 3.14 and with the lowest direct dependency versions. A release uploads no coverage report.
- `docs` runs the documentation workflow, `docs.yml`: the documentation tests, the complete MyST build using committed notebooks, and the nitpicky Sphinx API build with reference coverage. Tutorial execution remains a separate Modal action.
- `build` first requires the release tag to still name that commit, then builds the wheel and the source distribution from a clean checkout and checks their metadata against the tag.
- `smoke` runs `tests/smoke_wheel.py` on the built wheel on Linux with Python 3.12, 3.13, and 3.14 and on Windows with Python 3.12.

`tests/smoke_wheel.py` checks that the wheel is pure Python and complete, installs it into a clean environment with `uv venv` and `uv pip install`, imports the public modules from the installed wheel, and runs `tests/smoke_workflow.py` with that environment's interpreter in isolated mode.
That script imports a synthetic three-population count matrix with `SparseToZarr` and analyzes it with `pipeline.run`; the run must complete, its UMAP coordinates must be finite, and Leiden must find at least two clusters.
It then exports the run's raw counts and normalized values with `to_h5ad` in that environment, which has no `anndata`, and reads both files back with h5py.
On Linux x86_64 the environment installs the `tsne` extra and the run must also produce finite t-SNE coordinates; on every other platform the environment has no `sgtsnepi`, and `embeddings.tsne` must raise the `ImportError` that names the extra.
No smoke environment may have an `sgtsne` executable on `PATH`.

Nothing is uploaded when a gate fails, and runs for the same tag publish one at a time.
Fix the cause on `master`, then publish a release whose tag names the fixed commit.
Re-running a failed workflow tests the same commit again, which helps only when the failure came from outside the commit, such as a network error.

Run the smoke locally against a wheel you build. It installs the wheel's dependencies, so it needs network access or a warm uv cache:

    uv build --wheel --clear --out-dir dist
    uv run --no-project python tests/smoke_wheel.py dist/cytearc-*.whl

## Acknowledgements

### Contributors

Contributors to the CyteArc repository.
Thank you everyone!

Gautam Ahuja, Saatvik Viniak, and Yi Su.

### Open-source stack

A diverse number of open-source packages in Python scientific stack are being used to build CyteArc.
Here we acknowledge some of them (at least those with pretty logos).

[Python](https://www.python.org), [Zarr](https://zarr.readthedocs.io),
[NumPy](https://numpy.org), [SciPy](https://scipy.org),
[UMAP](https://umap-learn.readthedocs.io), [scikit-learn](https://scikit-learn.org),
[Gensim](https://radimrehurek.com/gensim), [pandas](https://pandas.pydata.org),
[statsmodels](https://www.statsmodels.org), [NetworkX](https://networkx.org),
[Numba](https://numba.pydata.org), [Matplotlib](https://matplotlib.org),
[seaborn](https://seaborn.pydata.org), [Jupyter](https://jupyter.org), and
[Hugging Face](https://huggingface.co).

[guideline]: https://www.dataschool.io/how-to-contribute-on-github
[discussion page]: https://github.com/NygenAnalytics/CyteArc/discussions
[Ruff]: https://docs.astral.sh/ruff/
[Sphinx]: https://www.sphinx-doc.org
[MyST]: https://mystmd.org/guide
[Jupytext]: https://jupytext.readthedocs.io/en/latest/index.html
[pytest]: https://docs.pytest.org/
