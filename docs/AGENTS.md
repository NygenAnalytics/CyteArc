# Documentation instructions

These instructions cover documentation sources, notebook execution, and site builds.

- Author executable pages as tracked `.ipynb` files in `docs/source/`. Each notebook is the
  page's sole source and contains its prose, code, outputs, and execution metadata.
- Keep prose guides and API references in Markdown. Excluded agent pages retain their
  source-only MyST Markdown.
- Explain each logical step in notebook code with a comment. Put comments on their own lines,
  with a blank line before each comment unless it starts the code cell.
- MyST builds guides and tutorials. Sphinx builds only `docs/source/reference/api/`.
- `docs/source/developers/contributing.md` describes the authoring workflow.

## Setup and rendering

Use Node.js 24 and install the Python tools:

```bash
uv sync --group docs-modal --extra agent --extra docs --extra test --extra extra --extra cytebase
make -C docs html
```

The site is written to ignored `docs/build/site/_build/html/`. Rendering does not execute code.
Every executable notebook needs a completed execution timestamp, a matching code fingerprint,
executed code cells, and no errors. Prose edits can reuse its outputs. Changed code requires
re-execution and the updated notebook in the same change. For a lightweight source preview,
use `make -C docs preview`; unexecuted pages display a warning that figures and results are
unavailable. Preview builds still reject changed code and incomplete or failed executions.

Pages excluded by `project.exclude` in `docs/source/myst.yml` remain source-only and need no saved
execution. Builds clear the generated site and API output so excluded pages leave no old routes.
Package, dependency, and dataset changes require judgment about which pages to refresh; there
is no dependency fingerprint or cache invalidation service.

Pull requests, release checks, and GitHub Pages validate and render the notebooks from the
checkout with `make -C docs html`. CI does not execute them. Missing, failed, or stale executions
fail the build. GitHub Pages publishes from `master` at `https://docs.nygen.io/CyteArc/`.
The workflow sets `BASE_URL=/CyteArc` and places the generated site inside the artifact's
`CyteArc/` directory. Keep rendered site files outside Git.

## Execute affected pages on Modal

Ask before paid or long-running cloud work unless the current request already authorizes it.
Use an ephemeral app; never deploy. Select the environment explicitly:

```bash
make -C docs execute-docs-modal PAGES="tea_seq cell_cycle" MODAL_ENVIRONMENT=your_environment
```

One invocation submits all selected source notebooks concurrently, one container per page.
Each successful execution atomically replaces the same `.ipynb` file in `docs/source/` and
records `metadata.executed_at` and `metadata.executed_code_sha256`. The latter identifies the
executed code; do not edit this fingerprint by hand. Failed pages raise an error and leave
their previous notebook intact. Rerun the failed pages directly.

The runner automatically records each code cell's UTC start time and elapsed wall time and
displays them below its output. CPU time is not measured. Keep `%%time` out of notebook cells.

Do not start another invocation writing the same notebooks, or render while notebooks are being
replaced. Notebook pages choose their published datasets explicitly. Dataset regeneration and
publication remain separate actions requiring explicit authorization. Provider-backed pages
also need their own credentials and authorization; this runner does not inject secrets.

## Checks

```bash
uv run pytest -n 0 tests/test_docs_structure.py tests/test_docs_render.py \
    tests/test_docs_execution.py
make -C docs check-reference
```

The API check is a nitpicky, warnings-as-errors Sphinx build. Its inventory must document every
public `DataStore` method exactly once and every top-level `cytearc.__all__` export.
Inspect the rendered figures and downloaded notebook after changing execution or presentation.
