# Agent validation

Agent tests live in `tests/test_agent_*.py` and participate in normal repository discovery.
See [README.md](README.md#code-map-migration-and-local-verification) for focused test commands
and the line-coverage gate.

## Offline fixtures

`tests/fixtures_agent.py` supplies a temporary RNA source and disables live model requests for
agent test modules. It is registered by `tests/conftest.py`. Independent H5AD scalar parsing and
nullable metadata fingerprint checks live in `tests/test_h5ad_inspect_columns.py` and
`tests/test_metadata_blocks.py`. `tests/test_import_architecture.py` checks dependency isolation.

The [agent tutorial](../../docs/source/tutorials/agent_workflow.md) uses a small synthetic prepared
store and a local `FunctionModel`. It leaves synthetic identities unassigned and requires no
dataset download or model credentials. Offline checks do not establish biological quality or
live-provider reliability.

## Persistence and scientific results

The readable audit stays outside the numerical store. Omitting `run_dir` selects
`Path.cwd() / "agent_runs" / <runId>`; an explicit external directory is also supported. The same
ID identifies the immutable manifest and the compact completed result under local Zarr
`agent_results/<runId>`.

The compact record connects the final core pipeline and its configuration to the selected
candidate, selection rationale, workspace, source and procedure identities, and external audit
locator. `AnalysisRun.compact_result` verifies the source and final run before reading it, and
never creates or migrates the record. Publication failures stay separate from scientific
completion; an explicit resume of a completed run can retry publication.

Checks cover cohort and feature alignment, exact final-artifact binding, marker reuse, source
relocation, saved decision replay, incomplete statuses, interruption, provider budgets, and
validation before any replacement locator is saved. Missing measurements stay distinct from
zeroes, and policy resolutions stay separate from accepted model responses.

## Reports and exports

Report checks cover offline rendering, escaped evidence, bounded image reads, embedded assets
and their licenses, all saved outcome statuses, natural cluster ordering, and exports aligned
with the final clustering. Report regeneration must preserve scientific records.

Plot checks cover read-only access, bounded marker selection, duplicate gene names, missing
statistics versus zero expression, fraction-to-area scaling, atomic saves, figure cleanup, and
plot failures that preserve completed scientific status.
