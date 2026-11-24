"""HVG selections that CyteArc 1.4.0 saved stay usable.

Revision 3 of select_hvgs made every selection save the selectedDetectionRatio
diagnostic, which 1.4.0 did not save. A selection of an earlier revision reads
it as None, not measured, so the calls that require the selection and the runs
that reference it still work, while a call that selects anew computes a
selection that saves it and logs why. The tests rewrite selections that this
release saved into the record that 1.4.0 saved for the same call: the same
arrays, parameters and inputs, revision 2 where revision 2 applies and none
otherwise, diagnostics without selectedDetectionRatio, and cytearc_version
1.4.0. A selection of 2,000 cells resolves the default min_cells to 20 in both
releases, so a default call has the provenance of the saved selection.

Revision 4 marks the batch-aware selections of the cytearc flavor that record
no normalizer, as 1.4.0 saved them: such selections now record the
normalization_method and size_factor of the values they score, so a call
computes them anew, while they stay usable and lineage shows them stale.
"""

import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cytearc import DataStore, DiagnosticWarning, SparseToZarr
from cytearc.datastore._operations.features import _warn_hvg_selection
from cytearc.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    require_complete_artifact,
)
from cytearc.storage.operation_revisions import OPERATION_REVISIONS
from tests.storage_helpers import (
    by_cell_id,
    recomputation,
    recomputations,
    rewrite_as_saved_by_1_4,
)

# Batch b expresses only its first 10 genes, so a selection by batch with a
# larger top_n has a batch of too few candidates.
_CELLS, _GENES, _B_GENES = 2_000, 40, 10
_TOP_N = 20
SETTINGS: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "hvg": {"top_n": _TOP_N},
    "pca": {"dims": 5},
    "neighbors": {"k": 5},
    "umap": False,
    "leiden": False,
    "doublets": False,
    "markers": False,
}
_REVISION_2, _REVISION_3, _REVISION_4 = OPERATION_REVISIONS["select_hvgs"]
_RATIO = "selectedDetectionRatio"


def _write_store(path: Path) -> DataStore:
    """2,000 cells in batches a and b, of which b expresses only 10 of 40 genes."""
    rng = np.random.default_rng(7)
    counts = np.zeros((_CELLS, _GENES), dtype=np.int64)
    half = _CELLS // 2
    counts[:half] = rng.poisson(rng.gamma(2.0, 1.0, _GENES), size=(half, _GENES))
    counts[half:, :_B_GENES] = rng.poisson(
        rng.gamma(2.0, 1.0, _B_GENES), size=(half, _B_GENES)
    )
    SparseToZarr(
        counts,
        str(path),
        [f"c{index}" for index in range(_CELLS)],
        [f"g{index}" for index in range(_GENES)],
        shuffle_cells=False,
    ).dump()
    store = DataStore(str(path), min_features_per_cell=0)
    store.cells.insert(
        "batch", by_cell_id(store.cells, np.repeat(["a", "b"], half).astype(object))
    )
    return store


def _as_saved_by_1_4(store: DataStore, ref: ArtifactRef) -> None:
    """Rewrite a selection's record as CyteArc 1.4.0 saved it for the same call.

    The selection of revision 3 becomes revision 2 where revision 2 applies to
    its record, and revision 1 otherwise. A batch-aware selection of the
    cytearc flavor loses its normalizer, which 1.4.0 did not record.
    """
    provenance = artifact_group(store.zw, ref).attrs["provenance"]
    assert provenance["revision"] == _REVISION_3.revision
    assert _REVISION_2.applies is not None
    earlier = (
        _REVISION_2.revision
        if _REVISION_2.applies(ref.kind, provenance["parameters"], provenance["inputs"])
        else 1
    )
    rewrite_as_saved_by_1_4(store, ref, revision=earlier, without=_RATIO)
    if "batch" in provenance["inputs"] and "flavor" not in provenance["parameters"]:
        _without_normalizer(store, ref)


def _without_normalizer(store: DataStore, ref: ArtifactRef) -> None:
    """Drop the normalizer and size factor from a selection's record."""
    group = artifact_group(store.zw, ref)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    del parameters["normalization_method"], parameters["size_factor"]
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> SimpleNamespace:
    """A store whose RNA run, labeled ``saved``, holds a selection that 1.4.0 saved."""
    location = tmp_path_factory.mktemp("earlier_hvg_selections") / "data.zarr"
    store = _write_store(location)
    run = store.pipeline.rna(label="saved", **SETTINGS)
    selection = run["highly_variable_features"]
    _as_saved_by_1_4(store, selection)
    return SimpleNamespace(location=location, selection=selection)


@pytest.fixture
def copy(saved, tmp_path) -> SimpleNamespace:
    """A copy of the saved store, its run and the run's selection."""
    target = tmp_path / "data.zarr"
    shutil.copytree(saved.location, target)
    store = DataStore(str(target), min_features_per_cell=0)
    return SimpleNamespace(
        store=store,
        run=store.pipeline.open(label="saved"),
        selection=saved.selection,
    )


def _expected_recomputation(selection: ArtifactRef) -> str:
    return recomputation("select_hvgs", selection, 2, 3, _REVISION_3.change)


def _genes(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(store.artifacts.load(ref)["values"][:], dtype=bool)


def _diagnostic_warnings(call: Callable[[], Any]) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DiagnosticWarning)
    ]


def test_a_saved_selection_reads_the_ratio_as_not_measured(copy) -> None:
    status = copy.store.artifacts.inspect(copy.selection)

    # The saved run's 2,000 cells resolved min_cells 20, which revision 2
    # applies to, and only revision 3 supersedes the selection.
    assert status.parameters is not None and status.parameters["min_cells"] == 20
    assert (status.revision, status.current_revision, status.is_current) == (
        2,
        3,
        False,
    )
    assert status.superseded_by == (_REVISION_3,)
    assert status.cytearc_version == "1.4.0"
    assert status.diagnostics == {
        "selectedFeatures": _TOP_N,
        _RATIO: None,
        "batches": None,
        "batchesBelowTopN": None,
        "batchBelowTopNExamples": [],
    }
    genes = _genes(copy.store, copy.selection)
    assert int(genes.sum()) == _TOP_N
    assert require_complete_artifact(copy.store.zw, copy.selection).complete
    # The run's frozen feature view reads the selection.
    np.testing.assert_array_equal(
        np.asarray(copy.run.features.fetch("highly_variable_features"), dtype=bool),
        genes,
    )


def test_normalizing_with_a_saved_selection_reuses_the_runs_result(copy) -> None:
    run = copy.run

    normalized = copy.store.features.normalize(
        run["analysis_cell_selection"], run["highly_variable_features"]
    )

    assert normalized == run["normalized"]


def test_a_mapping_reference_builds_on_a_saved_selection(copy) -> None:
    store = copy.store

    reference = store.mapping.load_reference(
        store.mapping.build_reference(copy.run["neighbors"])
    )

    assert reference.feature_selection == copy.selection
    ids = np.asarray(store.RNA.feats.fetch_all("ids")).astype(str)
    assert reference.feature_ids.tolist() == ids[_genes(store, copy.selection)].tolist()


def test_a_plan_of_a_saved_run_names_the_revision_that_supersedes_its_selection(
    copy,
) -> None:
    plan = copy.store.pipeline.plan(copy.run)

    hvg = plan["highly_variable_features"]
    assert (hvg.status, hvg.reason, hvg.operation, hvg.existing) == (
        "computed",
        "superseded",
        "select_hvgs",
        copy.selection,
    )
    assert hvg.detail == (
        f"A revision supersedes the stored select_hvgs artifact: {_REVISION_3.change}"
    )
    assert plan["input_snapshot"].status == "reused"
    later = plan.computed[plan.computed.index("highly_variable_features") + 1 :]
    assert later and {plan[stage].reason for stage in later} == {"upstream"}


def test_a_rerun_of_a_saved_run_selects_the_same_genes_and_logs_why(copy) -> None:
    store = copy.store

    with recomputations() as logged:
        again = store.pipeline.rerun(copy.run, label="again")

    assert logged == [_expected_recomputation(copy.selection)]
    selection = again["highly_variable_features"]
    status = store.artifacts.inspect(selection)
    assert selection != copy.selection
    assert status.revision == 3 and status.is_current
    assert isinstance(status.diagnostics[_RATIO], float)
    assert status.diagnostics[_RATIO] > 0
    np.testing.assert_array_equal(
        _genes(store, selection), _genes(store, copy.selection)
    )
    # The saved run still opens with its own selection, which still reads.
    saved_run = store.pipeline.open(label="saved")
    assert saved_run["highly_variable_features"] == copy.selection
    assert store.artifacts.inspect(copy.selection).diagnostics[_RATIO] is None


def test_a_default_call_selects_anew_and_logs_why(copy) -> None:
    store = copy.store
    cells = copy.run["analysis_cell_selection"]

    with recomputations() as logged:
        selection = store.features.hvgs(cells, top_n=_TOP_N, show_plot=False)

    assert logged == [_expected_recomputation(copy.selection)]
    assert selection != copy.selection
    assert store.artifacts.inspect(selection).revision == 3
    np.testing.assert_array_equal(
        _genes(store, selection), _genes(store, copy.selection)
    )
    # The selection of the current revision is reused, without a log line.
    with recomputations() as logged:
        assert store.features.hvgs(cells, top_n=_TOP_N, show_plot=False) == selection
    assert logged == []


def test_a_batch_aware_selection_without_a_normalizer_stays_usable_and_is_stale(
    copy,
) -> None:
    store = copy.store
    cells = copy.run["analysis_cell_selection"]

    def select(top_n: int) -> ArtifactRef:
        return store.features.hvgs(cells, batch="batch", top_n=top_n, show_plot=False)

    # A selection of this release records its normalizer and is current.
    saved_by_1_4 = select(_TOP_N)
    assert store.artifacts.inspect(saved_by_1_4).revision == 3
    assert store.artifacts.inspect(saved_by_1_4).is_current
    # As 1.4.0 saved it, and as builds before revision 4 saved another.
    _as_saved_by_1_4(store, saved_by_1_4)
    earlier_build = select(_TOP_N // 2)
    _without_normalizer(store, earlier_build)

    for ref, revision, superseded_by in (
        (saved_by_1_4, 2, (_REVISION_3, _REVISION_4)),
        (earlier_build, 3, (_REVISION_4,)),
    ):
        status = store.artifacts.inspect(ref)
        assert "normalization_method" not in (status.parameters or {})
        assert (status.revision, status.current_revision, status.is_current) == (
            revision,
            4,
            False,
        )
        assert status.superseded_by == superseded_by
        # It still loads, resolves, and serves a call that passes it.
        assert int(_genes(store, ref).sum()) > 0
        assert store.features.resolve("RNA", ref) == ref
        normalized = store.features.normalize(cells, ref)
        assert store.artifacts.inspect(normalized).input_ref("feature_selection") == ref
        markdown = store.artifacts.lineage(ref).to_markdown()
        assert "- Status: `stale`" in markdown
        assert f"- Revision: `{revision}, current 4: " in markdown

    # A call computes the selection anew, with its normalizer.
    computed = select(_TOP_N)
    assert computed not in (saved_by_1_4, earlier_build)
    status = store.artifacts.inspect(computed)
    assert (status.revision, status.is_current) == (3, True)
    assert "normalization_method" in (status.parameters or {})
    # Revision 4 leaves out selections of another flavor, which read counts,
    # those that this release saves and those that 1.4.0 saved.
    other = store.features.hvgs(
        cells, flavor="seurat_v3", batch="batch", top_n=_TOP_N, show_plot=False
    )
    status = store.artifacts.inspect(other)
    assert (status.revision, status.is_current) == (3, True)
    _as_saved_by_1_4(store, other)
    assert store.artifacts.inspect(other).superseded_by == (_REVISION_3,)


def test_a_current_selection_without_the_ratio_is_refused_and_computed_anew(
    copy,
) -> None:
    store = copy.store
    cells = copy.run["analysis_cell_selection"]
    current = store.features.hvgs(cells, top_n=_TOP_N, show_plot=False)
    group = artifact_group(store.zw, current)
    # A selection of revision 3 saves the ratio, so damage or a bug lost it.
    group.attrs["diagnostics"] = {
        key: value for key, value in group.attrs["diagnostics"].items() if key != _RATIO
    }

    message = (
        "holds other diagnostics than select_hvgs saves: .*missing "
        "selectedDetectionRatio. To read it, call ds.features.hvgs again with "
        "the same arguments"
    )
    with pytest.raises(ValueError, match=message):
        store.artifacts.inspect(current).diagnostics
    with pytest.raises(ValueError, match=message):
        store.features.normalize(cells, current)
    anew = store.features.hvgs(cells, top_n=_TOP_N, show_plot=False)
    assert anew not in (current, copy.selection)
    assert isinstance(store.artifacts.inspect(anew).diagnostics[_RATIO], float)
    np.testing.assert_array_equal(_genes(store, anew), _genes(store, current))


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        # A key that select_hvgs never declared.
        (
            {
                "selectedFeatures": _TOP_N,
                "batches": None,
                "batchesBelowTopN": None,
                "batchBelowTopNExamples": [],
                "selectedGenes": _TOP_N,
            },
            "holds other diagnostics than select_hvgs saves: .*not declared "
            "selectedGenes",
        ),
        # A key that 1.4.0 saved.
        (
            {"selectedFeatures": _TOP_N, "batches": None, "batchesBelowTopN": None},
            "holds other diagnostics than select_hvgs saves: .*missing "
            "batchBelowTopNExamples",
        ),
        # No diagnostics, as releases before 1.4.0 saved selections.
        (None, "holds no diagnostics, which select_hvgs saves with every artifact"),
    ],
)
def test_a_saved_selection_is_refused_for_what_its_revision_does_not_explain(
    copy, diagnostics, message
) -> None:
    group = artifact_group(copy.store.zw, copy.selection)
    if diagnostics is None:
        del group.attrs["diagnostics"]
    else:
        group.attrs["diagnostics"] = diagnostics
    run = copy.run

    expected = f"{message}.*call ds.features.hvgs again with the same arguments"
    with pytest.raises(ValueError, match=expected):
        copy.store.artifacts.inspect(copy.selection).diagnostics
    with pytest.raises(ValueError, match=expected):
        copy.store.features.normalize(run["analysis_cell_selection"], copy.selection)
    with pytest.raises(ValueError, match=expected):
        copy.store.mapping.build_reference(run["neighbors"])


def test_a_saved_selection_warns_as_its_computation_did(copy) -> None:
    store = copy.store
    cells = copy.run["analysis_cell_selection"]
    # Every gene is a candidate of the whole selection. By batch, min_mean=-1
    # leaves out the genes that a batch does not detect, whose mean over the
    # cells that detect them is 0, so batch b has its 10 genes.
    expected = {
        500: [
            f"Only {_GENES} features passed the HVG candidate filters, fewer than "
            "top_n=500; all of them are selected"
        ],
        15: [
            "1 of 2 batches have fewer HVG candidates than top_n=15, so all of their "
            f"candidates count as highly variable in them: b ({_B_GENES})"
        ],
    }
    for top_n, options in ((500, {}), (15, {"batch": "batch", "min_mean": -1})):
        computed, given = _diagnostic_warnings(
            lambda: store.features.hvgs(cells, top_n=top_n, show_plot=False, **options)
        )
        assert given == expected[top_n]
        _as_saved_by_1_4(store, computed)
        assert store.artifacts.inspect(computed).diagnostics[_RATIO] is None

        # The warning that a computed or reused selection gives, from what the
        # saved selection measured.
        _, again = _diagnostic_warnings(
            lambda: _warn_hvg_selection(store.zw, computed, top_n)
        )

        assert again == expected[top_n]
