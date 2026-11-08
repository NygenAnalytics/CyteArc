"""The diagnostics that artifacts save, and the warnings that operations derive from them.

An operation declares its diagnostics on its arguments class; the writer
refuses others, the reader refuses an artifact that holds others or none,
and ``ds.artifacts.inspect(ref).diagnostics`` reads them for any artifact.
Operations warn from saved diagnostics, so a reused artifact warns as a
computed one does.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cytearc.metadata.arguments as arguments_module
from cytearc import DiagnosticWarning, ExportSummary
from cytearc.clones.allele_clones import ALLELE_DIAGNOSTICS
from cytearc.clones.barcode_clones import BARCODE_DIAGNOSTICS
from cytearc.metadata.arguments import (
    CLONE_DIAGNOSTICS,
    GROUPING_DIAGNOSTICS,
    JOIN_DIAGNOSTICS,
    AlleleCloneArguments,
    BarcodeCloneArguments,
    ClonalExpansionArguments,
    CloneImportArguments,
    GuideCallImportArguments,
)
from cytearc.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from cytearc.storage.artifacts import artifact_group, require_complete_artifact
from cytearc.storage.diagnostics import (
    EXAMPLES,
    DiagnosticRule,
    checked_diagnostics,
    declare_diagnostics,
    declared_diagnostics,
    export_summary,
    warn_diagnostics,
)
from cytearc.utils.logging import collect_warnings
from cytearc.utils.warnings import warn
from tests.storage_helpers import open_store, write_count_store, written

# An operation that only these tests run: it saves a count, a share and
# examples of what it counted.
_OPERATION = "count_test_cells"
_KEYS = ("testedCells", "testedShare", "testedCellExamples")
_CALL = "ds.tests.count_cells"
declare_diagnostics(_OPERATION, _KEYS, call=_CALL)
# An operation that declares no diagnostics.
_UNDECLARED = "count_test_cells_without_diagnostics"
_SAVED = {"testedCells": 3, "testedShare": 0.75, "testedCellExamples": ["c0", "c2"]}


@pytest.fixture
def store(tmp_path):
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.random.default_rng(2).poisson(2.0, size=(4, 3))},
        "uint16",
        cell_ids=["c0", "c1", "c2", "c3"],
    )
    return open_store(tmp_path / "store.zarr")


def _plan(store: Any, operation: str = _OPERATION, value: int = 1) -> Any:
    selection = store.snapshot_cell_selection()
    return plan_cell_data_artifact(
        store.zw,
        scope="datastore",
        kind="clones",
        operation=operation,
        parameters={"value": value},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"values": ((4,), None)},
    )


def _write(store: Any, planned: Any, **options: Any) -> None:
    write_cell_data_artifact(
        store.zw,
        planned,
        {"values": np.asarray(["a", "b", "", "a"], dtype=object)},
        **options,
    )


_KEYS_OF_EVERY_KIND = (
    "count",
    "share",
    "label",
    "unmeasured",
    "barcodeExamples",
)


@pytest.mark.parametrize(
    ("values", "saved"),
    [
        (
            {
                "count": np.int64(3),
                "share": np.float32(0.5),
                "label": "x",
                "unmeasured": None,
                "barcodeExamples": ("a -> s1__a",),
            },
            {
                "count": 3,
                "share": 0.5,
                "label": "x",
                "unmeasured": None,
                "barcodeExamples": ["a -> s1__a"],
            },
        ),
        (
            {
                "count": 0,
                "share": 1e300,
                "label": "",
                "unmeasured": 7,
                "barcodeExamples": ["1", "2", "3", "4", "5"],
            },
            {
                "count": 0,
                "share": 1e300,
                "label": "",
                "unmeasured": 7,
                "barcodeExamples": ["1", "2", "3", "4", "5"],
            },
        ),
    ],
)
def test_diagnostics_are_numbers_texts_none_or_a_few_example_texts(
    values, saved
) -> None:
    checked = checked_diagnostics(values, _KEYS_OF_EVERY_KIND, operation="op")
    assert checked == saved
    # NumPy numbers become Python numbers, in the declared order.
    assert [type(value) for value in checked.values()] == [
        type(value) for value in saved.values()
    ]
    assert list(checked) == list(_KEYS_OF_EVERY_KIND)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("count", True, "not the boolean True"),
        ("count", np.bool_(False), r"not the boolean (np\.)?False"),
        ("share", float("nan"), "must be finite"),
        ("share", float("inf"), "must be finite"),
        ("share", np.float64("-inf"), "must be finite"),
        ("label", {"nested": 1}, "must be a number, a text or None"),
        ("label", [1, 2], "must be a number, a text or None"),
        ("barcodeExamples", "a", "must be a list of texts"),
        ("barcodeExamples", [1], "must be a list of texts"),
        ("barcodeExamples", [str(n) for n in range(EXAMPLES + 1)], "at most 5"),
    ],
)
def test_other_values_are_refused(key, value, message) -> None:
    values: dict[str, Any] = {
        "count": 1,
        "share": 0.5,
        "label": "x",
        "unmeasured": None,
        "barcodeExamples": [],
    }
    values[key] = value
    with pytest.raises(ValueError, match=message):
        checked_diagnostics(values, _KEYS_OF_EVERY_KIND, operation="op")


def test_diagnostics_hold_exactly_the_declared_keys() -> None:
    with pytest.raises(ValueError, match="must be exactly a, b: missing b$"):
        checked_diagnostics({"a": 1}, ("a", "b"), operation="op")
    with pytest.raises(ValueError, match="missing b; not declared c, d$"):
        checked_diagnostics({"a": 1, "d": 2, "c": 3}, ("a", "b"), operation="op")
    with pytest.raises(ValueError, match="must be a mapping"):
        checked_diagnostics([("a", 1)], ("a",), operation="op")


def test_an_operation_declares_camel_case_keys_once() -> None:
    for key in ("UnmatchedBarcodes", "n_unmatched", "unmatched-barcodes", "", "x y"):
        with pytest.raises(ValueError, match="must be camelCase"):
            declare_diagnostics("declare_test_keys", (key,), call="ds.x")
    with pytest.raises(ValueError, match="declares a diagnostic twice"):
        declare_diagnostics("declare_test_keys", ("a", "a"), call="ds.x")
    with pytest.raises(ValueError, match="must name the call"):
        declare_diagnostics("declare_test_keys", ("a",), call="")
    assert declared_diagnostics("declare_test_keys") == ()
    declare_diagnostics("declare_test_keys", ("a", "b"), call="ds.x")
    # The same declaration again, as a rebuilt dataclass makes it, holds.
    declare_diagnostics("declare_test_keys", ("a", "b"), call="ds.x")
    assert declared_diagnostics("declare_test_keys") == ("a", "b")
    with pytest.raises(ValueError, match=r"already, not \['a'\]"):
        declare_diagnostics("declare_test_keys", ("a",), call="ds.x")
    assert declared_diagnostics(None) == ()


#: The keys that each operation of the clone stack, repertoire, variants and
#: perturbation screens saves, beside the shared ones, as the API reference
#: gives them.
_OWN_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "CloneImportArguments": (
        (*JOIN_DIAGNOSTICS, *CLONE_DIAGNOSTICS),
        ("fileRows", "cellsInSeveralClones"),
    ),
    "CloneColumnArguments": (CLONE_DIAGNOSTICS, ()),
    "ImmuneReceptorImportArguments": (
        JOIN_DIAGNOSTICS,
        ("fileContigs", "unmatchedContigs", "cellsWithChain"),
    ),
    "ClonotypeArguments": (CLONE_DIAGNOSTICS, ()),
    "LineageBarcodeImportArguments": (
        JOIN_DIAGNOSTICS,
        ("fileRows", "unmatchedRows", "emptyRows"),
    ),
    "BarcodeCloneArguments": (CLONE_DIAGNOSTICS, BARCODE_DIAGNOSTICS),
    "RecorderAlleleImportArguments": (
        JOIN_DIAGNOSTICS,
        ("fileRows", "unmatchedRows", "slotsBeyondLength", "rowsWithoutIntegration"),
    ),
    "AlleleCloneArguments": (CLONE_DIAGNOSTICS, ALLELE_DIAGNOSTICS),
    "PhylogenyImportArguments": (
        JOIN_DIAGNOSTICS,
        ("cellsInNoTree", "unnamedLeaves", "leafTimeSpread"),
    ),
    "PhylogenyCloneArguments": (CLONE_DIAGNOSTICS, ("cellsBeforeCut",)),
    "VariantCountImportArguments": (
        JOIN_DIAGNOSTICS,
        (
            "listedCells",
            "sortedRuns",
            "repeatedCalls",
            "skippedEmptyAlternatives",
            "skippedOtherAlleles",
            "skippedVariantExamples",
        ),
    ),
    "VariantSelectionArguments": (
        (),
        ("cells", "variants", "informativeVariants", "maskedVariants"),
    ),
    "FrequencyCloneArguments": (
        CLONE_DIAGNOSTICS,
        (
            "variants",
            "cellsScored",
            "cellsWithoutCoverage",
            "cellsWithoutAlternative",
            "cellsWithoutNeighbors",
            "edges",
        ),
    ),
    "VariantGenotypeArguments": (
        (),
        (
            "wildTypeCalls",
            "heterozygousCalls",
            "homozygousAlternativeCalls",
            "alternativePresentCalls",
            "missingCalls",
            "filteredCalls",
            "alternativesNotShown",
        ),
    ),
    "GenotypeCloneArguments": (
        CLONE_DIAGNOSTICS,
        (
            "sites",
            "mutantCells",
            "ambiguousCells",
            "cellsInSmallClones",
            "dropoutProbability",
            "errorProbability",
            "cappedSearches",
            "cellsInCappedSearches",
        ),
    ),
    "ClonalExpansionArguments": (GROUPING_DIAGNOSTICS, ()),
    "CloneFateBiasArguments": (
        GROUPING_DIAGNOSTICS,
        ("earlyCellsWithoutLateCells", "earlyCellsWithTiedFates", "testedClones"),
    ),
    "GuideCallImportArguments": (JOIN_DIAGNOSTICS, ()),
    "GuideCallArguments": (
        (),
        (
            "guideFits",
            "fitsCallingNoCell",
            "fitsNoCells",
            "fitsTooSparse",
            "fitsNoFit",
            "fitsNoPositive",
            "fitCallingNoCellExamples",
            "cellsWithCall",
            "cellsWithSeveralGuides",
        ),
    ),
    "PerturbationEfficiencyArguments": (
        (),
        ("rowsWithoutFeature", "rowWithoutFeatureExamples"),
    ),
    "PerturbationAssociationArguments": (
        (),
        (
            "calibrationStatus",
            "controlGuides",
            "calibrationGrade",
            "falseDiscoveries",
            "negativeControlPairs",
            "rowsWithoutStatistic",
            "rowWithoutStatisticExamples",
            "calibrationGroupsWithoutStatistic",
            "calibrationGroupWithoutStatisticExamples",
        ),
    ),
    "PerturbationProfileArguments": (
        (),
        ("leftOutPerturbations", "leftOutPerturbationExamples"),
    ),
}


@pytest.mark.parametrize("name", sorted(_OWN_KEYS))
def test_each_operation_declares_the_keys_of_the_table(name) -> None:
    arguments = getattr(arguments_module, name)
    shared, own = _OWN_KEYS[name]
    assert arguments.diagnostics == (*shared, *own)
    assert declared_diagnostics(arguments.operation) == arguments.diagnostics
    assert arguments.call is not None and arguments.call.startswith("ds.")


def test_the_arguments_classes_declare_their_shared_keys() -> None:
    assert GuideCallImportArguments.diagnostics == JOIN_DIAGNOSTICS
    assert ClonalExpansionArguments.diagnostics == GROUPING_DIAGNOSTICS
    # Barcode and allele clones save the writer's keys and the numbers their
    # calling gives, which the calling code names.
    assert BarcodeCloneArguments.diagnostics == (
        *CLONE_DIAGNOSTICS,
        *BARCODE_DIAGNOSTICS,
    )
    assert AlleleCloneArguments.diagnostics == (*CLONE_DIAGNOSTICS, *ALLELE_DIAGNOSTICS)
    assert declared_diagnostics(CloneImportArguments.operation) == (
        CloneImportArguments.diagnostics
    )
    from cytearc.graph.arguments import OperationArguments

    with pytest.raises(TypeError, match="must name the call"):

        class _Uncalled(OperationArguments):
            operation = "declare_test_uncalled"
            artifact_kind = "clones"
            diagnostics = ("a",)


def test_the_writer_saves_the_declared_diagnostics_and_refuses_others(store) -> None:
    planned = _plan(store)
    before = written(store)
    for options, message in (
        ({}, "saves the diagnostics testedCells, testedShare, testedCellExamples"),
        ({"diagnostics": {**_SAVED, "otherCells": 1}}, "not declared otherCells"),
        ({"diagnostics": {"testedCells": 3}}, "missing testedShare"),
        ({"diagnostics": {**_SAVED, "testedShare": float("nan")}}, "must be finite"),
        (
            {"diagnostics": _SAVED, "attributes": {"diagnostics": _SAVED}},
            "passes its diagnostics as an attribute",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _write(store, planned, **options)
        # Nothing is written before the refusal.
        assert written(store) == before
    _write(store, planned, diagnostics=_SAVED)
    status = store.artifacts.inspect(planned.ref)
    assert status.diagnostics == _SAVED
    assert list(status.diagnostics) == list(_KEYS)


def test_an_operation_without_declared_diagnostics_saves_and_reads_none(
    store,
) -> None:
    planned = _plan(store, _UNDECLARED)
    with pytest.raises(ValueError, match="declares no diagnostics"):
        _write(store, planned, diagnostics={"testedCells": 1})
    _write(store, planned, diagnostics={})
    assert store.artifacts.inspect(planned.ref).diagnostics == {}
    # What such an artifact holds under the attribute, such as the list of a
    # Paris cut, is its own, and the reader leaves it out.
    artifact_group(store.zw, planned.ref).attrs["diagnostics"] = [{"cluster": 1}]
    assert store.artifacts.inspect(planned.ref).diagnostics == {}
    require_complete_artifact(store.zw, planned.ref)


def test_the_reader_refuses_an_artifact_without_its_operations_diagnostics(
    store,
) -> None:
    planned = _plan(store)
    _write(store, planned, diagnostics=_SAVED)
    group = artifact_group(store.zw, planned.ref)
    # An artifact that other code saved with other keys.
    group.attrs["diagnostics"] = {**_SAVED, "otherCells": 2}
    with pytest.raises(
        ValueError,
        match=f"holds other diagnostics than {_OPERATION} saves: .*not declared "
        f"otherCells. To read it, call {_CALL} again with the same arguments",
    ):
        store.artifacts.inspect(planned.ref).diagnostics
    # An artifact saved before CyteArc recorded diagnostics: loaders refuse it
    # and name the call that computes it anew.
    del group.attrs["diagnostics"]
    message = (
        f"holds no diagnostics, which {_OPERATION} saves with every artifact; "
        f"call {_CALL} again with the same arguments, which computes the "
        "artifact anew and leaves this one unused"
    )
    with pytest.raises(ValueError, match=message):
        store.artifacts.inspect(planned.ref).diagnostics
    with pytest.raises(ValueError, match=message):
        require_complete_artifact(store.zw, planned.ref)
    # Planning the same call reuses it no longer: a new artifact is computed,
    # and the old one stays in the store.
    again = _plan(store)
    assert not again.reused and again.ref != planned.ref
    _write(store, again, diagnostics=_SAVED)
    assert _plan(store).ref == again.ref
    assert _plan(store).reused
    assert planned.ref in store.artifacts.list(scope="datastore")


def test_a_loader_refuses_clones_saved_without_diagnostics(store) -> None:
    store.cells.insert("clone", np.asarray(["a", "a", "b", ""], dtype=object))
    cells = store.snapshot_cell_selection()
    clones = store.clones.from_column("clone", cell_selection=cells)
    assert store.clones.load_clones(clones).diagnostics == {
        "cellsWithClone": 3,
        "largestCloneShare": 2 / 3,
    }
    # The clones as CyteArc saved them before it recorded diagnostics.
    del artifact_group(store.zw, clones).attrs["diagnostics"]
    with pytest.raises(ValueError, match="call ds.clones.from_column again"):
        store.clones.load_clones(clones)
    with pytest.raises(ValueError, match="call ds.clones.from_column again"):
        store.clones.expansion(clones)
    # The call computes them anew.
    anew = store.clones.from_column("clone", cell_selection=cells)
    assert anew != clones
    assert store.clones.load_clones(anew).diagnostics["cellsWithClone"] == 3


def test_expansion_counts_the_cells_that_it_leaves_out(store) -> None:
    # Three cells with a clone, one of them without a sample.
    store.cells.insert("clone", np.asarray(["a", "a", "b", ""], dtype=object))
    store.cells.insert("sample", np.asarray(["s1", "s2", "", "s1"], dtype=object))
    clones = store.clones.from_column(
        "clone", cell_selection=store.snapshot_cell_selection()
    )
    with collect_warnings():
        expansion = store.clones.expansion(clones, group_by="sample")
    # Cell c3 has no clone, c2 has a clone but no sample, and s1 and s2 are
    # the groups.
    assert store.artifacts.inspect(expansion).diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 1,
        "groups": 2,
    }


def test_a_diagnostic_warning_names_its_operation_diagnostic_and_value() -> None:
    warning = DiagnosticWarning(
        "3 barcodes match no cell", "import_clones", "unmatchedBarcodes", 3
    )
    assert isinstance(warning, UserWarning)
    assert str(warning) == "3 barcodes match no cell"
    for value in (3, 0.25, "text", None, ["a", "b"]):
        given = DiagnosticWarning("message", "op", "someCount", value)
        copied = pickle.loads(pickle.dumps(given))
        assert type(copied) is DiagnosticWarning
        assert (copied.args, copied.operation, copied.diagnostic, copied.value) == (
            ("message",),
            "op",
            "someCount",
            value,
        )
        assert str(copied) == "message"


class _Arguments:
    operation = _OPERATION


_RULES = (
    DiagnosticRule(
        "testedCells",
        applies=lambda numbers: numbers["testedCells"] > 2,
        message=lambda numbers: f"{numbers['testedCells']} cells were tested",
    ),
)


def test_rules_warn_from_saved_diagnostics_in_their_own_category() -> None:
    with pytest.warns(DiagnosticWarning) as caught, collect_warnings() as messages:
        warn_diagnostics(_Arguments(), _SAVED, _RULES)
    (record,) = caught
    assert str(record.message) == "3 cells were tested"
    assert isinstance(record.message, DiagnosticWarning)
    assert (record.message.operation, record.message.diagnostic) == (
        _OPERATION,
        "testedCells",
    )
    assert record.message.value == 3
    # The warning points at the caller's code, outside the package.
    assert record.filename == __file__
    assert list(messages) == ["3 cells were tested"]
    # A rule that does not apply gives nothing, and callers silence CyteArc's
    # own warnings by their category.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_diagnostics(_Arguments(), {**_SAVED, "testedCells": 2}, _RULES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DiagnosticWarning)
            warn_diagnostics(_Arguments(), _SAVED, _RULES)
    with pytest.raises(
        KeyError, match="'testedCells', which its artifact does not hold"
    ):
        warn_diagnostics(_Arguments(), {}, _RULES)


def test_a_diagnostic_warning_can_be_issued_from_text_alone() -> None:
    # Python's warnings make a warning of a category from its text.
    with pytest.warns(DiagnosticWarning) as caught:
        warnings.warn("a study's own threshold", DiagnosticWarning, stacklevel=1)
    (record,) = caught
    assert str(record.message) == "a study's own threshold"
    assert (
        record.message.operation,
        record.message.diagnostic,
        record.message.value,
    ) == (None, None, None)


def test_text_warnings_keep_their_category() -> None:
    with pytest.warns(UserWarning) as caught, collect_warnings() as messages:
        warn("plain text")
    assert type(caught[0].message) is UserWarning
    assert list(messages) == ["plain text"]
    with pytest.warns(DeprecationWarning):
        warn("old", DeprecationWarning)


def test_reading_the_diagnostics_of_an_absent_artifact_reads_none(
    store, tmp_path: Path
) -> None:
    planned = _plan(store)
    # The plan names an artifact that is not written yet.
    assert store.artifacts.inspect(planned.ref).diagnostics == {}


def test_a_stage_reports_the_diagnostics_of_an_output_without_a_summary(store) -> None:
    from cytearc.datastore._pipeline.summaries import summarize_stage

    store.cells.insert("clone", np.asarray(["a", "a", "b", ""], dtype=object))
    clones = store.clones.from_column(
        "clone", cell_selection=store.snapshot_cell_selection()
    )
    assert summarize_stage(store, "clones", [("clones", clones)]) == {
        "cellsWithClone": 3,
        "largestCloneShare": 2 / 3,
    }
    # An output whose operation saves no diagnostics reports nothing.
    planned = _plan(store, _UNDECLARED)
    _write(store, planned)
    assert summarize_stage(store, "clones", [("clones", planned.ref)]) is None
    assert summarize_stage(
        store, "clones", [("clones", clones), ("other", planned.ref)]
    ) == {"cellsWithClone": 3, "largestCloneShare": 2 / 3}


def test_the_clone_writer_refuses_a_source_that_passes_its_own_keys(store) -> None:
    from cytearc.clones.assignments import CloneAssignment
    from cytearc.datastore._operations.clone_artifacts import (
        clone_diagnostics,
        write_clones,
    )
    from cytearc.utils.memory_budget import MemoryLedger

    assignment = CloneAssignment(
        codes=np.asarray([0, 0, 1, -1]),
        clone_ids=np.asarray(["a", "b"], dtype=object),
        clone_sizes=np.asarray([2, 1]),
        probabilities=None,
        n_cells_in_several_clones=0,
    )
    # Three cells given a clone, two of them in the largest, by hand.
    assert clone_diagnostics(assignment.codes, assignment.clone_sizes) == {
        "cellsWithClone": 3,
        "largestCloneShare": 2 / 3,
    }
    planned = _plan(store)
    before = written(store)
    with pytest.raises(ValueError, match="saves cellsWithClone for every source"):
        write_clones(
            store.zw,
            planned,
            assignment,
            {},
            ledger=MemoryLedger(None, "Saving clones"),
            diagnostics={"cellsWithClone": 3},
        )
    assert written(store) == before


def test_an_export_summary_checks_its_keys_and_warns_by_its_rules(tmp_path) -> None:
    rules = (
        DiagnosticRule(
            "renamed",
            lambda saved: saved["renamed"] > 0,
            lambda saved: f"{saved['renamed']} renamed",
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = export_summary(
            "export_test",
            tmp_path / "out.tsv",
            np.int64(4),
            {"renamed": 2},
            ("renamed",),
            rules,
        )
    assert summary == ExportSummary(
        path=tmp_path / "out.tsv", rows=4, diagnostics={"renamed": 2}
    )
    assert type(summary.rows) is int
    (warning,) = caught
    assert isinstance(warning.message, DiagnosticWarning)
    assert (
        warning.message.operation,
        warning.message.diagnostic,
        warning.message.value,
    ) == (
        "export_test",
        "renamed",
        2,
    )
    assert str(warning.message) == "2 renamed"
    # Nothing to warn of warns nothing, and keys other than those declared,
    # or a value that no diagnostic holds, are refused before any warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_summary("export_test", "out.tsv", 0, {"renamed": 0}, ("renamed",), rules)
    assert caught == []
    with pytest.raises(ValueError, match="missing renamed; not declared other"):
        export_summary("export_test", "out.tsv", 0, {"other": 1}, ("renamed",), rules)
    with pytest.raises(ValueError, match="must be finite"):
        export_summary(
            "export_test", "out.tsv", 0, {"renamed": float("nan")}, ("renamed",), rules
        )
