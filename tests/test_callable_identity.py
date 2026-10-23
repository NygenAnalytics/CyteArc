"""Custom callables are identified by their code, with an operation revision."""

import functools
import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import FunctionType

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.assay import norm_clr, norm_lib_size
from cytearc.storage.artifact_writer import artifact_transaction, plan_artifact
from cytearc.storage.artifacts import (
    _without_callable_code,
    artifact_path,
    callable_identity,
    inspect_artifact,
)
from cytearc.storage.lineage import ArtifactLineage
from cytearc.storage.operation_revisions import (
    OPERATION_REVISIONS,
    _NORMALIZER_CODE,
    effective_revision,
)
from cytearc.utils.logging import logger
from tests.qc_helpers import QC_VALUES, fresh_qc_store, open_qc_store

_NORMALIZER = """
def normalize(assay, counts, scale={scale}):
    return counts * scale{offset}
"""


def _cell(source: str, *, name: str = "normalize") -> FunctionType:
    """Define a function as a notebook cell does, in ``__main__``."""
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, "<notebook-cell>", "exec"), namespace)
    function = namespace[name]
    assert isinstance(function, FunctionType)
    return function


def _normalizer(scale: float = 1.0, offset: str = "") -> FunctionType:
    return _cell(_NORMALIZER.format(scale=scale, offset=offset))


@contextmanager
def _messages(level: str) -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level=level
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def test_a_function_defined_outside_cytearc_is_identified_by_its_code():
    original = callable_identity(_normalizer())

    assert set(original) == {"module", "qualname", "code"}
    assert (original["module"], original["qualname"]) == ("__main__", "normalize")
    # The same source in another cell or notebook has the same identity, also
    # when it moves within its cell.
    assert callable_identity(_normalizer()) == original
    assert (
        callable_identity(_cell("\n\n\n" + _NORMALIZER.format(scale=1.0, offset="")))
        == original
    )
    # An edited body, an edited constant, and an edited default each change it.
    assert callable_identity(_normalizer(offset=" + 1")) != original
    assert callable_identity(_normalizer(offset=" * 100.0")) != original
    assert callable_identity(_normalizer(scale=100.0)) != original


def test_nested_code_names_and_keyword_defaults_are_part_of_the_identity():
    def cell(body: str, default: str = "1") -> dict[str, str]:
        return callable_identity(
            _cell(
                "import numpy as np\n"
                f"def normalize(assay, counts, *, floor={default}):\n"
                f"    {body}\n"
            )
        )

    base = cell("return [np.log1p(value) for value in counts]")
    assert cell("return [np.log1p(value) for value in counts]") == base
    assert cell("return [np.log(value) for value in counts]") != base
    assert cell("return [np.log1p(value) + 0.5 for value in counts]") != base
    assert cell("return [np.log1p(value) for value in counts]", "2") != base
    # A generator expression compiles to nested code, unlike a comprehension.
    nested = "return list(np.log1p(value) for value in counts)"
    assert cell(nested) == cell(nested)
    assert cell(nested) != cell(nested.replace("log1p", "log"))
    assert cell("return counts[1:3]") != cell("return counts[1:4]")
    literals = "return counts in {1.5, 'a'} or (1j, b'x', None, ..., True, -0.0)"
    assert cell(literals) == cell(literals)
    assert cell(literals) != cell(literals.replace("-0.0", "0.0"))


def test_every_kind_of_code_constant_has_a_canonical_form():
    from cytearc.storage.artifacts import _code_constant

    assert _code_constant(slice(1, 3, None)) == [
        "slice",
        ["int", "1"],
        ["int", "3"],
        ["None"],
    ]
    assert _code_constant(...) == ["Ellipsis"]
    assert _code_constant(frozenset({"b", "a"})) == [
        "frozenset",
        [["str", "a"], ["str", "b"]],
    ]
    # A constant of a kind that compiled code does not hold today keeps its
    # type and representation.
    assert _code_constant(range(3)) == ["range", "range(0, 3)"]


def test_array_defaults_are_fingerprinted_and_opaque_defaults_refused():
    first = _cell("def normalize(assay, counts, weights=None):\n    return counts\n")
    second = _cell("def normalize(assay, counts, weights=None):\n    return counts\n")
    first.__defaults__ = (np.array([1.0, 2.0]),)
    second.__defaults__ = (np.array([1.0, 3.0]),)
    assert callable_identity(first) != callable_identity(second)

    second.__defaults__ = (object(),)
    with pytest.raises(
        ValueError,
        match=r"Function 'normalize' has a default value that CyteArc cannot "
        r"fingerprint \(Unsupported provenance value: object\); give it an "
        "artifact_identity",
    ):
        callable_identity(second)


def test_cytearc_functions_and_explicit_identities_keep_their_identities():
    # Every normalized artifact records these, so they must not change.
    assert callable_identity(norm_lib_size) == {
        "module": "cytearc.assay",
        "qualname": "norm_lib_size",
    }
    assert callable_identity(norm_clr) == {
        "identity": "cytearc.assay.norm_clr:feature-axis"
    }
    # A builtin has no Python code to digest.
    assert callable_identity(math.fsum) == {"module": "math", "qualname": "fsum"}

    named = _normalizer()
    named.artifact_identity = "lab-normalizer:v2"  # type: ignore[attr-defined]
    assert callable_identity(named) == {"identity": "lab-normalizer:v2"}


def _closure() -> object:
    scale = 2.0

    def normalize(assay, counts):
        return counts * scale

    return normalize


class _Normalizer:
    def __call__(self, assay, counts):
        return counts

    def method(self, assay, counts):
        return counts


@pytest.mark.parametrize(
    "value",
    [
        lambda assay, counts: counts,
        _closure(),
        _Normalizer(),
        _Normalizer().method,
        functools.partial(_normalizer(), scale=2.0),
    ],
    ids=["lambda", "closure", "callable-object", "method", "partial"],
)
def test_callables_whose_code_does_not_name_their_behavior_need_an_identity(value):
    with pytest.raises(
        ValueError,
        match="Dynamic or stateful callables must define artifact_identity: ",
    ):
        callable_identity(value)


def test_an_edited_normalizer_is_computed_again_instead_of_reused():
    store, _ = fresh_qc_store()
    ds = open_qc_store(store)
    cells = ds.snapshot_cell_selection()
    features = ds.features.universe(from_assay="RNA")
    ds.RNA.normMethod = _normalizer()
    first = ds.features.normalize(cells, features, log_transform=False)

    # The analyst edits the function in the notebook and runs the cell again.
    ds.RNA.normMethod = _normalizer(offset=" * 100.0")
    edited = ds.features.normalize(cells, features, log_transform=False)
    ds.RNA.normMethod = _normalizer(offset=" * 100.0")
    again = ds.features.normalize(cells, features, log_transform=False)

    assert edited != first and again == edited
    rows = ds.cells.active_index("I")
    np.testing.assert_allclose(ds.artifacts.load(first)["data"][:], QC_VALUES[rows])
    np.testing.assert_allclose(
        ds.artifacts.load(edited)["data"][:], 100.0 * QC_VALUES[rows]
    )
    status = ds.artifacts.inspect(edited)
    assert status.revision == 2 and status.is_current
    assert status.parameters["normalization_method"] == {
        "external_hook": True,
        **callable_identity(ds.RNA.normMethod),
    }


def _legacy(group: zarr.Group) -> None:
    """Rewrite an artifact's provenance as releases before 1.3.0 recorded it."""
    provenance = dict(group.attrs["provenance"])
    provenance.pop("revision", None)
    group.attrs["provenance"] = _without_callable_code(provenance)


def test_a_result_recorded_before_code_digests_is_recomputed_and_reported_stale():
    store, _ = fresh_qc_store()
    ds = open_qc_store(store)
    cells = ds.snapshot_cell_selection()
    features = ds.features.universe(from_assay="RNA")
    ds.RNA.normMethod = _normalizer()
    old = ds.features.normalize(cells, features, log_transform=False)
    _legacy(ds.zw[artifact_path(old)])

    legacy = ds.artifacts.inspect(old)
    assert legacy.revision == 1 and legacy.current_revision == 2
    assert not legacy.is_current
    assert legacy.superseded_by == (_NORMALIZER_CODE,)
    with _messages("INFO") as messages:
        new = ds.features.normalize(cells, features, log_transform=False)

    assert new != old
    assert ds.artifacts.inspect(new).revision == 2
    assert (
        f"Recomputing run_normalization: artifact {old.artifact_id[:12]} is "
        f"revision 1, current 2: {_NORMALIZER_CODE.change}"
    ) in messages
    assert "Status: `stale`" in ArtifactLineage.from_store(ds.zw, old).to_markdown()


def test_a_cytearc_normalizer_result_records_no_revision_and_is_reused():
    store, _ = fresh_qc_store()
    ds = open_qc_store(store)
    cells = ds.snapshot_cell_selection()
    features = ds.features.universe(from_assay="RNA")
    assert ds.RNA.normMethod is not None
    first = ds.features.normalize(cells, features)

    status = ds.artifacts.inspect(first)
    assert "revision" not in status.provenance
    assert status.revision == 1 and status.is_current
    with _messages("INFO") as messages:
        assert ds.features.normalize(cells, features) == first
    assert not any(message.startswith("Recomputing") for message in messages)


_NORMALIZER_OPERATIONS = {
    "run_marker_search": "marker_table",
    "run_normalization": "normalized",
    "run_pseudotime_aggregation": "pseudotime_aggregation",
    "run_pseudotime_marker_search": "pseudotime_markers",
    "run_statistical_testing": "statistical_tests",
    "summarize_atac_features": "feature_summary",
    "summarize_rna_features": "feature_summary",
}


@pytest.mark.parametrize(("operation", "kind"), sorted(_NORMALIZER_OPERATIONS.items()))
def test_every_operation_that_records_a_normalizer_scopes_the_revision(operation, kind):
    assert OPERATION_REVISIONS[operation] == (_NORMALIZER_CODE,)
    assert _NORMALIZER_CODE.release == "1.3.0"
    external = callable_identity(_normalizer())
    legacy = {"module": "__main__", "qualname": "normalize"}
    for method, revision in (
        (external, 2),
        ({"external_hook": True, **external}, 2),
        (legacy, 2),
        ({"module": "lab.norms", "qualname": "scaled"}, 2),
        (callable_identity(norm_lib_size), 1),
        ({"module": "cytearc", "qualname": "norm"}, 1),
        (callable_identity(norm_clr), 1),
        ({"module": 3, "qualname": "normalize"}, 1),
        (None, 1),
        ("norm_lib_size", 1),
    ):
        parameters = {"normalization_method": method}
        assert effective_revision(operation, kind, parameters, {}) == revision
    assert effective_revision(operation, kind, {}, {}) == 1

    # A new artifact that records a custom normalizer records the revision.
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation=operation,
        parameters={"normalization_method": external},
        inputs={},
        execution_options={},
    )
    assert planned.provenance["revision"] == 2
    with artifact_transaction(root, planned) as group:
        group.attrs["note"] = "payload"
    assert inspect_artifact(root, planned.ref).revision == 2
