"""The errors each loader raises, and that its docstring documents them.

A loader of a saved result raises ``TypeError`` for an argument that is not
an ArtifactRef, ``ValueError`` for an artifact of another kind, ``KeyError``
for one the store does not hold and ``RuntimeError`` for one whose write
stopped before it finished. Its message names the argument passed, and its
``Raises:`` section lists every class it raises here. ``artifacts.load_values``
reads cell-aligned arrays of any kind and raises ``ValueError`` for a missing
or incomplete artifact alike.
"""

import inspect
import re
import warnings

import numpy as np
import pytest

from cytearc.storage.artifacts import artifact_path
from cytearc.storage.refs import ArtifactRef
from tests.storage_helpers import open_store, write_count_store

# Each loader: its namespace and method, the kind it reads, and its scope.
LOADERS = [
    ("clones", "load_clones", "clones", "datastore"),
    ("clones", "load_expansion", "clonal_expansion", "datastore"),
    ("clones", "load_fate_bias", "clone_fate_bias", "datastore"),
    ("clones", "load_barcodes", "lineage_barcodes", "datastore"),
    ("clones", "load_alleles", "lineage_alleles", "datastore"),
    ("clones", "load_phylogeny", "cell_phylogeny", "datastore"),
    ("clones", "diversity", "clones", "datastore"),
    ("clones", "overlap", "clones", "datastore"),
    ("clones", "coupling", "clones", "datastore"),
    ("clones", "track", "clones", "datastore"),
    ("repertoire", "load_receptors", "immune_receptors", "datastore"),
    ("repertoire", "load_clonotypes", "clonotypes", "datastore"),
    ("repertoire", "gene_usage", "clonotypes", "datastore"),
    ("repertoire", "spectratype", "clonotypes", "datastore"),
    ("variants", "load_counts", "variant_counts", "datastore"),
    ("variants", "load_informative", "variant_selection", "datastore"),
    ("variants", "load_genotypes", "variant_genotypes", "datastore"),
    ("perturbation", "load_guide_calls", "guide_calls", "assay"),
    ("perturbation", "load_guide_thresholds", "guide_thresholds", "assay"),
    ("perturbation", "load_labels", "perturbation_labels", "assay"),
    ("perturbation", "load_efficiency", "perturbation_efficiency", "assay"),
    ("perturbation", "load_edistance", "perturbation_edistance", "assay"),
    ("perturbation", "load_signature", "perturbation_signature", "assay"),
    ("perturbation", "load_mixscape", "mixscape_classes", "assay"),
    ("perturbation", "load_associations", "perturbation_associations", "assay"),
    ("perturbation", "load_profiles", "perturbation_profiles", "assay"),
    ("perturbation", "load_clusters", "perturbation_clusters", "assay"),
    ("artifacts", "load", "clones", "datastore"),
]


def _documented(method) -> set[str]:
    """The error classes that a method's ``Raises:`` section lists."""
    doc = inspect.getdoc(inspect.unwrap(method)) or ""
    found = re.search(r"^Raises:\n(.*?)(?=^\S|\Z)", doc, re.S | re.M)
    assert found, f"{method.__qualname__} documents no errors"
    return set(re.findall(r"^\s+(\w+Error):", found.group(1), re.M))


@pytest.fixture(scope="module")
def loader_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("loaders") / "store.zarr"
    write_count_store(
        str(path), {"RNA": np.arange(24, dtype=np.uint16).reshape(6, 4)}, "uint16"
    )
    store = open_store(path)
    store.cells.insert("every", np.ones(6, dtype=bool))
    return store, store.snapshot_cell_selection("every")


def _refs(store, kind: str, scope: str, salt: int) -> tuple[ArtifactRef, ArtifactRef]:
    """A ref that the store does not hold, and one whose write stopped."""
    assay = "RNA" if scope == "assay" else None
    missing = ArtifactRef(
        scope=scope, kind=kind, artifact_id=f"{2 * salt:064x}", assay=assay
    )
    incomplete = ArtifactRef(
        scope=scope, kind=kind, artifact_id=f"{2 * salt + 1:064x}", assay=assay
    )
    # An interrupted write leaves the group, marked incomplete.
    group = store.zw.require_group(artifact_path(incomplete))
    group.attrs.update(
        {"artifact_id": incomplete.artifact_id, "kind": kind, "complete": False}
    )
    return missing, incomplete


@pytest.mark.parametrize(
    ("namespace", "name", "kind", "scope"),
    LOADERS,
    ids=[f"{namespace}.{name}" for namespace, name, _kind, _scope in LOADERS],
)
def test_each_loader_raises_the_documented_class_for_each_artifact_it_refuses(
    loader_store, namespace, name, kind, scope
) -> None:
    store, selection = loader_store
    method = getattr(getattr(store, namespace), name)
    argument = next(iter(inspect.signature(method).parameters))
    missing, incomplete = _refs(
        store, kind, scope, LOADERS.index((namespace, name, kind, scope)) + 1
    )
    raised = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyError, match="Artifact does not exist") as error:
            method(missing)
        raised["missing"] = error.type
        with pytest.raises(RuntimeError, match="^Artifact is incomplete") as error:
            method(incomplete)
        raised["incomplete"] = error.type
        with pytest.raises(TypeError, match=f"^{argument} must be an ArtifactRef$"):
            method(kind)
        raised["text"] = TypeError
        if namespace != "artifacts":
            # A selection of cells is no saved result of any of these kinds.
            with pytest.raises(ValueError, match=f"^{argument} must be ") as error:
                method(selection)
            raised["other kind"] = error.type
    documented = _documented(method)
    assert {cls.__name__ for cls in raised.values()} <= documented, documented


def test_load_values_raises_value_errors_for_missing_and_incomplete_artifacts(
    loader_store,
) -> None:
    store, selection = loader_store
    method = store.artifacts.load_values
    missing, incomplete = _refs(store, "cluster_labels", "assay", len(LOADERS) + 1)
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        method(missing)
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        method(incomplete)
    with pytest.raises(ValueError, match="'cell_selection' is not a cell-aligned"):
        method(selection)
    with pytest.raises(TypeError, match="^ref must be an ArtifactRef or a CellField$"):
        method("cluster_labels")
    with pytest.raises(TypeError, match="^cell_selection must be an ArtifactRef$"):
        method(missing, cell_selection="every")
    assert {"TypeError", "ValueError"} <= _documented(method)


def test_inspect_reports_a_missing_artifact_and_refuses_other_arguments(
    loader_store,
) -> None:
    store, _selection = loader_store
    missing, incomplete = _refs(store, "clones", "datastore", len(LOADERS) + 2)
    status = store.artifacts.inspect(missing)
    assert (status.exists, status.complete) == (False, False)
    status = store.artifacts.inspect(incomplete)
    assert (status.exists, status.complete) == (True, False)
    with pytest.raises(TypeError, match="^ref must be an ArtifactRef$"):
        store.artifacts.inspect("clones")
    assert "TypeError" in _documented(store.artifacts.inspect)


@pytest.mark.parametrize("plot", ["variant_statistics", "allele_frequencies"])
def test_the_variant_plots_name_their_own_argument(loader_store, plot) -> None:
    import cytearc.plotting as splt

    store, selection = loader_store
    draw = getattr(splt, plot)
    # The plots pass variants on to load_informative, whose argument is named
    # informative, so they check theirs first.
    with pytest.raises(TypeError, match="^variants must be an ArtifactRef$"):
        draw(store, "variant_selection", show=False)
    with pytest.raises(
        ValueError, match="^variants must be a variant_selection artifact from"
    ):
        draw(store, selection, show=False)
