"""Soup profiles, contamination estimates, and corrected assays on simulated droplets.

Cells are drawn from three known cluster profiles and contaminated with a
known fraction of a known soup, and empty droplets hold soup alone. SoupX is
GPL-2 licensed and R is not installed here, so the expectations come from the
simulation and from independent NumPy, SciPy, and statsmodels references of
the equations in Young and Behjati (2020), GigaScience 9:giaa151.
"""

import inspect
import math
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from scipy.stats import poisson
from statsmodels.stats.multitest import multipletests

import cytearc.datastore._operations.ambient as ambient_operations
from cytearc import DataStore, SparseToZarr
from cytearc.assay.classification import declared_assay_type
from cytearc.datastore.namespaces import QcAccessor
from cytearc.metadata.artifacts import plan_cell_data_artifact
from cytearc.metadata.membership import UnmeasuredCellsError, membership_attributes
from cytearc.quality_control.ambient import (
    AmbientContamination,
    remove_ambient_counts,
)
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.identity import read_dataset_fingerprint
from cytearc.storage.schema import create_cell_data, derived_assay_transaction
from cytearc.utils.logging import logger
from tests.storage_helpers import write_count_store

N_GENES = 120
CELLS_PER_CLUSTER = 120
N_CELLS = 3 * CELLS_PER_CLUSTER
N_EMPTY = 2000
N_LARGE = 20
N_ZERO = 3
RHO = 0.1
HAEMOGLOBIN = ["HBB", "HBA1", "HBA2", "HBD"]
IMMUNOGLOBULIN = ["IGKC", "IGHM", "IGLC2", "JCHAIN"]
T_CELL = ["CD3E", "CD3D", "IL7R", "LTB"]
NAMES = [
    *HAEMOGLOBIN,
    *IMMUNOGLOBULIN,
    *T_CELL,
    *(f"GENE{index}" for index in range(N_GENES - 14)),
    "CELLONLY",
    "ABSENT",
]
MARKERS = (np.arange(0, 4), np.arange(4, 8), np.arange(8, 12))
# Cells of the third cluster express this gene, but the soup holds none of it.
CELL_ONLY = N_GENES - 2
GENE_SETS = {"haemoglobin": HAEMOGLOBIN, "immunoglobulin": IMMUNOGLOBULIN}
CLUSTER_NAMES = np.array(["k0", "k1", "k2"])


@dataclass(frozen=True)
class Droplets:
    """Simulated counts of cells, empty droplets, and their known sources."""

    counts: np.ndarray
    endogenous: np.ndarray
    labels: np.ndarray
    soup: np.ndarray

    @property
    def cells(self) -> np.ndarray:
        return self.counts[:N_CELLS]

    @property
    def empty(self) -> np.ndarray:
        return self.counts[N_CELLS : N_CELLS + N_EMPTY]


@cache
def simulate() -> Droplets:
    """Draw contaminated cells, empty droplets, larger droplets, and zero rows.

    Each cluster expresses four marker genes with 40% of its counts and shares
    the other genes, except one that no droplet holds. The third cluster also
    expresses a gene that the soup lacks. The soup is the mean cluster profile
    without that gene, every cell holds exactly ``round(RHO * N)`` soup counts,
    empty droplets hold 2 to 10 soup counts, and larger droplets 30 to 60.
    """
    rng = np.random.default_rng(20201211)
    background = rng.gamma(1.0, 1.0, N_GENES)
    background[:12] = 0
    background[CELL_ONLY:] = 0
    profiles = []
    for cluster, markers in enumerate(MARKERS):
        profile = background.copy()
        profile[markers] = background.sum() * 0.4 / len(markers)
        if cluster == 2:
            profile[CELL_ONLY] = background.sum() * 0.05
        profiles.append(profile / profile.sum())
    profiles_array = np.asarray(profiles)
    soup = profiles_array.mean(axis=0)
    soup[CELL_ONLY] = 0
    soup /= soup.sum()
    labels = np.repeat(np.arange(3), CELLS_PER_CLUSTER)
    totals = rng.integers(1500, 4000, size=N_CELLS)
    endogenous = np.array(
        [
            rng.multinomial(total - round(RHO * total), profiles_array[label])
            for total, label in zip(totals, labels, strict=True)
        ]
    )
    ambient = np.array([rng.multinomial(round(RHO * total), soup) for total in totals])
    empty = np.array(
        [rng.multinomial(total, soup) for total in rng.integers(2, 11, N_EMPTY)]
    )
    large = np.array(
        [rng.multinomial(total, soup) for total in rng.integers(30, 61, N_LARGE)]
    )
    counts = np.vstack(
        [
            endogenous + ambient,
            empty,
            large,
            np.zeros((N_ZERO, N_GENES), dtype=np.int64),
        ]
    ).astype(np.uint32)
    return Droplets(counts=counts, endogenous=endogenous, labels=labels, soup=soup)


def _open(path: Path, **options) -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        **options,
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    droplets = simulate()
    path = tmp_path_factory.mktemp("ambient") / "droplets.zarr"
    SparseToZarr(
        csr_matrix(droplets.counts),
        str(path),
        cell_ids=[f"d{index}" for index in range(len(droplets.counts))],
        feature_ids=[f"g{index}" for index in range(N_GENES)],
        feature_names=NAMES,
        assay_name="RNA",
        nthreads=1,
    ).dump()
    store = _open(path)
    rows = np.arange(store.cells.N)
    store.cells.insert("is_cell", rows < N_CELLS)
    store.cells.insert("is_empty", (rows >= N_CELLS) & (rows < N_CELLS + N_EMPTY))
    store.cells.insert(
        "cluster",
        np.concatenate(
            [CLUSTER_NAMES[droplets.labels], np.full(len(rows) - N_CELLS, "empty")]
        ),
    )
    store.cells.insert("nothing", np.zeros(len(rows), dtype=bool))
    # Each copy reuses these inputs instead of computing them again.
    _cells_and_clusters(store)
    store.qc.ambient_profile(total_counts=(2, 10))
    return path


@pytest.fixture
def path(template, tmp_path) -> Path:
    target = tmp_path / "droplets.zarr"
    shutil.copytree(template, target)
    return target


@pytest.fixture
def store(path) -> DataStore:
    return _open(path)


def _cells_and_clusters(store: DataStore) -> tuple[ArtifactRef, ArtifactRef]:
    cells = store.snapshot_cell_selection("is_cell")
    return cells, store.snapshot_cluster_labels("cluster", cell_selection=cells)


def _stored_counts(store: DataStore, assay: str) -> np.ndarray:
    return np.vstack(list(store.get_assay(assay).rawData.stream_blocks(nthreads=1)))


def _reference_evidence(
    soup: np.ndarray,
    gene_sets: dict[str, list[str]],
    *,
    clusters: np.ndarray | None,
    max_contamination: float = 1.0,
    fdr: float = 0.05,
) -> dict[str, np.ndarray]:
    """Equations S2 to S4 for the simulated cells, written independently."""
    cells = simulate().cells.astype(np.float64)
    totals = cells.sum(axis=1)
    columns = [
        np.flatnonzero(np.isin(np.char.upper(NAMES), np.char.upper(gene_sets[name])))
        for name in sorted(gene_sets)
    ]
    observed = np.column_stack([cells[:, genes].sum(axis=1) for genes in columns])
    expected = np.column_stack([totals * soup[genes].sum() for genes in columns])
    p_values = poisson.sf(observed - 1, max_contamination * expected)
    expressing = np.column_stack(
        [
            multipletests(p_values[:, s], method="fdr_bh")[1] < fdr
            for s in range(len(columns))
        ]
    )
    used = ~expressing
    if clusters is not None:
        for s in range(len(columns)):
            used[:, s] = ~np.isin(clusters, np.unique(clusters[expressing[:, s]]))
    return {
        "observed": observed,
        "expected": expected,
        "p_values": p_values,
        "expressing": expressing,
        "used": used,
    }


def _ratio(
    evidence: dict[str, np.ndarray], rows: np.ndarray | slice = slice(None)
) -> float:
    used = evidence["used"][rows]
    return float(
        evidence["observed"][rows][used].sum() / evidence["expected"][rows][used].sum()
    )


def test_soup_profile_from_a_count_range_sums_exactly_those_droplets(store):
    droplets = simulate()

    profile = store.qc.ambient_profile(total_counts=(2, 10))
    frame = store.qc.load_ambient_profile(profile)

    assert profile.kind == "ambient_profile" and profile.assay == "RNA"
    assert frame.index.tolist() == [f"g{index}" for index in range(N_GENES)]
    assert frame.index.name == "ids"
    assert frame["name"].tolist() == NAMES
    counts = droplets.empty.sum(axis=0)
    np.testing.assert_array_equal(frame["counts"].to_numpy(), counts)
    assert frame["counts"].dtype == np.uint64
    # Equation S1: each gene's share of the empty droplets' counts.
    np.testing.assert_array_equal(frame["fraction"].to_numpy(), counts / counts.sum())
    assert frame.attrs == {"n_droplets": N_EMPTY, "n_counts": int(counts.sum())}
    assert frame.loc["g118", "fraction"] == frame.loc["g119", "fraction"] == 0.0
    # The estimate is close to the soup that the droplets were drawn from.
    assert np.abs(frame["fraction"].to_numpy() - droplets.soup).max() < 0.01
    status = store.artifacts.inspect(profile)
    assert status.operation == "estimate_ambient_profile"
    assert status.parameters == {"total_counts": [2.0, 10.0]}
    assert set(status.inputs) == {"dataset_fingerprint"}


def test_soup_profile_from_a_droplet_selection_matches_the_count_range(store):
    droplets = store.snapshot_cell_selection("is_empty")

    by_selection = store.qc.ambient_profile(droplets)
    by_range = store.qc.ambient_profile(total_counts=[2, 10])

    assert by_selection != by_range
    assert store.artifacts.inspect(by_selection).inputs["empty_droplets"] == (
        droplets.to_dict()
    )
    assert store.artifacts.inspect(by_selection).parameters == {}
    selected = store.qc.load_ambient_profile(by_selection)
    ranged = store.qc.load_ambient_profile(by_range)
    np.testing.assert_array_equal(selected.to_numpy(), ranged.to_numpy())
    assert selected.attrs == ranged.attrs
    # Integer and float bounds give the same identity.
    assert store.qc.ambient_profile(total_counts=(2.0, 10.0)) == by_range


def test_soup_profile_is_reused_recomputed_and_refused_read_only(store, path):
    first = store.qc.ambient_profile(total_counts=(2, 10))
    assert store.qc.ambient_profile(total_counts=(2, 10)) == first
    fresh = store.qc.ambient_profile(total_counts=(2, 10), invalidate_cache=True)
    assert fresh != first
    np.testing.assert_array_equal(
        store.qc.load_ambient_profile(fresh).to_numpy(),
        store.qc.load_ambient_profile(first).to_numpy(),
    )

    read_only = _open(path, zarr_mode="r")
    assert read_only.qc.ambient_profile(total_counts=(2, 10)) in {first, fresh}
    with pytest.raises(PermissionError, match="estimate_ambient_profile requires"):
        read_only.qc.ambient_profile(total_counts=(3, 9))


@pytest.mark.parametrize(
    ("arguments", "keywords", "error", "message"),
    [
        ((), {}, ValueError, "exactly one of empty_droplets and total_counts"),
        (("droplets",), {"total_counts": (2, 10)}, ValueError, "exactly one"),
        (("is_empty",), {}, TypeError, "empty_droplets must be an ArtifactRef"),
        ((), {"total_counts": "2-10"}, TypeError, r"\(low, high\) pair"),
        ((), {"total_counts": 10}, TypeError, r"\(low, high\) pair"),
        ((), {"total_counts": (2,)}, ValueError, r"\(low, high\) pair"),
        ((), {"total_counts": (2, 10, 20)}, ValueError, r"\(low, high\) pair"),
        ((), {"total_counts": (True, 10)}, TypeError, "real number"),
        ((), {"total_counts": (2, np.nan)}, ValueError, "finite"),
        ((), {"total_counts": (-1, 10)}, ValueError, "0 <= low <= high"),
        ((), {"total_counts": (10, 2)}, ValueError, "0 <= low <= high"),
        ((), {"total_counts": (10**6, 10**7)}, ValueError, "No droplet of assay 'RNA'"),
        ((), {"total_counts": (0, 0)}, ValueError, "hold no counts"),
        (("nothing",), {}, ValueError, "selects no droplet"),
    ],
)
def test_soup_profile_validates_its_arguments(
    store, arguments, keywords, error, message
):
    resolved = [
        store.snapshot_cell_selection("is_empty" if value == "droplets" else value)
        if value in {"droplets", "nothing"}
        else value
        for value in arguments
    ]
    before = store.artifacts.list(kind="ambient_profile", from_assay="RNA")

    with pytest.raises(error, match=message):
        store.qc.ambient_profile(*resolved, **keywords)
    assert store.artifacts.list(kind="ambient_profile", from_assay="RNA") == before


def test_ambient_operations_need_an_rna_assay_of_raw_counts(tmp_path):
    counts = np.arange(24).reshape(6, 4) % 5
    write_count_store(
        str(tmp_path / "two.zarr"), {"RNA": counts, "ADT": counts}, "uint8"
    )
    two = DataStore(
        str(tmp_path / "two.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    with pytest.raises(
        TypeError, match="needs an RNA assay; assay 'ADT' is a ADTassay"
    ):
        two.qc.ambient_profile(total_counts=(0, 5), from_assay="ADT")

    write_count_store(str(tmp_path / "float.zarr"), {"RNA": counts + 0.5}, "float32")
    floats = DataStore(
        str(tmp_path / "float.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    with pytest.raises(TypeError, match="stores float32 values"):
        floats.qc.ambient_profile(total_counts=(0, 5))


def test_loading_a_soup_profile_checks_the_reference(store):
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    with pytest.raises(TypeError, match="profile must be an ArtifactRef"):
        store.qc.load_ambient_profile("profile")
    with pytest.raises(
        ValueError, match="ambient_profile artifact, not 'cell_selection'"
    ):
        store.qc.load_ambient_profile(store.snapshot_cell_selection("is_cell"))
    artifact_group(store.zw, profile).attrs["ordered_feature_ids_fingerprint"] = "other"
    with pytest.raises(ValueError, match="does not match the features of assay 'RNA'"):
        store.qc.load_ambient_profile(profile)


def test_contamination_recovers_the_simulated_fraction(store):
    droplets = simulate()
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()

    ref = store.qc.ambient_contamination(
        cells, profile, gene_sets=GENE_SETS, clusters=clusters
    )
    result = store.qc.load_ambient_contamination(ref)

    assert isinstance(result, AmbientContamination)
    # The per-cell tables stay out of the summary that the result prints.
    assert repr(result).startswith("AmbientContamination(estimate=")
    assert "evidence" not in repr(result)
    assert (result.method, result.level) == ("gene_sets", "global")
    assert abs(result.estimate - RHO) < 0.01
    assert set(result.set_estimates) == {"haemoglobin", "immunoglobulin"}
    assert all(abs(value - RHO) < 0.015 for value in result.set_estimates.values())
    assert result.cluster_estimates == {}
    reference = _reference_evidence(
        soup, GENE_SETS, clusters=CLUSTER_NAMES[droplets.labels]
    )
    assert result.estimate == pytest.approx(_ratio(reference), rel=1e-12)
    for position, name in enumerate(sorted(GENE_SETS)):
        used = reference["used"][:, position]
        assert result.set_estimates[name] == pytest.approx(
            reference["observed"][used, position].sum()
            / reference["expected"][used, position].sum(),
            rel=1e-12,
        )
    evidence = result.evidence
    assert evidence is not None
    assert evidence.columns.tolist() == [
        "ids",
        "gene_set",
        "observed",
        "expected",
        "p_value",
        "expressing",
        "used",
    ]
    for position, name in enumerate(sorted(GENE_SETS)):
        rows = evidence[evidence["gene_set"] == name]
        assert rows["ids"].tolist() == [f"d{index}" for index in range(N_CELLS)]
        np.testing.assert_array_equal(
            rows["observed"], reference["observed"][:, position]
        )
        np.testing.assert_allclose(
            rows["expected"], reference["expected"][:, position], rtol=1e-12
        )
        np.testing.assert_allclose(
            rows["p_value"], reference["p_values"][:, position], rtol=1e-9, atol=1e-300
        )
        np.testing.assert_array_equal(
            rows["expressing"], reference["expressing"][:, position]
        )
        np.testing.assert_array_equal(rows["used"], reference["used"][:, position])
    # Every cell of the expressing cluster expresses its set, and no other cell
    # does; a cluster with an expressing cell is left out of that set.
    for position, cluster in enumerate((0, 1)):
        rows = evidence[evidence["gene_set"] == sorted(GENE_SETS)[position]]
        np.testing.assert_array_equal(rows["expressing"], droplets.labels == cluster)
        np.testing.assert_array_equal(rows["used"], droplets.labels != cluster)
    np.testing.assert_array_equal(
        result.fractions.to_numpy(), np.full(N_CELLS, result.estimate)
    )
    assert result.fractions.name == "fraction"
    assert result.fractions.index.tolist() == [f"d{index}" for index in range(N_CELLS)]

    # The fractions are cell-aligned values that selections and loaders read.
    used = store.artifacts.load_values(ref, value="used")
    assert used.categorical and used.values.shape == (N_CELLS, 2)
    selected = store.select_cells(ref, high=1.0)
    assert np.asarray(store.artifacts.load(selected)["values"][:]).sum() == N_CELLS
    status = store.artifacts.inspect(ref)
    assert status.parameters == {
        "method": "gene_sets",
        "gene_sets": {"haemoglobin": [0, 1, 2, 3], "immunoglobulin": [4, 5, 6, 7]},
        "level": "global",
        "max_contamination": 1.0,
        "fdr": 0.05,
    }
    assert status.inputs["profile"] == profile.to_dict()
    assert status.inputs["clusters"] == clusters.to_dict()
    assert (
        store.qc.ambient_contamination(
            cells, profile, gene_sets=GENE_SETS, clusters=clusters
        )
        == ref
    )


def test_contamination_per_cluster_uses_each_clusters_cells(store):
    droplets = simulate()
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()

    result = store.qc.load_ambient_contamination(
        store.qc.ambient_contamination(
            cells, profile, gene_sets=GENE_SETS, clusters=clusters, level="cluster"
        )
    )

    assert result.level == "cluster"
    assert list(result.cluster_estimates) == ["k0", "k1", "k2"]
    reference = _reference_evidence(
        soup, GENE_SETS, clusters=CLUSTER_NAMES[droplets.labels]
    )
    for label, name in enumerate(CLUSTER_NAMES):
        members = droplets.labels == label
        assert result.cluster_estimates[name] == pytest.approx(
            _ratio(reference, members), rel=1e-12
        )
        assert abs(result.cluster_estimates[name] - RHO) < 0.015
        np.testing.assert_array_equal(
            result.fractions.to_numpy()[members], result.cluster_estimates[name]
        )
    assert result.estimate == pytest.approx(_ratio(reference), rel=1e-12)


def test_contamination_without_clusters_leaves_out_only_expressing_cells(store):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()

    result = store.qc.load_ambient_contamination(
        store.qc.ambient_contamination(cells, profile, gene_sets=GENE_SETS)
    )

    reference = _reference_evidence(soup, GENE_SETS, clusters=None)
    assert result.evidence is not None
    np.testing.assert_array_equal(
        result.evidence["used"], ~result.evidence["expressing"]
    )
    assert result.estimate == pytest.approx(_ratio(reference), rel=1e-12)
    assert abs(result.estimate - RHO) < 0.01


def test_a_supplied_fraction_applies_to_every_cell(store):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))

    ref = store.qc.ambient_contamination(cells, profile, fraction=0.2)
    result = store.qc.load_ambient_contamination(ref)

    assert (result.method, result.level, result.estimate) == ("fixed", "global", 0.2)
    np.testing.assert_array_equal(result.fractions.to_numpy(), np.full(N_CELLS, 0.2))
    assert result.evidence is None
    assert result.set_estimates == {} and result.cluster_estimates == {}
    assert store.artifacts.inspect(ref).parameters == {
        "method": "fixed",
        "fraction": 0.2,
    }
    # Integer and float fractions give one identity, and both bounds are allowed.
    zero = store.qc.ambient_contamination(cells, profile, fraction=0)
    assert store.qc.ambient_contamination(cells, profile, fraction=0.0) == zero
    assert (
        store.qc.load_ambient_contamination(
            store.qc.ambient_contamination(cells, profile, fraction=1)
        ).estimate
        == 1.0
    )


def _masked_clusters(store: DataStore, cells: ArtifactRef) -> ArtifactRef:
    """Store cluster labels whose linked mask flags the first cell as missing."""
    labels = np.repeat(np.arange(3), CELLS_PER_CLUSTER).astype(np.int64)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[0] = True
    planned = plan_cell_data_artifact(
        store.zw,
        scope="datastore",
        kind="cluster_labels",
        operation="masked_test_clusters",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=cells,
        arrays={"values": ((N_CELLS,), "i")},
    )
    mask_name = f"{MISSING_MASK_PREFIX}values"
    with artifact_transaction(store.zw, planned) as group:
        group.create_array("values", data=labels)
        group.create_array(mask_name, data=missing)
        group["values"].attrs["missing_mask"] = mask_name
    return planned.ref


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("cells_not_ref", TypeError, "cell_selection must be an ArtifactRef"),
        ("bad_level", ValueError, "level must be 'global' or 'cluster'"),
        ("clusters_not_ref", TypeError, "clusters must be an ArtifactRef or None"),
        ("both", ValueError, "exactly one of gene_sets and fraction"),
        ("neither", ValueError, "exactly one of gene_sets and fraction"),
        (
            "zero_max",
            ValueError,
            "max_contamination must be greater than 0 and at most 1",
        ),
        (
            "large_max",
            ValueError,
            "max_contamination must be greater than 0 and at most 1",
        ),
        ("bool_max", TypeError, "max_contamination must be a real number"),
        ("zero_fdr", ValueError, "fdr must be greater than 0 and less than 1"),
        ("one_fdr", ValueError, "fdr must be greater than 0 and less than 1"),
        ("profile_not_ref", TypeError, "profile must be an ArtifactRef"),
        ("profile_kind", ValueError, "ambient_profile artifact, not 'cell_selection'"),
        (
            "fraction_with_clusters",
            ValueError,
            "apply only to estimates from gene_sets",
        ),
        ("fraction_with_level", ValueError, "apply only to estimates from gene_sets"),
        ("fraction_with_max", ValueError, "apply only to estimates from gene_sets"),
        ("fraction_with_fdr", ValueError, "apply only to estimates from gene_sets"),
        ("large_fraction", ValueError, "fraction must lie between 0 and 1"),
        ("text_fraction", TypeError, "fraction must be a real number"),
        ("cluster_level_alone", ValueError, "level='cluster' needs clusters"),
        ("sets_not_mapping", TypeError, "gene_sets must map gene-set names"),
        ("no_sets", ValueError, "gene_sets must name at least one gene set"),
        ("set_name", TypeError, "keyed by non-empty gene-set names"),
        ("text_set", TypeError, "must be a collection of gene names"),
        ("set_values", TypeError, "must hold gene names"),
        ("unmatched_set", ValueError, "No gene of gene set 'mouse' matches"),
        (
            "shared_genes",
            ValueError,
            "'immunoglobulin' shares genes with gene set 'haemoglobin'",
        ),
        ("absent_soup", ValueError, "'absent' hold no counts in the soup profile"),
        ("no_cells", ValueError, "cell_selection must select at least one cell"),
        ("uncovered", ValueError, "contain|within|subset|selection"),
        ("not_categorical", ValueError, "categorical cell labels"),
        ("masked", ValueError, "clusters has no label for some selected cells"),
        ("cluster_without_evidence", ValueError, "clusters have no cell .*: k0"),
        ("everywhere", ValueError, "No cell can measure the contamination"),
    ],
)
def test_contamination_validates_its_arguments(store, case, error, message):
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    arguments: list = [cells, profile]
    keywords: dict = {"gene_sets": GENE_SETS}
    if case == "cells_not_ref":
        arguments[0] = "is_cell"
    elif case == "bad_level":
        keywords["level"] = "cell"
    elif case == "clusters_not_ref":
        keywords["clusters"] = "cluster"
    elif case == "both":
        keywords["fraction"] = 0.1
    elif case == "neither":
        keywords = {}
    elif case in {"zero_max", "large_max", "bool_max"}:
        keywords["max_contamination"] = {
            "zero_max": 0,
            "large_max": 1.5,
            "bool_max": True,
        }[case]
    elif case in {"zero_fdr", "one_fdr"}:
        keywords["fdr"] = 0 if case == "zero_fdr" else 1
    elif case == "profile_not_ref":
        arguments[1] = "profile"
    elif case == "profile_kind":
        arguments[1] = cells
    elif case.startswith("fraction_with_"):
        option = {
            "fraction_with_clusters": {"clusters": clusters},
            "fraction_with_level": {"level": "cluster"},
            "fraction_with_max": {"max_contamination": 0.5},
            "fraction_with_fdr": {"fdr": 0.01},
        }[case]
        keywords = {"fraction": 0.1, **option}
    elif case in {"large_fraction", "text_fraction"}:
        keywords = {"fraction": 1.5 if case == "large_fraction" else "0.1"}
    elif case == "cluster_level_alone":
        keywords["level"] = "cluster"
    elif case == "sets_not_mapping":
        keywords["gene_sets"] = [HAEMOGLOBIN]
    elif case == "no_sets":
        keywords["gene_sets"] = {}
    elif case == "set_name":
        keywords["gene_sets"] = {"": HAEMOGLOBIN}
    elif case == "text_set":
        keywords["gene_sets"] = {"haemoglobin": "HBB"}
    elif case == "set_values":
        keywords["gene_sets"] = {"haemoglobin": ["HBB", 3]}
    elif case == "unmatched_set":
        keywords["gene_sets"] = {"mouse": ["Hbb-bs", "Hba-a1"]}
    elif case == "shared_genes":
        keywords["gene_sets"] = {**GENE_SETS, "immunoglobulin": ["IGKC", "hbb"]}
    elif case == "absent_soup":
        keywords["gene_sets"] = {**GENE_SETS, "absent": ["ABSENT"]}
    elif case == "no_cells":
        arguments[0] = store.snapshot_cell_selection("nothing")
    elif case == "uncovered":
        first = store.select_cells(clusters, include=["k0"])
        keywords["clusters"] = store.snapshot_cluster_labels(
            "cluster", cell_selection=first
        )
    elif case == "not_categorical":
        keywords["clusters"] = store.qc.ambient_contamination(
            cells, profile, fraction=0.1
        )
    elif case == "masked":
        keywords["clusters"] = _masked_clusters(store, cells)
    elif case == "cluster_without_evidence":
        keywords.update(
            gene_sets={"haemoglobin": HAEMOGLOBIN}, clusters=clusters, level="cluster"
        )
    elif case == "everywhere":
        # Every cell of the second cluster expresses immunoglobulin genes.
        arguments[0] = store.select_cells(clusters, include=["k1"])
        keywords["gene_sets"] = {"immunoglobulin": IMMUNOGLOBULIN}
    before = store.artifacts.list(kind="ambient_contamination", from_assay="RNA")

    with pytest.raises(error, match=message):
        store.qc.ambient_contamination(*arguments, **keywords)
    assert (
        store.artifacts.list(kind="ambient_contamination", from_assay="RNA") == before
    )


def test_contamination_logs_unmatched_genes_and_estimates_above_the_maximum(store):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        matched = store.qc.ambient_contamination(
            cells, profile, gene_sets={"haemoglobin": [*HAEMOGLOBIN, "hbm"]}
        )
        # Below the true fraction, the test leaves only cells with few counts.
        store.qc.ambient_contamination(
            cells, profile, gene_sets=GENE_SETS, max_contamination=0.05
        )
    finally:
        logger.remove(sink)

    assert messages[0] == (
        "1 of 5 genes of gene set 'haemoglobin' are not assay features: hbm"
    )
    assert messages[1].startswith("An estimated contamination fraction, ")
    assert "exceeds max_contamination=0.05" in messages[1]
    assert store.artifacts.inspect(matched).parameters["gene_sets"] == {
        "haemoglobin": [0, 1, 2, 3]
    }


def test_contamination_warns_about_an_estimate_of_zero_and_keeps_it(store):
    droplets = simulate()
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()
    # GENE5 is the rarest gene of the soup; many cells hold none of it.
    rare = NAMES.index("GENE5")
    without = droplets.cells[:, rare] == 0
    store.cells.insert(
        "no_gene5", np.concatenate([without, np.zeros(store.cells.N - N_CELLS, bool)])
    )
    cells = store.snapshot_cell_selection("no_gene5")
    expected = float((droplets.cells[without].sum(axis=1) * soup[rare]).sum())
    assert expected > 40

    def message(subject: str, genes: str, consequence: str) -> str:
        return (
            f"{subject} is 0: no count of {genes} was observed in the "
            f"{int(without.sum())} cells measuring it, where pure soup with their "
            f"totals would give {expected:.3g}. A count of 0 puts the fraction "
            f"below {-math.log(0.05) / expected:.3g} (the one-sided 95% Poisson "
            f"bound, 3 / {expected:.3g}), not at 0, so {consequence}. Check that "
            "the gene sets and the soup profile suit these cells."
        )

    # With a second gene set the global estimate is positive, and the warning
    # names the set of the estimate of 0.
    with pytest.warns(UserWarning) as caught:
        both = store.qc.ambient_contamination(
            cells, profile, gene_sets={"rare": ["GENE5"], "haemoglobin": HAEMOGLOBIN}
        )
    assert [str(warning.message) for warning in caught] == [
        message(
            "The contamination estimate from gene set 'rare'",
            "the set",
            "a correction with an estimate from this gene set alone would "
            f"leave {int(without.sum())} cells uncorrected",
        )
    ]
    result = store.qc.load_ambient_contamination(both)
    assert result.set_estimates["rare"] == 0.0
    assert result.estimate > 0

    with pytest.warns(UserWarning) as caught:
        alone = store.qc.ambient_contamination(
            cells, profile, gene_sets={"rare": ["GENE5"]}
        )
    assert [str(warning.message) for warning in caught] == [
        message(
            "The global contamination estimate",
            "the measured gene sets",
            f"a correction with this estimate would leave {int(without.sum())} "
            "cells uncorrected",
        )
    ]
    # The estimate of 0 is saved as computed, and a correction with it leaves
    # the cells' counts as they were.
    result = store.qc.load_ambient_contamination(alone)
    assert result.estimate == 0.0 and result.set_estimates == {"rare": 0.0}
    np.testing.assert_array_equal(result.fractions.to_numpy(), 0.0)
    store.qc.correct_ambient(alone, assay_label="unchanged")
    np.testing.assert_array_equal(
        _stored_counts(store, "unchanged")[:N_CELLS][without],
        droplets.cells[without],
    )
    assert store.get_assay("unchanged").z.attrs["ambient_removed_counts"] == 0


def test_contamination_is_reused_from_a_read_only_store(store, path):
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    ref = store.qc.ambient_contamination(
        cells, profile, gene_sets=GENE_SETS, clusters=clusters
    )

    read_only = _open(path, zarr_mode="r")
    assert (
        read_only.qc.ambient_contamination(
            cells, profile, gene_sets=GENE_SETS, clusters=clusters
        )
        == ref
    )
    with pytest.raises(
        PermissionError, match="estimate_ambient_contamination requires"
    ):
        read_only.qc.ambient_contamination(cells, profile, gene_sets=GENE_SETS)
    with pytest.raises(
        PermissionError, match="estimate_ambient_contamination requires"
    ):
        read_only.qc.ambient_contamination(cells, profile, fraction=0.3)


def test_loading_a_contamination_estimate_checks_the_reference(store):
    with pytest.raises(TypeError, match="contamination must be an ArtifactRef"):
        store.qc.load_ambient_contamination("estimate")
    with pytest.raises(ValueError, match="ambient_contamination artifact, not"):
        store.qc.load_ambient_contamination(store.snapshot_cell_selection("is_cell"))


def _assert_cells_lose_their_share(
    counts: np.ndarray, corrected: np.ndarray, fractions: np.ndarray, soup: np.ndarray
) -> None:
    """Each cell loses its share, or every soup count, rounded down or up."""
    totals = counts.sum(axis=1).astype(np.float64)
    in_soup = counts[:, soup > 0].sum(axis=1).astype(np.float64)
    targets = np.minimum(fractions * totals, in_soup)
    lost = counts.sum(axis=1, dtype=np.int64) - corrected.sum(axis=1, dtype=np.int64)
    assert ((lost == np.floor(targets)) | (lost == np.ceil(targets))).all()


def test_corrected_counts_remove_the_estimated_soup(store):
    droplets = simulate()
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()
    estimate = store.qc.ambient_contamination(
        cells, profile, gene_sets=GENE_SETS, clusters=clusters
    )
    fraction = store.qc.load_ambient_contamination(estimate).estimate

    assert store.qc.correct_ambient(estimate, assay_label="RNA_soup") is None

    assert "RNA_soup" in store.assay_names
    corrected = _stored_counts(store, "RNA_soup")
    raw = droplets.counts
    assert corrected.dtype == store.RNA.rawData.dtype
    assert (corrected <= raw).all()
    np.testing.assert_array_equal(corrected[N_CELLS:], 0)
    removed = raw[:N_CELLS].astype(np.int64) - corrected[:N_CELLS].astype(np.int64)
    _assert_cells_lose_their_share(
        raw[:N_CELLS], corrected[:N_CELLS], np.full(N_CELLS, fraction), soup
    )
    # The counts are those of the removal with the default seed, each cell's
    # row in the store keying its rounding draw.
    np.testing.assert_array_equal(
        corrected[:N_CELLS],
        remove_ambient_counts(
            raw[:N_CELLS],
            np.full(N_CELLS, fraction),
            soup,
            random_seed=0,
            row_keys=np.arange(N_CELLS),
        ),
    )
    np.testing.assert_array_equal(removed[:, soup == 0], 0)
    # The third cluster expresses a gene that the soup lacks, and keeps it all.
    assert raw[:N_CELLS][droplets.labels == 2, CELL_ONLY].min() > 0
    np.testing.assert_array_equal(
        corrected[:N_CELLS, CELL_ONLY], raw[:N_CELLS, CELL_ONLY]
    )
    attributes = store.get_assay("RNA_soup").z.attrs
    assert attributes["ambient_source_assay"] == "RNA"
    assert attributes["ambient_contamination"] == estimate.to_dict()
    assert attributes["ambient_profile"] == profile.to_dict()
    assert attributes["ambient_random_seed"] == 0
    assert attributes["ambient_corrected_cells"] == N_CELLS
    assert attributes["ambient_removed_counts"] == int(removed.sum())
    # Only the corrected cells are measured, and the assay is a prepared RNA
    # assay whose countsT holds the same counts.
    np.testing.assert_array_equal(
        store.cells.fetch_all("RNA_soup_I"), np.arange(store.cells.N) < N_CELLS
    )
    assay = store.get_assay("RNA_soup")
    assert declared_assay_type(assay) == "RNA"
    assert assay.rawDataT is not None
    np.testing.assert_array_equal(assay.rawDataT[:], corrected.T)
    np.testing.assert_array_equal(
        store.cells.fetch_all("RNA_soup_nCounts"), corrected.sum(axis=1)
    )
    assert assay.feats.fetch_all("names").tolist() == NAMES

    # The correction moves the counts towards the uncontaminated truth: their
    # squared error falls, and each cell correlates better with its truth.
    truth = droplets.endogenous.astype(np.float64)
    before = ((raw[:N_CELLS] - truth) ** 2).mean()
    after = ((corrected[:N_CELLS] - truth) ** 2).mean()
    assert after < 0.2 * before

    def correlations(values: np.ndarray) -> np.ndarray:
        centred = values - values.mean(axis=1, keepdims=True)
        reference = truth - truth.mean(axis=1, keepdims=True)
        return (centred * reference).sum(axis=1) / np.sqrt(
            (centred**2).sum(axis=1) * (reference**2).sum(axis=1)
        )

    raw_correlation = correlations(raw[:N_CELLS].astype(np.float64))
    corrected_correlation = correlations(corrected[:N_CELLS].astype(np.float64))
    assert (corrected_correlation > raw_correlation).mean() > 0.95
    assert corrected_correlation.mean() > raw_correlation.mean()
    for cluster, markers in enumerate(MARKERS):
        others = droplets.labels != cluster
        raw_elsewhere = raw[:N_CELLS][others][:, markers].sum()
        assert corrected[:N_CELLS][others][:, markers].sum() < 0.25 * raw_elsewhere
        own = raw[:N_CELLS][~others][:, markers].sum()
        assert corrected[:N_CELLS][~others][:, markers].sum() > 0.9 * own


def test_correction_is_deterministic_in_any_memory_budget(store, path, monkeypatch):
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    estimate = store.qc.ambient_contamination(
        cells, profile, gene_sets=GENE_SETS, clusters=clusters, level="cluster"
    )
    store.qc.correct_ambient(estimate, assay_label="first", random_seed=9)
    store.qc.correct_ambient(estimate, assay_label="second", random_seed=9)
    first = _stored_counts(store, "first")
    np.testing.assert_array_equal(_stored_counts(store, "second"), first)

    calls: list[tuple[int, np.ndarray, int]] = []
    removal = ambient_operations.remove_ambient_counts

    def recorded(counts, fractions, soup, *, random_seed, row_keys):
        calls.append((len(counts), np.asarray(row_keys), random_seed))
        return removal(
            counts, fractions, soup, random_seed=random_seed, row_keys=row_keys
        )

    monkeypatch.setattr(ambient_operations, "remove_ambient_counts", recorded)
    small = _open(path, mem_budget="2M")
    small.qc.correct_ambient(estimate, assay_label="small", random_seed=9)

    np.testing.assert_array_equal(_stored_counts(small, "small"), first)
    # Bands of fewer rows than the cells: the cells were read in several
    # blocks, and each removal call took a bounded number of rows, each keyed
    # by its row in the store.
    assert small.get_assay("small").rawData._backing.shards[0] < N_CELLS
    sizes = [size for size, _, _ in calls]
    assert len(calls) > 2
    assert sum(sizes) == N_CELLS
    assert max(sizes) < N_CELLS // 2
    np.testing.assert_array_equal(
        np.concatenate([keys for _, keys, _ in calls]), np.arange(N_CELLS)
    )
    assert {seed for _, _, seed in calls} == {9}


def test_correction_seed_sets_the_counts_and_the_assay_identity(store):
    droplets = simulate()
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()
    estimate = store.qc.ambient_contamination(
        cells, profile, gene_sets=GENE_SETS, clusters=clusters
    )
    fraction = store.qc.load_ambient_contamination(estimate).estimate

    store.qc.correct_ambient(estimate, assay_label="default")
    seeds = {"zero": 0, "one": 1, "large": 2**64 + 3}
    for label, seed in seeds.items():
        store.qc.correct_ambient(estimate, assay_label=label, random_seed=seed)

    counts = {label: _stored_counts(store, label) for label in ("default", *seeds)}
    identities = {
        label: read_dataset_fingerprint(store.get_assay(label).z) for label in counts
    }
    # The default seed is 0; a seed gives its counts and its identity, and
    # another seed others.
    np.testing.assert_array_equal(counts["default"], counts["zero"])
    assert identities["default"] == identities["zero"]
    assert (counts["one"] != counts["zero"]).any()
    assert len({identities[label] for label in ("zero", "one", "large")}) == 3
    for label, seed in {"default": 0, **seeds}.items():
        assert store.get_assay(label).z.attrs["ambient_random_seed"] == seed
        np.testing.assert_array_equal(
            counts[label][:N_CELLS],
            remove_ambient_counts(
                droplets.cells,
                np.full(N_CELLS, fraction),
                soup,
                random_seed=seed,
                row_keys=np.arange(N_CELLS),
            ),
        )
        _assert_cells_lose_their_share(
            droplets.cells, counts[label][:N_CELLS], np.full(N_CELLS, fraction), soup
        )


def test_correction_gives_a_cell_the_same_counts_in_any_selection(store):
    droplets = simulate()
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    second = store.select_cells(clusters, include=["k1"])

    store.qc.correct_ambient(
        store.qc.ambient_contamination(cells, profile, fraction=0.1),
        assay_label="every_cell",
        random_seed=4,
    )
    store.qc.correct_ambient(
        store.qc.ambient_contamination(second, profile, fraction=0.1),
        assay_label="second_cluster",
        random_seed=4,
    )

    # A cell's draw follows its row in the store, not its place in the
    # selection, so the second cluster's cells get the same counts.
    members = np.flatnonzero(droplets.labels == 1)
    assert members[0] > 0
    alone = _stored_counts(store, "second_cluster")
    np.testing.assert_array_equal(
        alone[members], _stored_counts(store, "every_cell")[members]
    )
    assert (alone[members] < droplets.cells[members]).any()
    np.testing.assert_array_equal(np.delete(alone, members, axis=0), 0)


def test_correction_needs_a_budget_for_one_row(store, path):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    estimate = store.qc.ambient_contamination(cells, profile, fraction=0.1)

    tiny = _open(path, mem_budget="256K")
    with pytest.raises(MemoryError, match="count shards of one row"):
        tiny.qc.correct_ambient(estimate, assay_label="tiny")
    assert "tiny" not in zarr.open_group(str(path), mode="r")


def test_correction_without_contamination_keeps_the_cells_counts(store):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    store.qc.correct_ambient(
        store.qc.ambient_contamination(cells, profile, fraction=0.0),
        assay_label="unchanged",
    )

    corrected = _stored_counts(store, "unchanged")
    np.testing.assert_array_equal(corrected[:N_CELLS], simulate().cells)
    np.testing.assert_array_equal(corrected[N_CELLS:], 0)
    assert store.get_assay("unchanged").z.attrs["ambient_removed_counts"] == 0


def test_correction_of_every_cell_measures_every_cell(store):
    droplets = simulate()
    everything = store.snapshot_cell_selection("I")
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()
    store.qc.correct_ambient(
        store.qc.ambient_contamination(everything, profile, fraction=0.5),
        assay_label="halved",
    )

    assert "halved_I" not in store.cells.columns
    corrected = _stored_counts(store, "halved")
    halves = np.full(len(corrected), 0.5)
    _assert_cells_lose_their_share(droplets.counts, corrected, halves, soup)
    np.testing.assert_array_equal(
        corrected,
        remove_ambient_counts(
            droplets.counts, halves, soup, row_keys=np.arange(len(corrected))
        ),
    )


def test_corrected_counts_feed_operations_by_assay_name(store):
    cells, clusters = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    store.qc.correct_ambient(
        store.qc.ambient_contamination(
            cells, profile, gene_sets=GENE_SETS, clusters=clusters
        ),
        assay_label="RNA_soup",
    )

    features = store.features.universe(from_assay="RNA_soup")
    normalized = store.features.normalize(cells, features)
    assert normalized.assay == "RNA_soup"
    with pytest.raises(UnmeasuredCellsError, match="RNA_soup"):
        store.features.normalize(store.snapshot_cell_selection("I"), features)
    # The corrected assay measured no empty droplet, so a count range leaves
    # out the droplets that it holds no counts of.
    droplets = store.snapshot_cell_selection("is_empty")
    with pytest.raises(UnmeasuredCellsError, match="estimate_ambient_profile"):
        store.qc.ambient_profile(droplets, from_assay="RNA_soup")
    with pytest.raises(ValueError, match="No droplet of assay 'RNA_soup'"):
        store.qc.ambient_profile(total_counts=(0, 0), from_assay="RNA_soup")


def test_correction_keeps_its_signature():
    assert str(inspect.signature(QcAccessor.correct_ambient)) == (
        "(self, contamination: cytearc.storage.refs.ArtifactRef, *, "
        "assay_label: str, random_seed: int = 0) -> None"
    )


def test_correction_validates_its_arguments(store, path):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    estimate = store.qc.ambient_contamination(cells, profile, fraction=0.1)

    with pytest.raises(TypeError, match="contamination must be an ArtifactRef"):
        store.qc.correct_ambient("estimate", assay_label="x")
    with pytest.raises(
        ValueError, match="ambient_contamination artifact, not 'ambient_profile'"
    ):
        store.qc.correct_ambient(profile, assay_label="x")
    with pytest.raises(TypeError, match="assay_label must be a string"):
        store.qc.correct_ambient(estimate, assay_label=1)
    with pytest.raises(ValueError, match="random_seed must be at least 0"):
        store.qc.correct_ambient(estimate, assay_label="x", random_seed=-1)
    for seed in (1.0, True, "0"):
        with pytest.raises(TypeError, match="random_seed must be an integer"):
            store.qc.correct_ambient(estimate, assay_label="x", random_seed=seed)
    with pytest.raises(
        ValueError, match="'RNA' already has metadata or a count matrix"
    ):
        store.qc.correct_ambient(estimate, assay_label="RNA")
    read_only = _open(path, zarr_mode="r")
    with pytest.raises(PermissionError, match="correct_ambient requires"):
        read_only.qc.correct_ambient(estimate, assay_label="x")
    assert store.assay_names == ["RNA"]


def test_failed_correction_leaves_no_assay_and_can_be_retried(store, path, monkeypatch):
    cells, _ = _cells_and_clusters(store)
    profile = store.qc.ambient_profile(total_counts=(2, 10))
    estimate = store.qc.ambient_contamination(cells, profile, fraction=0.1)

    def failing(counts, fractions, soup, **keywords):
        raise RuntimeError("removal failed")

    monkeypatch.setattr(ambient_operations, "remove_ambient_counts", failing)
    with pytest.raises(RuntimeError, match="removal failed"):
        store.qc.correct_ambient(estimate, assay_label="RNA_soup")
    monkeypatch.undo()

    root = zarr.open_group(str(path), mode="r")
    assert "RNA_soup" not in root
    assert "RNA_soup_I" not in root["cellData"]
    assert store.assay_names == ["RNA"]
    store.qc.correct_ambient(estimate, assay_label="RNA_soup")
    assert store.assay_names == ["RNA", "RNA_soup"]


def test_correction_in_a_workspace_publishes_its_matrices(tmp_path):
    droplets = simulate()
    path = tmp_path / "workspace.zarr"
    SparseToZarr(
        csr_matrix(droplets.counts),
        str(path),
        cell_ids=[f"d{index}" for index in range(len(droplets.counts))],
        feature_ids=[f"g{index}" for index in range(N_GENES)],
        feature_names=NAMES,
        assay_name="RNA",
        workspace="ws1",
        nthreads=1,
    ).dump()
    store = DataStore(
        str(path),
        workspace="ws1",
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )
    store.cells.insert("is_cell", np.arange(store.cells.N) < N_CELLS)
    cells = store.snapshot_cell_selection("is_cell")
    profile = store.qc.ambient_profile(total_counts=(2, 10))

    store.qc.correct_ambient(
        store.qc.ambient_contamination(cells, profile, fraction=0.1),
        assay_label="RNA_soup",
    )

    root = zarr.open_group(str(path), mode="r")
    assert root["ws1/RNA_soup"].attrs["is_assay"] is True
    assert root["ws1"].attrs["assayTypes"]["RNA_soup"] == "RNA"
    assert root["matrices/RNA_soup/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        root["ws1/cellData/RNA_soup_I"][:], np.arange(store.cells.N) < N_CELLS
    )
    corrected = _stored_counts(store, "RNA_soup")
    np.testing.assert_array_equal(corrected[N_CELLS:], 0)
    soup = store.qc.load_ambient_profile(profile)["fraction"].to_numpy()
    tenths = np.full(N_CELLS, 0.1)
    _assert_cells_lose_their_share(droplets.cells, corrected[:N_CELLS], tenths, soup)
    np.testing.assert_array_equal(
        corrected[:N_CELLS],
        remove_ambient_counts(
            droplets.cells, tenths, soup, row_keys=np.arange(N_CELLS)
        ),
    )
    reopened = DataStore(str(path), workspace="ws1", zarr_mode="r")
    assert "RNA_soup" in reopened.assay_names


def test_derived_assay_transaction_writes_membership_from_a_column_or_a_mask(
    tmp_path,
):
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    ids = np.array([f"c{index}" for index in range(5)])
    cells = create_cell_data(root, None, ids=ids, names=ids)
    mask = np.array([True, False, True, False, False])
    cells.create_array("flag", data=~mask)

    with derived_assay_transaction(
        root, "copied", None, operation="test", membership="flag"
    ) as transaction:
        counts = transaction.create_counts(5, ["f0"], ["f0"], np.uint8)
        counts.attrs["complete"] = True
    np.testing.assert_array_equal(root["cellData/copied_I"][:], ~mask)

    with pytest.raises(
        ValueError, match=r"needs one value per cell \(5\); got shape \(4,\)"
    ):
        with derived_assay_transaction(
            root, "derived", None, operation="test", membership=mask[:4]
        ) as transaction:
            transaction.create_counts(5, ["f0"], ["f0"], np.uint8)
    assert "derived" not in root

    with derived_assay_transaction(
        root, "derived", None, operation="test", membership=mask
    ) as transaction:
        counts = transaction.create_counts(5, ["f0"], ["f0"], np.uint8)
        counts[:] = np.arange(5, dtype=np.uint8)[:, None]
        counts.attrs["complete"] = True

    column = root["cellData/derived_I"]
    np.testing.assert_array_equal(column[:], mask)
    expected = membership_attributes("derived")
    assert {key: column.attrs[key] for key in expected} == expected
    assert root["derived"].attrs["is_assay"] is True
