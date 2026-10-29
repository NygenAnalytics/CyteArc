"""The perturbation recipe: a pooled CRISPR screen analyzed in one durable run.

The store simulates a low-MOI knockout screen: four targets with planted
effects on their own genes and on two programs of genes, four non-targeting
guides, cells without a guide and cells with guides of two targets, in two
lanes. Module fixtures run the recipe once with its defaults and once with
every stage; a test that writes copies the store, so that its runs reuse
theirs.
"""

import inspect
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc import (
    CancellationToken,
    DataStore,
    PipelineCancelled,
    PipelineExecutionError,
    PipelineRun,
)
from cytearc.datastore._pipeline import summaries
from cytearc.datastore._pipeline.executor import RecipePlan
from cytearc.datastore._pipeline.recipes.common import require_labeled_cells
from cytearc.datastore._pipeline.recipes.perturbation import resolve_perturbation
from cytearc.datastore._pipeline.settings import LABELS_NEED_CONTROLS, stage_settings
from cytearc.datastore.pipeline_accessor import PipelineAccessor
from cytearc.metadata.membership import UnmeasuredCellsError, membership_attributes
from tests.storage_helpers import insert_nullable_cell_column, write_count_store

N_GENES = 120
# Each guide's target and its target gene's feature ID: two guides of GENE_A
# and of GENE_B, one of GENE_C and of GENE_D, and four non-targeting guides.
GUIDES = [
    ("GENE_A", "RNA0"),
    ("GENE_A", "RNA0"),
    ("GENE_B", "RNA1"),
    ("GENE_B", "RNA1"),
    ("GENE_C", "RNA2"),
    ("GENE_D", "RNA3"),
    ("NT", ""),
    ("NT", ""),
    ("NT", ""),
    ("NT", ""),
]
LABELS = {"controls": ["NT"]}
# The head of a quick run: 50 HVGs of the 120 genes and 5 PCA dimensions.
HEAD: dict[str, Any] = {"hvg": {"top_n": 50}, "pca": {"dims": 5}}
# A quick run of every cell, without QC filtering.
QUICK: dict[str, Any] = {**HEAD, "filtering": False}
# Every stage, with a fixed UMI floor, no filtering, Harmony over the lanes,
# and few random sets for the association tests, as a list, which a JSON
# configuration gives.
FULL: dict[str, Any] = {
    **HEAD,
    "filtering": False,
    "guide_calling": {"method": "fixed", "min_umi": 10},
    "harmony": {"batch_columns": ["lane"]},
    "signature": {"dims": 5, "sample_column": "lane"},
    "mixscape": True,
    "associations": {"n_resamples": [99, 499, 999], "sample_column": "lane"},
    "profiles": {"sample_column": "lane"},
    "clusters": {"min_cluster_size": 2},
}
STAGES = [
    "input_snapshot",
    "filtering",
    "guide_calling",
    "perturbation_labels",
    "efficiency",
    "highly_variable_features",
    "normalization",
    "pca",
    "harmony",
    "edistance",
    "signature",
    "mixscape",
    "associations",
    "profiles",
    "perturbation_clusters",
]


def _screen(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the RNA counts, the guide counts and the lane of each cell.

    60 of GENE_A's 80 cells respond and 20 escape. Each responding cell's own
    gene falls tenfold; GENE_A and GENE_B lower genes 10 to 19, and GENE_C
    and GENE_D raise genes 20 to 29. Ten cells carry no guide, and the last
    four carry guides of GENE_A and GENE_C.
    """
    rng = np.random.default_rng(seed)
    guide_of = np.r_[
        np.repeat([0, 1], 40),
        np.repeat([2, 3], 30),
        np.full(40, 4),
        np.full(40, 5),
        np.repeat([6, 7, 8, 9], 50),
    ]
    n_cells = len(guide_of) + 14
    library = rng.lognormal(np.log(1500), 0.25, n_cells)
    base = rng.normal(-6.8, 0.6, N_GENES)
    base[:30] = -5.9
    mu = np.exp(base[None, :] + np.log(library)[:, None])
    lanes = np.where(np.arange(n_cells) % 2 == 0, "L1", "L2")
    mu[lanes == "L2", 30:35] *= 1.5
    gene_a = np.flatnonzero(guide_of <= 1)
    responding = gene_a[rng.permutation(len(gene_a))[:60]]
    gene_b = np.flatnonzero((guide_of == 2) | (guide_of == 3))
    gene_c = np.flatnonzero(guide_of == 4)
    gene_d = np.flatnonzero(guide_of == 5)
    for cells, own in ((responding, 0), (gene_b, 1), (gene_c, 2), (gene_d, 3)):
        mu[cells, own] *= 0.1
    mu[np.r_[responding, gene_b][:, None], np.arange(10, 20)] *= 0.1
    mu[np.r_[gene_c, gene_d][:, None], np.arange(20, 30)] *= 4.0
    rna = rng.poisson(mu)
    guides = rng.poisson(0.2, size=(n_cells, len(GUIDES)))
    carried = np.round(2.0 ** rng.normal(6, 0.5, len(guide_of))).astype(np.int64)
    guides[np.arange(len(guide_of)), guide_of] += carried
    guides[-4:, [0, 4]] += 60
    return rna.astype(np.uint16), guides.astype(np.uint16), lanes


RNA, GUIDE_COUNTS, LANES = _screen()
N_CELLS = len(RNA)
# The cells that every cell column of the screen fills: the columns with gaps
# leave out every tenth cell, 44 of the 434.
FILLED = np.arange(N_CELLS) % 10 != 0


def _open(location: str | Path, **options: Any) -> DataStore:
    return DataStore(
        str(location), default_assay="RNA", min_features_per_cell=0, **options
    )


def _write(location: Path) -> DataStore:
    write_count_store(str(location), {"RNA": RNA, "CRISPR": GUIDE_COUNTS}, "uint16")
    store = _open(location)
    features = store._get_assay("CRISPR").feats
    features.insert(
        "target_gene_name", np.asarray([target for target, _ in GUIDES], dtype=object)
    )
    features.insert(
        "target_gene_id", np.asarray([gene for _, gene in GUIDES], dtype=object)
    )
    depth = np.log1p(RNA.sum(axis=1).astype(np.float64))
    store.cells.insert("lane", np.asarray(LANES, dtype=object))
    store.cells.insert("depth", depth)
    # A lane label and a depth that the cells outside FILLED lack; a lane label
    # of empty or whitespace text there; and a depth of NaN or infinity there.
    insert_nullable_cell_column(store, "lane_gaps", np.asarray(LANES), ~FILLED)
    insert_nullable_cell_column(store, "depth_gaps", depth, ~FILLED)
    first = np.arange(N_CELLS) % 20 == 0
    store.cells.insert("lane_blank", np.where(FILLED, LANES, np.where(first, "", "  ")))
    store.cells.insert(
        "depth_inf", np.where(FILLED, depth, np.where(first, np.nan, np.inf))
    )
    store.cells.insert("filled", FILLED)
    return store


def _copy(location: Path, tmp_path: Path) -> DataStore:
    target = tmp_path / "copy.zarr"
    shutil.copytree(location, target)
    return _open(target)


def _written(store: DataStore) -> tuple[Any, ...]:
    """Return what a run writes: run records, and every artifact it can make."""
    return (
        [run.run_id for run in store.pipeline.list_runs(limit=1000)],
        store.artifacts.list(scope="datastore"),
        store.artifacts.list(from_assay="RNA"),
        store.artifacts.list(from_assay="CRISPR"),
    )


def _stages(run: PipelineRun) -> dict[str, dict[str, Any]]:
    return {stage["stage"]: stage for stage in run.report()["stages"]}


def _statuses(run: PipelineRun) -> dict[str, str]:
    return {name: stage["status"] for name, stage in _stages(run).items()}


def _reused(run: PipelineRun) -> dict[str, bool]:
    """Return whether each output of each stage was reused, by output key."""
    return {
        output["outputKey"]: output["reused"]
        for stage in run.report()["stages"]
        for output in stage["outputs"]
    }


def _resolve(store: DataStore, **keywords: Any) -> RecipePlan:
    """Resolve a call as ``ds.pipeline.perturbation`` does, writing nothing.

    Every omitted keyword takes the public method's default, and ``labels``
    those of the screen.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            PipelineAccessor.perturbation
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in {"callback", "cancel", "labels"}
    }
    return resolve_perturbation(store, **{**defaults, "labels": LABELS, **keywords})


def _call_file(path: Path, prefix: str = "") -> Path:
    """Write Cell Ranger's call file of the screen's carried guides.

    Every cell with more than 10 UMIs of a guide calls it, and one barcode
    matches no cell.
    """
    rows = [("unknown-barcode", "1", "CRISPR0", "50")]
    for cell in range(N_CELLS):
        called = np.flatnonzero(GUIDE_COUNTS[cell] > 10)
        if len(called):
            rows.append(
                (
                    f"cell{cell}",
                    str(len(called)),
                    "|".join(f"CRISPR{guide}" for guide in called),
                    "|".join(str(GUIDE_COUNTS[cell, guide]) for guide in called),
                )
            )
    frame = pd.DataFrame(
        rows, columns=["cell_barcode", "num_features", "feature_call", "num_umis"]
    )
    frame["cell_barcode"] = prefix + frame["cell_barcode"]
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def screen(tmp_path_factory) -> Path:
    """The location of the simulated screen, written once."""
    location = tmp_path_factory.mktemp("perturbation_screen") / "screen.zarr"
    _write(location)
    return location


@pytest.fixture(scope="module")
def runs(screen: Path, tmp_path_factory) -> SimpleNamespace:
    """A default run and a run of every stage, committed on a copy of the screen."""
    location = tmp_path_factory.mktemp("perturbation_runs") / "screen.zarr"
    shutil.copytree(screen, location)
    store = _open(location)
    default = store.pipeline.perturbation(labels=LABELS, label="default", **HEAD)
    full = store.pipeline.perturbation(labels=LABELS, label="full", **FULL)
    return SimpleNamespace(location=location, store=store, default=default, full=full)


@pytest.fixture(scope="module")
def readonly(screen: Path) -> DataStore:
    """The screen opened read-only; resolving a call writes nothing."""
    return _open(screen, zarr_mode="r")


@pytest.fixture(scope="module")
def rejection_store(screen: Path, tmp_path_factory) -> DataStore:
    """One writable copy for calls that must be refused before any write."""
    location = tmp_path_factory.mktemp("perturbation_rejections") / "screen.zarr"
    shutil.copytree(screen, location)
    return _open(location)


# Runs ------------------------------------------------------------------------


def test_a_default_run_labels_the_cells_and_measures_each_target(runs) -> None:
    run = runs.default

    assert (run.recipe, run.assay, run.status, run.label) == (
        "perturbation",
        "RNA",
        "completed",
        "default",
    )
    assert _statuses(run) == {
        stage: "skipped"
        if stage
        in {
            "harmony",
            "signature",
            "mixscape",
            "associations",
            "profiles",
            "perturbation_clusters",
        }
        else "completed"
        for stage in STAGES
    }
    assert list(run) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "guide_thresholds",
        "guide_calls",
        "perturbation_labels",
        "efficiency",
        "highly_variable_features",
        "normalized",
        "pca",
        "edistance",
    ]
    kinds = {key: ref.kind for key, ref in run.items()}
    assert kinds["guide_thresholds"] == "guide_thresholds"
    assert kinds["guide_calls"] == "guide_calls"
    assert kinds["perturbation_labels"] == "perturbation_labels"
    assert kinds["efficiency"] == "perturbation_efficiency"
    assert kinds["edistance"] == "perturbation_edistance"
    # The guide calls and labels belong to the guide assay, the results to RNA.
    assert run["guide_calls"].assay == run["perturbation_labels"].assay == "CRISPR"
    assert run["efficiency"].assay == run["edistance"].assay == "RNA"
    # The guides of the analysis cells are called, and E-distances read the PCA.
    store = runs.store
    calls = store.artifacts.inspect(run["guide_calls"])
    assert calls.inputs["cell_selection"] == run["analysis_cell_selection"].to_dict()
    assert calls.parameters["method"] == "poisson_gaussian"
    distances = store.artifacts.inspect(run["edistance"])
    assert distances.inputs["coordinates"] == run["pca"].to_dict()
    assert run.cells.columns == ("I", "ids", "names")
    assert run.features.columns == ("I", "ids", "names", "highly_variable_features")
    config = run.report()["run"]["config"]
    assert (config["guideAssay"], config["profileLabels"]) == ("CRISPR", None)
    filtering = config["stages"]["filtering"]
    assert (filtering["enabled"], filtering["method"], filtering["attrs"]) == (
        True,
        "mad",
        ["RNA_nCounts", "RNA_nFeatures"],
    )
    assert config["stages"] == {
        "filtering": filtering,
        "guide_calling": {},
        "labels": {"controls": ["NT"]},
        "efficiency": {},
        "hvg": {"top_n": 50},
        "normalization": {},
        "pca": {"dims": 5},
        "harmony": False,
        "edistance": {},
        "signature": False,
        "mixscape": False,
        "associations": False,
        "profiles": False,
        "clusters": False,
    }


def test_a_full_run_runs_every_stage(runs) -> None:
    run, store = runs.full, runs.store

    assert _statuses(run) == {
        stage: "skipped" if stage == "filtering" else "completed" for stage in STAGES
    }
    assert list(run) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "guide_calls",
        "perturbation_labels",
        "efficiency",
        "highly_variable_features",
        "normalized",
        "pca",
        "harmony",
        "edistance",
        "signature",
        "mixscape",
        "associations",
        "profiles",
        "perturbation_clusters",
    ]
    assert {key: run[key].kind for key in list(run)[-6:]} == {
        "edistance": "perturbation_edistance",
        "signature": "perturbation_signature",
        "mixscape": "mixscape_classes",
        "associations": "perturbation_associations",
        "profiles": "perturbation_profiles",
        "perturbation_clusters": "perturbation_clusters",
    }
    inspect_ = store.artifacts.inspect
    # E-distances and the signature read the Harmony coordinates.
    assert inspect_(run["edistance"]).inputs["coordinates"] == run["harmony"].to_dict()
    signature = inspect_(run["signature"])
    assert signature.inputs["coordinates"] == run["harmony"].to_dict()
    assert signature.inputs["features"] == run["highly_variable_features"].to_dict()
    # Trans pairs over the HVGs, with the list of random sets as given.
    tests = inspect_(run["associations"])
    assert tests.inputs["features"] == run["highly_variable_features"].to_dict()
    assert tests.parameters["n_resamples"] == [99, 499, 999]
    # A run with Mixscape profiles the knocked-out cells of each target.
    profiles = inspect_(run["profiles"])
    assert profiles.inputs["labels"] == run["mixscape"].to_dict()
    assert (
        profiles.inputs["feature_selection"]
        == run["highly_variable_features"].to_dict()
    )
    config = run.report()["run"]["config"]
    assert config["profileLabels"] == "mixscape"
    assert config["stages"] == {
        "filtering": {"enabled": False},
        "guide_calling": {"method": "fixed", "min_umi": 10},
        "labels": {"controls": ["NT"]},
        "efficiency": {},
        "hvg": {"top_n": 50},
        "normalization": {},
        "pca": {"dims": 5},
        "harmony": {"batch_columns": ["lane"]},
        "edistance": {},
        "signature": {"dims": 5, "sample_column": "lane"},
        "mixscape": {},
        "associations": {"n_resamples": [99, 499, 999], "sample_column": "lane"},
        "profiles": {"sample_column": "lane"},
        "clusters": {"min_cluster_size": 2},
    }


def test_the_run_cells_hold_each_cells_mixscape_class(runs) -> None:
    run, store = runs.full, runs.store

    # The guide calls and labels belong to the guide assay, which the run's
    # RNA cell view cannot hold, so its cell fields are Mixscape's alone.
    assert run.cells.columns == (
        "I",
        "ids",
        "names",
        "mixscape_class",
        "mixscape_posterior",
    )
    classes = store.perturbation.load_mixscape(run["mixscape"]).cells
    view = run.cells.to_pandas_dataframe(["mixscape_class", "mixscape_posterior"])
    np.testing.assert_array_equal(
        view["mixscape_class"].astype(str), classes["mixscape_class"].to_numpy()
    )
    np.testing.assert_array_equal(
        view["mixscape_posterior"], classes["posterior"].to_numpy()
    )
    assert set(classes["mixscape_class"]) >= {"GENE_A KO", "GENE_A NP", "control"}


def test_each_stage_summary_equals_its_loader(runs) -> None:
    store = runs.store
    default = {name: stage["summary"] for name, stage in _stages(runs.default).items()}
    full = {name: stage["summary"] for name, stage in _stages(runs.full).items()}

    # Each summary key is pinned, since a renamed array drops it silently.
    assert {
        name: None if summary is None else set(summary)
        for name, summary in full.items()
    } == {
        "input_snapshot": {"input_cell_selection", "feature_universe"},
        "filtering": None,
        "guide_calling": {"cells", "negative", "single", "multiple", "calledGuides"},
        "perturbation_labels": {
            "cells",
            "controls",
            "negative",
            "multipleTargets",
            "targets",
            "targetCells",
            "fewCellTargets",
            "fewCellTargetCount",
        },
        "efficiency": {
            "by",
            "rows",
            "statuses",
            "knockedDown",
            "medianRemainingFraction",
            "noFeature",
        },
        "highly_variable_features": {"features"},
        "normalization": {"cells", "features"},
        "pca": {"cells", "dims", "varianceRatio", "varianceExplained"},
        "harmony": {"rounds", "converged"},
        "edistance": {
            "by",
            "rows",
            "statuses",
            "significant",
            "atFloor",
            "nPermutations",
        },
        "signature": {"cells", "features", "controls", "samples", "neighbors"},
        "mixscape": {
            "rows",
            "statuses",
            "cells",
            "knockedOut",
            "fewDeGenes",
            "perturbationType",
        },
        "associations": {
            "discoveryPairs",
            "testedDiscoveryPairs",
            "discoveries",
            "calibrationPairs",
            "falseDiscoveries",
            "calibrationGrade",
            "calibrationStatus",
            "falseDiscoveriesByControlGuides",
            "positivePairs",
            "positiveDetected",
            "alpha",
        },
        "profiles": {
            "perturbations",
            "features",
            "leftOut",
            "zeroSdFeatures",
            "samples",
        },
        "perturbation_clusters": {
            "perturbations",
            "clusters",
            "unassigned",
            "largestCluster",
        },
    }
    # Fitted guide calls also summarize their fits.
    assert set(default["guide_calling"]) == {"guide_thresholds", "guide_calls"}
    fits = store.perturbation.load_guide_thresholds(runs.default["guide_calls"])
    assert default["guide_calling"]["guide_thresholds"] == {
        "samples": 1,
        "guides": len(fits.guides),
        "statuses": {
            status: int((fits.table["status"] == status).sum())
            for status in ("fitted", "no_cells", "too_sparse", "no_fit", "no_positive")
        },
    }
    for run, summary in (
        (runs.default, default["guide_calling"]["guide_calls"]),
        (runs.full, full["guide_calling"]),
    ):
        calls = store.perturbation.load_guide_calls(run["guide_calls"])
        n_guides = calls.cells["n_guides"]
        assert summary == {
            "cells": len(calls.cells),
            "negative": int((n_guides == 0).sum()),
            "single": int((n_guides == 1).sum()),
            "multiple": int((n_guides > 1).sum()),
            "calledGuides": calls.calls["guide"].nunique(),
        }
    labelled = store.perturbation.load_labels(runs.full["perturbation_labels"])
    cells = labelled.cells
    targets = sorted(
        labelled.guides.loc[~labelled.guides["is_control"], "target"].unique()
    )
    single = cells.loc[cells["n_targets"] == 1, "label"].value_counts()
    per_target = single.reindex(targets, fill_value=0)
    assert full["perturbation_labels"] == {
        "cells": len(cells),
        "controls": int(cells["is_control"].sum()),
        "negative": int((cells["label"] == "Negative").sum()),
        "multipleTargets": int((cells["n_targets"] > 1).sum()),
        "targets": len(targets),
        "targetCells": {
            "min": int(per_target.min()),
            "median": float(per_target.median()),
            "max": int(per_target.max()),
        },
        "fewCellTargets": {},
        "fewCellTargetCount": 0,
    }
    assert full["perturbation_labels"]["multipleTargets"] == 4
    table = store.perturbation.load_efficiency(runs.full["efficiency"]).table
    tested = table["status"] == "tested"
    assert full["efficiency"] == {
        "by": "target",
        "rows": len(table),
        "statuses": {
            status: int((table["status"] == status).sum())
            for status in ("tested", "no_feature", "no_cells", "no_control_expression")
        },
        "knockedDown": int(
            (
                tested
                & (table["adjusted_p_value"] < 0.05)
                & (table["remaining_fraction"] < 1)
            ).sum()
        ),
        "medianRemainingFraction": pytest.approx(
            float(table.loc[tested, "remaining_fraction"].median()), abs=1e-4
        ),
        "noFeature": [],
    }
    assert full["efficiency"]["knockedDown"] == 4
    distances = store.perturbation.load_edistance(runs.full["edistance"])
    tested = distances.table["status"] == "tested"
    floor = 1 / (distances.n_permutations + 1)
    assert full["edistance"] == {
        "by": "target",
        "rows": len(distances.table),
        "statuses": {
            "tested": int(tested.sum()),
            "too_few_cells": int((~tested).sum()),
        },
        "significant": int(
            (tested & (distances.table["adjusted_p_value"] < 0.05)).sum()
        ),
        "atFloor": int((tested & np.isclose(distances.table["p_value"], floor)).sum()),
        "nPermutations": distances.n_permutations,
    }
    selected = np.asarray(
        runs.full.features.fetch_all("highly_variable_features"), dtype=bool
    )
    hvg_ids = np.asarray(runs.full.features.fetch_all("ids")).astype(str)[selected]
    signature = store.perturbation.load_signature(
        runs.full["signature"], features=[hvg_ids[0]]
    )
    assert full["signature"] == {
        "cells": len(signature.cells),
        "features": len(hvg_ids),
        "controls": int(signature.cells["is_control"].sum()),
        "samples": signature.cells["sample"].nunique(),
        "neighbors": signature.n_neighbors,
    }
    mixscape = store.perturbation.load_mixscape(runs.full["mixscape"])
    rows = mixscape.targets
    assert full["mixscape"] == {
        "rows": len(rows),
        "statuses": {
            status: int((rows["status"] == status).sum())
            for status in ("fitted", "few_de_genes", "few_cells", "no_fit")
        },
        "cells": int(rows["n_cells"].sum()),
        "knockedOut": int(rows["n_ko"].sum()),
        "fewDeGenes": sorted(set(rows.loc[rows["status"] == "few_de_genes", "target"])),
        "perturbationType": mixscape.perturbation_type,
    }
    tests = store.perturbation.load_associations(runs.full["associations"])
    false = tests.calibration.loc[tests.calibration["significant"], "control_guides"]
    assert full["associations"] == {
        "discoveryPairs": len(tests.discovery),
        "testedDiscoveryPairs": int(tests.discovery["pass_qc"].sum()),
        "discoveries": int(tests.discovery["significant"].sum()),
        "calibrationPairs": len(tests.calibration),
        "falseDiscoveries": tests.n_false_discoveries,
        "calibrationGrade": tests.calibration_grade,
        "calibrationStatus": "checked",
        "falseDiscoveriesByControlGuides": false.value_counts().to_dict(),
        "positivePairs": len(tests.positive),
        "positiveDetected": int(tests.positive["significant"].sum()),
        "alpha": tests.alpha,
    }
    profiles = store.perturbation.load_profiles(runs.full["profiles"])
    assert full["profiles"] == {
        "perturbations": len(profiles.values),
        "features": profiles.values.shape[1],
        "leftOut": len(profiles.left_out),
        "zeroSdFeatures": int((profiles.features["n_zero_sd"] > 0).sum()),
        "samples": len(profiles.samples),
    }
    clusters = store.perturbation.load_clusters(runs.full["perturbation_clusters"])
    assigned = clusters.table["cluster"][clusters.table["cluster"] >= 0]
    assert full["perturbation_clusters"] == {
        "perturbations": len(clusters.table),
        "clusters": assigned.nunique(),
        "unassigned": int((clusters.table["cluster"] < 0).sum()),
        "largestCluster": int(assigned.value_counts().max()),
    }
    # The two programs make two clusters of two targets.
    assert full["perturbation_clusters"]["clusters"] == 2
    # The report shows each summary, as its Diagnostics section.
    markdown = runs.full.report(format="markdown")
    assert "### perturbation_labels" in markdown
    assert '"calibrationGrade"' in markdown


def test_a_signature_without_a_sample_column_counts_no_sample(runs, tmp_path) -> None:
    store = _copy(runs.location, tmp_path)
    run = runs.full
    signature = store.perturbation.signature(
        run["perturbation_labels"], run["harmony"], dims=5
    )

    summary = summaries.summarize_stage(store, "signature", (("signature", signature),))

    # Every cell is in one sample, which the loader names None.
    loaded = store.perturbation.load_signature(signature)
    assert loaded.sample_column is None
    assert summary is not None
    assert summary["samples"] == loaded.cells["sample"].nunique() == 0
    assert summary["cells"] == len(loaded.cells)


# Reuse -----------------------------------------------------------------------


def test_the_stage_methods_return_the_runs_artifacts(runs) -> None:
    run, store = runs.full, runs.store
    labels = run["perturbation_labels"]
    hvgs = run["highly_variable_features"]

    # Each method called with the run's inputs and settings returns the run's
    # artifact, as one call of the recipe would have made it.
    assert (
        store.perturbation.call_guides(
            run["analysis_cell_selection"], method="fixed", min_umi=10
        )
        == run["guide_calls"]
    )
    assert store.perturbation.labels(run["guide_calls"], controls=["NT"]) == labels
    assert store.perturbation.efficiency(labels) == run["efficiency"]
    assert store.perturbation.edistance(labels, run["harmony"]) == run["edistance"]
    # The signature's features default to those of its coordinates, the HVGs.
    assert (
        store.perturbation.signature(
            labels, run["harmony"], dims=5, sample_column="lane"
        )
        == run["signature"]
    )
    assert store.perturbation.mixscape(run["signature"]) == run["mixscape"]
    # A tuple of random sets records the same tests as the run's list.
    assert (
        store.perturbation.associations(
            labels,
            features=hvgs,
            n_resamples=(99, 499, 999),
            sample_column="lane",
        )
        == run["associations"]
    )
    assert (
        store.perturbation.profiles(run["mixscape"], hvgs, sample_column="lane")
        == run["profiles"]
    )
    assert (
        store.perturbation.cluster(run["profiles"], min_cluster_size=2)
        == run["perturbation_clusters"]
    )


def test_an_identical_second_call_creates_no_artifacts(runs, tmp_path) -> None:
    store = _copy(runs.location, tmp_path)

    again = store.pipeline.perturbation(labels=LABELS, **FULL)

    summary = again.report()["summary"]
    assert summary["createdArtifacts"] == []
    assert summary["reusedArtifacts"]
    assert dict(again) == dict(runs.full)
    assert again.run_id != runs.full.run_id
    # The reused stages report the summaries of the computed ones.
    assert [stage["summary"] for stage in again.report()["stages"]] == [
        stage["summary"] for stage in runs.full.report()["stages"]
    ]


def test_the_recipe_reuses_the_artifacts_of_standalone_calls(screen, tmp_path) -> None:
    store = _copy(screen, tmp_path)
    cells = store.select_measured_cells("CRISPR")
    hvgs = store.features.hvgs(cells, top_n=50, show_plot=False)
    pca = store.reduction.pca(store.features.normalize(cells, hvgs), dims=5)
    calls = store.perturbation.call_guides(cells, method="fixed", min_umi=10)
    labels = store.perturbation.labels(calls, controls=["NT"])
    efficiency = store.perturbation.efficiency(labels, by="guide")
    distances = store.perturbation.edistance(labels, pca, n_permutations=99)
    tests = store.perturbation.associations(
        labels, features=hvgs, n_resamples=(99, 499, 999), positive_controls=None
    )
    profiles = store.perturbation.profiles(labels, hvgs, mask_targets=False)
    clusters = store.perturbation.cluster(profiles, min_cluster_size=2)

    run = store.pipeline.perturbation(
        cells=cells,
        guide_calling={"method": "fixed", "min_umi": 10},
        labels=LABELS,
        efficiency={"by": "guide"},
        edistance={"n_permutations": 99},
        associations={"n_resamples": [99, 499, 999], "positive_controls": None},
        profiles={"mask_targets": False},
        clusters={"min_cluster_size": 2},
        **HEAD,
    )

    expected = {
        "highly_variable_features": hvgs,
        "pca": pca,
        "guide_calls": calls,
        "perturbation_labels": labels,
        "efficiency": efficiency,
        "edistance": distances,
        "associations": tests,
        "profiles": profiles,
        "perturbation_clusters": clusters,
    }
    assert {key: run[key] for key in expected} == expected
    reused = _reused(run)
    assert all(reused[key] for key in expected)
    # Without Mixscape the profiles average every cell of each target.
    assert run.report()["run"]["config"]["profileLabels"] == "perturbation_labels"


def test_a_run_after_an_rna_run_reuses_its_cells_genes_and_coordinates(
    screen, tmp_path
) -> None:
    store = _copy(screen, tmp_path)
    rna = store.pipeline.rna(
        cell_cycle=False,
        umap=False,
        leiden=False,
        doublets=False,
        markers=False,
        harmony={"batch_columns": ["lane"]},
        **HEAD,
    )
    head = ("analysis_cell_selection", "highly_variable_features", "normalized", "pca")

    # The same cells and settings filter, select, normalize and reduce alike.
    same = store.pipeline.perturbation(
        labels=LABELS, harmony={"batch_columns": ["lane"]}, **HEAD
    )
    # A run given the RNA run's cells takes them as they are.
    given = store.pipeline.perturbation(
        cells=rna, labels=LABELS, harmony={"batch_columns": ["lane"]}, **HEAD
    )

    for run in (same, given):
        assert all(run[key] == rna[key] for key in (*head, "harmony"))
        reused = _reused(run)
        assert all(
            reused[key]
            for key in ("highly_variable_features", "normalized", "pca", "harmony")
        )
    assert same["input_cell_selection"] != rna["analysis_cell_selection"]
    assert given["input_cell_selection"] == rna["analysis_cell_selection"]
    assert given.report()["run"]["config"]["cells"]["run"] == {
        "runId": rna.run_id,
        "label": None,
    }


# Guide calls imported from Cell Ranger ----------------------------------------


def test_guide_calling_imports_cell_rangers_calls(screen, tmp_path) -> None:
    store = _copy(screen, tmp_path)
    path = _call_file(tmp_path / "protospacer_calls_per_cell.csv")

    # A path object is recorded as the text of its path.
    run = store.pipeline.perturbation(
        labels=LABELS, guide_calling={"path": path}, efficiency=False, **QUICK
    )

    calls = store.artifacts.inspect(run["guide_calls"])
    assert calls.operation == "import_guide_calls"
    assert "guide_thresholds" not in run
    assert run.report()["run"]["config"]["stages"]["guide_calling"] == {
        "path": str(path)
    }
    summary = _stages(run)["guide_calling"]["summary"]
    loaded = store.perturbation.load_guide_calls(run["guide_calls"])
    assert summary["cells"] == len(loaded.cells)
    assert (summary["fileBarcodes"], summary["unmatchedBarcodes"]) == (
        N_CELLS - 10 + 1,
        1,
    )
    # The import reads the same calls as a fixed floor of 11 UMIs.
    fixed = store.perturbation.load_guide_calls(
        store.perturbation.call_guides(
            run["analysis_cell_selection"], method="fixed", min_umi=11
        )
    )
    pd.testing.assert_series_equal(loaded.cells["call"], fixed.cells["call"])
    assert _stages(run)["guide_calling"]["warnings"]


def test_one_call_file_reaches_the_cells_of_one_sample(screen, tmp_path) -> None:
    location = tmp_path / "merged.zarr"
    shutil.copytree(screen, location)
    # A store merged from the two lanes names each cell <lane>__<barcode>.
    ids = np.asarray([f"{lane}__cell{cell}" for cell, lane in enumerate(LANES)])
    zarr.open_group(str(location / "cellData"), mode="r+").create_array(
        "ids", data=ids, overwrite=True
    )
    store = _open(location)
    store.cells.insert("lane1", LANES == "L1")
    path = _call_file(tmp_path / "protospacer_calls_per_cell.csv")
    remedy = (
        "or call the guides from their counts with guide_calling settings without path"
    )
    before = _written(store)

    # The file reaches neither the cells of both lanes without a prefix nor the
    # other lane's cells with one.
    for cells, prefix, message in (
        (
            "I",
            None,
            "The call file holds the barcodes of one sample, but the run's cells "
            "come from several samples of a merged store, such as 'L1' and 'L2'; "
            "restrict cells to the cells of one sample and pass its barcode_prefix, "
            f"such as 'L1__', {remedy}",
        ),
        (
            "I",
            "L1__",
            "The call file's barcodes reach only the cells whose IDs start with "
            "barcode_prefix 'L1__', but 217 of the run's cells, such as 'L2__cell1', "
            f"do not; restrict cells to the cells of that sample, {remedy}",
        ),
        (
            "lane1",
            "L2__",
            "barcode_prefix 'L2__', but 217 of the run's cells, such as 'L1__cell0'",
        ),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            store.pipeline.perturbation(
                cells=cells,
                labels=LABELS,
                guide_calling={"path": path, "barcode_prefix": prefix},
                **HEAD,
            )
    assert _written(store) == before

    # One lane's cells with its prefix: the barcodes of the other lane's cells,
    # and of the cells that QC filtering removed, match no analysis cell.
    run = store.pipeline.perturbation(
        cells="lane1",
        labels=LABELS,
        guide_calling={"path": path, "barcode_prefix": "L1__"},
        efficiency=False,
        **HEAD,
    )
    selection = store.artifacts.load(run["analysis_cell_selection"])
    analysed = np.asarray(selection["values"][:], dtype=bool)
    called = (GUIDE_COUNTS > 10).any(axis=1)
    removed = int(np.count_nonzero(called & (LANES == "L1") & ~analysed))
    assert removed
    summary = _stages(run)["guide_calling"]["summary"]
    assert (summary["fileBarcodes"], summary["unmatchedBarcodes"]) == (
        N_CELLS - 10 + 1,
        1 + int(np.count_nonzero(called & (LANES == "L2"))) + removed,
    )
    # Every analysed cell that the file calls has its calls.
    cells = store.perturbation.load_guide_calls(run["guide_calls"]).cells
    assert cells.index.str.startswith("L1__").all()
    assert int((cells["n_guides"] > 0).sum()) == int(
        np.count_nonzero(called & analysed)
    )


# Failures and cancellation ----------------------------------------------------


def test_a_called_guide_without_a_target_fails_the_labels_stage(
    screen, tmp_path
) -> None:
    store = _copy(screen, tmp_path)
    # Guide 5 loses its target; cells call it, which only labelling finds.
    targets = np.asarray([target for target, _ in GUIDES], dtype=object)
    targets[5] = ""
    store._get_assay("CRISPR").feats.insert("target_gene_name", targets, overwrite=True)

    with pytest.raises(PipelineExecutionError) as raised:
        store.pipeline.perturbation(labels=LABELS, **QUICK)

    assert raised.value.stage == "perturbation_labels"
    assert "called guides have no target" in str(raised.value.__cause__)
    failed = store.pipeline.open(run_id=raised.value.run_id)
    assert failed.status == "failed"
    statuses = _statuses(failed)
    assert statuses["guide_calling"] == "completed"
    assert statuses["perturbation_labels"] == "failed"


def test_a_callback_follows_the_run_and_its_token_stops_it(screen, tmp_path) -> None:
    store = _copy(screen, tmp_path)
    token = CancellationToken()
    events: list[Any] = []

    def follow(event: Any) -> None:
        events.append(event)
        if event.kind == "stage_completed" and event.stage == "perturbation_labels":
            token.cancel("enough")

    with pytest.raises(PipelineCancelled) as raised:
        store.pipeline.perturbation(
            labels=LABELS, callback=follow, cancel=token, **QUICK
        )

    run = store.pipeline.open(run_id=raised.value.run_id)
    assert run.status == "interrupted"
    assert run.report()["run"]["interruption"]["kind"] == "cancelled"
    statuses = _statuses(run)
    assert statuses["perturbation_labels"] == "completed"
    assert "efficiency" not in statuses
    completed = {
        event.stage: event.summary
        for event in events
        if event.kind == "stage_completed"
    }
    assert (
        completed["perturbation_labels"]
        == (_stages(run)["perturbation_labels"]["summary"])
    )
    assert [event.kind for event in events][-1] == "run_interrupted"


# Checks before the run --------------------------------------------------------


def test_a_resolved_call_runs_its_stages_in_order(readonly) -> None:
    plan = _resolve(readonly, **HEAD)

    assert plan.recipe == "perturbation"
    assert plan.assay == "RNA"
    assert list(plan.stage_order) == STAGES
    assert {stage.name for stage in plan.stages if stage.action is None} == {
        "harmony",
        "signature",
        "mixscape",
        "associations",
        "profiles",
        "perturbation_clusters",
    }
    # Profiles of a run without Mixscape average every cell of a target.
    plan = _resolve(readonly, clusters=True, profiles={"min_cells": 10}, **HEAD)
    assert plan.config["profileLabels"] == "perturbation_labels"
    assert plan.config["stages"]["profiles"] == {"min_cells": 10}
    # Without clusters, empty profile settings are no settings.
    plan = _resolve(readonly, profiles={}, signature={}, **HEAD)
    assert plan.config["stages"]["profiles"] is False
    assert plan.config["stages"]["signature"] is False
    # The measuring stages switch off.
    plan = _resolve(readonly, efficiency=False, edistance=False, **HEAD)
    assert (
        plan.config["stages"]["efficiency"],
        plan.config["stages"]["edistance"],
    ) == (
        False,
        False,
    )
    assert {"efficiency", "edistance"} <= {
        stage.name for stage in plan.stages if stage.action is None
    }


def test_the_guide_assay_is_the_only_crispr_assay_or_the_named_one(
    readonly, tmp_path
) -> None:
    assert _resolve(readonly, guide_assay="CRISPR", **HEAD).config["guideAssay"] == (
        "CRISPR"
    )
    # A store without a CRISPR assay, or with two, names none by itself.
    alone = tmp_path / "alone.zarr"
    write_count_store(str(alone), {"RNA": RNA}, "uint16")
    pair = tmp_path / "pair.zarr"
    write_count_store(
        str(pair),
        {"RNA": RNA, "CRISPR": GUIDE_COUNTS, "GUIDES": GUIDE_COUNTS},
        "uint16",
    )
    for store, crispr in (
        (_open(alone), []),
        (_open(pair, assay_types={"GUIDES": "CRISPR"}), ["CRISPR", "GUIDES"]),
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The perturbation recipe needs an assay declared 'CRISPR'; name it "
                f"with guide_assay=...; CRISPR assays: {crispr!r}"
            ),
        ):
            _resolve(store, **HEAD)


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        # Labels need controls, which CyteArc never guesses.
        ({"labels": None}, ValueError, re.escape(LABELS_NEED_CONTROLS)),
        ({"labels": {"policy": "single"}}, ValueError, "labels needs controls"),
        ({"labels": True}, TypeError, "labels must be a mapping because"),
        ({"labels": {"controls": "NT"}}, TypeError, "controls must be a sequence"),
        (
            {"labels": {"controls": ["Non-Targeting"]}},
            ValueError,
            "controls name targets that no guide has: 'Non-Targeting'",
        ),
        (
            {"labels": {"controls": ["NT"], "targets": {"nope": "GENE_X"}}},
            ValueError,
            "targets names 1 guides that are not guides of the assay",
        ),
        (
            {"labels": {"controls": ["NT"], "target_column": "missing"}},
            ValueError,
            "target_column 'missing' is not a feature column of assay 'CRISPR'",
        ),
        (
            {"labels": {"controls": ["NT"], "targets": {"CRISPR0": "control"}}},
            ValueError,
            "Targets conflict with labels reserved",
        ),
        (
            {"labels": {"controls": ["NT"], "unknown": 1}},
            ValueError,
            r"Unknown labels settings \['unknown'\]",
        ),
        # Guide calling calls guides or imports Cell Ranger's calls.
        ({"guide_calling": True}, TypeError, "always runs"),
        ({"guide_calling": {"method": "best"}}, ValueError, "method must be"),
        ({"guide_calling": {"method": "fixed"}}, ValueError, "needs min_umi"),
        (
            {"guide_calling": {"fraction_threshold": 0.5}},
            ValueError,
            "applies only to method='top_guide'",
        ),
        (
            {
                "guide_calling": {
                    "method": "fixed",
                    "min_umi": 3,
                    "sample_column": "lane",
                }
            },
            ValueError,
            "sample_column applies only to",
        ),
        (
            {"guide_calling": {"sample_column": "missing"}},
            KeyError,
            "guide_calling\\['sample_column'\\] names 'missing'",
        ),
        (
            {"guide_calling": {"sample_column": "lane_gaps"}},
            ValueError,
            "Guide calling per sample needs a 'lane_gaps' label for every cell, but "
            "44 of the run's cells have none",
        ),
        # Empty or whitespace text is no label, as the methods' samples take it.
        (
            {"guide_calling": {"sample_column": "lane_blank"}},
            ValueError,
            "Guide calling per sample needs a 'lane_blank' label for every cell, "
            "but 44 of the run's cells have none",
        ),
        (
            {"guide_calling": {"path": "calls.csv", "method": "fixed"}},
            ValueError,
            "not both",
        ),
        ({"guide_calling": {"barcode_prefix": "L1__"}}, ValueError, "not both"),
        ({"guide_calling": {"path": 3}}, TypeError, "path must be a path"),
        (
            {"guide_calling": {"path": "/no/such/calls.csv"}},
            FileNotFoundError,
            "is not a file",
        ),
        # The stages that measure each perturbation.
        ({"efficiency": {"by": "gene"}}, ValueError, "by must be 'target' or 'guide'"),
        (
            {"efficiency": {"id_column": "missing"}},
            ValueError,
            r"efficiency\['id_column'\] 'missing' is not a feature column",
        ),
        ({"edistance": {"n_permutations": -1}}, ValueError, "n_permutations must be"),
        ({"pca": {"dims": 0}}, ValueError, "pca dims must be at least 1"),
        (
            {"hvg": {"top_n": 5}, "pca": {"dims": 5}},
            ValueError,
            r"pca dims \(5\) must be below hvg top_n \(5\)",
        ),
        (
            {"mixscape": True},
            ValueError,
            r"signature dims \(15\) must be at most pca dims \(5\)",
        ),
        (
            {"signature": {"dims": 3}},
            ValueError,
            "signature settings apply only with mixscape",
        ),
        ({"signature": True}, TypeError, "signature must be a mapping or None"),
        (
            {"mixscape": {"min_cells": 0}, "signature": {"dims": 5}},
            ValueError,
            "min_cells must be at least 1",
        ),
        (
            {"mixscape": {"split_column": "missing"}, "signature": {"dims": 5}},
            KeyError,
            "mixscape\\['split_column'\\] names 'missing'",
        ),
        (
            {"mixscape": True, "signature": {"dims": 5, "sample_column": "missing"}},
            KeyError,
            "signature\\['sample_column'\\] names 'missing'",
        ),
        (
            {"mixscape": {"split_column": "lane_gaps"}, "signature": {"dims": 5}},
            ValueError,
            "Mixscape per split needs a 'lane_gaps' label",
        ),
        (
            {"mixscape": True, "signature": {"dims": 5, "sample_column": "lane_gaps"}},
            ValueError,
            "The signature per sample needs a 'lane_gaps' label",
        ),
        # The recipe tests trans pairs over the run's HVGs.
        ({"associations": {"pairs": "trans"}}, ValueError, "tests trans pairs"),
        ({"associations": {"features": None}}, ValueError, "tests trans pairs"),
        (
            {
                "associations": {
                    "positive_controls": pd.DataFrame(
                        {"target": ["GENE_A"], "feature": ["RNA0"]}
                    )
                }
            },
            ValueError,
            "tests trans pairs",
        ),
        (
            {"associations": {"positive_controls": "genes"}},
            ValueError,
            "positive_controls must be 'targets'",
        ),
        (
            {"associations": {"n_resamples": [1, 1, 1]}},
            ValueError,
            r"n_resamples\[1\] must be at least 2",
        ),
        (
            {"associations": {"covariates": ["missing"]}},
            KeyError,
            "names 'missing', which is not a cell column",
        ),
        (
            {"associations": {"covariates": ["lane"]}},
            ValueError,
            "covariate 'lane' must be numeric",
        ),
        (
            {"associations": {"covariates": "depth"}},
            TypeError,
            "covariates must be a sequence of column names",
        ),
        (
            {"associations": {"sample_column": 5}},
            TypeError,
            r"associations\['sample_column'\] must be a non-empty string",
        ),
        (
            {"associations": {"sample_column": "lane_gaps"}},
            ValueError,
            "Association testing per sample needs a 'lane_gaps' label for every "
            "cell, but 44 of the run's cells have none",
        ),
        (
            {"associations": {"covariates": ["depth_gaps"]}},
            ValueError,
            "Covariate 'depth_gaps' needs a finite value for every cell, but 44 of "
            "the run's cells have a missing or non-finite value",
        ),
        (
            {"associations": {"covariates": ["depth", "depth_inf"]}},
            ValueError,
            "Covariate 'depth_inf' needs a finite value for every cell, but 44 of "
            "the run's cells have a missing or non-finite value",
        ),
        (
            {"associations": {"id_column": "missing"}},
            ValueError,
            r"associations\['id_column'\] 'missing' is not a feature column",
        ),
        ({"clusters": {"metric": "cosine"}}, ValueError, "metric must be"),
        (
            {"clusters": {"perturbations": "GENE_A"}},
            TypeError,
            "perturbations must be a sequence",
        ),
        (
            {"profiles": {"min_cells": 5}},
            ValueError,
            "profiles settings apply only with clusters",
        ),
        (
            {"clusters": True, "profiles": {"sample_column": "missing"}},
            KeyError,
            "profiles\\['sample_column'\\] names 'missing'",
        ),
        (
            {"clusters": True, "profiles": {"id_column": "missing"}},
            ValueError,
            r"profiles\['id_column'\] 'missing' is not a feature column",
        ),
        (
            {"clusters": True, "profiles": {"sample_column": "lane_gaps"}},
            ValueError,
            "Profiling per sample needs a 'lane_gaps' label for every cell, but 44 "
            "of the run's cells have none",
        ),
        # The run's assays, cells and fields.
        ({"assay": "CRISPR"}, TypeError, "The perturbation recipe analyzes RNA"),
        ({"guide_assay": "RNA"}, TypeError, "declared with type 'CRISPR'"),
        ({"guide_assay": 3}, TypeError, "guide_assay must be a non-empty string"),
        ({"cells": "missing"}, KeyError, "Cell selection column 'missing'"),
        ({"harmony": {"batch_columns": ["missing"]}}, KeyError, "Harmony columns"),
        (
            {"snapshot_columns": ["mixscape_class"]},
            ValueError,
            "collide with reserved run fields",
        ),
        ({"callback": 3}, TypeError, "callback must be callable"),
    ],
)
def test_a_call_is_refused_before_anything_is_written(
    rejection_store, keywords, error, match
) -> None:
    store = rejection_store
    before = _written(store)
    call: dict[str, Any] = {"labels": LABELS, **HEAD, **keywords}

    with pytest.raises(error, match=match):
        store.pipeline.perturbation(**call)

    assert _written(store) == before


def test_label_settings_need_controls_wherever_they_are_checked() -> None:
    # The settings check names the missing controls, as the recipe does,
    # rather than looking for a default that labels do not have.
    with pytest.raises(ValueError, match=re.escape(LABELS_NEED_CONTROLS)):
        stage_settings("labels", {"policy": "single"}, optional=False)


def test_controls_are_a_required_keyword(rejection_store) -> None:
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        rejection_store.pipeline.perturbation(**HEAD)


def test_a_read_only_store_runs_nothing(readonly) -> None:
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        readonly.pipeline.perturbation(labels=LABELS, **HEAD)


def test_cells_that_the_guide_assay_did_not_measure_are_refused(
    screen, tmp_path
) -> None:
    location = tmp_path / "screen.zarr"
    shutil.copytree(screen, location)
    # The guide assay measured all cells but the first ten.
    cells = zarr.open_group(str(location / "cellData"), mode="r+")
    measured = cells.create_array(
        "CRISPR_I", data=np.arange(N_CELLS) >= 10, chunks=(N_CELLS,)
    )
    measured.attrs.update(membership_attributes("CRISPR"))
    store = _open(location)
    before = _written(store)

    with pytest.raises(UnmeasuredCellsError) as raised:
        store.pipeline.perturbation(labels=LABELS, **HEAD)

    assert (raised.value.assay, raised.value.operation) == (
        "CRISPR",
        "pipeline.perturbation",
    )
    assert _written(store) == before


def test_a_path_object_names_the_call_file(readonly, tmp_path) -> None:
    path = _call_file(tmp_path / "calls.csv")
    plan = _resolve(readonly, guide_calling={"path": path, "barcode_prefix": None})

    assert plan.config["stages"]["guide_calling"] == {
        "path": os.fspath(path),
        "barcode_prefix": None,
    }
    with pytest.raises(TypeError, match="barcode_prefix must be text or None"):
        _resolve(readonly, guide_calling={"path": path, "barcode_prefix": 1})


def test_only_the_runs_cells_need_labels_and_covariates(readonly) -> None:
    # Every column with gaps fills the cells of FILLED, so a run of those cells
    # passes the checks that a run of every cell fails.
    plan = _resolve(
        readonly,
        cells="filled",
        guide_calling={"sample_column": "lane_blank"},
        signature={"dims": 5, "sample_column": "lane_gaps"},
        mixscape={"split_column": "lane_blank"},
        associations={
            "sample_column": "lane_gaps",
            "covariates": ["depth_gaps", "depth_inf"],
        },
        clusters=True,
        profiles={"sample_column": "lane_blank"},
        **HEAD,
    )

    assert plan.config["cells"] == {"column": "filled"}
    assert plan.config["stages"]["associations"] == {
        "sample_column": "lane_gaps",
        "covariates": ["depth_gaps", "depth_inf"],
    }


def test_a_blank_capture_is_a_label_and_a_blank_sample_is_none() -> None:
    labels = np.asarray(["L1", "", "  ", "L2", "L1"])
    missing = np.asarray([False, False, False, True, False])
    active = np.asarray([True, True, True, True, False])
    # pipeline.rna takes a capture as written, blank text included.
    with pytest.raises(ValueError, match="but 1 of the run's cells has none"):
        require_labeled_cells(
            labels, missing, active, "lane", purpose="Cell calling per capture"
        )
    require_labeled_cells(
        labels, None, active, "lane", purpose="Cell calling per capture"
    )
    # The perturbation recipe refuses it, as the samples of its methods do.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Guide calling per sample needs a 'lane' label for every cell, but 3 of "
            "the run's cells have none"
        ),
    ):
        require_labeled_cells(
            labels,
            missing,
            active,
            "lane",
            purpose="Guide calling per sample",
            blank_is_label=False,
        )


# Summaries of stored arrays ---------------------------------------------------


def _group(arrays: dict[str, Any], **attributes: Any) -> zarr.Group:
    """Return an in-memory group of ``arrays`` with ``attributes``."""
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    for name, values in arrays.items():
        group.create_array(name, data=np.asarray(values))
    group.attrs.update(attributes)
    return group


def test_summaries_of_artifacts_without_their_arrays_are_empty() -> None:
    empty = _group({})
    for summarize in (
        summaries._guide_thresholds,
        summaries._guide_calls,
        summaries._perturbation_labels,
        summaries._efficiency,
        summaries._signature,
        summaries._profiles,
        summaries._perturbation_clusters,
    ):
        assert summarize(empty) == {}
    for summarize in (
        summaries._edistance,
        summaries._mixscape,
        summaries._associations,
    ):
        assert summarize(empty, {}) == {}
    # Fits without a sample axis summarize nothing.
    assert summaries._guide_thresholds(_group({"status": [0, 1]})) == {}


def test_summaries_name_what_their_tables_lack() -> None:
    # Labels of control cells only have no target and no cells per target.
    labels = summaries._perturbation_labels(
        _group(
            {
                "n_targets": np.zeros(3, dtype=np.int32),
                "is_control": [True, True, False],
                "target_codes": np.full((3, 1), -1, dtype=np.int32),
                "targets": np.asarray(["NT"]),
                "target_is_control": [True],
            }
        )
    )
    assert labels == {
        "cells": 3,
        "controls": 2,
        "negative": 1,
        "multipleTargets": 0,
        "targets": 0,
        "targetCells": None,
        "fewCellTargets": {},
        "fewCellTargetCount": 0,
    }
    # Targets under 25 cells are named, fewest first.
    labels = summaries._perturbation_labels(
        _group(
            {
                "n_targets": np.ones(6, dtype=np.int32),
                "is_control": np.zeros(6, dtype=bool),
                "target_codes": np.asarray([[0], [0], [1], [2], [2], [2]], np.int32),
                "targets": np.asarray(["B", "A", "C"]),
                "target_is_control": np.zeros(3, dtype=bool),
            }
        )
    )
    assert labels["fewCellTargets"] == {"A": 1, "B": 2, "C": 3}
    assert labels["fewCellTargetCount"] == 3
    # A table without a tested row has no median fraction, and names the
    # targets without a feature.
    efficiency = summaries._efficiency(
        _group(
            {
                "status": [1, 2],
                "names": np.asarray(["GENE_X", "GENE_Y"]),
                "remaining_fraction": [np.nan, np.nan],
                "adjusted_p_value": [np.nan, np.nan],
            },
            status_names=["tested", "no_feature", "no_cells", "no_control_expression"],
            by="target",
        )
    )
    assert efficiency["medianRemainingFraction"] is None
    assert efficiency["noFeature"] == ["GENE_X"]
    # Without permutations no p-value sits at the floor.
    distances = summaries._edistance(
        _group(
            {
                "status": [0],
                "p_value": [np.nan],
                "adjusted_p_value": [np.nan],
            },
            status_names=["tested", "too_few_cells"],
            by="target",
        ),
        {"n_permutations": 0},
    )
    assert (distances["significant"], distances["atFloor"]) == (0, 0)


def test_false_discoveries_are_counted_by_their_control_guides() -> None:
    # Two discovery pairs and five calibration pairs of three control groups,
    # three of them false discoveries, two in one group.
    group = _group(
        {
            "sets": [0, 0, 1, 1, 1, 1, 1, 2],
            "pass_qc": [True, False, True, True, True, True, True, True],
            "significant": [True, False, True, True, False, True, False, True],
            "groups": [0, 0, 1, 1, 2, 3, 3, 0],
            "group_guides": np.asarray(["", "nt1+nt2", "nt3", "nt4"]),
        },
        set_names=["discovery", "calibration", "positive"],
        n_false_discoveries=3,
        calibration_grade="mild",
        calibration_status="checked",
    )

    summary = summaries._associations(group, {"alpha": 0.1})

    assert summary == {
        "discoveryPairs": 2,
        "testedDiscoveryPairs": 1,
        "discoveries": 1,
        "calibrationPairs": 5,
        "falseDiscoveries": 3,
        "calibrationGrade": "mild",
        "calibrationStatus": "checked",
        "falseDiscoveriesByControlGuides": {"nt1+nt2": 2, "nt4": 1},
        "positivePairs": 1,
        "positiveDetected": 1,
        "alpha": 0.1,
    }
    assert list(summary["falseDiscoveriesByControlGuides"]) == ["nt1+nt2", "nt4"]
