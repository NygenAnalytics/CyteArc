"""The description of each recipe's stage keywords, their settings, and the defaults that the recipe uses.

The description needs no store. Its defaults are checked against the public
methods' signatures, and the defaults that a recipe sets itself against the
settings that plans on small synthetic stores record.
"""

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytearc import DataStore
from cytearc.datastore import namespaces
from cytearc.datastore._pipeline import description
from cytearc.datastore._pipeline.filtering import FILTERING_SETTINGS
from cytearc.datastore._pipeline.settings import STAGE_SETTINGS
from cytearc.datastore.pipeline_accessor import (
    DEFAULT_PANEL_EXCLUDE,
    PipelineAccessor,
    RecipeDescription,
    SettingDescription,
    StageDescription,
)
from tests.perturbation_helpers import write_screen
from tests.storage_helpers import open_store, write_count_store

RECIPES = ("rna", "atac", "adt", "multimodal", "perturbation")
# The keywords of the recipes' methods that are not stages.
_NOT_STAGES = frozenset(
    {
        "self",
        "assay",
        "guide_assay",
        "label",
        "cells",
        "runs",
        "method",
        "snapshot_columns",
        "species",
        "callback",
        "cancel",
    }
)
# The public methods that each stage passes its settings to: filtering's
# manual bounds go to qc.filter, and guide calling imports Cell Ranger's
# calls with perturbation.import_guide_calls. The panel's only setting
# configures the recipe.
_METHODS: dict[str, tuple[str, ...]] = {
    "cell_calling": ("qc.call_cells",),
    "hashing": ("qc.hto_demultiplexing",),
    "filtering": ("qc.auto_filter", "qc.filter"),
    "cell_cycle": ("scores.cell_cycle",),
    "hvg": ("features.hvgs",),
    "prevalent_peaks": ("features.prevalent_peaks",),
    "panel": (),
    "normalization": ("features.normalize",),
    "pca": ("reduction.pca",),
    "lsi": ("reduction.lsi",),
    "harmony": ("reduction.harmony",),
    "ann_index": ("graph.ann_index",),
    "neighbors": ("graph.neighbors",),
    "connectivity": ("graph.connectivity",),
    "integration": ("integration.modalities",),
    "embedding_initialization": ("embeddings.initialization",),
    "umap": ("embeddings.umap",),
    "tsne": ("embeddings.tsne",),
    "leiden": ("clusters.leiden",),
    "membership_strength": ("clusters.membership_strength",),
    "paris": ("clusters.paris",),
    "doublets": ("qc.doublets",),
    "markers": ("markers.search",),
    "guide_calling": ("perturbation.call_guides", "perturbation.import_guide_calls"),
    "labels": ("perturbation.labels",),
    "efficiency": ("perturbation.efficiency",),
    "edistance": ("perturbation.edistance",),
    "signature": ("perturbation.signature",),
    "mixscape": ("perturbation.mixscape",),
    "associations": ("perturbation.associations",),
    "profiles": ("perturbation.profiles",),
    "clusters": ("perturbation.cluster",),
}
_ACCESSORS = {
    "qc": namespaces.QcAccessor,
    "scores": namespaces.ScoresAccessor,
    "features": namespaces.FeaturesAccessor,
    "reduction": namespaces.ReductionAccessor,
    "graph": namespaces.GraphAccessor,
    "integration": namespaces.IntegrationAccessor,
    "embeddings": namespaces.EmbeddingsAccessor,
    "clusters": namespaces.ClustersAccessor,
    "markers": namespaces.MarkersAccessor,
    "perturbation": namespaces.PerturbationAccessor,
}
_RNA_COLUMNS = (
    "<assay>_nCounts, <assay>_nFeatures, <assay>_percentMito and <assay>_percentRibo"
)
_WITHOUT_TOTALS = ", without <assay>_nCounts for an assay of normalized values"
_MANUAL_BOUNDS = "required with method 'manual', one bound for each column of attrs"
_ALL = "*"
# The settings whose type, default or rule the recipe sets itself, by
# recipe ("*" for every recipe with the stage), stage keyword and setting:
# (method, annotation, default, rule).
_OWN: dict[tuple[str, str, str], tuple[str | None, str | None, Any, str | None]] = {
    (_ALL, "cell_calling", "sample_column"): (None, "str | None", None, None),
    (_ALL, "hashing", "assay"): (
        None,
        "str | None",
        None,
        "the store's only assay of type HTO",
    ),
    (_ALL, "filtering", "method"): (
        None,
        "Literal['auto', 'mad', 'gaussian', 'manual']",
        "auto",
        None,
    ),
    ("rna", "filtering", "attrs"): (
        "qc.auto_filter",
        "collections.abc.Sequence[str]",
        None,
        f"the RNA assay's QC columns that preparation wrote, {_RNA_COLUMNS}"
        f"{_WITHOUT_TOTALS}",
    ),
    ("perturbation", "filtering", "attrs"): (
        "qc.auto_filter",
        "collections.abc.Sequence[str]",
        None,
        f"the RNA assay's QC columns that preparation wrote, {_RNA_COLUMNS}"
        f"{_WITHOUT_TOTALS}",
    ),
    ("atac", "filtering", "attrs"): (
        "qc.auto_filter",
        "collections.abc.Sequence[str]",
        None,
        "the ATAC assay's QC columns that preparation wrote, <assay>_nCounts and "
        f"<assay>_nFeatures{_WITHOUT_TOTALS}",
    ),
    ("adt", "filtering", "attrs"): (
        "qc.auto_filter",
        "collections.abc.Sequence[str]",
        None,
        f"the ADT assay's QC columns that preparation wrote, <assay>_nCounts"
        f"{_WITHOUT_TOTALS}",
    ),
    (_ALL, "filtering", "lows"): (
        "qc.filter",
        "collections.abc.Iterable[float | None]",
        None,
        _MANUAL_BOUNDS,
    ),
    (_ALL, "filtering", "highs"): (
        "qc.filter",
        "collections.abc.Iterable[float | None]",
        None,
        _MANUAL_BOUNDS,
    ),
    (_ALL, "hvg", "batch"): ("features.hvgs", "str | None", None, None),
    ("atac", "prevalent_peaks", "top_n"): (
        "features.prevalent_peaks",
        "int",
        None,
        "the smaller of 10000 and one fewer than the assay's peaks",
    ),
    (_ALL, "panel", "exclude"): (None, "str | None", DEFAULT_PANEL_EXCLUDE, None),
    ("adt", "pca", "dims"): (
        "reduction.pca",
        "int",
        None,
        "the smaller of 21 and one fewer than the panel's antibodies",
    ),
    (_ALL, "leiden", "partitions"): (
        None,
        "collections.abc.Sequence[float]",
        (0.5, 0.75, 1.0, 1.25),
        None,
    ),
    ("multimodal", "leiden", "partitions"): (
        None,
        "collections.abc.Sequence[float]",
        (1.0,),
        None,
    ),
    (_ALL, "leiden", "selected"): (None, "float | None", None, None),
    ("rna", "doublets", "sample_column"): (
        "qc.doublets",
        "str | None",
        None,
        "the Harmony batch_columns when the run corrects batches with Harmony, "
        "else None",
    ),
    (_ALL, "guide_calling", "path"): (
        "perturbation.import_guide_calls",
        "str | os.PathLike[str] | None",
        None,
        None,
    ),
    (_ALL, "associations", "positive_controls"): (
        "perturbation.associations",
        "Literal['targets'] | None",
        "targets",
        None,
    ),
    (_ALL, "profiles", "cells"): (
        None,
        "Literal['knocked_out', 'labelled']",
        None,
        "'knocked_out' when mixscape runs, else 'labelled'",
    ),
}
# Whether each stage runs when its keyword is omitted, where that differs
# from its default in the recipe's signature, or the rule that decides.
_RUNS: dict[tuple[str, str], bool | str] = {
    (_ALL, "harmony"): False,
    ("rna", "filtering"): (
        "when cells is a cell column, or when cell calling or hashing runs"
    ),
    ("atac", "filtering"): "when cells is a cell column",
    ("perturbation", "filtering"): "when cells is a cell column",
    ("adt", "filtering"): False,
    ("perturbation", "signature"): "when mixscape runs",
    ("perturbation", "profiles"): "when clusters runs",
}


def _own(table: dict[tuple[str, ...], Any], recipe: str, *key: str) -> Any:
    return table.get((recipe, *key), table.get((_ALL, *key)))


def _public_parameter(method: str, name: str) -> inspect.Parameter:
    """Return the parameter ``name`` of the public method ``method``, such as ``features.hvgs``."""
    namespace, _dot, attribute = method.partition(".")
    signature = inspect.signature(getattr(_ACCESSORS[namespace], attribute))
    return signature.parameters[name]


def _stage_keywords(recipe: str) -> list[tuple[str, inspect.Parameter]]:
    signature = inspect.signature(getattr(PipelineAccessor, recipe))
    return [
        (name, parameter)
        for name, parameter in signature.parameters.items()
        if name not in _NOT_STAGES
    ]


@pytest.mark.parametrize("recipe", RECIPES)
def test_each_stage_keyword_is_described_with_every_setting_and_its_default(recipe):
    described = PipelineAccessor.describe(recipe)

    assert isinstance(described, RecipeDescription) and described.recipe == recipe
    keywords = _stage_keywords(recipe)
    assert [stage.keyword for stage in described.stages] == [
        name for name, _parameter in keywords
    ]
    for (keyword, parameter), stage in zip(keywords, described.stages, strict=True):
        assert isinstance(stage, StageDescription)
        assert stage.methods == _METHODS[keyword], keyword
        names = (
            FILTERING_SETTINGS if keyword == "filtering" else STAGE_SETTINGS[keyword]
        )
        assert sorted(setting.name for setting in stage.settings) == sorted(names)
        runs = _own(_RUNS, recipe, keyword)
        if runs is None:
            # A stage that takes a switch runs as its default says, and one
            # that takes settings, or a mapping that a call must give, runs.
            default = parameter.default
            runs = default if isinstance(default, bool) else True
        expected = (runs, None) if isinstance(runs, bool) else (None, runs)
        assert (stage.runs, stage.rule) == expected, keyword
        for setting in stage.settings:
            assert isinstance(setting, SettingDescription)
            own = _own(_OWN, recipe, keyword, setting.name)
            if own is not None:
                method, annotation, default, rule = own
                assert (
                    setting.method,
                    setting.annotation,
                    setting.default,
                    setting.rule,
                    setting.required,
                ) == (method, annotation, default, rule, False), setting.name
                continue
            # Every other setting is the keyword of the stage's first method
            # that takes it, with that method's annotation and default.
            assert setting.method is not None
            assert setting.method == next(
                method
                for method in stage.methods
                if setting.name in _public_signature_names(method)
            )
            public = _public_parameter(setting.method, setting.name)
            assert setting.annotation == inspect.formatannotation(public.annotation)
            assert setting.rule is None
            if public.default is inspect.Parameter.empty:
                assert (setting.required, setting.default) == (True, None)
            else:
                assert not setting.required
                assert setting.default == public.default, setting.name


def _public_signature_names(method: str) -> set[str]:
    namespace, _dot, attribute = method.partition(".")
    return set(inspect.signature(getattr(_ACCESSORS[namespace], attribute)).parameters)


def test_the_settings_of_a_stage_list_recipe_keys_first_then_each_method_in_order():
    rna = PipelineAccessor.describe("rna")

    leiden = [setting.name for setting in rna["leiden"].settings]
    assert leiden == [
        "partitions",
        "selected",
        "backend",
        "symmetric_graph",
        "graph_upper_only",
        "random_seed",
    ]
    filtering = [setting.name for setting in rna["filtering"].settings]
    assert filtering[:2] == ["method", "attrs"]
    assert filtering[-3:] == ["lows", "highs", "keep_bounds"]
    hvg_order = [
        name
        for name in inspect.signature(namespaces.FeaturesAccessor.hvgs).parameters
        if name in STAGE_SETTINGS["hvg"]
    ]
    assert [setting.name for setting in rna["hvg"].settings] == hvg_order
    # Settings that a call must give are marked, with no default.
    assert rna["harmony"]["batch_columns"].required
    perturbation = PipelineAccessor.describe("perturbation")
    assert perturbation["labels"]["controls"].required
    assert perturbation["labels"].runs is True


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_description_is_json_and_needs_no_store(recipe):
    described = PipelineAccessor.describe(recipe)

    values = described.to_dict()

    assert json.loads(json.dumps(values, allow_nan=False)) == values
    assert [stage["keyword"] for stage in values["stages"]] == [
        stage.keyword for stage in described.stages
    ]
    # An accessor of no store describes the same recipe.
    assert PipelineAccessor(None).describe(recipe) == described


def test_defaults_that_json_cannot_hold_are_written_as_text():
    hvg = PipelineAccessor.describe("rna")["hvg"]

    assert (hvg["min_var"].default, hvg["max_mean"].default) == (-np.inf, np.inf)
    written = {setting["name"]: setting for setting in hvg.to_dict()["settings"]}
    assert written["min_var"]["default"] == "-inf"
    assert written["max_mean"]["default"] == "inf"
    associations = PipelineAccessor.describe("perturbation")["associations"]
    assert associations["n_resamples"].default == (499, 4999, 24999)
    assert associations.to_dict()["settings"][
        [setting.name for setting in associations.settings].index("n_resamples")
    ]["default"] == [499, 4999, 24999]
    assert description._json_value(np.float32(np.nan)) == "nan"
    with pytest.raises(TypeError, match="must be a JSON value"):
        description._json_value(object())


def test_a_description_refuses_unknown_recipes_stages_and_settings():
    with pytest.raises(ValueError, match="recipe must be one of"):
        PipelineAccessor.describe("scanpy")
    with pytest.raises(ValueError, match="recipe must be one of"):
        PipelineAccessor.describe(None)
    rna = PipelineAccessor.describe("rna")
    with pytest.raises(KeyError, match="has no stage keyword 'lsi'"):
        rna["lsi"]
    with pytest.raises(KeyError, match="takes no setting 'k'"):
        rna["pca"]["k"]


def test_a_stage_setting_must_be_a_method_keyword_or_declare_its_default(
    monkeypatch,
):
    # A setting that no method of its stage takes, and that declares no
    # default or rule of the recipe, cannot be described.
    settings = {**STAGE_SETTINGS, "pca": STAGE_SETTINGS["pca"] | {"whiten"}}
    monkeypatch.setattr(description, "STAGE_SETTINGS", settings)

    with pytest.raises(KeyError, match=r"pca\['whiten'\] is no keyword"):
        PipelineAccessor.describe("rna")


# Plans on small stores record the defaults that the recipes fit.

_QUICK_RNA: dict[str, Any] = {
    "cell_cycle": False,
    "hvg": {"top_n": 20},
    "pca": {"dims": 3},
    "umap": False,
    "leiden": False,
    "doublets": False,
    "markers": False,
}


@pytest.mark.parametrize("n_peaks", [12, 400, 10_005])
def test_the_atac_peaks_follow_their_described_rule(tmp_path: Path, n_peaks: int):
    ids = [f"chr1:{index * 100}-{index * 100 + 50}" for index in range(n_peaks)]
    counts = np.random.default_rng(n_peaks).poisson(0.3, size=(30, n_peaks))
    write_count_store(
        str(tmp_path / "atac.zarr"),
        {"ATAC": counts},
        "uint16",
        feature_ids={"ATAC": ids},
    )
    store = DataStore(
        str(tmp_path / "atac.zarr"), default_assay="ATAC", min_features_per_cell=0
    )
    rule = PipelineAccessor.describe("atac")["prevalent_peaks"]["top_n"].rule

    plan = store.pipeline.plan(
        "atac", lsi={"dims": 3}, umap=False, leiden=False, markers=False
    )

    # The smaller of 10000 and one fewer than the assay's peaks.
    assert rule == "the smaller of 10000 and one fewer than the assay's peaks"
    stages = plan.config["stages"]
    assert stages["prevalent_peaks"] == {"top_n": min(10_000, n_peaks - 1)}
    # A default call filters the cells of the I column on the ATAC QC columns.
    assert stages["filtering"]["attrs"] == ["ATAC_nCounts", "ATAC_nFeatures"]


@pytest.mark.parametrize(("n_antibodies", "dims"), [(4, 2), (6, 4), (30, 21)])
def test_the_adt_dimensions_follow_their_described_rule(
    tmp_path: Path, n_antibodies: int, dims: int
):
    # The panel leaves out the isotype control, the last antibody.
    names = [f"CD{index}" for index in range(n_antibodies - 1)] + ["IgG1_isotype"]
    rng = np.random.default_rng(n_antibodies)
    write_count_store(
        str(tmp_path / "adt.zarr"),
        {
            "RNA": rng.poisson(1.0, size=(40, 30)),
            "ADT": rng.poisson(5.0, size=(40, n_antibodies)),
        },
        "uint16",
        feature_ids={"ADT": names},
    )
    store = open_store(tmp_path / "adt.zarr")
    described = PipelineAccessor.describe("adt")

    plan = store.pipeline.plan(
        "adt", neighbors={"k": 3}, umap=False, leiden=False, markers=False
    )

    stages = plan.config["stages"]
    assert stages["panel"] == {
        "exclude": described["panel"]["exclude"].default,
        "excluded": ["IgG1_isotype"],
    }
    # The smaller of 21 and one fewer than the panel's antibodies.
    assert stages["pca"] == {"dims": min(21, (n_antibodies - 1) - 1)} == {"dims": dims}
    # An ADT run filters only when asked.
    assert described["filtering"].runs is False
    assert stages["filtering"] == {"enabled": False}


def test_the_rna_defaults_that_the_call_decides_follow_their_rules(tmp_path: Path):
    rng = np.random.default_rng(3)
    write_count_store(
        str(tmp_path / "rna.zarr"),
        {"RNA": rng.poisson(1.0, size=(40, 60)), "HTO": rng.poisson(5.0, (40, 3))},
        "uint16",
    )
    store = open_store(tmp_path / "rna.zarr")
    store.cells.insert("batch", np.where(np.arange(40) % 2 == 0, "a", "b"))
    store.cells.insert("chosen", np.arange(40) < 30)
    described = PipelineAccessor.describe("rna")
    graph = {**_QUICK_RNA, "leiden": True, "doublets": True}

    hashed = store.pipeline.plan("rna", hashing=True, **_QUICK_RNA).config
    harmony = store.pipeline.plan(
        "rna", harmony={"batch_columns": ["batch"]}, **graph
    ).config
    alone = store.pipeline.plan("rna", **graph).config
    chosen = store.pipeline.plan(
        "rna", cells=store.snapshot_cell_selection("chosen"), **_QUICK_RNA
    ).config

    # Hashing reads the store's only HTO assay.
    assert described["hashing"]["assay"].rule == "the store's only assay of type HTO"
    assert hashed["stages"]["hashing"] == {"assay": "HTO"}
    # Doublets are scored within the Harmony batches, and otherwise together.
    assert harmony["doubletSampleColumns"] == ["batch"]
    assert alone["doubletSampleColumns"] == []
    assert harmony["stages"]["leiden"]["partitions"] == list(
        described["leiden"]["partitions"].default
    )
    # Filtering runs for the cells of a column, here on the QC columns that
    # preparation wrote, and keeps a given selection's cells as they are.
    assert alone["stages"]["filtering"]["attrs"] == ["RNA_nCounts", "RNA_nFeatures"]
    assert chosen["stages"]["filtering"] == {"enabled": False}


@pytest.mark.parametrize(
    ("mixscape", "labels"), [(False, "perturbation_labels"), (True, "mixscape")]
)
def test_the_profiled_cells_follow_their_described_rule(
    tmp_path: Path, mixscape: bool, labels: str
):
    rng = np.random.default_rng(4)
    guides = np.zeros((80, 4), dtype=np.int64)
    guides[np.arange(80), np.arange(80) % 4] = 30
    store = write_screen(
        tmp_path / "screen.zarr",
        rng.poisson(2.0, size=(80, 40)),
        guides,
        targets=["GENE_A", "GENE_B", "NT", "NT"],
        target_ids=["RNA0", "RNA1", "", ""],
    )
    signature = {"signature": {"dims": 3}} if mixscape else {}

    plan = store.pipeline.plan(
        "perturbation",
        labels={"controls": ["NT"]},
        filtering=False,
        guide_calling={"method": "fixed", "min_umi": 10},
        hvg={"top_n": 20},
        pca={"dims": 3},
        mixscape=mixscape,
        clusters=True,
        **signature,
    )

    # Profiles average Mixscape's knocked-out cells when it runs, and every
    # cell of each label otherwise.
    cells = PipelineAccessor.describe("perturbation")["profiles"]["cells"]
    assert cells.rule == "'knocked_out' when mixscape runs, else 'labelled'"
    assert plan.config["profileLabels"] == labels
