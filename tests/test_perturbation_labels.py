"""Perturbation labels of the cells of a nine-cell screen.

The module writes the screen and its guide calls once, as a template. A test
that writes copies the template, and the tests of refused calls share one
copy, which they check is left as it was.
"""

from collections.abc import Iterator
from types import SimpleNamespace

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import CONTROL, MULTIPLE, NEGATIVE, PerturbationLabels
from cytearc.perturbation.labels import (
    POLICIES,
    guide_targets,
    label_memory_bytes,
    longest_label,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from cytearc.storage.artifacts import inspect_artifact
from cytearc.storage.selections import read_stored_selection_indices
from tests.perturbation_helpers import (
    copy_store,
    open_store,
    record_cell_reads,
    check_budget,
    write_screen,
    written,
)

# Guides CRISPR0 and CRISPR1 target GENE_A, CRISPR2 targets GENE_B, and
# CRISPR3 and CRISPR4 are controls. Every count of 20 is called at min_umi=10.
COUNTS = np.asarray(
    [
        [20, 0, 0, 0, 0],  # cell0: GENE_A
        [20, 20, 0, 0, 0],  # cell1: two guides of GENE_A
        [20, 0, 0, 20, 0],  # cell2: GENE_A and a control guide
        [0, 0, 0, 20, 0],  # cell3: a control guide only
        [0, 0, 0, 0, 3],  # cell4: no call
        [20, 0, 20, 0, 0],  # cell5: GENE_A and GENE_B
        [0, 0, 20, 20, 0],  # cell6: GENE_B and a control guide
        [0, 0, 0, 20, 20],  # cell7: two control targets
        [0, 20, 20, 0, 0],  # cell8: GENE_B and GENE_A
    ],
    dtype=np.uint16,
)
RNA = np.random.default_rng(7).poisson(2.0, size=(len(COUNTS), 10))
TARGETS = np.asarray(["GENE_A", "GENE_A", "GENE_B", "Non-Targeting", "Scrambled"])
CONTROLS = ["Non-Targeting", "Scrambled"]
SINGLE = [
    "GENE_A",
    "GENE_A",
    "GENE_A",
    CONTROL,
    NEGATIVE,
    MULTIPLE,
    "GENE_B",
    CONTROL,
    MULTIPLE,
]


def _calls(store: DataStore):
    cells = store.select_measured_cells("CRISPR")
    return store.perturbation.call_guides(cells, method="fixed", min_umi=10)


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen and its guide calls, which every test starts from."""
    path = tmp_path_factory.mktemp("labels") / "store.zarr"
    store = write_screen(path, RNA, COUNTS, targets=TARGETS)
    screen = SimpleNamespace(path=path, calls=_calls(store))
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the calls that must be refused."""
    return copy_store(template.path, tmp_path_factory.mktemp("labels_refused"))


def test_labels_name_the_genes_of_each_cells_guides(store, template) -> None:
    calls = template.calls

    single = store.perturbation.labels(calls, controls=CONTROLS)
    combined = store.perturbation.labels(
        calls, controls=CONTROLS, policy="combinations"
    )

    assert single.kind == "perturbation_labels"
    loaded = store.perturbation.load_labels(single)
    assert isinstance(loaded, PerturbationLabels)
    assert loaded.cells["label"].tolist() == SINGLE
    assert loaded.cells["n_targets"].tolist() == [1, 1, 1, 0, 0, 2, 1, 0, 2]
    assert loaded.cells["is_control"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert loaded.cells["guide_label"].tolist()[:5] == [
        "CRISPR0",
        MULTIPLE,
        MULTIPLE,
        "CRISPR3",
        NEGATIVE,
    ]
    assert loaded.controls == ("Non-Targeting", "Scrambled")
    assert loaded.policy == "single"
    assert loaded.targets.groupby("ids")["target"].apply(list).to_dict() == {
        "cell0": ["GENE_A"],
        "cell1": ["GENE_A"],
        "cell2": ["GENE_A"],
        "cell5": ["GENE_A", "GENE_B"],
        "cell6": ["GENE_B"],
        "cell8": ["GENE_A", "GENE_B"],
    }
    assert loaded.guides.to_dict("list") == {
        "guide": [f"CRISPR{index}" for index in range(5)],
        "target": TARGETS.tolist(),
        "is_control": [False, False, False, True, True],
    }
    labels = store.perturbation.load_labels(combined).cells["label"].tolist()
    assert labels[5] == labels[8] == "GENE_A+GENE_B"
    assert [label for index, label in enumerate(labels) if index not in (5, 8)] == [
        label for index, label in enumerate(SINGLE) if index not in (5, 8)
    ]
    # The labels group cells as cluster labels do.
    chosen = store.select_cells(single, include=["GENE_A"])
    indices = read_stored_selection_indices(
        store.zw,
        chosen,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    assert indices.tolist() == [0, 1, 2]
    codes = store.artifacts.load_values(single, value="target_codes").values
    assert np.asarray(codes).tolist()[5] == [0, 1]


def test_imported_calls_get_labels_too(store, tmp_path) -> None:
    cells = store.select_measured_cells("CRISPR")
    path = tmp_path / "calls.csv"
    path.write_text(
        "cell_barcode,num_features,feature_call,num_umis\n"
        "cell0,1,CRISPR0,20\n"
        "cell5,2,CRISPR0|CRISPR2,20|20\n"
    )
    imported = store.perturbation.import_guide_calls(path, cell_selection=cells)
    ref = store.perturbation.labels(imported, controls=["Non-Targeting"])
    labels = store.perturbation.load_labels(ref).cells["label"].tolist()
    assert labels[:6] == ["GENE_A", NEGATIVE, NEGATIVE, NEGATIVE, NEGATIVE, MULTIPLE]


def test_targets_replace_the_column_for_the_guides_they_name(store, template) -> None:
    calls = template.calls
    ref = store.perturbation.labels(
        calls, controls=["Non-Targeting"], targets={"CRISPR4": "GENE_C"}
    )
    labels = store.perturbation.load_labels(ref).cells["label"].tolist()
    assert labels[7] == "GENE_C"
    only = store.perturbation.labels(
        calls,
        controls=["Non-Targeting"],
        target_column=None,
        targets=dict(
            zip(
                [f"CRISPR{index}" for index in range(5)],
                ["A", "A", "B", "Non-Targeting", "C"],
                strict=True,
            )
        ),
    )
    labels = store.perturbation.load_labels(only).cells["label"].tolist()
    assert labels[:4] == ["A", "A", "A", CONTROL]
    assert labels[7] == "C"


def test_labels_are_reused_and_follow_their_inputs(store, template) -> None:
    calls = template.calls
    ref = store.perturbation.labels(calls, controls=CONTROLS)
    # The order and repeats of the controls do not change the identity.
    same = store.perturbation.labels(
        calls, controls=["Scrambled", "Non-Targeting", "Scrambled"]
    )
    assert same == ref
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {
        "controls": ["Non-Targeting", "Scrambled"],
        "policy": "single",
    }
    assert set(status.inputs) == {
        "cell_selection",
        "guide_calls",
        "targets_fingerprint",
    }
    other = store.perturbation.labels(
        calls, controls=CONTROLS, targets={"CRISPR1": "GENE_C"}
    )
    assert other != ref
    fresh = store.perturbation.labels(calls, controls=CONTROLS, invalidate_cache=True)
    labels = store.perturbation.load_labels(fresh).cells["label"].tolist()
    assert labels == SINGLE


def test_read_only_store_reuses_labels_but_cannot_label(store, template) -> None:
    calls = template.calls
    ref = store.perturbation.labels(calls, controls=CONTROLS)
    reader = open_store(store.zarr_loc, zarr_mode="r")
    assert reader.perturbation.labels(calls, controls=CONTROLS) == ref
    with pytest.raises(PermissionError, match="label_perturbations"):
        reader.perturbation.labels(calls, controls=CONTROLS, policy="combinations")


def test_labels_read_the_cell_ids_only_to_load_them(
    store, template, monkeypatch
) -> None:
    reads = record_cell_reads(monkeypatch, store, {"ids"})

    labels = store.perturbation.labels(template.calls, controls=CONTROLS)
    # The cells of one guide alone, which a comparison by guide reads.
    by_guide = store.perturbation.efficiency(labels, by="guide", id_column=None)

    assert reads["ids"] == []
    assert store.perturbation.load_efficiency(by_guide).table["guide"].tolist() == [
        "CRISPR0",
        "CRISPR1",
        "CRISPR2",
    ]
    loaded = store.perturbation.load_labels(labels)
    assert loaded.cells["label"].tolist() == SINGLE
    assert [rows.tolist() for rows in reads["ids"]] == [list(range(len(COUNTS)))]


def test_loading_labels_checks_the_budget_of_every_array_at_once(
    store, template, monkeypatch
) -> None:
    labels = store.perturbation.labels(template.calls, controls=CONTROLS)

    def fits(name: str, budget: int) -> bool:
        monkeypatch.setattr(store, "memoryBytes", budget)
        try:
            store.artifacts.load_values(labels, value=name)
        except MemoryError:
            return False
        return True

    def smallest(name: str) -> int:
        low, high = 0, 1 << 30
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (low, middle) if fits(name, middle) else (middle, high)
        return high

    # A budget that reads any one array with the cell IDs, but not all of them.
    names = ("values", "guide_label", "is_control", "n_targets", "target_codes")
    monkeypatch.setattr(store, "memoryBytes", max(smallest(name) for name in names))

    with pytest.raises(MemoryError, match="memory budget"):
        store.perturbation.load_labels(labels)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"controls": "Non-Targeting"}, TypeError, "a sequence of target names"),
        ({"controls": [3]}, TypeError, "a sequence of target names"),
        ({"controls": []}, ValueError, "name at least one target"),
        ({"controls": ["Non-Targeting", " "]}, ValueError, "no name may be blank"),
        ({"target_column": 3}, TypeError, "target_column must be a feature column"),
        ({"target_column": "gene"}, ValueError, "'gene' is not a feature column"),
        ({"targets": {"CRISPR0": 1}}, TypeError, "targets must map guide IDs"),
        ({"target_column": None}, ValueError, "Pass target_column, targets, or both"),
        (
            {"policy": "high_moi"},
            ValueError,
            "policy must be 'single' or 'combinations'",
        ),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be a boolean"),
        ({"targets": {"CRISPR9": "GENE_C"}}, ValueError, "not guides of the assay"),
        ({"targets": {"CRISPR0": " "}}, ValueError, "an empty target"),
        ({"targets": {"CRISPR0": "control"}}, ValueError, "reserved"),
        (
            {"targets": {"CRISPR0": "A+B"}, "policy": "combinations"},
            ValueError,
            "must not hold '\\+'",
        ),
        ({"controls": ["Non-targeting"]}, ValueError, "'Non-targeting'"),
    ],
)
def test_labels_refuse_arguments_they_cannot_use(
    shared_store, template, arguments, error, message
) -> None:
    saved = written(shared_store)
    with pytest.raises(error, match=message):
        shared_store.perturbation.labels(
            template.calls, **({"controls": CONTROLS} | arguments)
        )
    assert written(shared_store) == saved


def test_labels_refuse_calls_they_cannot_label(tmp_path, monkeypatch) -> None:
    store = write_screen(
        tmp_path / "store.zarr",
        RNA,
        COUNTS,
        targets=["GENE_A", "", "GENE_B", "Non-Targeting", None],
    )
    calls = _calls(store)
    with pytest.raises(TypeError, match="calls must be an ArtifactRef"):
        store.perturbation.labels("calls", controls=["Non-Targeting"])
    cells = store.select_measured_cells("CRISPR")
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.labels(cells, controls=["Non-Targeting"])
    # CRISPR1 and CRISPR4 are called but have no target in the column; a
    # read-only store says so too, rather than refusing to write.
    reader = open_store(store.zarr_loc, zarr_mode="r")
    for source in (store, reader):
        with pytest.raises(
            ValueError,
            match="2 called guides have no target, for example 'CRISPR1', 'CRISPR4'",
        ):
            source.perturbation.labels(calls, controls=["Non-Targeting"])
    targets = {"CRISPR1": "GENE_A", "CRISPR4": "Non-Targeting"}
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Labelling 9 cells"):
        store.perturbation.labels(calls, controls=["Non-Targeting"], targets=targets)
    assert not store.artifacts.list(kind="perturbation_labels")


def test_loading_labels_refuses_other_artifacts(shared_store, template) -> None:
    saved = written(shared_store)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        shared_store.perturbation.load_labels("labels")
    with pytest.raises(ValueError, match="not 'guide_calls'"):
        shared_store.perturbation.load_labels(template.calls)
    assert written(shared_store) == saved


def test_label_arrays_without_targets_keep_one_column() -> None:
    targets = guide_targets(
        np.asarray(["g0", "g1"]),
        np.asarray([np.nan, "Non-Targeting"], dtype=object),
        None,
        ("Non-Targeting",),
        policy="single",
    )
    assert targets.guide_codes.tolist() == [-1, 0]
    assert targets.guide_names().tolist() == ["", "Non-Targeting"]
    arrays = perturbation_label_arrays(
        np.asarray([[1], [-1]]), targets, policy="single"
    )
    assert arrays.values.tolist() == [CONTROL, NEGATIVE]
    assert arrays.target_codes.tolist() == [[-1], [-1]]
    # g0 has no target, so a cell that carries it cannot be labelled.
    with pytest.raises(ValueError, match="1 called guides have no target"):
        perturbation_label_arrays(np.asarray([[0], [1]]), targets, policy="single")
    cells, per_target, guides = perturbation_label_frames(
        np.asarray(["a", "b"]),
        values=arrays.values,
        guide_label=np.asarray(["g1", NEGATIVE]),
        is_control=arrays.is_control,
        n_targets=arrays.n_targets,
        target_codes=arrays.target_codes,
        target_names=targets.names,
        target_is_control=targets.is_control,
        guide_ids=np.asarray(["g0", "g1"]),
        guide_target_codes=targets.guide_codes,
    )
    assert cells["label"].tolist() == [CONTROL, NEGATIVE]
    assert per_target.empty
    assert guides["target"].tolist() == [None, "Non-Targeting"]
    assert guides["is_control"].tolist() == [False, True]
    assert isinstance(cells, pd.DataFrame)


def test_a_control_target_may_take_a_reserved_label(store, template) -> None:
    # The control guides target "control" and "Negative": control targets
    # never become labels, so a cell with control guides only is "control".
    reserved = {"CRISPR3": "control", "CRISPR4": "Negative"}
    labels = store.perturbation.labels(
        template.calls, controls=["control", "Negative"], targets=reserved
    )
    loaded = store.perturbation.load_labels(labels)
    assert loaded.cells["label"].tolist() == SINGLE
    assert loaded.guides["target"].tolist()[3:] == ["control", "Negative"]
    assert loaded.guides["is_control"].tolist() == [False] * 3 + [True] * 2
    # A target that is not a control still may not take a reserved label.
    with pytest.raises(
        ValueError, match="'Negative'. Give their guides other targets with targets"
    ):
        store.perturbation.labels(
            template.calls, controls=["control"], targets=reserved
        )


def test_a_control_target_may_hold_a_plus_under_combinations() -> None:
    # The control target "Non+Targeting" never joins the genes of a
    # combination: a cell with its guide only is "control", and a cell's
    # genes are joined without it.
    ids = np.asarray(["g0", "g1", "g2"])
    column = np.asarray(["GENE_A", "GENE_B", "Non+Targeting"], dtype=object)
    targets = guide_targets(
        ids, column, None, ("Non+Targeting",), policy="combinations"
    )
    codes = np.asarray([[2, -1, -1], [0, 2, -1], [1, 2, 0]])
    arrays = perturbation_label_arrays(codes, targets, policy="combinations")
    assert arrays.values.tolist() == [CONTROL, "GENE_A", "GENE_A+GENE_B"]
    # A target that is not a control still may not hold "+".
    with pytest.raises(ValueError, match="targets that are not controls must not"):
        guide_targets(ids, column, None, ("GENE_A",), policy="combinations")


def _own_sets(codes: np.ndarray, targets, policy: str) -> list[tuple]:
    """Each cell's label, control flag and sorted genes, from its own guides."""
    names = list(targets.names)
    found = []
    for cell in codes:
        called = [int(guide) for guide in cell if guide >= 0]
        target_of = [int(targets.guide_codes[guide]) for guide in called]
        controls = [code for code in target_of if targets.is_control[code]]
        genes = sorted({code for code in target_of if not targets.is_control[code]})
        if len(genes) == 1:
            label = names[genes[0]]
        elif genes:
            joined = "+".join(names[gene] for gene in genes)
            label = MULTIPLE if policy == "single" else joined
        else:
            label = CONTROL if controls else NEGATIVE
        found.append((label, bool(controls) and not genes, genes))
    return found


def test_label_arrays_equal_each_cells_own_sets() -> None:
    # Guides 0 to 7 target four genes in pairs, in an order that does not sort
    # them; guides 8 and 9 are controls of two control targets.
    column = np.asarray(
        ["GENE_D", "GENE_D", "GENE_B", "GENE_B", "GENE_C", "GENE_C"]
        + ["GENE_A", "GENE_A", "Non-Targeting", "Scrambled"],
        dtype=object,
    )
    ids = np.asarray([f"g{index}" for index in range(len(column))])
    rng = np.random.default_rng(12)
    codes = np.full((600, 7), -1, dtype=np.int32)
    for cell in range(len(codes)):
        called = rng.choice(len(column), rng.integers(0, 8), replace=False)
        codes[cell, : len(called)] = called
    for policy in POLICIES:
        targets = guide_targets(
            ids, column, None, ("Non-Targeting", "Scrambled"), policy=policy
        )
        arrays = perturbation_label_arrays(codes, targets, policy=policy)
        expected = _own_sets(codes, targets, policy)
        assert arrays.values.tolist() == [label for label, _, _ in expected]
        assert arrays.is_control.tolist() == [flag for _, flag, _ in expected]
        assert arrays.n_targets.tolist() == [len(genes) for _, _, genes in expected]
        assert arrays.target_codes.dtype == np.int32
        assert arrays.target_codes.tolist() == [
            genes + [-1] * (arrays.target_codes.shape[1] - len(genes))
            for _, _, genes in expected
        ]
        assert arrays.target_codes.shape[1] == 4


@pytest.mark.parametrize("policy", POLICIES)
def test_the_memory_estimate_bounds_the_labels_peak(policy) -> None:
    # 10,000 cells of up to 40 called guides of 30 genes and 4 controls, so
    # that the entries take the most, and under "combinations" a cell's label
    # joins up to 30 genes.
    n_cells, width, n_guides = 10_000, 40, 64
    rng = np.random.default_rng(13)
    ids = np.asarray([f"g{index}" for index in range(n_guides)])
    names = [f"GENE{index // 2}" for index in range(n_guides - 4)]
    column = np.asarray(names + ["Non-Targeting"] * 4, dtype=object)
    targets = guide_targets(ids, column, None, ("Non-Targeting",), policy=policy)
    order = np.argsort(rng.random((n_cells, n_guides)), axis=1)
    codes = order[:, :width].astype(np.int32)
    codes[np.arange(width) >= rng.integers(0, width + 1, n_cells)[:, None]] = -1
    # The first call creates what later calls reuse.
    perturbation_label_arrays(codes[:10], targets, policy=policy)
    tracemalloc.start()
    try:
        perturbation_label_arrays(codes, targets, policy=policy)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    estimate = label_memory_bytes(
        n_cells,
        width,
        label_length=longest_label(targets, width, policy=policy),
        joined=policy == "combinations",
    )
    assert peak <= estimate


@pytest.fixture(scope="module")
def wide_calls(tmp_path_factory) -> SimpleNamespace:
    """4,000 cells of one guide each, and 20 cells of 40 guides of 30 genes."""
    n_cells, n_guides = 4_000, 64
    rng = np.random.default_rng(11)
    guides = np.zeros((n_cells, n_guides), dtype=np.uint16)
    guides[np.arange(n_cells), rng.integers(0, n_guides, n_cells)] = 20
    for cell in range(0, n_cells, 200):
        guides[cell, rng.choice(n_guides, 40, replace=False)] = 20
    targets = [f"GENE{index // 2}" for index in range(n_guides - 4)]
    path = tmp_path_factory.mktemp("wide_labels") / "store.zarr"
    store = write_screen(
        path,
        rng.poisson(2.0, size=(n_cells, 3)) + 1,
        guides,
        targets=targets + ["Non-Targeting"] * 4,
    )
    return SimpleNamespace(path=path, calls=_calls(store))


@pytest.mark.slow
@pytest.mark.parametrize("policy", POLICIES)
def test_labelling_holds_no_more_than_the_budget_it_names(
    wide_calls, tmp_path, policy
) -> None:
    path = copy_store(wide_calls.path, tmp_path).zarr_loc

    def label(store, invalidate_cache=True):
        return store.perturbation.labels(
            wide_calls.calls,
            controls=["Non-Targeting"],
            policy=policy,
            invalidate_cache=invalidate_cache,
        )

    # One refusal names the budget that runs, which holds the whole call.
    _, refusals = check_budget(path, label)
    assert len(refusals) == 1
    assert refusals[0].startswith("Labelling 4000 cells holds about")
    reused = open_store(path)
    loaded = reused.perturbation.load_labels(label(reused, invalidate_cache=False))
    assert loaded.cells["n_targets"].max() > 20
