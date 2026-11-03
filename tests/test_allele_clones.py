"""Clones called from imported recorder alleles recover the truth's clones.

The allele calling truth gives each of the truth's clones two rare DARLIN
alleles, some of which its synthetic bank lists, beside common alleles,
unedited arrays, CARLIN's placeholder and dropout, written as CARLIN pairs
per sample and array. Each method's rules must give back the clones that the
truth's design derives, with its homoplasic allele and the allele that joins
two clones removed by the rules that remove them. DARLIN's skull library of
Zenodo 11929508 (CC BY 4.0) checks the graph on real alleles.
"""

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from tests.clone_helpers import (
    import_allele_truth,
    loaded_partition,
    traced_steps,
    write_truth_store,
)
from tests.clone_truth import (
    ALLELE_NOISE,
    DARLIN_ARRAYS,
    N_CELLS,
    SAMPLES,
    allele_calling_truth,
    clone_truth,
)
from tests.memory_helpers import smallest_budget, traced_peak
from tests.storage_helpers import copy_store, open_store, write_count_store

TRUTH = clone_truth()
DATA = Path(__file__).parent / "data" / "recorders"
# DARLIN's rules on the synthetic bank.
DARLIN_RULES = {"max_frequency": 2e-5, "max_samples": 1, "max_alleles": 3}


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("allele_clones") / "store.zarr"
    write_truth_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    return copy_store(template, tmp_path)


def _imported(store: DataStore, folder: Path, noise=ALLELE_NOISE, **options):
    calling = allele_calling_truth(TRUTH, noise=noise)
    return calling, import_allele_truth(store, calling, TRUTH, folder, **options)


@pytest.mark.parametrize("method", ["graph", "p_clonal"])
@pytest.mark.parametrize("max_samples", [None, 1])
@pytest.mark.parametrize("max_alleles", [None, 3])
def test_each_method_recovers_the_clones_of_the_truths_rare_alleles(
    template, tmp_path, method, max_samples, max_alleles
) -> None:
    store = copy_store(template, tmp_path)
    calling, alleles = _imported(store, tmp_path)
    rules = {"max_frequency": 2e-5} if method == "graph" else {}
    clones = store.clones.allele_clones(
        alleles,
        method=method,
        frequencies=calling.banks,
        max_samples=max_samples,
        max_alleles=max_alleles,
        **rules,
    )
    expected = calling.expected_partition(
        TRUTH, max_samples=max_samples, max_alleles=max_alleles
    )
    assert loaded_partition(store, clones) == expected
    # max_samples=1 keeps the homoplasic allele from joining its two clones,
    # and max_alleles=3 leaves out the two that the bank's missing allele
    # joins, five rare alleles between them.
    if max_samples == 1 and max_alleles == 3:
        joined = set(TRUTH.cell_ids[np.isin(TRUTH.clones, (2, 5))].tolist())
        assert expected == {clone for clone in TRUTH.partition() if not clone & joined}


def test_without_noise_the_rules_give_every_clone_of_the_truth(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path, noise=())
    assert calling.expected_partition(TRUTH) == TRUTH.partition()
    graph = store.clones.allele_clones(
        alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
    )
    tested = store.clones.allele_clones(
        alleles, method="p_clonal", frequencies=calling.banks, max_alleles=3
    )
    assert loaded_partition(store, graph) == TRUTH.partition()
    assert loaded_partition(store, tested) == TRUTH.partition()


def test_the_artifact_records_its_rules_what_they_removed_and_the_banks_coverage(
    store, tmp_path
) -> None:
    calling, alleles = _imported(store, tmp_path)
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
    )
    record = store.artifacts.inspect(clones)
    assert record.operation == "call_allele_clones"
    assert {name: record.parameters[name] for name in DARLIN_RULES} == DARLIN_RULES
    assert record.parameters["method"] == "graph"
    assert record.parameters["fdr"] is None
    attributes = dict(store.artifacts.load(clones).attrs)
    # Counted from the truth's tables: each array's distinct edited alleles,
    # the bank's frequent ones and the homoplasic allele of three mice.
    arrays = calling.recorders.arrays
    edited = arrays[arrays["read"] & ~arrays["allele"].isin(["[]", "X"])]
    tests = edited.drop_duplicates(["integration", "allele"])
    banks = pd.concat(
        [bank.assign(integration=name) for name, bank in calling.banks.items()]
    )
    tested = tests.merge(banks, on=["integration", "allele"], how="left")
    frequent = int((tested["normalized_count"] >= 2e-5).sum())
    homoplasic = int(
        ((tested["normalized_count"] < 2e-5) & (tested["sample_count"] > 1)).sum()
    )
    counts = attributes["rule_counts"]
    assert counts["alleles"] == len(tests)
    assert counts["alleles_too_frequent"] == frequent > 0
    assert counts["alleles_in_too_many_samples"] == homoplasic == 1
    assert counts["rare_alleles"] == len(tests) - frequent - homoplasic
    assert counts["alleles_not_significant"] == 0
    # The two clones that the missing allele joins hold five rare alleles.
    joined = int(np.isin(TRUTH.clones, (2, 5)).sum())
    assert counts["clones_with_too_many_alleles"] == 1
    assert counts["cells_in_clones_with_too_many_alleles"] == joined
    coverage = attributes["bank_coverage"]
    assert [entry["character"] for entry in coverage] == sorted(DARLIN_ARRAYS)
    for entry in coverage:
        array = tested[tested["integration"] == entry["character"]]
        cells = edited[edited["integration"] == entry["character"]]
        listed = set(calling.banks[entry["character"]]["allele"])
        assert entry == {
            "character": entry["character"],
            "alleles": len(array),
            "alleles_in_bank": int(array["normalized_count"].notna().sum()),
            "observations": len(cells),
            "observations_in_bank": int(cells["allele"].isin(listed).sum()),
        }
    assert attributes["n_clones"] == len(
        calling.expected_partition(
            TRUTH,
            **{name: DARLIN_RULES[name] for name in ("max_samples", "max_alleles")},
        )
    )


def test_load_clones_shows_each_cells_rare_alleles(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
    )
    loaded = store.clones.load_clones(clones)
    assert loaded.source == "call_allele_clones"
    assert list(loaded.cells.columns) == ["clone", "clone_size", "n_alleles"]
    # Each cell's rare alleles of the design, the homoplasic one left out by
    # max_samples=1.
    arrays = calling.recorders.arrays
    is_rare = np.asarray(
        [
            allele in calling.rare[array] and allele != calling.homoplasic
            for array, allele in zip(
                arrays["integration"], arrays["allele"], strict=True
            )
        ]
    )
    rare = arrays[arrays["read"].to_numpy() & is_rare]
    expected = np.bincount(rare["cell"], minlength=N_CELLS)
    np.testing.assert_array_equal(loaded.cells["n_alleles"].to_numpy(), expected)
    # Cells of the two joined clones keep their rare alleles without a clone.
    left_out = np.isin(TRUTH.clones, (2, 5))
    assert loaded.cells["clone"].isna().to_numpy()[left_out].all()
    assert (loaded.cells["n_alleles"].to_numpy()[left_out] > 0).all()


def _pair(folder: Path, lines: dict[str, list[str]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "AlleleAnnotations.txt").write_text("\n".join(lines) + "\n")
    (folder / "AlleleColonies.txt").write_text(
        "\n".join(",".join(cells) for cells in lines.values()) + "\n"
    )
    return folder


def _spelled(number: int) -> str:
    """A number written in bases, as the insertion of an allele of its own."""
    bases = []
    while True:
        bases.append("ACGT"[number % 4])
        number //= 4
        if not number:
            return "".join(bases)


def _clones_of_their_own_alleles(
    folder: Path,
    n_cells: int,
    n_clones: int,
    clone_size: int,
    arrays: tuple[str, ...],
) -> tuple[DataStore, object]:
    """A store whose clones each hold an allele of their own on each array.

    The clones' cells are drawn at random on each array, so clones of more
    than one cell need a single array to stay apart.
    """
    rng = np.random.default_rng(24)
    barcodes = sorted(
        {"".join(row) for row in rng.choice(list("ACGT"), size=(n_cells + 5_000, 16))}
    )[:n_cells]
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=barcodes,
    )
    store = open_store(folder / "store.zarr")
    paths = []
    for index, array in enumerate(arrays):
        drawn = rng.permutation(n_cells)[: n_clones * clone_size]
        lines = {
            f"{2 + index}_{25 + index}del,60_61ins{_spelled(clone)}": [
                barcodes[cell] for cell in cells
            ]
            for clone, cells in enumerate(drawn.reshape(n_clones, clone_size).tolist())
        }
        paths.append(_pair(folder / array, lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            paths,
            cell_selection=store.snapshot_cell_selection(),
            integrations=list(arrays),
        )
    return store, alleles


def test_unedited_arrays_placeholders_and_missing_sites_never_link_cells(
    tmp_path,
) -> None:
    rng = np.random.default_rng(12)
    barcodes = sorted({"".join(rng.choice(list("ACGT"), size=16)) for _ in range(12)})
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(barcodes), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=barcodes,
    )
    store = open_store(tmp_path / "store.zarr")
    # Six cells are unedited, three hold the placeholder X and three share an
    # allele; the third array is read in none of them.
    pair = _pair(
        tmp_path / "CA",
        {"[]": barcodes[:6], "X": barcodes[6:9], "23_268del": barcodes[9:]},
    )
    other = _pair(tmp_path / "TA", {"[]": barcodes[:3], "X": barcodes[6:9]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            [pair, other],
            cell_selection=store.snapshot_cell_selection(),
            integrations=["CA", "TA"],
        )
    # No bank lists [] or X, so if they were alleles they would be rare.
    lacking = pd.DataFrame({"allele": ["5_6del"], "normalized_count": [0.5]})
    for clones in (
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={"CA": lacking},
            max_frequency=1e-9,
        ),
        store.clones.allele_clones(alleles, method="graph", max_frequency=1.0),
        store.clones.allele_clones(
            alleles, method="p_clonal", frequencies={"CA": lacking}
        ),
    ):
        assert loaded_partition(store, clones) == {frozenset(barcodes[9:])}
        n_alleles = store.clones.load_clones(clones).cells["n_alleles"].tolist()
        assert sorted(n_alleles) == [0] * 9 + [1] * 3
        cells = store.clones.load_clones(clones).cells
        assert set(cells.index[cells["n_alleles"] > 0]) == set(barcodes[9:])


def test_within_calls_each_sample_as_it_would_alone(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    for method, rules in (("graph", DARLIN_RULES), ("p_clonal", {"max_alleles": 3})):
        within = store.clones.allele_clones(
            alleles,
            method=method,
            frequencies=calling.banks,
            within="sample",
            **rules,
        )
        expected = calling.expected_partition(
            TRUTH,
            max_samples=rules.get("max_samples"),
            max_alleles=3,
            groups=TRUTH.samples,
        )
        assert loaded_partition(store, within) == expected
        for sample in SAMPLES:
            store.cells.insert(f"is_{sample}", TRUTH.samples == sample, overwrite=True)
            alone = import_allele_truth(
                store,
                calling,
                TRUTH,
                tmp_path / f"{method}_{sample}",
                cell_selection=store.snapshot_cell_selection(f"is_{sample}"),
                samples=(sample,),
            )
            clones = store.clones.allele_clones(
                alone, method=method, frequencies=calling.banks, **rules
            )
            assert loaded_partition(store, clones) == {
                clone for clone in expected if next(iter(clone)).startswith(sample)
            }


def test_within_needs_a_label_for_every_cell_with_an_edit(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    labels = TRUTH.samples.astype(object).copy()
    edited = store.clones.load_alleles(alleles).cells["n_edits"].to_numpy() > 0
    labels[np.flatnonzero(edited)[0]] = None
    store.cells.insert("partly", labels)
    options = {"method": "graph", "frequencies": calling.banks, "max_frequency": 2e-5}
    with pytest.raises(ValueError, match="no label for 1 cells with an edit"):
        store.clones.allele_clones(alleles, within="partly", **options)
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.clones.allele_clones(alleles, within="mouse", **options)
    with pytest.raises(ValueError, match="also a column of the result"):
        store.clones.allele_clones(alleles, within="n_alleles", **options)


def test_saved_clones_are_reused_by_their_rules_and_banks(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    options = {"method": "graph", "max_frequency": 2e-5, "max_samples": 1}
    clones = store.clones.allele_clones(alleles, frequencies=calling.banks, **options)
    # The same banks in another row order and with columns calling does not
    # read are the same banks.
    shuffled = {
        name: bank.iloc[::-1].assign(smoothed_homoplasy=0.5)
        for name, bank in calling.banks.items()
    }
    assert (
        store.clones.allele_clones(alleles, frequencies=shuffled, **options) == clones
    )
    # Another frequency, another sample count, another rule or a bank without
    # samples calls clones again.
    changed = {name: bank.copy() for name, bank in calling.banks.items()}
    changed["CA"].loc[changed["CA"].index[-1], "normalized_count"] = 0.5
    recounted = {name: bank.copy() for name, bank in calling.banks.items()}
    recounted["TA"].loc[recounted["TA"].index[-1], "sample_count"] = 2
    others = [
        store.clones.allele_clones(alleles, frequencies=changed, **options),
        store.clones.allele_clones(alleles, frequencies=recounted, **options),
        store.clones.allele_clones(
            alleles, frequencies=calling.banks, **options, max_alleles=3
        ),
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={
                name: bank.drop(columns="sample_count")
                for name, bank in calling.banks.items()
            },
            max_frequency=2e-5,
        ),
    ]
    assert clones not in others and len(set(others)) == len(others)
    # The graph's identity ignores fdr, which it does not read.
    assert (
        store.clones.allele_clones(
            alleles, frequencies=calling.banks, fdr=0.2, **options
        )
        == clones
    )
    recomputed = store.clones.allele_clones(
        alleles, frequencies=calling.banks, invalidate_cache=True, **options
    )
    np.testing.assert_array_equal(
        store.artifacts.load_values(recomputed).values,
        store.artifacts.load_values(clones).values,
    )


def test_saved_clones_are_reused_only_with_each_cells_rare_alleles(
    store, tmp_path
) -> None:
    from cytearc.storage.artifacts import artifact_group

    calling, alleles = _imported(store, tmp_path)
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
    )
    group = artifact_group(store.zw, clones)
    assert group["n_alleles"].dtype == np.dtype(np.int32)
    n_alleles = np.asarray(group["n_alleles"][:])
    assert n_alleles.max() > 0
    assert (
        store.clones.allele_clones(
            alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
        )
        == clones
    )
    # Clones that lost each cell's rare alleles are called again, as a new
    # artifact, rather than reused without them.
    del group["n_alleles"]
    again = store.clones.allele_clones(
        alleles, method="graph", frequencies=calling.banks, **DARLIN_RULES
    )
    assert again != clones
    np.testing.assert_array_equal(
        store.artifacts.load(again)["n_alleles"][:], n_alleles
    )
    loaded = store.clones.load_clones(again)
    np.testing.assert_array_equal(loaded.cells["n_alleles"], n_alleles)


def test_other_labels_of_one_within_column_call_other_clones(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    options = {"method": "graph", "frequencies": calling.banks, **DARLIN_RULES}
    store.cells.insert("batch", TRUTH.samples.astype(object))
    by_sample = store.clones.allele_clones(alleles, within="batch", **options)
    store.cells.insert("batch", np.full(N_CELLS, "all", dtype=object), overwrite=True)
    together = store.clones.allele_clones(alleles, within="batch", **options)
    assert together != by_sample
    assert loaded_partition(store, together) == loaded_partition(
        store, store.clones.allele_clones(alleles, **options)
    )


def test_a_read_only_store_reuses_saved_clones_and_refuses_new_ones(
    store, tmp_path
) -> None:
    calling, alleles = _imported(store, tmp_path)
    options = {"method": "graph", "frequencies": calling.banks, "max_frequency": 2e-5}
    clones = store.clones.allele_clones(alleles, **options)
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.clones.allele_clones(alleles, **options) == clones
    with pytest.raises(PermissionError, match="call_allele_clones"):
        reader.clones.allele_clones(alleles, max_alleles=3, **options)


def test_calls_follow_neither_file_order_nor_chunks_nor_threads(
    tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import recorder_alleles as operations

    calling = allele_calling_truth(TRUTH, noise=ALLELE_NOISE)
    stored = []
    for case in ("plain", "files reversed", "chunks of one row", "4 threads"):
        folder = tmp_path / case.replace(" ", "_")
        folder.mkdir()
        write_truth_store(folder / "store.zarr", TRUTH)
        store = open_store(
            folder / "store.zarr", nthreads=4 if case == "4 threads" else 1
        )
        with monkeypatch.context() as patched:
            if case == "chunks of one row":
                patched.setattr(operations, "_CHUNK_SHARE", 10**15)
            alleles = import_allele_truth(
                store,
                calling,
                TRUTH,
                folder,
                samples=SAMPLES[::-1] if case == "files reversed" else SAMPLES,
                arrays=DARLIN_ARRAYS[::-1]
                if case == "files reversed"
                else DARLIN_ARRAYS,
            )
        for method, rules in (
            ("graph", DARLIN_RULES),
            ("p_clonal", {"max_alleles": 3}),
        ):
            clones = store.clones.allele_clones(
                alleles, method=method, frequencies=calling.banks, **rules
            )
            group = store.artifacts.load(clones)
            found = {
                name: np.asarray(group[name][:])
                for name in ("values", "codes", "n_alleles", "clone_sizes")
            }
            found["attributes"] = {
                name: dict(group.attrs)[name]
                for name in ("rule_counts", "bank_coverage")
            }
            stored.append((method, found))
    for method in ("graph", "p_clonal"):
        runs = [found for name, found in stored if name == method]
        for other in runs[1:]:
            assert other["attributes"] == runs[0]["attributes"]
            for name in ("values", "codes", "n_alleles", "clone_sizes"):
                np.testing.assert_array_equal(other[name], runs[0][name])


def test_allele_clones_refuse_other_artifacts_and_arguments(store, tmp_path) -> None:
    calling, alleles = _imported(store, tmp_path)
    clones = store.clones.allele_clones(alleles, method="graph", max_frequency=0.5)
    with pytest.raises(ValueError, match="lineage_alleles artifact"):
        store.clones.allele_clones(clones, method="graph", max_frequency=0.5)
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.clones.allele_clones("alleles", method="graph", max_frequency=0.5)
    with pytest.raises(TypeError, match="invalidate_cache"):
        store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=1
        )
    with pytest.raises(ValueError, match="method must be one of"):
        store.clones.allele_clones(alleles, method="DARLIN", max_frequency=0.5)
    with pytest.raises(ValueError, match="needs frequencies"):
        store.clones.allele_clones(alleles, method="p_clonal")
    with pytest.raises(ValueError, match="names 'CC', which is no character"):
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={**calling.banks, "CC": calling.banks["CA"]},
            max_frequency=2e-5,
        )
    with pytest.raises(ValueError, match="holds no bank of 'TA'"):
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={"CA": calling.banks["CA"], "RA": calling.banks["RA"]},
            max_frequency=2e-5,
        )
    for frequencies in ([calling.banks["CA"]], {"CA": dict(calling.banks["CA"])}):
        with pytest.raises(TypeError, match="must map each character|pandas DataFrame"):
            store.clones.allele_clones(
                alleles, method="graph", frequencies=frequencies, max_frequency=2e-5
            )
    with pytest.raises(ValueError, match="lacks the column 'normalized_count'"):
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={"CA": calling.banks["CA"].drop(columns="normalized_count")},
            max_frequency=2e-5,
        )


def test_allele_clones_name_missing_and_incomplete_alleles(store, tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    _calling, alleles = _imported(store, tmp_path)
    missing = ArtifactRef(
        scope="datastore", kind="lineage_alleles", artifact_id="f" * 64, assay=None
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.clones.allele_clones(missing, method="graph", max_frequency=0.5)
    # The arguments are checked before the store is read.
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.allele_clones(
            missing, method="graph", max_frequency=0.5, invalidate_cache=np.True_
        )
    # An import that stopped before it finished leaves an incomplete artifact.
    artifact_group(store.zw, alleles).attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store.clones.allele_clones(alleles, method="graph", max_frequency=0.5)


def test_calling_refuses_a_budget_too_small_before_writing(
    store, tmp_path, monkeypatch
) -> None:
    calling, alleles = _imported(store, tmp_path)
    before = len(store.artifacts.list(kind="clones", scope="datastore"))
    for budget in (10_000, 200_000):
        monkeypatch.setattr(store, "memoryBytes", budget)
        with pytest.raises(MemoryError, match="Raise mem_budget"):
            store.clones.allele_clones(
                alleles, method="graph", frequencies=calling.banks, max_frequency=2e-5
            )
    assert len(store.artifacts.list(kind="clones", scope="datastore")) == before


def test_a_budget_that_only_writing_the_clones_exceeds_is_refused_before_calling(
    tmp_path, monkeypatch
) -> None:
    import cytearc.datastore._operations.recorder_alleles as operations

    # 100 clones of two cells among 20,000: writing the clones, as if every
    # cell were a clone of its own, needs more than any other step.
    store, alleles = _clones_of_their_own_alleles(tmp_path, 20_000, 100, 2, ("CA",))

    def call() -> object:
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=True
        )

    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)
    # Every step but writing fits the most bytes that any of them states.
    checks, _peak = traced_steps(monkeypatch, call, budget)
    fitting = max(
        stated for what, stated in checks if not what.endswith("as they are written")
    )
    entered: list[object] = []
    monkeypatch.setattr(
        operations, "allele_clone_calls", lambda *args, **kwargs: entered.append(args)
    )
    monkeypatch.setattr(store, "memoryBytes", fitting)
    with pytest.raises(
        MemoryError, match="for the clones of 20000 cells as they are written, more"
    ):
        call()
    assert entered == []


def test_writing_the_clones_counts_a_buffer_for_each_write_zarr_runs_at_once(
    tmp_path, monkeypatch
) -> None:
    # Zarr writes each chunk and each array's metadata through a buffered
    # file, as many at once as its async concurrency allows, and Python 3.14
    # made each buffer 128 KiB, where it was the file system's block. Both
    # checks of writing the clones count them, and no other check does.
    import io

    import zarr

    import cytearc.datastore._operations.recorder_alleles as operations

    store, alleles = _clones_of_their_own_alleles(tmp_path, 2_000, 100, 2, ("CA",))
    monkeypatch.setattr(store, "memoryBytes", 1 << 34)
    check = operations._check_calling

    def stated(concurrency: int) -> list[tuple[str, int]]:
        found: list[tuple[str, int]] = []

        def recorded(needed: int, budget: int, what: str) -> None:
            found.append((what, needed))
            check(needed, budget, what)

        with monkeypatch.context() as patched:
            patched.setattr(operations, "_check_calling", recorded)
            with zarr.config.set({"async.concurrency": concurrency}):
                store.clones.allele_clones(
                    alleles, method="graph", max_frequency=0.5, invalidate_cache=True
                )
        return found

    ten, twenty = stated(10), stated(20)
    assert [what for what, _ in ten] == [what for what, _ in twenty]
    grown = {what: more - fewer for (what, fewer), (_, more) in zip(ten, twenty)}
    assert grown == {
        what: 10 * io.DEFAULT_BUFFER_SIZE if what.endswith("as they are written") else 0
        for what in grown
    }
    assert sum(what.endswith("as they are written") for what in grown) == 2
    # Before reading, saving is checked as if every cell were a clone of its
    # own: 16 bytes per cell for its group and edits, 8 for the calls' rare
    # alleles, each clone's label as a Python string at the size of the
    # longest, what saving that many clones holds, and 8 MiB whatever the
    # size.
    import sys

    from cytearc.datastore._operations.clone_artifacts import clone_write_bytes

    longest = "clone_2000"
    with zarr.config.set({"async.concurrency": 10}):
        saving = clone_write_bytes(
            2_000, 2_000, len(longest), source_arrays={"n_alleles": np.int32}
        )
    (before_reading,) = [
        needed
        for what, needed in ten
        if what == "the clones of 2000 cells as they are written"
    ]
    assert before_reading == (16 + 8 + sys.getsizeof(longest)) * 2_000 + saving + (
        8 << 20
    )


def test_real_darlin_alleles_join_the_cells_that_share_a_rare_one(tmp_path) -> None:
    """DARLIN's skull library on three arrays, its frequencies from its cells."""
    lines = {}
    for array in DARLIN_ARRAYS:
        folder = DATA / f"darlin_ll837_skull_{array}"
        annotations = (folder / "AlleleAnnotations.txt").read_text().split("\n")
        colonies = (folder / "AlleleColonies.txt").read_text().split("\n")
        lines[array] = list(zip(annotations, colonies, strict=True))
    barcodes = sorted(
        {
            cell
            for pairs in lines.values()
            for _allele, line in pairs
            for cell in line.split(",")
        }
    )
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(barcodes), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=barcodes,
    )
    store = open_store(tmp_path / "store.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            [DATA / f"darlin_ll837_skull_{array}" for array in DARLIN_ARRAYS],
            cell_selection=store.snapshot_cell_selection(),
            integrations=list(DARLIN_ARRAYS),
        )
    clones = store.clones.allele_clones(alleles, method="graph", max_frequency=0.1)
    # The same rule on the raw files: an edited allele held by less than a
    # tenth of its array's edited cells links the cells that hold it.
    graph = nx.Graph()
    for array, pairs in lines.items():
        # A placeholder is an edit of no known allele: it counts among the
        # edited cells and links none.
        edited = [
            (allele, line.split(","))
            for allele, line in pairs
            if allele not in ("", "[]")
        ]
        total = sum(len(cells) for _allele, cells in edited)
        for allele, cells in edited:
            if allele in ("X", "?") or len(cells) / total >= 0.1:
                continue
            for cell in cells:
                graph.add_edge(("cell", cell), (array, allele))
    components = list(nx.connected_components(graph))
    expected = {
        frozenset(node[1] for node in component if node[0] == "cell")
        for component in components
    }
    assert loaded_partition(store, clones) == expected
    # Five clones join cells, one of four cells, and 25 clones hold the rare
    # alleles of two or three arrays.
    assert sorted(len(clone) for clone in expected if len(clone) > 1) == [2, 2, 2, 2, 4]
    assert sum(len({node[0] for node in c} - {"cell"}) > 1 for c in components) == 25


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "rules"),
    [
        ("graph", DARLIN_RULES),
        ("p_clonal", {"max_alleles": 3}),
        ("graph", {"max_frequency": 0.01}),
    ],
)
def test_calling_holds_no_more_than_the_smallest_budget_it_accepts(
    tmp_path, monkeypatch, method, rules
) -> None:
    rng = np.random.default_rng(20)
    n_cells = 6_000
    # Bare 16-base barcodes, as CARLIN writes the cells of a 10x run.
    barcodes = sorted(
        {"".join(row) for row in rng.choice(list("ACGT"), size=(n_cells + 50, 16))}
    )[:n_cells]
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=barcodes,
    )
    store = open_store(tmp_path / "store.zarr")
    store.cells.insert(
        "mouse", np.asarray([f"m{index % 3}" for index in range(n_cells)], dtype=object)
    )
    banks, paths = {}, []
    for array in DARLIN_ARRAYS:
        # Alleles of a Zipf law, a long-tailed share of them unique; four
        # cells in five read, and one in fifty X.
        read = np.flatnonzero(rng.random(n_cells) < 0.8)
        drawn = np.minimum(rng.zipf(1.4, size=len(read)), 3_000)
        texts = np.asarray(
            [f"{20 + d % 200}_{40 + d % 200 + d // 200}del" for d in drawn],
            dtype=object,
        )
        texts[rng.random(len(read)) < 0.02] = "X"
        texts[drawn == 1] = "[]"
        lines = (
            pd.Series(np.asarray(barcodes, dtype=object)[read])
            .groupby(texts)
            .apply(list)
        )
        paths.append(
            _pair(tmp_path / array, {allele: cells for allele, cells in lines.items()})
        )
        listed = [allele for allele in lines.index if allele not in ("[]", "X")][::2]
        banks[array] = pd.DataFrame(
            {
                "allele": listed,
                "normalized_count": rng.random(len(listed)) * 1e-4,
                "sample_count": rng.integers(0, 4, size=len(listed)),
            }
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            paths,
            cell_selection=store.snapshot_cell_selection(),
            integrations=list(DARLIN_ARRAYS),
        )
    # The graph of frequencies below 0.01 takes them from the cells.
    frequencies = None if method == "graph" and "max_samples" not in rules else banks

    def call(budget: int) -> object:
        monkeypatch.setattr(store, "memoryBytes", budget)
        return store.clones.allele_clones(
            alleles,
            method=method,
            frequencies=frequencies,
            within="mouse",
            invalidate_cache=True,
            **rules,
        )

    call(1 << 32)
    budget = smallest_budget(call, 1 << 30)
    _, peak = traced_peak(lambda: call(budget), budget)
    assert peak <= budget


@pytest.mark.slow
@pytest.mark.parametrize(("n_cells", "placeholders"), [(2_000, None), (6_000, 0.97)])
def test_calling_on_wide_characters_holds_no_more_than_the_smallest_budget_it_accepts(
    tmp_path, monkeypatch, n_cells, placeholders
) -> None:
    """PEtracer's characters by 300 sites, drawn evenly or nearly all the placeholder !."""
    rng = np.random.default_rng(21)
    cells = [f"cell{index:06d}" for index in range(n_cells)]
    header = ["cell_barcode"] + [
        f"intID{k}-{site}" for k in range(100) for site in ("EMX1", "HEK3", "RNF2")
    ]
    weights = (
        None
        if placeholders is None
        else [placeholders] + [(1 - placeholders) / 10] * 10
    )
    tokens = rng.choice(
        np.asarray(list("!*-12345678"), dtype=object), size=(n_cells, 300), p=weights
    )
    path = tmp_path / "characters.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + "".join(
            cell + "," + ",".join(row) + "\n"
            for cell, row in zip(cells, tokens.tolist(), strict=True)
        )
    )
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection()
    )

    def call(budget: int) -> object:
        monkeypatch.setattr(store, "memoryBytes", budget)
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.15, invalidate_cache=True
        )

    call(1 << 32)
    budget = smallest_budget(call, 1 << 30)
    _, peak = traced_peak(lambda: call(budget), budget)
    assert peak <= budget
    # The records, not the fixed bytes, set the budget.
    assert budget > 4 * (8 << 20)


def _characters_store(tmp_path: Path, rows: list[str], header: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cells = [row.split(",")[0] for row in rows]
    path = tmp_path / "characters.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return store, alleles


# PEtracer's characters of six cells on three sites of two integrations: the
# mark 1 at intID1-EMX1 in c0 and c1 and at intID1-HEK3 in c2 and c3, the mark
# 4 at intID2-RNF2 in c0 and c4, the mark 2 in c5 alone; ! is a placeholder of
# an unknown mark, - a missing site and * an unedited one.
_PETRACER = (
    "cell_barcode,intID1-EMX1,intID1-HEK3,intID2-RNF2",
    [
        "c0,1,*,4",
        "c1,1,!,*",
        "c2,-,1,*",
        "c3,-,1,!",
        "c4,!,-,4",
        "c5,2,-,!",
    ],
)
_NAMES = ("intID1-EMX1", "intID1-HEK3", "intID2-RNF2")


def test_marks_of_named_sites_are_alleles_of_their_own_characters(tmp_path) -> None:
    header, rows = _PETRACER
    store, alleles = _characters_store(tmp_path, rows, header)
    # Banks keyed as export_characters names the characters, each lacking
    # every mark, so that every mark is rare.
    lacking = {
        name: pd.DataFrame({"allele": ["9"], "normalized_count": [0.5]})
        for name in _NAMES
    }
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=lacking, max_frequency=1e-9
    )
    # The same mark at two sites is two alleles: c0 and c1 share 1 at EMX1,
    # c2 and c3 share it at HEK3, and the placeholders of c3 and c5 at RNF2
    # name no allele.
    assert loaded_partition(store, clones) == {
        frozenset({"c0", "c1", "c4"}),
        frozenset({"c2", "c3"}),
        frozenset({"c5"}),
    }
    assert store.clones.load_clones(clones).cells["n_alleles"].tolist() == [
        2,
        1,
        1,
        1,
        1,
        1,
    ]
    # A bank keyed by the integration alone names no character of these sites.
    with pytest.raises(ValueError, match="names 'intID1', which is no character"):
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={**lacking, "intID1": lacking["intID1-EMX1"]},
            max_frequency=1e-9,
        )


def test_p_clonal_counts_placeholders_and_not_missing_sites_among_the_edited_cells(
    tmp_path,
) -> None:
    header, rows = _PETRACER
    store, alleles = _characters_store(tmp_path, rows, header)
    # At intID1-EMX1, c0, c1, c4 (a placeholder) and c5 hold an edit and c2
    # and c3 are missing: M = 4, so the mark 1 at rate 0.025 has λM = 0.1 and
    # p_clonal 0.0492, which survives beside three alleles missing from their
    # banks (p 0) since 0.0492·4 ≤ 4·0.05. Counted as edits, the missing sites
    # would make M = 6, λM = 0.15 and p 0.0731, which does not, and c1 would
    # lose its only rare allele.
    banks = {
        "intID1-EMX1": pd.DataFrame({"allele": ["1"], "normalized_count": [0.025]}),
        "intID1-HEK3": pd.DataFrame({"allele": ["9"], "normalized_count": [0.5]}),
        "intID2-RNF2": pd.DataFrame({"allele": ["9"], "normalized_count": [0.5]}),
    }
    clones = store.clones.allele_clones(alleles, method="p_clonal", frequencies=banks)
    assert loaded_partition(store, clones) == {
        frozenset({"c0", "c1", "c4"}),
        frozenset({"c2", "c3"}),
        frozenset({"c5"}),
    }
    assert (
        dict(store.artifacts.load(clones).attrs)["rule_counts"][
            "alleles_not_significant"
        ]
        == 0
    )


def test_bank_coverage_counts_each_allele_once_across_groups(tmp_path) -> None:
    header, rows = _PETRACER
    store, alleles = _characters_store(tmp_path, rows, header)
    store.cells.insert("mouse", np.asarray(["m1"] * 3 + ["m2"] * 3, dtype=object))
    # The bank of HEK3 lists the mark 1, which c2 of m1 and c3 of m2 hold, a
    # test in each group: one allele, two observations.
    banks = {
        "intID1-EMX1": pd.DataFrame({"allele": ["9"], "normalized_count": [0.5]}),
        "intID1-HEK3": pd.DataFrame({"allele": ["1"], "normalized_count": [1e-6]}),
        "intID2-RNF2": pd.DataFrame({"allele": ["9"], "normalized_count": [0.5]}),
    }
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=banks, max_frequency=2e-5, within="mouse"
    )
    attributes = dict(store.artifacts.load(clones).attrs)
    assert attributes["bank_coverage"] == [
        {
            "character": "intID1-EMX1",
            "alleles": 2,
            "alleles_in_bank": 0,
            "observations": 3,
            "observations_in_bank": 0,
        },
        {
            "character": "intID1-HEK3",
            "alleles": 1,
            "alleles_in_bank": 1,
            "observations": 2,
            "observations_in_bank": 2,
        },
        {
            "character": "intID2-RNF2",
            "alleles": 1,
            "alleles_in_bank": 0,
            "observations": 2,
            "observations_in_bank": 0,
        },
    ]
    # Each group's tests are its own: 1 at EMX1 in m1, 2 at EMX1 in m2, 1 at
    # HEK3 in each, and 4 at RNF2 in each.
    assert attributes["rule_counts"]["alleles"] == 6
    # The groups split c0, c1 and c4, and c2 and c3.
    assert loaded_partition(store, clones) == {
        frozenset({"c0", "c1"}),
        frozenset({"c2"}),
        frozenset({"c3"}),
        frozenset({"c4"}),
        frozenset({"c5"}),
    }


def test_a_bank_may_name_a_character_whose_cells_are_all_unedited(tmp_path) -> None:
    header = "cell_barcode,intID1-EMX1,intID1-HEK3"
    rows = ["c0,1,*", "c1,1,*", "c2,-,*"]
    store, alleles = _characters_store(tmp_path, rows, header)
    banks = {
        "intID1-EMX1": pd.DataFrame({"allele": ["2"], "normalized_count": [0.5]}),
        "intID1-HEK3": pd.DataFrame({"allele": ["2"], "normalized_count": [0.5]}),
    }
    clones = store.clones.allele_clones(
        alleles, method="graph", frequencies=banks, max_frequency=1e-9
    )
    assert loaded_partition(store, clones) == {frozenset({"c0", "c1"})}
    with pytest.raises(
        ValueError,
        match="names 'RNF2', which is no character of these alleles; their "
        "characters are 'intID1-EMX1', 'intID1-HEK3'",
    ):
        store.clones.allele_clones(
            alleles,
            method="graph",
            frequencies={**banks, "RNF2": banks["intID1-EMX1"]},
            max_frequency=1e-9,
        )
    # An artifact without a record has no character at all.
    _store, empty = _characters_store(tmp_path / "empty", ["c0,-,-"], header)
    with pytest.raises(ValueError, match="their characters are 'intID1-EMX1'"):
        _store.clones.allele_clones(
            empty,
            method="graph",
            frequencies={"XA": banks["intID1-EMX1"]},
            max_frequency=0.5,
        )


def _assert_steps_within(traced: tuple[list[tuple[str, int]], int]) -> None:
    """No step holds more than its check states: the traced peak, which holds each
    step's shortfall from the largest check beside it, stays within the largest."""
    checks, peak = traced
    ceiling = max(stated for _what, stated in checks)
    assert peak <= ceiling, (peak - ceiling, checks)
    assert len(checks) > 10


@pytest.mark.slow
@pytest.mark.parametrize(
    "tokens",
    [
        # Nearly every site the placeholder !, so the edits set the bytes.
        {"!": 0.97, "*": 0.003, "-": 0.003, "1": 0.024},
        # Nearly every site unedited or missing, so the records set them.
        {"*": 0.5, "-": 0.45, "!": 0.01, "1": 0.02, "2": 0.02},
    ],
)
def test_each_step_on_millions_of_records_holds_no_more_than_its_check_states(
    tmp_path, monkeypatch, tokens
) -> None:
    """PEtracer's characters of 12,000 cells by 300 sites, 3.6 million records."""
    rng = np.random.default_rng(22)
    n_cells = 12_000
    cells = [f"cell{index:06d}" for index in range(n_cells)]
    header = ["cell_barcode"] + [
        f"intID{k}-{site}" for k in range(100) for site in ("EMX1", "HEK3", "RNF2")
    ]
    drawn = rng.choice(
        np.asarray(list(tokens), dtype=object),
        size=(n_cells, 300),
        p=list(tokens.values()),
    )
    path = tmp_path / "characters.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + "".join(
            cell + "," + ",".join(row) + "\n"
            for cell, row in zip(cells, drawn.tolist(), strict=True)
        )
    )
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection()
    )
    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)

    def call() -> object:
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=True
        )

    _assert_steps_within(traced_steps(monkeypatch, call, budget))


@pytest.mark.slow
@pytest.mark.parametrize("ending", ["é", "語", "😀"])
def test_each_step_on_wide_alleles_holds_no_more_than_its_check_states(
    tmp_path, monkeypatch, ending
) -> None:
    """1,000 alleles of 5,000 characters, each shared by two cells, 5 MB of text.

    An allele that ends in a character beyond ASCII is a Python string with
    a longer header; one beyond U+00FF makes Python hold each of the
    allele's characters in two bytes, and one beyond U+FFFF, such as an
    emoji, in four, so decoding the text holds up to ten bytes per stored
    byte and the decoded alleles four, which the checks count by the
    alleles' largest UTF-8 byte.
    """
    rng = np.random.default_rng(24)
    n_cells = 2_000
    cells = [f"cell{index:06d}" for index in range(n_cells)]
    letters = np.asarray(list("ACGT"))
    # Each pair of cells shares an allele, which links them.
    texts = [
        "".join(rng.choice(letters, 4_987)) + f"{pair:012d}{ending}"
        for pair in range(n_cells // 2)
    ]
    path = tmp_path / "alleles.tsv"
    path.write_text(
        "cell\tallele\n"
        + "".join(f"{cell}\t{texts[index // 2]}\n" for index, cell in enumerate(cells)),
        encoding="utf-8",
    )
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path,
        cell_selection=store.snapshot_cell_selection(),
        columns={"cell": "cell", "allele": "allele"},
        integrations="I",
    )
    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)

    def call() -> object:
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=True
        )

    assert store.clones.load_clones(call()).clones["n_cells"].tolist() == [2] * (
        n_cells // 2
    )
    _assert_steps_within(traced_steps(monkeypatch, call, budget))


@pytest.mark.slow
@pytest.mark.parametrize("ending", ["", "é"])
def test_each_step_on_few_long_alleles_holds_no_more_than_its_check_states(
    tmp_path, monkeypatch, ending
) -> None:
    """100 alleles of 50,000 characters, each shared by two of 200 cells.

    Few cells and nodes leave the steps of the calls little room beyond
    their counts, so what the operation holds whatever its size, such as
    its arrays' headers and objects, must be left out of their budget.
    """
    rng = np.random.default_rng(25)
    n_cells = 200
    cells = [f"cell{index:06d}" for index in range(n_cells)]
    letters = np.asarray(list("ACGT"))
    texts = [
        "".join(rng.choice(letters, 49_987)) + f"{pair:012d}{ending}"
        for pair in range(n_cells // 2)
    ]
    path = tmp_path / "alleles.tsv"
    path.write_text(
        "cell\tallele\n"
        + "".join(f"{cell}\t{texts[index // 2]}\n" for index, cell in enumerate(cells)),
        encoding="utf-8",
    )
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        path,
        cell_selection=store.snapshot_cell_selection(),
        columns={"cell": "cell", "allele": "allele"},
        integrations="I",
    )
    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)

    def call() -> object:
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=True
        )

    _assert_steps_within(traced_steps(monkeypatch, call, budget))


@pytest.mark.slow
def test_each_step_with_banks_and_groups_holds_no_more_than_its_check_states(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(23)
    n_cells = 60_000
    barcodes = sorted(
        {"".join(row) for row in rng.choice(list("ACGT"), size=(n_cells + 500, 16))}
    )[:n_cells]
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((n_cells, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=barcodes,
    )
    store = open_store(tmp_path / "store.zarr")
    store.cells.insert(
        "mouse", np.asarray([f"m{index % 7}" for index in range(n_cells)], dtype=object)
    )
    banks, paths = {}, []
    for array in DARLIN_ARRAYS:
        read = np.flatnonzero(rng.random(n_cells) < 0.8)
        drawn = np.minimum(rng.zipf(1.4, size=len(read)), 20_000)
        texts = np.asarray(
            [f"{20 + d % 200}_{40 + d % 200 + d // 200}del" for d in drawn],
            dtype=object,
        )
        texts[rng.random(len(read)) < 0.02] = "X"
        texts[drawn == 1] = "[]"
        lines = (
            pd.Series(np.asarray(barcodes, dtype=object)[read])
            .groupby(texts)
            .apply(list)
        )
        paths.append(_pair(tmp_path / array, dict(lines.items())))
        listed = [allele for allele in lines.index if allele not in ("[]", "X")][::2]
        filler = [f"{k}_{k + 5}delinsA" for k in range(50_000)]
        banks[array] = pd.DataFrame(
            {
                "allele": listed + filler,
                "normalized_count": rng.random(len(listed) + len(filler)) * 1e-4,
                "sample_count": rng.integers(0, 4, size=len(listed) + len(filler)),
            }
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            paths,
            cell_selection=store.snapshot_cell_selection(),
            integrations=list(DARLIN_ARRAYS),
        )
    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)
    for method, rules in (("graph", DARLIN_RULES), ("p_clonal", {"max_alleles": 3})):

        def call(method=method, rules=rules) -> object:
            return store.clones.allele_clones(
                alleles,
                method=method,
                frequencies=banks,
                invalidate_cache=True,
                **rules,
            )

        _assert_steps_within(traced_steps(monkeypatch, call, budget))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "n_clones", "clone_size", "arrays"),
    [
        # Every cell holds an allele of its own on each of three arrays, so
        # the alleles' text sets the bytes of reading the records.
        (100_000, 100_000, 1, DARLIN_ARRAYS),
        # Clones of four cells cover every cell, so the cells in clones set
        # the bytes of writing them.
        (400_000, 100_000, 4, ("CA",)),
    ],
)
def test_each_step_on_many_alleles_and_clones_holds_no_more_than_its_check_states(
    tmp_path, monkeypatch, n_cells, n_clones, clone_size, arrays
) -> None:
    store, alleles = _clones_of_their_own_alleles(
        tmp_path, n_cells, n_clones, clone_size, arrays
    )
    budget = 1 << 34
    monkeypatch.setattr(store, "memoryBytes", budget)

    def call() -> object:
        return store.clones.allele_clones(
            alleles, method="graph", max_frequency=0.5, invalidate_cache=True
        )

    traced = traced_steps(monkeypatch, call, budget)
    written = f"the {n_clones} clones of {n_cells} cells as they are written"
    assert traced[0][-1][0] == written
    _assert_steps_within(traced)


@pytest.mark.slow
def test_the_edits_check_counts_the_chunks_that_reading_the_edits_decodes(
    tmp_path, monkeypatch
) -> None:
    # The edits of 300,000 cells are three stored chunks of int64 counts,
    # which Zarr decodes at once, each into a buffer of its own, beside the
    # counts that it returns: more than the counts and their mask, nine bytes
    # a cell.
    import cytearc.datastore._operations.recorder_alleles as operations
    from cytearc.datastore._operations.record_tables import read_array
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.geometry import READ_OBJECT_BYTES

    n_cells = 300_000
    store, alleles = _clones_of_their_own_alleles(tmp_path, n_cells, 75_000, 4, ("CA",))
    stated: dict[str, int] = {}
    check = operations._check_calling

    def noted(needed: int, budget: int, what: str) -> None:
        stated[what] = needed
        check(needed, budget, what)

    monkeypatch.setattr(operations, "_check_calling", noted)
    store.clones.allele_clones(alleles, method="graph", max_frequency=0.5)
    group = artifact_group(store.zw, alleles)

    _result, peak = traced_peak(lambda: read_array(group, "n_edits") > 0, 1 << 30)

    edits = stated[f"the edits of {n_cells} cells"] - operations._FIXED_BYTES
    # Zarr's objects for the read are the fixed bytes' to cover.
    assert 9 * n_cells < peak <= edits + READ_OBJECT_BYTES, (peak, edits)
