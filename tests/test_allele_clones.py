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
    copy_clone_store,
    import_allele_truth,
    loaded_partition,
    open_clone_store,
    smallest_budget,
    traced_peak,
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
from tests.storage_helpers import write_count_store

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
    return copy_clone_store(template, tmp_path)


def _imported(store: DataStore, folder: Path, noise=ALLELE_NOISE, **options):
    calling = allele_calling_truth(TRUTH, noise=noise)
    return calling, import_allele_truth(store, calling, TRUTH, folder, **options)


@pytest.mark.parametrize("method", ["graph", "p_clonal"])
@pytest.mark.parametrize("max_samples", [None, 1])
@pytest.mark.parametrize("max_alleles", [None, 3])
def test_each_method_recovers_the_clones_of_the_truths_rare_alleles(
    template, tmp_path, method, max_samples, max_alleles
) -> None:
    store = copy_clone_store(template, tmp_path)
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
    store = open_clone_store(tmp_path / "store.zarr")
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
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
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
        store = open_clone_store(
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
    store = open_clone_store(tmp_path / "store.zarr")
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
    store = open_clone_store(tmp_path / "store.zarr")
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
    assert traced_peak(lambda: call(budget)) <= budget


@pytest.mark.slow
def test_calling_on_wide_characters_holds_no_more_than_the_smallest_budget_it_accepts(
    tmp_path, monkeypatch
) -> None:
    """PEtracer's characters of 2,000 cells by 300 sites: 600,000 records."""
    rng = np.random.default_rng(21)
    cells = [f"cell{index:06d}" for index in range(2_000)]
    header = ["cell_barcode"] + [
        f"intID{k}-{site}" for k in range(100) for site in ("EMX1", "HEK3", "RNF2")
    ]
    tokens = rng.choice(
        np.asarray(list("*-12345678!"), dtype=object), size=(2_000, 300)
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
    store = open_clone_store(tmp_path / "store.zarr")
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
    peak = traced_peak(lambda: call(budget))
    assert peak <= budget
    # The records, not the fixed bytes, set the budget.
    assert budget > 4 * (8 << 20)
