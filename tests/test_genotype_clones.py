"""Clones called from the combinations of mutant calls that cells carry.

Hand-written GoT tables fix each cell's calls at two targets; the expected
clones follow from the rule written out: cells link where they agree at
every target both were called at, and cells that agree with two clones that
disagree at a target they were not called at are ambiguous and join neither.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import every_cell, write_cells_store
from tests.storage_helpers import copy_store
from tests.variant_helpers import write_variant_store
from tests.variant_truth import mito_clone_truth, variant_truth, write_mgatk

TRUTH = variant_truth()
IDS = [str(cell) for cell in TRUTH.clones.cell_ids]
# Each cell's wild-type and mutant UMIs at S1 and S2, None where not read:
# A (cells 0, 1) is mutant at S1 alone, B (2, 3) at both, X (4) at S1 with S2
# unread, W (7) at neither, and Z (8) at S2 with S1 unread.
CALLS = {
    0: ((0, 2), (3, 0)),
    1: ((1, 1), (2, 0)),
    2: ((0, 3), (1, 2)),
    3: ((2, 1), (0, 1)),
    4: ((0, 2), None),
    7: ((4, 0), (3, 0)),
    8: (None, (0, 2)),
}


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("genotype_clones") / "store.zarr"
    write_variant_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def _genotypes(store, folder: Path, calls=CALLS):
    paths = []
    for target, name in enumerate(("S1", "S2")):
        lines = ["BC\twtUMIs\tmutUMIs"] + [
            f"{IDS[cell]}\t{umis[target][0]}\t{umis[target][1]}"
            for cell, umis in calls.items()
            if umis[target] is not None
        ]
        path = folder / f"{name}.tsv"
        path.write_text("\n".join(lines) + "\n")
        paths.append(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    return store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)


def _clones(store, clones) -> dict[str, list[int]]:
    cells = store.clones.load_clones(clones).cells
    found: dict[str, list[int]] = {}
    for position, cell in enumerate(IDS):
        label = cells.loc[cell, "clone"]
        if isinstance(label, str):
            found.setdefault(label, []).append(position)
    return found


def _ambiguous(store, clones) -> list[int]:
    cells = store.clones.load_clones(clones).cells
    return sorted(IDS.index(cell) for cell in cells.index[cells["ambiguous"]])


def test_cells_link_where_they_agree_and_ambiguous_cells_join_none(
    store, tmp_path
) -> None:
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path), method="exact")
    # Z, unread at S1, joins B, with which it agrees at S2; X agrees with A
    # and B, which disagree at S2, where X was not read, so it is ambiguous.
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    assert _ambiguous(store, clones) == [4]
    diagnostics = store.artifacts.inspect(clones).diagnostics
    assert diagnostics["ambiguousCells"] == 1
    assert diagnostics["mutantCells"] == 6
    assert diagnostics["sites"] == 2
    # Five of the six mutant cells are in a clone, three of them in the
    # largest.
    assert diagnostics["cellsWithClone"] == 5
    assert diagnostics["largestCloneShare"] == 3 / 5
    record = store.artifacts.inspect(clones)
    assert record.operation == "call_genotype_clones"
    assert record.parameters["similarity"] == 1.0
    assert record.parameters["inclusive"] is True


def test_an_unread_site_is_unknown_rather_than_wild_type(store, tmp_path) -> None:
    # Without X, nothing joins A and B; Z still joins B, which it could not
    # if its unread S1 counted as wild type, where B is mutant.
    calls = {cell: umis for cell, umis in CALLS.items() if cell != 4}
    clones = store.variants.genotype_clones(
        _genotypes(store, tmp_path, calls), method="exact"
    )
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    # Read as wild type at S1, Z disagrees with B and is a clone of its own.
    calls[8] = ((3, 0), (0, 2))
    clones = store.variants.genotype_clones(
        _genotypes(store, tmp_path, calls), method="exact"
    )
    assert _clones(store, clones) == {
        "clone_1": [0, 1],
        "clone_2": [2, 3],
        "clone_3": [8],
    }


def test_many_cells_that_cannot_be_placed_join_no_clone(store, tmp_path) -> None:
    # Two cells mutant at S1 and unread at S2 agree with A and B alike; as
    # one node of two cells they would join A and B if only cells of their
    # own were left out.
    calls = {**CALLS, 5: ((1, 2), None)}
    clones = store.variants.genotype_clones(
        _genotypes(store, tmp_path, calls), method="exact"
    )
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    assert _ambiguous(store, clones) == [4, 5]


def test_a_cell_read_where_partial_cells_were_not_stays_with_them(
    store, tmp_path
) -> None:
    # Cell 2 is mutant at S1 and S2; cells 0 and 1 were read at S1 alone and
    # cells 3 and 4 at S2 alone, so the partial cells share no read site.
    # Nothing disagrees, so all five are one clone, with cell 2 kept rather
    # than taken for a cell that joins two others.
    calls = {
        0: ((0, 2), None),
        1: ((1, 1), None),
        2: ((0, 3), (1, 2)),
        3: (None, (0, 1)),
        4: (None, (2, 2)),
    }
    clones = store.variants.genotype_clones(
        _genotypes(store, tmp_path, calls), method="exact"
    )
    assert _clones(store, clones) == {"clone_1": [0, 1, 2, 3, 4]}
    assert _ambiguous(store, clones) == []


def test_a_doublet_of_two_read_clones_forms_a_clone_of_its_own(store, tmp_path) -> None:
    # A is mutant at S1 and wild type at S2, B the reverse; cell 4 carries
    # both mutations where both were read, a combination of its own.
    calls = {
        0: ((0, 2), (3, 0)),
        1: ((1, 1), (2, 0)),
        2: ((3, 0), (0, 2)),
        3: ((2, 0), (1, 1)),
        4: ((0, 2), (0, 2)),
    }
    clones = store.variants.genotype_clones(
        _genotypes(store, tmp_path, calls), method="exact"
    )
    assert _clones(store, clones) == {
        "clone_1": [0, 1],
        "clone_2": [2, 3],
        "clone_3": [4],
    }


def test_a_subclone_under_dropout_stays_apart_from_its_founder(tmp_path) -> None:
    # 1,000 cells carry a founder mutation at S1 and 400 of them a subclone's
    # at S2; each site is read in about half of the cells. Cells read at S2
    # with its mutation are the subclone, cells read at both sites with S2
    # wild type the founder alone, and cells mutant at S1 but unread at S2
    # could be either.
    rng = np.random.default_rng(16)
    n_cells = 1_000
    ids = [f"cell{cell:04d}-1" for cell in range(n_cells)]
    subclone = np.arange(n_cells) < 400
    read = rng.random((n_cells, 2)) < 0.5
    store = write_cells_store(tmp_path / "store.zarr", ids)
    paths = []
    for site, name in enumerate(("S1", "S2")):
        lines = ["BC\twtUMIs\tmutUMIs"]
        for cell in np.flatnonzero(read[:, site]).tolist():
            mutant = site == 0 or subclone[cell]
            lines.append(f"{ids[cell]}\t{int(not mutant)}\t{int(mutant)}")
        path = tmp_path / f"{name}.tsv"
        path.write_text("\n".join(lines) + "\n")
        paths.append(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    genotypes = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    clones = store.variants.genotype_clones(genotypes, method="exact")
    cells = store.clones.load_clones(clones).cells.loc[ids]
    members = {
        str(label): set(np.flatnonzero((cells["clone"] == label).to_numpy()).tolist())
        for label in cells["clone"].unique()
        if isinstance(label, str) and label
    }
    expected_subclone = set(np.flatnonzero(read[:, 1] & subclone).tolist())
    expected_founder = set(np.flatnonzero(read[:, 0] & read[:, 1] & ~subclone).tolist())
    expected_ambiguous = set(np.flatnonzero(read[:, 0] & ~read[:, 1]).tolist())
    assert sorted(members.values(), key=len) == sorted(
        [expected_subclone, expected_founder], key=len
    )
    assert set(np.flatnonzero(cells["ambiguous"].to_numpy()).tolist()) == (
        expected_ambiguous
    )


def test_clones_of_chosen_sites_and_within_groups(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    # At S1 alone, A, B and X carry one combination; Z carries none there.
    alone = store.variants.genotype_clones(genotypes, sites=["S1"], method="exact")
    assert _clones(store, alone) == {"clone_1": [0, 1, 2, 3, 4]}
    donors = np.where(np.arange(len(IDS)) == 4, "d2", "d1").astype(object)
    store.cells.insert("donor", donors)
    within = store.variants.genotype_clones(genotypes, within="donor", method="exact")
    # X's donor holds no other cell, so it agrees with no clone that
    # disagrees with another and forms a clone of its own.
    assert _clones(store, within) == {
        "clone_1": [2, 3, 8],
        "clone_2": [0, 1],
        "clone_3": [4],
    }
    assert _ambiguous(store, within) == []
    assert within != store.variants.genotype_clones(genotypes, method="exact")


def test_every_genotype_method_names_sites_by_label_or_position(tmp_path) -> None:
    # mgatk's counts lie on one contig, whose positions name its sites: a
    # position resolves to the label that the genotypes record, so naming
    # sites either way calls, loads and clones the same.
    truth = mito_clone_truth()
    mito = write_variant_store(tmp_path / "store.zarr", truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = mito.variants.import_counts(
            write_mgatk(truth, tmp_path / "final"), cell_selection=every_cell(mito)
        )
    (first, first_ref, first_alt), (second, second_ref, second_alt) = (
        truth.variant_sites()[:2]
    )
    labels = [f"chrM:{first}", f"chrM:{second}"]
    alleles = {labels[0]: (first_ref, first_alt), labels[1]: (second_ref, second_alt)}
    genotypes = mito.variants.genotypes(
        counts, method="threshold", sites=labels, alleles=alleles, min_mutant_umis=1
    )
    assert (
        mito.variants.genotypes(
            counts,
            method="threshold",
            sites=[first, second],
            alleles=alleles,
            min_mutant_umis=1,
        )
        == genotypes
    )
    by_position = mito.variants.load_genotypes(genotypes, sites=[np.int64(second)])
    by_label = mito.variants.load_genotypes(genotypes, sites=[labels[1]])
    assert by_position.sites.index.tolist() == [labels[1]]
    pd.testing.assert_frame_equal(by_position.calls, by_label.calls)
    assert len(by_label.calls) > 0
    clones = mito.variants.genotype_clones(genotypes, sites=[labels[0]])
    assert mito.variants.genotype_clones(genotypes, sites=[first]) == clones
    assert mito.artifacts.inspect(clones).parameters["sites"] == [labels[0]]
    with pytest.raises(ValueError, match="The genotypes have no site 4"):
        mito.variants.genotype_clones(genotypes, sites=[4])
    with pytest.raises(TypeError, match="must hold site labels or positions"):
        mito.variants.load_genotypes(genotypes, sites=[True])


def test_a_position_names_every_variant_genotyped_there_and_no_named_site(
    tmp_path,
) -> None:
    import h5py

    from tests.variant_truth import _texts_dataset

    # A DAb-seq panel of one contig: two variants at chr13:100, which share
    # its reference reads, and FLT3-ITD, a named site without a position.
    n_cells = 12
    barcodes = [f"AAAC{cell:04d}GGTT-1-all" for cell in range(n_cells)]
    itd = "FLT3-ITD:chr13:28608251:T/TCTC"
    names = ["FLT3:chr13:100:A/G", "FLT3:chr13:100:A/T", itd]
    depth = np.full((n_cells, 3), 40, dtype=np.int16)
    alternative = np.zeros((n_cells, 3), dtype=np.int16)
    calls = np.zeros((n_cells, 3), dtype=np.int8)
    alternative[:6, 0], calls[:6, 0] = 20, 1
    alternative[6:9, 1], calls[6:9, 1] = 10, 1
    alternative[::2, 2], calls[::2, 2] = 15, 1
    reference = depth - alternative
    reference[:, 0] = reference[:, 1] = depth[:, 0] - alternative[:, :2].sum(axis=1)
    path = tmp_path / "run.hdf5"
    with h5py.File(path, "w") as handle:
        for name, values in (
            ("AD", alternative),
            ("RD", reference),
            ("DP", depth),
            ("GQ", np.full((n_cells, 3), 90, dtype=np.int8)),
            ("GT", calls),
        ):
            handle.create_dataset(name, data=values)
        _texts_dataset(handle, "VARIANTS", names)
        _texts_dataset(handle, "CELL_BARCODES", barcodes)
    store = write_cells_store(tmp_path / "store.zarr", barcodes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(path, cell_selection=every_cell(store))
    genotypes = store.variants.genotypes(counts, method="source")
    at_100 = ["chr13:100:A/G", "chr13:100:A/T"]
    assert store.variants.load_genotypes(genotypes).sites.index.tolist() == [
        *at_100,
        itd,
    ]
    # Position 100 names both of its variants in all three methods.
    assert (
        store.variants.load_genotypes(
            store.variants.genotypes(counts, method="source", sites=[100])
        ).sites.index.tolist()
        == at_100
    )
    assert store.variants.load_genotypes(
        genotypes, sites=[100]
    ).sites.index.tolist() == (at_100)
    clones = store.variants.genotype_clones(genotypes, sites=[100])
    assert store.artifacts.inspect(clones).parameters["sites"] == at_100
    assert store.variants.genotype_clones(genotypes, sites=at_100) == clones
    # A named site has no position, so no position names it, -1 included.
    with pytest.raises(ValueError, match="^The artifact has no site -1"):
        store.variants.genotypes(counts, method="source", sites=[-1])
    with pytest.raises(ValueError, match="^The genotypes have no site -1"):
        store.variants.load_genotypes(genotypes, sites=[-1])
    with pytest.raises(ValueError, match="^The genotypes have no site -1"):
        store.variants.genotype_clones(genotypes, sites=[-1])


@pytest.mark.parametrize("method", ["likelihood", "exact"])
def test_saved_genotype_clones_are_reused_and_called_again(
    store, tmp_path, method
) -> None:
    genotypes = _genotypes(store, tmp_path)
    first = store.variants.genotype_clones(genotypes, method=method)
    assert store.variants.genotype_clones(genotypes, method=method) == first
    again = store.variants.genotype_clones(
        genotypes, method=method, invalidate_cache=True
    )
    assert _clones(store, again) == _clones(store, first)
    del artifact_group(store.zw, first)["ambiguous"]
    second = store.variants.genotype_clones(genotypes, method=method)
    assert second != first
    if method == "likelihood":
        # The likelihood's clones are reused only with the sites' rates.
        del artifact_group(store.zw, second)["dropout_probability"]
        assert store.variants.genotype_clones(genotypes, method=method) != second


def test_genotype_clones_refuse_other_artifacts_and_arguments(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    with pytest.raises(ValueError, match="variant_genotypes artifact"):
        store.variants.genotype_clones(every_cell(store))
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.variants.genotype_clones("genotypes")
    with pytest.raises(ValueError, match="no site 'S3'"):
        store.variants.genotype_clones(genotypes, sites=["S3"])
    with pytest.raises(ValueError, match="sites names no site"):
        store.variants.genotype_clones(genotypes, sites=[])
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.variants.genotype_clones(genotypes, within="mouse")
    store.cells.insert(
        "half", np.where(np.arange(len(IDS)) == 2, "", "a").astype(object)
    )
    with pytest.raises(ValueError, match="no label for 1 cells with a mutant call"):
        store.variants.genotype_clones(genotypes, within="half", method="exact")
    # W, wild type at both sites, takes part in the likelihood's fit, so it
    # needs a label there and not in the exact rule.
    store.cells.insert(
        "wild", np.where(np.arange(len(IDS)) == 7, "", "a").astype(object)
    )
    with pytest.raises(ValueError, match="no label for 1 cells with a call"):
        store.variants.genotype_clones(genotypes, within="wild")
    assert _clones(
        store, store.variants.genotype_clones(genotypes, within="wild", method="exact")
    ) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}


def test_within_may_name_a_column_of_the_loaded_clones(store, tmp_path) -> None:
    # The within labels join no table, so a column named like the loaded
    # clones' ambiguous flag keeps clones apart as any column does.
    genotypes = _genotypes(store, tmp_path)
    donors = np.where(np.arange(len(IDS)) == 4, "d2", "d1").astype(object)
    store.cells.insert("donor", donors)
    store.cells.insert("ambiguous", donors.copy())
    named = store.variants.genotype_clones(
        genotypes, within="ambiguous", method="exact"
    )
    assert _clones(store, named) == _clones(
        store, store.variants.genotype_clones(genotypes, within="donor", method="exact")
    )
    assert list(store.clones.load_clones(named).cells.columns) == [
        "clone",
        "clone_size",
        "ambiguous",
    ]


@pytest.mark.parametrize("method", ["likelihood", "exact"])
def test_genotypes_without_a_known_call_give_no_clones(store, tmp_path, method) -> None:
    # Above every cell's UMIs, each call is missing or filtered: no cell
    # takes part, so no clone is called and no rate is estimated.
    counts = store.artifacts.inspect(_genotypes(store, tmp_path)).input_ref("counts")
    unread = store.variants.genotypes(
        counts, method="threshold", min_mutant_umis=1, min_umis=50
    )
    clones = store.variants.genotype_clones(unread, method=method)
    assert _clones(store, clones) == {}
    diagnostics = store.artifacts.inspect(clones).diagnostics
    assert (diagnostics["mutantCells"], diagnostics["ambiguousCells"]) == (0, 0)
    assert diagnostics["dropoutProbability"] is None
    assert diagnostics["errorProbability"] is None


@pytest.mark.parametrize("method", ["likelihood", "exact"])
def test_genotype_clones_refuse_a_budget_too_small_before_writing(
    store, tmp_path, method
) -> None:
    from cytearc.utils.memory_budget import MemoryBudgetError

    genotypes = _genotypes(store, tmp_path)
    before = len(store.artifacts.list(kind="clones", scope="datastore"))
    store.memoryBytes = 500
    with pytest.raises(MemoryBudgetError, match="Raise mem_budget to at least"):
        store.variants.genotype_clones(genotypes, method=method)
    assert len(store.artifacts.list(kind="clones", scope="datastore")) == before
