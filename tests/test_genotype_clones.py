"""Clones called from the combinations of mutant calls that cells carry.

Hand-written GoT tables fix each cell's calls at two targets; the expected
clones follow from the rule written out: cells link where they agree at
every target both were called at, and cells that agree with two clones that
disagree at a target they were not called at are ambiguous and join neither.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from cytearc.storage.artifacts import artifact_group
from tests.clone_helpers import (
    every_cell,
    smallest_budget,
    write_cells_store,
)
from tests.perturbation_helpers import traced_peak
from tests.variant_helpers import copy_variant_store, write_variant_store
from tests.variant_truth import variant_truth

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
    return copy_variant_store(template, tmp_path)


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
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path))
    # Z, unread at S1, joins B, with which it agrees at S2; X agrees with A
    # and B, which disagree at S2, where X was not read, so it is ambiguous.
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    assert _ambiguous(store, clones) == [4]
    attributes = dict(artifact_group(store.zw, clones).attrs)
    assert attributes["n_ambiguous"] == 1
    assert attributes["n_cells_mutant"] == 6
    assert attributes["n_sites"] == 2
    record = store.artifacts.inspect(clones)
    assert record.operation == "genotype_clones"
    assert record.parameters["similarity"] == 1.0
    assert record.parameters["inclusive"] is True


def test_an_unread_site_is_unknown_rather_than_wild_type(store, tmp_path) -> None:
    # Without X, nothing joins A and B; Z still joins B, which it could not
    # if its unread S1 counted as wild type, where B is mutant.
    calls = {cell: umis for cell, umis in CALLS.items() if cell != 4}
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
    assert _clones(store, clones) == {"clone_1": [2, 3, 8], "clone_2": [0, 1]}
    # Read as wild type at S1, Z disagrees with B and is a clone of its own.
    calls[8] = ((3, 0), (0, 2))
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
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
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
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
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
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
    clones = store.variants.genotype_clones(_genotypes(store, tmp_path, calls))
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
    clones = store.variants.genotype_clones(genotypes)
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
    alone = store.variants.genotype_clones(genotypes, sites=["S1"])
    assert _clones(store, alone) == {"clone_1": [0, 1, 2, 3, 4]}
    donors = np.where(np.arange(len(IDS)) == 4, "d2", "d1").astype(object)
    store.cells.insert("donor", donors)
    within = store.variants.genotype_clones(genotypes, within="donor")
    # X's donor holds no other cell, so it agrees with no clone that
    # disagrees with another and forms a clone of its own.
    assert _clones(store, within) == {
        "clone_1": [2, 3, 8],
        "clone_2": [0, 1],
        "clone_3": [4],
    }
    assert _ambiguous(store, within) == []
    assert within != store.variants.genotype_clones(genotypes)


def test_saved_genotype_clones_are_reused_and_called_again(store, tmp_path) -> None:
    genotypes = _genotypes(store, tmp_path)
    first = store.variants.genotype_clones(genotypes)
    assert store.variants.genotype_clones(genotypes) == first
    again = store.variants.genotype_clones(genotypes, invalidate_cache=True)
    assert _clones(store, again) == _clones(store, first)
    del artifact_group(store.zw, first)["ambiguous"]
    assert store.variants.genotype_clones(genotypes) != first


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
    with pytest.raises(ValueError, match="also a column of the result"):
        store.variants.genotype_clones(genotypes, within="ambiguous")
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.variants.genotype_clones(genotypes, within="mouse")
    store.cells.insert(
        "half", np.where(np.arange(len(IDS)) == 2, "", "a").astype(object)
    )
    with pytest.raises(ValueError, match="no label for 1 cells with a mutant call"):
        store.variants.genotype_clones(genotypes, within="half")


def test_genotype_clones_refuse_a_budget_too_small_before_writing(
    store, tmp_path
) -> None:
    genotypes = _genotypes(store, tmp_path)
    before = len(store.artifacts.list(kind="clones", scope="datastore"))
    store.memoryBytes = 500
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.genotype_clones(genotypes)
    # A budget that reads the calls but not the cells' sets and masks.
    from cytearc.datastore._operations import variants

    n_cells = len(IDS)
    n_rows = int(artifact_group(store.zw, genotypes)["calls"].shape[0])
    store.memoryBytes = (
        variants._GENOTYPE_FIXED_BYTES
        + variants._GENOTYPE_CLONE_BYTES_PER_CELL * n_cells
        + variants._GENOTYPE_CLONE_BYTES_PER_ROW * n_rows
    )
    with pytest.raises(MemoryError, match="Building the sites of 6 mutant cells"):
        store.variants.genotype_clones(genotypes)
    assert len(store.artifacts.list(kind="clones", scope="datastore")) == before


@pytest.mark.slow
def test_genotype_clones_hold_no_more_than_the_smallest_budget_they_accept(
    store, tmp_path
) -> None:
    genotypes = _genotypes(store, tmp_path)

    def call(budget: int):
        store.memoryBytes = budget
        try:
            return store.variants.genotype_clones(genotypes, invalidate_cache=True)
        finally:
            store.memoryBytes = 1 << 30

    budget = smallest_budget(call, 1 << 28)
    store.memoryBytes = budget
    _clones, peak = traced_peak(
        lambda: store.variants.genotype_clones(genotypes, invalidate_cache=True),
        budget,
    )
    assert peak <= budget


@pytest.mark.slow
def test_genotype_clones_of_many_cells_hold_no_more_than_the_smallest_budget(
    tmp_path,
) -> None:
    from tests.test_variant_genotype_calls import _many_cell_counts

    # 4,000 cells at 40 targets: the calls, the cells' sets and their masks
    # of known sites outweigh what the store's objects hold.
    ids = [f"c{cell:06d}-1" for cell in range(4_000)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    counts = _many_cell_counts(store, ids, tmp_path, "threshold", n_sites=40)
    genotypes = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)

    def call(budget: int):
        store.memoryBytes = budget
        try:
            return store.variants.genotype_clones(genotypes, invalidate_cache=True)
        finally:
            store.memoryBytes = 1 << 30

    budget = smallest_budget(call, 1 << 30)
    assert traced_peak(lambda: call(budget), budget)[1] <= budget
