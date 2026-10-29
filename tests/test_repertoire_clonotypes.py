import itertools

import numpy as np
import pytest

from cytearc.repertoire.clonotypes import (
    DUAL_IR,
    RECEPTOR_ARMS,
    assign_clonotypes,
    clonotype_labels,
)

# Sequences by name, so the tests read as receptors: a* are VJ junctions and
# b* VDJ junctions.
_CODES = {name: code for code, name in enumerate(["a1", "a2", "a3", "b1", "b2", "b3"])}


def _slots(*cells: tuple) -> np.ndarray:
    """Encode cells given as (VJ_1, VJ_2, VDJ_1, VDJ_2) sequence names."""
    return np.asarray(
        [[-1 if name is None else _CODES[name] for name in cell] for cell in cells],
        dtype=np.int64,
    )


def _partition(codes: np.ndarray) -> list[set[int]]:
    """Return the cells of each clonotype, as sets, in a canonical order."""
    groups: dict[int, set[int]] = {}
    for cell, code in enumerate(codes.tolist()):
        if code >= 0:
            groups.setdefault(code, set()).add(cell)
    return sorted(groups.values(), key=min)


def _assign(cells: np.ndarray, groups=None, **options):
    if groups is None:
        groups = np.zeros(len(cells), dtype=np.int64)
    return assign_clonotypes(cells, groups=np.asarray(groups), **options)


def test_identical_pairs_share_a_clonotype() -> None:
    cells = _slots(
        ("a1", None, "b1", None),
        ("a1", None, "b1", None),
        ("a1", None, "b2", None),
        ("a2", None, "b1", None),
    )
    assignment = _assign(cells)
    assert _partition(assignment.codes) == [{0, 1}, {2}, {3}]
    assert assignment.sizes.tolist() == [2, 1, 1]
    assert assignment.n_clonotypes == 3
    # The largest clonotype is first.
    assert assignment.codes[0] == 0


def test_dual_chains_link_cells_through_either_chain() -> None:
    cells = _slots(
        ("a1", None, "b1", None),
        ("a1", "a2", "b1", None),
        ("a2", None, "b1", None),
    )
    # Under "any", cell 0 and cell 2 join through cell 1.
    assert _partition(_assign(cells).codes) == [{0, 1, 2}]
    # Under "all", only cells with the same chains join.
    assert _partition(_assign(cells, dual_ir="all").codes) == [{0}, {1}, {2}]
    # Under "primary_only", the second chain is ignored.
    assert _partition(_assign(cells, dual_ir="primary_only").codes) == [{0, 1}, {2}]


def test_all_chains_match_in_either_order() -> None:
    cells = _slots(
        ("a1", "a2", "b1", None),
        ("a2", "a1", "b1", None),
        ("a1", "a2", "b1", "b2"),
    )
    assert _partition(_assign(cells, dual_ir="all").codes) == [{0, 1}, {2}]


def test_cells_with_one_arm_join_only_cells_with_one_arm() -> None:
    cells = _slots(
        (None, None, "b1", None),
        (None, None, "b1", None),
        ("a1", None, "b1", None),
        ("a1", None, None, None),
    )
    assert _partition(_assign(cells).codes) == [{0, 1}, {2}, {3}]
    # Under "any", one shared arm is enough.
    assert _partition(_assign(cells, receptor_arms="any").codes) == [{0, 1, 2, 3}]


def test_one_arm_compares_that_arm_alone() -> None:
    cells = _slots(
        ("a1", None, "b1", None),
        ("a1", None, "b2", None),
        (None, None, "b1", None),
    )
    vj = _assign(cells, receptor_arms="VJ")
    assert _partition(vj.codes) == [{0, 1}]
    # A cell without a VJ chain has no clonotype.
    assert vj.codes[2] == -1 and vj.configurations[2] == -1
    assert _partition(_assign(cells, receptor_arms="VDJ").codes) == [{0, 2}, {1}]


def test_genes_must_match_when_asked() -> None:
    cells = _slots(
        ("a1", None, "b1", None),
        ("a1", None, "b1", None),
        ("a1", None, "b1", None),
    )
    v_genes = np.asarray([[1, -1, 5, -1], [2, -1, 5, -1], [1, -1, 5, -1]])
    j_genes = np.asarray([[7, -1, 9, -1], [7, -1, 9, -1], [7, -1, 8, -1]])
    assert _partition(_assign(cells, v_genes=v_genes).codes) == [{0, 2}, {1}]
    assert _partition(_assign(cells, j_genes=j_genes).codes) == [{0, 1}, {2}]
    assert _partition(_assign(cells, v_genes=v_genes, j_genes=j_genes).codes) == [
        {0},
        {1},
        {2},
    ]
    # Missing calls match each other.
    missing = np.full((3, 4), -1)
    assert _partition(_assign(cells, v_genes=missing).codes) == [{0, 1, 2}]


def test_groups_keep_clonotypes_apart() -> None:
    cells = _slots(
        ("a1", None, "b1", None),
        ("a1", None, "b1", None),
        ("a1", None, "b1", None),
    )
    assignment = _assign(cells, groups=[0, 1, -1])
    assert _partition(assignment.codes) == [{0}, {1}]
    assert assignment.codes[2] == -1


def test_ranks_follow_size_then_configuration_whatever_the_cell_order() -> None:
    cells = _slots(
        ("a2", None, "b2", None),
        ("a1", None, "b1", None),
        ("a3", None, "b3", None),
        ("a3", None, "b3", None),
        ("a1", None, "b2", None),
    )
    first = _assign(cells).codes
    # The pair is largest; ties of one cell follow the sorted configurations.
    assert first.tolist() == [3, 1, 0, 0, 2]
    order = np.asarray([4, 2, 0, 3, 1])
    again = _assign(cells[order]).codes
    np.testing.assert_array_equal(again, first[order])


def test_no_eligible_cell_gives_no_clonotype() -> None:
    cells = _slots((None, None, None, None), ("a1", None, None, None))
    assignment = _assign(cells, groups=[0, -1])
    assert assignment.codes.tolist() == [-1, -1]
    assert assignment.n_clonotypes == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"receptor_arms": "both"}, "receptor_arms must be one of"),
        ({"dual_ir": "some"}, "dual_ir must be one of"),
        ({"groups": np.zeros(3)}, "one group per cell"),
        ({"v_genes": np.zeros((2, 3))}, "Gene codes must align"),
    ],
)
def test_arguments_are_checked(kwargs, message) -> None:
    cells = _slots(("a1", None, "b1", None), ("a1", None, "b1", None))
    arguments = {"groups": np.zeros(2, dtype=np.int64)} | kwargs
    with pytest.raises(ValueError, match=message):
        assign_clonotypes(cells, **arguments)
    with pytest.raises(ValueError, match="four slots"):
        assign_clonotypes(np.zeros((2, 3)), groups=np.zeros(2))


def test_labels_name_clonotypes_by_rank() -> None:
    assert clonotype_labels(3).tolist() == ["clonotype_1", "clonotype_2", "clonotype_3"]
    assert clonotype_labels(0).tolist() == []


def _pairwise_partition(
    sequences: np.ndarray,
    groups: np.ndarray,
    genes: np.ndarray,
    receptor_arms: str,
    dual_ir: str,
) -> list[set[int]]:
    """Clonotypes from the pairwise rules, checked for every pair of cells."""
    chains = [0] if dual_ir == "primary_only" else [0, 1]
    arms = {
        "VJ": [0 + chain for chain in chains],
        "VDJ": [2 + chain for chain in chains],
    }
    taking = ["VJ", "VDJ"] if receptor_arms in ("all", "any") else [receptor_arms]

    def elements(cell: int, arm: str) -> list[tuple[int, int]]:
        return [
            (int(sequences[cell, slot]), int(genes[cell, slot]))
            for slot in arms[arm]
            if sequences[cell, slot] >= 0
        ]

    def eligible(cell: int) -> bool:
        return groups[cell] >= 0 and any(elements(cell, arm) for arm in taking)

    def arm_match(first: int, second: int, arm: str) -> bool:
        a, b = elements(first, arm), elements(second, arm)
        if dual_ir == "all":
            return bool(a) and sorted(a) == sorted(b)
        if dual_ir == "primary_only":
            slot = arms[arm][0]
            return sequences[first, slot] >= 0 and (
                sequences[first, slot],
                genes[first, slot],
            ) == (sequences[second, slot], genes[second, slot])
        return any(x == y for x in a for y in b)

    def primary_arms(cell: int) -> int:
        return sum(sequences[cell, arms[arm][0]] >= 0 for arm in ("VJ", "VDJ"))

    def match(first: int, second: int) -> bool:
        if groups[first] != groups[second]:
            return False
        if receptor_arms in ("VJ", "VDJ"):
            return arm_match(first, second, receptor_arms)
        matched = [arm_match(first, second, arm) for arm in ("VJ", "VDJ")]
        if receptor_arms == "any":
            return any(matched)
        count = primary_arms(first)
        if count != primary_arms(second):
            return False
        return all(matched) if count == 2 else any(matched)

    cells = [cell for cell in range(len(groups)) if eligible(cell)]
    parent = {cell: cell for cell in cells}

    def find(cell: int) -> int:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for first, second in itertools.combinations(cells, 2):
        if match(first, second):
            parent[find(first)] = find(second)
    components: dict[int, set[int]] = {}
    for cell in cells:
        components.setdefault(find(cell), set()).add(cell)
    return sorted(components.values(), key=min)


@pytest.mark.parametrize("seed", range(6))
def test_keys_link_exactly_the_cells_that_the_pairwise_rules_link(seed) -> None:
    rng = np.random.default_rng(seed)
    n_cells = 80
    # Few distinct sequences, so chains collide often.
    sequences = rng.integers(0, 4, size=(n_cells, 4))
    filled = rng.random((n_cells, 4)) < [0.8, 0.3, 0.8, 0.2]
    # A secondary slot is filled only after its primary, as ranked chains are,
    # but a primary chain may lack the compared sequence.
    filled[:, 1] &= filled[:, 0] | (rng.random(n_cells) < 0.1)
    filled[:, 3] &= filled[:, 2] | (rng.random(n_cells) < 0.1)
    sequences = np.where(filled, sequences, -1)
    genes = np.where(filled, rng.integers(-1, 2, size=(n_cells, 4)), -1)
    groups = rng.integers(-1, 2, size=n_cells)
    for receptor_arms, dual_ir in itertools.product(RECEPTOR_ARMS, DUAL_IR):
        for use_genes in (False, True):
            gene_codes = genes if use_genes else np.zeros_like(genes)
            assignment = assign_clonotypes(
                sequences,
                groups=groups,
                v_genes=genes if use_genes else None,
                receptor_arms=receptor_arms,
                dual_ir=dual_ir,
            )
            expected = _pairwise_partition(
                sequences, groups, gene_codes, receptor_arms, dual_ir
            )
            assert _partition(assignment.codes) == expected, (
                receptor_arms,
                dual_ir,
                use_genes,
            )


def test_representatives_are_first_cells_of_the_most_frequent_configuration() -> None:
    from cytearc.repertoire.clonotypes import representative_cells

    cells = _slots(
        ("a1", None, "b1", None),  # 0: clonotype of cells 0 to 4, configuration x
        ("a1", "a2", "b1", None),  # 1: configuration y
        ("a1", "a2", "b1", None),  # 2: configuration y
        ("a1", None, "b1", None),  # 3: configuration x
        ("a2", None, "b1", None),  # 4: configuration z
        ("a3", None, "b3", None),  # 5: a clonotype of its own
        (None, None, None, None),  # 6: no clonotype
    )
    assignment = _assign(cells, groups=[0, 0, 0, 0, 0, 0, -1])
    representatives, n_configurations = representative_cells(assignment)
    # x and y tie with two cells; x comes first in sorted order.
    assert representatives.tolist() == [0, 5]
    assert n_configurations.tolist() == [3, 1]
    empty = _assign(_slots((None, None, None, None)))
    assert [array.tolist() for array in representative_cells(empty)] == [[], []]


def test_cluster_pairs_are_checked_against_the_memory_left() -> None:
    from cytearc.repertoire.clonotypes import assign_clonotype_clusters

    # Three cells whose beta chains differ by one residue: two pairs of
    # chains within the cutoff.
    sequences = np.asarray([[-1, -1, 0, -1], [-1, -1, 1, -1], [-1, -1, 2, -1]])
    texts = ["CASSF", "CASTF", "CAGSF"]
    groups = np.zeros(3, dtype=np.int64)
    options = dict(groups=groups, metric="hamming", cutoff=1, receptor_arms="VDJ")
    linked = assign_clonotype_clusters(sequences, texts, max_bytes=160, **options)
    assert linked.codes.tolist() == [0, 0, 0]
    with pytest.raises(MemoryError, match="2 pairs of chains within the cutoff"):
        assign_clonotype_clusters(sequences, texts, max_bytes=159, **options)
