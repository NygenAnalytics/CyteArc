"""Marker tables of any group label: the two layouts that name a slot's
groups, their readers, and the tables of earlier releases."""

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.features.markers as marker_algorithms
from cytearc.datastore._operations.features import _validate_marker_group_label
from cytearc.datastore._pipeline.summaries import _markers as marker_summary
from cytearc.datastore.datastore import DataStore
from cytearc.features.markers import RankMarkerResult
from cytearc.features.markers.table import (
    MARKER_FOLD_CHANGE_POLICY,
    MARKER_P_VALUE_POLICY,
    MARKER_REVISION_1_STAT_COLUMNS,
    MARKER_STAT_COLUMNS,
    MarkerSlot,
    can_name_marker_group,
)
from cytearc.storage.artifacts import artifact_path
from cytearc.metadata.selection import resolve_complete_labels
from tests.storage_helpers import write_count_store

EVERY_ROW = {"min_frac_exp": None, "min_auc": None, "max_p_adj": None}
# Six groups of ten cells, and one cell of a seventh group that marker search
# skips, under labels that cannot name a child group of a marker slot.
SIZES = {
    "%2F": 10,
    ".": 10,
    "..": 10,
    "T/NK": 10,
    "a\\b": 10,
    "rare/1": 1,
    "γδ T": 10,
}
# The same cells under labels that can name child groups. Both sets sort alike
# by code point, so the search ranks the groups in the same order.
TWINS = {
    "%2F": "a_pct2F",
    ".": "b_dot",
    "..": "c_dotdot",
    "T/NK": "d_T_NK",
    "a\\b": "e_a_b",
    "rare/1": "f_rare_1",
    "γδ T": "g_gd_T",
}
# Natural order compares casefolded text, so "T/NK" follows "rare/1".
NATURAL = ["%2F", ".", "..", "a\\b", "rare/1", "T/NK", "γδ T"]
TESTED = [label for label in NATURAL if label != "rare/1"]
# The search sorts its groups by code point, which gives each label its code.
BY_CODE_POINT = ["%2F", ".", "..", "T/NK", "a\\b", "rare/1", "γδ T"]
# The marker metadata of every slot, in the order that every release writes it.
MARKER_METADATA = [
    ("stat_columns", list(MARKER_STAT_COLUMNS)),
    ("method", "mannwhitneyu"),
    ("alternative", "two-sided"),
    ("tie_correction", True),
    ("continuity_correction", True),
    ("adjustment_method", "fdr_bh"),
    ("adjustment_scope", "within_group_all_tested_features"),
    ("fold_change_policy", MARKER_FOLD_CHANGE_POLICY),
    ("p_value_policy", MARKER_P_VALUE_POLICY),
]


def _refuse_search(*_args, **_kwargs):
    raise AssertionError("the rank marker search must not run")


def _store(directory, labels: np.ndarray, counts: np.ndarray) -> DataStore:
    path = str(directory / "markers.zarr")
    write_count_store(path, {"RNA": counts}, "float32")
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
    store.cells.insert("labels", np.asarray(labels, dtype=object), overwrite=True)
    return store


@pytest.fixture(scope="module")
def labeled(tmp_path_factory):
    """One store, its marker table under ``SIZES``' labels, and under their twins.

    Feature ``2 * k`` marks the cells of the ``k``-th group of ten.
    """
    rng = np.random.default_rng(31)
    labels = np.repeat(np.array(list(SIZES), dtype=object), list(SIZES.values()))
    counts = rng.poisson(2.0, size=(len(labels), 12)).astype(np.float32)
    for position, label in enumerate(label for label in SIZES if SIZES[label] > 1):
        counts[labels == label, 2 * position] += rng.integers(8, 20, 10)
    store = _store(tmp_path_factory.mktemp("labels"), labels, counts)
    store.cells.insert(
        "twins", np.array([TWINS[label] for label in labels], dtype=object)
    )
    cells = store.snapshot_cell_selection()
    features = store.features.universe(from_assay="RNA")
    tables = {}
    for column in ("labels", "twins"):
        clusters = store.snapshot_cluster_labels(column, cell_selection=cells)
        with pytest.warns(UserWarning, match="skips 1 groups"):
            tables[column] = (
                clusters,
                store.markers.search(clusters, features=features),
            )
    return store, features, tables


def _slot(store: DataStore, marker) -> zarr.Group:
    return store.zw[artifact_path(marker)]


def test_a_table_of_labels_that_cannot_name_child_groups_stores_groups_by_code(
    labeled,
) -> None:
    store, _, tables = labeled
    slot = _slot(store, tables["labels"][1])

    assert slot.attrs["group_labels"] == BY_CODE_POINT
    assert slot.attrs["skipped_group_codes"] == {"5": 1}
    # A release that reads only labels refuses a slot without skipped_groups,
    # instead of showing the codes as labels.
    assert "skipped_groups" not in slot.attrs
    assert sorted(slot.group_keys()) == ["0", "1", "2", "3", "4", "6"]
    assert dict(slot["3"].attrs) == {"n_group": 10, "n_reference": 51}


def test_marker_readers_name_every_group_by_its_label_in_natural_order(
    labeled, tmp_path
) -> None:
    store, _, tables = labeled
    marker = tables["labels"][1]

    with pytest.warns(UserWarning, match=r"'rare/1' \(1 cell\)"):
        table = store.markers.load(marker, **EVERY_ROW)
    assert list(dict.fromkeys(table["group_id"])) == TESTED
    assert table.groupby("group_id").size().to_dict() == dict.fromkeys(TESTED, 12)

    one = store.markers.load(marker, group_id="T/NK", **EVERY_ROW)
    # Joined with the skipped group's empty table, text columns become object.
    pd.testing.assert_frame_equal(
        one,
        table[table["group_id"] == "T/NK"].reset_index(drop=True),
        check_dtype=False,
    )
    with pytest.warns(UserWarning, match=r"'rare/1' \(1 cell\)"):
        assert store.markers.load(marker, group_id="rare/1").empty
    with pytest.raises(ValueError, match="no group 'T_NK'"):
        store.markers.load(marker, group_id="T_NK")

    out = tmp_path / "markers.csv"
    with pytest.warns(UserWarning, match=r"'rare/1' \(1 cell\)"):
        store.markers.export_csv(marker, str(out), **EVERY_ROW)
    exported = pd.read_csv(out)
    assert exported.columns.tolist() == NATURAL
    assert exported["rare/1"].isna().all()
    assert exported["T/NK"].tolist() == one["feature_name"].tolist()


def test_statistics_do_not_depend_on_whether_labels_name_child_groups(
    labeled,
) -> None:
    store, _, tables = labeled
    twin_slot = _slot(store, tables["twins"][1])
    assert dict(twin_slot.attrs["skipped_groups"]) == {"f_rare_1": 1}
    assert "group_labels" not in twin_slot.attrs
    assert sorted(twin_slot.group_keys()) == sorted(TWINS[label] for label in TESTED)

    with pytest.warns(UserWarning, match="rare/1"):
        by_code = store.markers.load(tables["labels"][1], **EVERY_ROW)
    with pytest.warns(UserWarning, match="f_rare_1"):
        by_label = store.markers.load(tables["twins"][1], **EVERY_ROW)
    order = ["group_id", "feature_index"]
    pd.testing.assert_frame_equal(
        by_code.assign(group_id=by_code["group_id"].map(TWINS))
        .sort_values(order)
        .reset_index(drop=True),
        by_label.sort_values(order).reset_index(drop=True),
        check_exact=True,
    )


def test_tables_of_either_layout_are_reused(labeled, monkeypatch) -> None:
    store, features, tables = labeled
    monkeypatch.setattr(marker_algorithms, "find_markers_by_rank", _refuse_search)

    for clusters, marker in tables.values():
        with pytest.warns(UserWarning, match="skips 1 groups"):
            assert store.markers.search(clusters, features=features) == marker


def test_marker_heatmap_and_run_summary_read_groups_stored_by_code(labeled) -> None:
    from cytearc.plotting.heatmaps import _prepare_marker_heatmap

    store, _, tables = labeled
    marker = tables["labels"][1]

    prepared = _prepare_marker_heatmap(store, marker=marker, topn=1, log_transform=None)
    planted = [label for label in SIZES if SIZES[label] > 1]
    assert prepared["markers"].set_index("group")["feature_index"].to_dict() == {
        label: 2 * position for position, label in enumerate(planted)
    }
    assert list(prepared["matrix"].columns) == NATURAL
    for column in ("labels", "twins"):
        summary = marker_summary(_slot(store, tables[column][1]))
        assert summary == {"groups": 6, "skippedGroups": 1}


def test_a_table_of_an_earlier_release_with_labels_now_stored_by_code_is_read_and_reused(
    tmp_path, monkeypatch
) -> None:
    # Earlier releases named a child group .zattrs in a Zarr v3 store, which
    # this release stores by code, since that name is a Zarr v2 document.
    rng = np.random.default_rng(5)
    labels = np.repeat(np.array([".zattrs", "B"], dtype=object), 10)
    counts = rng.poisson(3.0, size=(20, 6)).astype(np.float32)
    counts[labels == ".zattrs", 1] += 12.0
    store = _store(tmp_path, labels, counts)
    clusters = store.snapshot_cluster_labels(
        "labels", cell_selection=store.snapshot_cell_selection()
    )
    features = store.features.universe(from_assay="RNA")
    marker = store.markers.search(clusters, features=features)
    expected = store.markers.load(marker, **EVERY_ROW)
    slot = _slot(store, marker)
    assert slot.attrs["group_labels"] == [".zattrs", "B"]

    # The layout of earlier releases: each child group named by its label.
    for code, label in enumerate(slot.attrs["group_labels"]):
        stored = slot[str(code)]
        child = slot.create_group(label, attributes=dict(stored.attrs))
        child.create_array("stats", data=np.asarray(stored["stats"][:]))
        del slot[str(code)]
    del slot.attrs["group_labels"]
    del slot.attrs["skipped_group_codes"]
    slot.attrs["skipped_groups"] = {}
    assert sorted(slot.group_keys()) == [".zattrs", "B"]

    pd.testing.assert_frame_equal(
        store.markers.load(marker, **EVERY_ROW), expected, check_exact=True
    )
    monkeypatch.setattr(marker_algorithms, "find_markers_by_rank", _refuse_search)
    assert store.markers.search(clusters, features=features) == marker


def test_a_blank_label_is_refused_before_the_search(labeled, monkeypatch) -> None:
    store, features, tables = labeled
    clusters, _ = tables["labels"]
    resolved = resolve_complete_labels(store.zw, clusters, name="clusters")
    values = np.asarray(resolved.values, dtype=object)
    values[values == "rare/1"] = "  "
    monkeypatch.setattr(marker_algorithms, "find_markers_by_rank", _refuse_search)

    with pytest.raises(ValueError, match="'  ' is blank"):
        store._run_marker_search_artifact(
            assay=store._get_assay("RNA"),
            cell_selection=resolved.source_cell_selection,
            clusters=clusters,
            cluster_values=values,
            feature_selection=store._features_resolve("RNA", features),
        )
    for label in ("", " \t"):
        with pytest.raises(ValueError, match="snapshot_cluster_labels"):
            _validate_marker_group_label(label)


@pytest.mark.parametrize(
    ("label", "names_a_group"),
    [
        ("B cell", True),
        ("a.b", True),
        ("x.", True),
        ("%2F", True),
        ("γδ T", True),
        ("__x", True),
        ("feature_index_2", True),
        # Earlier releases named child groups by these labels.
        (".x", True),
        ("..x", True),
        ("...", True),
        (".hidden.", True),
        (".zattrs2", True),
        ("", False),
        ("T/NK", False),
        ("a\\b", False),
        ("a\x00b", False),
        (".", False),
        ("..", False),
        ("zarr.json", False),
        (".zattrs", False),
        (".zgroup", False),
        (".zarray", False),
        (".zmetadata", False),
        ("feature_index", False),
        ("feature_names", False),
        ("feature_ids", False),
    ],
)
def test_labels_that_can_name_a_child_group_of_a_marker_slot(
    label, names_a_group
) -> None:
    assert can_name_marker_group(label) is names_a_group


def _write_slot(group_ids, group_sizes) -> tuple[zarr.Group, RankMarkerResult]:
    """Write a marker slot of three features for the groups given."""
    rng = np.random.default_rng(3)
    statistics = rng.random((3, len(group_ids), 8))
    # Column 6 holds Mann-Whitney z statistics.
    statistics[:, :, 6] = rng.normal(scale=3.0, size=statistics.shape[:2])
    result = RankMarkerResult(
        group_ids=np.asarray(group_ids),
        group_sizes=np.asarray(group_sizes),
        feature_index=np.array([0, 2, 5]),
        statistics=statistics,
    )
    slot = zarr.open_group(store=MemoryStore(), mode="w").create_group("slot")
    DataStore._write_marker_slot(
        slot,
        result,
        workers=2,
        feature_names=np.array([f"g{index}" for index in range(6)]),
        feature_ids=np.array([f"id{index}" for index in range(6)]),
    )
    return slot, result


def test_labels_that_name_child_groups_keep_the_layout_of_earlier_releases() -> None:
    labels = ["%2F", "B cell", "a.b", "x", "γδ T"]
    slot, result = _write_slot(labels, [10, 20, 30, 1, 40])

    assert list(slot.attrs.items()) == [*MARKER_METADATA, ("skipped_groups", {"x": 1})]
    assert sorted(slot.array_keys()) == [
        "feature_ids",
        "feature_index",
        "feature_names",
    ]
    assert sorted(slot.group_keys()) == ["%2F", "B cell", "a.b", "γδ T"]
    for label, n_group in (("%2F", 10), ("B cell", 20), ("a.b", 30), ("γδ T", 40)):
        assert dict(slot[label].attrs) == {
            "n_group": n_group,
            "n_reference": 101 - n_group,
        }
        np.testing.assert_array_equal(
            slot[label]["stats"][:], result.stored_statistics(label)
        )


def test_one_label_that_cannot_name_a_child_group_stores_every_group_by_code() -> None:
    slot, result = _write_slot(["B cell", "T/NK", "x"], [10, 20, 1])

    assert list(slot.attrs.items()) == [
        *MARKER_METADATA,
        ("group_labels", ["B cell", "T/NK", "x"]),
        ("skipped_group_codes", {"2": 1}),
    ]
    assert sorted(slot.group_keys()) == ["0", "1"]
    for code, label, n_group in (("0", "B cell", 10), ("1", "T/NK", 20)):
        assert dict(slot[code].attrs) == {
            "n_group": n_group,
            "n_reference": 31 - n_group,
        }
        np.testing.assert_array_equal(
            slot[code]["stats"][:], result.stored_statistics(label)
        )
    read = MarkerSlot.read(slot, 6)
    assert read.groups == {"B cell": "0", "T/NK": "1"}
    assert read.skipped_groups == {"x": 1}
    table = read.table("T/NK", np.array([f"g{index}" for index in range(6)]))
    pd.testing.assert_frame_equal(
        table, result.table("T/NK", np.array([f"g{index}" for index in range(6)]))
    )


# Labels that start with a dot, which earlier releases stored by label.
DOTTED = [".x", "..x", "...", ".hidden.", ".zattrs2"]


def _stored_files(root) -> dict[str, bytes]:
    """Every file of a local store under ``root``, by its path."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_labels_that_start_with_a_dot_keep_the_layout_of_earlier_releases_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    import cytearc.datastore._operations.features as features_operations
    from zarr.storage import LocalStore

    labels = sorted([*DOTTED, "B cell", "x"])
    sizes = [1 if label == "x" else 10 for label in labels]
    rng = np.random.default_rng(3)
    statistics = rng.random((3, len(labels), 8))
    statistics[:, :, 6] = rng.normal(scale=3.0, size=statistics.shape[:2])
    result = RankMarkerResult(
        group_ids=np.asarray(labels),
        group_sizes=np.asarray(sizes),
        feature_index=np.array([0, 2, 5]),
        statistics=statistics,
    )

    def write(directory) -> dict[str, bytes]:
        slot = zarr.open_group(LocalStore(directory), mode="w").create_group("slot")
        DataStore._write_marker_slot(
            slot,
            result,
            feature_names=np.array([f"g{index}" for index in range(6)]),
            feature_ids=np.array([f"id{index}" for index in range(6)]),
        )
        return _stored_files(directory)

    written = write(tmp_path / "written")
    slot = zarr.open_group(str(tmp_path / "written"), mode="r")["slot"]
    assert list(slot.attrs.items()) == [*MARKER_METADATA, ("skipped_groups", {"x": 1})]
    assert sorted(slot.group_keys()) == sorted([*DOTTED, "B cell"])
    # Earlier releases named the child group of every label they accepted
    # by the label, which the writer does for every label when told it can.
    monkeypatch.setattr(
        features_operations, "can_name_marker_group", lambda _label: True
    )
    assert written == write(tmp_path / "by_label")
    assert any(path.startswith("slot/.x/") for path in written)


@pytest.mark.parametrize(
    "label",
    [
        "",
        "a/b",
        "a\\b",
        "a\x00b",
        ".",
        "..",
        "zarr.json",
        ".zattrs",
        ".zgroup",
        ".zarray",
        ".zmetadata",
        "feature_index",
        "feature_names",
        "feature_ids",
    ],
)
def test_a_label_that_no_child_group_can_take_stores_the_groups_by_code(label) -> None:
    labels = sorted([label, "B cell", "x"])
    slot, result = _write_slot(labels, [1 if name == "x" else 10 for name in labels])

    assert slot.attrs["group_labels"] == labels
    assert slot.attrs["skipped_group_codes"] == {str(labels.index("x")): 1}
    assert "skipped_groups" not in slot.attrs
    assert sorted(slot.group_keys()) == sorted(
        str(code) for code, name in enumerate(labels) if name != "x"
    )
    read = MarkerSlot.read(slot, 6)
    assert read.groups == {
        name: str(code) for code, name in enumerate(labels) if name != "x"
    }
    names = np.array([f"g{index}" for index in range(6)])
    pd.testing.assert_frame_equal(read.table(label, names), result.table(label, names))


def _add_group(slot, name: str) -> None:
    slot.create_group(name)


_CODE_LAYOUT_CORRUPTIONS = {
    "labels-and-skipped-groups": (
        lambda slot: slot.attrs.update({"skipped_groups": {}}),
        "not in skipped_groups",
    ),
    "codes-without-labels": (
        lambda slot: slot.attrs.__delitem__("group_labels"),
        "must list their labels in group_labels",
    ),
    "labels-text": (
        lambda slot: slot.attrs.update({"group_labels": "B cell"}),
        "distinct text labels",
    ),
    "labels-number": (
        lambda slot: slot.attrs.update({"group_labels": ["B cell", 1, "x"]}),
        "distinct text labels",
    ),
    "labels-repeated": (
        lambda slot: slot.attrs.update({"group_labels": ["B cell", "x", "x"]}),
        "distinct text labels",
    ),
    "codes-missing": (
        lambda slot: slot.attrs.__delitem__("skipped_group_codes"),
        "skipped_group_codes metadata that maps group codes",
    ),
    "codes-count": (
        lambda slot: slot.attrs.update({"skipped_group_codes": {"2": 0}}),
        "skipped_group_codes metadata that maps group codes",
    ),
    "code-outside-labels": (
        lambda slot: slot.attrs.update({"skipped_group_codes": {"2": 1, "3": 1}}),
        "named by the codes",
    ),
    "skipped-code-stored": (
        lambda slot: slot.attrs.update({"skipped_group_codes": {"1": 20, "2": 1}}),
        "named by the codes",
    ),
    "group-missing": (lambda slot: slot.__delitem__("1"), "named by the codes"),
    "group-named-by-label": (
        lambda slot: _add_group(slot, "B cell"),
        "named by the codes",
    ),
}


@pytest.mark.parametrize("corruption", sorted(_CODE_LAYOUT_CORRUPTIONS))
def test_marker_slots_that_break_the_code_layout_raise(corruption) -> None:
    slot, _ = _write_slot(["B cell", "T/NK", "x"], [10, 20, 1])
    corrupt, message = _CODE_LAYOUT_CORRUPTIONS[corruption]
    corrupt(slot)

    with pytest.raises(ValueError, match=message):
        MarkerSlot.read(slot, 6)


def test_a_revision_1_table_never_stores_groups_by_code() -> None:
    slot, _ = _write_slot(["B cell", "T/NK", "x"], [10, 20, 1])
    del slot.attrs["p_value_policy"]
    slot.attrs["stat_columns"] = list(MARKER_REVISION_1_STAT_COLUMNS)

    with pytest.raises(
        ValueError, match="must not record skipped_groups, group_labels"
    ):
        MarkerSlot.read(slot, 6)
