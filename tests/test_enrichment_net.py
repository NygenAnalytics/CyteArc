import numpy as np
import pandas as pd
import pytest

from cytearc.features.enrichment.net import prepare_network, read_gmt
from cytearc.utils.logging import logger


def test_read_gmt_parses_sets_and_ignores_descriptions(tmp_path):
    path = tmp_path / "sets.gmt"
    path.write_text(
        "\nSet B\tdescription\tGene3\tGene4\nSet A\tna\tGene1\tGene2\n",
        encoding="utf-8",
    )

    result = read_gmt(path)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "source": ["Set B", "Set B", "Set A", "Set A"],
                "target": ["Gene3", "Gene4", "Gene1", "Gene2"],
            }
        ),
    )


def test_read_gmt_ignores_a_utf8_byte_order_mark(tmp_path):
    path = tmp_path / "bom.gmt"
    # Some editors start UTF-8 files with a byte order mark.
    path.write_bytes(b"\xef\xbb\xbfSET_A\tdescription\tCD3D\tCD3E\nSET_B\t-\tMS4A1\n")

    result = read_gmt(path)

    assert result["source"].tolist() == ["SET_A", "SET_A", "SET_B"]
    assert result["target"].tolist() == ["CD3D", "CD3E", "MS4A1"]


@pytest.mark.parametrize(
    "contents, match",
    [
        ("set\tdescription\n", "at least 3"),
        ("\tdescription\tgene\n", "source is empty"),
        ("set\tdescription\t\n", "no targets"),
        ("\n", "no gene sets"),
    ],
)
def test_read_gmt_rejects_malformed_files(tmp_path, contents, match):
    path = tmp_path / "bad.gmt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_gmt(path)


def test_prepare_network_matches_prunes_and_is_canonical():
    original = pd.DataFrame(
        {
            "source": ["Beta", "Alpha", "Alpha", "Beta", "Dropped"],
            "target": ["geneD", "geneA", "GENEC", "GENEB", "missing"],
            "weight": [4.0, -1.0, 2.0, 3.0, 8.0],
        }
    )
    original.index = ["edge-a", "edge-b", "edge-c", "edge-d", "edge-e"]
    unchanged = original.copy(deep=True)
    kwargs = {
        "active_feature_names": np.array(["GeneA", "GeneB", "GeneC", "GeneD"]),
        "active_feature_index": np.array([8, 2, 5, 9]),
        "tmin": 2,
        "weighted": True,
    }

    prepared = prepare_network(original, **kwargs)
    shuffled = prepare_network(
        original.sample(frac=1.0, random_state=7).reset_index(drop=True),
        **kwargs,
    )

    pd.testing.assert_frame_equal(original, unchanged)
    np.testing.assert_array_equal(prepared.source_names, ["Alpha", "Beta"])
    np.testing.assert_array_equal(prepared.source_sizes, [2, 2])
    np.testing.assert_array_equal(prepared.matched_feature_index, [2, 5, 8, 9])
    for field in (
        "source_names",
        "source_sizes",
        "matched_feature_index",
        "edge_source_index",
        "edge_feature_index",
        "edge_weight",
    ):
        np.testing.assert_array_equal(
            getattr(prepared, field),
            getattr(shuffled, field),
        )
    assert prepared.network_digest == shuffled.network_digest


def test_prepare_network_detects_duplicate_and_ambiguous_matches():
    features = np.array(["GeneA", "GeneB"])
    indices = np.array([0, 1])
    duplicate = pd.DataFrame({"source": ["Set", "Set"], "target": ["GeneA", "GeneA"]})
    collapsed_duplicate = pd.DataFrame(
        {"source": ["Set", "Set"], "target": ["GeneA", "GENEA"]}
    )

    with pytest.raises(ValueError, match="duplicate source-target"):
        prepare_network(
            duplicate,
            active_feature_names=features,
            active_feature_index=indices,
            tmin=1,
            weighted=False,
        )
    with pytest.raises(ValueError, match="after case-insensitive"):
        prepare_network(
            collapsed_duplicate,
            active_feature_names=features,
            active_feature_index=indices,
            tmin=1,
            weighted=False,
        )
    with pytest.raises(ValueError, match="multiple active assay features"):
        prepare_network(
            pd.DataFrame({"source": ["Set"], "target": ["genea"]}),
            active_feature_names=np.array(["GeneA", "GENEA"]),
            active_feature_index=indices,
            tmin=1,
            weighted=False,
            ambiguous_targets="error",
        )
    with pytest.raises(ValueError, match="ambiguous_targets must be"):
        prepare_network(
            pd.DataFrame({"source": ["Set"], "target": ["GeneB"]}),
            active_feature_names=features,
            active_feature_index=indices,
            tmin=1,
            weighted=False,
            ambiguous_targets="keep",  # type: ignore[arg-type]
        )


def test_prepare_network_drops_ambiguous_targets_by_default():
    # 10x GRCh38 feature tables repeat symbols such as GGT1 under two ids.
    names = np.array(["GGT1", "CD3D", "ggt1", "CD3E", "MATR3", "MATR3"])
    indices = np.array([10, 11, 12, 13, 14, 15])
    net = pd.DataFrame(
        {
            "source": ["T", "T", "T", "Other", "Other", "Other"],
            "target": ["GGT1", "CD3D", "CD3E", "MATR3", "CD3D", "Ggt1"],
            "weight": [5.0, 1.0, 2.0, 7.0, 3.0, 9.0],
        }
    )
    kwargs = {
        "active_feature_names": names,
        "active_feature_index": indices,
        "weighted": True,
    }

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        prepared = prepare_network(net, tmin=2, **kwargs)
    finally:
        logger.remove(sink)
    unambiguous = prepare_network(
        net.loc[net["target"].isin(["CD3D", "CD3E"])],
        tmin=2,
        **kwargs,
    )

    assert prepared.dropped_ambiguous_targets == ("GGT1", "Ggt1", "MATR3")
    assert len(messages) == 1
    assert "GGT1, Ggt1, MATR3" in messages[0]
    # Dropped edges do not count toward tmin, so "Other" keeps one edge only.
    np.testing.assert_array_equal(prepared.source_names, ["T"])
    np.testing.assert_array_equal(prepared.matched_feature_index, [11, 13])
    np.testing.assert_array_equal(prepared.edge_weight, [1.0, 2.0])
    # The digest hashes only the retained edges.
    assert prepared.network_digest == unambiguous.network_digest
    assert unambiguous.dropped_ambiguous_targets == ()

    with pytest.raises(ValueError, match="after dropping 3 ambiguous targets"):
        prepare_network(
            net.loc[net["target"].isin(["GGT1", "MATR3", "Ggt1"])],
            tmin=1,
            **kwargs,
        )
    with pytest.raises(ValueError, match="multiple active assay features"):
        prepare_network(net, tmin=2, ambiguous_targets="error", **kwargs)


def test_network_digest_only_uses_score_affecting_weights():
    first = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [1.0, 2.0],
        }
    )
    second = first.assign(weight=[3.0, 4.0])
    kwargs = {
        "active_feature_names": np.array(["GeneA", "GeneB"]),
        "active_feature_index": np.array([0, 1]),
        "tmin": 2,
    }

    weighted_first = prepare_network(first, weighted=True, **kwargs)
    weighted_second = prepare_network(second, weighted=True, **kwargs)
    unweighted_first = prepare_network(first, weighted=False, **kwargs)
    unweighted_second = prepare_network(second, weighted=False, **kwargs)

    assert weighted_first.network_digest != weighted_second.network_digest
    assert unweighted_first.network_digest == unweighted_second.network_digest
    np.testing.assert_array_equal(unweighted_first.edge_weight, [1.0, 1.0])


def test_weighted_network_drops_zero_edges_before_tmin():
    net = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="tmin=2"):
        prepare_network(
            net,
            active_feature_names=np.array(["GeneA", "GeneB"]),
            active_feature_index=np.array([0, 1]),
            tmin=2,
            weighted=True,
        )


def test_network_rejects_boolean_weights():
    net = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [True, False],
        }
    )

    with pytest.raises(ValueError, match="not boolean"):
        prepare_network(
            net,
            active_feature_names=np.array(["GeneA", "GeneB"]),
            active_feature_index=np.array([0, 1]),
            tmin=1,
            weighted=True,
        )


@pytest.mark.parametrize(
    "dropped",
    [("Ggt1", "GGT1"), ("GGT1", "GGT1"), ("GGT1", ""), ["GGT1"]],
    ids=["unsorted", "repeated", "empty", "list"],
)
def test_prepared_network_rejects_malformed_dropped_targets(dropped):
    import dataclasses

    prepared = prepare_network(
        pd.DataFrame({"source": ["Set", "Set"], "target": ["GeneA", "GeneB"]}),
        active_feature_names=np.array(["GeneA", "GeneB"]),
        active_feature_index=np.array([0, 1]),
        tmin=2,
        weighted=False,
    )
    assert dataclasses.replace(
        prepared, dropped_ambiguous_targets=("GGT1", "Ggt1")
    ).dropped_ambiguous_targets == ("GGT1", "Ggt1")

    with pytest.raises(ValueError, match="dropped targets must be sorted unique"):
        dataclasses.replace(prepared, dropped_ambiguous_targets=dropped)


def _legacy_prepare(
    net: pd.DataFrame,
    *,
    active_feature_names: np.ndarray,
    active_feature_index: np.ndarray,
    tmin: int,
    weighted: bool,
    ambiguous_targets: str = "drop",
) -> tuple:
    """Prepare a network one edge at a time, as prepare_network did before.

    Argument checks are left out: the generated networks pass them.
    """
    from cytearc.features.enrichment.net import _network_digest

    feature_index = np.asarray(active_feature_index, dtype=np.int64)
    columns = ["source", "target"] + (["weight"] if "weight" in net.columns else [])
    frame = net.loc[:, columns].copy(deep=True).reset_index(drop=True)
    if frame[["source", "target"]].isna().any().any():
        raise ValueError("Network source and target values must not be missing")
    frame["source"] = frame["source"].map(lambda value: str(value).strip())
    frame["target"] = frame["target"].map(lambda value: str(value).strip())
    if (frame["source"] == "").any() or (frame["target"] == "").any():
        raise ValueError("Network source and target values must be non-empty")
    if frame.duplicated(subset=["source", "target"]).any():
        raise ValueError("Network contains duplicate source-target edges")
    if "weight" not in frame.columns:
        frame["weight"] = 1.0
    if frame["weight"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Network weights must be numeric and not boolean")
    try:
        weights = pd.to_numeric(frame["weight"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Network weights must be numeric") from exc
    if not np.isfinite(weights).all():
        raise ValueError("Network weights must be finite")
    frame["weight"] = weights
    if weighted:
        frame = frame.loc[frame["weight"] != 0.0].copy()
        if frame.empty:
            raise ValueError("Weighted network contains no non-zero edges")
    else:
        frame["weight"] = 1.0
    positions: dict[str, list[int]] = {}
    for position, name in enumerate(np.asarray(active_feature_names)):
        positions.setdefault(str(name).upper(), []).append(position)
    matched_rows, matched_indices, ambiguous = [], [], set()
    for row_index, target in zip(frame.index, frame["target"], strict=True):
        matches = positions.get(str(target).upper(), [])
        if not matches:
            continue
        if len(matches) > 1:
            if ambiguous_targets == "error":
                raise ValueError(
                    f"Network target {target!r} matches multiple active assay "
                    "features. Pass ambiguous_targets='drop' to drop its edges."
                )
            ambiguous.add(str(target))
            continue
        matched_rows.append(int(row_index))
        matched_indices.append(int(feature_index[matches[0]]))
    dropped = tuple(sorted(ambiguous))
    if not matched_rows:
        raise ValueError(
            "Network has no targets overlapping the active assay features"
            + (f" after dropping {len(dropped)} ambiguous targets" if dropped else "")
        )
    frame = frame.loc[matched_rows].copy()
    frame["feature_index"] = np.asarray(matched_indices, dtype=np.int64)
    if frame.duplicated(subset=["source", "feature_index"]).any():
        raise ValueError(
            "Network contains duplicate edges after case-insensitive feature matching"
        )
    counts = frame.groupby("source", sort=False, observed=True).size()
    kept_sources = sorted(str(source) for source in counts[counts >= tmin].index)
    if not kept_sources:
        raise ValueError(
            "No network sources remain after feature overlap and "
            f"tmin={tmin} pruning ({len(frame)} matched edges)"
        )
    frame = frame.loc[frame["source"].isin(kept_sources)].copy()
    source_names = np.asarray(kept_sources, dtype="U")
    source_to_index = {source: index for index, source in enumerate(kept_sources)}
    frame["source_index"] = frame["source"].map(source_to_index).astype(np.int64)
    frame = frame.sort_values(
        ["source_index", "feature_index"], kind="mergesort"
    ).reset_index(drop=True)
    edge_source_index = frame["source_index"].to_numpy(dtype=np.int64)
    edge_feature_index = frame["feature_index"].to_numpy(dtype=np.int64)
    edge_weight = frame["weight"].to_numpy(dtype=np.float64)
    return (
        source_names,
        np.bincount(edge_source_index, minlength=len(source_names)).astype(np.int64),
        np.unique(edge_feature_index).astype(np.int64),
        edge_source_index,
        edge_feature_index,
        edge_weight,
        _network_digest(
            source_names,
            edge_source_index,
            edge_feature_index,
            edge_weight,
            weighted=weighted,
        ),
        dropped,
    )


def _prepared_outcome(prepare, net: pd.DataFrame, **kwargs) -> tuple:
    try:
        result = prepare(net, **kwargs)
    except (TypeError, ValueError) as error:
        return type(error), str(error)
    if not isinstance(result, tuple):
        result = (
            result.source_names,
            result.source_sizes,
            result.matched_feature_index,
            result.edge_source_index,
            result.edge_feature_index,
            result.edge_weight,
            result.network_digest,
            result.dropped_ambiguous_targets,
        )
    return tuple(
        (value.dtype.str, value.tolist()) if isinstance(value, np.ndarray) else value
        for value in result
    )


# Texts whose upper case differs in length or merges distinct values, beside
# blanks, padding, case variants, and text that looks numeric.
_TEXTS = [
    "GeneA",
    "genea",
    " GeneB",
    "GeneB ",
    "Straße",
    "STRASSE",
    "ﬁsh",
    "FISH",
    "İx",
    "ix",
    "ǅ",
    "ǆ",
    "x\x1fy",
    "x y",
    "",
    " ",
    "nan",
    "1",
    "1.0",
    "Set",
]
_FEATURES = ["GeneA", "geneb", "STRASSE", "Straße", "FISH", "ix", "ǅ", "x y", "1"]


def _generated_networks(seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    networks = []
    for kind in range(8):
        n_edges = int(rng.integers(0, 30))
        net = pd.DataFrame(
            {
                "source": rng.choice([*_TEXTS, "S1", "S2", "S3"], size=n_edges),
                "target": rng.choice([*_TEXTS, *_FEATURES], size=n_edges),
            }
        )
        if kind == 1:
            net = net.astype(object)
        elif kind == 2:
            # Equal values whose text differs: 1, 1.0, True, and -0.0.
            net["target"] = pd.Series(
                [
                    [1, 1.0, True, np.int64(1), "1", 2, -0.0, 0.0][index]
                    for index in rng.integers(0, 8, size=n_edges)
                ],
                dtype=object,
            )
        elif kind == 3:
            net["source"] = net["source"].astype("category")
        elif kind == 4:
            net["target"] = rng.integers(0, 3, size=n_edges)
        elif kind == 5:
            net = net.drop_duplicates(subset=["source", "target"])
            net.index = [f"edge{index}" for index in range(len(net))]
        if kind == 7:
            net["weight"] = rng.choice([True, False], size=len(net))
        elif kind % 3 == 0:
            net["weight"] = rng.choice([0.0, 1.0, -2.5, 3.0], size=len(net))
        elif kind % 3 == 1:
            net["weight"] = pd.Series(rng.integers(0, 3, size=len(net)), dtype=object)
        networks.append(net)
    return networks


@pytest.mark.parametrize("seed", range(6))
def test_prepare_network_matches_the_per_edge_preparation(seed):
    rng = np.random.default_rng(100 + seed)
    names = rng.choice(_FEATURES, size=int(rng.integers(1, 9)), replace=False)
    if seed % 2:
        names = names.astype(object)
    indices = rng.permutation(50)[: len(names)]

    for net in _generated_networks(seed):
        for weighted in (False, True):
            for ambiguous_targets in ("drop", "error"):
                for tmin in (1, 2):
                    options = {
                        "active_feature_names": names,
                        "active_feature_index": indices,
                        "tmin": tmin,
                        "weighted": weighted,
                        "ambiguous_targets": ambiguous_targets,
                    }
                    assert _prepared_outcome(
                        prepare_network, net, **options
                    ) == _prepared_outcome(_legacy_prepare, net, **options)


def test_prepare_network_matches_targets_as_the_case_insensitive_index(monkeypatch):
    from cytearc.metadata.table import CaseInsensitiveIndex

    names = np.array(["Straße", "ǅ", "ﬁsh", "İx", "GeneA", "genea"])
    targets = ["STRASSE", "ǆ", "Ǆ", "FISH", "İX", "i̇x", "GENEA", "straße"]
    index = CaseInsensitiveIndex(names)
    expected = {
        f"S{edge}": index.positions(target)[0]
        for edge, target in enumerate(targets)
        if len(index.positions(target)) == 1
    }
    # The preparation looks up no edge one at a time.
    monkeypatch.setattr(
        CaseInsensitiveIndex,
        "positions",
        lambda self, name: pytest.fail("prepare_network looked up an edge"),
    )

    prepared = prepare_network(
        pd.DataFrame(
            {"source": [f"S{edge}" for edge in range(len(targets))], "target": targets}
        ),
        active_feature_names=names,
        active_feature_index=np.arange(len(names)),
        tmin=1,
        weighted=False,
    )

    matched = dict(
        zip(
            prepared.source_names[prepared.edge_source_index].tolist(),
            prepared.edge_feature_index.tolist(),
            strict=True,
        )
    )
    assert matched == expected
    assert expected == {"S0": 0, "S1": 1, "S2": 1, "S3": 2, "S4": 3, "S7": 0}
    assert prepared.dropped_ambiguous_targets == ("GENEA",)
