"""The type that a DataStore resolves for an assay is the type its store declares.

Merge, subset, and HTO demultiplexing read the declared type of an open assay.
It is the type that the open resolved, which a writable open records and a
read-only open must match, so no reader sees a type that differs from the
assay's class or from the store's record.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc import DataStore
from cytearc.assay import ADTassay, Assay
from cytearc.assay.classification import declared_assay_type
from cytearc.merge import DataStoreMerge
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.storage.stores import load_zarr
from cytearc.writers import SubsetZarr
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

_RNA = np.random.default_rng(3).poisson(4.0, size=(12, 5)).astype(np.uint32) + 1
_TAGS = np.random.default_rng(4).poisson(6.0, size=(12, 3)).astype(np.uint32) + 1


def _write_store(
    path: Path,
    counts: Mapping[str, np.ndarray],
    *,
    workspace: str | None = None,
    feature_ids: Mapping[str, list[str]] | None = None,
) -> str:
    """Write a fresh import whose assays share one set of cells.

    A feature is named ``f"{assay}{index}"`` unless ``feature_ids`` names the
    features of its assay.
    """
    root = load_zarr(zarr_loc=str(path), mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    cell_ids = np.asarray([f"cell{index}" for index in range(n_cells)])
    create_cell_data(root, workspace, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        feature_ids_of_assay = np.asarray(
            (feature_ids or {}).get(
                assay, [f"{assay}{index}" for index in range(values.shape[1])]
            )
        )
        array = create_zarr_count_assay(
            root,
            assay,
            workspace,
            n_cells,
            feat_ids=feature_ids_of_assay,
            feat_names=feature_ids_of_assay,
            dtype="uint32",
        )
        array[:] = values
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, workspace)
    return str(path)


def _open(location: str, **options: Any) -> DataStore:
    return DataStore(
        location,
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        **options,
    )


def _attribute_root(location: str, workspace: str | None) -> zarr.Group:
    root = zarr.open_group(location, mode="r")
    return root if workspace is None else root[workspace]


def _snapshot(location: str) -> dict[str, bytes]:
    root = Path(location)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.slow
@pytest.mark.parametrize("workspace", [None, "ws"], ids=["root", "workspace"])
def test_an_explicit_type_is_the_declared_type_of_the_whole_session(
    tmp_path, workspace, monkeypatch
) -> None:
    # Before, a workspace store opened with assay_types={"hashtags": "HTO"}
    # recorded HTO and opened the assay as ADTassay, while merge and subset
    # read the workspace attributes that the assay captured before the open
    # recorded the type, and declared it as Assay.
    counts = {"RNA": _RNA, "hashtags": _TAGS}
    first = _open(
        _write_store(tmp_path / "first.zarr", counts, workspace=workspace),
        workspace=workspace,
        assay_types={"hashtags": "HTO"},
    )
    second = _open(
        _write_store(tmp_path / "second.zarr", counts, workspace=workspace),
        workspace=workspace,
        assay_types={"hashtags": "HTO"},
    )
    hashtags = first.get_assay("hashtags")
    assert type(hashtags) is ADTassay
    assert hashtags.assayType == "HTO"
    assert declared_assay_type(hashtags) == "HTO"
    assert _attribute_root(str(first.zarr_loc), workspace).attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }

    merger = DataStoreMerge(
        [first, second], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    )
    plan = merger.plan()
    assert {assay.assayName: assay.assayType for assay in plan.assays} == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }
    assert plan.manifest["assayTypes"] == {"RNA": "RNA", "hashtags": "HTO"}
    merger.dump()
    merged = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    assert merged.attrs["assayTypes"] == {"RNA": "RNA", "hashtags": "HTO"}

    subset_path = str(tmp_path / "subset.zarr")
    SubsetZarr(
        subset_path,
        assays=[first.RNA, hashtags],
        out_workspace=workspace,
        cell_ids=first.cells.fetch_all("ids")[:8],
        nthreads=1,
    ).dump()
    assert _attribute_root(subset_path, workspace).attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }

    # Demultiplexing accepts the assay in the session that declared it.
    monkeypatch.setattr(
        "cytearc.datastore._operations.quality_control._hto_calls",
        lambda frame, **_kwargs: (
            frame.idxmax(axis=1),
            dict.fromkeys(frame.columns, 0),
            [],
        ),
    )
    cells = first.snapshot_cell_selection("I")
    identity = first.qc.hto_demultiplexing(cells, from_assay="hashtags")
    assert identity.kind == "hto_identity"


def test_the_declared_type_is_the_one_the_open_resolved(tmp_path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    datastore = _open(location)
    assert type(datastore.get_assay("hashtags")) is Assay

    # A record changed behind the open does not change the open assays, whose
    # class and normalization follow the type the open resolved.
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = {
        "RNA": "RNA",
        "hashtags": "HTO",
    }
    assert declared_assay_type(datastore.get_assay("hashtags")) == "Assay"
    cells = datastore.snapshot_cell_selection("I")
    with pytest.raises(TypeError, match="'hashtags' is declared as 'Assay'"):
        datastore.qc.hto_demultiplexing(cells, from_assay="hashtags")
    reopened = _open(location)
    assert declared_assay_type(reopened.get_assay("hashtags")) == "HTO"


def test_a_read_only_open_rejects_a_type_that_differs_from_the_store(
    tmp_path,
) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    _open(location)
    assert zarr.open_group(location, mode="r").attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "Assay",
    }
    before = _snapshot(location)

    # Before, the assay opened as ADTassay while merge and subset read the
    # recorded Assay.
    message = (
        r"assay_types declares assay 'hashtags' as 'HTO', but the store declares "
        r"it as 'Assay'\. A read-only open cannot record a type.*"
        r"zarr_mode='r\+' and assay_types=\{'hashtags': 'HTO'\}"
    )
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r", assay_types={"hashtags": "HTO"})
    # An assay without a record entry declares the preset of its name.
    with pytest.raises(ValueError, match="declares assay 'RNA' as 'URNA'"):
        _open(location, zarr_mode="r", assay_types={"RNA": "URNA"})
    assert _snapshot(location) == before

    matching = _open(
        location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "Assay"}
    )
    assert declared_assay_type(matching.get_assay("hashtags")) == "Assay"
    assert declared_assay_type(matching.RNA) == "RNA"
    assert _snapshot(location) == before

    # A writable open records the new type, and read-only opens then match it.
    _open(location, assay_types={"hashtags": "HTO"})
    reopened = _open(location, zarr_mode="r", assay_types={"hashtags": "HTO"})
    assert type(reopened.get_assay("hashtags")) is ADTassay
    assert declared_assay_type(reopened.get_assay("hashtags")) == "HTO"

    # Without a record, an assay declares the preset of its name or Assay.
    del zarr.open_group(location, mode="r+").attrs["assayTypes"]
    matching = _open(
        location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "Assay"}
    )
    assert declared_assay_type(matching.get_assay("hashtags")) == "Assay"
    with pytest.raises(ValueError, match="declares it as 'Assay'"):
        _open(location, zarr_mode="r", assay_types={"hashtags": "ADT"})


def test_a_record_that_is_not_a_mapping_raises_its_remedy(tmp_path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    _open(location)
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = ["RNA", "HTO"]
    before = _snapshot(location)

    message = (
        r"The assayTypes attribute of the store is .*, which is not a mapping.*"
        r"zarr_mode='r\+'.*assay_types=\{'RNA': <preset>, 'hashtags': <preset>\}"
    )
    # Before, the record was treated as empty and a writable open replaced it.
    with pytest.raises(ValueError, match=message):
        _open(location)
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r")
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "HTO"})
    # Declaring only some assays leaves the others without a type.
    with pytest.raises(ValueError, match=message):
        _open(location, assay_types={"hashtags": "HTO"})
    assert _snapshot(location) == before

    # A writable open that declares every assay replaces the record.
    datastore = _open(location, assay_types={"RNA": "RNA", "hashtags": "HTO"})
    assert declared_assay_type(datastore.get_assay("hashtags")) == "HTO"
    assert zarr.open_group(location, mode="r").attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }


def test_repack_rejects_a_record_that_is_not_a_mapping(tmp_path) -> None:
    from cytearc.tools.repack_zarr import repack_store

    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA})
    _open(location)
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = ["RNA"]
    output = tmp_path / "repacked.zarr"
    with pytest.raises(ValueError, match="which is not a mapping"):
        repack_store(location, str(output))
    assert not output.exists()


def test_a_derived_assay_declares_the_type_it_was_registered_with(tmp_path) -> None:
    peaks = ["chr1:100-200", "chr1:250-350", "chr2:100-200"]
    counts = np.array([[2, 0, 1], [0, 3, 0], [1, 0, 4], [3, 1, 1]], dtype=np.uint32)
    root = load_zarr(zarr_loc=str(tmp_path / "atac.zarr"), mode="w")
    cell_ids = np.asarray([f"cell{index}" for index in range(len(counts))])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    array = create_zarr_count_assay(
        root,
        "ATAC",
        None,
        len(counts),
        feat_ids=np.asarray(peaks),
        feat_names=np.asarray(peaks),
        dtype="uint32",
    )
    array[:] = counts
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "ATAC", None)
    datastore = DataStore(
        str(tmp_path / "atac.zarr"),
        default_assay="ATAC",
        min_features_per_cell=-1,
        nthreads=1,
    )
    bed = tmp_path / "genes.bed"
    pd.DataFrame(
        [
            ("chr1", 120, 300, "gene_a", "GENE_A"),
            ("chr2", 120, 180, "gene_b", "GENE_B"),
        ]
    ).to_csv(bed, sep="\t", header=False, index=False)

    datastore.add_melded_assay(
        from_assay="ATAC",
        external_bed_fn=str(bed),
        assay_label="GENES",
        assay_type="GeneActivity",
        renormalization=False,
    )
    datastore.ATAC.feats.insert("module", np.array([1, 1, 2]), overwrite=True)
    datastore.add_grouped_assay("module", from_assay="ATAC", assay_label="MODULES")
    # Gene scores and group means are not counts, so steps that need counts
    # refuse them; the ATAC counts carry no record.
    derived = {"importedAs": None, "cellTotal": None}
    assert datastore.get_assay("GENES").normalizedValues == derived
    assert datastore.get_assay("MODULES").normalizedValues == derived
    assert datastore.ATAC.normalizedValues is None

    # Every assay of the session carries its type, including the ones that
    # the session kept when a later derived assay was registered.
    assert {
        name: declared_assay_type(datastore.get_assay(name))
        for name in datastore.assay_names
    } == {"ATAC": "ATAC", "GENES": "GeneActivity", "MODULES": "Assay"}
    assert zarr.open_group(str(tmp_path / "atac.zarr"), mode="r").attrs[
        "assayTypes"
    ] == {"ATAC": "ATAC", "GENES": "GeneActivity", "MODULES": "Assay"}


@pytest.mark.parametrize(
    ("declared", "opened_as"), [(None, Assay), ("ADT", ADTassay)], ids=["Assay", "ADT"]
)
def test_adding_an_assay_opens_again_an_assay_whose_type_another_session_changed(
    tmp_path, declared, opened_as
) -> None:
    location = _write_store(
        tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS}, workspace="ws"
    )
    types = None if declared is None else {"hashtags": declared}
    datastore = _open(location, workspace="ws", assay_types=types)
    rna = datastore.RNA
    rna.sf = 2000
    held = datastore.get_assay("hashtags")
    assert type(held) is opened_as
    datastore.RNA.feats.insert("module", np.array([1, 1, 2, 2, 3]), overwrite=True)
    # Another session declares the hashtags as HTO, which opens as the class
    # that ADT opens as.
    _open(location, workspace="ws", assay_types={"hashtags": "HTO"})

    datastore.add_grouped_assay("module", assay_label="MODULES")

    # The assay whose stored type changed is opened again with that type, as
    # a new session opens it, and the RNA assay keeps the session's object.
    hashtags = datastore.get_assay("hashtags")
    assert hashtags is not held
    assert type(hashtags) is ADTassay
    assert declared_assay_type(hashtags) == "HTO"
    assert datastore.RNA is rna and rna.sf == 2000
    assert _attribute_root(location, "ws").attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
        "MODULES": "Assay",
    }


_PEAKS = ["chr1:100-200", "chr1:250-350", "chr2:100-200"]
_PEAK_COUNTS = np.random.default_rng(5).poisson(3.0, size=(12, 3)).astype(np.uint32)


@pytest.mark.parametrize(
    ("call", "label", "label_type"),
    [
        ("add_grouped_assay", "MODULES", "Assay"),
        ("add_melded_assay", "GeneScores", "Assay"),
        ("correct_ambient", "RNA_soup", "RNA"),
    ],
)
@pytest.mark.parametrize("workspace", [None, "ws"], ids=["root", "workspace"])
def test_adding_an_assay_keeps_what_another_session_recorded(
    tmp_path, workspace, call, label, label_type
) -> None:
    location = _write_store(
        tmp_path / "data.zarr",
        {"RNA": _RNA, "ATAC": _PEAK_COUNTS, "hashtags": _TAGS},
        workspace=workspace,
        feature_ids={"ATAC": _PEAKS},
    )
    datastore = _open(location, workspace=workspace)
    datastore.RNA.feats.insert("module", np.array([1, 1, 2, 2, 3]), overwrite=True)
    ids = datastore.cells.fetch_all("ids")
    datastore.cells.insert("is_cell", pd.Series(np.arange(12) < 6, index=ids))
    datastore.cells.insert("is_empty", pd.Series(np.arange(12) >= 6, index=ids))
    profile = datastore.qc.ambient_profile(
        empty_droplets=datastore.snapshot_cell_selection("is_empty")
    )
    estimate = datastore.qc.ambient_contamination(
        datastore.snapshot_cell_selection("is_cell"), profile, fraction=0.1
    )
    pd.DataFrame([("chr1", 120, 300, "gene_a", "GENE_A", "+")]).to_csv(
        tmp_path / "genes.bed", sep="\t", header=False, index=False
    )
    calls = {
        "add_grouped_assay": lambda: datastore.add_grouped_assay(
            "module", assay_label=label
        ),
        "add_melded_assay": lambda: datastore.add_melded_assay(
            from_assay="ATAC",
            external_bed_fn=str(tmp_path / "genes.bed"),
            assay_label=label,
        ),
        "correct_ambient": lambda: datastore.qc.correct_ambient(
            estimate, assay_label=label
        ),
    }
    # Another session declares the hashtags as HTO and makes them the store's
    # default assay.
    DataStore(
        location,
        workspace=workspace,
        default_assay="hashtags",
        assay_types={"hashtags": "HTO"},
        nthreads=1,
    )

    calls[call]()

    # Before, the call wrote back every attribute that this session read when
    # it opened the group: the hashtags became Assay again and the default
    # assay RNA, except where a workspace store's grouped or melded assay read
    # the workspace group again.
    attributes = _attribute_root(location, workspace).attrs
    assert attributes["defaultAssay"] == "hashtags"
    assert attributes["assayTypes"] == {
        "RNA": "RNA",
        "ATAC": "ATAC",
        "hashtags": "HTO",
        label: label_type,
    }
    # This session reads the record again, so it opens the hashtags as HTO.
    assert declared_assay_type(datastore.get_assay("hashtags")) == "HTO"
