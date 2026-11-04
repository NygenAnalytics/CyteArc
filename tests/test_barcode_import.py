"""Lineage barcodes imported from every supported format recover a synthetic truth.

The module writes a store of the truth's cells once, as a template. Tests that
only import share one copy, since each import saves its own artifact; tests
that check reuse, read-only stores or the memory budget copy the template.
Each writer emits the truth's molecules as a tool writes them, one file per
sample, with sequencing errors, ambient reads and droplets outside the store;
the import must give back exactly the molecules or barcode counts that the
truth fixes.
"""

import gzip
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.barcodes import LineageBarcodes
from cytearc.utils.logging import collect_warnings
from cytearc.utils.memory_budget import MemoryBudgetError
from tests.clone_helpers import (
    assert_joined,
    every_cell,
    example_ids,
    write_truth_store,
)
from tests.clone_truth import (
    N_CELLS,
    SAMPLES,
    barcode_truth,
    clone_truth,
    write_bartab_annotation,
    write_bartab_counts,
    write_celltag_reads,
    write_fatemap,
    write_molecule_info,
    write_pycashier,
    write_quicat,
    write_quicat_h5ad,
    write_quicat_wide,
    write_trex_molecules,
)
from tests.memory_helpers import charges
from tests.storage_helpers import (
    copy_store,
    open_store,
    rechunk_column,
    write_count_store,
)

TRUTH = clone_truth()
BARCODES = barcode_truth(TRUTH)


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("barcode_import") / "store.zarr"
    write_truth_store(path, TRUTH)
    return path


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    """One writable copy for the imports that save artifacts of their own."""
    store = copy_store(template, tmp_path_factory.mktemp("barcode_shared"))
    return SimpleNamespace(store=store, cells=every_cell(store))


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def _records(loaded: LineageBarcodes) -> pd.DataFrame:
    """The loaded records as plain columns: cell, barcode, umi, umis, reads."""
    records = loaded.records
    return pd.DataFrame(
        {
            "cell": records["ids"].astype(str).to_numpy(),
            "barcode": records["barcode"].astype(str).to_numpy(),
            "umi": records["umi"].astype(object).where(records["umi"].notna(), None),
            "umis": records["umis"].fillna(-1).to_numpy(dtype=np.int64),
            "reads": records["reads"].fillna(-1).to_numpy(dtype=np.int64),
        }
    )


def _molecules(*, corrected: bool = False, reads: bool = True) -> pd.DataFrame:
    expected = BARCODES.expected_molecules(TRUTH, corrected=corrected)
    return pd.DataFrame(
        {
            "cell": expected["cell"].to_numpy(dtype=object),
            "barcode": expected["barcode"].to_numpy(dtype=object),
            "umi": expected["umi"].to_numpy(dtype=object),
            "umis": np.ones(len(expected), dtype=np.int64),
            "reads": expected["reads"].to_numpy(dtype=np.int64)
            if reads
            else np.full(len(expected), -1, dtype=np.int64),
        }
    )


def _pairs(*, corrected: bool = False, umis: bool = True, reads: bool = False):
    expected = BARCODES.expected_pairs(TRUTH, corrected=corrected)
    return pd.DataFrame(
        {
            "cell": expected["cell"].to_numpy(dtype=object),
            "barcode": expected["barcode"].to_numpy(dtype=object),
            "umi": np.full(len(expected), None, dtype=object),
            "umis": expected["umis"].to_numpy(dtype=np.int64)
            if umis
            else np.full(len(expected), -1, dtype=np.int64),
            "reads": expected["reads"].to_numpy(dtype=np.int64)
            if reads
            else np.full(len(expected), -1, dtype=np.int64),
        }
    )


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    """Records in one order, text as Python strings with None for no UMI."""
    frame = pd.DataFrame(
        {
            "cell": frame["cell"].astype(str).to_numpy(dtype=object),
            "barcode": frame["barcode"].astype(str).to_numpy(dtype=object),
            "umi": np.asarray(
                [None if pd.isna(umi) else str(umi) for umi in frame["umi"]],
                dtype=object,
            ),
            "umis": frame["umis"].to_numpy(dtype=np.int64),
            "reads": frame["reads"].to_numpy(dtype=np.int64),
        }
    )
    keys = frame.assign(umi_key=frame["umi"].fillna(""))
    return (
        keys.sort_values(["cell", "barcode", "umi_key"])
        .drop(columns="umi_key")
        .reset_index(drop=True)
    )


def _per_sample(
    writer: Callable[..., Path], name: str, **options
) -> Callable[[Path], list[Path]]:
    def write(folder: Path) -> list[Path]:
        return [
            writer(BARCODES, folder / f"{sample}_{name}", sample=sample, **options)
            for sample in SAMPLES
        ]

    return write


_PREFIXES = [f"{sample}__" for sample in SAMPLES]
# The molecule_info file's features are the corrected barcodes, named LB<k>.
_FEATURES = {
    sample: {
        barcode: f"LB{index}"
        for index, barcode in enumerate(
            sorted(BARCODES.sample_molecules(sample)["true_barcode"].unique())
        )
    }
    for sample in SAMPLES
}


def _feature_named(expected: pd.DataFrame) -> pd.DataFrame:
    samples = [cell.split("__", 1)[0] for cell in expected["cell"]]
    named = expected.copy()
    named["barcode"] = [
        _FEATURES[sample][barcode]
        for sample, barcode in zip(samples, expected["barcode"], strict=True)
    ]
    return named


# Each case: the writer of each sample's file (or of one file), the import's
# options, the detected format, and the records the import must give.
_CASES: dict[
    str, tuple[Callable[[Path], list[Path]], dict, str, Callable[[], pd.DataFrame]]
] = {
    "trex": (
        _per_sample(write_trex_molecules, "molecules.txt"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "trex",
        _molecules,
    ),
    "trex_corrected": (
        _per_sample(
            write_trex_molecules, "molecules_corrected.txt", generation="corrected"
        ),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "trex",
        lambda: _molecules(corrected=True),
    ),
    "trex_2022": (
        _per_sample(write_trex_molecules, "molecules.txt", generation="2022"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "trex",
        lambda: _molecules(reads=False),
    ),
    "pycashier": (
        _per_sample(write_pycashier, "test.umi_cell_labeled.barcode.tsv"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "pycashier",
        _molecules,
    ),
    "pycashier_without_header": (
        _per_sample(
            write_pycashier, "test.cell_record_labeled.barcode.tsv", header=False
        ),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "pycashier",
        _molecules,
    ),
    "celltag_rna": (
        _per_sample(write_celltag_reads, "parse.txt.gz"),
        {"barcode_prefix": _PREFIXES},
        "celltag",
        _molecules,
    ),
    "celltag_atac": (
        _per_sample(write_celltag_reads, "parse_atac.txt.gz", assay="ATAC"),
        {"barcode_prefix": _PREFIXES},
        "celltag",
        lambda: _pairs(umis=False, reads=True),
    ),
    "biddy": (
        _per_sample(write_celltag_reads, "v1.celltag.parsed.tsv", biddy=True),
        {"barcode_prefix": _PREFIXES},
        "celltag",
        _molecules,
    ),
    "fatemap": (
        _per_sample(write_fatemap, "stepThreeStarcodeShavedReads.txt"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "fatemap",
        lambda: _molecules(corrected=True, reads=False),
    ),
    "fatemap_without_header": (
        _per_sample(write_fatemap, "stepThree.txt", header=False),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "fatemap",
        lambda: _molecules(corrected=True, reads=False),
    ),
    "quicat": (
        _per_sample(write_quicat, "output.csv"),
        {"barcode_prefix": _PREFIXES},
        "quicat",
        _pairs,
    ),
    "bartab": (
        _per_sample(write_bartab_counts, "counts.tsv.gz"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "bartab",
        lambda: _pairs(corrected=True),
    ),
    "bartab_annotation": (
        _per_sample(write_bartab_annotation, "cell_barcode_annotation.tsv"),
        {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"},
        "bartab_annotation",
        lambda: _pairs(corrected=True),
    ),
    "bartab_annotation_legacy": (
        _per_sample(write_bartab_annotation, "cell-barcode-anno.tsv", legacy=True),
        {"barcode_prefix": _PREFIXES},
        "bartab_annotation",
        lambda: _pairs(corrected=True),
    ),
    "molecule_info": (
        _per_sample(write_molecule_info, "molecule_info.h5"),
        {"barcode_prefix": _PREFIXES},
        "molecule_info",
        lambda: _feature_named(_molecules(corrected=True)),
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_format_recovers_the_truth(shared, tmp_path, case) -> None:
    write, options, detected, expected = _CASES[case]
    paths = write(tmp_path)

    with collect_warnings() as messages:
        ref = shared.store.clones.import_barcodes(
            paths, cell_selection=shared.cells, **options
        )

    loaded = shared.store.clones.load_barcodes(ref)
    pd.testing.assert_frame_equal(_sorted(_records(loaded)), _sorted(expected()))
    sources = shared.store.artifacts.inspect(ref).inputs["sources"]
    assert {source["format"] for source in sources} == {detected}
    # The droplets outside the store are d2's file's only unmatched cells, and
    # every cell of the store is selected.
    diagnostics = shared.store.artifacts.inspect(ref).diagnostics
    assert diagnostics["unmatchedBarcodes"] == len(TRUTH.outside)
    assert diagnostics["unselectedBarcodes"] == 0
    shown = example_ids(diagnostics["unmatchedBarcodeExamples"])
    assert len(shown) == min(5, len(TRUTH.outside))
    assert all(any(drop in cell for drop in TRUTH.outside) for cell in shown)
    assert len(list(messages)) == 1


def _wide_options(prefix_pattern: str) -> dict:
    """List a merged QuiCAT file once per sample, keeping that sample's cells."""
    return {
        "barcode_prefix": _PREFIXES,
        "barcode_pattern": [prefix_pattern.format(sample=sample) for sample in SAMPLES],
    }


@pytest.mark.parametrize(
    "write",
    [
        lambda folder: write_quicat_wide(BARCODES, folder / "barcodes_output.csv"),
        lambda folder: write_quicat_h5ad(BARCODES, folder / "barcodes_output.h5ad"),
        lambda folder: write_quicat_h5ad(
            BARCODES, folder / "barcodes_output.h5ad", layout="csc"
        ),
        lambda folder: write_quicat_h5ad(
            BARCODES, folder / "barcodes_output.h5ad", layout="dense"
        ),
        lambda folder: write_quicat_h5ad(
            BARCODES, folder / "barcodes_output.h5ad", with_anndata=True
        ),
        lambda folder: write_quicat_h5ad(
            BARCODES, folder / "barcodes_output.h5ad", layout="csc", with_anndata=True
        ),
        lambda folder: write_quicat_h5ad(
            BARCODES,
            folder / "barcodes_output.h5ad",
            layout="dense",
            with_anndata=True,
        ),
    ],
    ids=[
        "wide_csv",
        "h5ad_csr",
        "h5ad_csc",
        "h5ad_dense",
        "anndata_csr",
        "anndata_csc",
        "anndata_dense",
    ],
)
def test_quicats_merged_files_name_cells_with_their_sample(
    shared, tmp_path, write
) -> None:
    path = write(tmp_path)
    # QuiCAT names merged cells "<barcode>-1_<sample>"; a merged store names
    # them "<sample>__<barcode>-1".
    with collect_warnings():
        ref = shared.store.clones.import_barcodes(
            [path, path],
            cell_selection=shared.cells,
            **_wide_options(r"^(?P<barcode>[ACGT]+-1)_{sample}$"),
        )

    loaded = shared.store.clones.load_barcodes(ref)
    pd.testing.assert_frame_equal(_sorted(_records(loaded)), _sorted(_pairs()))
    assert loaded.libraries["library"].tolist() == ["barcodes_output"]


def test_counts_of_cells_and_libraries_follow_the_records(shared, tmp_path) -> None:
    paths = _per_sample(write_trex_molecules, "molecules.txt")(tmp_path)
    ref = shared.store.clones.import_barcodes(
        paths,
        cell_selection=shared.cells,
        barcode_prefix=_PREFIXES,
        barcode_suffix="-1",
        library="lineage",
    )

    loaded = shared.store.clones.load_barcodes(ref)
    expected = BARCODES.expected_molecules(TRUTH)
    by_cell = expected.groupby("cell")
    cells = loaded.cells
    assert cells.index.tolist() == TRUTH.cell_ids.tolist()
    np.testing.assert_array_equal(
        cells["n_barcodes"],
        by_cell["barcode"].nunique().reindex(cells.index, fill_value=0),
    )
    np.testing.assert_array_equal(
        cells["n_umis"], by_cell.size().reindex(cells.index, fill_value=0)
    )
    np.testing.assert_array_equal(
        cells["n_reads"], by_cell["reads"].sum().reindex(cells.index, fill_value=0)
    )
    np.testing.assert_array_equal(cells["n_records"], cells["n_umis"])
    assert loaded.libraries.to_dict("list") == {
        "library": ["lineage"],
        "level": ["molecules"],
        "n_records": [len(expected)],
        "has_umis": [True],
        "has_reads": [True],
    }
    # The canonical per-cell values are the distinct barcodes.
    values = shared.store.artifacts.load_values(ref)
    np.testing.assert_array_equal(values.values, cells["n_barcodes"])
    pairs = loaded.pairs()
    observed = (
        pd.DataFrame(
            {
                "cell": pairs["ids"].astype(str).to_numpy(dtype=object),
                "barcode": pairs["barcode"].astype(str).to_numpy(dtype=object),
                "umis": pairs["umis"].to_numpy(dtype=np.int64),
                "reads": pairs["reads"].to_numpy(dtype=np.int64),
                "molecules": pairs["n_molecules"].to_numpy(dtype=np.int64),
            }
        )
        .sort_values(["cell", "barcode"])
        .reset_index(drop=True)
    )
    expected_pairs = BARCODES.expected_pairs(TRUTH)
    assert observed["cell"].tolist() == expected_pairs["cell"].tolist()
    assert observed["barcode"].tolist() == expected_pairs["barcode"].tolist()
    assert observed["umis"].tolist() == expected_pairs["umis"].tolist()
    assert observed["molecules"].tolist() == expected_pairs["umis"].tolist()
    assert observed["reads"].tolist() == expected_pairs["reads"].tolist()


def test_rna_molecules_and_atac_reads_share_an_import_in_two_libraries(
    shared, tmp_path
) -> None:
    rna = _per_sample(write_celltag_reads, "rna.txt.gz")(tmp_path)
    atac = _per_sample(write_celltag_reads, "atac.txt.gz", assay="ATAC")(tmp_path)

    with collect_warnings():
        ref = shared.store.clones.import_barcodes(
            [*rna, *atac],
            cell_selection=shared.cells,
            barcode_prefix=_PREFIXES * 2,
            library=["RNA", "RNA", "ATAC", "ATAC"],
        )

    loaded = shared.store.clones.load_barcodes(ref)
    assert loaded.libraries.to_dict("list") == {
        "library": ["ATAC", "RNA"],
        "level": ["pairs", "molecules"],
        "n_records": [len(_pairs()), len(_molecules())],
        "has_umis": [False, True],
        "has_reads": [True, True],
    }
    # A cell's UMIs are unknown, since its ATAC pairs count reads only.
    assert (loaded.cells["n_umis"][loaded.cells["n_records"] > 0] == -1).all()
    rna_records = _records(loaded)[loaded.records["library"] == "RNA"]
    pd.testing.assert_frame_equal(_sorted(rna_records), _sorted(_molecules()))


def test_a_file_lets_its_names_and_chunk_go_once_it_is_counted(
    shared, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import lineage_barcodes as operations

    # The reads of one sample in two files, counted in turn. Reading a file
    # holds its names and a chunk of its rows, which go once the file is
    # counted, so a file leaves charged only what the counter's records grew
    # by at their merge, beside the join's hashes of barcodes that match no
    # selected cell, of which these files name none.
    path = write_pycashier(BARCODES, tmp_path / "all.tsv", sample="d4")
    lines = path.read_text().splitlines()
    first = tmp_path / "lane1.tsv"
    second = tmp_path / "lane2.tsv"
    first.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    second.write_text("\n".join([lines[0], *lines[len(lines) // 2 :]]) + "\n")
    mixin = operations._LineageBarcodeOperationsMixin
    count_file = mixin._count_file
    left: list[tuple[int, int]] = []

    def counted_file(self, counter, join, file, layout, *, ledger, counted):
        before = ledger.held_bytes
        result = count_file(
            self, counter, join, file, layout, ledger=ledger, counted=counted
        )
        assert join.n_unmatched_barcodes == 0
        left.append((ledger.held_bytes - before, result[1] - counted))
        return result

    monkeypatch.setattr(mixin, "_count_file", counted_file)
    shared.store.clones.import_barcodes(
        [first, second],
        cell_selection=shared.cells,
        barcode_prefix="d4__",
        barcode_suffix="-1",
        library="d4",
        invalidate_cache=True,
    )
    assert len(left) == 2 and left[0][1] > 0
    assert all(held == grown for held, grown in left), left


def test_a_library_of_molecules_and_pairs_is_refused(shared, tmp_path) -> None:
    rna = write_celltag_reads(BARCODES, tmp_path / "rna.txt.gz", sample="d2")
    atac = write_celltag_reads(
        BARCODES, tmp_path / "atac.txt.gz", sample="d2", assay="ATAC"
    )
    with pytest.raises(ValueError, match="holds molecules in one file and barcode"):
        shared.store.clones.import_barcodes(
            [rna, atac], cell_selection=shared.cells, barcode_prefix="d2__", library="X"
        )


def test_reads_of_one_library_in_two_files_merge_into_its_molecules(
    shared, tmp_path
) -> None:
    # The reads of one sample, split between two files, as two lanes are.
    path = write_pycashier(BARCODES, tmp_path / "all.tsv", sample="d4")
    lines = path.read_text().splitlines()
    first = tmp_path / "lane1.tsv"
    second = tmp_path / "lane2.tsv"
    first.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    second.write_text("\n".join([lines[0], *lines[len(lines) // 2 :]]) + "\n")

    ref = shared.store.clones.import_barcodes(
        [first, second],
        cell_selection=shared.cells,
        barcode_prefix="d4__",
        barcode_suffix="-1",
        library="d4",
    )

    expected = _molecules()
    expected = expected[expected["cell"].str.startswith("d4__")]
    loaded = shared.store.clones.load_barcodes(ref)
    pd.testing.assert_frame_equal(
        _sorted(_records(loaded)), _sorted(expected.reset_index(drop=True))
    )


def test_trexs_reads_table_is_refused_for_repeating_its_molecules(
    shared, tmp_path
) -> None:
    path = write_trex_molecules(
        BARCODES, tmp_path / "reads.txt", sample="d4", generation="reads"
    )
    with pytest.raises(ValueError, match="as TREX's reads.txt lists each read"):
        shared.store.clones.import_barcodes(
            path,
            cell_selection=shared.cells,
            barcode_prefix="d4__",
            barcode_suffix="-1",
        )


def test_tables_read_by_columns_give_reads_molecules_and_pairs(
    shared, tmp_path
) -> None:
    reads = BARCODES.reads("d4")
    path = tmp_path / "reads.csv"
    pd.DataFrame(
        {"cb": reads["cell_barcode"], "ub": reads["umi"], "lineage": reads["barcode"]}
    ).to_csv(path, index=False)
    columns = {"cell": "cb", "umi": "ub", "barcode": "lineage"}
    options = {"cell_selection": shared.cells, "barcode_prefix": "d4__"}

    as_reads = shared.store.clones.import_barcodes(
        path, format="reads", columns=columns, barcode_suffix="-1", **options
    )

    expected = _molecules()
    expected = expected[expected["cell"].str.startswith("d4__")].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        _sorted(_records(shared.store.clones.load_barcodes(as_reads))),
        _sorted(expected),
    )
    # Read as molecules without reads, each read is a molecule: equal rows
    # name one molecule, whose reads are unknown.
    as_molecules = shared.store.clones.import_barcodes(
        path, format="molecules", columns=columns, barcode_suffix="-1", **options
    )
    pd.testing.assert_frame_equal(
        _sorted(_records(shared.store.clones.load_barcodes(as_molecules))),
        _sorted(expected.assign(reads=-1)),
    )


def test_two_barcode_columns_join_as_clonetracker_xps_halves(shared, tmp_path) -> None:
    pairs = BARCODES.expected_pairs(TRUTH)
    pairs = pairs[pairs["cell"].str.startswith("d2__")]
    path = tmp_path / "clonetracker.tsv"
    pd.DataFrame(
        {
            "cell": pairs["cell"].str.removeprefix("d2__"),
            "bc14": pairs["barcode"].str[:8],
            "bc30": pairs["barcode"].str[8:],
            "n": pairs["umis"],
        }
    ).to_csv(path, sep="\t", index=False)

    ref = shared.store.clones.import_barcodes(
        path,
        cell_selection=shared.cells,
        barcode_prefix="d2__",
        format="pairs",
        columns={"cell": "cell", "barcode": ["bc14", "bc30"], "umis": "n"},
    )

    records = _records(shared.store.clones.load_barcodes(ref))
    joined = (pairs["barcode"].str[:8] + "__" + pairs["barcode"].str[8:]).tolist()
    assert sorted(zip(records["cell"], records["barcode"], strict=True)) == sorted(
        zip(pairs["cell"], joined, strict=True)
    )
    assert sorted(records["umis"]) == sorted(pairs["umis"])


@pytest.mark.parametrize("separator", [";", ",", ":", "|"])
def test_list_fields_split_into_pairs_with_their_counts(
    shared, tmp_path, separator
) -> None:
    pairs = BARCODES.expected_pairs(TRUTH)
    pairs = pairs[pairs["cell"].str.startswith("d4__")]
    path = tmp_path / "lists.tsv"
    rows = [
        (
            cell.removeprefix("d4__"),
            separator.join(group["barcode"]),
            separator.join(str(count) for count in group["umis"]),
        )
        for cell, group in pairs.groupby("cell")
    ]
    # A cell without a barcode lists none.
    rows.append(("TTTTTTTTTTTTTTTT-1", "NA", "NA"))
    pd.DataFrame(rows, columns=["cell", "barcodes", "umis"]).to_csv(
        path, sep="\t", index=False
    )

    with collect_warnings() as messages:
        ref = shared.store.clones.import_barcodes(
            path,
            cell_selection=shared.cells,
            barcode_prefix="d4__",
            format="pairs",
            columns={
                "cell": "cell",
                "barcode": "barcodes",
                "umis": "umis",
                "separator": separator,
            },
        )

    records = _records(shared.store.clones.load_barcodes(ref))
    assert sorted(
        zip(records["cell"], records["barcode"], records["umis"], strict=True)
    ) == sorted(zip(pairs["cell"], pairs["barcode"], pairs["umis"], strict=True))
    assert "1 of" in list(messages)[0]


def test_lists_whose_counts_do_not_align_are_refused_with_their_lines(
    shared, tmp_path
) -> None:
    path = tmp_path / "lists.tsv"
    path.write_text("cell\tbarcodes\tumis\nAAAA-1\tx;y\t3;4\nCCCC-1\tx;y\t3\n")
    with pytest.raises(
        ValueError, match="another number of umis than of barcodes on line 3"
    ):
        shared.store.clones.import_barcodes(
            path,
            cell_selection=shared.cells,
            format="pairs",
            columns={
                "cell": "cell",
                "barcode": "barcodes",
                "umis": "umis",
                "separator": ";",
            },
        )


def test_barcodes_counted_as_an_assays_features_are_pairs(tmp_path) -> None:
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.6, size=(N_CELLS, 5)).astype(np.uint16)
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((N_CELLS, 3), dtype=np.uint16), "LINEAGE": counts},
        "uint16",
        cell_ids=TRUTH.cell_ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    cells = store.snapshot_cell_selection("every_cell")

    ref = store.clones.import_barcodes(cell_selection=cells, from_assay="LINEAGE")

    loaded = store.clones.load_barcodes(ref)
    rows, columns = np.nonzero(counts)
    expected = pd.DataFrame(
        {
            "cell": TRUTH.cell_ids[rows],
            "barcode": [f"LINEAGE{column}" for column in columns],
            "umi": None,
            "umis": counts[rows, columns].astype(np.int64),
            "reads": -1,
        }
    )
    pd.testing.assert_frame_equal(
        _sorted(_records(loaded)), _sorted(expected.astype({"reads": np.int64}))
    )
    assert loaded.libraries["library"].tolist() == ["LINEAGE"]
    # Reused with the same assay counts.
    assert (
        store.clones.import_barcodes(cell_selection=cells, from_assay="LINEAGE") == ref
    )
    named = store.clones.import_barcodes(
        cell_selection=cells, from_assay="LINEAGE", library="larry"
    )
    assert named != ref
    assert store.clones.load_barcodes(named).libraries["library"].tolist() == ["larry"]


def test_an_assay_import_refuses_file_options(tmp_path) -> None:
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((N_CELLS, 3), dtype=np.uint16)},
        "uint16",
        cell_ids=TRUTH.cell_ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    cells = store.snapshot_cell_selection("every_cell")
    with pytest.raises(ValueError, match="not both"):
        store.clones.import_barcodes(path, cell_selection=cells, from_assay="RNA")
    with pytest.raises(ValueError, match="barcode_suffix, format would change"):
        store.clones.import_barcodes(
            cell_selection=cells, from_assay="RNA", barcode_suffix="-1", format="trex"
        )
    with pytest.raises(ValueError, match="paths must name the barcode files"):
        store.clones.import_barcodes(cell_selection=cells)
    with pytest.raises(ValueError, match="Assay 'ABSENT' not found"):
        store.clones.import_barcodes(cell_selection=cells, from_assay="ABSENT")
    with pytest.raises(TypeError, match="library must be text"):
        store.clones.import_barcodes(
            cell_selection=cells, from_assay="RNA", library=["a"]
        )
    with pytest.raises(ValueError, match="library must name a library"):
        store.clones.import_barcodes(cell_selection=cells, from_assay="RNA", library="")


def test_import_is_reused_by_contents_and_warns_again(store, tmp_path) -> None:
    paths = _per_sample(write_trex_molecules, "molecules.txt")(tmp_path)
    cells = every_cell(store)
    options = {"barcode_prefix": _PREFIXES, "barcode_suffix": "-1"}
    with collect_warnings() as first:
        ref = store.clones.import_barcodes(paths, cell_selection=cells, **options)
    moved = tmp_path / "moved"
    moved.mkdir()
    for path in paths:
        (moved / path.name).write_bytes(path.read_bytes())
    with collect_warnings() as again:
        assert (
            store.clones.import_barcodes(
                [moved / path.name for path in paths], cell_selection=cells, **options
            )
            == ref
        )
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    with collect_warnings() as read_only:
        assert (
            reader.clones.import_barcodes(paths, cell_selection=cells, **options) == ref
        )
    assert list(first) == list(again) == list(read_only)
    assert "3 of" in list(first)[0]
    paths[0].write_text(paths[0].read_text().replace("\t1\n", "\t2\n", 1))
    with pytest.raises(PermissionError, match="import_lineage_barcodes"):
        reader.clones.import_barcodes(paths, cell_selection=cells, **options)
    assert store.clones.import_barcodes(paths, cell_selection=cells, **options) != ref


def test_a_subset_of_cells_holds_only_its_barcodes(shared, tmp_path) -> None:
    paths = _per_sample(write_trex_molecules, "molecules.txt")(tmp_path)
    samples = TRUTH.samples
    shared.store.cells.insert("in_d4", samples == "d4", overwrite=True)
    cells = shared.store.snapshot_cell_selection("in_d4")

    with collect_warnings() as messages:
        ref = shared.store.clones.import_barcodes(
            paths, cell_selection=cells, barcode_prefix=_PREFIXES, barcode_suffix="-1"
        )

    loaded = shared.store.clones.load_barcodes(ref)
    assert loaded.cells.index.tolist() == TRUTH.cell_ids[samples == "d4"].tolist()
    expected = _molecules()
    expected = expected[expected["cell"].str.startswith("d4__")].reset_index(drop=True)
    pd.testing.assert_frame_equal(_sorted(_records(loaded)), _sorted(expected))
    # The d2 file's cells are cells of the store outside the selection, and
    # its droplets outside the store match no cell.
    n_d2 = int(np.count_nonzero(samples == "d2"))
    listed = {
        f"{sample}__{barcode}-1"
        for sample in SAMPLES
        for barcode in BARCODES.sample_molecules(sample)["cell_barcode"]
    }
    stored = set(TRUTH.cell_ids.tolist())
    unselected = sorted(cell for cell in listed & stored if cell.startswith("d2__"))
    assert len(unselected) >= n_d2 // 2
    assert_joined(
        shared.store.artifacts.inspect(ref).diagnostics,
        n_file=len(listed),
        unmatched=sorted(listed - stored),
        unselected=unselected,
    )
    assert list(messages)


def test_a_file_without_barcodes_of_the_selected_cells_saves_none(
    shared, tmp_path
) -> None:
    path = tmp_path / "molecules.txt"
    path.write_text("#cell_id\tumi\tclone_id\tread_count\n")
    ref = shared.store.clones.import_barcodes(path, cell_selection=shared.cells)
    loaded = shared.store.clones.load_barcodes(ref)
    assert loaded.records.empty
    assert (loaded.cells["n_barcodes"] == 0).all()
    assert (loaded.cells["n_umis"] == 0).all()
    assert loaded.libraries.to_dict("list") == {
        "library": ["molecules"],
        "level": ["molecules"],
        "n_records": [0],
        "has_umis": [False],
        "has_reads": [False],
    }
    assert loaded.pairs().empty


def test_rows_without_a_barcode_or_counted_zero_hold_none(shared, tmp_path) -> None:
    cell = TRUTH.cell_ids[0]
    path = tmp_path / "pairs.csv"
    path.write_text(
        "cell,barcode,umis,reads\n"
        f"{cell},AAAA,3,9\n{cell},NA,4,4\n{cell},CCCC,0,5\n{cell},GGGG,,0\n"
        f"{cell},TTTT,,2\n"
    )
    ref = shared.store.clones.import_barcodes(
        path,
        cell_selection=shared.cells,
        format="pairs",
        columns={
            "cell": "cell",
            "barcode": "barcode",
            "umis": "umis",
            "reads": "reads",
        },
    )
    records = _records(shared.store.clones.load_barcodes(ref))
    # TTTT's UMIs are unknown, and its reads count.
    assert records[["barcode", "umis", "reads"]].values.tolist() == [
        ["AAAA", 3, 9],
        ["GGGG", -1, 0],
        ["TTTT", -1, 2],
    ]
    assert shared.store.artifacts.load(ref).attrs["n_empty_rows"] == 2


def test_import_refuses_a_budget_too_small_and_writes_nothing(
    store, tmp_path, monkeypatch
) -> None:
    paths = _per_sample(write_pycashier, "reads.tsv")(tmp_path)
    cells = every_cell(store)
    monkeypatch.setattr(store, "memoryBytes", 60_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.clones.import_barcodes(
            paths, cell_selection=cells, barcode_prefix=_PREFIXES, barcode_suffix="-1"
        )
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == []
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(
        MemoryError, match="Importing barcodes for 48 selected cells holds about"
    ):
        store.clones.import_barcodes(
            paths, cell_selection=cells, barcode_prefix=_PREFIXES, barcode_suffix="-1"
        )


def test_a_read_level_file_of_many_chunks_holds_its_molecules_not_its_reads(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module
    from cytearc.readers import lineage_barcodes as reader

    # One sample's molecules read 40 and 400 times: the second file is ten
    # times longer, and both hold the same molecules.
    molecules = BARCODES.sample_molecules("d4")
    peaks = []
    for depth in (40, 400):
        path = tmp_path / f"depth{depth}.tsv"
        rows = molecules.loc[molecules.index.repeat(depth)].sample(
            frac=1, random_state=1
        )
        rows[["cell_barcode", "umi", "barcode"]].rename(
            columns={"cell_barcode": "cell"}
        ).assign(info="@r").loc[:, ["info", "umi", "cell", "barcode"]].to_csv(
            path, sep="\t", index=False
        )
        held: list[int] = []
        original = barcodes_module.BarcodeCounter.add

        def add(self, *args, _held=held, _original=original, **kwargs):
            _original(self, *args, **kwargs)
            _held.append(self.n_rows)

        monkeypatch.setattr(barcodes_module.BarcodeCounter, "add", add)
        # Chunks of 200 rows, merged once 1,000 rows wait.
        monkeypatch.setattr(reader, "_MAX_CHUNK_ROWS", 200)
        monkeypatch.setattr(store, "_merge_rows", lambda: 1_000)
        ref = store.clones.import_barcodes(
            path,
            cell_selection=every_cell(store),
            barcode_prefix="d4__",
            barcode_suffix="-1",
            invalidate_cache=True,
        )
        monkeypatch.undo()
        loaded = store.clones.load_barcodes(ref)
        assert len(loaded.records) == len(molecules)
        assert (loaded.records["reads"] == depth).all()
        assert len(held) == -(-len(molecules) * depth // 200)
        peaks.append(max(held))
    # The counter holds the molecules, the rows that wait to merge and one
    # chunk's, however many reads the file lists.
    assert peaks[0] <= len(molecules) + 1_000 + 200
    assert peaks[1] <= len(molecules) + 1_000 + 200


def test_load_refuses_other_artifacts_and_tables_larger_than_the_budget(
    store, tmp_path, monkeypatch
) -> None:
    path = write_trex_molecules(BARCODES, tmp_path / "molecules.txt", sample="d2")
    cells = every_cell(store)
    with collect_warnings():
        ref = store.clones.import_barcodes(
            path, cell_selection=cells, barcode_prefix="d2__", barcode_suffix="-1"
        )
    with pytest.raises(ValueError, match="must be a lineage_barcodes artifact"):
        store.clones.load_barcodes(cells)
    with pytest.raises(TypeError, match="barcodes must be an ArtifactRef"):
        store.clones.load_barcodes("barcodes")
    monkeypatch.setattr(store, "memoryBytes", 2_000)
    with pytest.raises(MemoryError, match="The tables of 48 cells"):
        store.clones.load_barcodes(ref)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (
            {"cell_selection": "cells"},
            TypeError,
            "cell_selection must be an ArtifactRef",
        ),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be a boolean"),
        ({"format": 3}, TypeError, "format must be a format name"),
        ({"format": "larry"}, ValueError, "format must be one of"),
        ({"library": 3}, TypeError, "library must be text"),
        ({"library": ["a", "b"]}, ValueError, "lists 2 values for 1 barcode files"),
        (
            {"columns": {"cell": "a"}},
            ValueError,
            "columns names the columns of a table",
        ),
        (
            {"format": "pairs", "columns": {"cel": "a"}},
            ValueError,
            "unknown roles 'cel'",
        ),
    ],
)
def test_import_refuses_arguments_it_cannot_use(
    shared, tmp_path, arguments, error, message
) -> None:
    path = write_trex_molecules(BARCODES, tmp_path / "molecules.txt", sample="d2")
    call = {"cell_selection": shared.cells, **arguments}
    with pytest.raises(error, match=message):
        shared.store.clones.import_barcodes(path, **call)


def test_import_refuses_files_it_cannot_read(shared, tmp_path) -> None:
    clones = tmp_path / "clones.txt"
    clones.write_text("clone_nr\tcell_id\n1\tAAAC\n")
    with pytest.raises(ValueError, match="import it with ds.clones.import_clones"):
        shared.store.clones.import_barcodes(clones, cell_selection=shared.cells)
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="format='reads', 'molecules' or 'pairs'"):
        shared.store.clones.import_barcodes(unknown, cell_selection=shared.cells)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        shared.store.clones.import_barcodes(empty, cell_selection=shared.cells)
    compressed = tmp_path / "pycashier.tsv.gz"
    compressed.write_bytes(
        gzip.compress(b"info\tumi\tcell\tbarcode\n@r\tAC_G\tAAAC\tGG\n")
    )
    with pytest.raises(ValueError, match="a read name that holds '_' shifts"):
        shared.store.clones.import_barcodes(compressed, cell_selection=shared.cells)
    with pytest.raises(FileNotFoundError):
        shared.store.clones.import_barcodes(
            tmp_path / "absent.txt", cell_selection=shared.cells
        )


def test_import_refuses_an_empty_selection_and_assays_without_integer_counts(
    tmp_path,
) -> None:
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {
            "RNA": np.ones((N_CELLS, 3), dtype=np.float32),
            "LINEAGE": np.full((N_CELLS, 2), 0.5, dtype=np.float32),
        },
        "float32",
        cell_ids=TRUTH.cell_ids,
    )
    store = open_store(path)
    store.cells.insert("no_cell", np.zeros(N_CELLS, dtype=bool))
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    none = store.snapshot_cell_selection("no_cell")
    cells = store.snapshot_cell_selection("every_cell")
    counts = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d2")
    with pytest.raises(ValueError, match="cell_selection selects no cell"):
        store.clones.import_barcodes(counts, cell_selection=none)
    with pytest.raises(TypeError, match="must be integers, not float32"):
        store.clones.import_barcodes(cell_selection=cells, from_assay="LINEAGE")
    with pytest.raises(TypeError, match="from_assay must be an assay name"):
        store.clones.import_barcodes(cell_selection=cells, from_assay=3)


def test_load_refuses_an_artifact_whose_libraries_are_not_recorded(
    store, tmp_path
) -> None:
    from cytearc.storage.artifacts import artifact_group

    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    ref = store.clones.import_barcodes(
        path,
        cell_selection=every_cell(store),
        barcode_prefix="d4__",
        barcode_suffix="-1",
    )
    artifact_group(store.zw, ref).attrs["libraries"] = [{"library": "x"}]
    with pytest.raises(ValueError, match="no valid list of libraries"):
        store.clones.load_barcodes(ref)


def test_a_trex_table_and_amplicon_reads_of_one_library_merge(shared, tmp_path) -> None:
    # TREX lists each molecule once; pycashier's reads of the same library add
    # one read per row to the same molecules.
    trex = write_trex_molecules(BARCODES, tmp_path / "molecules.txt", sample="d4")
    reads = write_pycashier(BARCODES, tmp_path / "amplicon.tsv", sample="d4")

    ref = shared.store.clones.import_barcodes(
        [trex, reads],
        cell_selection=shared.cells,
        barcode_prefix="d4__",
        barcode_suffix="-1",
        library="d4",
    )

    expected = _molecules()
    expected = expected[expected["cell"].str.startswith("d4__")].reset_index(drop=True)
    records = _records(shared.store.clones.load_barcodes(ref))
    pd.testing.assert_frame_equal(
        _sorted(records), _sorted(expected.assign(reads=2 * expected["reads"]))
    )


def test_two_trex_runs_of_one_library_merge_their_reads(shared, tmp_path) -> None:
    first = write_trex_molecules(BARCODES, tmp_path / "run1.txt", sample="d2")
    second = tmp_path / "run2.txt"
    second.write_text(first.read_text() + "\n")

    with collect_warnings():
        ref = shared.store.clones.import_barcodes(
            [first, second],
            cell_selection=shared.cells,
            barcode_prefix="d2__",
            barcode_suffix="-1",
            library="d2",
        )

    expected = _molecules()
    expected = expected[expected["cell"].str.startswith("d2__")].reset_index(drop=True)
    records = _records(shared.store.clones.load_barcodes(ref))
    pd.testing.assert_frame_equal(
        _sorted(records), _sorted(expected.assign(reads=2 * expected["reads"]))
    )


def test_a_trex_reads_table_beside_molecules_is_named_in_the_refusal(
    shared, tmp_path
) -> None:
    molecules = write_trex_molecules(BARCODES, tmp_path / "molecules.txt", sample="d4")
    reads = write_trex_molecules(
        BARCODES, tmp_path / "reads.txt", sample="d4", generation="reads"
    )
    with pytest.raises(ValueError, match="reads.txt lists .* on several rows"):
        shared.store.clones.import_barcodes(
            [molecules, reads],
            cell_selection=shared.cells,
            barcode_prefix="d4__",
            barcode_suffix="-1",
            library=["d4", "other"],
        )


def test_barcodes_of_an_assay_join_cells_across_its_blocks(
    tmp_path, monkeypatch
) -> None:
    from cytearc.matrix.chunked import ChunkedArray

    rng = np.random.default_rng(8)
    counts = rng.poisson(0.7, size=(N_CELLS, 4)).astype(np.uint16)
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((N_CELLS, 3), dtype=np.uint16), "LINEAGE": counts},
        "uint16",
        cell_ids=TRUTH.cell_ids,
    )
    store = open_store(path)
    keep = np.arange(N_CELLS) % 3 != 0
    store.cells.insert("kept", keep)
    cells = store.snapshot_cell_selection("kept")
    stream = ChunkedArray._stream_blocks

    def in_blocks_of_five(self, **options):
        for block in stream(self, **options):
            for start in range(0, block.shape[0], 5):
                yield block[start : start + 5]

    monkeypatch.setattr(ChunkedArray, "_stream_blocks", in_blocks_of_five)
    ref = store.clones.import_barcodes(cell_selection=cells, from_assay="LINEAGE")

    kept = counts[keep]
    rows, columns = np.nonzero(kept)
    expected = sorted(
        zip(
            TRUTH.cell_ids[keep][rows],
            [f"LINEAGE{column}" for column in columns],
            kept[rows, columns].tolist(),
            strict=True,
        )
    )
    records = _records(store.clones.load_barcodes(ref))
    assert (
        sorted(zip(records["cell"], records["barcode"], records["umis"], strict=True))
        == expected
    )


def test_rows_of_unselected_cells_are_unmatched_rather_than_empty(
    shared, tmp_path
) -> None:
    path = tmp_path / "pairs.csv"
    cell = TRUTH.cell_ids[0]
    path.write_text(
        f"cell,barcode,umis\n{cell},AAAA,3\noutside-1,NA,2\noutside-2,,1\n{cell},,4\n"
    )
    with collect_warnings():
        ref = shared.store.clones.import_barcodes(
            path,
            cell_selection=shared.cells,
            format="pairs",
            columns={"cell": "cell", "barcode": "barcode", "umis": "umis"},
        )
    attributes = shared.store.artifacts.load(ref).attrs
    assert (attributes["n_file_rows"], attributes["n_unmatched_rows"]) == (4, 2)
    assert attributes["n_empty_rows"] == 1


def test_pairs_counted_by_reads_alone_hold_no_barcode_at_zero(shared, tmp_path) -> None:
    cell = TRUTH.cell_ids[0]
    path = tmp_path / "pairs.csv"
    path.write_text(f"cell,barcode,reads\n{cell},AAAA,5\n{cell},CCCC,0\n")
    ref = shared.store.clones.import_barcodes(
        path,
        cell_selection=shared.cells,
        format="pairs",
        columns={"cell": "cell", "barcode": "barcode", "reads": "reads"},
    )
    records = _records(shared.store.clones.load_barcodes(ref))
    assert records[["barcode", "umis", "reads"]].values.tolist() == [["AAAA", -1, 5]]
    assert shared.store.artifacts.load(ref).attrs["n_empty_rows"] == 1


def test_the_budget_check_counts_the_lookup_and_a_chunk_beside_the_records(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module

    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    # Beyond the read of a stored chunk of the cells' IDs, which is checked
    # before the IDs are read.
    budget = 64 * 2**20
    monkeypatch.setattr(store, "memoryBytes", budget)
    # The records alone fit the budget; with the lookup of the cells and a
    # chunk of text beside them, they do not. Before the first row, the
    # lookup and a chunk fit.
    monkeypatch.setattr(
        barcodes_module.BarcodeCounter,
        "peak_bytes",
        property(lambda self: budget - 64 if self.n_rows else 0),
    )
    with pytest.raises(MemoryError, match="molecules and pairs at their merge"):
        store.clones.import_barcodes(
            path,
            cell_selection=every_cell(store),
            barcode_prefix="d4__",
            barcode_suffix="-1",
        )
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == []


def test_a_small_budget_reads_a_file_in_several_chunks(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module
    from cytearc.datastore._operations import lineage_barcodes as operations

    path = write_pycashier(BARCODES, tmp_path / "reads.tsv", sample="d4")
    n_reads = len(path.read_text().splitlines()) - 1
    chunks = []
    original = barcodes_module.BarcodeCounter.add

    def add(self, cells, *args, **kwargs):
        chunks.append(len(cells))
        return original(self, cells, *args, **kwargs)

    monkeypatch.setattr(barcodes_module.BarcodeCounter, "add", add)
    # A budget beyond the read of a stored chunk of the cells' IDs, of which a
    # chunk of the file may take a 640th: the chunks of a 2 MiB budget.
    monkeypatch.setattr(store, "memoryBytes", 64 * 2**20)
    monkeypatch.setattr(operations, "_CHUNK_SHARE", 640)
    store.clones.import_barcodes(
        path,
        cell_selection=every_cell(store),
        barcode_prefix="d4__",
        barcode_suffix="-1",
    )
    assert n_reads > 500
    assert len(chunks) > 10


def test_an_empty_library_name_is_refused(shared, tmp_path) -> None:
    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    with pytest.raises(ValueError, match="library must name a library"):
        shared.store.clones.import_barcodes(
            path, cell_selection=shared.cells, barcode_prefix="d4__", library=""
        )
    with pytest.raises(ValueError, match="library must name a library"):
        shared.store.clones.import_barcodes(
            [path, path],
            cell_selection=shared.cells,
            barcode_prefix=["d2__", "d4__"],
            library=["a", ""],
        )


def test_a_file_whose_reading_cannot_fit_the_budget_is_refused_before_a_row(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.readers import lineage_barcodes as reader

    path = write_quicat_h5ad(BARCODES, tmp_path / "barcodes_output.h5ad")
    read = []
    iterate = reader.iter_barcode_rows

    def recorded(layout, chunk_rows):
        read.append(chunk_rows)
        return iterate(layout, chunk_rows)

    monkeypatch.setattr(reader, "iter_barcode_rows", recorded)
    # Names of more cells and barcodes than the budget holds.
    monkeypatch.setattr(
        reader.BarcodeLayout, "held_bytes", lambda self: store.memoryBytes
    )
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.clones.import_barcodes(path, cell_selection=every_cell(store))
    assert read == []
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == []


def _held_at(store, run: Callable[[], object], item: str) -> int:
    """What the import's ledger planned at its first charge of ``item``, in a run at the store's budget."""
    _result, taken = charges(run)
    return next(planned for name, _nbytes, planned in taken if name == item)


def _most_through(store, run: Callable[[], object], item: str) -> int:
    """The most that the import's ledger planned up to its first charge of ``item``, in a run at the store's budget."""
    _result, taken = charges(run)
    plans = []
    for name, _nbytes, planned in taken:
        plans.append(planned)
        if name == item:
            return max(plans)
    raise AssertionError(f"No charge of {item!r}")


def _refused_before_reading(
    store, path, monkeypatch, *, match: str = "Raise mem_budget", **options
) -> list:
    """Import ``path`` expecting a refusal; return the chunk sizes that were read."""
    from cytearc.readers import lineage_barcodes as reader

    read: list = []
    iterate = reader.iter_barcode_rows

    def recorded(layout, chunk_rows):
        read.append(chunk_rows)
        return iterate(layout, chunk_rows)

    monkeypatch.setattr(reader, "iter_barcode_rows", recorded)
    saved = store.artifacts.list(kind="lineage_barcodes", scope="datastore")
    with pytest.raises(MemoryError, match=match):
        store.clones.import_barcodes(
            path, cell_selection=every_cell(store), invalidate_cache=True, **options
        )
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == saved
    return read


def test_a_dense_matrix_of_one_large_chunk_is_refused_before_it_is_read(
    store, tmp_path, monkeypatch
) -> None:
    import h5py

    from tests.clone_truth import _obs_frame

    # One gzip chunk of 4,000 cells by 1,000 barcodes, 16 MB once inflated:
    # the tile read and HDF5's buffer of the chunk each take that much.
    path = tmp_path / "dense.h5ad"
    with h5py.File(path, "w") as handle:
        _obs_frame(handle, [f"C{index}" for index in range(4_000)])
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset(
            "_index",
            data=np.asarray([f"B{index}" for index in range(1_000)], dtype=object),
            dtype=h5py.string_dtype(),
        )
        values = np.zeros((4_000, 1_000), dtype=np.float32)
        values[np.arange(4_000), np.arange(4_000) % 1_000] = 2
        handle.create_dataset(
            "X", data=values, chunks=(4_000, 1_000), compression="gzip"
        )
    monkeypatch.setattr(store, "memoryBytes", 24 * 2**20)
    assert _refused_before_reading(store, path, monkeypatch) == []


def test_the_keys_of_a_trex_table_count_against_the_budget(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import lineage_barcodes as operations

    # 2,000 molecules of cells outside the store: no record is held, so only
    # the keys that find repeated molecules fill the budget, here counted at
    # 40,000 bytes each, beyond a budget that holds the read of a stored
    # chunk of the cells' IDs.
    lines = ["#cell_id\tumi\tclone_id\tread_count"] + [
        f"OUT{row % 50:04d}\tU{row:06d}\tTTTT\t3" for row in range(2_000)
    ]
    path = tmp_path / "molecules.txt"
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(operations, "_KEY_BYTES", 40_000)
    monkeypatch.setattr(store, "memoryBytes", 64 * 2**20)
    with collect_warnings():
        read = _refused_before_reading(store, path, monkeypatch)
    assert len(read) == 1


def test_a_molecule_info_file_of_many_barcodes_in_one_chunk_is_refused(
    store, tmp_path, monkeypatch
) -> None:
    import h5py

    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    # 1,500,000 cell barcodes stored as one chunk of 24 MB, read at once.
    with h5py.File(path, "a") as handle:
        listed = handle["barcodes"][:]
        filler = np.char.add(
            b"F", np.char.zfill(np.arange(1_500_000).astype("S15"), 15)
        ).astype(listed.dtype)
        del handle["barcodes"]
        barcodes = np.concatenate([listed, filler])
        handle.create_dataset("barcodes", data=barcodes, chunks=(len(barcodes),))
    monkeypatch.setattr(store, "memoryBytes", 24 * 2**20)
    assert (
        _refused_before_reading(store, path, monkeypatch, barcode_prefix="d4__") == []
    )


def test_a_library_column_of_many_long_labels_is_refused_before_it_is_read(
    store, tmp_path, monkeypatch
) -> None:
    import h5py

    path = write_quicat_h5ad(BARCODES, tmp_path / "barcodes_output.h5ad")
    # A column that names each cell's library by 1,000 characters of its own,
    # 21 MB decoded, more than the budget leaves beside the check of the
    # cells' IDs, which needs less than them.
    with h5py.File(path, "a") as handle:
        obs = handle["obs"]
        n_cells = obs["_index"].shape[0]
        del handle["obs"]
        names = [f"C{index}" for index in range(20_000)]
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray(names, dtype=h5py.string_dtype()))
        labels = obs.create_dataset(
            "lane",
            data=np.asarray(
                [f"{index:01000d}" for index in range(20_000)],
                dtype=h5py.string_dtype(),
            ),
        )
        labels.attrs["encoding-type"] = "string-array"
        labels.attrs["encoding-version"] = "0.2.0"
        del handle["X"]
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        n_barcodes = handle["var/_index"].shape[0]
        group.attrs["shape"] = np.asarray([20_000, n_barcodes])
        group.create_dataset("data", data=np.ones(20_000, dtype=np.float32))
        group.create_dataset("indices", data=np.zeros(20_000, dtype=np.int32))
        group.create_dataset("indptr", data=np.arange(20_001, dtype=np.int64))
    assert n_cells < 20_000
    # The most that the import holds until the join holds its lookup of the
    # cells, to the byte, which the column's labels pass before they are read.
    budget = _most_through(
        store,
        lambda: store.clones.import_barcodes(
            write_quicat_h5ad(BARCODES, tmp_path / "plain.h5ad"),
            cell_selection=every_cell(store),
            invalidate_cache=True,
        ),
        "the join's lookup of the selected cells' IDs",
    )
    monkeypatch.setattr(store, "memoryBytes", budget)
    read = _refused_before_reading(
        store,
        path,
        monkeypatch,
        match="the barcode file's names and a chunk",
        format="h5ad",
        columns={"library": "lane"},
    )
    assert read == []


def test_a_refusal_that_reading_causes_does_not_ask_for_fewer_cells(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.readers import lineage_barcodes as reader

    path = write_quicat_h5ad(BARCODES, tmp_path / "barcodes_output.h5ad")
    monkeypatch.setattr(
        reader.BarcodeLayout, "held_bytes", lambda self: store.memoryBytes
    )
    with pytest.raises(MemoryError) as refused:
        store.clones.import_barcodes(path, cell_selection=every_cell(store))
    # A smaller selection holds the file's names all the same.
    assert "Raise mem_budget" in str(refused.value)
    assert "cell_selection" not in str(refused.value)


def test_the_join_s_unmatched_barcodes_count_beside_the_records(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module
    from cytearc.datastore._operations import cell_barcodes

    counted = write_bartab_counts(BARCODES, tmp_path / "counts.tsv.gz", sample="d4")
    rows = gzip.decompress(counted.read_bytes()).decode()
    # 300 droplets outside the store, counted here at 134,000 bytes each, 40
    # MB beside a budget that holds the read of a stored chunk of the cells'
    # IDs.
    droplets = "".join(f"LB{row}\tOUT{row:05d}\t2\n" for row in range(300))
    path = tmp_path / "counts.tsv"
    path.write_text(rows + droplets)
    budget = 64 * 2**20
    monkeypatch.setattr(store, "memoryBytes", budget)
    monkeypatch.setattr(cell_barcodes, "_UNMATCHED_BYTES_PER_BARCODE", 134_000)
    # The records take half the budget, the droplets' hashes more than the
    # rest: each fits alone, but not both.
    monkeypatch.setattr(
        barcodes_module.BarcodeCounter,
        "peak_bytes",
        property(lambda self: budget // 2 if self.n_rows else 0),
    )
    with pytest.raises(
        MemoryError, match=r"Importing barcodes for \d+ selected cells holds about"
    ):
        store.clones.import_barcodes(
            path,
            cell_selection=every_cell(store),
            barcode_prefix="d4__",
            barcode_suffix="-1",
        )
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == []


def test_the_lookup_of_the_selected_cells_counts_as_the_records_grow(
    tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module

    # 2,000 selected cells, whose lookup outweighs a chunk of a small file.
    cell_ids = np.asarray([f"C{index:05d}-1" for index in range(2_000)])
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((2_000, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cell_ids,
    )
    store = open_store(tmp_path / "store.zarr")
    path = tmp_path / "counts.tsv"
    path.write_text(
        "gene\tcell\tcount\n"
        + "".join(f"LB{row % 3}\tC{row:05d}-1\t2\n" for row in range(20))
    )
    # Beyond the read of a stored chunk of the cells' IDs.
    budget = 64 * 2**20
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    selection = store.snapshot_cell_selection("every_cell")
    # What the cells hold once the join holds its lookup of their IDs, in a
    # run that saves barcodes, which the refused run leaves as they are.
    lookup = _held_at(
        store,
        lambda: store.clones.import_barcodes(
            path, cell_selection=selection, invalidate_cache=True
        ),
        "the join's lookup of the selected cells' IDs",
    )
    monkeypatch.setattr(store, "memoryBytes", budget)
    # Once rows arrive, the records fill what the lookup leaves of the budget.
    monkeypatch.setattr(
        barcodes_module.BarcodeCounter,
        "peak_bytes",
        property(lambda self: budget - lookup + 1 if self.n_rows else 0),
    )
    with pytest.raises(
        MemoryError, match="molecules and pairs at their merge"
    ) as refused:
        store.clones.import_barcodes(
            path, cell_selection=selection, invalidate_cache=True
        )
    assert refused.value.needed_bytes > budget
    assert len(store.artifacts.list(kind="lineage_barcodes", scope="datastore")) == 1


def test_the_selected_cells_and_the_read_of_their_ids_are_checked_first(
    store, tmp_path, monkeypatch
) -> None:
    import cytearc.metadata.rows as rows

    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    options = {"barcode_prefix": "d4__", "barcode_suffix": "-1"}

    def run() -> object:
        return store.clones.import_barcodes(
            path, cell_selection=every_cell(store), invalidate_cache=True, **options
        )

    # The 64 bytes of arrays of each of the 48 cells, and the read of their
    # IDs as text, which decodes a stored chunk of them.
    with collect_warnings():
        reading = _held_at(store, run, "the selected cells' IDs")
    read: list[object] = []
    original = rows.read_array_rows_chunkwise

    def counted(*args):
        read.append(args)
        return original(*args)

    saved = store.artifacts.list(kind="lineage_barcodes", scope="datastore")
    monkeypatch.setattr(rows, "read_array_rows_chunkwise", counted)
    monkeypatch.setattr(store, "memoryBytes", reading - 1)
    with pytest.raises(
        MemoryError,
        match=f"Importing barcodes for {N_CELLS} selected cells holds about "
        f"{reading} bytes for the selected cells' IDs",
    ):
        run()
    # Refused before a cell ID is read, and nothing is written. At the bound
    # the IDs, in the one chunk of their 48 rows, are read once. The import's
    # chunks then take what the ledger's room leaves, so this small import
    # runs there unless a later step, such as the save beside Zarr's file
    # buffers, needs more than the read of the IDs; that step then refuses,
    # writing nothing. At the budget that the import's refusals name, the IDs
    # are read again.
    assert read == []
    assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == saved
    assert store.cells._get_array("ids").chunks == (N_CELLS,)
    monkeypatch.setattr(store, "memoryBytes", reading)
    try:
        with collect_warnings():
            bound = run()
    except MemoryBudgetError as later:
        assert later.item != "the selected cells' IDs"
        assert later.needed_bytes > reading
        assert store.artifacts.list(kind="lineage_barcodes", scope="datastore") == saved
    else:
        saved = store.artifacts.list(kind="lineage_barcodes", scope="datastore")
        assert bound in saved
    assert len(read) == 1
    monkeypatch.setattr(store, "memoryBytes", 64 * 2**20)
    with collect_warnings():
        ref = run()
    assert len(read) == 2
    assert ref in store.artifacts.list(kind="lineage_barcodes", scope="datastore")


def test_the_selected_cells_count_their_ids_as_the_join_holds_them(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.clones import barcodes as barcodes_module
    from cytearc.readers import lineage_barcodes as reader

    ids = [str(cell_id) for cell_id in store.cells.fetch_all("ids")]
    # Three of the store's cells, so that no barcode leaves a hash beside them.
    path = tmp_path / "counts.tsv"
    path.write_text(
        "gene\tcell\tcount\n" + "".join(f"LB{k}\t{ids[k]}\t2\n" for k in range(3))
    )
    # The 64 bytes of arrays of each of the 48 cells, their IDs as the
    # Python strings that the lookup holds, and the join's lookup.
    cells = _held_at(
        store,
        lambda: store.clones.import_barcodes(
            path, cell_selection=every_cell(store), invalidate_cache=True
        ),
        "the join's lookup of the selected cells' IDs",
    )
    # Beyond the read of a stored chunk of IDs, which the check before the
    # IDs are read counts.
    budget = 64 * 2**20
    monkeypatch.setattr(store, "memoryBytes", budget)
    # Reading holds nothing else, so the records at their merge may take what
    # the cells leave; the finished table of the 3 records is held beside the
    # merge before the merge goes, and the save fits in what the merge leaves.
    _result, taken = charges(
        lambda: store.clones.import_barcodes(
            path, cell_selection=every_cell(store), invalidate_cache=True
        )
    )
    table = next(nbytes for item, nbytes, _planned in taken if item == "the 3 records")
    monkeypatch.setattr(reader.BarcodeLayout, "held_bytes", lambda self: 0)
    monkeypatch.setattr(reader.BarcodeLayout, "chunk_bytes", lambda self, rows: 0)
    refused = {
        budget - cells + 1: "the 3 molecules and pairs at their merge",
        budget - cells: "the 3 records",
    }
    for records in (budget - cells + 1, budget - cells, budget - cells - table):
        monkeypatch.setattr(
            barcodes_module.BarcodeCounter,
            "peak_bytes",
            property(lambda self, records=records: records if self.n_rows else 0),
        )
        if records in refused:
            needed = cells + records + (table if records == budget - cells else 0)
            with pytest.raises(
                MemoryError,
                match=f"holds about {needed} bytes for {refused[records]},",
            ):
                store.clones.import_barcodes(
                    path, cell_selection=every_cell(store), invalidate_cache=True
                )
        else:
            ref = store.clones.import_barcodes(
                path, cell_selection=every_cell(store), invalidate_cache=True
            )
    assert store.clones.load_barcodes(ref).records["barcode"].tolist() == [
        "LB0",
        "LB1",
        "LB2",
    ]


def test_load_names_missing_and_incomplete_barcodes(store, tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    with collect_warnings():
        ref = store.clones.import_barcodes(
            path,
            cell_selection=every_cell(store),
            barcode_prefix="d4__",
            barcode_suffix="-1",
        )
    missing = ArtifactRef(
        scope="datastore", kind="lineage_barcodes", artifact_id="f" * 64, assay=None
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.clones.load_barcodes(missing)
    # An import that stopped before it finished leaves an incomplete artifact.
    artifact_group(store.zw, ref).attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store.clones.load_barcodes(ref)


def test_the_hashes_a_file_leaves_count_before_the_next_file_is_read(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import cell_barcodes
    from cytearc.readers import lineage_barcodes as reader

    # Each file holds 300 droplets outside the store, whose hashes are
    # counted here at 100,000 bytes each: 30 MB once the first file, in the
    # files' canonical order, is read. The second file's reading, set at
    # 40 MiB, fits the budget beside them only if they are left out.
    paths = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.tsv"
        path.write_text(
            "gene\tcell\tcount\n"
            + "".join(f"LB{row}\t{name}{row:05d}\t2\n" for row in range(300))
        )
        paths.append(path)
    asked: list[str] = []

    def held_bytes(layout) -> int:
        if str(layout.path) not in asked:
            asked.append(str(layout.path))
        return 0 if str(layout.path) == asked[0] else 40 * 2**20

    monkeypatch.setattr(cell_barcodes, "_UNMATCHED_BYTES_PER_BARCODE", 100_000)
    monkeypatch.setattr(reader.BarcodeLayout, "held_bytes", held_bytes)
    monkeypatch.setattr(store, "memoryBytes", 64 * 2**20)
    with collect_warnings():
        read = _refused_before_reading(
            store,
            paths,
            monkeypatch,
            match="the barcode file's names and a chunk",
        )
    # The first file was read; the second was refused before its first row.
    assert len(read) == 1
    assert len(asked) == 2


def test_the_load_counts_its_records_texts_and_cells_to_the_byte(
    store, tmp_path, monkeypatch
) -> None:
    path = write_trex_molecules(BARCODES, tmp_path / "molecules.txt", sample="d2")
    with collect_warnings():
        ref = store.clones.import_barcodes(
            path,
            cell_selection=every_cell(store),
            barcode_prefix="d2__",
            barcode_suffix="-1",
        )
    records = store.clones.load_barcodes(ref).records
    n_barcodes = records["barcode"].dropna().nunique()
    n_umis = records["umi"].dropna().nunique()
    # 120 bytes per record and 64 per cell, and 160 for each cell and each
    # distinct barcode and UMI, each stored text with one bound more than its
    # values.
    needed = (
        120 * len(records)
        + 160 * ((n_barcodes + 1) + (n_umis + 1) + N_CELLS)
        + 64 * N_CELLS
    )
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"hold about {needed} bytes"):
        store.clones.load_barcodes(ref)
    # At the bound the tables fit, and the read of the cells' barcode counts,
    # which counts Zarr's objects beside each array it reads, names the budget
    # it needs.
    monkeypatch.setattr(store, "memoryBytes", needed)
    with pytest.raises(MemoryBudgetError) as counted:
        store.clones.load_barcodes(ref)
    assert counted.value.item == "'n_barcodes' of the lineage_barcodes artifact"
    reading = counted.value.needed_bytes
    assert reading > needed
    # There the load runs, and reading the cells' IDs decodes the one chunk of
    # their 48 rows.
    monkeypatch.setattr(store, "memoryBytes", reading)
    pd.testing.assert_frame_equal(store.clones.load_barcodes(ref).records, records)
    # The IDs in a chunk of 100,000 rows, as a store written before cell
    # columns were chunked by their rows keeps them, decode that whole chunk,
    # which the read's own check refuses at that budget.
    rechunk_column(store.cells._group, "ids")
    reopened = open_store(store.zarr_loc, mem_budget=reading)
    with pytest.raises(MemoryBudgetError) as refused:
        reopened.clones.load_barcodes(ref)
    assert "The tables of" not in str(refused.value)
    assert refused.value.item == counted.value.item
    assert refused.value.needed_bytes > reading


def test_selecting_cells_by_their_barcodes_follows_the_documented_bounds(
    shared, tmp_path
) -> None:
    # The API page's example: select_cells keeps the values above low, so
    # low=0 keeps the cells with a barcode and low=1 those with two or more.
    path = write_bartab_counts(BARCODES, tmp_path / "counts.tsv", sample="d4")
    with collect_warnings():
        barcodes = shared.store.clones.import_barcodes(
            path,
            cell_selection=shared.cells,
            barcode_prefix="d4__",
            barcode_suffix="-1",
        )
    records = shared.store.clones.load_barcodes(barcodes).records
    # Each cell's distinct barcodes, counted from its records.
    distinct = records.groupby("ids", observed=True)["barcode"].nunique()
    assert (distinct == 1).any() and (distinct >= 2).any()
    for low, fewest in ((0, 1), (1, 2)):
        selected = shared.store.select_cells(barcodes, low=low)
        kept = shared.store.artifacts.load_values(barcodes, cell_selection=selected)
        expected = {str(cell) for cell in distinct.index[distinct >= fewest]}
        assert set(kept.cell_ids.tolist()) == expected, low
