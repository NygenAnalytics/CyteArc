"""Reading barcode files: formats recognised from their contents, and their rows.

Each tool's header is recognised, as are pycashier's and FateMap's tables
without one; tables of other tools are read by the columns that ``columns``
names. Cell Ranger's ``molecule_info.h5`` is read in its documented layout,
with UMIs decoded as Cell Ranger packs them.
"""

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cytearc.readers.cellranger import (
    decode_umis,
    is_molecule_info,
    iter_molecule_info,
    molecule_info_library,
)
from cytearc.readers.lineage_barcodes import barcode_layout, iter_barcode_rows
from tests.clone_truth import (
    barcode_truth,
    clone_truth,
    pack_umis,
    write_molecule_info,
    write_quicat_h5ad,
)

TRUTH = clone_truth()
BARCODES = barcode_truth(TRUTH)


def _file(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text)
    return path


def _rows(layout, chunk_rows: int = 2):
    chunks = list(iter_barcode_rows(layout, chunk_rows))
    return chunks


# Each case: a file's text, and the format, level and columns it is read by.
_HEADERS = {
    "trex_current": (
        "#cell_id\tumi\tclone_id\tread_count\nAAAC\tGG\tTTT-0\t3\n",
        "trex",
        "molecules",
        {
            "cell": ["cell_id"],
            "barcode": ["clone_id"],
            "umi": "umi",
            "reads": "read_count",
        },
    ),
    "trex_corrected": (
        "#cell_id\tumi\toriginal_clone_id\tclone_id\tread_count\nAAAC\tGG\tTTA\tTTT\t3\n",
        "trex",
        "molecules",
        {
            "cell": ["cell_id"],
            "barcode": ["clone_id"],
            "umi": "umi",
            "reads": "read_count",
        },
    ),
    "trex_2022": (
        "#cell_id\tumi\tclone_id\nAAAC\tGG\tTTT\n",
        "trex",
        "molecules",
        {"cell": ["cell_id"], "barcode": ["clone_id"], "umi": "umi", "reads": None},
    ),
    "trex_smartseq2": (
        "#cell_id\tclone_id\nAAAC\tTTT\n",
        "trex",
        "reads",
        {"cell": ["cell_id"], "barcode": ["clone_id"], "umi": None, "reads": None},
    ),
    "pycashier": (
        "info\tumi\tcell\tbarcode\n@r1\tGG\tAAAC\tTTT\n",
        "pycashier",
        "reads",
        {"cell": ["cell"], "barcode": ["barcode"], "umi": "umi", "reads": None},
    ),
    "celltag_atac": (
        "Cell.BC\tCell.Tag\nAAAC-1\tTTT\n",
        "celltag",
        "reads",
        {"cell": ["Cell.BC"], "barcode": ["Cell.Tag"], "umi": None, "reads": None},
    ),
    "fatemap": (
        "cellID\tUMI\tOriginalBarcode\tBC50StarcodeD8\tSampleNum\nAAAC\tGG\tTTA\tTTT\t1\n",
        "fatemap",
        "molecules",
        {
            "cell": ["cellID"],
            "barcode": ["BC50StarcodeD8"],
            "umi": "UMI",
            "reads": None,
        },
    ),
    "quicat": (
        "sample,cell,barcode,counts,replicate\ns1,AAAC-1,TTT,4,r1\n",
        "quicat",
        "pairs",
        {"cell": ["cell"], "barcode": ["barcode"], "umis": "counts"},
    ),
    "quicat_wide": (
        "sample,cell,TTT,GGG\ns1,AAAC-1_s1,0,4\n",
        "quicat_wide",
        "pairs",
        {"cell": ["cell"], "barcode": []},
    ),
    "bartab": (
        "gene\tcell\tcount\nLB1\tAAAC\t4\n",
        "bartab",
        "pairs",
        {"cell": ["cell"], "barcode": ["gene"], "umis": "count"},
    ),
    "bartab_annotation": (
        "barcode\tbc.umi.count\nAAAC\tLB1;LB2\t4;1\n",
        "bartab_annotation",
        "pairs",
        {"cell": [], "barcode": ["barcode"], "umis": "bc.umi.count", "separator": ";,"},
    ),
}


@pytest.mark.parametrize("case", sorted(_HEADERS))
def test_each_tool_is_recognised_by_its_header(tmp_path, case) -> None:
    text, format, level, columns = _HEADERS[case]
    path = _file(tmp_path, "file.txt", text)

    layout = barcode_layout(path)

    assert (layout.format, layout.level) == (format, level)
    recorded = layout.record()
    assert {role: recorded["columns"][role] for role in columns} == columns
    assert recorded["library"] == "file"
    # The format named is read alike.
    assert barcode_layout(path, format=format) == layout


@pytest.mark.parametrize(
    ("text", "format", "names"),
    [
        ("@read1\tGGAT\tAAACCC\tTTTGG\n@read2\tGGAT\tAAACCC\tTTTGG\n", "pycashier", 4),
        (
            "AAAC\tGGAT\tTTA\tTTT\tTT\tT\tT\t1\nAAAC\tGGCC\tTTA\tTTT\tTT\tT\tT\t1\n",
            "fatemap",
            8,
        ),
    ],
)
def test_tables_without_a_header_are_recognised_by_their_rows(
    tmp_path, text, format, names
) -> None:
    path = _file(tmp_path, "rows.txt", text)

    layout = barcode_layout(path)

    assert layout.format == format
    (chunk,) = _rows(layout, 10)
    assert chunk.first_line == 1
    assert len(chunk.cells) == 2
    assert barcode_layout(path, format=format) == layout


def test_a_named_format_without_its_header_needs_its_columns_in_order(tmp_path) -> None:
    path = _file(tmp_path, "rows.txt", "@read1\tGGAT\tAAAC\n")
    with pytest.raises(
        ValueError, match="must hold the 4 columns of format='pycashier'"
    ):
        barcode_layout(path, format="pycashier")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("clone_nr\tcell_id\n1\tAAAC\n", "import it with ds.clones.import_clones"),
        ("a,b\n1,2\n", "format='reads', 'molecules' or 'pairs'"),
        (
            "#cell_id\tumi\tclone_id\tinfo\tcell\tbarcode\nA\tG\tT\t@r\tA\tT\n",
            "the columns of several barcode tables: trex, pycashier",
        ),
        ("\n\n", "is empty; a barcode table needs a header line"),
        ("barcode\tbc.umi.count\n", "R's row-name layout"),
        ("sample,cell,replicate\ns,A,r\n", "has no barcode columns"),
    ],
)
def test_files_that_name_no_known_table_are_refused(tmp_path, text, message) -> None:
    path = _file(tmp_path, "file.txt", text)
    format = None
    if "bc.umi.count" in text:
        format = "bartab_annotation"
    elif "replicate" in text:
        format = "quicat_wide"
    with pytest.raises(ValueError, match=message):
        barcode_layout(path, format=format)


@pytest.mark.parametrize(
    ("format", "columns", "error", "message"),
    [
        ("pairs", {"barcode": "b"}, ValueError, "has no row names, so name its cell"),
        ("pairs", {"cell": "c"}, ValueError, "needs the barcode column"),
        (
            "molecules",
            {"cell": "c", "barcode": "b"},
            ValueError,
            "needs each molecule's UMI",
        ),
        (
            "reads",
            {"cell": "c", "barcode": "b", "reads": "n"},
            ValueError,
            "reads no column for 'reads'",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "umi": "u"},
            ValueError,
            "reads no column for 'umi'",
        ),
        ("pairs", {"cell": "c", "barcode": "x"}, ValueError, "has no column 'x'"),
        (
            "pairs",
            {"cell": "c", "barcode": ["b", "n"], "separator": ";"},
            ValueError,
            "must name one column",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "separator": "/"},
            ValueError,
            "must be one of ';'",
        ),
        (
            "pairs",
            {"cell": 3, "barcode": "b"},
            TypeError,
            "a column name or a sequence",
        ),
        (
            "pairs",
            {"cell": "c", "barcode": "b", "umis": ["n"]},
            TypeError,
            "must be a column name",
        ),
        ("pairs", {"cell": "", "barcode": "b"}, ValueError, "names an empty column"),
        ("pairs", ["cell"], TypeError, "columns must map roles"),
        ("trex", {"cell": "c"}, ValueError, "format='trex' reads fixed columns"),
    ],
)
def test_columns_that_a_level_cannot_read_are_refused(
    tmp_path, format, columns, error, message
) -> None:
    path = _file(tmp_path, "table.tsv", "c\tb\tn\tu\nAAAC\tTT\t3\tGG\n")
    with pytest.raises(error, match=message):
        barcode_layout(path, format=format, columns=columns)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"format": "larry"}, ValueError, "format must be one of"),
        ({"library": 3}, TypeError, "library must be text"),
        ({"library": ""}, ValueError, "must name a library"),
        (
            {
                "format": "pairs",
                "columns": {"cell": "c", "barcode": "b", "library": "u"},
                "library": "x",
            },
            ValueError,
            "library= would name no row",
        ),
        ({"format": "h5ad"}, ValueError, "is a text table, not format='h5ad'"),
    ],
)
def test_options_that_do_not_fit_the_file_are_refused(
    tmp_path, options, error, message
) -> None:
    path = _file(tmp_path, "table.tsv", "c\tb\tn\tu\nAAAC\tTT\t3\tGG\n")
    with pytest.raises(error, match=message):
        barcode_layout(path, **options)
    with pytest.raises(TypeError, match="path must be a path"):
        barcode_layout(3)


def test_a_file_without_a_name_before_its_extensions_needs_a_library(tmp_path) -> None:
    path = _file(tmp_path, ".tsv", "gene\tcell\tcount\nLB\tA\t1\n")
    with pytest.raises(ValueError, match="give it a library"):
        barcode_layout(path)
    assert barcode_layout(path, library="lib").library == "lib"


def test_rows_name_their_lines_when_a_value_is_wrong(tmp_path) -> None:
    missing_cell = _file(tmp_path, "a.tsv", "gene\tcell\tcount\nLB\tA\t1\nLB\t\t2\n")
    with pytest.raises(ValueError, match="without a cell barcode on lines 3"):
        _rows(barcode_layout(missing_cell))
    bad_count = _file(tmp_path, "b.tsv", "gene\tcell\tcount\nLB\tA\t1\nLB\tB\t2.5\n")
    with pytest.raises(ValueError, match="count must be a whole number .* line 3"):
        _rows(barcode_layout(bad_count))
    missing_umi = _file(
        tmp_path, "c.tsv", "#cell_id\tumi\tclone_id\nA\tGG\tT\nB\tNA\tT\n"
    )
    with pytest.raises(ValueError, match="without a UMI \\(umi\\) on line 3"):
        _rows(barcode_layout(missing_umi))
    missing_library = _file(tmp_path, "d.tsv", "c\tb\tl\nA\tT\tx\nB\tT\t\n")
    with pytest.raises(ValueError, match="without a library \\(l\\) on line 3"):
        _rows(
            barcode_layout(
                missing_library,
                format="pairs",
                columns={"cell": "c", "barcode": "b", "library": "l"},
            )
        )
    shifted = _file(
        tmp_path, "e.tsv", "info\tumi\tcell\tbarcode\n@r\tGG\tAA\tT\n@r_2\tx\tAA\tT\n"
    )
    with pytest.raises(ValueError, match="holds 'AA' and 'x' on line 3"):
        _rows(barcode_layout(shifted))


def test_fatemap_tables_of_several_samples_are_refused(tmp_path) -> None:
    path = _file(
        tmp_path,
        "fm.txt",
        "cellID\tUMI\tBC50StarcodeD8\tSampleNum\nAAAC\tGG\tTT\t1\nAAAC\tGC\tTT\t1\n"
        "AAAC\tGA\tTT\t2\n",
    )
    with pytest.raises(ValueError, match="holds the samples 1, 2"):
        _rows(barcode_layout(path))


def test_two_barcode_columns_join_and_a_missing_half_names_no_barcode(tmp_path) -> None:
    path = _file(tmp_path, "xp.csv", "cell,bc14,bc30,n\nA,AC,GT,3\nB,AC,NA,2\n")
    layout = barcode_layout(
        path,
        format="pairs",
        columns={"cell": "cell", "barcode": ["bc14", "bc30"], "umis": "n"},
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.barcodes.tolist() == ["AC__GT", None]
    assert chunk.umi_counts.tolist() == [3, 2]


def test_list_fields_split_with_their_counts_and_keep_cells_without_any(
    tmp_path,
) -> None:
    path = _file(
        tmp_path,
        "lists.tsv",
        "cell\tbarcodes\tumis\treads\nA\tx|y\t3|4\t30|40\nB\tNA\tNA\tNA\nC\tz|\t5|\t50|\n",
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={
            "cell": "cell",
            "barcode": "barcodes",
            "umis": "umis",
            "reads": "reads",
            "separator": "|",
        },
    )
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["A", "A", "B", "C", "C"]
    assert chunk.barcodes.tolist() == ["x", "y", None, "z", None]
    assert chunk.umi_counts.tolist() == [3, 4, -1, 5, -1]
    assert chunk.reads.tolist() == [30, 40, -1, 50, -1]
    bad = _file(tmp_path, "bad.tsv", "cell\tbarcodes\tumis\nA\tx|y\t3|4\nB\tx|y\t3|z\n")
    with pytest.raises(
        ValueError,
        match="umis must be a whole number of at least 0, but holds 'z' on line 3",
    ):
        _rows(
            barcode_layout(
                bad,
                format="pairs",
                columns={
                    "cell": "cell",
                    "barcode": "barcodes",
                    "umis": "umis",
                    "separator": "|",
                },
            ),
            10,
        )


def test_bartab_annotations_split_both_list_separators(tmp_path) -> None:
    current = _file(
        tmp_path,
        "current.tsv",
        "barcode\tbc.umi.count\nAAAC\tLB1;LB2\t15;12\nCCCG\tLB3\t1\n",
    )
    legacy = _file(
        tmp_path,
        "legacy.tsv",
        "barcode bc.umi.count\nAAAC-1 LB1,LB2 15,12\nCCCG-1 LB3 1\n",
    )
    for path, suffix in ((current, ""), (legacy, "-1")):
        (chunk,) = _rows(barcode_layout(path), 10)
        assert chunk.cells.tolist() == [f"AAAC{suffix}"] * 2 + [f"CCCG{suffix}"]
        assert chunk.barcodes.tolist() == ["LB1", "LB2", "LB3"]
        assert chunk.umi_counts.tolist() == [15, 12, 1]


def test_quicats_wide_table_gives_its_nonzero_counts(tmp_path) -> None:
    path = _file(
        tmp_path,
        "barcodes_output.csv",
        "sample,cell,condition,TTT,GGG\ns1,A_s1,c,0,4\ns1,B_s1,c,2,0\ns1,C_s1,c,0,0\n",
    )
    chunks = _rows(barcode_layout(path), 2)
    assert [chunk.first_line for chunk in chunks] == [2, 4]
    cells = np.concatenate([chunk.cells for chunk in chunks]).tolist()
    barcodes = np.concatenate([chunk.barcodes for chunk in chunks]).tolist()
    counts = np.concatenate([chunk.umi_counts for chunk in chunks]).tolist()
    assert list(zip(cells, barcodes, counts, strict=True)) == [
        ("A_s1", "GGG", 4),
        ("B_s1", "TTT", 2),
    ]


def test_gzip_compressed_tables_read_whatever_their_names(tmp_path) -> None:
    path = tmp_path / "counts.tsv"
    path.write_bytes(gzip.compress(b"gene\tcell\tcount\nLB1\tAAAC\t4\n"))
    (chunk,) = _rows(barcode_layout(path), 10)
    assert chunk.cells.tolist() == ["AAAC"]


def _quicat_h5ad(tmp_path: Path, layout: str = "csr") -> Path:
    return write_quicat_h5ad(BARCODES, tmp_path / f"{layout}.h5ad", layout=layout)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_anndata_counts_are_pairs_of_their_cells_and_barcodes(tmp_path, layout) -> None:
    path = _quicat_h5ad(tmp_path, layout)
    with h5py.File(path, "r") as handle:
        cells = [value.decode() for value in handle["obs/_index"][:]]
        names = [value.decode() for value in handle["var/_index"][:]]
    read = barcode_layout(path)
    assert (read.format, read.level, read.umis) == ("h5ad", "pairs", "X")
    chunks = _rows(read, 7)
    pairs = sorted(
        (cell, barcode, int(count))
        for chunk in chunks
        for cell, barcode, count in zip(
            chunk.cells, chunk.barcodes, chunk.umi_counts, strict=True
        )
    )
    expected = []
    for sample in ("d2", "d4"):
        molecules = BARCODES.sample_molecules(sample)
        counted = molecules.groupby(["cell_barcode", "barcode"]).size()
        expected += [
            (f"{cell}-1_{sample}", barcode, int(count))
            for (cell, barcode), count in counted.items()
        ]
    assert pairs == sorted(expected)
    assert set(names) >= {barcode for _cell, barcode, _count in expected}
    assert len(cells) == len({cell for cell, _barcode, _count in expected})


def test_anndata_counts_may_be_reads_and_libraries_an_obs_column(tmp_path) -> None:
    path = _quicat_h5ad(tmp_path)
    layout = barcode_layout(
        path, format="h5ad", columns={"reads": "X", "library": "sample"}
    )
    (chunk,) = _rows(layout, 10_000)
    assert chunk.umi_counts is None
    assert (chunk.reads > 0).all()
    assert sorted(set(chunk.libraries)) == ["d2", "d4"]
    assert all(
        cell.endswith(library)
        for cell, library in zip(chunk.cells, chunk.libraries, strict=True)
    )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"cell": "x"}, "columns takes only 'umis' or 'reads'"),
        ({"umis": "X", "reads": "X"}, "name one of them"),
        ({"umis": "layers/counts"}, "must name 'X'"),
        ({"library": "absent"}, "has no obs column 'absent'"),
    ],
)
def test_anndata_columns_that_it_cannot_read_are_refused(
    tmp_path, columns, message
) -> None:
    path = _quicat_h5ad(tmp_path)
    with pytest.raises(ValueError, match=message):
        barcode_layout(path, format="h5ad", columns=columns)


def test_anndata_files_without_their_tables_or_with_wrong_counts_are_refused(
    tmp_path,
) -> None:
    path = _quicat_h5ad(tmp_path, "dense")
    with h5py.File(path, "a") as handle:
        handle["X"][0, 0] = 1.5
    with pytest.raises(ValueError, match="must be whole numbers"):
        _rows(barcode_layout(path), 1_000)
    with h5py.File(path, "a") as handle:
        del handle["X"]
        handle.create_dataset("X", data=np.ones((2, 2)))
    with pytest.raises(ValueError, match="is 2 by 2, but the file names"):
        _rows(barcode_layout(path), 1_000)
    with h5py.File(path, "a") as handle:
        del handle["X"]
    with pytest.raises(ValueError, match="has no X matrix"):
        barcode_layout(path)
    with h5py.File(path, "a") as handle:
        del handle["var"]
    with pytest.raises(ValueError, match="no AnnData var table"):
        barcode_layout(path)
    with h5py.File(path, "a") as handle:
        del handle["obs"]
    with pytest.raises(ValueError, match="no AnnData obs table"):
        barcode_layout(path)
    with pytest.raises(ValueError, match="is an HDF5 file, not format='trex'"):
        barcode_layout(path, format="trex")


def test_umis_decode_as_cell_ranger_packs_them() -> None:
    umis = [
        "AGTACTCCGAAA",
        "TAGATTCACGCT",
        "ATTTATTAATGT",
        "AAAAAAAAAAAA",
        "TTTTTTTTTTTT",
    ]
    # The values that Cell Ranger's molecule_info.h5 of two 10x Genomics
    # datasets (CC BY 4.0) store for these UMIs.
    assert pack_umis(umis[:3]).tolist() == [2913664, 13169767, 4144187]
    assert decode_umis(pack_umis(umis), 12).tolist() == umis
    assert decode_umis(np.asarray([0, 3], dtype=np.uint32), 2).tolist() == ["AA", "AT"]
    assert decode_umis(np.empty(0, dtype=np.uint32), 10).tolist() == []


def test_molecule_info_gives_the_molecules_of_its_lineage_library(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    assert is_molecule_info(path)
    library = molecule_info_library(path)
    assert (library.library_type, library.library_indices, library.umi_length) == (
        "Custom",
        (1,),
        12,
    )
    rows = list(iter_molecule_info(library, chunk_rows=1))
    molecules = BARCODES.sample_molecules("d4")
    barcodes = np.concatenate([chunk.barcodes for chunk in rows])
    umis = np.concatenate([chunk.umis for chunk in rows])
    reads = np.concatenate([chunk.reads for chunk in rows])
    features = np.concatenate([chunk.features for chunk in rows])
    names = {
        barcode: f"LB{index}"
        for index, barcode in enumerate(sorted(molecules["true_barcode"].unique()))
    }
    assert sorted(zip(barcodes, umis, features, reads, strict=True)) == sorted(
        (f"{row.cell_barcode}-1", row.umi, names[row.true_barcode], row.reads)
        for row in molecules.itertuples(index=False)
    )
    layout = barcode_layout(path)
    assert (layout.format, layout.library, layout.level) == (
        "molecule_info",
        "Custom",
        "molecules",
    )


def test_molecule_info_names_the_library_to_read(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    assert molecule_info_library(path, "Custom").library_type == "Custom"
    with pytest.raises(ValueError, match="holds no library of type 'Antibody Capture'"):
        molecule_info_library(path, "Antibody Capture")
    libraries = [
        {"library_id": 0, "library_type": "Gene Expression", "gem_group": 1},
        {"library_id": 1, "library_type": "Custom", "gem_group": 1},
        {"library_id": 2, "library_type": "Antibody Capture", "gem_group": 1},
    ]
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries)], dtype="S4096")
        )
    with pytest.raises(ValueError, match="'Antibody Capture', 'Custom'; name the"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries[:1])], dtype="S4096")
        )
    with pytest.raises(ValueError, match="holds no feature barcode library"):
        molecule_info_library(path)


def test_molecule_info_without_a_recorded_umi_length_is_refused(tmp_path) -> None:
    path = write_molecule_info(
        BARCODES, tmp_path / "molecule_info.h5", sample="d2", umi_length=None
    )
    with pytest.raises(ValueError, match="records no UMI length"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["metrics_json"]
        handle.create_dataset(
            "metrics_json",
            data=json.dumps({"chemistry_defs": {"Custom": {"umi": [{"length": 18}]}}}),
            dtype=h5py.string_dtype(),
        )
    with pytest.raises(ValueError, match="a 32-bit packed UMI holds 1 to 16 bases"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["metrics_json"]
        handle.create_dataset(
            "metrics_json",
            data=json.dumps({"chemistry_defs": {"Custom": {"umi": {"length": 10}}}}),
            dtype=h5py.string_dtype(),
        )
    assert molecule_info_library(path).umi_length == 10


def test_molecule_info_files_that_break_the_layout_are_refused(tmp_path) -> None:
    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d2")
    with h5py.File(path, "a") as handle:
        # A molecule of the lineage library names a barcode the file lacks.
        lineage = int(np.flatnonzero(handle["library_idx"][:] == 1)[0])
        handle["barcode_idx"][lineage] = 10**6
    with pytest.raises(ValueError, match="barcodes or features that it does not hold"):
        list(iter_molecule_info(molecule_info_library(path)))
    with h5py.File(path, "a") as handle:
        del handle["count"]
        handle.create_dataset("count", data=np.ones(3, dtype=np.uint32))
    with pytest.raises(ValueError, match="differ in length"):
        molecule_info_library(path)
    with h5py.File(path, "a") as handle:
        del handle["umi"]
    with pytest.raises(ValueError, match="lacks umi"):
        molecule_info_library(path)
    assert not is_molecule_info(path)
    with h5py.File(path, "a") as handle:
        del handle["library_info"]
        handle.create_dataset("library_info", data=np.asarray([b"[1, 2]"]))
        handle.create_dataset("umi", data=np.ones(3, dtype=np.uint32))
    with pytest.raises(ValueError, match="lists no library types"):
        molecule_info_library(path)


def test_a_table_in_rs_row_name_layout_takes_its_cells_from_the_row_names(
    tmp_path,
) -> None:
    # Watermelon's GEO metadata: R row names, and comma lists of barcodes.
    path = _file(
        tmp_path,
        "metaData_with_lineage.txt",
        '"sample_id" "lineage_barcode"\n"AAAC-1" "1" "TTT,GGG"\n"CCCG-1" "1" NA\n',
    )
    layout = barcode_layout(
        path,
        format="pairs",
        columns={"barcode": "lineage_barcode", "separator": ","},
    )
    assert layout.cell == ()
    (chunk,) = _rows(layout, 10)
    assert chunk.cells.tolist() == ["AAAC-1", "AAAC-1", "CCCG-1"]
    assert chunk.barcodes.tolist() == ["TTT", "GGG", None]
    assert chunk.umi_counts is None


def test_anndata_counts_skip_blocks_without_counts_and_refuse_odd_matrices(
    tmp_path,
) -> None:
    path = _quicat_h5ad(tmp_path, "dense")
    with h5py.File(path, "a") as handle:
        values = handle["X"][:]
        values[0] = 0
        del handle["X"]
        handle.create_dataset("X", data=values)
    chunks = _rows(barcode_layout(path), 1)
    assert chunks[0].first_line == 2
    assert len(chunks) == len(values) - 1
    with h5py.File(path, "a") as handle:
        del handle["X"]
        group = handle.create_group("X")
        group.attrs["shape"] = np.asarray(values.shape)
    with pytest.raises(ValueError, match="neither a dense nor a sparse matrix"):
        _rows(barcode_layout(path), 10)


def test_an_obs_column_that_lacks_a_cells_library_is_refused(tmp_path) -> None:
    path = _quicat_h5ad(tmp_path)
    with h5py.File(path, "a") as handle:
        codes = handle["obs/sample/codes"][:]
        codes[0] = -1
        handle["obs/sample/codes"][:] = codes
    layout = barcode_layout(path, format="h5ad", columns={"library": "sample"})
    with pytest.raises(ValueError, match="lacks the library of some cells"):
        _rows(layout, 10_000)


def test_molecule_info_skips_rows_of_other_libraries(tmp_path, monkeypatch) -> None:
    from cytearc.readers import cellranger

    path = write_molecule_info(BARCODES, tmp_path / "molecule_info.h5", sample="d4")
    library = molecule_info_library(path)
    monkeypatch.setattr(cellranger, "_MIN_MOLECULE_ROWS", 1)
    chunks = list(iter_molecule_info(library, chunk_rows=1))
    with h5py.File(path, "r") as handle:
        lineage = np.flatnonzero(handle["library_idx"][:] == 1)
    assert [chunk.first_row for chunk in chunks] == lineage.tolist()
