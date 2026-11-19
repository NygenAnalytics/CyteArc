"""IDs and names that the 10x HDF5, Matrix Market and Tapestri readers refuse.

A cell ID, feature ID or feature name that is not UTF-8 text, and a missing
or blank ID, are refused with a ValueError that names the file, the dataset
or the column, and the rows, when the reader is created, so an import writes
nothing. IDs that only look odd, such as the text NA, import as the file
holds them.
"""

import gzip
import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix

import cytearc.readers._text as text_module
import cytearc.readers.mtx as mtx_module
from cytearc.readers import (
    CrDirReader,
    CrH5Reader,
    MtxReader,
    TapestriReader,
    inspect_mtx,
)
from cytearc.readers._text import (
    _identifier_texts,
    require_unique_identifiers,
    unique_cell_identifiers,
    unique_feature_identifiers,
    utf8_texts,
)
from cytearc.readers.tapestri import tapestri_source
from cytearc.utils.arguments import quoted_value
from cytearc.writers import CrToZarr, MtxToZarr, TapestriToZarr
from tests.variant_truth import tapestri_truth, write_tapestri

# Two cells by three features.
_COUNTS = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
_FEATURES = b"G1\tA\tGene Expression\nG2\tB\tGene Expression\nG3\tC\tGene Expression\n"
_BARCODES = b"c0\nc1\n"
# "Ünïcödé" and "細胞" (cells), whose characters take several bytes each.
_ACCENTED = "Ünïcödé"
_CELLS_WORD = "細胞"


def _refused_import(
    make_reader: Callable[[], Any], writer: Callable[..., Any], destination: str
) -> ValueError:
    """Return the refusal of an import of the reader that ``make_reader`` creates."""
    with pytest.raises(ValueError) as refused:
        reader = make_reader()
        try:
            writer(reader, destination, mem_budget="64M", nthreads=1).dump()
        finally:
            reader.close()
    return refused.value


def _imported(make_reader: Callable[[], Any], writer: Callable[..., Any], store: Path):
    reader = make_reader()
    try:
        writer(
            reader, str(store), mem_budget="64M", nthreads=1, shuffle_cells=False
        ).dump()
    finally:
        reader.close()
    return zarr.open_group(str(store), mode="r")


def _texts(values: Any) -> list[str]:
    return [str(value) for value in values]


# The shared check of IDs, which the 10x HDF5, Matrix Market, Tapestri and CSV
# readers call.

_PLACE = "column 1 of 'barcodes.tsv'"
_CHECKS: dict[str, Callable[..., object]] = {
    "require_unique": lambda values, **place: require_unique_identifiers(
        values, "Cell IDs", **place
    ),
    "unique_cells": lambda values, **place: unique_cell_identifiers(
        np.asarray(values, dtype=object), "Cell IDs", "make_unique", **place
    ),
    "unique_features": lambda values, **place: unique_feature_identifiers(
        np.asarray(values, dtype=object), "Cell IDs", "error", **place
    ),
}


@pytest.mark.parametrize("check", list(_CHECKS.values()), ids=list(_CHECKS))
@pytest.mark.parametrize(
    ("values", "refusal"),
    [
        pytest.param(
            ["c0", float("nan"), "c2"],
            "must contain non-empty values; row 2 of column 1 of 'barcodes.tsv' is "
            "missing or blank",
            id="nan",
        ),
        pytest.param(
            np.array([0.5, np.nan, 2.5, np.nan], dtype=np.float32),
            "must contain non-empty values; rows 2 and 4 of column 1 of "
            "'barcodes.tsv' are missing or blank",
            id="float32-nan",
        ),
        pytest.param(
            # A space, and an ideographic space, which is blank too.
            ["c0", None, " ", "　"],
            "must contain non-empty values; rows 2, 3 and 4 of column 1 of "
            "'barcodes.tsv' are missing or blank",
            id="none-and-blanks",
        ),
        pytest.param(
            ["", "", "", "c3", " ", " ", " ", " "],
            "must contain non-empty values; rows 1, 2, 3, 5, 6 and 2 more of "
            "column 1 of 'barcodes.tsv' are missing or blank",
            id="the-first-rows-and-how-many-more",
        ),
        pytest.param(
            [b"c0", b"caf\xe9"],
            "must be UTF-8 text; row 2 of column 1 of 'barcodes.tsv' holds b'caf\\xe9'",
            id="latin-1",
        ),
        pytest.param(
            # A UTF-16 surrogate, which UTF-8 never encodes, after a blank ID.
            [b"c0", " ", b"\xed\xa0\x80"],
            "must be UTF-8 text; row 3 of column 1 of 'barcodes.tsv' holds "
            "b'\\xed\\xa0\\x80'",
            id="surrogate",
        ),
    ],
)
def test_the_shared_check_names_the_rows_of_the_ids_that_it_refuses(
    check, values, refusal
) -> None:
    with pytest.raises(ValueError) as refused:
        check(values, place=_PLACE)

    # A ValueError of its own, not the UnicodeDecodeError of a decode.
    assert type(refused.value) is ValueError
    assert str(refused.value) == f"Cell IDs {refusal}"


def test_the_shared_check_names_the_rows_alone_without_a_place() -> None:
    # CSVReader names no place.
    with pytest.raises(ValueError) as blank:
        require_unique_identifiers(np.array(["c0", ""]), "CSV cell IDs")
    with pytest.raises(ValueError) as not_text:
        require_unique_identifiers([b"\xff"], "CSV cell IDs")

    assert str(blank.value) == (
        "CSV cell IDs must contain non-empty values; row 2 is missing or blank"
    )
    assert (
        str(not_text.value) == "CSV cell IDs must be UTF-8 text; row 1 holds b'\\xff'"
    )


def test_the_shared_check_keeps_ids_that_merely_look_odd() -> None:
    values = ["NA", "nan", "None", "N/A", _ACCENTED, _CELLS_WORD]
    mixed = [*values, "café".encode(), 7, np.str_("seven")]

    texts = _identifier_texts(mixed, "Cell IDs", _PLACE)

    assert texts.tolist() == [*values, "café", "7", "seven"]
    # Text alone takes the check that skips the conversion, and gives the same.
    assert _identifier_texts(values, "Cell IDs", _PLACE).tolist() == values
    require_unique_identifiers(mixed, "Cell IDs", place=_PLACE)
    # Bytes and the str of one ID repeat it, as an import would store both.
    with pytest.raises(ValueError, match=r"^Cell IDs must contain unique values$"):
        require_unique_identifiers(["café".encode(), "café"], "Cell IDs")
    renamed = unique_cell_identifiers(
        np.array(["NA", "NA", "nan"], dtype=object), "Cell IDs", "make_unique"
    )
    assert renamed.tolist() == ["NA", "NA-1", "nan"]


def test_utf8_texts_decodes_bytes_and_names_the_first_that_are_not_text() -> None:
    assert utf8_texts(
        np.array([b"a", "b", 3, "é".encode()], dtype=object),
        "IDs must be UTF-8 text",
        "x",
    ) == ["a", "b", "3", "é"]
    with pytest.raises(ValueError) as refused:
        utf8_texts(np.array([b"ok", b"\xff", b"\xfe"]), "IDs must be UTF-8 text", "x")
    assert str(refused.value) == "IDs must be UTF-8 text; row 2 of x holds b'\\xff'"


def test_utf8_texts_decodes_in_blocks_and_counts_rows_across_them(
    monkeypatch,
) -> None:
    # Blocks of two values, so the third block holds row 5.
    monkeypatch.setattr(text_module, "_DECODE_ROWS", 2)
    values = np.array([b"a", b"b", "é".encode(), b"d", b"e\xff", b"f"])

    with pytest.raises(ValueError) as refused:
        utf8_texts(values, "IDs must be UTF-8 text", "x")

    assert str(refused.value) == "IDs must be UTF-8 text; row 5 of x holds b'e\\xff'"
    assert utf8_texts(values[:4], "IDs must be UTF-8 text", "x") == ["a", "b", "é", "d"]


def test_utf8_texts_never_passes_a_decode_that_failed(monkeypatch) -> None:
    # Were the search for the bytes to find none, the decode's own error
    # would still stop the read.
    monkeypatch.setattr(text_module, "check_utf8", lambda *_args: None)
    with pytest.raises(UnicodeDecodeError):
        utf8_texts(np.array([b"\xff"]), "IDs must be UTF-8 text", "x")


# 10x HDF5 files.


def _write_h5(
    path: Path,
    *,
    barcodes: tuple[bytes, ...] = (b"AAAC-1", b"AAGC-1"),
    ids: tuple[bytes, ...] = (b"G1", b"G2", b"G3"),
    names: tuple[bytes, ...] = (b"A", b"B", b"C"),
    types: tuple[bytes, ...] = (b"Gene Expression",) * 3,
    legacy: bool = False,
) -> Path:
    matrix = csr_matrix(_COUNTS)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("GRCh38" if legacy else "matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        group.create_dataset("barcodes", data=np.array(barcodes))
        if legacy:
            group.create_dataset("genes", data=np.array(ids))
            group.create_dataset("gene_names", data=np.array(names))
        else:
            features = group.create_group("features")
            features.create_dataset("id", data=np.array(ids))
            features.create_dataset("name", data=np.array(names))
            features.create_dataset("feature_type", data=np.array(types))
    return path


@pytest.mark.parametrize(
    ("members", "message"),
    [
        pytest.param(
            {"barcodes": (b"AAAC-1", b"AAGC\xff-1")},
            "Cell Ranger barcodes must be UTF-8 text; row 2 of the dataset "
            "'matrix/barcodes' of 'refused.h5' holds b'AAGC\\xff-1'",
            id="barcode",
        ),
        pytest.param(
            {"ids": (b"G1", b"G2", b"G\xe93")},
            "Cell Ranger feature IDs must be UTF-8 text; row 3 of the dataset "
            "'matrix/features/id' of 'refused.h5' holds b'G\\xe93'",
            id="feature-id",
        ),
        pytest.param(
            # The first name ends within a character of two bytes.
            {"names": (b"caf\xc3", b"B", b"C")},
            "Cell Ranger feature names must be UTF-8 text; row 1 of the dataset "
            "'matrix/features/name' of 'refused.h5' holds b'caf\\xc3'",
            id="feature-name",
        ),
        pytest.param(
            {"types": (b"Gene Expression", b"Antibody \xff", b"Gene Expression")},
            "Cell Ranger feature types must be UTF-8 text; row 2 of the dataset "
            "'matrix/features/feature_type' of 'refused.h5' holds b'Antibody \\xff'",
            id="feature-type",
        ),
        pytest.param(
            {"names": (b"A", b"\xed\xa0\x80", b"C"), "legacy": True},
            "Cell Ranger feature names must be UTF-8 text; row 2 of the dataset "
            "'GRCh38/gene_names' of 'refused.h5' holds b'\\xed\\xa0\\x80'",
            id="cell-ranger-2-feature-name",
        ),
        pytest.param(
            {"barcodes": (b"AAAC-1", b" ")},
            "Cell Ranger barcodes must contain non-empty values; row 2 of the dataset "
            "'matrix/barcodes' of 'refused.h5' is missing or blank",
            id="blank-barcode",
        ),
        pytest.param(
            {"ids": (b"", b"G2", "　".encode())},
            "Cell Ranger feature IDs must contain non-empty values; rows 1 and 3 of "
            "the dataset 'matrix/features/id' of 'refused.h5' are missing or blank",
            id="blank-feature-ids",
        ),
        pytest.param(
            {"ids": (b"G1", b"G2", b""), "legacy": True},
            "Cell Ranger feature IDs must contain non-empty values; row 3 of the "
            "dataset 'GRCh38/genes' of 'refused.h5' is missing or blank",
            id="cell-ranger-2-blank-feature-id",
        ),
    ],
)
def test_cellranger_h5_imports_refuse_ids_and_names_before_writing(
    tmp_path, monkeypatch, members, message
) -> None:
    # A relative path, which the messages name as given.
    monkeypatch.chdir(tmp_path)
    _write_h5(Path("refused.h5"), **members)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    refused = _refused_import(
        lambda: CrH5Reader("refused.h5", temp_dir=temporary), CrToZarr, "refused.zarr"
    )

    assert type(refused) is ValueError
    assert str(refused) == message
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(temporary.iterdir())


def test_cellranger_h5_imports_keep_ids_and_names_that_merely_look_odd(
    tmp_path,
) -> None:
    path = _write_h5(
        tmp_path / "odd.h5",
        barcodes=(b"NA", b"nan"),
        ids=(b"NA", b"nan", b"None"),
        names=(_ACCENTED.encode(), _CELLS_WORD.encode(), b"N/A"),
    )

    root = _imported(lambda: CrH5Reader(str(path)), CrToZarr, tmp_path / "odd.zarr")

    assert _texts(root["cellData/ids"][:]) == ["NA", "nan"]
    assert _texts(root["RNA/featureData/ids"][:]) == ["NA", "nan", "None"]
    assert _texts(root["RNA/featureData/names"][:]) == [_ACCENTED, _CELLS_WORD, "N/A"]
    np.testing.assert_array_equal(root["RNA/counts"][:], _COUNTS)


# Matrix Market files: the 10x layout, Parse Biosciences', peaks, ZIP
# archives and feature references.


def _write_mex(
    directory: Path,
    *,
    features: bytes = _FEATURES,
    barcodes: bytes = _BARCODES,
    gz: bool = False,
    features_name: str = "features.tsv",
    reference: bytes | None = None,
) -> Path:
    directory.mkdir()
    n_features = len([line for line in features.splitlines() if line.strip()])
    n_cells = len([line for line in barcodes.splitlines() if line.strip()])
    (directory / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{n_features} {n_cells} 2\n1 1 1\n{n_features} {n_cells} 2\n"
    )
    suffix = ".gz" if gz else ""
    for name, data in ((features_name, features), ("barcodes.tsv", barcodes)):
        target = directory / f"{name}{suffix}"
        target.write_bytes(gzip.compress(data) if gz else data)
    if reference is not None:
        (directory / "feature_reference.csv").write_bytes(reference)
    return directory


def _mex_refusal(directory: Path, temporary: Path, reader_type: str) -> ValueError:
    if reader_type == "CrDirReader":
        return _refused_import(
            lambda: CrDirReader(str(directory), temp_dir=str(temporary)),
            MtxToZarr,
            str(directory.parent / "refused.zarr"),
        )
    candidate = inspect_mtx(directory)[0]
    if reader_type == "explicit":
        return _refused_import(
            lambda: MtxReader(
                candidate.matrixPath,
                candidate.featurePath,
                candidate.cellPath,
                feature_reference_path=candidate.featureReferencePath,
                temp_dir=str(temporary),
            ),
            MtxToZarr,
            str(directory.parent / "refused.zarr"),
        )
    return _refused_import(
        lambda: MtxReader(candidate, temp_dir=str(temporary)),
        MtxToZarr,
        str(directory.parent / "refused.zarr"),
    )


@pytest.mark.parametrize("reader_type", ["MtxReader", "explicit", "CrDirReader"])
@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize(
    ("members", "file", "message"),
    [
        pytest.param(
            {"features": _FEATURES.replace(b"\tB\t", b"\tB\xe9\t")},
            "features.tsv",
            "Feature names must be UTF-8 text; row 2 of column 2 of {file} holds "
            "b'B\\xe9'",
            id="feature-name",
        ),
        pytest.param(
            # UTF-8 text of several bytes a character comes first.
            {
                "features": _FEATURES.replace(b"\tA\t", f"\t{_ACCENTED}\t".encode())
                .replace(b"\tB\t", f"\t{_CELLS_WORD}\t".encode())
                .replace(b"G3", b"G\xff3")
            },
            "features.tsv",
            "Feature IDs must be UTF-8 text; row 3 of column 1 of {file} holds "
            "b'G\\xff3'",
            id="feature-id",
        ),
        pytest.param(
            # The first name ends within a character of three bytes.
            {"features": _FEATURES.replace(b"\tA\t", b"\t\xe7\xb4\t")},
            "features.tsv",
            "Feature names must be UTF-8 text; row 1 of column 2 of {file} holds "
            "b'\\xe7\\xb4'",
            id="cut-feature-name",
        ),
        pytest.param(
            {"features": b"G1\tA\tGene \xff\nG2\tB\tGene Expression\n"},
            "features.tsv",
            "Feature types must be UTF-8 text; row 1 of column 3 of {file} holds "
            "b'Gene \\xff'",
            id="feature-type",
        ),
        pytest.param(
            # A column that the reader does not keep, which pandas decodes too.
            {
                "features": (
                    b"G1\tA\tGene Expression\tchr1\nG2\tB\tGene Expression\tchr\xff\n"
                )
            },
            "features.tsv",
            "Matrix Market sidecars must be UTF-8 text; row 2 of column 4 of {file} "
            "holds b'chr\\xff'",
            id="other-feature-column",
        ),
        pytest.param(
            # Row 2 holds such bytes in its second and third columns, and row
            # 3 in its first: the refusal names the first of each.
            {
                "features": (
                    b"G1\tA\tGene Expression\n"
                    b"G2\tB\xe9\tGene \xff\n"
                    b"G\xfe3\tC\tGene Expression\n"
                )
            },
            "features.tsv",
            "Feature names must be UTF-8 text; row 2 of column 2 of {file} holds "
            "b'B\\xe9'",
            id="first-of-several",
        ),
        pytest.param(
            {"barcodes": b"c0\nc\xe91\n"},
            "barcodes.tsv",
            "Cell IDs must be UTF-8 text; row 2 of column 1 of {file} holds b'c\\xe91'",
            id="cell-id",
        ),
        pytest.param(
            # A blank line holds no cell; a blank field is a missing ID.
            {"barcodes": b"c0\tx\n\n\ty\n"},
            "barcodes.tsv",
            "Cell IDs must contain non-empty values; row 2 of column 1 of {file} is "
            "missing or blank",
            id="blank-cell-id",
        ),
        pytest.param(
            {"features": b"G1\tA\n \tB\nG3\tC\n"},
            "features.tsv",
            "Feature IDs must contain non-empty values; row 2 of column 1 of {file} "
            "is missing or blank",
            id="blank-feature-id",
        ),
    ],
)
def test_matrix_market_imports_refuse_ids_and_names_before_writing(
    tmp_path, members, file, message, gz, reader_type
) -> None:
    directory = _write_mex(tmp_path / "mex", gz=gz, **members)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    path = directory / f"{file}.gz" if gz else directory / file

    refused = _mex_refusal(directory, temporary, reader_type)

    assert type(refused) is ValueError
    assert str(refused) == message.format(file=quoted_value(str(path)))
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(temporary.iterdir())


def _write_parse(directory: Path, *, genes: bytes, cells: bytes) -> Path:
    directory.mkdir()
    n_genes = len(genes.splitlines()) - 1
    n_cells = len(cells.splitlines()) - 1
    # Parse Biosciences writes cells by genes.
    (directory / "DGE.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{n_cells} {n_genes} 2\n1 1 1\n{n_cells} {n_genes} 2\n"
    )
    (directory / "all_genes.csv").write_bytes(genes)
    (directory / "cell_metadata.csv").write_bytes(cells)
    return directory


_GENES = b"gene_id,gene_name,genome\nG1,A,GRCh38\nG2,B,GRCh38\n"
_PARSE_CELLS = b"bc_wells,sample\nc0,s1\nc1,s2\n"


@pytest.mark.parametrize(
    ("genes", "cells", "file", "message"),
    [
        pytest.param(
            _GENES.replace(b",B,", b",B\xe9,"),
            _PARSE_CELLS,
            "all_genes.csv",
            "Feature names must be UTF-8 text; row 2 of the column 'gene_name' of "
            "{file} holds b'B\\xe9'",
            id="feature-name",
        ),
        pytest.param(
            _GENES.replace(b"G1", b"G\xff1"),
            _PARSE_CELLS,
            "all_genes.csv",
            "Feature IDs must be UTF-8 text; row 1 of the column 'gene_id' of {file} "
            "holds b'G\\xff1'",
            id="feature-id",
        ),
        pytest.param(
            b"gene_id,gene_n\xe4me\nG1,A\nG2,B\n",
            _PARSE_CELLS,
            "all_genes.csv",
            "Matrix Market sidecars must be UTF-8 text; the header of {file} holds "
            "b'gene_n\\xe4me'",
            id="header",
        ),
        pytest.param(
            _GENES,
            b"bc_wells,sample\nc0,s1\nc\xff1,s2\n",
            "cell_metadata.csv",
            "Cell IDs must be UTF-8 text; row 2 of the column 'bc_wells' of {file} "
            "holds b'c\\xff1'",
            id="cell-id",
        ),
        pytest.param(
            _GENES,
            b"bc_wells,sample\nc0,s\xe91\nc1,s2\n",
            "cell_metadata.csv",
            "Matrix Market sidecars must be UTF-8 text; row 1 of the column 'sample' "
            "of {file} holds b's\\xe91'",
            id="cell-metadata",
        ),
        pytest.param(
            # The second row lacks the field of its ID: a missing ID, which
            # pandas read as NaN and the reader imported as "nan".
            b"gene_name,gene_id\nA,G1\nB\n",
            _PARSE_CELLS,
            "all_genes.csv",
            "Feature IDs must contain non-empty values; row 2 of the column "
            "'gene_id' of {file} is missing or blank",
            id="nan-feature-id",
        ),
        pytest.param(
            b"gene_id,gene_name\nG1,A\n,B\n",
            _PARSE_CELLS,
            "all_genes.csv",
            "Feature IDs must contain non-empty values; row 2 of the column "
            "'gene_id' of {file} is missing or blank",
            id="empty-feature-id",
        ),
        pytest.param(
            _GENES,
            b"bc_wells,sample\n,s1\nc1,s2\n",
            "cell_metadata.csv",
            "Cell IDs must contain non-empty values; row 1 of the column 'bc_wells' "
            "of {file} is missing or blank",
            id="empty-cell-id",
        ),
    ],
)
def test_parse_imports_refuse_ids_and_names_before_writing(
    tmp_path, genes, cells, file, message
) -> None:
    directory = _write_parse(tmp_path / "parse", genes=genes, cells=cells)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    refused = _mex_refusal(directory, temporary, "MtxReader")

    assert type(refused) is ValueError
    assert str(refused) == message.format(file=quoted_value(str(directory / file)))
    assert not (tmp_path / "refused.zarr").exists()


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        pytest.param(
            b"id,sequence\nG\xe91,ACGT\n",
            "Feature-reference IDs must be UTF-8 text; row 1 of the column 'id' of "
            "{file} holds b'G\\xe91'",
            id="id",
        ),
        pytest.param(
            b"id,sequence\nG1,ACGT\nG2,AC\xffT\n",
            "Feature references must be UTF-8 text; row 2 of the column 'sequence' "
            "of {file} holds b'AC\\xffT'",
            id="other-column",
        ),
        pytest.param(
            b"id,sequence\nG1,ACGT\n,TTTT\n",
            "Feature-reference IDs must contain non-empty values; row 2 of the "
            "column 'id' of {file} is missing or blank",
            id="empty-id",
        ),
    ],
)
def test_feature_references_refuse_ids_and_text_that_is_not_utf8(
    tmp_path, reference, message
) -> None:
    directory = _write_mex(tmp_path / "mex", reference=reference)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    refused = _mex_refusal(directory, temporary, "MtxReader")

    assert type(refused) is ValueError
    path = directory / "feature_reference.csv"
    assert str(refused) == message.format(file=quoted_value(str(path)))


def test_peak_sidecars_refuse_chromosomes_that_are_not_utf8(tmp_path) -> None:
    directory = _write_mex(
        tmp_path / "atac",
        features=b"chr1\t10\t20\nchr\xff\t30\t40\n",
        features_name="peaks.bed",
    )

    refused = _mex_refusal(directory, tmp_path, "MtxReader")

    assert str(refused) == (
        "Feature IDs must be UTF-8 text; row 2 of column 1 of "
        f"{quoted_value(str(directory / 'peaks.bed'))} holds b'chr\\xff'"
    )


def test_archive_imports_name_the_member_and_remove_their_temporary_files(
    tmp_path,
) -> None:
    archive = tmp_path / "counts.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "sample/matrix.mtx.gz",
            gzip.compress(
                b"%%MatrixMarket matrix coordinate integer general\n3 2 1\n1 1 1\n"
            ),
        )
        handle.writestr(
            "sample/features.tsv.gz",
            gzip.compress(_FEATURES.replace(b"\tB\t", b"\tB\xe9\t")),
        )
        handle.writestr("sample/barcodes.tsv.gz", gzip.compress(_BARCODES))
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    candidate = inspect_mtx(archive)[0]
    refused = _refused_import(
        lambda: MtxReader(candidate, temp_dir=str(temporary)),
        MtxToZarr,
        str(tmp_path / "refused.zarr"),
    )

    assert (candidate.nFeatures, candidate.nCells) == (3, 2)
    assert str(refused) == (
        "Feature names must be UTF-8 text; row 2 of column 2 of "
        f"'sample/features.tsv.gz' in {quoted_value(str(archive))} holds b'B\\xe9'"
    )
    # The members extracted for the reader are removed.
    assert not list(temporary.iterdir())


def test_inspection_counts_the_rows_of_sidecars_that_are_not_utf8(tmp_path) -> None:
    mex = _write_mex(
        tmp_path / "mex",
        features=b"G1\tA\nG2\tB\xe9\nG3\tC\n",
        barcodes=b"c\xff0\nc1\n",
        gz=True,
    )
    parse = _write_parse(
        tmp_path / "parse", genes=_GENES, cells=b"bc_wells,s\xe9\nc0,1\nc1,2\n"
    )

    candidate = inspect_mtx(mex)[0]
    parse_candidate = inspect_mtx(parse)[0]

    assert (candidate.nFeatures, candidate.nCells) == (3, 2)
    assert (parse_candidate.nFeatures, parse_candidate.nCells) == (2, 2)
    # The header names the column of the cell IDs, which the text that is not
    # UTF-8 does not hide.
    assert parse_candidate.cellIdKeys == ("bc_wells",)


@pytest.mark.parametrize(
    ("features", "file", "message"),
    [
        pytest.param(
            b"".join(f"G{row}\tN{row}\n".encode() for row in range(1, 6))
            + b"G6\tN\xe96\nG7\tN7\n",
            "features.tsv",
            "Feature names must be UTF-8 text; row 6 of column 2 of {file} holds "
            "b'N\\xe96'",
            id="without-header",
        ),
    ],
)
def test_the_search_for_text_that_is_not_utf8_counts_rows_across_its_chunks(
    tmp_path, monkeypatch, features, file, message
) -> None:
    # Chunks of two rows, so the row of the refusal is in the third.
    monkeypatch.setattr(mtx_module, "_TEXT_SEARCH_ROWS", 2)
    mex = _write_mex(tmp_path / "mex", features=features)
    parse = _write_parse(
        tmp_path / "parse",
        genes=b"gene_id,gene_name\n"
        + b"".join(f"G{row},N{row}\n".encode() for row in range(1, 4))
        + b"G4,N\xe94\nG5,N5\n",
        cells=b"bc_wells\n" + b"".join(f"c{row}\n".encode() for row in range(5)),
    )

    refused = _mex_refusal(mex, tmp_path, "MtxReader")
    parse_refused = _mex_refusal(parse, tmp_path, "MtxReader")

    assert str(refused) == message.format(file=quoted_value(str(mex / file)))
    # A header is the first row of the first chunk; rows count after it.
    assert str(parse_refused) == (
        "Feature names must be UTF-8 text; row 4 of the column 'gene_name' of "
        f"{quoted_value(str(parse / 'all_genes.csv'))} holds b'N\\xe94'"
    )


def test_a_sidecar_whose_search_finds_no_such_text_is_still_refused(
    tmp_path, monkeypatch
) -> None:
    # A read that fails to decode refuses the file, even if the search for
    # the bytes, which reads the file again, finds none.
    directory = _write_mex(tmp_path / "mex")
    read_csv = pd.read_csv
    failed: list[str] = []

    def failing_once(path, *args, **kwargs):
        if not failed and str(path).endswith("features.tsv"):
            failed.append(str(path))
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(mtx_module.pd, "read_csv", failing_once)

    refused = _mex_refusal(directory, tmp_path, "MtxReader")

    assert type(refused) is ValueError
    assert str(refused) == (
        "Matrix Market sidecars must be UTF-8 text, but "
        f"{quoted_value(str(directory / 'features.tsv'))} is not"
    )


def test_matrix_market_imports_keep_ids_and_names_that_merely_look_odd(
    tmp_path,
) -> None:
    features = (
        f"NA\tN/A\tGene Expression\nnan\t{_ACCENTED}\tGene Expression\n"
        f"None\t{_CELLS_WORD}\tGene Expression\n"
    ).encode()
    directory = _write_mex(
        tmp_path / "mex",
        features=features,
        barcodes=b"NA\nnan\n",
        reference=b"id,sequence\nNA,ACGT\n",
    )

    reader = MtxReader(inspect_mtx(directory)[0])
    try:
        columns = dict(reader.get_feature_columns())
    finally:
        reader.close()
    root = _imported(
        lambda: CrDirReader(str(directory)), MtxToZarr, tmp_path / "odd.zarr"
    )

    # The reference's ID NA is the feature NA, not a missing ID.
    assert list(columns["sequence"]) == ["ACGT", None, None]
    assert _texts(root["cellData/ids"][:]) == ["NA", "nan"]
    assert _texts(root["RNA/featureData/ids"][:]) == ["NA", "nan", "None"]
    assert _texts(root["RNA/featureData/names"][:]) == ["N/A", _ACCENTED, _CELLS_WORD]


def test_parse_imports_keep_the_text_of_their_ids_and_names(tmp_path) -> None:
    # pandas read NA as a missing value, which imported as "nan", and 007 as
    # the number 7.
    directory = _write_parse(
        tmp_path / "parse",
        genes=f"gene_id,gene_name\nNA,N/A\n007,{_ACCENTED}\n".encode(),
        cells=b"bc_wells,sample,count\nNA,NA,1\n001,B,\n",
    )

    root = _imported(
        lambda: MtxReader(inspect_mtx(directory)[0]), MtxToZarr, tmp_path / "parse.zarr"
    )

    assert _texts(root["cellData/ids"][:]) == ["NA", "001"]
    assert _texts(root["RNA/featureData/ids"][:]) == ["NA", "007"]
    assert _texts(root["RNA/featureData/names"][:]) == ["N/A", _ACCENTED]
    # The metadata columns read missing values as they did.
    assert root["cellData/__cytearc_missing__sample"][:].tolist() == [True, False]
    assert root["cellData/sample"][1] == "B"
    assert np.isnan(root["cellData/count"][1])


# Tapestri files.


def _tapestri_with(path: Path, dataset: str, row: int, value: bytes) -> Path:
    """Write a Tapestri file whose ``dataset`` holds ``value`` in ``row``."""
    write_tapestri(tapestri_truth(), path)
    with h5py.File(path, "r+") as handle:
        values = handle[dataset][:].tolist()
        values[row] = value
        del handle[dataset]
        handle.create_dataset(dataset, data=np.array(values))
    return path


@pytest.mark.parametrize(
    ("dataset", "row", "value", "message"),
    [
        pytest.param(
            "assays/dna_read_counts/ra/barcode",
            1,
            b"ACGT\xffACGT-1",
            "The barcodes of assays/dna_read_counts must be UTF-8 text; row 2 of the "
            "dataset 'assays/dna_read_counts/ra/barcode' of 'refused.h5' holds "
            "b'ACGT\\xffACGT-1'",
            id="barcode",
        ),
        pytest.param(
            "assays/dna_read_counts/ra/sample_name",
            0,
            b"d\xe9",
            "The sample names of assays/dna_read_counts must be UTF-8 text; row 1 of "
            "the dataset 'assays/dna_read_counts/ra/sample_name' of 'refused.h5' "
            "holds b'd\\xe9'",
            id="sample-name",
        ),
        pytest.param(
            "assays/protein_read_counts/ca/id",
            2,
            b"Ig\xffG1",
            "The feature IDs of assays/protein_read_counts must be UTF-8 text; row 3 "
            "of the dataset 'assays/protein_read_counts/ca/id' of 'refused.h5' holds "
            "b'Ig\\xffG1'",
            id="feature-id",
        ),
        pytest.param(
            "assays/dna_read_counts/ca/id",
            1,
            b" ",
            "The feature IDs of assays/dna_read_counts must contain non-empty values; "
            "row 2 of the dataset 'assays/dna_read_counts/ca/id' of 'refused.h5' is "
            "missing or blank",
            id="blank-feature-id",
        ),
    ],
)
def test_tapestri_imports_refuse_ids_before_writing(
    tmp_path, monkeypatch, dataset, row, value, message
) -> None:
    monkeypatch.chdir(tmp_path)
    _tapestri_with(Path("refused.h5"), dataset, row, value)
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    refused = _refused_import(
        lambda: TapestriReader("refused.h5", temp_dir=temporary),
        TapestriToZarr,
        "refused.zarr",
    )

    assert type(refused) is ValueError
    assert str(refused) == message
    assert not (tmp_path / "refused.zarr").exists()
    assert not list(temporary.iterdir())


def test_tapestri_variants_refuse_barcodes_that_are_not_utf8(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _tapestri_with(Path("refused.h5"), "assays/dna_variants/ra/barcode", 0, b"\xff")

    with pytest.raises(ValueError) as refused:
        tapestri_source("refused.h5")

    assert str(refused.value) == (
        "The barcodes of assays/dna_variants must be UTF-8 text; row 1 of the "
        "dataset 'assays/dna_variants/ra/barcode' of 'refused.h5' holds b'\\xff'"
    )


def test_pandas_refuses_these_sidecars_with_a_decode_error() -> None:
    # The refusals rest on pandas failing to decode such text, wherever it is,
    # rather than reading it some other way.
    with pytest.raises(UnicodeDecodeError):
        pd.read_csv(io.BytesIO(b"a\tb\nc\t\xff\n"), sep="\t", header=None, dtype=str)
