"""Clones imported from files that the tools and studies published.

TREX's expected ``clones.txt`` (MIT) and six clones of CellTag-multi's iEP
clone table (Zenodo 23442375, CC BY 4.0) are checked in under
``tests/data/clones/``. LARRY's cytokine clone matrix and metadata (GEO
GSE140802, 1.9 MB) are downloaded; that test skips when GEO is unreachable or
cannot serve a file whole.
"""

import gzip
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmread

from tests.clone_helpers import open_clone_store
from tests.storage_helpers import write_count_store

DATA = Path(__file__).parent / "data" / "clones"
_LARRY = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4185nnn/GSM4185644/suppl/"
    "GSM4185644_stateFate_cytokinePerturbation_{}"
)


def _store(path: Path, cell_ids: list[str]):
    counts = np.ones((len(cell_ids), 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=cell_ids)
    store = open_clone_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store, store.snapshot_cell_selection("every_cell")


def test_trex_clones_join_10x_cells_through_a_suffix(tmp_path) -> None:
    table = pd.read_csv(DATA / "trex_clones.txt", sep="\t", dtype=str)
    # Cell Ranger names the cells with "-1"; one more cell has no clone.
    cell_ids = [f"{barcode}-1" for barcode in table["cell_id"]] + ["ACGTACGTACGTACGT-1"]
    store, cells = _store(tmp_path / "store.zarr", cell_ids)

    ref = store.clones.import_clones(
        DATA / "trex_clones.txt", cell_selection=cells, barcode_suffix="-1"
    )

    loaded = store.clones.load_clones(ref)
    assert loaded.clones.to_dict("list") == {
        "clone": ["4", "3", "1", "2"],
        "n_cells": [30, 8, 2, 1],
    }
    expected = dict(zip(cell_ids, [*table["clone_nr"], None], strict=True))
    observed = (
        loaded.cells["clone"].astype(object).where(loaded.cells["clone"].notna(), None)
    )
    assert observed.to_dict() == expected
    assert store.artifacts.load(ref).attrs["n_unmatched_barcodes"] == 0


def test_celltag_multi_clones_join_cells_of_both_assays(tmp_path) -> None:
    table = pd.read_csv(DATA / "celltag_multi_iep_clones.csv", dtype=str, index_col=0)
    store, cells = _store(tmp_path / "store.zarr", table["cell.bc"].tolist())

    ref = store.clones.import_clones(
        DATA / "celltag_multi_iep_clones.csv", cell_selection=cells
    )

    loaded = store.clones.load_clones(ref)
    assert loaded.clones.to_dict("list") == {
        "clone": ["r2_1817", "r1_101", "r2_1818", "r1_1036", "r1_1001", "r1_1002"],
        "n_cells": [7, 6, 4, 3, 2, 2],
    }
    assert loaded.cells["clone"].astype(str).to_dict() == dict(
        zip(table["cell.bc"], table["clone_id"], strict=True)
    )
    # Each row repeats its clone's size, which the import recovers.
    sizes = table["clone.size (RNA & ATAC)"].astype(int).to_numpy()
    np.testing.assert_array_equal(loaded.cells["clone_size"].to_numpy(), sizes)


def test_a_pattern_joins_celltag_multi_cells_to_a_store_of_one_sample(tmp_path) -> None:
    table = pd.read_csv(DATA / "celltag_multi_iep_clones.csv", dtype=str, index_col=0)
    sample = "B4D21-RNA-r2-1-"
    in_sample = table["cell.bc"].str.startswith(sample)
    # The store of one sample names cells by their barcodes alone.
    barcodes = table.loc[in_sample, "cell.bc"].str.removeprefix(sample).tolist()
    store, cells = _store(tmp_path / "store.zarr", barcodes)

    with pytest.warns(
        UserWarning, match=f"{len(table) - len(barcodes)} of 24 barcodes"
    ):
        ref = store.clones.import_clones(
            DATA / "celltag_multi_iep_clones.csv",
            cell_selection=cells,
            barcode_pattern=rf"^{sample}(?P<barcode>[ACGT]{{16}}-1)$",
        )

    loaded = store.clones.load_clones(ref).cells
    assert (
        loaded["clone"].astype(str).tolist()
        == table.loc[in_sample, "clone_id"].tolist()
    )


# What a download raises when GEO is unreachable or cuts the file short:
# IncompleteRead is an HTTPException, and a gzip stream that ends early raises
# EOFError when it is read.
_DOWNLOAD_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
    EOFError,
)


def _fetched(url: str) -> bytes:
    """Return a gzip file's bytes from ``url``, refusing a file that arrives cut short."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
        length = response.headers.get("Content-Length")
    if length is not None and len(data) != int(length):
        raise http.client.IncompleteRead(data, int(length) - len(data))
    gzip.decompress(data)
    return data


def _download(name: str, folder: Path) -> Path:
    """Download one of LARRY's files, or skip when GEO cannot serve it whole.

    A failed download is tried once more. The file is written only once it
    is whole, under a name of its own and then renamed, so no run reads part
    of it.
    """
    path = folder / f"GSM4185644_stateFate_cytokinePerturbation_{name}"
    for attempt in range(2):
        try:
            data = _fetched(_LARRY.format(name))
        except _DOWNLOAD_ERRORS as error:
            if attempt:
                pytest.skip(f"GEO could not serve LARRY's {name} whole: {error}")
            continue
        partial = path.with_name(f"{path.name}.partial")
        partial.write_bytes(data)
        partial.replace(path)
        return path
    raise AssertionError("unreachable: the second attempt returns or skips")


@pytest.mark.integration
def test_larrys_cytokine_clones_match_counts_taken_from_the_raw_files(tmp_path) -> None:
    matrix = _download("clone_matrix.mtx.gz", tmp_path)
    metadata_path = _download("metadata.txt.gz", tmp_path)
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    cell_ids = (metadata["Library"] + "__" + metadata["Cell barcode"]).tolist()
    # GEO's file is clones by cells.
    clones_by_cells = mmread(matrix).tocsr()
    assert clones_by_cells.shape == (5257, len(cell_ids))
    expected_sizes = np.sort(np.diff(clones_by_cells.indptr))[::-1]
    store, cells = _store(tmp_path / "store.zarr", cell_ids)

    ref = store.clones.import_clones(
        matrix, cell_selection=cells, columns={"cell": ["Library", "Cell barcode"]}
    )

    loaded = store.clones.load_clones(ref)
    assert len(loaded.clones) == 5257
    np.testing.assert_array_equal(loaded.clones["n_cells"].to_numpy(), expected_sizes)
    members = clones_by_cells.tocsc()
    in_clone = np.diff(members.indptr) > 0
    assert int(loaded.cells["clone"].notna().sum()) == int(in_clone.sum()) == 25749
    # Each cell's clone is the matrix row that holds it.
    rows = members.indices[members.indptr[:-1][in_clone]]
    labels = loaded.cells["clone"].astype(object).to_numpy()[in_clone]
    np.testing.assert_array_equal(labels.astype(str), rows.astype(str))


class _Response:
    """A response of GEO's server that serves ``data`` under ``length``."""

    def __init__(self, data: bytes, length: int | None, error: Exception | None):
        self._data, self._error = data, error
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        if self._error is not None:
            raise self._error
        return self._data


_WHOLE = gzip.compress(b"%%MatrixMarket matrix coordinate pattern general\n", mtime=0)


@pytest.mark.parametrize(
    "response",
    [
        # The server closes the connection before the length it announced.
        _Response(b"", None, http.client.IncompleteRead(_WHOLE[:10], len(_WHOLE))),
        # The body is shorter than the length it announced.
        _Response(_WHOLE[:10], len(_WHOLE), None),
        # Without a length, the gzip stream itself ends early.
        _Response(_WHOLE[:-8], None, None),
    ],
)
def test_a_download_cut_short_skips_after_one_retry_and_leaves_no_file(
    tmp_path, monkeypatch, response
) -> None:
    calls: list[str] = []

    def urlopen(url: str, timeout: float) -> _Response:
        calls.append(url)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(pytest.skip.Exception, match="could not serve"):
        _download("clone_matrix.mtx.gz", tmp_path)

    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_a_download_that_succeeds_on_its_retry_is_written_whole(
    tmp_path, monkeypatch
) -> None:
    responses = iter(
        [
            _Response(b"", None, http.client.IncompleteRead(b"", len(_WHOLE))),
            _Response(_WHOLE, len(_WHOLE), None),
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: next(responses))

    path = _download("clone_matrix.mtx.gz", tmp_path)

    assert path.read_bytes() == _WHOLE
    assert list(tmp_path.iterdir()) == [path]
