"""Stores and the counts that the variant truth implies, for the variant tests.

The expected tables come from the truth's arrays alone, so a test that
compares an import with them checks the importer against values it never
computed. ``download_or_skip`` fetches the files of tests marked
integration.
"""

import gzip
import http.client
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tests.clone_helpers import open_clone_store, write_truth_store
from tests.variant_truth import (
    BASES,
    CONTIG,
    PROBES,
    REFERENCE,
    SITE_POSITIONS,
    VariantTruth,
)

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore


def write_variant_store(path: Path, truth: VariantTruth) -> "DataStore":
    """Write a store of the truth's cells, every cell in ``every_cell``."""
    return write_truth_store(path, truth.clones)


def copy_variant_store(template: Path, folder: Path) -> "DataStore":
    target = folder / template.name
    shutil.copytree(template, target)
    return open_clone_store(target)


def mito_label(position: int, contig: str = CONTIG) -> str:
    """A site's label: ``chrM:310``, or the position alone on an unnamed contig."""
    return f"{contig}:{position}" if contig else str(position)


def expected_mito(
    truth: VariantTruth, *, swapped: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The depth and counts that an import of mgatk's files gives: every
    position but the N, every base, both strands.

    ``swapped`` gives what files with swapped strands give when read as
    mgatk files without ``swap_strands``.
    """
    depth_rows = []
    count_rows = []
    ids = truth.row_ids
    for row in range(truth.n_cells):
        for position in SITE_POSITIONS:
            reads = truth.mito[row, position - 1]
            total = int(reads.sum())
            if total:
                depth_rows.append((ids[row], mito_label(position), total))
            for base_index, base in enumerate(BASES):
                forward, reverse = (int(value) for value in reads[base_index])
                if swapped:
                    forward, reverse = reverse, forward
                if forward + reverse:
                    count_rows.append(
                        (
                            ids[row],
                            mito_label(position),
                            base,
                            forward + reverse,
                            forward,
                            reverse,
                        )
                    )
    depth = pd.DataFrame(depth_rows, columns=["cell", "site", "depth"])
    counts = pd.DataFrame(
        count_rows, columns=["cell", "site", "allele", "count", "forward", "reverse"]
    )
    return depth, counts


def expected_variant_sites(
    truth: VariantTruth,
    *,
    reference: bool = True,
    other: bool = False,
    contig: str = CONTIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The depth and counts at the clone variants, as the tools that count a
    reference and an alternative per site give them.

    ``reference`` keeps the reference counts, which MiTo leaves out; with
    ``other`` the other bases are an allele and count toward the depth, as in
    cellsnp-lite. Without either, the depth counts what the tool's DP holds:
    the reference and alternative, or every base for MiTo.
    """
    counts_per_base = truth.mito.sum(axis=3)
    ids = truth.row_ids
    depth_rows = []
    count_rows = []
    for position, ref, alt in truth.variant_sites():
        for row in range(truth.n_cells):
            per_base = counts_per_base[row, position - 1]
            ref_count = int(per_base[BASES.index(ref)])
            alt_count = int(per_base[BASES.index(alt)])
            other_count = int(per_base.sum()) - ref_count - alt_count
            if reference:
                depth = ref_count + alt_count + (other_count if other else 0)
            else:
                depth = int(per_base.sum())
            if depth:
                depth_rows.append((ids[row], mito_label(position, contig), depth))
            alleles = [(alt, alt_count)]
            if reference:
                alleles.append((ref, ref_count))
            if other:
                alleles.append(("other", other_count))
            for allele, count in alleles:
                if count:
                    count_rows.append(
                        (ids[row], mito_label(position, contig), allele, count)
                    )
    return (
        pd.DataFrame(depth_rows, columns=["cell", "site", "depth"]),
        pd.DataFrame(count_rows, columns=["cell", "site", "allele", "count"]),
    )


def expected_got(
    truth: VariantTruth,
    *,
    site: str,
    target: int = 0,
    ambiguous: bool = False,
    extra_wild_type: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The depth and WT, MUT (and ambiguous) counts of one GoT target."""
    ids = truth.row_ids
    depth_rows = []
    count_rows = []
    for row in range(truth.n_cells):
        wild_type, mutant, other = truth.got[row, target].tolist()
        if wild_type < 0:
            continue
        wild_type += extra_wild_type
        alleles = [("WT", wild_type), ("MUT", mutant)]
        if ambiguous:
            alleles.append(("ambiguous", other))
        total = sum(count for _allele, count in alleles)
        if total:
            depth_rows.append((ids[row], site, total))
        for allele, count in alleles:
            if count:
                count_rows.append((ids[row], site, allele, count))
    return (
        pd.DataFrame(depth_rows, columns=["cell", "site", "depth"]),
        pd.DataFrame(count_rows, columns=["cell", "site", "allele", "count"]),
    )


def expected_got_calls(
    truth: VariantTruth, *, site: str, target: int = 0, source: str
) -> pd.DataFrame:
    ids = truth.row_ids
    rows = [
        (ids[row], site, "MUT", source, truth.got_call(row, target))
        for row in range(truth.n_cells)
        if truth.got_call(row, target) != "NA"
    ]
    return pd.DataFrame(rows, columns=["cell", "site", "allele", "source", "call"])


def expected_gift(truth: VariantTruth) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The depth, UMIs, reads and support of each cell, probe and gapfill."""
    ids = truth.row_ids
    counts = truth.gift_counts()
    count_rows = []
    depth: dict[tuple[str, str], int] = {}
    for (row, probe, gapfill), (umis, reads, support) in sorted(counts.items()):
        if row >= truth.n_cells:
            continue
        name = PROBES[probe][0]
        count_rows.append((ids[row], name, gapfill, umis, reads, support))
        depth[(ids[row], name)] = depth.get((ids[row], name), 0) + umis
    return (
        pd.DataFrame(
            [(cell, site, value) for (cell, site), value in depth.items()],
            columns=["cell", "site", "depth"],
        ),
        pd.DataFrame(
            count_rows,
            columns=["cell", "site", "allele", "count", "reads", "support"],
        ),
    )


def table(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """A loaded or expected table as plain text and numbers, sorted, for comparison."""
    chosen = frame if columns is None else frame[columns]
    plain = pd.DataFrame(
        {
            name: (
                chosen[name].astype(str).to_numpy(dtype=object)
                if not pd.api.types.is_numeric_dtype(chosen[name])
                else chosen[name].to_numpy()
            )
            for name in chosen.columns
        }
    )
    keys = [
        name for name in ("cell", "site", "allele", "source") if name in plain.columns
    ]
    return plain.sort_values(keys, kind="stable").reset_index(drop=True)


def assert_tables(loaded: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert that a loaded table holds exactly the expected rows."""
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(
        table(loaded, columns), table(expected, columns), check_dtype=False
    )


def mito_reference_digest() -> str:
    import hashlib

    return hashlib.sha256(REFERENCE.encode()).hexdigest()


# What a download raises when its host is unreachable, slow or cuts the file
# short: IncompleteRead is an HTTPException rather than an OSError, and a gzip
# stream that ends early raises EOFError.
_UNREACHABLE = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
    EOFError,
)
# HTTP statuses of a host that refuses or fails for the moment, as GEO does
# under load; any other error status, such as 404 for a file that moved,
# fails the test.
_TRANSIENT_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})


def _fetched(url: str) -> bytes:
    """The bytes at ``url``, refusing a body shorter than its stated length
    and a gzip file that ends early."""
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
        length = response.headers.get("Content-Length")
    if length is not None and len(data) != int(length):
        raise http.client.IncompleteRead(data, int(length) - len(data))
    if url.endswith(".gz"):
        gzip.decompress(data)
    return data


def download_or_skip(url: str, path: Path) -> Path:
    """Download a file for a test marked integration, or skip the test when
    its host cannot serve the file whole.

    A failed download is tried once more, and the file is written only once
    whole, under a name of its own and then renamed. A status that names a
    missing file, such as 404, fails the test, since skipping would hide a
    file that moved.
    """
    failure: Exception | None = None
    for _attempt in range(2):
        try:
            data = _fetched(url)
        except urllib.error.HTTPError as error:
            if error.code not in _TRANSIENT_STATUSES:
                raise
            failure = error
            continue
        except _UNREACHABLE as error:
            failure = error
            continue
        partial = path.with_name(f"{path.name}.partial")
        partial.write_bytes(data)
        partial.replace(path)
        return path
    pytest.skip(f"{url} could not be downloaded whole: {failure}")
