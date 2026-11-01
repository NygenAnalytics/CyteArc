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

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore._operations.clone_artifacts import zarr_write_buffer_bytes
from tests.clone_helpers import open_clone_store, write_truth_store
from tests.variant_truth import (
    BASES,
    CONTIG,
    PROBES,
    REFERENCE,
    SITE_POSITIONS,
    TapestriTruth,
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


def expected_mito_statistics(truth: VariantTruth, rows: list[int]) -> pd.DataFrame:
    """mgatk's per-variant statistics of the chosen rows, from the truth's reads.

    Every alternative base that some chosen cell reads is scored over the
    chosen cells: the pooled frequency, the sample variance of the covered
    cells' frequencies, the cells read twice on each strand, the strand
    correlation of the cells that read it, the cells over each level, the
    largest frequency, the mean depth, and MAESTER's mean base quality.
    """
    reads = truth.mito[rows]
    quality = truth.quality[rows]
    depth = reads.sum(axis=(2, 3))
    records = {}
    for position in SITE_POSITIONS:
        reference = REFERENCE[position - 1]
        site_depth = depth[:, position - 1]
        covered = site_depth > 0
        for base_index, base in enumerate(BASES):
            if base == reference:
                continue
            forward = reads[:, position - 1, base_index, 0]
            reverse = reads[:, position - 1, base_index, 1]
            alternative = forward + reverse
            if not alternative.sum():
                continue
            frequency = alternative[covered] / site_depth[covered]
            shown = alternative > 0
            f, r = forward[shown].astype(float), reverse[shown].astype(float)
            correlation = (
                float(np.corrcoef(f, r)[0, 1])
                if shown.sum() > 1 and f.std() > 0 and r.std() > 0
                else np.nan
            )
            strand_quality = np.concatenate(
                [
                    quality[:, position - 1, base_index, 0][forward > 0],
                    quality[:, position - 1, base_index, 1][reverse > 0],
                ]
            )
            mean = alternative.sum() / site_depth.sum()
            variance = frequency.var(ddof=1) if covered.sum() > 1 else np.nan
            records[f"{CONTIG}:{position}{reference}>{base}"] = {
                "mean": mean,
                "variance": variance,
                "vmr": variance / (mean + 1e-11),
                "n_cells_conf_detected": int(((forward >= 2) & (reverse >= 2)).sum()),
                "strand_correlation": correlation,
                "n_cells_over_5": int((frequency >= 0.05).sum()),
                "n_cells_over_10": int((frequency >= 0.10).sum()),
                "n_cells_over_20": int((frequency >= 0.20).sum()),
                "n_cells_over_95": int((frequency >= 0.95).sum()),
                "max_heteroplasmy": frequency.max(),
                "mean_coverage": site_depth.sum() / len(rows),
                "n_cells_covered": int(covered.sum()),
                "mean_quality": strand_quality.mean(),
            }
    return pd.DataFrame.from_dict(records, orient="index")


def mito_variant_labels(truth: VariantTruth) -> set[str]:
    """The labels of the truth's clone variants, such as ``chrM:13A>G``."""
    return {
        f"{CONTIG}:{position}{reference}>{alternative}"
        for position, reference, alternative in truth.variant_sites()
    }


RCRS_CELLS = ("cell0", "cell1", "cell2")
# The alternative base of each position that write_rcrs_mgatk writes, None for
# C, or T where the reference reads C: masked positions, a NUMT position that
# the NUMT does not change to C, and 1000, which no mask lists.
RCRS_VARIANTS = {
    310: "C",
    16293: "C",
    16218: "T",
    2617: "G",
    12684: "C",
    1000: None,
}


def write_rcrs_mgatk(
    folder: Path, alternative_reads: tuple[int, int, int] = (4, 4, 4)
) -> Path:
    """Write an mgatk folder on the rCRS reference of MiTo's fixture: each of
    ``RCRS_CELLS`` reads 6 reference reads and its ``alternative_reads`` at
    each variant of ``RCRS_VARIANTS``, as many on each strand."""
    import gzip

    folder.mkdir(parents=True)
    data = Path(__file__).parent / "data" / "variants"
    with gzip.open(data / "mito_mtscatac" / "pL1000.refAllele.txt.gz", "rt") as handle:
        reference = {
            int(position): base.upper()
            for position, base in (line.split() for line in handle if line.strip())
        }
    (folder / "chrM_refAllele.txt").write_text(
        "".join(f"{position}\t{base}\n" for position, base in reference.items())
    )
    rows: dict[str, list[str]] = {base: [] for base in "ACGT"}
    for cell, reads in zip(RCRS_CELLS, alternative_reads, strict=True):
        for position, alternative in RCRS_VARIANTS.items():
            base = reference[position]
            if alternative is None:
                alternative = "C" if base != "C" else "T"
            rows[base].append(f"{position},{cell},6,6")
            if reads:
                rows[alternative].append(f"{position},{cell},{reads},{reads}")
    for base, lines in rows.items():
        (folder / f"s.{base}.txt.gz").write_bytes(
            gzip.compress("".join(f"{line}\n" for line in lines).encode())
        )
    (folder / "s.depthTable.txt").write_text(
        "".join(f"{cell}\t1\n" for cell in RCRS_CELLS)
    )
    return folder


def rcrs_store(path: Path) -> "DataStore":
    """Write a store of ``RCRS_CELLS``, every cell in ``every_cell``."""
    from cytearc.datastore.datastore import DataStore
    from tests.storage_helpers import write_count_store

    write_count_store(
        str(path),
        {"RNA": np.ones((len(RCRS_CELLS), 2), dtype=int)},
        "uint16",
        cell_ids=list(RCRS_CELLS),
    )
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    store.cells.insert("every_cell", np.ones(len(RCRS_CELLS), dtype=bool))
    return store


def _widest_chunk_bytes(group, names: list[str]) -> int:
    """The bytes of a chunk of the widest of these stored arrays."""
    return max(
        int(group[name].chunks[0]) * int(group[name].dtype.itemsize) for name in names
    )


def band_budget(store: "DataStore", counts, rows: int) -> int:
    """The memory budget at which ``ds.variants.informative`` reads bands of
    ``rows`` stored rows of ``counts``: its per-variant sums (240 bytes per
    allele and 16 per site), the two chunks of the widest array that a band's
    reads decompress beyond its rows, the file buffers of Zarr's writes and
    200 bytes per row."""
    group = store.artifacts.load(counts)
    fields = group.attrs["count_fields"]
    names = ["depth_sites", "depth", "count_alleles", "counts"]
    if "forward" in fields:
        names += ["counts_forward", "counts_reverse"]
    if "quality_forward" in fields:
        names += ["counts_quality_forward", "counts_quality_reverse"]
    n_alleles = int(group["allele_sites"].shape[0])
    n_sites = int(group["site_positions"].shape[0])
    return (
        240 * n_alleles
        + 16 * n_sites
        + 2 * _widest_chunk_bytes(group, names)
        + zarr_write_buffer_bytes()
        + 200 * rows
    )


def clone_band_budget(
    store: "DataStore", counts, rows: int, *, n_cells: int, n_variants: int
) -> int:
    """The memory budget at which ``ds.variants.clones`` reads bands of
    ``rows`` stored rows of ``counts``: the frequencies of ``n_cells`` cells
    at ``n_variants`` informative variants, the two chunks that a band's
    reads decompress beyond its rows, and per row 200 bytes and 32 per
    variant. It leaves out what writing the clones holds after the bands, the
    file buffers of Zarr's writes among them, which the budget must fit too."""
    group = store.artifacts.load(counts)
    names = ["depth_sites", "depth", "count_alleles", "counts"]
    return (
        8 * n_cells * n_variants
        + 2 * _widest_chunk_bytes(group, names)
        + (200 + 32 * n_variants) * rows
    )


#: The site and allele of each readable Tapestri variant, by position in
#: ``TAPESTRI_VARIANTS``, written out by hand: the deletion ``TC/T`` and the
#: SNV ``T/G`` of chr1:2000 are alleles ``T`` and ``GC`` of the reference
#: ``TC``, as VCF writes two alternatives of one record.
TAPESTRI_ALLELES = {
    0: ("chr1:1000", "A"),
    1: ("chr1:1000", "T"),
    3: ("chr1:2000", "T"),
    4: ("chr1:2000", "GC"),
    5: ("chr2:500", "G"),
    6: ("chr2:500", "C"),
    9: ("chr7:400", "A"),
}
TAPESTRI_REFERENCES = {
    "chr1:1000": "G",
    "chr1:2000": "TC",
    "chr2:500": "A",
    "chr7:400": "G",
}
#: The variants that pass and whose alleles are bases.
TAPESTRI_PASSING = (0, 1, 3, 4, 5, 9)
#: DAb-seq's FLT3-ITD is a site of its own, named as the file names it.
DABSEQ_ITD_SITE = "FLT3-ITD:chr13:28608251:T/TCTC"


def dabseq_ids(truth: TapestriTruth) -> list[str]:
    """Each row's barcode as DAb-seq writes it: barcode, tube 1 and sample."""
    return [
        f"{barcode}-1-{sample}"
        for barcode, sample in zip(truth.barcodes, truth.samples, strict=True)
    ]


def expected_tapestri(
    truth: TapestriTruth, columns: tuple[int, ...] = TAPESTRI_PASSING
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The depth, counts and calls that an import of the chosen variants gives.

    The depth is each position's DP, the counts each variant's alternative
    reads, and the calls each variant's NGT with its GQ where the cell has
    reads.
    """
    ids = truth.row_ids
    depth_rows, count_rows, call_rows = [], [], []
    for row in range(truth.n_rows):
        seen: set[str] = set()
        for column in columns:
            site, allele = TAPESTRI_ALLELES[column]
            depth = int(truth.depth[row, column])
            if site not in seen and depth:
                depth_rows.append((ids[row], site, depth))
            seen.add(site)
            reads = int(truth.alternative[row, column])
            if reads:
                count_rows.append((ids[row], site, allele, reads))
            if depth:
                call_rows.append(
                    (
                        ids[row],
                        site,
                        allele,
                        "NGT",
                        str(int(truth.genotypes[row, column])),
                        float(truth.quality[row, column]),
                    )
                )
    return (
        pd.DataFrame(depth_rows, columns=["cell", "site", "depth"]),
        pd.DataFrame(count_rows, columns=["cell", "site", "allele", "count"]),
        pd.DataFrame(
            call_rows,
            columns=["cell", "site", "allele", "source", "call", "quality"],
        ),
    )


def expected_dabseq(
    truth: TapestriTruth,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The depth, counts and calls of the truth's DAb-seq file.

    The reads follow ``write_dabseq``'s rules, derived from the truth here:
    the alternatives' reads, the overlap cell's second one dropped; the
    reference reads of a position, those that no alternative claims, once
    per position; FLT3-ITD's 25 of 60 reads in the cells of clone 4, without
    reference reads, at a site of its own; and the GT with its GQ where the
    cell has reads, NaN where GQ is missing.
    """
    ids = dabseq_ids(truth)
    columns = tuple(sorted(TAPESTRI_ALLELES))
    alternative = truth.alternative.copy()
    alternative[truth.overlap_row, 6] = 0
    itd_rows = set(np.flatnonzero(truth.clones.clones == 4).tolist())
    depth_rows, count_rows, call_rows = [], [], []
    for row in range(truth.n_rows):
        depth_at: dict[str, int] = {}
        claimed: dict[str, int] = {}
        for column in columns:
            site, allele = TAPESTRI_ALLELES[column]
            depth_at[site] = int(truth.depth[row, column])
            claimed[site] = claimed.get(site, 0) + int(alternative[row, column])
            if alternative[row, column]:
                count_rows.append(
                    (ids[row], site, allele, int(alternative[row, column]))
                )
            genotype = int(truth.genotypes[row, column])
            if truth.depth[row, column] > 0:
                quality = np.nan if genotype == 3 else float(truth.quality[row, column])
                call_rows.append((ids[row], site, allele, "GT", str(genotype), quality))
        for site, depth in depth_at.items():
            if depth > 0:
                depth_rows.append((ids[row], site, depth))
            if depth - claimed[site] > 0:
                count_rows.append(
                    (ids[row], site, TAPESTRI_REFERENCES[site], depth - claimed[site])
                )
        # DAb-seq's wild-type FLT3-ITD calls, in cells without ITDseek's
        # reads, are left out.
        if row in itd_rows:
            depth_rows.append((ids[row], DABSEQ_ITD_SITE, 60))
            count_rows.append((ids[row], DABSEQ_ITD_SITE, "TCTC", 25))
            call_rows.append((ids[row], DABSEQ_ITD_SITE, "TCTC", "GT", "1", 100.0))
    return (
        pd.DataFrame(depth_rows, columns=["cell", "site", "depth"]),
        pd.DataFrame(count_rows, columns=["cell", "site", "allele", "count"]),
        pd.DataFrame(
            call_rows,
            columns=["cell", "site", "allele", "source", "call", "quality"],
        ),
    )
