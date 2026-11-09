#!/usr/bin/env python3
"""Build the human and mouse STRING networks that potency scoring downloads.

Each network keeps the STRING v12.5 protein links whose score, recombined
without the `database` channel, is at least 0.700. STRING's `database` channel
draws on curated resources such as KEGG, whose own terms still apply, so it is
left out. The other channels are recombined with STRING's documented formula:
remove the prior of 0.041 from each channel score, combine the channels as
1 - prod(1 - s), add the prior back once, multiply by 1000 and truncate.
Proteins become genes through STRING's preferred names and `Ensembl_gene`
aliases, and the largest connected component is kept.

Each output is a NumPy `.npz` archive of `ensembl_ids`, `symbols`, `edges`
(two rows of gene positions, each edge once with the lower position first),
`scores` (the recombined scores out of 1000) and `metadata`, a JSON text that
names STRING, its version and license, the changes made, the cut-off, and the
URL, size and SHA-256 digest of every source file. Archive members carry a
fixed timestamp, so rebuilding from the same source files on the same platform
gives identical bytes. `manifest.json` lists each output's digest.

STRING data are under CC BY 4.0, and the outputs stay under it. The script
writes local files only. `cytearc.datasets.STRING_NETWORKS` records the size and
digest of each published file, so a rebuild that changes a file needs that
record updated with it.

Example:
    uv run python scripts/build_string_networks.py --out build/string_networks
"""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

STRING_VERSION = "12.5"
DOWNLOAD_ROOT = "https://stringdb-downloads.org/download"
SPECIES = {"human": 9606, "mouse": 10090}
PRIOR = 0.041
CUTOFF = 700
CHANNELS = (
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
    "textmining",
)
DROPPED_CHANNELS = ("database",)
CITATION = (
    "Szklarczyk D, Nastou K, Koutrouli M, et al. The STRING database in 2025: "
    "protein networks with directionality of regulation. Nucleic Acids Res. "
    "2025;53(D1):D730-D737. doi:10.1093/nar/gkae1113"
)
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def source_url(kind: str, taxid: int) -> str:
    version = f"v{STRING_VERSION}"
    return f"{DOWNLOAD_ROOT}/{kind}.{version}/{taxid}.{kind}.{version}.txt.gz"


def fetch(url: str, directory: Path) -> tuple[Path, dict]:
    """Download a file once, returning its path and its size and digest."""
    path = directory / url.rsplit("/", 1)[1]
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with (
            urllib.request.urlopen(url, timeout=600) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1 << 20)
        partial.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return path, {
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    with gzip.open(path, "rb") as handle:
        return pd.read_csv(handle, **kwargs)


def combine(scores: dict[str, np.ndarray], channels: tuple[str, ...]) -> np.ndarray:
    """STRING's combined score of the given channels, out of 1000 and truncated."""
    remaining = np.ones(len(next(iter(scores.values()))))
    for channel in channels:
        without_prior = np.clip(
            (scores[channel] / 1000.0 - PRIOR) / (1.0 - PRIOR), 0.0, None
        )
        remaining *= 1.0 - without_prior
    combined = 1.0 - remaining
    return np.floor((combined + PRIOR * (1.0 - combined)) * 1000.0 + 1e-9).astype(
        np.int64
    )


def build_network(species: str, work: Path) -> tuple[dict[str, np.ndarray], dict]:
    taxid = SPECIES[species]
    sources = {}
    info_path, sources["protein.info"] = fetch(source_url("protein.info", taxid), work)
    alias_path, sources["protein.aliases"] = fetch(
        source_url("protein.aliases", taxid), work
    )
    links_path, sources["protein.links.detailed"] = fetch(
        source_url("protein.links.detailed", taxid), work
    )

    info = read_table(
        info_path,
        sep="\t",
        usecols=[0, 1],
        names=["protein", "symbol"],
        header=0,
        dtype=str,
    )
    aliases = read_table(
        alias_path, sep="\t", names=["protein", "alias", "source"], header=0, dtype=str
    )
    ensembl = (
        aliases.loc[aliases["source"] == "Ensembl_gene"]
        .groupby("protein")["alias"]
        .agg(lambda values: sorted(set(values)))
    )
    several = int((ensembl.map(len) > 1).sum())
    info["ensembl"] = info["protein"].map(ensembl.map(lambda values: values[0]))
    info = info.dropna(subset=["symbol", "ensembl"]).reset_index(drop=True)
    position = pd.Series(np.arange(len(info)), index=info["protein"])

    links = read_table(
        links_path,
        sep=" ",
        header=0,
        dtype={
            **{channel: np.int64 for channel in CHANNELS},
            "combined_score": np.int64,
        },
    )
    first = position.reindex(links["protein1"]).to_numpy()
    second = position.reindex(links["protein2"]).to_numpy()
    # Every link is listed in both directions; keep each once.
    once = first < second
    channel_scores = {channel: links[channel].to_numpy()[once] for channel in CHANNELS}
    published = links["combined_score"].to_numpy()[once]
    first = first[once].astype(np.int64)
    second = second[once].astype(np.int64)
    del links

    # The formula must reproduce STRING's own combined score from every channel.
    recombined_all = combine(channel_scores, CHANNELS)
    largest_error = int(np.abs(recombined_all - published).max())
    if largest_error > 1:
        raise RuntimeError(
            f"STRING's documented formula reproduces its {species} combined scores only within "
            f"{largest_error} of 1000; the channel columns or the formula have changed"
        )
    kept_channels = tuple(
        channel for channel in CHANNELS if channel not in DROPPED_CHANNELS
    )
    recombined = combine(channel_scores, kept_channels)
    keep = recombined >= CUTOFF
    first, second, recombined = first[keep], second[keep], recombined[keep]

    n = len(info)
    adjacency = sparse.coo_matrix(
        (np.ones(len(first)), (first, second)), shape=(n, n)
    ).tocsr()
    adjacency = adjacency + adjacency.T
    linked = np.flatnonzero(np.diff(adjacency.indptr) > 0)
    _, component = connected_components(adjacency[linked][:, linked], directed=False)
    largest = linked[component == np.bincount(component).argmax()]
    renumber = np.full(n, -1, dtype=np.int64)
    renumber[largest] = np.arange(len(largest))
    inside = (renumber[first] >= 0) & (renumber[second] >= 0)
    a, b = renumber[first[inside]], renumber[second[inside]]
    scores = recombined[inside]
    low, high = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((high, low))
    edges = np.stack([low[order], high[order]]).astype(
        np.uint16 if len(largest) < 2**16 else np.uint32
    )

    metadata = {
        "source": "STRING",
        "stringVersion": STRING_VERSION,
        "species": species,
        "taxid": taxid,
        "license": LICENSE,
        "licenseUrl": LICENSE_URL,
        "citation": CITATION,
        "changes": (
            "Removed STRING's database channel, recombined the other channel scores with STRING's "
            f"documented formula (prior {PRIOR}), kept links scoring at least {CUTOFF} of 1000, "
            "mapped proteins to genes through STRING's preferred names and Ensembl_gene aliases, "
            "and kept the largest connected component."
        ),
        "channels": list(kept_channels),
        "droppedChannels": list(DROPPED_CHANNELS),
        "cutoff": CUTOFF,
        "prior": PRIOR,
        "genes": int(len(largest)),
        "edges": int(edges.shape[1]),
        "proteinsWithSeveralEnsemblGenes": several,
        "largestRecombinationError": largest_error,
        "sources": sources,
    }
    arrays = {
        "ensembl_ids": info["ensembl"].to_numpy()[largest].astype(str),
        "symbols": info["symbol"].to_numpy()[largest].astype(str),
        "edges": edges,
        "scores": scores[order].astype(np.uint16),
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    return arrays, metadata


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """A NumPy archive whose members carry a fixed timestamp, so its bytes are reproducible."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            member = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_TIME)
            member.external_attr = 0o644 << 16
            archive.writestr(
                member,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for the networks and manifest.json",
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=None,
        help="Directory that keeps the STRING source files between runs (default: <out>/sources)",
    )
    parser.add_argument(
        "--species", nargs="+", choices=sorted(SPECIES), default=sorted(SPECIES)
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    downloads = args.downloads or args.out / "sources"
    downloads.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stringVersion": STRING_VERSION,
        "cutoff": CUTOFF,
        "droppedChannels": list(DROPPED_CHANNELS),
        "license": LICENSE,
        "citation": CITATION,
        "files": {},
    }
    for species in args.species:
        arrays, metadata = build_network(species, downloads)
        name = f"{species}_{SPECIES[species]}.npz"
        path = args.out / name
        write_npz(path, arrays)
        manifest["files"][name] = {
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "genes": metadata["genes"],
            "edges": metadata["edges"],
            "taxid": metadata["taxid"],
            "sources": metadata["sources"],
        }
        print(
            f"{name}: {metadata['genes']} genes, {metadata['edges']} edges, {path.stat().st_size} bytes"
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
