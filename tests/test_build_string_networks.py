"""``scripts/build_string_networks.py`` on small STRING-like source files.

The source files go into the script's download directory, which it reads in
place of downloading, so nothing touches the network. Their channel scores are
chosen so that each expected score is known without the script's formula: a
link with one channel scores that channel, and the scores derived by hand are
spelled out where they are used.
"""

import gzip
import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from scripts import build_string_networks as build

CHANNELS = build.CHANNELS
INFO = [
    ("9606.P1", "GA"),
    ("9606.P2", "GB"),
    ("9606.P3", "GC"),
    ("9606.P4", "GD"),
    ("9606.P5", "GE"),
    ("9606.P6", "GF"),
    ("9606.P7", "GG"),
    ("9606.P8", "GH"),
]
ALIASES = [
    ("9606.P1", "ENSG00000000001", "Ensembl_gene"),
    # Two genes for one protein: the first in sorted order names it.
    ("9606.P2", "ENSG00000000009", "Ensembl_gene"),
    ("9606.P2", "ENSG00000000002", "Ensembl_gene"),
    ("9606.P3", "ENSG00000000003", "Ensembl_gene"),
    ("9606.P4", "ENSG00000000004", "Ensembl_gene"),
    ("9606.P5", "ENSG00000000005", "Ensembl_gene"),
    ("9606.P6", "ENSG00000000006", "Ensembl_gene"),
    ("9606.P7", "ENSG00000000007", "Ensembl_gene"),
    # P8 has no Ensembl gene, so it cannot be a node.
    ("9606.P8", "hsa:8", "KEGG_GENEID"),
    ("9606.P1", "GA-alias", "BioMart_HUGO"),
]
# Each link: its channel scores and STRING's combined score of all seven.
LINKS = [
    # One channel scores itself, so these keep 800 and 750.
    ("9606.P1", "9606.P2", {"experimental": 800}, 800),
    ("9606.P2", "9606.P3", {"textmining": 750}, 750),
    # Two channels of 500: (0.5 - 0.041) / 0.959 = 0.478624 each without the
    # prior, combined 1 - 0.521376**2 = 0.728166, plus 0.041 * 0.271834 gives
    # 0.739311, so 739.
    ("9606.P3", "9606.P4", {"experimental": 500, "textmining": 500}, 739),
    # STRING scores 958 with the database channel (900 with experiments at 600:
    # 1 - 0.104275 * 0.417101 = 0.956507, plus 0.041 * 0.043493 = 0.958290);
    # without it the link keeps 600 and falls below the cut-off.
    ("9606.P4", "9606.P1", {"database": 900, "experimental": 600}, 958),
    # A second component, smaller than P1 to P4, which the build leaves out.
    ("9606.P5", "9606.P6", {"experimental": 900}, 900),
    # Below the cut-off.
    ("9606.P7", "9606.P1", {"coexpression": 650}, 650),
    # Its protein has no Ensembl gene.
    ("9606.P8", "9606.P1", {"experimental": 950}, 950),
]


def write_gzip(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_sources(directory: Path, links=LINKS) -> dict[str, str]:
    """Write the three human source files where the script looks; return their digests."""
    directory.mkdir(parents=True, exist_ok=True)
    names = {
        kind: build.source_url(kind, 9606).rsplit("/", 1)[1]
        for kind in ("protein.info", "protein.aliases", "protein.links.detailed")
    }
    write_gzip(
        directory / names["protein.info"],
        ["#string_protein_id\tpreferred_name\tprotein_size\tannotation"]
        + [f"{protein}\t{name}\t100\tA protein" for protein, name in INFO],
    )
    write_gzip(
        directory / names["protein.aliases"],
        ["#string_protein_id\talias\tsource"]
        + [f"{protein}\t{alias}\t{source}" for protein, alias, source in ALIASES],
    )
    rows = ["protein1 protein2 " + " ".join(CHANNELS) + " combined_score"]
    for first, second, scores, combined in links:
        values = " ".join(str(scores.get(channel, 0)) for channel in CHANNELS)
        # STRING lists every link in both directions.
        rows.append(f"{first} {second} {values} {combined}")
        rows.append(f"{second} {first} {values} {combined}")
    write_gzip(directory / names["protein.links.detailed"], rows)
    return {
        kind: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for kind, name in names.items()
    }


@pytest.mark.parametrize("score", [41, 42, 500, 699, 700, 999])
def test_one_channel_combines_to_its_own_score(score):
    for channel in CHANNELS:
        scores = {name: np.array([0]) for name in CHANNELS}
        scores[channel] = np.array([score])
        assert build.combine(scores, CHANNELS).tolist() == [score]


def test_scores_below_the_prior_combine_to_the_prior_and_two_halves_to_739():
    # Three links: no evidence, every channel at 30, and two channels at 500.
    scores = {name: np.array([0, 30, 0]) for name in CHANNELS}
    scores["experimental"] = np.array([0, 30, 500])
    scores["textmining"] = np.array([0, 30, 500])
    assert build.combine(scores, CHANNELS).tolist() == [41, 41, 739]


def test_the_build_drops_the_database_channel_weak_links_and_small_components(
    tmp_path,
):
    digests = write_sources(tmp_path)

    arrays, metadata = build.build_network("human", tmp_path)

    assert arrays["symbols"].tolist() == ["GA", "GB", "GC", "GD"]
    assert arrays["ensembl_ids"].tolist() == [
        "ENSG00000000001",
        "ENSG00000000002",
        "ENSG00000000003",
        "ENSG00000000004",
    ]
    assert arrays["edges"].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert arrays["edges"].dtype == np.uint16
    assert arrays["scores"].tolist() == [800, 750, 739]
    assert json.loads(str(arrays["metadata"])) == metadata
    assert metadata["genes"] == 4 and metadata["edges"] == 3
    assert metadata["channels"] == [c for c in CHANNELS if c != "database"]
    assert metadata["droppedChannels"] == ["database"]
    assert metadata["cutoff"] == 700
    assert metadata["proteinsWithSeveralEnsemblGenes"] == 1
    assert metadata["largestRecombinationError"] == 0
    assert metadata["stringVersion"] == "12.5"
    assert metadata["license"] == "CC BY 4.0"
    assert "doi:10.1093/nar/gkae1113" in metadata["citation"]
    assert "Removed STRING's database channel" in metadata["changes"]
    assert {kind: source["sha256"] for kind, source in metadata["sources"].items()} == (
        digests
    )
    for kind, source in metadata["sources"].items():
        assert source["url"] == build.source_url(kind, 9606)


def test_the_build_refuses_combined_scores_that_its_formula_cannot_reproduce(
    tmp_path,
):
    # STRING would score this one-channel link 650, not 645.
    links = [
        link
        if link[0] != "9606.P7"
        else ("9606.P7", "9606.P1", {"coexpression": 650}, 645)
        for link in LINKS
    ]
    write_sources(tmp_path, links)

    with pytest.raises(RuntimeError, match="only within 5 of 1000"):
        build.build_network("human", tmp_path)


def test_main_writes_reproducible_networks_and_their_manifest(tmp_path, capsys):
    write_sources(tmp_path / "sources")
    arguments = ["--downloads", str(tmp_path / "sources"), "--species", "human"]

    assert build.main(["--out", str(tmp_path / "first"), *arguments]) == 0
    assert build.main(["--out", str(tmp_path / "second"), *arguments]) == 0

    first = (tmp_path / "first" / "human_9606.npz").read_bytes()
    assert first == (tmp_path / "second" / "human_9606.npz").read_bytes()
    assert sorted(path.name for path in (tmp_path / "first").iterdir()) == [
        "human_9606.npz",
        "manifest.json",
    ]
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    entry = manifest["files"]["human_9606.npz"]
    assert entry["sha256"] == hashlib.sha256(first).hexdigest()
    assert entry["bytes"] == len(first)
    assert (entry["genes"], entry["edges"], entry["taxid"]) == (4, 3, 9606)
    assert manifest["droppedChannels"] == ["database"]
    assert manifest["cutoff"] == 700
    with np.load(tmp_path / "first" / "human_9606.npz", allow_pickle=False) as stored:
        assert sorted(stored.files) == [
            "edges",
            "ensembl_ids",
            "metadata",
            "scores",
            "symbols",
        ]
    # The script prints what it built and nothing about publishing.
    printed = capsys.readouterr().out
    assert "human_9606.npz: 4 genes, 3 edges" in printed
    assert "hf " not in printed and "bucket" not in printed


def test_fetch_downloads_a_source_once_and_records_its_digest(tmp_path, monkeypatch):
    served = tmp_path / "served"
    write_sources(served / "protein.info.v12.5")
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            requests.append(self.path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=served))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setattr(build, "DOWNLOAD_ROOT", root)
        url = build.source_url("protein.info", 9606)
        expected = (served / "protein.info.v12.5" / url.rsplit("/", 1)[1]).read_bytes()

        path, record = build.fetch(url, tmp_path / "downloads")
        again, record_again = build.fetch(url, tmp_path / "downloads")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert path == again and path.read_bytes() == expected
    assert (
        record
        == record_again
        == {
            "url": url,
            "bytes": len(expected),
            "sha256": hashlib.sha256(expected).hexdigest(),
        }
    )
    assert len(requests) == 1
    assert [p.name for p in (tmp_path / "downloads").iterdir()] == [path.name]
