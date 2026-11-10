"""Download, cache and check the STRING networks that potency scoring reads.

``tests/data/networks/synthetic_network.npz`` stands in for a STRING network: 30
made-up genes in the layout that ``scripts/build_string_networks.py`` writes,
recorded here as the human network. ``FakeHub`` serves it for downloads, and
real requests that fail go to the loopback hub or a closed loopback port. Every
test but the live check runs with the network blocked and the cache under
``tmp_path``.
"""

import hashlib
import json
import shutil
import socket
from pathlib import Path

import httpx2
import numpy as np
import pytest
from huggingface_hub import hf_api
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
)

from cytearc import datasets
from cytearc.features.gene_reference import default_cache_dir
from cytearc.utils.cache import cache_directory
from cytearc.utils.digest import file_sha256
from tests.fixtures_hub import BUCKET_ID as LOOPBACK_BUCKET
from tests.fixtures_hub import FakeHub
from tests.network_helpers import NETWORK_SKIP, hub_or_skip

SYNTHETIC = Path(__file__).parent / "data" / "networks" / "synthetic_network.npz"
GENES = 30
COMMAND = (
    'cytearc.datasets.connect("cytearc_networks").download('
    '"string_v12.5/human_9606.npz", destination="<directory>")'
)
NETWORK_ARGUMENT = 'network="<directory>/string_v12.5/human_9606.npz"'


def synthetic_network_arrays() -> dict[str, np.ndarray]:
    """The synthetic network: a ring of 30 genes and a chord of length 7 from every third."""
    genes = np.arange(GENES)
    starts = np.arange(0, GENES, 3)
    pairs = np.concatenate(
        [
            np.stack([genes, (genes + 1) % GENES]),
            np.stack([starts, (starts + 7) % GENES]),
        ],
        axis=1,
    )
    low, high = pairs.min(axis=0), pairs.max(axis=0)
    order = np.lexsort((high, low))
    edges = np.stack([low[order], high[order]]).astype(np.uint16)
    metadata = {
        "source": "synthetic",
        "species": "human",
        "taxid": 9606,
        "genes": GENES,
        "edges": int(edges.shape[1]),
        "cutoff": 700,
    }
    return {
        "ensembl_ids": np.array([f"ENSG9{i:010d}" for i in range(1, GENES + 1)]),
        "symbols": np.array([f"SYN{i}" for i in range(1, GENES + 1)]),
        "edges": edges,
        "scores": (700 + (37 * np.arange(edges.shape[1])) % 300).astype(np.uint16),
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }


@pytest.fixture(autouse=True)
def _offline_unless_live(request):
    """Keep every test except the live check off the network."""
    if request.node.get_closest_marker("integration") is None:
        request.getfixturevalue("hub_offline")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """A cache root under ``tmp_path``; ``hub_offline`` would leave the user's."""
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    return root


@pytest.fixture
def synthetic(monkeypatch):
    """Record the synthetic network as the only network, the human one."""
    entry = datasets.NetworkFile(
        species="human",
        taxid=9606,
        string_version="12.5",
        name="human_9606.npz",
        size=SYNTHETIC.stat().st_size,
        sha256=file_sha256(SYNTHETIC),
    )
    monkeypatch.setattr(datasets, "STRING_NETWORKS", {"human": entry})
    return entry


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A ``FakeHub`` in place of the public bucket."""
    fake = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(datasets, "list_bucket_tree", fake.list_bucket_tree)
    monkeypatch.setattr(datasets, "download_bucket_files", fake.download_bucket_files)
    return fake


def publish(hub: FakeHub, entry: datasets.NetworkFile, data: bytes) -> None:
    hub.put(f"cytearc_networks/{entry.remote_path}", data, bucket_id="Nygen/cytebase")


def files_under(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def downloads(hub: FakeHub) -> int:
    return sum(call[0] == "download_bucket_files" for call in hub.calls)


def cached_path(cache: Path) -> Path:
    return cache / "cytearc" / "networks" / "string_v12.5" / "human_9606.npz"


def test_the_synthetic_network_holds_the_documented_graph():
    with np.load(SYNTHETIC, allow_pickle=False) as stored:
        assert sorted(stored.files) == sorted(synthetic_network_arrays())
        for name, expected in synthetic_network_arrays().items():
            np.testing.assert_array_equal(stored[name], expected)
            assert stored[name].dtype == expected.dtype
    edges = synthetic_network_arrays()["edges"]
    assert edges.shape == (2, 40)
    assert (edges[0] < edges[1]).all()
    assert len({tuple(pair) for pair in edges.T}) == 40


def test_cache_directory_follows_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_directory("networks") == tmp_path / "xdg" / "cytearc" / "networks"
    monkeypatch.setenv("XDG_CACHE_HOME", "~/.xdg")
    assert (
        cache_directory("networks")
        == tmp_path / "home" / ".xdg" / "cytearc" / "networks"
    )
    for unset in ("", None):
        if unset is None:
            monkeypatch.delenv("XDG_CACHE_HOME")
        else:
            monkeypatch.setenv("XDG_CACHE_HOME", unset)
        assert cache_directory("networks") == (
            tmp_path / "home" / ".cache" / "cytearc" / "networks"
        )


def test_gene_references_and_networks_share_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("CYTEARC_GENE_REFERENCE_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == cache_directory("gene_reference")
    assert default_cache_dir().parent == cache_directory("networks").parent


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../networks"])
def test_cache_directory_takes_one_name(name):
    with pytest.raises(ValueError, match="one path component"):
        cache_directory(name)


def test_first_use_downloads_the_network_and_later_uses_read_the_cache(
    cache, synthetic, hub
):
    publish(hub, synthetic, SYNTHETIC.read_bytes())

    path = datasets.string_network_file("human")

    assert path == cached_path(cache)
    assert path.read_bytes() == SYNTHETIC.read_bytes()
    # The staging directory is gone; the network is the only file.
    assert files_under(cache) == ["cytearc/networks/string_v12.5/human_9606.npz"]
    assert downloads(hub) == 1
    assert datasets.string_network_file("human") == path
    assert downloads(hub) == 1
    # Every request was anonymous.
    assert hub.calls and all(call[-1] is False for call in hub.calls)


def test_a_download_with_another_digest_raises_and_saves_nothing(cache, synthetic, hub):
    data = bytearray(SYNTHETIC.read_bytes())
    data[100] ^= 0xFF
    publish(hub, synthetic, bytes(data))

    with pytest.raises(ValueError) as raised:
        datasets.string_network_file("human")

    message = str(raised.value)
    assert message.startswith(
        f"The file downloaded from {synthetic.uri} does not match"
    )
    assert f"expected {len(data):,} bytes with SHA-256 {synthetic.sha256}" in message
    assert (
        f"found {len(data):,} bytes with SHA-256 {hashlib.sha256(data).hexdigest()}"
        in message
    )
    assert message.endswith("Nothing was saved.")
    # Neither the file nor the directories the download made are left.
    assert not cache.exists()


def test_a_download_of_another_size_raises_and_saves_nothing(cache, synthetic, hub):
    data = SYNTHETIC.read_bytes() + b"\0"
    publish(hub, synthetic, data)

    with pytest.raises(ValueError) as raised:
        datasets.string_network_file("human")

    message = str(raised.value)
    assert f"expected {synthetic.size:,} bytes" in message
    assert f"found {synthetic.size + 1:,} bytes" in message
    assert not cache.exists()


def test_a_transfer_that_fails_part_way_saves_nothing(
    cache, synthetic, hub, monkeypatch
):
    publish(hub, synthetic, SYNTHETIC.read_bytes())

    def cut_short(bucket_id, files, *, raise_on_missing_files=False, token=None):
        files[0][1].write_bytes(SYNTHETIC.read_bytes()[:500])
        raise httpx2.ReadError("connection reset")

    monkeypatch.setattr(datasets, "download_bucket_files", cut_short)

    with pytest.raises(FileNotFoundError, match="connection reset") as raised:
        datasets.string_network_file("human")

    assert isinstance(raised.value.__cause__, httpx2.ReadError)
    assert not cache.exists()


def test_a_failed_download_removes_only_what_it_created(
    cache, synthetic, hub, monkeypatch
):
    publish(hub, synthetic, SYNTHETIC.read_bytes())
    other = cache / "cytearc" / "networks" / "other.txt"

    def another_writer_then_fail(
        bucket_id, files, *, raise_on_missing_files=False, token=None
    ):
        # Another process writes into a directory this download created.
        other.write_text("kept")
        raise httpx2.ReadError("connection reset")

    monkeypatch.setattr(datasets, "download_bucket_files", another_writer_then_fail)

    with pytest.raises(FileNotFoundError, match="connection reset"):
        datasets.string_network_file("human")

    assert files_under(cache) == ["cytearc/networks/other.txt"]
    assert not (cache / "cytearc" / "networks" / "string_v12.5").exists()


def _connection_refused(bucket_id, prefix=None):
    raise httpx2.ConnectError("[Errno 111] Connection refused")


def _offline_mode(bucket_id, prefix=None):
    raise OfflineModeIsEnabled("Cannot reach the Hub in offline mode")


@pytest.mark.parametrize(
    ("setup", "cause", "reason"),
    [
        pytest.param(
            lambda hub: hub.hooks.setdefault("list_bucket_tree", []).append(
                _connection_refused
            ),
            httpx2.ConnectError,
            "Connection refused",
            id="unreachable",
        ),
        pytest.param(
            lambda hub: hub.hooks.setdefault("list_bucket_tree", []).append(
                _offline_mode
            ),
            OfflineModeIsEnabled,
            "offline mode",
            id="offline-mode",
        ),
        pytest.param(
            lambda hub: hub.put("cytearc_docs/README.md", "docs", "Nygen/cytebase"),
            KeyError,
            "'cytearc_networks' is not a CyteBase repository",
            id="not-published",
        ),
        pytest.param(
            lambda hub: hub.put(
                "cytearc_networks/string_v12.5/mouse_10090.npz", b"m", "Nygen/cytebase"
            ),
            EntryNotFoundError,
            "No bucket entries",
            id="file-missing",
        ),
    ],
)
def test_a_network_that_cannot_be_downloaded_names_the_way_to_supply_it(
    cache, synthetic, hub, setup, cause, reason
):
    setup(hub)

    with pytest.raises(FileNotFoundError) as raised:
        datasets.string_network_file("human")

    message = str(raised.value)
    assert message.startswith(
        f"The STRING v12.5 human network (human_9606.npz, {synthetic.size:,} bytes, "
        f"SHA-256 {synthetic.sha256}) is not cached at {cached_path(cache)} and could "
        f"not be downloaded from {synthetic.uri}: "
    )
    assert reason in message
    assert COMMAND in message
    assert NETWORK_ARGUMENT in message
    assert isinstance(raised.value.__cause__, cause)
    assert not cache.exists()


def test_a_bucket_path_that_holds_several_files_is_refused(cache, synthetic, hub):
    for part in ("a", "b"):
        hub.put(
            f"cytearc_networks/string_v12.5/human_9606.npz/{part}",
            part,
            "Nygen/cytebase",
        )

    with pytest.raises(FileNotFoundError, match="names 2 files rather than one"):
        datasets.string_network_file("human")

    assert not cache.exists()


def test_the_way_to_supply_a_network_offline_works(cache, synthetic, hub, tmp_path):
    publish(hub, synthetic, SYNTHETIC.read_bytes())
    directory = tmp_path / "copies"

    # The command and argument that the offline error gives, as written there.
    datasets.connect("cytearc_networks").download(
        "string_v12.5/human_9606.npz", destination=directory
    )
    copy = directory / "string_v12.5" / "human_9606.npz"

    assert datasets.string_network_file("human", network=copy) == copy
    assert not cache.exists()


@pytest.fixture
def real_client(monkeypatch):
    """Point ``cytearc.datasets`` at a Hugging Face client of the test's own, for an endpoint it names.

    The functions that ``cytearc.datasets`` imports are bound to the client that
    ``huggingface_hub`` shares across the process. A client of the test's own
    keeps the endpoint that the test sets out of that shared state.
    """

    def connect_to(endpoint: str) -> None:
        client = hf_api.HfApi(endpoint=endpoint)
        monkeypatch.setattr(datasets, "list_bucket_tree", client.list_bucket_tree)
        monkeypatch.setattr(
            datasets, "download_bucket_files", client.download_bucket_files
        )

    return connect_to


def test_a_refused_connection_on_the_real_client_names_the_way_to_supply_it(
    cache, synthetic, real_client
):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    real_client(f"http://127.0.0.1:{port}")

    with pytest.raises(FileNotFoundError) as raised:
        datasets.string_network_file("human")

    assert isinstance(raised.value.__cause__, httpx2.ConnectError)
    assert COMMAND in str(raised.value)
    assert not cache.exists()


def test_a_status_from_the_loopback_hub_names_the_way_to_supply_it(
    cache, synthetic, loopback_hub, monkeypatch, real_client
):
    real_client(loopback_hub.endpoint)
    monkeypatch.setattr(datasets, "_BUCKET_ID", LOOPBACK_BUCKET)
    published = loopback_hub.root / "cytearc_networks" / synthetic.remote_path
    published.parent.mkdir(parents=True)
    shutil.copyfile(SYNTHETIC, published)
    loopback_hub.plan["tree"] = [403]

    with pytest.raises(FileNotFoundError) as raised:
        datasets.string_network_file("human")

    assert isinstance(raised.value.__cause__, HfHubHTTPError)
    assert raised.value.__cause__.response.status_code == 403
    assert f"hf://buckets/{LOOPBACK_BUCKET}/cytearc_networks/" in str(raised.value)
    assert NETWORK_ARGUMENT in str(raised.value)
    assert [kind for kind, _ in loopback_hub.requests] == ["tree"]
    assert not cache.exists()


def test_a_cached_network_loads_with_the_hub_blocked(cache, synthetic):
    # No fake hub: hub_offline fails any request loudly.
    cached = cached_path(cache)
    cached.parent.mkdir(parents=True)
    shutil.copyfile(SYNTHETIC, cached)

    assert datasets.string_network_file("human") == cached
    assert cached.read_bytes() == SYNTHETIC.read_bytes()


def test_a_cached_network_that_changed_raises_and_stays(cache, synthetic, hub):
    cached = cached_path(cache)
    cached.parent.mkdir(parents=True)
    changed = SYNTHETIC.read_bytes()[:-1]
    cached.write_bytes(changed)

    with pytest.raises(ValueError) as raised:
        datasets.string_network_file("human")

    message = str(raised.value)
    assert message.startswith(f"The cached file {cached} does not match")
    assert f"SHA-256 {hashlib.sha256(changed).hexdigest()}" in message
    assert message.endswith(
        "Delete it to download the network again, or pass network= a copy that matches."
    )
    assert cached.read_bytes() == changed
    assert hub.calls == []


def test_a_local_network_is_checked_and_needs_no_hub(cache, synthetic, hub, tmp_path):
    copy = tmp_path / "copy.npz"
    shutil.copyfile(SYNTHETIC, copy)
    assert datasets.string_network_file("human", network=copy) == copy
    assert datasets.string_network_file("human", network=str(copy)) == copy

    changed = tmp_path / "changed.npz"
    changed.write_bytes(SYNTHETIC.read_bytes()[::-1])
    with pytest.raises(
        ValueError, match=r"^The file passed as network=, .*changed\.npz, "
    ):
        datasets.string_network_file("human", network=changed)

    for missing in (tmp_path / "absent.npz", tmp_path):
        with pytest.raises(FileNotFoundError) as raised:
            datasets.string_network_file("human", network=missing)
        assert "must be a copy of human_9606.npz" in str(raised.value)
        assert synthetic.sha256 in str(raised.value)
    assert hub.calls == []
    assert not cache.exists()


def test_species_and_network_must_name_a_recorded_network(cache):
    with pytest.raises(ValueError, match=r"Networks exist for: human, mouse$"):
        datasets.string_network_file("rat")
    with pytest.raises(TypeError, match="species must be text, got int"):
        datasets.string_network_file(9606)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="network must be a path, got bytes"):
        datasets.string_network_file("human", network=b"human_9606.npz")  # type: ignore[arg-type]


def test_the_record_names_each_published_network_once():
    records = datasets.STRING_NETWORKS
    assert {species: entry.taxid for species, entry in records.items()} == {
        "human": 9606,
        "mouse": 10090,
    }
    assert all(entry.species == species for species, entry in records.items())
    assert len({entry.name for entry in records.values()}) == len(records)
    for entry in records.values():
        assert entry.string_version == "12.5"
        assert entry.name == f"{entry.species}_{entry.taxid}.npz"
        assert entry.uri == (
            f"hf://buckets/Nygen/cytebase/cytearc_networks/string_v12.5/{entry.name}"
        )
        assert entry.size > 0
        assert len(entry.sha256) == 64 and set(entry.sha256) <= set("0123456789abcdef")
    with pytest.raises(TypeError):
        records["rat"] = records["human"]  # type: ignore[index]


@pytest.mark.integration
def test_the_published_networks_match_their_records(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with hub_or_skip("the STRING networks in Nygen/cytebase"):
        if "cytearc_networks" not in datasets.list_repositories():
            pytest.skip(
                f"{NETWORK_SKIP} cytearc_networks: Nygen/cytebase does not hold the "
                "STRING networks yet"
            )
        for species, entry in datasets.STRING_NETWORKS.items():
            try:
                path = datasets.string_network_file(species)
            except FileNotFoundError as error:
                # Let hub_or_skip judge why the hub could not serve the file.
                raise error.__cause__ or error
            assert path.stat().st_size == entry.size
            assert file_sha256(path) == entry.sha256
