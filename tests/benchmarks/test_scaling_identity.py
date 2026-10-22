"""Benchmarks of the identity checks whose cost grows with the cell count.

Preparation hashes the text of every cell ID into the dataset fingerprint, a
filter by sample hashes every cell's sample label into its provenance, and
every public call that takes a cell selection checks the selection against the
ordered cell IDs. Each benchmark calls the function its path calls on 10x
barcodes stored as CyteArc stores them, and checks the result against an
independent oracle, so it doubles as a test at the smoke size.
"""

import hashlib
from functools import lru_cache

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

LADDER = Ladder(sizes=(125_000, 250_000, 500_000, 1_000_000), smoke=2_000)


@lru_cache(maxsize=4)
def _barcodes(n_cells: int) -> np.ndarray:
    """Return ``n_cells`` distinct 10x cell IDs such as ``ACGT...-1``, shuffled."""
    rng = np.random.default_rng(inputs.SEED)
    codes = rng.permutation(n_cells).astype(np.int64)
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[
        (codes[:, None] >> (2 * np.arange(15, -1, -1))) & 3
    ]
    return np.char.add(letters.view("S16").reshape(-1).astype(str), "-1")


@lru_cache(maxsize=4)
def _cell_table(n_cells: int) -> zarr.Group:
    """Return a root whose cell table holds the barcodes as a metadata column."""
    from cytearc.storage.arrays import create_metadata_column
    from cytearc.storage.layout import PROFILE_METADATA_CHUNK

    root = zarr.open_group(store=MemoryStore(), mode="w")
    create_metadata_column(
        root.create_group("cellData"),
        "ids",
        data=_barcodes(n_cells),
        chunkSize=PROFILE_METADATA_CHUNK,
    )
    return root


def _records_digest(name: str, shape: list[int], values: list[str]) -> str:
    """Hash one shape record and one record per value, as preparation always has.

    Each record is the length of its name, the name, the length of its
    payload, and the payload, with lengths as eight big-endian bytes.
    """
    from cytearc.storage.artifacts import canonical_bytes

    digest = hashlib.blake2b(digest_size=32, person=b"cytearc-values")

    def record(label: str, payload: bytes) -> None:
        for part in (label.encode("utf-8"), payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)

    record(name, canonical_bytes({"shape": shape}))
    for value in values:
        record("value", value.encode("utf-8"))
    return digest.hexdigest()


def test_dataset_fingerprint_cell_ids(bench) -> None:
    from cytearc.storage.artifacts import ValueFingerprintBuilder
    from cytearc.storage.identity import _hash_column

    def make(n_cells: int):
        ids = _cell_table(n_cells)["cellData/ids"]

        def run() -> str:
            builder = ValueFingerprintBuilder()
            _hash_column(builder, "cell_ids", ids)
            return builder.hexdigest()

        return run

    def check(n_cells: int, digest) -> None:
        expected = _records_digest("cell_ids", [n_cells], _barcodes(n_cells).tolist())
        assert digest == expected

    # Preparation hashes the cell IDs once into each assay's dataset fingerprint.
    bench("identity.dataset_fingerprint_cell_ids", make, LADDER, check=check)


def test_cell_selection_check(bench) -> None:
    from cytearc.storage.selections import (
        resolve_generated_selection_artifact,
        validate_cell_selection,
    )
    from cytearc.storage.validation_scope import validation_scope

    @lru_cache(maxsize=4)
    def selection(n_cells: int):
        root = _cell_table(n_cells)
        values = np.arange(n_cells) % 3 != 0
        ref, _stored = resolve_generated_selection_artifact(
            root,
            scope="datastore",
            kind="cell_selection",
            values=values,
            row_ids=root["cellData/ids"],
            operation="benchmark_selection",
            parameters={},
            inputs={},
            source_column="I",
        )
        return root, ref

    def make(n_cells: int):
        root, ref = selection(n_cells)

        def run():
            # Each public call opens its own scope and checks its selection.
            with validation_scope():
                return validate_cell_selection(root, ref)

        return run

    def check(n_cells: int, validated) -> None:
        assert validated.selected_count == n_cells - (n_cells + 2) // 3

    bench("identity.cell_selection_check", make, LADDER, check=check)


def test_sample_label_fingerprint(bench) -> None:
    from cytearc.storage.artifacts import fingerprint_strings

    @lru_cache(maxsize=4)
    def labels(n_cells: int) -> np.ndarray:
        rng = np.random.default_rng(inputs.SEED)
        names = np.asarray(["donor_a", "donor_bb", "donor_ccc", "pool"])
        # A filter by sample reads typed labels and keeps them as objects.
        return names[rng.integers(0, names.size, size=n_cells)].astype(object)

    def make(n_cells: int):
        values = labels(n_cells)
        return lambda: fingerprint_strings(values)

    def check(n_cells: int, digest) -> None:
        from cytearc.storage.artifacts import canonical_bytes

        # The labels as text of the longest label's width, then their UTF-32.
        text = np.asarray([str(value) for value in labels(n_cells)], dtype="<U9")
        expected = hashlib.blake2b(digest_size=32, person=b"cytearc-values")
        metadata = canonical_bytes({"dtype": {"str": "<U9"}, "shape": [n_cells]})
        for part in (b"values:metadata", metadata):
            expected.update(len(part).to_bytes(8, "big"))
            expected.update(part)
        expected.update(text.view(np.uint8))
        assert digest == expected.hexdigest()

    bench("identity.sample_label_fingerprint", make, LADDER, check=check)
