#!/usr/bin/env python3
"""Publish the rebuilt documentation stores to the Cytebase bucket.

Each rebuilt archive and its manifest replace `<dataset>/data.zarr.tar.gz`
and `<dataset>/manifest.json` in place.

Run without `--apply` to print the plan. Uploading needs a write token for the
bucket.

Example:
    uv run python scripts/publish_docs_datasets.py
    uv run python scripts/publish_docs_datasets.py --apply tenx_5K_pbmc_rnaseq
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import BucketFile, batch_bucket_files, list_bucket_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "docs/source/developers/dataset_manifests"
BUCKET_ID = "Nygen/cytebase"
REPOSITORY = "cytearc_docs"
STORE_NAME = "data.zarr"
ARCHIVE_NAME = f"{STORE_NAME}.tar.gz"

# Datasets whose unpacked stores support direct remote access.
UNPACKED_DATASETS = frozenset({"tenx_5K_pbmc_rnaseq", "tabula_sapiens"})


@dataclass(slots=True)
class Plan:
    """Every bucket mutation needed to publish one dataset."""

    dataset: str
    upload: list[tuple[Path, str]] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.upload or self.remove)


def _published_files(prefix: str) -> dict[str, BucketFile]:
    files = {}
    for item in list_bucket_tree(
        BUCKET_ID,
        prefix=f"{REPOSITORY}/{prefix}",
        recursive=True,
        token=False,
    ):
        if isinstance(item, BucketFile):
            files[item.path] = item
    return files


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan_dataset(dataset: str, build_root: Path) -> Plan:
    manifest_path = MANIFEST_DIR / f"{dataset}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest for {dataset}, rebuild it first")
    manifest = json.loads(manifest_path.read_text())

    archive = build_root / dataset / ARCHIVE_NAME
    if not archive.is_file():
        raise FileNotFoundError(f"No archive at {archive}, rebuild {dataset} first")
    digest = _file_digest(archive)
    if digest != manifest["archiveSha256"]:
        raise RuntimeError(
            f"{archive} does not match its manifest checksum. Rebuild {dataset} "
            "so the manifest and the archive describe the same store."
        )

    plan = Plan(dataset=dataset)
    plan.upload.append((archive, f"{REPOSITORY}/{dataset}/{ARCHIVE_NAME}"))
    plan.upload.append((manifest_path, f"{REPOSITORY}/{dataset}/manifest.json"))

    if dataset in UNPACKED_DATASETS:
        published = _published_files(dataset)
        store = build_root / dataset / STORE_NAME
        if not store.is_dir():
            raise FileNotFoundError(f"No unpacked store at {store}")
        wanted = set()
        for item in sorted(store.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(store).as_posix()
            remote = f"{REPOSITORY}/{dataset}/{STORE_NAME}/{relative}"
            wanted.add(remote)
            plan.upload.append((item, remote))
        stale = [
            path
            for path in published
            if path.startswith(f"{REPOSITORY}/{dataset}/{STORE_NAME}/")
            and path not in wanted
        ]
        plan.remove.extend(sorted(stale))

    return plan


def _describe(plan: Plan) -> None:
    print(plan.dataset)
    archives = [item for item in plan.upload if item[1].endswith(ARCHIVE_NAME)]
    members = [item for item in plan.upload if not item[1].endswith(ARCHIVE_NAME)]
    for source, destination in archives:
        print(f"  upload   {destination} ({source.stat().st_size / 1e6:.1f} MB)")
    if members:
        total = sum(source.stat().st_size for source, _ in members)
        noun = "file" if len(members) == 1 else "files"
        print(f"  upload   {len(members)} additional {noun} ({total / 1e6:.1f} MB)")
    if plan.remove:
        print(f"  delete   {len(plan.remove)} stale unpacked file(s)")


def _apply(plan: Plan) -> None:
    if plan.upload:
        batch_bucket_files(
            BUCKET_ID,
            add=[(str(source), destination) for source, destination in plan.upload],
        )
        print(f"  uploaded {len(plan.upload)} file(s)")
    if plan.remove:
        batch_bucket_files(BUCKET_ID, delete=plan.remove)
        print(f"  deleted {len(plan.remove)} stale file(s)")


def main(argv: list[str] | None = None) -> int:
    available = sorted(
        path.name.removesuffix("_manifest.json")
        for path in MANIFEST_DIR.glob("*_manifest.json")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*available, []],
        help="Datasets to publish, defaults to every rebuilt dataset",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the plan instead of printing it",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_ROOT / "build/cytebase",
        help="Directory holding the rebuilt stores and archives",
    )
    args = parser.parse_args(argv)

    selected = list(args.datasets) or available
    if not selected:
        parser.error("no manifests found, rebuild a dataset first")

    plans = [_plan_dataset(dataset, args.build_root) for dataset in selected]
    for plan in plans:
        _describe(plan)
        if args.apply and not plan.empty:
            _apply(plan)
        print()

    if not args.apply:
        print("Dry run. Pass --apply to publish, which needs a bucket write token.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
