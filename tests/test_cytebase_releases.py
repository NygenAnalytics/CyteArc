"""Offline tests for the public CyteBase release list."""

import re

from cytearc import cytebase
from cytearc.cytebase import _releases, _storage


def _order(version: str) -> tuple[int, int]:
    month, _, patch = version.partition(".")
    return int(month), int(patch or 0)


def test_releases_list_valid_year_month_buckets_newest_first():
    listed = cytebase.releases()
    versions = [release.version for release in listed]

    assert versions
    assert len(set(versions)) == len(versions)
    assert versions == sorted(versions, key=_order, reverse=True)
    for release in listed:
        assert release.bucket == _storage.versioned_bucket(release.version)
        assert _storage.Bucket(release.bucket, token=False).bucket_id == release.bucket
        assert re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", release.released)
        assert release.released[2:4] + release.released[5:7] == release.version[:4]
        assert release.notes.strip()
        assert all(
            isinstance(entry, str) and entry.strip() for entry in release.changelog
        )


def test_exactly_the_latest_release_is_marked():
    listed = cytebase.releases()
    assert [release.version for release in listed if release.latest] == [
        _releases.LATEST
    ]


def test_first_release_is_october_2026():
    [first] = [release for release in cytebase.releases() if release.version == "2610"]
    assert (first.bucket, first.released) == ("Nygen/cytebase-2610", "2026-10")
