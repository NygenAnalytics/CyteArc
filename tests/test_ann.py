import numpy as np
import pytest

from cytearc.neighbors import index
from cytearc.neighbors.index import ann_index_peak_bytes, fix_knn_query


def test_fix_knn_query_removes_leading_self_matches():
    indices = np.array([[4, 1, 2], [7, 5, 6]])
    distances = np.array([[0.0, 0.2, 0.4], [0.0, 0.3, 0.6]])

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([4, 7]),
    )

    assert missed_self_hits == 0
    np.testing.assert_array_equal(fixed_indices, [[1, 2], [5, 6]])
    # The kept distances are copied, not recomputed.
    np.testing.assert_array_equal(fixed_distances, [[0.2, 0.4], [0.3, 0.6]])


def test_fix_knn_query_handles_each_self_neighbor_case():
    indices = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 1, 4],
            [0, 2, 3, 4],
        ]
    )
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    original_indices = indices.copy()
    original_distances = distances.copy()

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([0, 1, 5]),
    )

    assert missed_self_hits == 1
    np.testing.assert_array_equal(
        fixed_indices,
        np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [0, 2, 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        fixed_distances,
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.4],
                [0.1, 0.2, 0.3],
            ]
        ),
    )
    np.testing.assert_array_equal(indices, original_indices)
    np.testing.assert_array_equal(distances, original_distances)


def test_fix_knn_query_removes_only_the_first_repeated_self_label():
    indices = np.array([[3, 8, 3, 9]])
    distances = np.array([[0.0, 0.1, 0.2, 0.3]])

    fixed_indices, fixed_distances, missed_self_hits = fix_knn_query(
        indices,
        distances,
        ref_idx=np.array([3]),
    )

    assert missed_self_hits == 0
    np.testing.assert_array_equal(fixed_indices, [[8, 3, 9]])
    np.testing.assert_array_equal(fixed_distances, [[0.1, 0.2, 0.3]])


@pytest.mark.parametrize(
    ("indices", "distances", "ref_idx"),
    [
        (np.arange(3), np.zeros(3), np.arange(3)),
        (np.zeros((2, 3), dtype=int), np.zeros((2, 2)), np.arange(2)),
        (np.zeros((2, 3), dtype=int), np.zeros((2, 3)), np.arange(3)),
        (np.zeros((2, 1), dtype=int), np.zeros((2, 1)), np.arange(2)),
    ],
    ids=["one_dimensional", "distance_shape", "reference_length", "single_column"],
)
def test_fix_knn_query_rejects_incompatible_shapes(indices, distances, ref_idx):
    with pytest.raises(ValueError, match="incompatible shapes"):
        fix_knn_query(indices, distances, ref_idx)


@pytest.mark.parametrize(
    ("platform_name", "machine", "runtime_bytes"),
    [
        ("win32", "AMD64", 80 + 96),
        ("darwin", "arm64", 64 + 56),
        ("linux", "aarch64", 48 + 56),
        ("linux", "x86_64", 40 + 56),
        ("freebsd14", "amd64", 40 + 56),
    ],
)
def test_index_estimate_counts_each_platforms_mutex_and_label_entry(
    monkeypatch, platform_name, machine, runtime_bytes
):
    def estimate(platform_name: str, machine: str) -> int:
        monkeypatch.setattr(index.sys, "platform", platform_name)
        monkeypatch.setattr(index.platform, "machine", lambda: machine)
        return ann_index_peak_bytes(1000, 3, 2)

    x86_64_linux = estimate("linux", "x86_64")

    # Each cell holds one std::mutex and one entry of the label map: on
    # x86-64 Linux, glibc's 40-byte mutex and libstdc++'s 56-byte entry.
    assert estimate(platform_name, machine) - x86_64_linux == 1000 * (
        runtime_bytes - (40 + 56)
    )
