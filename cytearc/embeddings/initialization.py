from dataclasses import dataclass

import numpy as np

from ..utils.arguments import RANDOM_STATE_MAX, float_argument, integer_argument
from ..utils.arrays import rescale_array


@dataclass(frozen=True, slots=True)
class InitializationSettings:
    """The settings of an embedding initialization, as canonical Python values.

    Each attribute is the ``embeddings.initialization`` keyword of the same
    name.
    """

    n_centroids: int
    rand_state: int
    batch_size: int | None
    kmeans_sampling: float
    kmeans_batch_size: int


def _sampling_fraction(value: object, name: str) -> float:
    resolved = float_argument(value, name)
    if not 0 < resolved <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return resolved


def initialization_settings(
    *,
    n_centroids: object,
    rand_state: object,
    batch_size: object,
    kmeans_sampling: object,
    kmeans_batch_size: object,
) -> InitializationSettings:
    """Check the settings of an embedding initialization and return them as Python values.

    ``embeddings.initialization`` and the embedding initialization stage of
    a pipeline recipe check their settings with this function, so a recipe
    refuses a setting before its run starts, with the message that
    ``embeddings.initialization`` gives. K-means needs at least two
    centroids.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a setting is out of range.
    """
    return InitializationSettings(
        batch_size=(
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        ),
        n_centroids=integer_argument(n_centroids, "n_centroids", minimum=2),
        # The seed seeds scikit-learn's k-means.
        rand_state=integer_argument(
            rand_state, "rand_state", minimum=0, maximum=RANDOM_STATE_MAX
        ),
        kmeans_sampling=_sampling_fraction(kmeans_sampling, "kmeans_sampling"),
        kmeans_batch_size=integer_argument(
            kmeans_batch_size, "kmeans_batch_size", minimum=1
        ),
    )


def initial_embedding(
    cluster_centers: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Create cell coordinates from PCA-projected seed partitions."""
    from sklearn.decomposition import PCA

    centers = np.asarray(cluster_centers)
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1:
        raise ValueError("Cluster labels must be one-dimensional")
    if np.issubdtype(labels.dtype, np.floating):
        if not np.all(np.isfinite(labels)) or not np.array_equal(
            labels,
            np.floor(labels),
        ):
            raise ValueError("Floating cluster labels must be finite integers")
    elif not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Cluster labels must contain numeric integers")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError("Cluster labels are outside the center range")
    if n_components < 1 or n_components > min(centers.shape):
        raise ValueError(
            "Embedding components cannot exceed the center count or dimensions"
        )
    index_labels = labels.astype(np.intp, copy=False)
    # Many high-dimensional centers select the randomized solver; seed it so
    # the same centers always give the same coordinates.
    principal_components = PCA(
        n_components=n_components,
        random_state=0,
    ).fit_transform(centers)
    for component in range(n_components):
        principal_components[:, component] = rescale_array(
            principal_components[:, component]
        )
    return np.asarray(principal_components[index_labels], dtype=np.float32, order="C")
