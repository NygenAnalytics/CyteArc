#!/usr/bin/env python3
"""Write LocalMAP reference outputs of pacmap 0.9.1 for CyteArc's LocalMAP tests.

pacmap (Apache-2.0, https://github.com/YingfanWang/PaCMAP) holds LocalMAP's
reference implementation. This script draws 400 cells in four Gaussian
clusters of 12 dimensions, finds their 20 nearest neighbors exactly, and
records what pacmap's own functions compute from them:

- the neighbor pairs that PaCMAP's scaled-distance rule keeps,
- the gradients of the three phases and of LocalMAP's third phase,
- the layouts after 200 and 450 iterations from fixed pairs and a fixed
  start, which the optimizer reaches without random draws.

The pairs are drawn by pacmap's seeded samplers. A full LocalMAP run is not
recorded, because pacmap's threads share one random state while they draw
further pairs again, so its layout differs between runs.

Example:
    uv run --no-project --with pacmap==0.9.1 --with scikit-learn \\
        python scripts/export_localmap_golden.py --out tests/data/localmap
"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

SEED = 7
CLUSTERS, CELLS_PER_CLUSTER, DIMS = 4, 100, 12
N_NEIGHBORS, N_MID_NEAR, N_FURTHER, KNN = 10, 5, 20, 20
ITERATIONS = (100, 100, 250)
MAX_DISTANCE = 10.0


def main() -> None:
    from pacmap import pacmap as reference
    from sklearn.neighbors import NearestNeighbors

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    rng = np.random.default_rng(SEED)
    centers = rng.normal(scale=6.0, size=(CLUSTERS, DIMS))
    spreads = rng.uniform(0.5, 2.0, size=CLUSTERS)
    cells = np.concatenate(
        [
            center + rng.normal(scale=spread, size=(CELLS_PER_CLUSTER, DIMS))
            for center, spread in zip(centers, spreads, strict=True)
        ]
    ).astype(np.float32)
    n_cells = cells.shape[0]
    distances, indices = (
        NearestNeighbors(n_neighbors=KNN + 1).fit(cells).kneighbors(cells)
    )
    if not np.array_equal(indices[:, 0], np.arange(n_cells)):
        raise RuntimeError("every cell must be its own nearest neighbor")
    indices = np.ascontiguousarray(indices[:, 1:], dtype=np.int32)
    distances = np.ascontiguousarray(distances[:, 1:], dtype=np.float32)

    scale = np.maximum(np.mean(distances[:, 3:6], axis=1), 1e-10).astype(np.float32)
    scaled = reference.scale_dist(distances, scale, indices)
    neighbors = reference.sample_neighbors_pair(cells, scaled, indices, N_NEIGHBORS)
    mid_near = reference.sample_MN_pair_deterministic(cells, N_MID_NEAR, SEED, 0)
    further = reference.sample_FP_pair_deterministic(
        cells, neighbors, N_NEIGHBORS, N_FURTHER, SEED
    )
    start = (rng.normal(size=(n_cells, 2)) * 1e-4).astype(np.float32)
    state = (rng.normal(size=(n_cells, 2)) * 3.0).astype(np.float32)

    outputs: dict[str, np.ndarray] = {}
    for iteration in (0, 50, 150, 200):
        w_mid_near, w_neighbors, w_further = reference.find_weight(
            1000.0, iteration, num_iters=ITERATIONS
        )
        outputs[f"gradient_{iteration}"] = reference.pacmap_grad(
            state, neighbors, mid_near, further, w_neighbors, w_mid_near, w_further
        )[:-1]
    w_mid_near, w_neighbors, w_further = reference.find_weight(
        1000.0, 300, num_iters=ITERATIONS
    )
    outputs["gradient_local"] = reference.pacmap_grad_nearby_recip_sqrt(
        state,
        neighbors,
        mid_near,
        further,
        w_neighbors,
        w_mid_near,
        w_further,
        MAX_DISTANCE / 2,
    )[:-1]

    def optimize(iterations: int) -> np.ndarray:
        # pacmap.localmap's loop, without its draws of new further pairs.
        layout = start.copy()
        first_moment = np.zeros_like(layout)
        second_moment = np.zeros_like(layout)
        for iteration in range(iterations):
            w_mn, w_nb, w_fp = reference.find_weight(
                1000.0, iteration, num_iters=ITERATIONS
            )
            if iteration > ITERATIONS[0] + ITERATIONS[1]:
                gradient = reference.pacmap_grad_nearby_recip_sqrt(
                    layout,
                    neighbors,
                    mid_near,
                    further,
                    w_nb,
                    w_mn,
                    w_fp,
                    MAX_DISTANCE / 2,
                )
            else:
                gradient = reference.pacmap_grad(
                    layout, neighbors, mid_near, further, w_nb, w_mn, w_fp
                )
            reference.update_embedding_adam(
                layout,
                gradient,
                first_moment,
                second_moment,
                0.9,
                0.999,
                1.0,
                iteration,
            )
        return layout

    outputs["layout_200"] = optimize(200)
    outputs["layout_450"] = optimize(sum(ITERATIONS))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "pacmap_0_9_1_reference.npz",
        cells=cells,
        knn_indices=indices,
        knn_distances=distances,
        neighbor_pairs=neighbors[:, 1].reshape(n_cells, N_NEIGHBORS),
        mid_near_pairs=mid_near[:, 1].reshape(n_cells, N_MID_NEAR),
        further_pairs=further[:, 1].reshape(n_cells, N_FURTHER),
        start=start,
        state=state,
        **outputs,
    )
    provenance = {
        "pacmap": version("pacmap"),
        "numpy": version("numpy"),
        "numba": version("numba"),
        "scikit-learn": version("scikit-learn"),
        "python": platform.python_version(),
        "seed": SEED,
        "iterations": ITERATIONS,
        "max_distance": MAX_DISTANCE,
    }
    (out / "pacmap_0_9_1_reference.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
