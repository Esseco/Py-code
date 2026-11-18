"""Compare the three weighted samplers on synthetic descriptors (示例数据).

Replace the generated feature vectors with MLIP descriptors in real use; the
weights here are arbitrary and would come from QBC, energy or stress outside
the sampler. Importing this module does no work.
"""
import numpy as np

from Process_MLIP import (
    WeightedFarthestFirstSampler,
    WeightedKMeansPlusPlusSampler,
    WeightedNMSSampler,
)

N_CANDIDATES = 500
N_LABELED = 30
DIM = 32
N_SELECT = 20
RADIUS = 0.5
SEED = 0


def make_data():
    """Return synthetic candidates, labeled configurations and weights."""
    rng = np.random.default_rng(SEED)
    candidates = {
        f"struct_{index:03d}": {"features": rng.normal(size=DIM)}
        for index in range(N_CANDIDATES)
    }
    labeled = {
        f"dft_{index:03d}": {"features": rng.normal(size=DIM)}
        for index in range(N_LABELED)
    }
    weights = {key: float(rng.uniform(0.2, 1.0)) for key in candidates}
    return candidates, labeled, weights


def main():
    candidates, labeled, weights = make_data()
    samplers = {
        "kmeans++": WeightedKMeansPlusPlusSampler(random_state=SEED),
        "farthest-first": WeightedFarthestFirstSampler(),
        "nms": WeightedNMSSampler(radius=RADIUS),
    }
    for name, sampler in samplers.items():
        selected = sampler.select(
            candidates=candidates, labeled=labeled, weights=weights, n_select=N_SELECT
        )
        print(f"{name:14s} {len(selected):3d} selected, first: {selected[:3]}")


if __name__ == "__main__":
    main()
