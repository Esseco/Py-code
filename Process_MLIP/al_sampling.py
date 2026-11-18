"""Weighted active-learning selectors over precomputed descriptors.

All three samplers share one interface and one distance layer::

    selected = sampler.select(
        candidates={"struct_001": {"features": np.array([...])}},
        labeled={"dft_001": {"features": np.array([...])}},
        weights={"struct_001": 0.85},
        n_select=100,
    )

Features are used exactly as given: nothing is normalized or scaled here, and
``weights`` are final (QBC, energy, stress or any mixture is fused by the
caller). Distances are Euclidean in feature space; ``d_i`` is the nearest
distance from candidate ``i`` to any labeled configuration plus the candidates
already selected in this round.

* :class:`WeightedKMeansPlusPlusSampler` samples with ``P(i) ∝ w_i d_i²``.
* :class:`WeightedFarthestFirstSampler` picks ``argmax_i w_i d_i²``.
* :class:`WeightedNMSSampler` scans by descending weight and keeps candidates
  farther than ``radius`` from labeled and already selected configurations.

Distance work stays linear in the number of candidates per step: one chunked
nearest-distance pass against ``labeled`` and one vectorized update per
selected configuration. No candidate-by-candidate distance matrix is built.
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    "WeightedKMeansPlusPlusSampler",
    "WeightedFarthestFirstSampler",
    "WeightedNMSSampler",
]


def _feature_vector(item: Any, label: str) -> np.ndarray:
    """Return one finite 1-D feature vector from a dict entry or array."""
    value = item.get("features") if isinstance(item, Mapping) else item
    if value is None:
        raise ValueError(f"{label} has no 'features' entry")
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{label} features must be 1-D, got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{label} features must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} features contain NaN or inf")
    return vector


def _stack(items: Mapping[Hashable, Any], label: str) -> tuple[list[Hashable], np.ndarray]:
    """Validate a candidate/labeled mapping and stack its feature vectors."""
    if not isinstance(items, Mapping):
        raise TypeError(f"{label} must be a mapping of ID -> features")
    ids = list(items)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate IDs")
    if not ids:
        return [], np.empty((0, 0))
    vectors = [_feature_vector(items[key], f"{label}[{key!r}]") for key in ids]
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError(
            f"{label} features have inconsistent dimensions: {sorted(dimensions)}"
        )
    return ids, np.vstack(vectors)


def _prepare(
    candidates: Mapping[Hashable, Any],
    labeled: Mapping[Hashable, Any],
    weights: Mapping[Hashable, float],
    n_select: int,
) -> tuple[list[Hashable], np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate inputs and return IDs, feature matrix, weights, labeled features."""
    if isinstance(n_select, bool) or not isinstance(n_select, (int, np.integer)):
        raise TypeError("n_select must be an integer")
    if n_select < 1:
        raise ValueError("n_select must be >= 1")

    candidate_ids, points = _stack(candidates, "candidates")
    if not candidate_ids:
        raise ValueError("candidates must not be empty")
    _, reference = _stack(labeled, "labeled")
    if reference.size and reference.shape[1] != points.shape[1]:
        raise ValueError(
            "labeled features have dimension "
            f"{reference.shape[1]}, candidates have {points.shape[1]}"
        )
    if not reference.size:
        reference = np.empty((0, points.shape[1]))

    candidate_set = set(candidate_ids)
    missing = [key for key in candidate_ids if key not in weights]
    if missing:
        raise ValueError(f"weights are missing candidates, e.g. {missing[:5]}")
    extra = [key for key in weights if key not in candidate_set]
    if extra:
        raise ValueError(f"weights contain unknown candidates, e.g. {extra[:5]}")

    weight_array = np.array([weights[key] for key in candidate_ids], dtype=float)
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    if np.any(weight_array < 0):
        raise ValueError("weights must be non-negative")

    return candidate_ids, points, weight_array, reference, int(n_select)


def _nearest_distances(
    points: np.ndarray, reference: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Nearest Euclidean distance from each point to any reference point."""
    if reference.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    nearest = np.empty(points.shape[0])
    for start in range(0, points.shape[0], chunk_size):
        block = points[start:start + chunk_size]
        nearest[start:start + block.shape[0]] = cdist(block, reference).min(axis=1)
    return nearest


def _update_nearest(
    points: np.ndarray, nearest: np.ndarray, index: int
) -> np.ndarray:
    """Fold one newly selected point into the running nearest distances."""
    delta = points - points[index]
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    np.minimum(nearest, distance, out=nearest)
    return nearest


def _weighted_scores(weights: np.ndarray, nearest: np.ndarray) -> np.ndarray:
    """``w_i d_i²`` with infinite distances treated as no contribution."""
    distances = np.where(np.isfinite(nearest), nearest, 0.0)
    return weights * distances * distances


def _draw_index(
    rng: np.random.Generator, probabilities: np.ndarray, available: np.ndarray
) -> int | None:
    """Draw one available index, or ``None`` when every probability is zero."""
    values = np.where(available, probabilities, 0.0)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    return int(rng.choice(values.size, p=values / total))


def _fallback_index(weights: np.ndarray, available: np.ndarray) -> int:
    """Deterministic fallback: highest weight among available, first on ties."""
    masked = np.where(available, weights, -np.inf)
    if np.isfinite(masked).any():
        return int(np.argmax(masked))
    return int(np.flatnonzero(available)[0])


class _BaseWeightedSampler:
    """Shared validation, output mapping and distance bookkeeping."""

    def __init__(self, *, chunk_size=512):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be an integer")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = int(chunk_size)

    def select(
        self,
        candidates: Mapping[Hashable, Any],
        labeled: Mapping[Hashable, Any],
        weights: Mapping[Hashable, float],
        n_select: int = 100,
    ) -> list[Hashable]:
        """Select candidate IDs; returns at most ``min(n_select, len(candidates))``.

        Args:
            candidates: ID -> ``{"features": 1-D array-like}``; plain arrays are
                accepted too. Every ID must be unique.
            labeled: Already computed configurations in the same feature space;
                an empty mapping means no DFT data yet.
            weights: Final non-negative weight per candidate ID; must cover the
                candidates exactly. No normalization is applied.
            n_select: Maximum number of IDs to return.

        Returns:
            Selected IDs in selection order. Duplicate features are still valid
            configurations; they simply never help by distance.
        """
        ids, points, weight_array, reference, count = _prepare(
            candidates, labeled, weights, n_select
        )
        count = min(count, len(ids))
        indices = self._select_indices(points, weight_array, reference, count)
        return [ids[index] for index in indices]

    def _select_indices(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        reference: np.ndarray,
        n_select: int,
    ) -> list[int]:
        raise NotImplementedError


class WeightedKMeansPlusPlusSampler(_BaseWeightedSampler):
    """Weighted k-means++: sample the next point with ``P(i) ∝ w_i d_i²``.

    The first point follows the same rule against ``labeled``; when ``labeled``
    is empty it is drawn with probability ``∝ w_i``. ``random_state`` reseeds
    every ``select`` call, so identical inputs give identical selections.
    Zero probabilities (duplicate features, zero weights) fall back to the
    highest-weight available candidate, first position on ties.
    """

    def __init__(self, random_state=None, *, chunk_size=512):
        super().__init__(chunk_size=chunk_size)
        self.random_state = random_state

    def _select_indices(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        reference: np.ndarray,
        n_select: int,
    ) -> list[int]:
        rng = np.random.default_rng(self.random_state)
        nearest = _nearest_distances(points, reference, self.chunk_size)
        available = np.ones(points.shape[0], dtype=bool)
        selected: list[int] = []
        for step in range(n_select):
            if step == 0 and reference.shape[0] == 0:
                probabilities = weights
            else:
                probabilities = _weighted_scores(weights, nearest)
            index = _draw_index(rng, probabilities, available)
            if index is None:
                index = _fallback_index(weights, available)
            selected.append(index)
            available[index] = False
            _update_nearest(points, nearest, index)
        return selected


class WeightedFarthestFirstSampler(_BaseWeightedSampler):
    """Weighted farthest-first: pick ``argmax_i w_i d_i²`` every step.

    With empty ``labeled`` the first pick is the highest-weight candidate. Ties
    keep the earlier candidate position, so the sampler is deterministic.
    """

    def _select_indices(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        reference: np.ndarray,
        n_select: int,
    ) -> list[int]:
        nearest = _nearest_distances(points, reference, self.chunk_size)
        available = np.ones(points.shape[0], dtype=bool)
        selected: list[int] = []
        for step in range(n_select):
            if step == 0 and reference.shape[0] == 0:
                scores = weights
            else:
                scores = _weighted_scores(weights, nearest)
            index = int(np.argmax(np.where(available, scores, -np.inf)))
            selected.append(index)
            available[index] = False
            _update_nearest(points, nearest, index)
        return selected


class WeightedNMSSampler(_BaseWeightedSampler):
    """Weighted non-maximum suppression with a fixed feature-space radius.

    Candidates are visited by descending weight (stable on ties) and kept only
    when farther than ``radius`` from ``labeled`` and every accepted candidate.
    It may return fewer than ``n_select`` IDs and never pads the result.
    """

    def __init__(self, radius, *, chunk_size=512):
        super().__init__(chunk_size=chunk_size)
        radius_value = float(radius)
        if not np.isfinite(radius_value) or radius_value < 0:
            raise ValueError("radius must be a finite non-negative number")
        self.radius = radius_value

    def _select_indices(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        reference: np.ndarray,
        n_select: int,
    ) -> list[int]:
        nearest = _nearest_distances(points, reference, self.chunk_size)
        selected: list[int] = []
        for index in np.argsort(-weights, kind="stable"):
            if len(selected) >= n_select:
                break
            if nearest[index] > self.radius:
                selected.append(int(index))
                _update_nearest(points, nearest, int(index))
        return selected
