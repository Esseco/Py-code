"""Weighted active-learning samplers: selection rules, validation and edges."""
import unittest

import numpy as np

from Process_MLIP import (
    WeightedFarthestFirstSampler,
    WeightedKMeansPlusPlusSampler,
    WeightedNMSSampler,
    compute_weights,
    gaussian_kernel_density,
)


def _candidates(coords):
    return {f"c{i}": {"features": np.asarray(point, dtype=float)} for i, point in enumerate(coords)}


def _labeled(coords):
    return {f"l{i}": {"features": np.asarray(point, dtype=float)} for i, point in enumerate(coords)}


def _weights(count, value=1.0):
    return {f"c{i}": value for i in range(count)}


class KMeansPlusPlusTests(unittest.TestCase):
    def test_seeded_selection_is_reproducible(self):
        candidates = _candidates(np.random.default_rng(1).normal(size=(40, 6)))
        labeled = _labeled(np.random.default_rng(2).normal(size=(4, 6)))
        weights = _weights(40, 0.5)
        sampler = WeightedKMeansPlusPlusSampler(random_state=7)
        first = sampler.select(candidates, labeled, weights, n_select=5)
        second = sampler.select(candidates, labeled, weights, n_select=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)

    def test_empty_labeled_first_point_follows_weights(self):
        candidates = _candidates([[0.0], [1.0], [2.0]])
        weights = {"c0": 0.0, "c1": 1.0, "c2": 0.0}
        selected = WeightedKMeansPlusPlusSampler(random_state=0).select(
            candidates, {}, weights, n_select=1
        )
        self.assertEqual(selected, ["c1"])

    def test_labeled_duplicate_is_never_preferred(self):
        candidates = _candidates([[0.0, 0.0], [10.0, 0.0]])
        labeled = _labeled([[0.0, 0.0]])
        weights = {"c0": 100.0, "c1": 1.0}
        selected = WeightedKMeansPlusPlusSampler(random_state=0).select(
            candidates, labeled, weights, n_select=1
        )
        self.assertEqual(selected, ["c1"])

    def test_zero_weights_fall_back_without_failure(self):
        candidates = _candidates([[0.0], [1.0], [2.0]])
        selected = WeightedKMeansPlusPlusSampler(random_state=0).select(
            candidates, {}, _weights(3, 0.0), n_select=2
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)

    def test_duplicate_features_still_return_distinct_ids(self):
        candidates = _candidates([[1.0, 1.0]] * 5)
        selected = WeightedKMeansPlusPlusSampler(random_state=0).select(
            candidates, {}, _weights(5), n_select=3
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)


class FarthestFirstTests(unittest.TestCase):
    def test_empty_labeled_picks_highest_weight(self):
        candidates = _candidates([[0.0], [1.0], [2.0]])
        weights = {"c0": 0.1, "c1": 0.9, "c2": 0.5}
        selected = WeightedFarthestFirstSampler().select(candidates, {}, weights, n_select=1)
        self.assertEqual(selected, ["c1"])

    def test_weighted_distance_decides_after_labeled(self):
        labeled = _labeled([[0.0, 0.0]])
        candidates = _candidates([[1.0, 0.0], [5.0, 0.0]])
        equal = {"c0": 1.0, "c1": 1.0}
        self.assertEqual(
            WeightedFarthestFirstSampler().select(candidates, labeled, equal, n_select=1),
            ["c1"],
        )
        weighted = {"c0": 1.0, "c1": 0.01}
        self.assertEqual(
            WeightedFarthestFirstSampler().select(candidates, labeled, weighted, n_select=1),
            ["c0"],
        )

    def test_ties_keep_earlier_position_and_never_repeat(self):
        candidates = _candidates([[0.0], [0.0], [0.0]])
        selected = WeightedFarthestFirstSampler().select(
            candidates, {}, _weights(3, 0.0), n_select=3
        )
        self.assertEqual(selected, ["c0", "c1", "c2"])

    def test_more_than_candidates_selects_all(self):
        candidates = _candidates([[0.0], [1.0]])
        selected = WeightedFarthestFirstSampler().select(
            candidates, {}, _weights(2), n_select=10
        )
        self.assertEqual(len(selected), 2)


class NMSTests(unittest.TestCase):
    def test_radius_filters_and_may_return_fewer(self):
        labeled = _labeled([[0.0, 0.0]])
        candidates = _candidates([[0.5, 0.0], [2.0, 0.0], [2.5, 0.0]])
        weights = {"c0": 3.0, "c1": 2.0, "c2": 1.0}
        selected = WeightedNMSSampler(radius=1.0).select(
            candidates, labeled, weights, n_select=10
        )
        self.assertEqual(selected, ["c1"])

    def test_equal_distance_to_radius_is_excluded(self):
        labeled = _labeled([[0.0]])
        candidates = _candidates([[1.0]])
        selected = WeightedNMSSampler(radius=1.0).select(
            candidates, labeled, {"c0": 1.0}, n_select=1
        )
        self.assertEqual(selected, [])

    def test_empty_labeled_keeps_weight_order(self):
        candidates = _candidates([[0.0], [100.0], [200.0]])
        weights = {"c0": 0.2, "c1": 0.9, "c2": 0.5}
        selected = WeightedNMSSampler(radius=0.0).select(candidates, {}, weights, n_select=2)
        self.assertEqual(selected, ["c1", "c2"])

    def test_negative_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            WeightedNMSSampler(radius=-1.0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.candidates = _candidates([[0.0, 0.0], [1.0, 0.0]])
        self.weights = _weights(2, 1.0)

    def test_invalid_inputs_raise(self):
        cases = [
            ({}, {}, self.weights, 1),
            (self.candidates, {}, {"c0": 1.0}, 1),
            (self.candidates, {}, {**self.weights, "unknown": 1.0}, 1),
            (self.candidates, {}, {"c0": -1.0, "c1": 1.0}, 1),
            (self.candidates, {}, {"c0": np.nan, "c1": 1.0}, 1),
            (self.candidates, {}, {"c0": np.inf, "c1": 1.0}, 1),
            (self.candidates, {}, self.weights, 0),
            (self.candidates, {}, self.weights, 1.5),
            ({"a": {"features": [[0.0, 0.0]]}}, {}, {"a": 1.0}, 1),
            ({"a": {"features": [0.0, np.nan]}}, {}, {"a": 1.0}, 1),
            ({"a": {"features": [0.0]}, "b": {"features": [0.0, 1.0]}}, {}, {"a": 1.0, "b": 1.0}, 1),
            ({"a": np.array([0.0, 0.0])}, {"l": np.array([0.0])}, {"a": 1.0}, 1),
        ]
        for candidates, labeled, weights, n_select in cases:
            with self.subTest(candidates=str(candidates)[:40], n_select=n_select):
                with self.assertRaises((TypeError, ValueError)):
                    WeightedKMeansPlusPlusSampler().select(
                        candidates, labeled, weights, n_select=n_select
                    )

    def test_plain_feature_arrays_are_accepted(self):
        candidates = {"a": np.array([0.0, 0.0]), "b": np.array([1.0, 1.0])}
        selected = WeightedFarthestFirstSampler().select(
            candidates, {}, {"a": 1.0, "b": 1.0}, n_select=1
        )
        self.assertEqual(len(selected), 1)

    def test_missing_features_entry_raises(self):
        with self.assertRaises(ValueError):
            WeightedFarthestFirstSampler().select(
                {"a": {"coordinates": [0.0, 0.0]}}, {}, {"a": 1.0}, n_select=1
            )


class DensityAndMethodTests(unittest.TestCase):
    def test_gaussian_kernel_density_values_and_cluster_ordering(self):
        candidates = _candidates([[0.0], [0.1], [0.2], [10.0]])
        rho = gaussian_kernel_density(candidates, bandwidth=1.0, chunk_size=2)
        self.assertGreater(rho["c1"], rho["c3"])
        expected_c3 = (3.0 * np.exp(-0.5 * (9.9**2)) + 1.0) / 4.0
        # c3 is at 10.0; distances are 10.0, 9.9, 9.8, 0.0
        exact_c3 = (
            np.exp(-50.0) + np.exp(-0.5 * 9.9**2) + np.exp(-0.5 * 9.8**2) + 1.0
        ) / 4.0
        self.assertAlmostEqual(rho["c3"], exact_c3, places=12)
        self.assertGreater(expected_c3, 0.0)

    def test_gaussian_kernel_density_includes_labeled_and_default_bandwidth(self):
        candidates = _candidates([[0.0], [5.0]])
        labeled = _labeled([[0.0], [0.0]])
        rho = gaussian_kernel_density(candidates, labeled=labeled)
        self.assertGreater(rho["c0"], rho["c1"])

        identical = _candidates([[1.0, 1.0], [1.0, 1.0]])
        rho_same = gaussian_kernel_density(identical)
        self.assertAlmostEqual(rho_same["c0"], 1.0)
        self.assertAlmostEqual(rho_same["c1"], 1.0)

    def test_invalid_bandwidth_raises(self):
        candidates = _candidates([[0.0], [1.0]])
        for bad in (0.0, -1.0, np.nan, np.inf, True):
            with self.assertRaises((TypeError, ValueError)):
                gaussian_kernel_density(candidates, bandwidth=bad)

    def test_compute_weights_hook_preserves_base_and_supports_weight_fn(self):
        candidates = _candidates([[0.0], [0.1], [5.0]])
        rho = gaussian_kernel_density(candidates, bandwidth=1.0)
        base = {"c0": 2.0, "c1": 3.0, "c2": 4.0}
        self.assertEqual(compute_weights(candidates, base_weights=base, density=rho), base)
        self.assertEqual(compute_weights(candidates), {"c0": 1.0, "c1": 1.0, "c2": 1.0})
        adjusted = compute_weights(
            candidates,
            base_weights=base,
            density=rho,
            weight_fn=lambda _k, w, r: w / r,
        )
        self.assertAlmostEqual(adjusted["c2"], base["c2"] / rho["c2"])

    def test_kmeans_plus_plus_fps_method_matches_farthest_first(self):
        candidates = _candidates(np.random.default_rng(3).normal(size=(25, 4)))
        labeled = _labeled(np.random.default_rng(4).normal(size=(3, 4)))
        weights = {f"c{i}": float(i + 1) / 25.0 for i in range(25)}

        fps_via_param = WeightedKMeansPlusPlusSampler(method="fps").select(
            candidates, labeled, weights, n_select=6
        )
        fps_legacy = WeightedFarthestFirstSampler().select(
            candidates, labeled, weights, n_select=6
        )
        self.assertEqual(fps_via_param, fps_legacy)

        with self.assertRaises(ValueError):
            WeightedKMeansPlusPlusSampler(method="unknown")


if __name__ == "__main__":
    unittest.main()
