"""MACE 逐层描述符：层选择、切片与聚合（不加载真实权重）。"""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from Process_MLIP.direct_sampling.features import (
    MACEAtomicDescriptors,
    _normalize_layers,
    build_encoder,
)


class _Tensor:
    """Minimal stand-in for the torch tensor returned as ``node_feats``."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Model:
    def __init__(self, array):
        self._array = array
        self.calls = []

    def __call__(self, data, compute_force=True):
        self.calls.append(compute_force)
        return {"node_feats": _Tensor(self._array)}


class _Calculator:
    def __init__(self, array):
        self.models = [_Model(array)]

    def _atoms_to_batch(self, atoms):
        return SimpleNamespace(to_dict=lambda: {})


class _Atoms:
    def __init__(self, symbols):
        self.symbols = list(symbols)

    def __len__(self):
        return len(self.symbols)

    def get_chemical_symbols(self):
        return list(self.symbols)


def _descriptors(raw, dim=2):
    """Build a descriptor object around a fake 2-interaction model."""
    descriptor = object.__new__(MACEAtomicDescriptors)
    descriptor.num_interactions = 2
    descriptor.dim_per_layer = dim
    descriptor._layer_offsets = (0, dim * 4, dim * 5)
    descriptor.calculator = _Calculator(raw)
    return descriptor


class LayerNormalizationTests(unittest.TestCase):
    def test_all_aliases_select_every_layer(self):
        for layers in ("all", "ALL", "both", None, (1, 2)):
            self.assertEqual(_normalize_layers(layers, 2), (1, 2), layers)

    def test_single_and_sequence_layers_keep_order(self):
        self.assertEqual(_normalize_layers(1, 3), (1,))
        self.assertEqual(_normalize_layers("2", 3), (2,))
        self.assertEqual(_normalize_layers("2,1", 3), (2, 1))
        self.assertEqual(_normalize_layers([2, 1, 2], 3), (2, 1))

    def test_invalid_layers_raise(self):
        for layers in (0, 4, [1, 5], "", [], "first"):
            with self.assertRaises(ValueError, msg=repr(layers)):
                _normalize_layers(layers, 3)


class PerAtomSliceTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.arange(4 * 10, dtype=float).reshape(4, 10)
        self.descriptor = _descriptors(self.raw)
        self.atoms = _Atoms(["Na", "Cl", "Na", "Cl"])

    def _per_atom(self, layers):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(no_grad=nullcontext)}):
            return self.descriptor.per_atom(self.atoms, layers)

    def test_first_layer_uses_l0_block(self):
        np.testing.assert_array_equal(self._per_atom(1), self.raw[:, :2])

    def test_second_layer_skips_first_full_width(self):
        np.testing.assert_array_equal(self._per_atom(2), self.raw[:, 8:10])

    def test_all_layers_concatenate_in_requested_order(self):
        expected = np.hstack([self.raw[:, :2], self.raw[:, 8:10]])
        np.testing.assert_array_equal(self._per_atom("all"), expected)
        np.testing.assert_array_equal(
            self._per_atom([2, 1]), np.hstack([self.raw[:, 8:10], self.raw[:, :2]])
        )

    def test_force_computation_is_switched_off(self):
        self._per_atom("all")
        self.assertEqual(self.descriptor.calculator.models[0].calls, [False])


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.arange(4 * 10, dtype=float).reshape(4, 10)
        self.descriptor = _descriptors(self.raw)
        self.atoms = _Atoms(["Na", "Cl", "Na", "Cl"])

    def _aggregate(self, mode):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(no_grad=nullcontext)}):
            return self.descriptor.aggregate(self.atoms, mode)

    def test_atom_mean_pools_every_atom(self):
        values = self._aggregate("atom_mean")
        self.assertEqual(values.shape, (4,))
        both = np.hstack([self.raw[:, :2], self.raw[:, 8:10]])
        np.testing.assert_allclose(values, both.mean(axis=0))

    def test_element_mean_pools_per_symbol(self):
        blocks = self._aggregate("element_mean")
        self.assertEqual(sorted(blocks), ["Cl", "Na"])
        both = np.hstack([self.raw[:, :2], self.raw[:, 8:10]])
        np.testing.assert_allclose(blocks["Na"], both[[0, 2]].mean(axis=0))
        np.testing.assert_allclose(blocks["Cl"], both[[1, 3]].mean(axis=0))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self._aggregate("sum")


class BuildEncoderTests(unittest.TestCase):
    def test_m3gnet_rejects_layers(self):
        with self.assertRaises(ValueError):
            build_encoder("missing_dir", encoder="m3gnet", layers=1)


if __name__ == "__main__":
    unittest.main()
