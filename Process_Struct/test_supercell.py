from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

from Process_Struct import get_supercell_matrices

try:
    from ase.geometry import minkowski_reduce as _minkowski_reduce
except ImportError:
    _minkowski_reduce = None


@unittest.skipIf(_minkowski_reduce is None, "requires ASE minkowski_reduce")
class SupercellMatricesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = Structure(
            Lattice.cubic(3.0),
            ["Si"],
            [[0.0, 0.0, 0.0]],
        )

    def test_sizes_are_grouped_by_string_without_cross_size_deduplication(self):
        result = get_supercell_matrices(
            self.structure,
            sizes=[2, 4, 6],
            dimensions="abc",
        )

        self.assertEqual(list(result), ["2", "4", "6"])
        self.assertEqual(
            [len(result[key]["common"]) for key in result], [3, 9, 13]
        )
        self.assertEqual(
            [len(result[key]["recommended"]) for key in result], [3, 9, 13]
        )

    def test_deduplicate_removes_smaller_contained_supercells(self):
        result = get_supercell_matrices(
            self.structure,
            sizes=[2, 4],
            dimensions="abc",
            deduplicate=True,
        )

        self.assertEqual(result["2"]["common"], [])
        self.assertEqual(result["2"]["recommended"], [])
        self.assertEqual(len(result["4"]["common"]), 9)
        self.assertEqual(len(result["4"]["recommended"]), 9)

    def test_integer_size_remains_supported(self):
        result = get_supercell_matrices(self.structure, 2)

        self.assertEqual(set(result), {"common", "recommended"})
        self.assertEqual(len(result["common"]), 3)
        self.assertEqual(len(result["recommended"]), 3)


if __name__ == "__main__":
    unittest.main()
