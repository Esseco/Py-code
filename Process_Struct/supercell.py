"""Symmetry-distinct supercell matrices and shorter equivalent bases."""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np


def _normalize_sizes(sizes: Iterable[int] | int) -> list[int]:
    """Validate and sort requested volume multipliers."""
    if isinstance(sizes, (bool, np.bool_)):
        raise ValueError("sizes must contain positive integers")
    if isinstance(sizes, Integral):
        values = [sizes]
    elif isinstance(sizes, (str, bytes)):
        raise TypeError("sizes must be an iterable of positive integers")
    else:
        try:
            values = list(sizes)
        except TypeError as exc:
            if isinstance(sizes, (float, np.floating)):
                raise ValueError("sizes must contain positive integers") from exc
            raise TypeError("sizes must be an iterable of positive integers") from exc
    if not values:
        raise ValueError("sizes must not be empty")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value < 1 for value in values):
        raise ValueError("sizes must contain positive integers")
    result = sorted(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("sizes must contain unique values")
    return result


def _contains_supercell(large: np.ndarray, small: np.ndarray) -> bool:
    """Return whether *large* is an integer supercell of *small*."""
    transform = np.asarray(large, dtype=float) @ np.linalg.inv(small)
    rounded = np.rint(transform)
    if not np.allclose(transform, rounded, rtol=0, atol=1e-7):
        return False
    determinant = round(abs(float(np.linalg.det(rounded))))
    expected = abs(float(np.linalg.det(large))) / abs(float(np.linalg.det(small)))
    return determinant > 1 and np.isclose(determinant, expected, rtol=0, atol=1e-7)


def _deduplicate_larger_first(
    matrices_by_size: dict[int, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Remove smaller candidates contained by any larger candidate."""
    removed: dict[int, set[int]] = {size: set() for size in matrices_by_size}
    sizes = sorted(matrices_by_size, reverse=True)
    for large_pos, large_size in enumerate(sizes):
        large_matrices = matrices_by_size[large_size]
        for small_size in sizes[large_pos + 1:]:
            for small_index, (small_matrix, _) in enumerate(matrices_by_size[small_size]):
                if small_index in removed[small_size]:
                    continue
                if any(_contains_supercell(large, small_matrix) for large, _ in large_matrices):
                    removed[small_size].add(small_index)
    return {
        size: [pair for index, pair in enumerate(matrices_by_size[size])
               if index not in removed[size]]
        for size in sorted(matrices_by_size)
    }


def get_supercell_matrices(
    structure,
    sizes: Iterable[int] | int,
    dimensions: str = "abc",
    *,
    deduplicate: bool = False,
    symprec: float = 1e-5,
) -> dict[str, list[np.ndarray]] | dict[str, dict[str, list[np.ndarray]]]:
    """Return symmetry-inequivalent recommended matrices by requested size.

    Args:
        structure: Ordered pymatgen Structure or ASE Atoms, preferably a
            primitive cell. The input cell is used as supplied, without
            primitive-cell conversion.
        sizes: Positive integer volume multipliers relative to the input cell.
        dimensions: Nonempty subset of ``"abc"``. Unselected lattice
            directions remain unchanged.
        deduplicate: If true, process sizes from large to small and remove a
            smaller matrix when an actually larger matrix contains it. The
            default is false.
        symprec: icet/spglib symmetry tolerance in angstrom (default 1e-5).

    Returns:
        For a single integer size, return the legacy ``{"common": [...],
        "recommended": [...]}`` structure. For multiple sizes, return a
        dictionary keyed by size strings; each value contains ``common`` and
        ``recommended`` lists. ``common`` contains icet's symmetry-inequivalent
        matrices; ``recommended`` contains the corresponding Minkowski-reduced
        matrices. Both use ``new_cell = matrix @ old_cell`` and can be passed
        directly to pymatgen or ASE supercell methods.

    Notes:
        Equivalence uses the input's atomic species, positions and symmetry,
        not magnetic moments. Supply an ordered structure. No files are written
        and the input is not modified. Enumeration can grow rapidly with size.
    """
    if not isinstance(deduplicate, (bool, np.bool_)):
        raise TypeError("deduplicate must be bool")
    if not isinstance(dimensions, str) or not dimensions or not set(dimensions) <= set("abc") \
            or len(set(dimensions)) != len(dimensions):
        raise ValueError('dimensions must contain distinct lattice directions from "abc"')
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")

    requested_sizes = _normalize_sizes(sizes)

    from ase import Atoms
    from ase.geometry import minkowski_reduce
    from icet.tools import enumerate_supercells

    if isinstance(structure, Atoms):
        atoms = structure.copy()
    else:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        if not isinstance(structure, Structure):
            raise TypeError("structure must be an ASE Atoms or pymatgen Structure")
        if not structure.is_ordered:
            raise ValueError("structure must be ordered")
        atoms = AseAtomsAdaptor.get_atoms(structure)
    cell = np.asarray(atoms.cell)
    if not len(atoms) or not np.isfinite(cell).all() or np.linalg.matrix_rank(cell) != 3:
        raise ValueError("structure must be nonempty with a finite, full-rank cell")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("structure positions must be finite")

    pbc = [axis in dimensions for axis in "abc"]
    atoms.set_pbc(pbc)
    matrices_by_size: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for size in requested_sizes:
        matrices: list[tuple[np.ndarray, np.ndarray]] = []
        for supercell in enumerate_supercells(
            atoms, [size], niggli_reduce=False, symprec=symprec
        ):
            transform = np.linalg.solve(cell.T, np.asarray(supercell.cell).T).T
            matrix = np.rint(transform).astype(int)
            if not np.allclose(transform, matrix, rtol=0, atol=1e-7):
                raise RuntimeError("icet returned a non-integer supercell transformation")
            _, operation = minkowski_reduce(matrix @ cell, pbc=pbc)
            matrices.append((matrix, operation @ matrix))
        matrices_by_size[size] = matrices

    if deduplicate:
        matrices_by_size = _deduplicate_larger_first(matrices_by_size)
    if len(requested_sizes) == 1:
        common, recommended = zip(*matrices_by_size[requested_sizes[0]]) if matrices_by_size[requested_sizes[0]] else ([], [])
        return {"common": list(common), "recommended": list(recommended)}
    return {
        str(size): {
            "common": [common for common, _ in matrices_by_size[size]],
            "recommended": [recommended for _, recommended in matrices_by_size[size]],
        }
        for size in requested_sizes
    }
