"""Layered-oxide structure perturbation for local Process_MLIP workflows."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

_DEFAULTS = {
    "frames_per_parent": 36,
    "cell_scale_fraction": {"a": 0.01, "b": 0.01, "c": 0.02},
    "atom_displacement_A": {"Na": 0.12, "Li": 0.12, "TM": 0.05, "O": 0.08},
    "seed": 42,
    "min_distance_A": 1.0,
    "volume_ratio": (0.75, 1.25),
}


def _load(item):
    return item.copy() if isinstance(item, Atoms) else read(str(item), index=0)


def _category(symbol):
    if symbol in {"O", "F", "S", "Se"}:
        return "O"
    if symbol in {"Na", "Li", "K", "Rb", "Cs"}:
        return symbol
    return "TM"


def _valid(atoms, reference_volume, minimum, volume_ratio):
    volume = abs(float(atoms.get_volume()))
    if (not np.isfinite(volume) or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
        or volume <= 0 or not volume_ratio[0] * reference_volume <= volume <= volume_ratio[1] * reference_volume):
        return False
    # Include distinct atoms at distance zero and periodic self-images.
    return len(neighbor_list("d", atoms, minimum, self_interaction=False)) == 0


def perturb_layered_oxide_structure(structure, output_dir, *, frames_per_parent=36,
                                     seed=42, min_distance_A=1.0,
                                     volume_ratio=(0.75, 1.25), metadata=None,
                                     overwrite=False):
    """Generate default joint atom/cell perturbations for one supplied structure.

    Phase, composition/SOC and supercell generation are owned by the phase-diagram agent.
    """
    if isinstance(frames_per_parent, bool) or not isinstance(frames_per_parent, int) or frames_per_parent < 1:
        raise ValueError("frames_per_parent must be a positive integer")
    if (isinstance(min_distance_A, bool) or not np.isfinite(min_distance_A) or min_distance_A <= 0
        or len(volume_ratio) != 2 or not np.isfinite(volume_ratio).all()
        or not 0 < volume_ratio[0] <= 1 <= volume_ratio[1]):
        raise ValueError("invalid geometry filter parameters")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {root}")
    base = _load(structure)
    if not len(base) or not base.pbc.all() or abs(base.get_volume()) <= 0:
        raise ValueError("input structure must have atoms and a periodic nonzero cell")
    reference_volume = abs(float(base.get_volume()))
    if not _valid(base, reference_volume, min_distance_A, volume_ratio):
        raise ValueError("input structure fails geometry filter")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = [base.copy()]
    identities = [dict(perturbation_index=0, **(metadata or {}))]
    for perturbation_index in range(1, frames_per_parent):
        trial = base.copy()
        scales = np.array([1 + rng.uniform(-0.01, 0.01), 1 + rng.uniform(-0.01, 0.01), 1 + rng.uniform(-0.02, 0.02)])
        trial.set_cell(base.cell @ np.diag(scales), scale_atoms=True)
        for i, symbol in enumerate(trial.get_chemical_symbols()):
            amplitude = _DEFAULTS["atom_displacement_A"].get(_category(symbol), 0.05)
            trial.positions[i] += rng.normal(0.0, amplitude, 3)
        if _valid(trial, reference_volume, min_distance_A, volume_ratio):
            frames.append(trial)
            identities.append(dict(perturbation_index=perturbation_index, **(metadata or {})))
    structure_path = root / "structures.extxyz"
    write(structure_path, frames, format="extxyz")
    parameters = dict(_DEFAULTS, frames_per_parent=frames_per_parent, seed=int(seed), min_distance_A=float(min_distance_A), volume_ratio=list(volume_ratio))
    manifest = {"schema": "layered-oxide-perturbation-v1", "n_frames": len(frames), "metadata": dict(metadata or {}), "parameters": parameters, "structures": str(structure_path), "identities": identities}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return {"manifest": str(manifest_path), "structures": str(structure_path), "n_frames": len(frames)}


def generate_nepkit_initial_dataset(structure, output_dir, **kwargs):
    """Compatibility alias for the single-structure perturbation API."""
    return perturb_layered_oxide_structure(structure, output_dir, **kwargs)
