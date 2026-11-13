"""模型无关的单结构弛豫结果适配；底层统一使用 MLIPRelaxer。"""
from pathlib import Path

import numpy as np
from ase.io import write

from ._md_support import _copy_structure
from .MLIP_Relax import MLIPRelaxer


def _validate_relax_settings(fmax, steps):
    """模型加载前校验单结构弛豫参数。"""
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError("fmax must be finite and positive")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")


def _relax_structure(structure, calculator, *, output_path=None, fmax=0.05,
                     steps=150, relax_cell=True, step_callback=None):
    """Shared single-structure optimizer; no trajectory, copied input atoms."""
    _validate_relax_settings(fmax, steps)
    atoms = _copy_structure(structure)
    if len(atoms) == 0:
        raise ValueError("cannot relax an empty structure")
    relaxation = MLIPRelaxer(atoms, calculator, fmax=float(fmax), steps=int(steps),
                             optimizer="LBFGS", relax_cell=relax_cell).relax(
                                 traj_file=None, log_file=None, step_callback=step_callback)
    optimizer_converged = relaxation["optimizer_converged"]

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise RuntimeError("Model relaxation returned non-finite energy or forces")
    force_norms = np.linalg.norm(forces, axis=1)
    final_path = None
    if output_path is not None:
        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(final_path), atoms, format="vasp", direct=True, vasp5=True)

    final_feature_difference = None
    if step_callback is not None and hasattr(step_callback, "check_final"):
        final_feature_difference = step_callback.check_final(atoms)

    result = {
        "status": "completed",
        "relax_stopped_normally": True,
        "optimizer_converged": optimizer_converged,
        "energy": energy,
        "energy_per_atom": energy / len(atoms),
        "energy_unit": "eV",
        "forces_ev_per_angstrom": forces.tolist(),
        "force_max_ev_per_angstrom": float(force_norms.max(initial=0.0)),
        "force_rms_ev_per_angstrom": float(np.sqrt(np.mean(force_norms ** 2))),
        "atom_count": len(atoms),
        "relax_steps_used": relaxation["relax_steps_used"],
        "fmax_target_ev_per_angstrom": float(fmax),
        "cell_relaxed": bool(relax_cell),
        "structure_path": str(final_path) if final_path else None,
    }
    if final_feature_difference is not None:
        failures = list(getattr(step_callback, "failures", []))
        result["extrapolation_monitor"] = {
            "final_feature_difference": float(final_feature_difference),
            "extrapolation_failure": bool(failures),
            "failures": failures,
        }
    return result
