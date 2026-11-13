"""Single-point forces and DP-GEN-style model disagreement for identical frames."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from ase.io import iread

from .calculators import create_mlip_calculator
from .model_config import _resolve_model_config


def _write(path, content, overwrite, protected):
    path = Path(path).resolve()
    if path in protected:
        raise ValueError("Output must not overwrite an input")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
        if overwrite:
            os.replace(temporary, path)
        elif os.name == 'nt':
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _frame_identity(atoms):
    data = dict(numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
                cell=np.asarray(atoms.cell).tolist(), pbc=atoms.pbc.tolist())
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def evaluate_frame_forces(input_file, output_json, model, *, index=':', overwrite=False):
    """Evaluate raw forces (eV/Å) on ASE-readable frames; no relaxation.

    index follows ASE iread syntax; default reads all frames. The model is a
    Process_MLIP preset or model_config dict, loaded once. JSON stores exact
    geometry identities, atomic numbers, forces and resolved model settings.
    Existing outputs are rejected unless overwrite=True. A failure writes no result.
    """
    if type(overwrite) is not bool:
        raise TypeError('overwrite must be bool')
    source = Path(input_file).resolve()
    target = Path(output_json).resolve()
    if target == source:
        raise ValueError('Output must not overwrite input')
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    config = _resolve_model_config(model)
    if config.get('model_path') and target == Path(config['model_path']).resolve():
        raise ValueError('Output must not overwrite model weights')
    calculator = create_mlip_calculator(config)
    rows = []
    for atoms in iread(str(source), index=index):
        identity = _frame_identity(atoms)
        atoms.calc = calculator
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError('Invalid force array')
        rows.append(dict(identity=identity, atomic_numbers=atoms.numbers.tolist(), forces=forces.tolist()))
    if not rows:
        raise ValueError('No frames selected')
    result = dict(schema='process_mlip.forces.v1', source=str(source), selection=str(index),
                  model=config, force_unit='eV/angstrom', constraints_applied=False, frames=rows)
    _write(target, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), overwrite, {source})
    return result


def compare_force_results(result_files, output_json, *, overwrite=False):
    """Compare same ordered frames using DP-GEN population force deviation.

    Per-atom deviation is sqrt(mean_models(sum_xyz((F - mean_F)**2))).
    JSON includes per-atom deviations, variances, and frame min/mean/max.
    Model disagreement is not a calibrated error against reference forces.
    """
    if type(overwrite) is not bool:
        raise TypeError('overwrite must be bool')
    sources = [Path(p).resolve() for p in result_files]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError('At least two distinct result files are required')
    documents = [json.loads(p.read_text(encoding='utf-8')) for p in sources]
    first = documents[0]
    for doc in documents:
        if (doc.get('schema') != 'process_mlip.forces.v1' or doc.get('force_unit') != 'eV/angstrom'
                or doc.get('constraints_applied') is not False or not doc.get('frames')
                or len(doc['frames']) != len(first['frames'])):
            raise ValueError('Incompatible force result')
    rows = []
    for frame_index, group in enumerate(zip(*(d['frames'] for d in documents))):
        reference = group[0]
        if any(f['identity'] != reference['identity'] or f['atomic_numbers'] != reference['atomic_numbers'] for f in group):
            raise ValueError(f'Frame mismatch at index {frame_index}')
        forces = np.asarray([f['forces'] for f in group], dtype=float)
        if forces.shape != (len(documents), len(reference['atomic_numbers']), 3) or not np.isfinite(forces).all():
            raise ValueError('Invalid force array')
        variance = np.mean(np.sum((forces - forces.mean(axis=0)) ** 2, axis=2), axis=0)
        deviation = np.sqrt(variance)
        rows.append(dict(frame_index=frame_index, identity=reference['identity'],
                         atomic_numbers=reference['atomic_numbers'], atomic_variance=variance.tolist(),
                         atomic_deviation=deviation.tolist(), min=float(deviation.min()),
                         mean=float(deviation.mean()), max=float(deviation.max())))
    result = dict(schema='process_mlip.force_deviation.v1', result_files=[str(p) for p in sources],
                  models=[d['model'] for d in documents], deviation_unit='eV/angstrom',
                  variance_unit='(eV/angstrom)^2', variance_ddof=0, frames=rows)
    _write(output_json, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), overwrite, set(sources))
    return result
