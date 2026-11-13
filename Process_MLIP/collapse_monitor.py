"""Detect force jumps and compare M3GNet structure embeddings."""

import numpy as np


class CollapseDetected(RuntimeError):
    """Relaxation stopped after the configured structural difference was exceeded."""

    def __init__(self, evidence):
        super().__init__(f"structure collapse indicator exceeded: {evidence['score']:.6g}")
        self.evidence = evidence


class CollapseMonitor:
    def __init__(self, initial_atoms, encoder, *, stop_on_extrapolation=True):
        self.initial_atoms = initial_atoms.copy()
        self.encoder = encoder
        self.previous_force = None
        self.initial_embedding = None
        self.stop_on_extrapolation = stop_on_extrapolation
        self.failures = []

    def __call__(self, atoms):
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError("collapse monitor requires finite atomic forces")
        maximum = float(np.linalg.norm(forces, axis=1).max())
        previous = self.previous_force
        self.previous_force = maximum
        if previous is None or maximum - previous <= 1.0:
            return
        self._check_embedding(atoms, {"previous_force_ev_per_angstrom": previous,
                                     "force_ev_per_angstrom": maximum,
                                     "force_increase_ev_per_angstrom": maximum - previous,
                                     "trigger": "force_jump"})

    def check_final(self, atoms):
        """Check the final frame even when no force jump was observed."""
        return self._check_embedding(atoms, {"trigger": "final"})

    def _check_embedding(self, atoms, evidence):
        if self.initial_embedding is None:
            self.initial_embedding = np.asarray(self.encoder(self.initial_atoms), dtype=float).reshape(-1)
        current = np.asarray(self.encoder(atoms), dtype=float).reshape(-1)
        if (not current.size or current.shape != self.initial_embedding.shape
                or not np.isfinite(current).all() or not np.isfinite(self.initial_embedding).all()):
            raise ValueError("invalid M3GNet embeddings for collapse monitor")
        score = float(np.linalg.norm(current - self.initial_embedding))
        if score > 1.0:
            failure = {"score": score, **evidence}
            self.failures.append(failure)
            if self.stop_on_extrapolation:
                raise CollapseDetected(failure)
        return score


def m3gnet_monitor_factory(model_path, *, stop_on_extrapolation=True):
    """Create per-relaxation monitors with a lazily initialized shared encoder."""
    encoder = None

    def encode(atoms):
        nonlocal encoder
        if encoder is None:
            from Process_MLIP.direct_sampling.features import M3GNetEmbedding
            encoder = M3GNetEmbedding(model_path)
        return encoder(atoms)

    return lambda atoms: CollapseMonitor(atoms, encode, stop_on_extrapolation=stop_on_extrapolation)


def compare_m3gnet_features(structure_a, structure_b, model_path):
    """Return the L2 distance between two M3GNet structure embeddings.

    Structures may be ASE Atoms, pymatgen Structures, or paths readable by ASE.
    The explicitly supplied model_path is an M3GNet checkpoint directory.
    This dimensionless descriptor distance is not a stability verdict.
    """
    from ._md_support import _copy_structure

    first = _copy_structure(structure_a)
    second = _copy_structure(structure_b)
    monitor = m3gnet_monitor_factory(model_path, stop_on_extrapolation=False)(first)
    return monitor.check_final(second)
