import numpy as np
import pytest

from Process_MLIP.collapse_monitor import CollapseDetected, CollapseMonitor


def test_compare_m3gnet_features_accepts_paths_and_atoms(tmp_path, monkeypatch):
    from ase import Atoms as ASEAtoms
    from ase.io import write
    from Process_MLIP import compare_m3gnet_features
    from Process_MLIP.direct_sampling import features

    first = ASEAtoms('Cu', positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    second = first.copy()
    second.positions[0, 0] = 3
    path = tmp_path / 'first.traj'
    write(path, first)
    class Encoder:
        def __init__(self, model_path):
            assert model_path == 'checkpoint'
        def __call__(self, atoms):
            return [atoms.positions[0, 0], 4 * atoms.positions[0, 0] / 3]
    monkeypatch.setattr(features, 'M3GNetEmbedding', Encoder)
    assert compare_m3gnet_features(path, second, 'checkpoint') == pytest.approx(5)
    assert second.positions[0, 0] == 3


class Atoms:
    def __init__(self, force, feature):
        self.force, self.feature = force, feature

    def __len__(self):
        return 1

    def copy(self):
        return Atoms(self.force, self.feature)

    def get_forces(self):
        return [[self.force, 0, 0]]


def test_force_jump_triggers_feature_difference_and_only_above_threshold_stops():
    calls = []
    def encoder(atoms):
        calls.append(atoms.feature)
        return np.array([atoms.feature, 0])
    monitor = CollapseMonitor(Atoms(0, 0), encoder)
    monitor(Atoms(0, 0))
    monitor(Atoms(1, 5))  # exactly 1 eV/Å is not a jump
    assert calls == []
    monitor(Atoms(2.1, 1))  # exactly 1 feature unit does not stop
    assert calls == [0, 1]
    with pytest.raises(CollapseDetected) as caught:
        monitor(Atoms(3.2, 2))
    assert caught.value.evidence['score'] == 2
    assert calls == [0, 1, 2]


def test_nonfinite_embedding_is_not_treated_as_safe():
    monitor = CollapseMonitor(Atoms(0, 0), lambda atoms: [float('nan')])
    monitor(Atoms(0, 0))
    with pytest.raises(ValueError, match='invalid M3GNet'):
        monitor(Atoms(2, 0))


def test_final_frame_is_checked_without_force_jump():
    monitor = CollapseMonitor(Atoms(0, 0), lambda atoms: [atoms.feature])
    monitor(Atoms(0, 0))
    monitor(Atoms(0.5, 0.5))
    assert monitor.check_final(Atoms(0.5, 1)) == 1
    with pytest.raises(CollapseDetected) as caught:
        monitor.check_final(Atoms(0.5, 1.01))
    assert caught.value.evidence['trigger'] == 'final'
    assert caught.value.evidence['score'] == pytest.approx(1.01)


def test_monitor_can_record_failures_without_interrupting_relaxation():
    monitor = CollapseMonitor(Atoms(0, 0), lambda atoms: [atoms.feature],
                              stop_on_extrapolation=False)
    monitor(Atoms(0, 0))
    monitor(Atoms(2, 2))
    assert monitor.check_final(Atoms(2, 3)) == 3
    assert [row['trigger'] for row in monitor.failures] == ['force_jump', 'final']


def test_relax_callback_observes_initial_step_and_can_stop_optimizer():
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from Process_MLIP.MLIP_Relax import MLIPRelaxer

    atoms = ASEAtoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]])
    seen = []

    def stop(atoms):
        seen.append(atoms.get_positions().copy())
        if len(seen) == 2:
            raise RuntimeError('stop requested')

    with pytest.raises(RuntimeError, match='stop requested'):
        MLIPRelaxer(atoms, EMT(), steps=4).relax(traj_file=None, log_file=None,
                                                step_callback=stop)
    assert len(seen) == 2
    assert not np.array_equal(seen[0], seen[1])


def test_mc_relaxation_monitor_can_interrupt():
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from pymatgen.io.ase import AseAtomsAdaptor
    from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass

    runner = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
    runner.model = EMT()
    runner.relax_cell = False
    runner.fmax = 0.05
    runner.relax_steps = 4
    seen = []
    def factory(initial):
        def monitor(atoms):
            seen.append(atoms.get_positions().copy())
            if len(seen) == 2:
                raise RuntimeError('mc stop requested')
        return monitor
    runner.relaxation_monitor_factory = factory
    structure = AseAtomsAdaptor.get_structure(
        ASEAtoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True))
    with pytest.raises(RuntimeError, match='mc stop requested'):
        runner._relax_with_mace(structure)
    assert len(seen) == 2


def test_mc_relaxation_calls_final_check():
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from pymatgen.io.ase import AseAtomsAdaptor
    from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass

    runner = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
    runner.model, runner.relax_cell, runner.fmax, runner.relax_steps = EMT(), False, 0.05, 0
    seen = []
    class Monitor:
        def __call__(self, atoms):
            pass
        def check_final(self, atoms):
            seen.append(len(atoms))
    runner.relaxation_monitor_factory = lambda atoms: Monitor()
    structure = AseAtomsAdaptor.get_structure(
        ASEAtoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True))
    runner._relax_with_mace(structure)
    assert seen == [2]


def test_standalone_relaxation_checks_final_frame():
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from Process_MLIP._relaxation import _relax_structure

    atoms = ASEAtoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    seen = []
    class Monitor:
        def __call__(self, atoms):
            pass
        def check_final(self, atoms):
            seen.append(len(atoms))
    _relax_structure(atoms, EMT(), steps=0,
                     relax_cell=False, step_callback=Monitor())
    assert seen == [2]


def test_standalone_monitor_evidence_is_returned():
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from Process_MLIP._relaxation import _relax_structure

    atoms = ASEAtoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    calls = iter(([0.0], [2.0]))
    monitor = CollapseMonitor(atoms, lambda _: next(calls), stop_on_extrapolation=False)
    result = _relax_structure(atoms, EMT(), steps=0, relax_cell=False, step_callback=monitor)
    assert result['extrapolation_monitor']['extrapolation_failure'] is True
    assert result['extrapolation_monitor']['final_feature_difference'] == 2.0


def test_mc_extrapolation_frame_is_saved_and_rejected_from_pool():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from pymatgen.core import Lattice, Structure
    from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass, InitialExtrapolationFailure
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        runner = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
        runner.out_dir = Path(folder)
        structure = Structure(Lattice.cubic(4), ['Na', 'O'], [[0, 0, 0], [.5, .5, .5]])
        evidence = {'extrapolation_monitor': {'extrapolation_failure': True, 'failures': [{'score': 2.0}]}}
        result = runner._check_candidate(structure, dict(evidence))
        assert result['status'] == 'invalid' and result['reason'] == 'extrapolation_failure'
        saved = runner.extrapolation_failures[0]
        assert Path(saved['structure_path']).is_file() and len(saved['structure_checksum']) == 64
        with pytest.raises(InitialExtrapolationFailure) as error:
            runner._check_candidate(structure, dict(evidence), initial=True)
        assert Path(error.value.evidence['structure_path']).is_file()


def test_mc_records_monitored_relaxations_and_exact_input_identity(tmp_path):
    from ase import Atoms as ASEAtoms
    from ase.calculators.emt import EMT
    from pymatgen.io.ase import AseAtomsAdaptor
    from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass, _relaxation_input_identity
    atoms=ASEAtoms('Cu2',positions=[[0,0,0],[1.8,0,0]],cell=[10,10,10],pbc=True)
    structure=AseAtomsAdaptor.get_structure(atoms)
    assert _relaxation_input_identity(structure)==_relaxation_input_identity(structure.copy())
    runner=LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
    runner.model,runner.relax_cell,runner.fmax,runner.relax_steps=EMT(),False,0.05,0
    runner.out_dir = tmp_path
    class Monitor:
        failures=[]
        def __call__(self, atoms): pass
        def check_final(self, atoms): return 0.25
    runner.relaxation_monitor_factory=lambda initial: Monitor()
    runner._relax_with_mace(structure)
    observation = runner.relaxation_observations[0]
    assert observation['structure_id'] == _relaxation_input_identity(structure)
    assert observation['identity_scheme'] == 'periodic_input_v1'
    assert observation['monitor_complete'] is True and observation['extrapolation_failure'] is False
    import hashlib
    from pathlib import Path
    frame = Path(observation['structure_path'])
    assert frame.is_file() and hashlib.sha256(frame.read_bytes()).hexdigest() == observation['structure_checksum']
    runner.relaxation_observations = []
    with pytest.raises(FileExistsError, match="monitored_mc_frame_already_exists"):
        runner._relax_with_mace(structure)
