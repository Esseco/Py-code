import json
import numpy as np
from ase import Atoms
from ase.io import read
from Process_MLIP.nepkit_sampling import perturb_layered_oxide_structure


def _structure():
    return Atoms('Na2O2', cell=np.diag([3.0, 3.0, 5.0]), pbc=True,
                  scaled_positions=[[0,0,0],[.5,.5,0],[.25,.25,.5],[.75,.75,.5]])


def test_generates_36_frames_and_parent(tmp_path):
    result = perturb_layered_oxide_structure(_structure(), tmp_path/'out', seed=7,
                                             metadata={'phase':'O3','soc':'Na','na_content':1.0})
    assert result['n_frames'] == 36
    frames = read(tmp_path/'out'/'structures.extxyz', index=':')
    assert len(frames) == 36
    assert np.allclose(frames[0].cell, _structure().cell)


def test_manifest_preserves_metadata_and_defaults(tmp_path):
    perturb_layered_oxide_structure(_structure(), tmp_path/'out', metadata={'phase':'O3'}, seed=3)
    data = json.loads((tmp_path/'out'/'manifest.json').read_text())
    assert data['metadata']['phase'] == 'O3'
    assert data['parameters']['atom_displacement_A']['O'] == .08
    assert data['parameters']['cell_scale_fraction']['c'] == .02


def test_invalid_parent_rejected_before_output_creation(tmp_path):
    import pytest
    atoms = Atoms('NaO', positions=[[0,0,0],[0,0,0]], cell=[4,4,4], pbc=True)
    with pytest.raises(ValueError, match='geometry filter'):
        perturb_layered_oxide_structure(atoms, tmp_path/'out', frames_per_parent=1)
    assert not (tmp_path/'out').exists()


def test_periodic_self_image_too_close_is_rejected(tmp_path):
    import pytest
    atoms = Atoms('Na', positions=[[0,0,0]], cell=[0.5,4,4], pbc=True)
    with pytest.raises(ValueError, match='geometry filter'):
        perturb_layered_oxide_structure(atoms, tmp_path/'out', frames_per_parent=1)


def test_all_returned_frames_pass_same_filter(tmp_path):
    from Process_MLIP.nepkit_sampling import _valid
    parent = _structure()
    result = perturb_layered_oxide_structure(parent, tmp_path/'out', seed=7)
    frames = read(result['structures'], index=':')
    assert all(_valid(f, parent.get_volume(), 1.0, (0.75, 1.25)) for f in frames)
