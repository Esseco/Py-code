import pytest
import ase.constraints
from ase.filters import ExpCellFilter

from Process_MLIP.direct_sampling.features import _legacy_m3gnet_ase_import


def test_m3gnet_legacy_import_is_temporary_even_on_failure(monkeypatch):
    monkeypatch.delattr(ase.constraints, "ExpCellFilter", raising=False)
    with pytest.raises(RuntimeError, match="load failed"):
        with _legacy_m3gnet_ase_import():
            from ase.constraints import ExpCellFilter as legacy_filter

            assert legacy_filter is ExpCellFilter
            raise RuntimeError("load failed")
    assert not hasattr(ase.constraints, "ExpCellFilter")


def test_m3gnet_legacy_import_preserves_existing_attribute(monkeypatch):
    original = object()
    monkeypatch.setattr(ase.constraints, "ExpCellFilter", original, raising=False)
    with _legacy_m3gnet_ase_import():
        assert ase.constraints.ExpCellFilter is original
    assert ase.constraints.ExpCellFilter is original


def test_descriptor_checks_built_weights_and_acknowledges_unused_head(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from ase import Atoms
    from Process_MLIP.direct_sampling import features

    calls = []
    status = SimpleNamespace(assert_existing_objects_matched=lambda: calls.append("verified"),
                             expect_partial=lambda: calls.append("unused_head"))
    model = SimpleNamespace(load_weights=lambda path: status)
    encoder = SimpleNamespace(describer_model=model, transform_one=lambda structure: np.array([1.0, 2.0]))
    feature = object.__new__(features.M3GNetEmbedding)
    feature.encoder = encoder
    feature.metadata = {"model_path": str(tmp_path)}
    monkeypatch.setattr(features, "_windows_neighbor_compat", nullcontext)
    atoms = Atoms("Na", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    np.testing.assert_array_equal(feature(atoms), [1.0, 2.0])
    np.testing.assert_array_equal(feature(atoms), [1.0, 2.0])
    assert calls == ["verified", "unused_head"]
