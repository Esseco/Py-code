import math

import pytest
from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass


class Structure:
    def copy(self):
        return self


def evaluation(main, other):
    values = [other, main]
    return {
        "energy_per_model_per_atom": values,
        "energy_mean_per_atom": sum(values) / 2,
    }


def sampler():
    obj = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
    obj.main_model_index = 1
    obj.current_eval = evaluation(-10, 0)
    obj.temperature = 300
    obj.kb = 8.617333262145e-5
    obj.pool, obj.pool_size = [], 4
    return obj


def test_acceptance_uses_main_model_when_committee_mean_disagrees():
    obj = sampler()
    trial = evaluation(-11, 5)  # mean increases, main energy decreases
    accepted, probability, delta = obj._accept(trial)
    assert accepted and probability == 1.0 and delta == -1
    accepted, probability, delta = obj._accept(evaluation(-9, -10))
    assert delta == 1 and probability < 1e-10


def test_pool_ranks_and_deduplicates_by_main_model():
    obj = sampler()
    obj._update_pool(Structure(), evaluation(-10, -10), "a", 1)
    obj._update_pool(Structure(), evaluation(-11, 5), "b", 2)
    obj._update_pool(Structure(), evaluation(-12, 20), "a", 3)
    assert [x["step"] for x in obj.pool] == [3, 2]
    assert [x["energy"] for x in obj.pool] == [-12, -11]


def test_main_energy_requires_valid_member():
    obj = sampler()
    with pytest.raises((ValueError, IndexError)):
        obj._accept({"energy_mean_per_atom": -9, "energy_per_model_per_atom": [-9]})
    with pytest.raises(ValueError):
        obj._accept(evaluation(math.nan, -10))


def test_real_mc_loop_tracks_main_energy_when_means_rank_oppositely(monkeypatch):
    obj = sampler()
    obj.mode = "TM_MC_input"
    obj.max_steps = 2
    obj.save_every = 100
    obj.energy_tol = 0.001
    obj.patience = None
    obj.last_improve_step = 0
    obj.visited = set()
    obj.trace = []
    obj.out_dir = "."
    obj._configure_run = lambda **kwargs: None
    obj.prepare = lambda: setattr(
        obj, "best_energy", obj._main_energy(obj.current_eval)
    )
    trial_rows = iter([(Structure(), "b", None), (Structure(), "c", None)])
    obj._trial = lambda: next(trial_rows)
    obj._record_occupation = lambda key: None
    obj._check_candidate = lambda *a, **kw: {"status": "valid"}
    obj._save_checkpoint = lambda: None

    def complete(main, other):
        result = evaluation(main, other)
        result.update(
            energy_std_per_atom=1,
            energy_var_per_atom=1,
            force_uncertainty_per_atom=0.5,
            force_uncertainty_var_per_atom=0.25,
            force_uncertainty_max=0.5,
            force_uncertainty_var_max=0.25,
        )
        return result

    obj.current_eval = complete(-10, 0)
    values = iter([complete(-11, 5), complete(-10.5, -100)])
    obj._relax_and_evaluate = lambda *a, **kw: (Structure(), next(values), [])
    monkeypatch.setattr("Process_AL_MC.MC_sample.random.random", lambda: 0.5)
    obj.run()
    assert [row["accepted"] for row in obj.trace] == [True, False]
    assert obj.trace[0]["delta_E_main_model_per_atom"] == -1
    assert obj.trace[1]["delta_E_main_model_per_atom"] == 0.5
    assert obj.best_energy == -11
    assert obj.pool[0]["step"] == 1
    assert obj.trace[0]["trial_energy_mean_per_atom"] == -3


def test_trajectory_export_requires_and_uses_main_energy(tmp_path):
    import json
    from pymatgen.core import Lattice, Structure as PmgStructure
    from Process_AL_MC.Auc_code import extract_data_from_mcjson

    folder = tmp_path / "supercell" / "pool"
    folder.mkdir(parents=True)
    path = folder / "rank_000_traj.json"
    crystal = PmgStructure(Lattice.cubic(4), ["Na"], [[0, 0, 0]])
    row = {
        "structure": crystal.as_dict(),
        "energy_mean_per_atom": -5,
        "energy_std_per_atom": 1,
        "force_uncertainty_per_atom": 0.1,
        "force_uncertainty_max": 0.2,
    }
    path.write_text(json.dumps([row]), encoding="utf8")
    with pytest.raises(ValueError, match="main_model_energy_missing"):
        extract_data_from_mcjson(path, tmp_path / "exports")
    row["energy_main_model_per_atom"] = -8
    path.write_text(json.dumps([row]), encoding="utf8")
    assert extract_data_from_mcjson(path, tmp_path / "exports")[0]["c_e"] == -8
