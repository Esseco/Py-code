"""Migration endpoint generation and MLIP climbing-image NEB."""
from pathlib import Path
import csv
import json
import math

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

from .MLIP_Relax import MLIPRelaxer
from .calculators import create_mlip_calculator, _symmetrix_species_from_atoms
from .model_config import _resolve_model_config


def _structure(value):
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    if isinstance(value, Structure):
        return value.copy()
    atoms = value.copy() if isinstance(value, Atoms) else read(str(value))
    return AseAtomsAdaptor.get_structure(atoms)


def _validated_structures(structure, parent_structure, mobile_ion, match_tol, overlap_tol):
    target, parent = _structure(structure), _structure(parent_structure)
    if not np.allclose(target.lattice.matrix, parent.lattice.matrix, atol=1e-6, rtol=0):
        raise ValueError("structure and parent_structure must have the same lattice")
    if not isinstance(mobile_ion, str) or not mobile_ion:
        raise ValueError("mobile_ion must be a nonempty element symbol")
    for name, value in (("match_tol", match_tol), ("overlap_tol", overlap_tol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    target_host = sorted(s.specie.symbol for s in target if s.specie.symbol != mobile_ion)
    parent_host = sorted(s.specie.symbol for s in parent if s.specie.symbol != mobile_ion)
    if target_host != parent_host:
        raise ValueError("parent and structure must have the same nonmobile composition")
    from scipy.optimize import linear_sum_assignment
    for symbol in sorted(set(target_host + [mobile_ion])):
        target_sites = [s.frac_coords for s in target if s.specie.symbol == symbol]
        parent_sites = [s.frac_coords for s in parent if s.specie.symbol == symbol]
        if not target_sites or len(parent_sites) < len(target_sites):
            raise ValueError(f"parent has insufficient {symbol} sites")
        distances = target.lattice.get_all_distances(target_sites, parent_sites)
        rows, cols = linear_sum_assignment(distances)
        if len(rows) != len(target_sites) or np.any(distances[rows, cols] > match_tol):
            raise ValueError(f"{symbol} sites do not match the parent within match_tol")
    return target, parent


def generate_neb_pairs(structure, parent_structure, *, mobile_ion="Na", mode="random",
                       random_num=3, indices=None, num_paths_per_ion=1,
                       max_distance=5.0, match_tol=0.6, overlap_tol=0.5, seed=0):
    """Generate mobile-ion/vacancy endpoint pairs using the parent site catalog.

    Inputs may be ASE Atoms, pymatgen Structures, or structure paths. Parent
    and target must use the same lattice; parent may contain mutually exclusive
    candidate sites. This generates unrelaxed endpoints, not energy barriers.
    """
    from Process_Vasp.structures.structure import generate_neb_endpoints
    target, parent = _validated_structures(structure, parent_structure, mobile_ion, match_tol, overlap_tol)
    return generate_neb_endpoints(parent, target, mobile_ion, mode=mode, indices=indices,
                                  random_num=random_num, num_paths_per_ion=num_paths_per_ion,
                                  max_distance=max_distance, match_tol=match_tol,
                                  overlap_tol=overlap_tol, seed=seed)


def run_mlip_neb(structure, parent_structure, output_dir, *, model="mace-mh1",
                 calculator=None, mobile_ion="Na", mode="random", random_num=3,
                 indices=None, num_paths_per_ion=1, max_distance=5.0,
                 match_tol=0.6, overlap_tol=0.5, seed=0, intermediate_images=3,
                 relax_endpoints=True, endpoint_fmax=0.1, endpoint_steps=2000,
                 endpoint_optimizer="LBFGS", neb_fmax=0.1, neb_steps=500,
                 cineb_fmax=0.05, cineb_steps=500, endpoint_pairs=None):
    """Run fixed-cell endpoint relaxation, IDPP, NEB and CI-NEB for each pair.

    ``model`` is any existing model_config (MACE, ORB, SevenNet, etc.); a
    supplied ASE ``calculator`` bypasses model loading. All energies/barriers
    are eV, distances Å, and forces eV/Å. The parent is a site catalog in the
    same cell as the unrelaxed target. Output must not exist. Independent path
    failures are recorded and the remaining paths continue; no restart is implied.
    """
    if not isinstance(intermediate_images, int) or isinstance(intermediate_images, bool) or intermediate_images < 1:
        raise ValueError("intermediate_images must be a positive integer")
    for name, value in (("endpoint_fmax", endpoint_fmax), ("neb_fmax", neb_fmax), ("cineb_fmax", cineb_fmax)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("endpoint_steps", endpoint_steps), ("neb_steps", neb_steps), ("cineb_steps", cineb_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(relax_endpoints) is not bool:
        raise TypeError("relax_endpoints must be bool")
    if mode == "select" and indices is not None and not isinstance(indices, int):
        indices = list(indices)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(out)
    if endpoint_pairs is None:
        pairs = generate_neb_pairs(structure, parent_structure, mobile_ion=mobile_ion,
            mode=mode, random_num=random_num, indices=indices, num_paths_per_ion=num_paths_per_ion,
            max_distance=max_distance, match_tol=match_tol, overlap_tol=overlap_tol, seed=seed)
    else:
        target, parent = _validated_structures(structure, parent_structure, mobile_ion, match_tol, overlap_tol)
        pairs = [(_structure(a), _structure(b)) for a, b in endpoint_pairs]
        expected = [s.specie.symbol for s in target]
        parent_mobile = [s.frac_coords for s in parent if s.specie.symbol == mobile_ion]
        for initial, final in pairs:
            if (not np.allclose(initial.lattice.matrix, target.lattice.matrix, atol=1e-6, rtol=0)
                or not np.allclose(final.lattice.matrix, target.lattice.matrix, atol=1e-6, rtol=0)
                or [s.specie.symbol for s in initial] != expected or [s.specie.symbol for s in final] != expected):
                raise ValueError("preselected endpoint cells and atom order must match structure")
            from pymatgen.io.ase import AseAtomsAdaptor
            aa, bb, tt = map(AseAtomsAdaptor.get_atoms, (initial, final, target))
            _, start_d = find_mic(aa.positions-tt.positions, tt.cell, tt.pbc)
            _, moved_d = find_mic(bb.positions-aa.positions, tt.cell, tt.pbc)
            moved = np.flatnonzero(moved_d > 1e-6)
            if np.any(start_d > 1e-5) or len(moved) != 1 or aa[moved[0]].symbol != mobile_ion:
                raise ValueError("preselected pair must start at structure and move one mobile ion")
            if np.min(target.lattice.get_all_distances([final[int(moved[0])].frac_coords], parent_mobile)) > match_tol:
                raise ValueError("preselected destination is not a parent mobile site")
    if not pairs:
        raise ValueError("No valid migration pairs were generated")
    from pymatgen.io.ase import AseAtomsAdaptor
    config = None
    if calculator is None:
        config = _resolve_model_config(model)
        if config['type'] == 'mace':
            config['symmetrix_species'] = _symmetrix_species_from_atoms(
                config['model_path'], config['enable_symmetrix'], config['symmetrix_species'],
                [AseAtomsAdaptor.get_atoms(pairs[0][0])])
        calculator = create_mlip_calculator(config)
    out.mkdir(parents=True, exist_ok=False)
    run_info = dict(mobile_ion=mobile_ion, mode=mode, seed=seed, random_num=random_num, indices=indices,
                    num_paths_per_ion=num_paths_per_ion, max_distance=max_distance,
                    match_tol=match_tol, overlap_tol=overlap_tol, pair_count=len(pairs),
                    structure_source=str(structure) if isinstance(structure, (str, Path)) else type(structure).__name__,
                    parent_source=str(parent_structure) if isinstance(parent_structure, (str, Path)) else type(parent_structure).__name__,
                    model=config, supplied_calculator=calculator.__class__.__name__,
                    preselected_pairs=endpoint_pairs is not None,
                    relax_endpoints=relax_endpoints, endpoint_fmax=endpoint_fmax,
                    endpoint_steps=endpoint_steps, endpoint_optimizer=endpoint_optimizer,
                    intermediate_images=intermediate_images, neb_fmax=neb_fmax,
                    neb_steps=neb_steps, cineb_fmax=cineb_fmax, cineb_steps=cineb_steps)
    (out / "run.json").write_text(json.dumps(run_info, indent=2, allow_nan=False), encoding="utf-8")
    results = []
    for number, (initial, final) in enumerate(pairs, 1):
        folder = out / f"path_{number:03d}"
        folder.mkdir()
        row = {"path": folder.name, "status": "failed"}
        try:
            a, b = AseAtomsAdaptor.get_atoms(initial), AseAtomsAdaptor.get_atoms(final)
            delta, dist = find_mic(b.positions-a.positions, a.cell, a.pbc)
            moved = np.flatnonzero(dist > 1e-6)
            if len(moved) != 1 or a[moved[0]].symbol != mobile_ion:
                raise ValueError("Endpoint pair must move exactly one mobile ion")
            row.update(moving_index=int(moved[0]), hop_distance_A=float(dist[moved[0]]))
            write(folder / "initial.vasp", a, format="vasp", direct=True)
            write(folder / "final.vasp", b, format="vasp", direct=True)
            if relax_endpoints:
                for name, atoms in (("initial", a), ("final", b)):
                    report = MLIPRelaxer(atoms, calculator, fmax=endpoint_fmax, steps=endpoint_steps,
                        optimizer=endpoint_optimizer, relax_cell=False).relax(
                        traj_file=str(folder / f"{name}_relax.traj"),
                        log_file=str(folder / f"{name}_relax.log"),
                        final_structure=str(folder / f"{name}_relaxed.traj"))
                    row[name+"_converged"] = report["optimizer_converged"]
                    row[name+"_force_max_eV_A"] = report["force_max_ev_per_angstrom"]
                    if not report["optimizer_converged"]:
                        raise RuntimeError(f"{name} endpoint relaxation did not converge")
                _, separation = find_mic(b.positions-a.positions, a.cell, a.pbc)
                row["relaxed_mobile_separation_A"] = float(separation[moved[0]])
                if separation[moved[0]] < 0.5:
                    raise RuntimeError("relaxed endpoints collapsed to the same mobile-ion site")
            images = [a] + [a.copy() for _ in range(intermediate_images)] + [b]
            neb = NEB(images, climb=False, method="improvedtangent", allow_shared_calculator=True)
            neb.interpolate(method="idpp", mic=True)
            for image in images:
                image.calc = calculator
            with FIRE(neb, trajectory=str(folder / "neb.traj"), logfile=str(folder / "neb.log")) as opt:
                row["neb_converged"] = bool(opt.run(fmax=neb_fmax, steps=neb_steps))
            neb.climb = True
            with FIRE(neb, trajectory=str(folder / "cineb.traj"), logfile=str(folder / "cineb.log")) as opt:
                row["cineb_converged"] = bool(opt.run(fmax=cineb_fmax, steps=cineb_steps))
            energies = [float(im.get_potential_energy()) for im in images]
            if not np.all(np.isfinite(energies)):
                raise RuntimeError("calculator returned nonfinite image energy")
            write(folder / "images.traj", images)
            row.update(status="completed" if row["cineb_converged"] else "not_converged",
                energies_eV=energies, forward_barrier_eV=max(energies)-energies[0],
                reverse_barrier_eV=max(energies)-energies[-1],
                endpoint_delta_eV=energies[-1]-energies[0])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        (folder / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
        results.append(row)
    columns = ["path", "status", "moving_index", "hop_distance_A", "neb_converged", "cineb_converged",
               "forward_barrier_eV", "reverse_barrier_eV", "endpoint_delta_eV", "error"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    return {"output_dir": str(out), "results": results, "summary_csv": str(out / "summary.csv")}
