"""All-origin FFT MSD (DXR) and Nernst-Einstein conductivity from a fixed cell.

Second, independent estimator beside the pymatgen route (``export_msd`` /
``MP_MSD``). ``DXR_MSD`` unwraps wrapped NVT frames with minimum-image
increments in fractional coordinates, computes every time origin of the MSD by
FFT autocorrelation (lags from one saved frame to ``max_lag_ratio`` of the
run), and fits an ordinary least-squares free-intercept line, returning the
slope uncertainty. The transport definitions are identical to ``fit_msd``:
D_total = slope_total/6, sigma_NE = N/V * (z e)^2 D / (k_B T), only sampling
and weighting change (all origins and uniform weights here; pymatgen uses its
max-mode lag grid near/above 1 ps with 1/lag weights).
"""
import csv
import gzip
import json
import os
import shutil
import tempfile
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from ase.io import iread

from .msd import _trajectory_times

_ELEMENTARY_CHARGE = 1.602176634e-19
_BOLTZMANN = 1.380649e-23
_CSV_COLUMNS = ["segment", "frame_index", "time_fs", "time_ps", "species", "atom_count",
                "msd_x_A2", "msd_y_A2", "msd_z_A2", "msd_total_A2"]


def _iter_atoms(source, skip_frames):
    """Stream frames, transparently expanding gzip-compressed archives."""
    temporary = None
    try:
        target = str(source)
        if source.suffix.lower() == ".gz":
            inner = Path(source.name[:-3]).suffix or ".traj"
            handle, temporary = tempfile.mkstemp(suffix=inner, prefix="dxr_msd_")
            with os.fdopen(handle, "wb") as out, gzip.open(source, "rb") as raw:
                shutil.copyfileobj(raw, out)
            target = temporary
        for index, atoms in enumerate(iread(target)):
            if index >= skip_frames:
                yield atoms
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def _unwrap_fractional(scaled):
    """Accumulate minimum-image fractional increments to remove wrapping."""
    delta = np.diff(scaled, axis=0)
    delta -= np.round(delta)
    return np.concatenate([scaled[:1], scaled[:1] + np.cumsum(delta, axis=0)], axis=0)


def _all_origin_msd(displacement, max_lag):
    """MSD per axis at every lag, averaged over atoms and all time origins."""
    frame_count = displacement.shape[0]
    lags = np.arange(max_lag + 1)
    x = np.ascontiguousarray(displacement.transpose(1, 2, 0), dtype=np.float64)
    atom_count = x.shape[0]
    cumulative = np.concatenate([np.zeros(x.shape[:2] + (1,)), np.cumsum(x * x, axis=-1)],
                                axis=-1)
    length = 1 << int(np.ceil(np.log2(max(2 * frame_count, 1))))
    transformed = np.fft.rfft(x, n=length, axis=-1)
    autocorrelation = np.fft.irfft(transformed * np.conj(transformed), n=length, axis=-1)
    autocorrelation = autocorrelation[:, :, :frame_count]
    squared = (cumulative[:, :, frame_count][:, :, None] - cumulative[:, :, lags]
               + cumulative[:, :, frame_count - lags] - 2 * autocorrelation[:, :, lags])
    counts = (frame_count - lags).astype(float)
    return lags, squared.mean(axis=0).T / counts[:, None]


def _linear_fit(x, y):
    """Vectorised OLS with free intercept for several columns at once."""
    mean_x = x.mean()
    centered = x - mean_x
    sxx = float(np.sum(centered * centered))
    if sxx <= 0:
        raise ValueError("Fit window needs at least two distinct lag times")
    mean_y = y.mean(axis=0)
    slopes = (centered[:, None] * (y - mean_y)).sum(axis=0) / sxx
    intercepts = mean_y - slopes * mean_x
    residual = y - (x[:, None] * slopes + intercepts)
    variance = np.sum((y - mean_y) ** 2, axis=0)
    r_squared = np.where(variance > 0, 1 - np.sum(residual ** 2, axis=0) / variance, None)
    dof = len(x) - 2
    stderr = np.full(slopes.shape, np.nan)
    if dof > 0:
        stderr = np.sqrt(np.sum(residual ** 2, axis=0) / dof / sxx)
    return slopes, intercepts, r_squared, stderr


def DXR_MSD(trajectory, output_csv=None, *, species=None, frame_interval_fs=None,
            skip_frames=0, max_lag_ratio=0.7, fit_start_ps=1.0, fit_end_ps=None,
            temperature_K=None, volume_A3=None, ion_charge=1, drift_reference="framework",
            output_json=None, overwrite=False):
    """All-origin FFT MSD and optional Nernst-Einstein conductivity.

    trajectory: `.traj`, `.xyz` or gzip-compressed archive readable by ASE.
    species: element string, list/tuple, or None (every element), analysed
    element by element. skip_frames drops equilibration frames.
    frame_interval_fs: time between saved frames; when None it is read from an
    adjacent report.json, which requires equally spaced frames.
    max_lag_ratio: largest lag as a fraction of the run (0 < r < 1).
    fit_start_ps/fit_end_ps: inclusive fit window in ps; None ends at the last
    lag. At least three positive-lag samples are required.
    drift_reference='framework' subtracts the mass-weighted centre-of-mass
    displacement of the atoms outside ``species`` (same convention as
    ``export_msd``); None keeps the raw diffusion. Wrapped frames are always
    unwrapped with fractional minimum-image increments; a constant cell and
    fixed atom order are required (NVT only).
    temperature_K: positive Kelvin for the conductivity; None returns only the
    diffusion coefficient. volume_A3 defaults to the trajectory cell volume.
    ion_charge: signed nonzero charge number, or a dict per element.
    output_csv: writes the MSD curve with the same columns as ``export_msd``;
    output_json: writes the full result. Both refuse to overwrite by default.

    Returns slopes, intercepts, R^2 and slope standard errors for x/y/z/total,
    D = slope/6 (cm^2/s and m^2/s), directional D = slope_axis/2, and the
    uncorrelated sigma_NE in S/cm and S/m. No models are loaded and the input
    trajectory is never modified.
    """
    for key, value, minimum in (("skip_frames", skip_frames, 0),):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key, value in (("max_lag_ratio", max_lag_ratio), ("fit_start_ps", fit_start_ps)):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 < max_lag_ratio < 1:
        raise ValueError("max_lag_ratio must satisfy 0 < ratio < 1")
    if fit_start_ps <= 0:
        raise ValueError("fit_start_ps must be positive")
    if fit_end_ps is not None and (isinstance(fit_end_ps, bool)
                                   or not isinstance(fit_end_ps, Real)
                                   or not np.isfinite(fit_end_ps) or fit_end_ps <= fit_start_ps):
        raise ValueError("fit_end_ps must be finite and greater than fit_start_ps")
    if temperature_K is not None and (isinstance(temperature_K, bool)
                                      or not isinstance(temperature_K, Real)
                                      or not np.isfinite(temperature_K) or temperature_K <= 0):
        raise ValueError("temperature_K must be finite and positive")
    if volume_A3 is not None and (isinstance(volume_A3, bool)
                                  or not isinstance(volume_A3, Real)
                                  or not np.isfinite(volume_A3) or volume_A3 <= 0):
        raise ValueError("volume_A3 must be finite and positive")
    for key, value in (("overwrite", overwrite),):
        if type(value) is not bool:
            raise TypeError(f"{key} must be bool")
    if drift_reference not in (None, "framework"):
        raise ValueError("drift_reference must be None or framework")
    source = Path(trajectory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destinations = {}
    for name, value in (("output_csv", output_csv), ("output_json", output_json)):
        if value is None:
            continue
        destination = Path(value)
        if destination.is_symlink() or destination.is_dir():
            raise FileExistsError(destination)
        destination = destination.resolve()
        if destination == source:
            raise ValueError(f"{name} cannot overwrite the input trajectory")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destinations[name] = destination

    cells, symbols, scaled = [], None, []
    for atoms in _iter_atoms(source, skip_frames):
        if len(atoms) < 1:
            raise ValueError("Trajectory must contain atoms")
        if not np.all(atoms.pbc):
            raise ValueError("DXR MSD requires fully periodic 3D cells")
        cell = np.asarray(atoms.get_cell().array, dtype=float)
        if not np.all(np.isfinite(cell)):
            raise ValueError("Trajectory cell must be finite")
        frame_symbols = np.asarray(atoms.get_chemical_symbols())
        if symbols is None:
            symbols, first_cell = frame_symbols, cell
        elif not np.array_equal(frame_symbols, symbols):
            raise ValueError("Atom identities and order must remain unchanged")
        elif not np.allclose(cell, first_cell, rtol=1e-6, atol=1e-6):
            raise ValueError("DXR MSD requires a fixed cell (use an NVT trajectory)")
        cells.append(cell)
        scaled.append(np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float))
    frame_count = len(scaled)
    if frame_count < 3:
        raise ValueError("DXR MSD needs at least three saved frames")
    scaled = np.asarray(scaled)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Trajectory positions must be finite")
    cell = cells[0]
    volume = float(np.abs(np.linalg.det(cell))) if volume_A3 is None else float(volume_A3)
    if volume <= 0:
        raise ValueError("DXR MSD requires a nonzero cell volume")

    times, time_source = _trajectory_times(source, frame_count + skip_frames, frame_interval_fs)
    times = np.asarray(times)[skip_frames:]
    if frame_count > 2:
        spacing = np.diff(times)
        if not np.allclose(spacing, spacing[0], rtol=1e-9, atol=1e-9):
            raise ValueError("DXR MSD requires equally spaced saved frames")
    interval_fs = float(times[1] - times[0])
    max_lag = int(np.floor((frame_count - 1) * max_lag_ratio))
    max_lag = max(2, min(max_lag, frame_count - 1))

    cartesian = _unwrap_fractional(scaled) @ cell
    if species is None:
        requested = sorted(set(symbols))
    elif isinstance(species, str):
        requested = [species]
    elif isinstance(species, (list, tuple)) and all(isinstance(s, str) for s in species):
        requested = list(dict.fromkeys(species))
    else:
        raise TypeError("species must be an element string, list/tuple, or None")
    missing = [element for element in requested if element not in set(symbols)]
    if missing:
        raise ValueError(f"Requested species not present: {missing}")
    if isinstance(ion_charge, dict):
        charges = {}
        for element in requested:
            if element not in ion_charge:
                raise ValueError(f"Missing ion_charge for {element}")
            value = ion_charge[element]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value == 0):
                raise ValueError("ion_charge must be a finite nonzero number")
            charges[element] = float(value)
    else:
        if (isinstance(ion_charge, bool) or not isinstance(ion_charge, Real)
                or not np.isfinite(ion_charge) or ion_charge == 0):
            raise ValueError("ion_charge must be a finite nonzero number")
        charges = {element: float(ion_charge) for element in requested}
    masses = None
    for atoms in _iter_atoms(source, skip_frames):
        masses = np.asarray(atoms.get_masses(), dtype=float)
        break
    selected = np.isin(symbols, requested)
    reference = np.flatnonzero(~selected)
    if drift_reference == "framework" and not len(reference):
        raise ValueError("framework drift correction requires atoms outside selected species")
    if drift_reference == "framework":
        weights = masses[reference]
        centre = (cartesian[:, reference] * weights[None, :, None]).sum(axis=1)
        centre /= weights.sum()
        cartesian = cartesian - (centre - centre[0])[:, None, :]

    lag_times_ps = np.arange(max_lag + 1) * interval_fs / 1000.0
    results, rows, total_conductivity = [], [], 0.0
    for element in requested:
        mask = symbols == element
        population = int(np.count_nonzero(mask))
        if population < 1:
            raise ValueError(f"Species {element} has no atoms")
        displacement = cartesian[:, mask]
        _, components = _all_origin_msd(displacement, max_lag)
        total = components.sum(axis=1)
        curve = np.column_stack([components, total])
        end = lag_times_ps[-1] if fit_end_ps is None else min(float(fit_end_ps), lag_times_ps[-1])
        chosen = (lag_times_ps >= fit_start_ps) & (lag_times_ps <= end + 1e-12)
        if np.count_nonzero(chosen) < 3:
            raise ValueError("Fit window needs at least three MSD samples; lower fit_start_ps "
                             "or increase max_lag_ratio")
        x, y = lag_times_ps[chosen], curve[chosen]
        slopes, intercepts, r_squared, stderr = _linear_fit(x, y)
        names = ("x", "y", "z", "total")
        diffusion_cm2_s = float(slopes[3]) * 1e-4 / 6
        diffusion_stderr = float(stderr[3]) * 1e-4 / 6
        conductivity = conductivity_stderr = None
        if temperature_K is not None:
            factor = (population / (volume * 1e-30)
                      * (charges[element] * _ELEMENTARY_CHARGE) ** 2 / (_BOLTZMANN * temperature_K))
            conductivity = factor * diffusion_cm2_s * 1e-4
            conductivity_stderr = factor * diffusion_stderr * 1e-4
            total_conductivity += conductivity
        results.append(dict(
            species=element, atom_count=population, ion_charge=charges[element],
            samples=int(np.count_nonzero(chosen)),
            fit_range_ps=[float(x[0]), float(x[-1])],
            slope_A2_per_ps=dict(zip(names, slopes.tolist())),
            slope_stderr_A2_per_ps=dict(zip(names, stderr.tolist())),
            intercept_A2=dict(zip(names, intercepts.tolist())),
            r_squared=dict(zip(names, [None if value is None else float(value)
                                       for value in r_squared])),
            diffusion_m2_s=diffusion_cm2_s * 1e-4,
            diffusion_cm2_s=diffusion_cm2_s,
            diffusion_stderr_cm2_s=diffusion_stderr,
            directional_diffusion_m2_s=(np.asarray(slopes[:3]) * 1e-4 / 2).tolist(),
            conductivity_NE_S_m=None if conductivity is None else conductivity,
            conductivity_NE_S_cm=None if conductivity is None else conductivity / 100,
            conductivity_stderr_S_cm=(None if conductivity_stderr is None
                                      else conductivity_stderr / 100),
        ))
        for index, value in enumerate(curve):
            rows.append(dict(segment=1, frame_index=index,
                             time_fs=float(lag_times_ps[index] * 1000),
                             time_ps=float(lag_times_ps[index]), species=element,
                             atom_count=population, msd_x_A2=float(value[0]),
                             msd_y_A2=float(value[1]), msd_z_A2=float(value[2]),
                             msd_total_A2=float(value[3])))
    result = dict(trajectory_file=str(source), time_source=time_source,
                  frame_count=int(frame_count), skip_frames=int(skip_frames),
                  frame_interval_fs=interval_fs, max_lag_ratio=float(max_lag_ratio),
                  max_lag_ps=float(lag_times_ps[-1]),
                  temperature_K=None if temperature_K is None else float(temperature_K),
                  volume_A3=volume, drift_reference=drift_reference,
                  fit_method="ols", fit_weighting="uniform", fit_range_ps=[float(fit_start_ps),
                                                                          None if fit_end_ps is None
                                                                          else float(fit_end_ps)],
                  method="FFT all-origin MSD; OLS free intercept; Nernst-Einstein",
                  results=results,
                  conductivity_NE_sum_S_m=float(total_conductivity),
                  conductivity_NE_sum_S_cm=float(total_conductivity / 100),
                  limitations="Uncorrelated estimate; a linear MSD window is assumed, not "
                              "detected; uniform weights emphasise long lags more than the "
                              "pymatgen 1/lag fit")
    if "output_csv" in destinations:
        destination = destinations["output_csv"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        result["csv_file"] = str(destination)
    if "output_json" in destinations:
        destination = destinations["output_json"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w" if overwrite else "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        result["json_file"] = str(destination)
    return result
