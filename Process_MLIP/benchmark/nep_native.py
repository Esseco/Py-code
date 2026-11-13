"""Native GPUMD MD adapter used for GPU NEP benchmark runs."""
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

import numpy as np
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


_TIME_RE = re.compile(r"Time used for this run\s*=\s*([+\-0-9.eE]+)\s*second")
_SPEED_RE = re.compile(
    r"Speed of this run\s*=\s*([+\-0-9.eE]+)\s*atom\*step/second"
)
_STEP_RE = re.compile(r"\b(\d+)\s+steps completed\.")


def gpumd_md_parameters(parameters):
    """Translate benchmark NVT settings into one continuous GPUMD run."""
    timestep = float(parameters["timestep_fs"])
    temperature = float(parameters["temperature_K"])
    tdamp = float(parameters.get("tdamp_fs", 100.0))
    if timestep <= 0 or tdamp <= 0 or temperature <= 0:
        raise ValueError("GPUMD NVT timestep, thermostat damping, and temperature must be positive")
    return [
        ("time_step", timestep),
        ("ensemble", ("nvt_nhc", temperature, temperature, tdamp / timestep)),
        ("dump_thermo", int(parameters.get("log_interval", 10))),
        ("dump_xyz", (int(parameters.get("trajectory_interval", 10)),
                      "movie.xyz", "precision", "double", "velocity")),
        ("run", int(parameters["steps"])),
    ]


def parse_gpumd_timing(stdout, natoms, steps):
    """Read GPUMD's native run duration/throughput without using process wall time."""
    if isinstance(natoms, bool) or not isinstance(natoms, int) or natoms <= 0:
        raise ValueError("natoms must be a positive integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    time_match = _TIME_RE.search(stdout or "")
    speed_match = _SPEED_RE.search(stdout or "")
    elapsed = float(time_match.group(1)) if time_match else None
    atom_steps = float(speed_match.group(1)) if speed_match else None
    if elapsed is not None and (not np.isfinite(elapsed) or elapsed <= 0):
        elapsed = None
    if atom_steps is not None and (not np.isfinite(atom_steps) or atom_steps <= 0):
        atom_steps = None
    if elapsed is None and atom_steps is None:
        raise ValueError("native GPUMD timing was absent or invalid in run.stdout")
    steps_per_second = atom_steps / natoms if atom_steps is not None else None
    if elapsed is None:
        elapsed = natoms * steps / atom_steps
    if steps_per_second is None:
        steps_per_second = steps / elapsed
    return dict(elapsed_seconds=elapsed,
                atom_steps_per_second=atom_steps,
                steps_per_second=steps_per_second)


def parse_gpumd_completed_steps(stdout, expected_steps):
    """Require GPUMD's final progress record to match the requested step count."""
    completed = [int(value) for value in _STEP_RE.findall(stdout or "")]
    if not completed or completed[-1] != expected_steps:
        actual = completed[-1] if completed else "not reported"
        raise ValueError(f"GPUMD reported {actual} completed steps; expected {expected_steps}")
    return completed[-1]


def native_nep_molecular_dynamics(atoms, directory, md_parameters, model, seed):
    """Run NEP NVT through one GPUMD process, retaining native output and timing."""
    if model.get("type") != "nep" or model.get("md_backend", "ase") != "gpumd_native":
        raise ValueError("Native GPUMD MD requires a NEP model configured with md_backend=gpumd_native")
    options = model.get("parameters", {})
    if options.get("backend") != "gpu" or not str(model.get("device", "")).startswith("cuda"):
        raise ValueError("Native GPUMD MD requires the NEP GPU backend and a CUDA device")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gpumd_dir = directory / f"gpumd_{uuid4().hex}"
    initial = atoms.copy()
    MaxwellBoltzmannDistribution(
        initial, temperature_K=float(md_parameters["temperature_K"]),
        rng=np.random.default_rng(seed), force_temp=True,
    )
    Stationary(initial)

    start = perf_counter()
    try:
        from ..nep_md import run_nep_md
        native_report = run_nep_md(
            initial, options["model_path"], gpumd_dir,
            parameters=gpumd_md_parameters(md_parameters),
            executable=options.get("gpumd_command", "gpumd"),
        )
        stage_wall_seconds = perf_counter() - start
        stdout = (gpumd_dir / "run.stdout").read_text(encoding="utf-8", errors="replace")
        timing = parse_gpumd_timing(stdout, len(initial), int(md_parameters["steps"]))
        steps = parse_gpumd_completed_steps(stdout, int(md_parameters["steps"]))

        movie = gpumd_dir / "movie.xyz"
        thermo = gpumd_dir / "thermo.out"
        if not movie.is_file() or movie.stat().st_size == 0:
            raise RuntimeError("Native GPUMD MD completed without a nonempty movie.xyz")
        if not thermo.is_file() or thermo.stat().st_size == 0:
            raise RuntimeError("Native GPUMD MD completed without a nonempty thermo.out")
        final_atoms = read(movie, index=-1)
        final_path = directory / "final.extxyz"
        write(final_path, final_atoms, format="extxyz", write_results=False)
    except Exception as exc:
        exc.benchmark_measurement = dict(
            elapsed_seconds=None, stage_wall_seconds=perf_counter() - start,
            peak_gpu_allocated_MiB=None, peak_gpu_reserved_MiB=None,
            gpu_memory_scope="unavailable_external_gpu_subprocess",
            native_gpumd_directory=str(gpumd_dir),
        )
        raise

    return dict(
        elapsed_seconds=timing["elapsed_seconds"],
        stage_wall_seconds=stage_wall_seconds,
        md_backend="gpumd_native",
        md_parameters=dict(md_parameters),
        thermostat="nvt_nhc",
        thermostat_chain_length=4,
        velocity_initialization="ASE Maxwell-Boltzmann, force_temp=True, COM removed",
        velocity_seed=int(seed),
        steps_completed=steps,
        steps_per_second=timing["steps_per_second"],
        atom_steps_per_second=timing["atom_steps_per_second"],
        simulated_time_ps=steps * float(md_parameters["timestep_fs"]) / 1000,
        timing_scope="gpumd_native_run_internal_timer",
        timing_source="GPUMD run.stdout; excludes Python and model staging wall time",
        stage_wall_scope="includes input writing, model staging, GPUMD startup and MD",
        gpu_memory_scope="unavailable_external_gpu_subprocess",
        native_gpumd_directory=str(gpumd_dir),
        native_gpumd_returncode=native_report["returncode"],
        trajectory_file=str(movie), log_file=str(thermo),
        final_structure=str(final_path),
    )
