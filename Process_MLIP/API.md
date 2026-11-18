# Process_MLIP 参数索引

源码静态索引；保留原有函数名。类构造和公开方法分别列出。默认值不代表适用于所有体系。

## `WeightedKMeansPlusPlusSampler`

源文件：[al_sampling.py](al_sampling.py)，第 213 行。

```python
WeightedKMeansPlusPlusSampler
```

Weighted k-means++: sample the next point with ``P(i) ∝ w_i d_i²``.

The first point follows the same rule against ``labeled``; when ``labeled``
is empty it is drawn with probability ``∝ w_i``. ``random_state`` reseeds
every ``select`` call, so identical inputs give identical selections.
Zero probabilities (duplicate features, zero weights) fall back to the
highest-weight available candidate, first position on ties.

## `WeightedKMeansPlusPlusSampler.__init__`

源文件：[al_sampling.py](al_sampling.py)，第 223 行。

```python
WeightedKMeansPlusPlusSampler.__init__(self, random_state=None, *, chunk_size=512)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `WeightedFarthestFirstSampler`

源文件：[al_sampling.py](al_sampling.py)，第 252 行。

```python
WeightedFarthestFirstSampler
```

Weighted farthest-first: pick ``argmax_i w_i d_i²`` every step.

With empty ``labeled`` the first pick is the highest-weight candidate. Ties
keep the earlier candidate position, so the sampler is deterministic.

## `WeightedNMSSampler`

源文件：[al_sampling.py](al_sampling.py)，第 281 行。

```python
WeightedNMSSampler
```

Weighted non-maximum suppression with a fixed feature-space radius.

Candidates are visited by descending weight (stable on ties) and kept only
when farther than ``radius`` from ``labeled`` and every accepted candidate.
It may return fewer than ``n_select`` IDs and never pads the result.

## `WeightedNMSSampler.__init__`

源文件：[al_sampling.py](al_sampling.py)，第 289 行。

```python
WeightedNMSSampler.__init__(self, radius, *, chunk_size=512)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[benchmark/__main__.py](benchmark/__main__.py)，第 8 行。

```python
main(argv=None)
```

CLI; validate never loads models, runs MD, or creates output files.

## `plot_comparison`

源文件：[benchmark/comparison.py](benchmark/comparison.py)，第 100 行。

```python
plot_comparison(result_dirs, cps_csv, output_dir, structures=None)
```

Plot MD speed, speed versus CPS and relaxation time from benchmark results.

result_dirs contain speed/md.csv and speed/relax.csv. structures maps material
labels to exact structure IDs selected for CPS; omitted selects LCO_756/NVP_720.
Existing files in output_dir with these plot names are overwritten.

## `validate_md_backend`

源文件：[benchmark/config.py](benchmark/config.py)，第 35 行。

```python
validate_md_backend(model)
```

Apply and validate a model's MD engine without importing model libraries.

## `load_config`

源文件：[benchmark/config.py](benchmark/config.py)，第 47 行。

```python
load_config(path)
```

Read YAML, resolve paths relative to it, validate all benchmark parameters.

## `validate_endmembers`

源文件：[benchmark/ehull.py](benchmark/ehull.py)，第 10 行。

```python
validate_endmembers(compositions)
```

Require elemental terminal phases; never invent/reference-download competitors.

## `compute_hull`

源文件：[benchmark/ehull.py](benchmark/ehull.py)，第 20 行。

```python
compute_hull(records, tolerance=1e-06)
```

Return PhaseDiagram and per-ID eV/atom values from relaxed whole-cell eV.

## `read_reference`

源文件：[benchmark/ehull.py](benchmark/ehull.py)，第 47 行。

```python
read_reference(path)
```

Strict ID,ehull_eV_per_atom CSV; references are already-normalized Ehull.

## `compare_reference`

源文件：[benchmark/ehull.py](benchmark/ehull.py)，第 69 行。

```python
compare_reference(rows, reference, tolerance=1e-06)
```

Match IDs exactly, report coverage, errors and stable-set confusion counts.

## `make_demo`

源文件：[benchmark/examples/make_demo.py](benchmark/examples/make_demo.py)，第 10 行。

```python
make_demo(directory, quick=False)
```

Create synthetic structures and YAML; EMT is only an end-to-end demonstration.

## `gpumd_md_parameters`

源文件：[benchmark/nep_native.py](benchmark/nep_native.py)，第 19 行。

```python
gpumd_md_parameters(parameters)
```

Translate benchmark NVT settings into one continuous GPUMD run.

## `parse_gpumd_timing`

源文件：[benchmark/nep_native.py](benchmark/nep_native.py)，第 36 行。

```python
parse_gpumd_timing(stdout, natoms, steps)
```

Read GPUMD's native run duration/throughput without using process wall time.

## `parse_gpumd_completed_steps`

源文件：[benchmark/nep_native.py](benchmark/nep_native.py)，第 62 行。

```python
parse_gpumd_completed_steps(stdout, expected_steps)
```

Require GPUMD's final progress record to match the requested step count.

## `native_nep_molecular_dynamics`

源文件：[benchmark/nep_native.py](benchmark/nep_native.py)，第 71 行。

```python
native_nep_molecular_dynamics(atoms, directory, md_parameters, model, seed)
```

Run NEP NVT through one GPUMD process, retaining native output and timing.

## `Measurement`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 13 行。

```python
Measurement
```

Synchronized wall time and same-process torch allocator peak in MiB.

## `Measurement.__init__`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 16 行。

```python
Measurement.__init__(self, device, external_gpu=False, backend='torch')
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `Measurement.synchronize`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 37 行。

```python
Measurement.synchronize(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `warmup`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 69 行。

```python
warmup(atoms, calculator, device)
```

One uncached energy/force/stress probe per structure, outside both timers.

## `relax`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 83 行。

```python
relax(atoms, calculator, directory, parameters, device)
```

Full-cell LBFGS; count uncached ASE force requests, save final structure.

## `molecular_dynamics`

源文件：[benchmark/operations.py](benchmark/operations.py)，第 114 行。

```python
molecular_dynamics(atoms, calculator, directory, parameters, device, seed)
```

Reuse NHC NVT; return the existing integrator-only timer, not outer wall time.

## `speed_plots`

源文件：[benchmark/plotting.py](benchmark/plotting.py)，第 9 行。

```python
speed_plots(rows, root, stage)
```

Separate curves per stage; same atom count replicates remain separate points.

## `phase_plot`

源文件：[benchmark/plotting.py](benchmark/plotting.py)，第 37 行。

```python
phase_plot(diagram, root, rows)
```

Plot <=4-component hulls; always provide an ID/Ehull plot for any dimension.

## `register_model`

源文件：[benchmark/registry.py](benchmark/registry.py)，第 9 行。

```python
register_model(name, factory)
```

Register factory(model_config_dict) -> ASE calculator; reject replacement.

## `create_calculator`

源文件：[benchmark/registry.py](benchmark/registry.py)，第 83 行。

```python
create_calculator(config)
```

Build one configured calculator without silent model/backend fallback.

## `run_benchmark`

源文件：[benchmark/runner.py](benchmark/runner.py)，第 186 行。

```python
run_benchmark(config_path, mode, model_ids=None)
```

Run speed or ehull; resume completed stages, retry failed stages, reject input drift.

## `write_json`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 15 行。

```python
write_json(path, value)
```

Atomically replace one owned JSON report, rejecting NaN/Infinity.

## `write_csv`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 37 行。

```python
write_csv(path, rows)
```

Write union-of-columns CSV, retaining explicit error records.

## `file_hash`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 50 行。

```python
file_hash(path)
```

Stream SHA256 without loading large checkpoints into memory.

## `source_hash`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 59 行。

```python
source_hash(path)
```

Hash model files or a SavedModel directory, including relative filenames.

## `identity`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 81 行。

```python
identity(config)
```

Bind outputs to input bytes, checkpoints, configuration and core implementation.

## `output_lock`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 99 行。

```python
output_lock(root)
```

Prevent concurrent writers; stale locks require explicit user inspection/removal.

## `environment`

源文件：[benchmark/storage.py](benchmark/storage.py)，第 115 行。

```python
environment()
```

Record environment and GPU inventory without requiring torch on CPU.

## `create_mace_calculator`

源文件：[calculators.py](calculators.py)，第 29 行。

```python
create_mace_calculator(model_path, *, device='cuda', head=None, default_dtype='float64', enable_cueq=False, enable_symmetrix=False, symmetrix_use_kokkos=False, symmetrix_species=None)
```

Load a local MACE checkpoint. MH-1 filenames default to omat_pbe.

Only load trusted checkpoints: MACE deserializes Python model objects.
Pass head explicitly for renamed multihead weights or other DFT tasks.

## `create_sevennet_calculator`

源文件：[calculators.py](calculators.py)，第 106 行。

```python
create_sevennet_calculator(model='7net-omni', *, device='cpu', modal='mpa', enable_flash=False)
```

Load SevenNet-Omni by official name or trusted local checkpoint.

Named models may download weights through SevenNet on first use. Local
paths must exist. modal selects the DFT task (mpa, omat24, etc.); do not
combine energies from different tasks. No dtype conversion is applied.

## `create_chgnet_calculator`

源文件：[calculators.py](calculators.py)，第 139 行。

```python
create_chgnet_calculator(model_path=None, *, device='cuda', model_name='0.3.0', model=None)
```

Load CHGNet pretrained version, trusted local checkpoint, or loaded model.

The official calculator converts stress from GPa to ASE eV/Å³. Only one
of model_path and model may be supplied; no dependencies are installed.

## `create_orb_calculator`

源文件：[calculators.py](calculators.py)，第 158 行。

```python
create_orb_calculator(model_path=None, *, device='cuda', model_name='orb-v3-conservative-inf-omat', precision='float32-highest', compile=True)
```

Load an ORB pretrained architecture, optionally with local weights.

Supports the official model+adapter API and the older model-only API.
model_path must match model_name's architecture. No fallback model is used.

## `create_mlip_calculator`

源文件：[calculators.py](calculators.py)，第 184 行。

```python
create_mlip_calculator(model='mace-mh1')
```

Build exactly one calculator from a preset name or model_config dict.

## `CollapseDetected`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 6 行。

```python
CollapseDetected
```

Relaxation stopped after the configured structural difference was exceeded.

## `CollapseDetected.__init__`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 9 行。

```python
CollapseDetected.__init__(self, evidence)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CollapseMonitor`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 14 行。

```python
CollapseMonitor
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CollapseMonitor.__init__`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 15 行。

```python
CollapseMonitor.__init__(self, initial_atoms, encoder, *, stop_on_extrapolation=True)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CollapseMonitor.check_final`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 37 行。

```python
CollapseMonitor.check_final(self, atoms)
```

Check the final frame even when no force jump was observed.

## `m3gnet_monitor_factory`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 57 行。

```python
m3gnet_monitor_factory(model_path, *, stop_on_extrapolation=True)
```

Create per-relaxation monitors with a lazily initialized shared encoder.

## `compare_m3gnet_features`

源文件：[collapse_monitor.py](collapse_monitor.py)，第 71 行。

```python
compare_m3gnet_features(structure_a, structure_b, model_path)
```

Return the L2 distance between two M3GNet structure embeddings.

Structures may be ASE Atoms, pymatgen Structures, or paths readable by ASE.
The explicitly supplied model_path is an M3GNet checkpoint directory.
This dimensionless descriptor distance is not a stability verdict.

## `sample`

源文件：[direct_sampling/core.py](direct_sampling/core.py)，第 7 行。

```python
sample(features, n_samples, *, components='kaiser', threshold=0.5, clusters=None, k=1, seed=42)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MACEAtomicDescriptors`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 69 行。

```python
MACEAtomicDescriptors
```

Per-atom invariant MACE descriptors with selectable interaction layers.

``layers`` counts MACE interaction blocks: ``1`` is the first layer, ``2``
the second, ``"all"`` (default) every layer in the checkpoint.  The public
``MACECalculator.get_descriptors`` only returns cumulative layers, so a
single layer is sliced out of the model's ``node_feats`` (the l=0 block of
that layer).  The slice reproduces the official cumulative output exactly;
``Process_MLIP/tests/test_mace_descriptors.py`` guards that equality.

## `MACEAtomicDescriptors.__init__`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 80 行。

```python
MACEAtomicDescriptors.__init__(self, model_path, device='cpu', head=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MACEAtomicDescriptors.per_atom`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 96 行。

```python
MACEAtomicDescriptors.per_atom(self, atoms, layers='all')
```

Return ``(n_atoms, dim_per_layer * len(layers))`` invariant descriptors.

## `MACEAtomicDescriptors.aggregate`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 111 行。

```python
MACEAtomicDescriptors.aggregate(self, atoms, mode='atom_mean', layers='all')
```

Pool per-atom descriptors into one vector or one vector per element.

## `MACEEmbedding`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 121 行。

```python
MACEEmbedding
```

Mean-pooled MACE descriptors, the structure-level encoder of the CLI.

## `MACEEmbedding.__init__`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 124 行。

```python
MACEEmbedding.__init__(self, model_path, device='cpu', head=None, layers='all')
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `M3GNetEmbedding`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 136 行。

```python
M3GNetEmbedding
```

Paper formation-energy structure encoder, provided by maml.

## `M3GNetEmbedding.__init__`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 138 行。

```python
M3GNetEmbedding.__init__(self, model_path)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `build_encoder`

源文件：[direct_sampling/features.py](direct_sampling/features.py)，第 215 行。

```python
build_encoder(model_path, encoder='m3gnet', *, device='cpu', head=None, layers='all')
```

Return a structure encoder; both encoders import their dependencies lazily.

``layers`` selects MACE interaction layers (``1``, ``2``, ``"all"``); the
M3GNet encoder has a single graph readout and rejects any other value.

## `structures_to_pca_csv`

源文件：[direct_sampling/pca.py](direct_sampling/pca.py)，第 68 行。

```python
structures_to_pca_csv(structures, output_csv, *, extractor=None, model=None, encoder='m3gnet', device='cpu', head=None, n_components=2, labels=None, overwrite=False, progress=None)
```

Encode structures, project them on principal components, write one CSV.

structures: iterable of ASE Atoms / pymatgen Structure / structure file path.
    A path is expanded to all its frames, so one trajectory contributes one
    row per frame.
output_csv: CSV destination; one row per structure.
extractor: callable Atoms -> 1D feature vector; overrides model/encoder.
model: checkpoint directory (M3GNet) or model file (MACE); required unless
    an explicit extractor is given.
encoder: "m3gnet" (default) or "mace", mirroring the DIRECT command line.
n_components: number of principal component columns; 2 by default.
labels: optional per-structure grouping column used to colour a 2D plot.
overwrite: allow replacing an existing CSV.

Returns a report dict with the CSV path, explained variance ratios and the
extractor metadata. Features are held in memory, so a huge trajectory should
be sampled first (for example with `export_traj_frames`).

## `extract`

源文件：[direct_sampling/workflow.py](direct_sampling/workflow.py)，第 15 行。

```python
extract(paths, extractor, cache, *, progress=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `run`

源文件：[direct_sampling/workflow.py](direct_sampling/workflow.py)，第 35 行。

```python
run(cache, output, n_samples, *, reference_count=0, **options)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `select_with_reference`

源文件：[direct_sampling/workflow.py](direct_sampling/workflow.py)，第 62 行。

```python
select_with_reference(features, reference_count, max_samples=100, **options)
```

Select new DIRECT cluster representatives not covered by reference rows.

Reference rows come first. Any cluster containing a reference is considered
covered; only the original DIRECT center from remaining clusters is returned.

## `DXR_MSD`

源文件：[dxr_msd.py](dxr_msd.py)，第 100 行。

```python
DXR_MSD(trajectory, output_csv=None, *, species=None, frame_interval_fs=None, skip_frames=0, max_lag_ratio=0.7, fit_start_ps=1.0, fit_end_ps=None, temperature_K=None, volume_A3=None, ion_charge=1, drift_reference='framework', output_json=None, overwrite=False)
```

All-origin FFT MSD and optional Nernst-Einstein conductivity.

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

## `make_data`

源文件：[examples/al_sampling.py](examples/al_sampling.py)，第 23 行。

```python
make_data()
```

Return synthetic candidates, labeled configurations and weights.

## `main`

源文件：[examples/al_sampling.py](examples/al_sampling.py)，第 38 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `make_calculator`

源文件：[examples/ase_relax.py](examples/ase_relax.py)，第 13 行。

```python
make_calculator()
```

Replace with a calculator compatible with the material and environment.

## `main`

源文件：[examples/ase_relax.py](examples/ase_relax.py)，第 18 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/batch_postprocess_md.py](examples/batch_postprocess_md.py)，第 25 行。

```python
main()
```

每个 NVT 阶段一行；失败也记录，避免一个任务中断整批。

## `main`

源文件：[examples/extract_frames.py](examples/extract_frames.py)，第 11 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/mace_relax.py](examples/mace_relax.py)，第 11 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/benchmark_runner.py](examples/md_benchmark/benchmark_runner.py)，第 20 行。

```python
main(case)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/01_mace_mh1/run_md.py](examples/md_benchmark/tasks/01_mace_mh1/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/02_mace_omat/run_md.py](examples/md_benchmark/tasks/02_mace_omat/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/03_mace_omat_sym/run_md.py](examples/md_benchmark/tasks/03_mace_omat_sym/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/04_sevennet_omni/run_md.py](examples/md_benchmark/tasks/04_sevennet_omni/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/05_sevennet_omni_flash/run_md.py](examples/md_benchmark/tasks/05_sevennet_omni_flash/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/06_sevennet_nano6/run_md.py](examples/md_benchmark/tasks/06_sevennet_nano6/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/07_sevennet_nano6_flash/run_md.py](examples/md_benchmark/tasks/07_sevennet_nano6_flash/run_md.py)，第 21 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/md_benchmark/tasks/08_chgnet/run_md.py](examples/md_benchmark/tasks/08_chgnet/run_md.py)，第 22 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/mlip_md.py](examples/mlip_md.py)，第 197 行。

```python
main(argv=None)
```

Preview configuration or run MD after editing the parameters above.

## `main`

源文件：[examples/msd.py](examples/msd.py)，第 44 行。

```python
main()
```

Export native max-mode lag sampling; fit only an explicitly chosen range.

## `main`

源文件：[examples/postprocess_md.py](examples/postprocess_md.py)，第 25 行。

```python
main()
```

复用项目后处理函数；只写 msd.csv 和 transport.json。

## `main`

源文件：[examples/sevennet_relax.py](examples/sevennet_relax.py)，第 11 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_500K_30ps/run_md.py](examples/sym_md_scan/sym_500K_30ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_700K_10ps/run_md.py](examples/sym_md_scan/sym_700K_10ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_700K_30ps/run_md.py](examples/sym_md_scan/sym_700K_30ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_700K_50ps/run_md.py](examples/sym_md_scan/sym_700K_50ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_700K_70ps/run_md.py](examples/sym_md_scan/sym_700K_70ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/sym_md_scan/sym_900K_30ps/run_md.py](examples/sym_md_scan/sym_900K_30ps/run_md.py)，第 25 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `evaluate_frame_forces`

源文件：[force_analysis.py](force_analysis.py)，第 45 行。

```python
evaluate_frame_forces(input_file, output_json, model, *, index=':', overwrite=False)
```

Evaluate raw forces (eV/Å) on ASE-readable frames; no relaxation.

index follows ASE iread syntax; default reads all frames. The model is a
Process_MLIP preset or model_config dict, loaded once. JSON stores exact
geometry identities, atomic numbers, forces and resolved model settings.
Existing outputs are rejected unless overwrite=True. A failure writes no result.

## `compare_force_results`

源文件：[force_analysis.py](force_analysis.py)，第 81 行。

```python
compare_force_results(result_files, output_json, *, overwrite=False)
```

Compare same ordered frames using DP-GEN population force deviation.

Per-atom deviation is sqrt(mean_models(sum_xyz((F - mean_F)**2))).
JSON includes per-atom deviations, variances, and frame min/mean/max.
Model disagreement is not a calibrated error against reference forces.

## `assess_forgetting`

源文件：[forgetting.py](forgetting.py)，第 4 行。

```python
assess_forgetting(structure_ids, collapse_ids, *, coverage_complete)
```

Count distinct relaxed structures; incomplete coverage cannot clear a round.

## `relax_structure_mace`

源文件：[mace_relax.py](mace_relax.py)，第 8 行。

```python
relax_structure_mace(structure, model_path, *, output_path=None, device='cuda', head=None, default_dtype='float64', enable_cueq=False, enable_symmetrix=False, symmetrix_use_kokkos=False, symmetrix_species=None, fmax=0.05, steps=150, relax_cell=True, step_callback=None)
```

Relax one structure and return final energy, forces, and optimizer facts.

``structure`` may be a path, ASE ``Atoms``, or pymatgen ``Structure``.
No trajectory or per-step files are written. If ``output_path`` is given,
only the final structure is written there.

## `run_md`

源文件：[md.py](md.py)，第 37 行。

```python
run_md(structure, calculator, output_dir, *, ensemble='nvt', temperature_K=300.0, steps=1000, timestep_fs=None, tdamp_fs=100.0, pressure_bar=1.0, pdamp_fs=1000.0, initialize_velocities=None, seed=42, remove_com=True, zero_rotation=False, trajectory_interval=10, log_interval=10, overwrite=False, time_offset_fs=0.0, model_metadata=None, measure_steps_per_second=False, npt_cell_mode='isotropic', tchain=3, pchain=3, tloop=1, ploop=1, segment_steps=None, on_segment=None)
```

Run one NHC NVT or MTK NPT stage with an ASE calculator.

Positions/cell use Å; energies eV, temperature K, time/damping fs, pressure
bar (positive compression). Input atoms are copied. initialize_velocities
None initializes only when momenta are absent; False preserves velocities;
True samples Maxwell-Boltzmann momenta using seed. COM removal/rotation
options apply only to initialization. Constraints are not supported.
npt_cell_mode selects isotropic volume scaling or full flexible-cell MTK.
tchain/tloop control the thermostat; pchain/ploop apply only to NPT.
Chain lengths and integration substeps must be positive integers.

segment_steps optionally advances the SAME integrator in blocks. It must
be a positive multiple of trajectory_interval and log_interval; total
steps must align with trajectory_interval as well. After
each block, on_segment(event) sees flushed output and completed steps,
time_fs, paths, temperature_K and saved-frame interval. Return True to
request a normal early stop; False/None continues. No decision rules are
built in. This is in-process continuity, not cross-process checkpointing.
timestep_fs=None selects 2 fs for NVT or 1 fs for NPT; explicit values override.

Writes md.traj, md.csv, final.traj (with momenta), report.json. Existing
owned files require overwrite=True; unrelated files are never removed.
No exact thermostat/barostat restart is provided. Returns final_atoms,
output paths, final thermodynamic state and actual completed step count.
measure_steps_per_second times dyn.run (including observers/output), with
CUDA synchronization. Loading, pre-relaxation, property probes and final
export are excluded. steps_per_second is None when off or steps is zero.

## `run_mlip_md`

源文件：[md_workflow.py](md_workflow.py)，第 15 行。

```python
run_mlip_md(structure, model='mace-mh1', output_dir='md_results', stages=None, *, pre_relax=None, **md_parameters)
```

预弛豫与 MD 共用模型；输出总报告、分阶段轨迹和最终结构。

structure: 结构路径 / ASE Atoms / pymatgen Structure；model: 预设名或配置字典。
stages=None: 单阶段；列表: 顺序衔接，参数优先级为阶段 > 公共 > 默认。
衔接保留位置、晶胞和动量，控温/控压内部状态重置；非精确断点续跑。

pre_relax=None/False: 不弛豫；True/字典: 开启，字典默认值如下：
  enabled=True；relax_cell=True（原子与晶胞一起弛豫，False 仅弛豫原子）。
  optimizer="LBFGS"（BFGS / FIRE / LBFGS，须大写）。
  fmax=0.05 eV/Å；steps=1000（非负整数）；require_convergence=True（未收敛停止）。
  maxstep=0.05 Å；exp_cell_factor=None（FrechetCellFilter 按原子数缩放）。
MD 默认：NVT、300 K、1000 步；步长 NVT=2 fs、NPT=1 fs，可显式覆盖。
tdamp_fs=100，NPT pdamp_fs=1000；timestep_fs=None 按各阶段系综选择。
NPT 晶胞：isotropic（默认）/ full；tchain/pchain=3，tloop/ploop=1。
trajectory_interval/log_interval=10（步）；seed=42；overwrite=False。
measure_steps_per_second=True 可记录吞吐，排除加载、弛豫与后处理。

参考：pre_relax={"relax_cell": True, "optimizer": "LBFGS", "steps": 1000}，
配合 temperature_K=600, steps=5000, trajectory_interval=1, log_interval=1。
返回 final_atoms、stage_results、pre_relax_result、steps_per_second 和报告路径；
elapsed_time_fs 为模拟时长。其余 MD 参数见 run_md。

## `MLIPRelaxer`

源文件：[MLIP_Relax.py](MLIP_Relax.py)，第 7 行。

```python
MLIPRelaxer
```

Relax ASE Atoms with an explicitly supplied calculator.

Modifies the supplied Atoms and retains its constraints. Energies are eV,
fmax is eV/Å. Cell relaxation also tests filter stress degrees of freedom.
maxstep=None uses the ASE optimizer default; positive values limit optimizer
steps (Å for atom variables). FrechetCellFilter scales cell variables by
exp_cell_factor, defaulting to atom count; no direct strain bound is imposed.

## `MLIPRelaxer.__init__`

源文件：[MLIP_Relax.py](MLIP_Relax.py)，第 17 行。

```python
MLIPRelaxer.__init__(self, atoms, calculator, fmax=0.05, steps=200, optimizer='BFGS', relax_cell=False, *, maxstep=None, exp_cell_factor=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MLIPRelaxer.relax`

源文件：[MLIP_Relax.py](MLIP_Relax.py)，第 64 行。

```python
MLIPRelaxer.relax(self, traj_file='relax.traj', log_file='relax.log', final_structure=None, step_callback=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `model_config`

源文件：[model_config.py](model_config.py)，第 25 行。

```python
model_config(preset='mace-mh1', **overrides)
```

Return an independent dict of model defaults with explicit overrides.

Presets: mace-mh1, sevennet-omni, chgnet, orb-v3 and nep. Default device
is cuda for the original model loaders; NEP uses an explicit cpu/gpu
backend. Accelerators default off. model_path overrides pretrained weights. This
function does not check local files, import models, or download anything.

## `export_msd`

源文件：[msd.py](msd.py)，第 70 行。

```python
export_msd(trajectory, output_csv, *, species=None, frame_interval_fs=None, skip_frames=0, number_of_segments=1, unwrap=False, drift_reference=None, overwrite=False, min_obs=30)
```

Export pymatgen max-mode MSD (Å²), averaged over atoms and time origins.

species: element/list/None (all). skip_frames removes equilibration;
number_of_segments splits equally, each analyzed independently. min_obs=30
controls pymatgen's maximum lag; native sampling starts near 1 ps.
CSV columns are unchanged; time_fs/time_ps are lag times, frame_index is
segment start + lag in frames (not a unique displacement endpoint).

Times come from adjacent report.json or frame_interval_fs. A short final
saved interval is excluded with a warning; other unequal spacing is rejected.
unwrap=True uses ASE MIC for wrapped input; default preserves continuous
MD positions. drift_reference='framework' subtracts the mass-weighted COM
displacement of unselected atoms; None applies no drift correction.

Requires pymatgen-analysis-diffusion only when called; no models are loaded.
Writes only output_csv, refuses overwrite by default. Internal pymatgen
diffusion/conductivity fits are discarded; use fit_msd explicitly.

## `generate_neb_pairs`

源文件：[neb.py](neb.py)，第 54 行。

```python
generate_neb_pairs(structure, parent_structure, *, mobile_ion='Na', mode='random', random_num=3, indices=None, num_paths_per_ion=1, max_distance=5.0, match_tol=0.6, overlap_tol=0.5, seed=0)
```

Generate mobile-ion/vacancy endpoint pairs using the parent site catalog.

Inputs may be ASE Atoms, pymatgen Structures, or structure paths. Parent
and target must use the same lattice; parent may contain mutually exclusive
candidate sites. This generates unrelaxed endpoints, not energy barriers.

## `run_mlip_neb`

源文件：[neb.py](neb.py)，第 71 行。

```python
run_mlip_neb(structure, parent_structure, output_dir, *, model='mace-mh1', calculator=None, mobile_ion='Na', mode='random', random_num=3, indices=None, num_paths_per_ion=1, max_distance=5.0, match_tol=0.6, overlap_tol=0.5, seed=0, intermediate_images=3, relax_endpoints=True, endpoint_fmax=0.1, endpoint_steps=2000, endpoint_optimizer='LBFGS', neb_fmax=0.1, neb_steps=500, cineb_fmax=0.05, cineb_steps=500, endpoint_pairs=None)
```

Run fixed-cell endpoint relaxation, IDPP, NEB and CI-NEB for each pair.

``model`` is any existing model_config (MACE, ORB, SevenNet, etc.); a
supplied ASE ``calculator`` bypasses model loading. All energies/barriers
are eV, distances Å, and forces eV/Å. The parent is a site catalog in the
same cell as the unrelaxed target. Output must not exist. Independent path
failures are recorded and the remaining paths continue; no restart is implied.

## `create_nep_calculator`

源文件：[nep.py](nep.py)，第 6 行。

```python
create_nep_calculator(model_path, *, backend='cpu', gpumd_command=None, directory=None)
```

Load a NEP4 model through Calorine's explicit CPU/GPU backend.

``backend='cpu'`` constructs ``CPUNEP``. ``backend='gpu'`` constructs
``GPUNEP`` and optionally forwards a GPUMD command and persistent working
directory. Optional dependencies are imported only when this function is
called; no fallback between CPU and GPU is attempted.

## `vasp_to_nep_xyz`

源文件：[nep_data.py](nep_data.py)，第 109 行。

```python
vasp_to_nep_xyz(inputs, output_file='train.xyz', *, index=':', energy_type='free_energy', include_virial=True, overwrite=False)
```

Convert labelled VASP results to GPUMD NEP extended XYZ.

## `validate_nep_dataset`

源文件：[nep_data.py](nep_data.py)，第 149 行。

```python
validate_nep_dataset(path, *, expected_species=None, require_forces=True, require_virial=False)
```

Validate a GPUMD NEP extended XYZ dataset and return a summary.

## `split_nep_dataset`

源文件：[nep_data.py](nep_data.py)，第 193 行。

```python
split_nep_dataset(input_file, output_dir, *, train_fraction=0.9, seed=42, overwrite=False)
```

Deterministically split a NEP dataset into train.xyz and test.xyz.

## `run_nep_md`

源文件：[nep_md.py](nep_md.py)，第 105 行。

```python
run_nep_md(structure, model_path, output_dir, *, parameters, executable='gpumd', overwrite=False, env=None, timeout=None)
```

Run a native GPUMD task with an explicit NEP model and run parameters.

The task directory contains ``model.xyz``, ``nep.txt``, ``run.in``,
captured stdout/stderr, and ``report.json``. GPUMD output files such as
``thermo.out`` and ``dump.xyz`` are preserved for later analysis.

## `relax_structure_nep`

源文件：[nep_relax.py](nep_relax.py)，第 6 行。

```python
relax_structure_nep(structure, model_path, *, output_path=None, backend='cpu', gpumd_command=None, directory=None, fmax=0.05, steps=150, relax_cell=True)
```

Relax one structure with a Calorine NEP calculator.

## `prepare_nep_training`

源文件：[nep_training.py](nep_training.py)，第 129 行。

```python
prepare_nep_training(output_dir, train_xyz, test_xyz, *, species, parameters=None, overwrite=False, mode='train')
```

Create a self-contained GPUMD NEP training/fine-tuning directory.

## `fine_tune_nep`

源文件：[nep_training.py](nep_training.py)，第 166 行。

```python
fine_tune_nep(output_dir, foundation_model, foundation_restart, *, species, train_xyz=None, test_xyz=None, parameters=None, executable='nep', env=None, timeout=None, overwrite=False)
```

Prepare and run a NEP fine-tuning task from a foundation model/restart.

## `train_nep`

源文件：[nep_training.py](nep_training.py)，第 231 行。

```python
train_nep(output_dir, *, executable='nep', env=None, timeout=None, overwrite=False)
```

Run the external ``nep`` executable in a prepared task directory.

## `perturb_layered_oxide_structure`

源文件：[nepkit_sampling.py](nepkit_sampling.py)，第 45 行。

```python
perturb_layered_oxide_structure(structure, output_dir, *, frames_per_parent=36, seed=42, min_distance_A=1.0, volume_ratio=(0.75, 1.25), metadata=None, overwrite=False)
```

Generate default joint atom/cell perturbations for one supplied structure.

Phase, composition/SOC and supercell generation are owned by the phase-diagram agent.

## `generate_nepkit_initial_dataset`

源文件：[nepkit_sampling.py](nepkit_sampling.py)，第 91 行。

```python
generate_nepkit_initial_dataset(structure, output_dir, **kwargs)
```

Compatibility alias for the single-structure perturbation API.

## `relax_structure_sevennet`

源文件：[sevennet_relax.py](sevennet_relax.py)，第 6 行。

```python
relax_structure_sevennet(structure, model='7net-omni', *, output_path=None, device='cpu', modal='mpa', fmax=0.05, steps=150, relax_cell=True)
```

Return energy (eV), forces (eV/Å), convergence and selected modal.

Inputs are a structure path, ASE Atoms or pymatgen Structure. Atoms are
copied. Only output_path is written (VASP format, overwrites if present).
No random sampling, trajectory or log. Named models may download weights;
supply a trusted local checkpoint for offline repeatable calculations.

## `vasp_to_xyz`

源文件：[training_data.py](training_data.py)，第 12 行。

```python
vasp_to_xyz(inputs, output_file='mace_train.xyz', *, index=':', energy_type='free_energy', include_stress=True, overwrite=False)
```

Export VASP results to MACE extxyz; return the number of written frames.

inputs: OUTCAR/vasprun.xml path, calculation directory, or a list of these.
Directories prefer vasprun.xml over OUTCAR; .gz versions are supported.
index: ASE frame selector, e.g. ":" (all), -1 (last), or "::10".
REF_energy is the whole-cell energy in eV. Default "free_energy" uses
VASP's force-consistent free energy; "energy" selects ASE's extrapolated
zero-smearing energy. Never mix energy conventions within a dataset.
REF_forces are raw DFT forces in eV/Angstrom, without applying constraints.
REF_stress is ASE tensile-positive stress in eV/Angstrom^3 (3x3), including
ASE's VASP sign/unit conversion. Set include_stress=False if unavailable.

Coordinates, cell, species and PBC are retained; constraints/calculators
are removed. Missing/non-finite labels fail without changing an existing
output. The output is replaced only after every selected frame succeeds.
No DFT/model calculation, random sampling or convergence filtering occurs;
the caller must select electronically converged, consistent DFT results.
Uses the existing ASE/NumPy stack (py1); no MACE training dependency.

## `export_traj_frames`

源文件：[trajectory.py](trajectory.py)，第 8 行。

```python
export_traj_frames(trajectory, output_dir=None, frames=-1, *, format='vasp', overwrite=False, uniform_count=None)
```

按保存帧索引导出结构，返回 frame_indices、structure_files 和 frame_count。

frames: 整数/整数列表/slice；0 为首帧，-1 为末帧，None 为全部。
uniform_count: 按帧索引均匀选取指定数量，包含首末帧；1 帧时取末帧。
output_dir=None: 输出到输入轨迹所在目录。
format: vasp（默认）/ cif / xyz / extxyz / traj；文件名 frame_000000.格式。
索引不是 MD 步号；多起点 MSD 的 frame_index 仅作 lag 参考，不代表唯一终点。
负索引转为实际索引。
原子顺序保持不变；不弛豫、不加载模型、不修改源轨迹，默认不覆盖文件。
VASP 使用分数坐标；需完整晶胞。保留 ASE 动量等信息请用 traj。

## `fit_msd`

源文件：[transport.py](transport.py)，第 125 行。

```python
fit_msd(msd_csv, *, fit_start_ps=None, fit_end_ps=None, temperature_K, volume_A3, ion_charge, species=None, output_json=None, overwrite=False, diffusion_mode='auto', axes='xy', anisotropy_threshold=10, min_msd_A2=1.0)
```

Fit MSD CSV and estimate uncorrelated Nernst–Einstein conductivity.

Inclusive fit window in ps; None defaults to the first/last positive lag
of each species/segment separately. Zero lag is excluded. A manual bound
may be supplied on either side; this does not detect the diffusive regime.
Positive temperature in K and fixed cell volume
in Å³. ion_charge is a signed nonzero charge number (e.g. +1), or a dict
mapping selected elements to charge numbers. species is a string/list or
None (all CSV elements). Fits each segment separately, with a free intercept
and at least three positive-lag samples. Calls pymatgen's native fitter
with its defaults (max-mode, weights 1/lag, positive slope floor), after
selecting the window. D_total=slope_total/6; D_axis=slope_axis/2.
Slopes/intercepts describe the raw weighted line before its slope floor;
r_squared remains ordinary, unweighted R² of the fitted line.
Native slope-floor flags identify zero/negative slopes replaced by its
small positive value; these results do not establish resolved diffusion.
Returns SI and cm²/s diffusion, S/m and S/cm conductivity, and diagnostics.
Optional JSON refuses overwrite by default. No uncertainty or correlation
correction is inferred; R² alone does not establish a diffusive regime.

diffusion_mode='auto' keeps 3D results, then checks Cartesian MSD at the
last lag in the fit window: largest/middle > threshold adds 1D; otherwise
middle/smallest > threshold adds 2D. All zero with the MSD gate disabled
makes it indeterminate.
Zero denominators are recorded as None; positive/zero triggers a supplement.
Native slope-floor flags remain diagnostics, not evidence of diffusion.
'3d' disables the check; '1d'/'2d' manually projects onto axes ('x'/'xy', etc.).
axes must be distinct Cartesian letters, not crystallographic directions.
Threshold must be finite and >1 (default 10); it is a heuristic, not a
proof of dimensionality. Projected fits use the same lag window and native
defaults, with D=slope/(2*d). Conductivity still uses the full volume.
New anisotropy/reduced_dimensional and directional conductivity fields
supplement, never replace, the original 3D fields. No source files change.
min_msd_A2=1.0 skips auto supplements when endpoint total MSD is below
this nonnegative Å² threshold (0 disables this gate). low_msd flags each
curve; 3D results and manual 1D/2D fits remain available. This does not
identify thermal vibration or establish a diffusion detection limit.

