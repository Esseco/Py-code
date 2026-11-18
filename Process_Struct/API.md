# Process_Struct 参数索引

源码静态索引；保留原有函数名。类构造和公开方法分别列出。默认值不代表适用于所有体系。

## `theoretical_specific_capacity`

源文件：[Cal_capacity.py](Cal_capacity.py)，第 43 行。

```python
theoretical_specific_capacity(formula: str, migrating_ion=None)
```

Calculate theoretical specific capacity in mAh/g.

The number of transferable ions is taken from the stoichiometric coefficient
of ``migrating_ion`` in ``formula``. The molar mass includes the complete
input formula.

Args:
    formula: Electrode formula, for example ``"LiFePO4"``.

Returns:
    Theoretical specific capacity in mAh/g.

## `calc_formation_energy`

源文件：[Cal_eform.py](Cal_eform.py)，第 4 行。

```python
calc_formation_energy(df, comp_col, energy_col, fix_ion=None, ion=None, calc_voltage=False)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `read_channel_net`

源文件：[ccnb_net.py](ccnb_net.py)，第 5 行。

```python
read_channel_net(path: Path)
```

Read NET rows, retaining periodic images and reverse connections.

## `prepare_anion_groups`

源文件：[Chemical_Capacity_Constraints/anion_context.py](Chemical_Capacity_Constraints/anion_context.py)，第 13 行。

```python
prepare_anion_groups(formula: str, amounts: Mapping[str, float], rules: Mapping, anion_groups: Mapping[str, float] | None=None)
```

Remove recognized fixed-charge groups and return their net charge.

Groups in parentheses are treated as explicit. Without explicit grouping,
common groups are inferred from the composition and the result is marked as
heuristic. Pass ``anion_groups={"PO4": 2, "SO4": 1}`` to resolve ambiguous
or unparenthesized formulas. Context-dependent groups are never inferred.

## `estimate_mobile_ion_window`

源文件：[Chemical_Capacity_Constraints/framework_ion_window.py](Chemical_Capacity_Constraints/framework_ion_window.py)，第 37 行。

```python
estimate_mobile_ion_window(formula: str, *, mobile_ion: str='Na', environment: str='auto', anion_redox_scenario: str='none', anion_groups: Mapping[str, float] | None=None, rules_path: str=RULES_PATH)
```

Estimate the min/max Li or Na content charge-balanced by a framework.

``formula`` may be either a bare framework (for example ``FePO4``) or a
full composition containing the selected mobile ion (for example
``NaFePO4``). The selected Li/Na is removed before framework charges are
evaluated. If present, its input amount is returned as ``initial_x``.

For each framework element, the configured oxidation-state endpoints are
combined independently. The guest-ion range is therefore a permissive
envelope that allows mixed valence / partial occupancy between endpoints.

## `batch_estimate_mobile_ion_windows`

源文件：[Chemical_Capacity_Constraints/framework_ion_window.py](Chemical_Capacity_Constraints/framework_ion_window.py)，第 267 行。

```python
batch_estimate_mobile_ion_windows(formulas: Iterable[str], *, mobile_ion: str='Na', environment: str='auto', anion_redox_scenario: str='none', anion_groups: Mapping[str, float] | None=None, rules_path: str=RULES_PATH)
```

Estimate guest-ion windows for many formulas, retaining row-level errors.

## `load_rules`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 25 行。

```python
load_rules(path: str | Path=RULES_PATH)
```

Load the editable oxidation-state rules.

## `infer_environment`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 35 行。

```python
infer_environment(formula: str | Composition)
```

Infer a broad anion family; pass environment explicitly when ambiguous.

## `estimate_redox_capacity`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 210 行。

```python
estimate_redox_capacity(formula: str, *, environment: str='auto', mobile_ions: Mapping[str, float] | None=None, active_elements: Iterable[str] | None=None, anion_redox_scenario: str='none', anion_groups: Mapping[str, float] | None=None, rules_path: str | Path=RULES_PATH, max_solutions: int=10000)
```

Estimate charge-neutral valences and a heuristic electron/capacity budget.

Pass the full composition at the intended initial SOC (normally the
alkali-containing, discharged composition). Alkali ions are assigned their
fixed formal charges for neutrality and their extractable charge inventory
caps the reported capacity. If no mobile ion from mobile_ions is present,
no mobile-ion cap is applied.

The default Q_redox includes transition-metal/cation oxidation only.
Optional anion scenarios are reported separately and added only when
explicitly selected.

## `batch_estimate_redox_capacity`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 417 行。

```python
batch_estimate_redox_capacity(formulas: Iterable[str], *, environment: str='auto', capacity_cutoff_mAh_g: float=100.0, mobile_ions: Mapping[str, float] | None=None, anion_redox_scenario: str='none', anion_groups: Mapping[str, float] | None=None, rules_path: str | Path=RULES_PATH)
```

Apply the estimator to many formulas and return flat, CSV-friendly rows.

## `screen_substitutions`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 487 行。

```python
screen_substitutions(formula: str, replaced_element: str, candidate_elements: Sequence[str] | None=None, *, environment: str='auto', capacity_cutoff_mAh_g: float=100.0, q_cutoff_e_per_formula: float | None=None, mobile_ions: Mapping[str, float] | None=None, anion_redox_scenario: str='none', anion_groups: Mapping[str, float] | None=None, rules_path: str | Path=RULES_PATH)
```

Screen same-stoichiometry elemental substitutions by charge/capacity only.

This does not create or relax structures and does not predict synthesis,
phase stability, percolation, ion transport, or electronic conductivity.

## `main`

源文件：[Chemical_Capacity_Constraints/redox_capacity.py](Chemical_Capacity_Constraints/redox_capacity.py)，第 604 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `deduplicate`

源文件：[deduplication.py](deduplication.py)，第 9 行。

```python
deduplicate(struc_lst: list)
```

deduplicate 的 Docstring

:param struc_lst: structs list
:type struc_lst: list
:return: deduplicated list
:rtype: list

## `deduplicate_dict`

源文件：[deduplication.py](deduplication.py)，第 33 行。

```python
deduplicate_dict(struc_dict: dict)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `deduplicate_df`

源文件：[deduplication.py](deduplication.py)，第 59 行。

```python
deduplicate_df(df: pd.DataFrame, n_keep: int, struct_col: str='struct', energy_col: str='energy_mean_per_atom')
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/capacity.py](examples/capacity.py)，第 5 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/deduplicate_structures.py](examples/deduplicate_structures.py)，第 10 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `rocksalt_nacl`

源文件：[examples/feature_sets.py](examples/feature_sets.py)，第 13 行。

```python
rocksalt_nacl()
```

Cubic NaCl with oxidation states, so CrystalNN resolves six neighbours.

## `main`

源文件：[examples/feature_sets.py](examples/feature_sets.py)，第 24 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/octahedra.py](examples/octahedra.py)，第 8 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/percolation.py](examples/percolation.py)，第 8 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `export_sites`

源文件：[examples/percolation_sites.py](examples/percolation_sites.py)，第 28 行。

```python
export_sites(folder: Path, migrant: str | None)
```

Export channel NET sites and edges; retain every periodic connection.

## `main`

源文件：[examples/percolation_sites.py](examples/percolation_sites.py)，第 97 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/structure_features.py](examples/structure_features.py)，第 9 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `main`

源文件：[examples/wyckoff_sites.py](examples/wyckoff_sites.py)，第 7 行。

```python
main()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureResult`

源文件：[features/api.py](features/api.py)，第 56 行。

```python
FeatureResult
```

Feature table for one structure.

Attributes:
    values: Structure-level column name to value, in column order.
    sites: Site-level rows; each row always has ``site_index`` and
        ``element`` plus the prefixed site columns of the enabled sets.
    metadata: Set selection, level, per-set provider metadata, structure
        summary and interpreter details needed to reproduce the run.
    warnings: Warning texts emitted while computing; de-duplicated in
        emission order.

## `FeatureResult.names`

源文件：[features/api.py](features/api.py)，第 75 行。

```python
FeatureResult.names(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureResult.site_names`

源文件：[features/api.py](features/api.py)，第 79 行。

```python
FeatureResult.site_names(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureResult.to_vector`

源文件：[features/api.py](features/api.py)，第 87 行。

```python
FeatureResult.to_vector(self)
```

Structure-level values as a float array in ``names`` order.

## `FeatureResult.to_frame`

源文件：[features/api.py](features/api.py)，第 92 行。

```python
FeatureResult.to_frame(self)
```

One-row DataFrame with structure-level columns.

## `FeatureResult.to_site_frame`

源文件：[features/api.py](features/api.py)，第 99 行。

```python
FeatureResult.to_site_frame(self)
```

Site-level rows as a DataFrame; no rows give an empty frame.

## `FeatureExtractor`

源文件：[features/api.py](features/api.py)，第 107 行。

```python
FeatureExtractor
```

Reusable extractor; expensive models are loaded once at construction.

Args:
    sets: Feature sets to compute; ``None`` means ``("geometry",)``.
    level: ``"structure"``, ``"site"`` or ``"both"``.
    encoder: MLIP encoder, ``"mace"`` or ``"m3gnet"``.
    model_path: MLIP model path; required when ``sets`` contains ``"mlip"``.
    device: Torch device for MACE.
    head: Optional MACE head.
    mlip_layers: MACE interaction layers, ``1``, ``2``, ``"all"`` or a list.
    mlip_aggregate: ``"atom_mean"`` (one vector) or ``"element_mean"``
        (one block per element, columns like ``mlip.Na.d000``).
    geometry_options: Keyword arguments for ``StructureFeatureExtractor``.
    cnn_options: Keyword arguments forwarded to pymatgen ``CrystalNN``.
    cnn_site_filter: ``"all"`` or ``"tm"`` for the native CN layer.
    cnn_fp_preset: matminer fingerprint preset, ``"cn"`` or ``"ops"``.
    include: Glob patterns kept, matched against full column names.
    exclude: Glob patterns dropped after ``include``.

## `FeatureExtractor.__init__`

源文件：[features/api.py](features/api.py)，第 128 行。

```python
FeatureExtractor.__init__(self, sets: str | Sequence[str] | None=None, *, level: str='structure', encoder: str='mace', model_path: str | None=None, device: str='cpu', head: str | None=None, mlip_layers: str | int | Sequence[int]='all', mlip_aggregate: str='atom_mean', geometry_options: dict[str, Any] | None=None, cnn_options: dict[str, Any] | None=None, cnn_site_filter: str='all', cnn_fp_preset: str='cn', include: Sequence[str] | None=None, exclude: Sequence[str] | None=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractor.metadata`

源文件：[features/api.py](features/api.py)，第 186 行。

```python
FeatureExtractor.metadata(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractor.extract`

源文件：[features/api.py](features/api.py)，第 203 行。

```python
FeatureExtractor.extract(self, structure_or_path: Any)
```

Compute the selected features for one structure.

## `load_structure`

源文件：[features/api.py](features/api.py)，第 347 行。

```python
load_structure(structure_or_path: Any)
```

Return a pymatgen ``Structure`` from a Structure, ASE Atoms or path.

## `extract_features`

源文件：[features/api.py](features/api.py)，第 377 行。

```python
extract_features(structure_or_path: Any, **kwargs: Any)
```

One-shot convenience wrapper; rebuilds the extractor on every call.

Use :class:`FeatureExtractor` for batches so models are loaded once.

## `extract_many`

源文件：[features/api.py](features/api.py)，第 386 行。

```python
extract_many(structures: Iterable[Any], *, return_sites: bool=False, **kwargs: Any)
```

Extract features for many structures; one bad structure does not abort.

Returns a DataFrame with ``status``, ``error`` and ``formula`` columns in
addition to the feature columns.  With ``return_sites=True`` returns
``(structure_frame, site_frame)``, where the site frame carries a
``structure_index`` column.

## `CrystalNNCounter`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 52 行。

```python
CrystalNNCounter
```

Coordination numbers and their distribution from pymatgen ``CrystalNN``.

Args:
    options: Keyword arguments forwarded to ``CrystalNN``; empty by default,
        which keeps pymatgen's own defaults (no oxidation-state guessing).
    site_filter: ``"all"`` for every site or ``"tm"`` for transition-metal
        sites only.
    tm_elements: Elements treated as transition metals when
        ``site_filter="tm"``.

## `CrystalNNCounter.__init__`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 66 行。

```python
CrystalNNCounter.__init__(self, options: dict[str, Any] | None=None, site_filter: str='all', tm_elements: Sequence[str] | None=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNCounter.metadata`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 80 行。

```python
CrystalNNCounter.metadata(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNCounter.compute`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 96 行。

```python
CrystalNNCounter.compute(self, structure: Structure, *, with_sites: bool=True)
```

Return CrystalNN coordination statistics and optional per-site rows.

## `CrystalNNFingerprints`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 164 行。

```python
CrystalNNFingerprints
```

matminer CrystalNN fingerprints, imported lazily.

``preset="cn"`` gives 24 site-level values and 48 structure-level values
(mean/std/min/max of the site vector); ``preset="ops"`` gives 61 and 122.

## `CrystalNNFingerprints.__init__`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 173 行。

```python
CrystalNNFingerprints.__init__(self, preset: str='cn')
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNFingerprints.metadata`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 192 行。

```python
CrystalNNFingerprints.metadata(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNFingerprints.site_names`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 203 行。

```python
CrystalNNFingerprints.site_names(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNFingerprints.structure_names`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 207 行。

```python
CrystalNNFingerprints.structure_names(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `CrystalNNFingerprints.compute`

源文件：[features/crystalnn.py](features/crystalnn.py)，第 210 行。

```python
CrystalNNFingerprints.compute(self, structure: Structure, *, with_sites: bool=True)
```

Return structure-level statistics and optional per-site rows.

## `GeometryFeatures`

源文件：[features/geometry.py](features/geometry.py)，第 21 行。

```python
GeometryFeatures
```

Wrap ``StructureFeatureExtractor`` behind one ``compute`` call.

## `GeometryFeatures.__init__`

源文件：[features/geometry.py](features/geometry.py)，第 26 行。

```python
GeometryFeatures.__init__(self, options: dict[str, Any] | None=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `GeometryFeatures.metadata`

源文件：[features/geometry.py](features/geometry.py)，第 33 行。

```python
GeometryFeatures.metadata(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `GeometryFeatures.compute`

源文件：[features/geometry.py](features/geometry.py)，第 40 行。

```python
GeometryFeatures.compute(self, structure: Structure, *, with_sites: bool=True)
```

Return ``(structure_values, site_rows)`` for one structure.

Both tables come from a single ``extract`` call so the neighbour search
is not repeated.  Keys are *not* prefixed here; :mod:`api` adds prefixes.

## `MLIPFeatures`

源文件：[features/mlip.py](features/mlip.py)，第 25 行。

```python
MLIPFeatures
```

MACE / M3GNet embedding of one structure.

Args:
    model_path: Path of the MLIP model directory or file.
    encoder: ``"mace"`` or ``"m3gnet"``.
    device: Torch device for MACE; ignored by M3GNet.
    head: Optional MACE head name.
    layers: MACE interaction layers to use: ``1`` (first), ``2`` (second),
        ``"all"`` (every layer, the default), or a sequence of layer
        numbers. M3GNet has a single readout and only accepts ``"all"``.
    aggregate: ``"atom_mean"`` pools every atom into one vector;
        ``"element_mean"`` pools per chemical element and emits one block
        per element (``mlip.Na.d000``, ``mlip.Cl.d000``, ...). M3GNet only
        supports ``"atom_mean"``.

## `MLIPFeatures.__init__`

源文件：[features/mlip.py](features/mlip.py)，第 44 行。

```python
MLIPFeatures.__init__(self, model_path: str, encoder: str='mace', device: str='cpu', head: str | None=None, layers: str | int | tuple[int, ...] | list[int]='all', aggregate: str='atom_mean')
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MLIPFeatures.metadata`

源文件：[features/mlip.py](features/mlip.py)，第 87 行。

```python
MLIPFeatures.metadata(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MLIPFeatures.descriptors`

源文件：[features/mlip.py](features/mlip.py)，第 101 行。

```python
MLIPFeatures.descriptors(self)
```

The loaded :class:`MACEAtomicDescriptors` (MACE only).

## `MLIPFeatures.names`

源文件：[features/mlip.py](features/mlip.py)，第 109 行。

```python
MLIPFeatures.names(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MLIPFeatures.compute`

源文件：[features/mlip.py](features/mlip.py)，第 120 行。

```python
MLIPFeatures.compute(self, structure: Structure, *, with_sites: bool=False)
```

Return the embedding as ``{"d000": ...}`` or ``{"Na.d000": ...}``.

## `FeatureSet`

源文件：[features/registry.py](features/registry.py)，第 35 行。

```python
FeatureSet
```

One selectable group of feature columns.

Attributes:
    name: Public name used in ``sets=[...]``.
    prefix: Prefix every column produced by the set carries, e.g. ``geom.``.
    levels: Levels the set can fill: ``structure``, ``site`` or both.
    requires: Importable module names that must be installed.
    description: One-line human description.

## `check_dependencies`

源文件：[features/registry.py](features/registry.py)，第 91 行。

```python
check_dependencies(feature_set: FeatureSet)
```

Raise :class:`ImportError` when a required package is missing.

The message names the interpreter so a wrong conda environment is easy to
spot, which is the most common cause on this machine.

## `resolve_sets`

源文件：[features/registry.py](features/registry.py)，第 107 行。

```python
resolve_sets(sets: str | Sequence[str] | None)
```

Return the requested feature sets, in request order, without duplicates.

## `resolve_level`

源文件：[features/registry.py](features/registry.py)，第 134 行。

```python
resolve_level(level: str)
```

Map ``level`` onto the tables that must be filled.

## `select_names`

源文件：[features/registry.py](features/registry.py)，第 144 行。

```python
select_names(names: Iterable[str], include: Sequence[str] | None=None, exclude: Sequence[str] | None=None)
```

Filter full column names with glob patterns.

Patterns are matched against the prefixed name, e.g. ``"cnn.cn_*"``.
``include`` keeps matches only; ``exclude`` then drops matches.

## `available_sets`

源文件：[features/registry.py](features/registry.py)，第 171 行。

```python
available_sets()
```

Describe every registered set, including whether it can run right now.

## `sanitize_for_json`

源文件：[Get_Oct_distortion.py](Get_Oct_distortion.py)，第 8 行。

```python
sanitize_for_json(df)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `analyze_octahedra_distortion`

源文件：[Get_Oct_distortion.py](Get_Oct_distortion.py)，第 25 行。

```python
analyze_octahedra_distortion(struct, TM_elements=['Fe', 'Mn'], cutoff_O=2.5, cutoff_TM1=3.5, cutoff_TM2=5.5, ligands_max_dist=3.0)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `StructureFeatureExtractor`

源文件：[Get_structfeatures_all.py](Get_structfeatures_all.py)，第 27 行。

```python
StructureFeatureExtractor
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `StructureFeatureExtractor.__init__`

源文件：[Get_structfeatures_all.py](Get_structfeatures_all.py)，第 104 行。

```python
StructureFeatureExtractor.__init__(self, global_features: str | list[str] | None='all', local_features: str | list[str] | None='all', ligand: str='O', tm_elements: list[str] | None=None, na_species: tuple=('Na',), tm_species: tuple=('Mn', 'Fe', 'Co', 'Ni', 'Cu'), layer_axis: int=2, layer_tol: float=0.08, ligands_max_dist: float=3.0)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `StructureFeatureExtractor.cnn`

源文件：[Get_structfeatures_all.py](Get_structfeatures_all.py)，第 156 行。

```python
StructureFeatureExtractor.cnn(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `StructureFeatureExtractor.extract`

源文件：[Get_structfeatures_all.py](Get_structfeatures_all.py)，第 163 行。

```python
StructureFeatureExtractor.extract(self, structure: Structure, supercell_matrix=None)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `StructureFeatureExtractor.extract_features`

源文件：[Get_structfeatures_all.py](Get_structfeatures_all.py)，第 295 行。

```python
StructureFeatureExtractor.extract_features(cls, structure: Structure, **kwargs)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `analyze_percolation`

源文件：[percolation.py](percolation.py)，第 167 行。

```python
analyze_percolation(structure: Structure | str | Path, migrant: str | None=None, *, cutoff_radius: float, output_dir: str | Path | None=None)
```

Return geometric channel connectivity at a probe radius in Å.

``structure`` accepts a pymatgen Structure or any file format it can read.
An empty ``migrant`` keeps all atoms and measures the original void network.
A specified element is removed by CCNB before channel analysis. The
returned dimension is 0, 1, 2, or 3; it is not a migration energy barrier.
Generated files are retained under ``output_dir`` when that is provided.
On Windows, transient locks on generated files are retried during cleanup;
a persistent cleanup failure is returned in ``cleanup_warning`` together
with the remaining temporary directory path.

## `estimate_spatial_capacity`

源文件：[spatial_capacity.py](spatial_capacity.py)，第 54 行。

```python
estimate_spatial_capacity(structure: Structure | str | Path, mobile_ion: str='Na', *, cutoff_radius: float=0.8, min_site_distance_A: float=2.0, output_dir: str | Path | None=None, verbose: bool=True)
```

Estimate feasible loading of CCNB percolating-channel sites in mAh/g.

Input: ordered Structure or readable structure file; Li/Na is removed by
CCNB. First find channels at cutoff_radius (angstrom), then use only nodes
in their 1D/2D/3D percolating components, with free radius >= cutoff_radius.
Isolated voids and 0D components are excluded; no percolating channel means
zero capacity. Repeated periodic positions are counted once. Bottleneck
points are not storage sites; equivalent labels do not collapse different
positions. Pair distances include periodic boundaries and must be >=
min_site_distance_A (default 2 angstrom).

One deterministic radius-priority greedy pass returns a feasible site count,
using distance rows instead of a full matrix. It is NOT a proven
maximum or an experimental reversible capacity. Each selected site holds
one monovalent Li/Na. Default mass basis is framework + selected ions:
Q = 26801.481145 * N / (M_framework + N * M_ion), using the input cell's
composition and site count. Input-mass and framework-mass bases are also
returned. Geometric percolation is required, but charge balance, site energy
and migration barriers are not evaluated.

Run in ccnb. No random seed is used. CCNB writes a unique temporary folder;
default cleanup preserves the returned data on lock failure. output_dir
retains a new unique folder each run; existing results are not overwritten.
verbose prints elapsed CCNB and packing stages; set False to silence them.

## `get_supercell_matrices`

源文件：[supercell.py](supercell.py)，第 8 行。

```python
get_supercell_matrices(structure, size: int, dimensions: str='abc', *, symprec: float=1e-05)
```

Return inequivalent matrices and a reduced matrix for each one.

Args:
    structure: Ordered pymatgen Structure or ASE Atoms, preferably a
        primitive cell. The input cell is used as supplied, without
        primitive-cell conversion. Its original PBC flags are not used.
    size: Positive integer volume multiplier relative to the input cell,
        not the final atom count.
    dimensions: Nonempty subset of "abc", e.g. "ab", "ac", or "abc".
        These are lattice directions, not Cartesian axes. Unselected
        directions remain unchanged and are never mixed into selected ones.
    symprec: icet/spglib symmetry tolerance in angstrom (default 1e-5).

Returns:
    Dict with two equally long lists of integer (3, 3) numpy arrays:
    ``all`` contains icet's symmetry-inequivalent HNF representatives;
    ``recommended[i]`` is a Minkowski-reduced basis of ``all[i]``, using
    only selected directions. Both use ``new_cell = matrix @ old_cell``
    and work with pymatgen ``make_supercell`` or ASE ``make_supercell``.
    Reduction shortens basis vectors; it does not select a globally most
    cubic cell or rank the inequivalent supercells.

Notes:
    Requires numpy, ASE and icet in py1; pymatgen only for Structure input.
    Equivalence uses the input's atomic species, positions and symmetry,
    not magnetic moments. Supply an ordered structure. No files are
    written and the input is not modified. No randomness is used; repeated
    calls with the same input and library versions give the same order.
    Enumeration can grow rapidly with size, especially in three dimensions.

## `DeduplicationTests`

源文件：[test_deduplication.py](test_deduplication.py)，第 8 行。

```python
DeduplicationTests
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `DeduplicationTests.test_compatibility_and_order`

源文件：[test_deduplication.py](test_deduplication.py)，第 9 行。

```python
DeduplicationTests.test_compatibility_and_order(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `rocksalt_nacl`

源文件：[test_features.py](test_features.py)，第 22 行。

```python
rocksalt_nacl()
```

Eight-site NaCl with oxidation states so CrystalNN finds six neighbours.

## `layered_licoo2`

源文件：[test_features.py](test_features.py)，第 33 行。

```python
layered_licoo2()
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests`

源文件：[test_features.py](test_features.py)，第 42 行。

```python
FeatureExtractionTests
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.setUpClass`

源文件：[test_features.py](test_features.py)，第 44 行。

```python
FeatureExtractionTests.setUpClass(cls)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_top_level_export_is_the_same_function`

源文件：[test_features.py](test_features.py)，第 48 行。

```python
FeatureExtractionTests.test_top_level_export_is_the_same_function(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_default_sets_are_geometry_only`

源文件：[test_features.py](test_features.py)，第 51 行。

```python
FeatureExtractionTests.test_default_sets_are_geometry_only(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_crystalnn_counts_six_neighbours_for_rocksalt`

源文件：[test_features.py](test_features.py)，第 60 行。

```python
FeatureExtractionTests.test_crystalnn_counts_six_neighbours_for_rocksalt(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_site_level_rows`

源文件：[test_features.py](test_features.py)，第 68 行。

```python
FeatureExtractionTests.test_site_level_rows(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_structure_only_level_has_no_sites`

源文件：[test_features.py](test_features.py)，第 78 行。

```python
FeatureExtractionTests.test_structure_only_level_has_no_sites(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_include_and_exclude`

源文件：[test_features.py](test_features.py)，第 82 行。

```python
FeatureExtractionTests.test_include_and_exclude(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_non_sodium_structure_does_not_raise`

源文件：[test_features.py](test_features.py)，第 96 行。

```python
FeatureExtractionTests.test_non_sodium_structure_does_not_raise(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_metadata_records_structure_and_sets`

源文件：[test_features.py](test_features.py)，第 100 行。

```python
FeatureExtractionTests.test_metadata_records_structure_and_sets(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_load_structure_accepts_poscar`

源文件：[test_features.py](test_features.py)，第 107 行。

```python
FeatureExtractionTests.test_load_structure_accepts_poscar(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_batch_isolates_a_bad_entry`

源文件：[test_features.py](test_features.py)，第 122 行。

```python
FeatureExtractionTests.test_batch_isolates_a_bad_entry(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_batch_returns_site_frame`

源文件：[test_features.py](test_features.py)，第 128 行。

```python
FeatureExtractionTests.test_batch_returns_site_frame(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_mlip_requires_a_model_path`

源文件：[test_features.py](test_features.py)，第 136 行。

```python
FeatureExtractionTests.test_mlip_requires_a_model_path(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_unavailable_optional_set_raises_import_error`

源文件：[test_features.py](test_features.py)，第 140 行。

```python
FeatureExtractionTests.test_unavailable_optional_set_raises_import_error(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `FeatureExtractionTests.test_available_sets_reports_prefix_and_availability`

源文件：[test_features.py](test_features.py)，第 146 行。

```python
FeatureExtractionTests.test_available_sets_reports_prefix_and_availability(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MaceDescriptorTests`

源文件：[test_features.py](test_features.py)，第 158 行。

```python
MaceDescriptorTests
```

Per-layer and per-element MACE descriptor selection (real checkpoint).

## `MaceDescriptorTests.setUpClass`

源文件：[test_features.py](test_features.py)，第 162 行。

```python
MaceDescriptorTests.setUpClass(cls)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MaceDescriptorTests.test_first_layer_only`

源文件：[test_features.py](test_features.py)，第 183 行。

```python
MaceDescriptorTests.test_first_layer_only(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MaceDescriptorTests.test_second_layer_only_differs_from_first`

源文件：[test_features.py](test_features.py)，第 189 行。

```python
MaceDescriptorTests.test_second_layer_only_differs_from_first(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MaceDescriptorTests.test_all_layers_concatenate`

源文件：[test_features.py](test_features.py)，第 197 行。

```python
MaceDescriptorTests.test_all_layers_concatenate(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `MaceDescriptorTests.test_element_mean_emits_one_block_per_element`

源文件：[test_features.py](test_features.py)，第 202 行。

```python
MaceDescriptorTests.test_element_mean_emits_one_block_per_element(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `M3GNetEncoderTests`

源文件：[test_features.py](test_features.py)，第 217 行。

```python
M3GNetEncoderTests
```

The M3GNet checkpoint path, loaded once per class.

## `M3GNetEncoderTests.setUpClass`

源文件：[test_features.py](test_features.py)，第 221 行。

```python
M3GNetEncoderTests.setUpClass(cls)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `M3GNetEncoderTests.test_structure_vector_has_64_named_columns`

源文件：[test_features.py](test_features.py)，第 235 行。

```python
M3GNetEncoderTests.test_structure_vector_has_64_named_columns(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `M3GNetEncoderTests.test_mace_only_options_are_rejected`

源文件：[test_features.py](test_features.py)，第 243 行。

```python
M3GNetEncoderTests.test_mace_only_options_are_rejected(self)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `save_traj_xyz`

源文件：[Traj_Convert.py](Traj_Convert.py)，第 6 行。

```python
save_traj_xyz(traj, save_name, include_e=True, include_f=False, include_s=False)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `traj_to_info`

源文件：[Traj_Convert.py](Traj_Convert.py)，第 26 行。

```python
traj_to_info(traj_file, include_e=True, include_f=False, include_s=False)
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `Octahedron`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 21 行。

```python
Octahedron
```

原源码尚未说明输入/返回含义；执行前阅读实现，勿根据函数名猜测。

## `Octahedron.__init__`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 33 行。

```python
Octahedron.__init__(self, core_site, structure, ligands_min_distance=0.0, ligands_max_distance=2.7, forbidden_ligands=[], possible_ligands=None)
```

The initialisation of an instance of the Octahedron class.

Arguments:
-core_site: the atom in the centre (pymatgen PeriodicSite object)
-structure: a pymatgen Structure object corresponding to the unit  
  cell containing the core_site
-ligands_min_distance: an optional argument (defaults to 0.0 if not 
  includes) which gives the minimum distance from core_site (in 
  Angstroms) for atoms to be considered possible ligands.
-ligands_max_distance: an optional argument (defaults to 2.7 if not
  included) which gives the maximum distance from core_site (in 
  Angstroms) for atoms to be considered possible ligands.
-forbidden_ligands: an optional argument which lists the names of 
  species which specifically cannot be considered as ligands; for 
  instance, if the core_site is "Ni" you may which to exclude "Ni" 
  from being considered as a possible ligand. If not included, 
  defaults to an empty list.
-possible_ligands: an optional argument which lists atom types 
  which can be considered Ligands. Defaults to a None value in 
  which case all species which are not listed in forbidden_ligands
  are considered as possible ligands.

## `Octahedron.calculate_core_ligand_distance_for_perfect_octahedron`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 263 行。

```python
Octahedron.calculate_core_ligand_distance_for_perfect_octahedron(self)
```

This method will calculate the core-ligand distance for a perfect
  octahedron of equal volume to the octahedron.

A perfect octahedron can be split into two square-based pyramids 
  with base area a.
  Hence the volume is V=sqrt(2)*a^3/3
  from Pythagoras's theorem, 2a^2 = (2l_0)^2 so a = sqrt(2)*l_0
  therefore l_0 = (6*V)^(1/3)

return:
  float: the core-ligand distance for a perfect octahedron of 
    equal volume to the octahedron, in Angstroms.

## `Octahedron.calculate_quadratic_elongation`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 281 行。

```python
Octahedron.calculate_quadratic_elongation(self)
```

This method calculates the quadratic elongation of the octahedron, 
  as defined in:
    Robinson, Keith, G. V. Gibbs, and P. H. Ribbe.
    "Quadratic elongation: a quantitative measure of distortion 
    in coordination polyhedra."
    Science 172.3983 (1971): 567-570.

Return:
  float: the quadratic elongation of the octahedron.

## `Octahedron.get_intersite_distance`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 301 行。

```python
Octahedron.get_intersite_distance(self, site1, site2, override_lattice_check=False)
```

This method will give the distance between two sites. Pymatgen 
  does include functionality to do this, but it does not work well 
  when it encounters periodicity, i.e. if a neighbour is in a 
  neighbouring unit cell. This custom method should therefore be 
  used.

Arguments:
  site1: a Pymatgen PeriodicSite object
  site2: a Pymatgen PeriodicSite object
  override_lattice_check: bool. optional, defaults to False. If 
    True, there will be no check to make sure site1 and site2 are 
    in the same cell.

Return: 
  float: the distance (in Angstroms) between the two sites.

## `Octahedron.get_neighbours`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 338 行。

```python
Octahedron.get_neighbours(self, ligands_min_distance=0.0, ligands_max_distance=2.7, forbidden_ligands=[], possible_ligands=None)
```

This method will list all ligands, in ascending order of bond 
  length from the core_site, provided they meet the criteria 
  given in the __init__ method.

Arguments (all optional):
  ligands_min_distance (float): the minimum distance of a site to 
    be considered a neighbour (in Angstroms)
  ligands_max_distance (float) the maximum distance of a site to 
    be considered a neighbour (in Angstroms)
  forbidden_ligands (list): a list of strings or site species which 
    may not be considered neighbours
  possible_ligands (list or None): a list of strings or site 
    species which only may be considered neighbours. Defaults to 
    None, meaning no restriction
  
Return:
  Python list: containing PeriodicSite objects

## `Octahedron.calculate_average_ligand_bond_length`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 421 行。

```python
Octahedron.calculate_average_ligand_bond_length(self, octahedral_centre='core_atom')
```

Calculates the average bond length between the centre of the 
  octahedron and its ligands.

Arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.

Returns:
  float: The average bond length.

## `Octahedron.calculate_bond_length_distortion_index`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 495 行。

```python
Octahedron.calculate_bond_length_distortion_index(self, octahedral_centre='core_atom')
```

Calculates and returns the bond length distortion index of a 
  polyhedron as defined in Baur (1974).
The bond length distortion index is calculated as the average 
  absolute deviation of individual centre-ligand bond lengths 
  from the average bond length, normalized by the average
  bond length.

Reference:
  Baur, W. H. "The geometry of polyhedral distortions. Predictive 
  relationships for the phosphate group." Acta Crystallographica 
  Section B: Structural Crystallography and Crystal Chemistry,
  1974, 30(5), 1195-1215.

Arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.

Returns:
  float: the bond length distortion index.

## `Octahedron.get_bond_angle`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 574 行。

```python
Octahedron.get_bond_angle(self, ligand1, ligand2, degrees=True)
```

Calculates the angle between the bonds of two ligand ions 
  to the core atom.

Arguments:
    ligand1 (PeriodicSite): the first ligand ion
    ligand2 (PeriodicSite): the second ligand ion
    degrees: if True, output is in degrees. Otherwise output 
      is radians. Defaults to True

Returns:
    float: the angle between the bonds in degrees

## `Octahedron.list_all_ligand_core_ligand_angles`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 628 行。

```python
Octahedron.list_all_ligand_core_ligand_angles(self, degrees=True)
```

Returns an array containing all bond angles (including bond angles 
  between opposite ligands).

Arguments:
  degrees (bool): an optional argument, defaults to True. 
    Calculated values are degrees (True) or radians (False)

## `Octahedron.calculate_bond_angle_variance`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 654 行。

```python
Octahedron.calculate_bond_angle_variance(self, degrees=True)
```

Calculates the bond angle variance, as defined in
    Baur, W. H. "The geometry of polyhedral distortions. 
    Predictive relationships for the phosphate group."
    Acta Crystallographica Section B: Structural Crystallography 
    and Crystal Chemistry 30.5 (1974): 1195-1215.

Arguments:
  degrees (bool): an optional argument, defaults to True. 
    Calculated value is degrees (True) or radians (False)

return:
  float: the bond angle variance, in units of either degrees or
    radians depending on user input

## `Octahedron.calculate_effective_coordination_number`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 687 行。

```python
Octahedron.calculate_effective_coordination_number(self)
```

This method calculates the effective coordination number of the 
  octahedron as first defined in the following paper:
    Hoppe, Rudolf. "Effective coordination numbers (ECoN) and 
    mean fictive ionic radii (MEFIR)." Zeitschrift für 
    Kristallographie-Crystalline Materials 150.1-4 (1979): 23-52.

return:
  float: the effective coordination number

## `Octahedron.calculate_off_centering_distance`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 715 行。

```python
Octahedron.calculate_off_centering_distance(self)
```

This method calculates the off-centering distance defined in Eq 7 
  of:
    Koçer, Can P., et al.
    "Cation disorder and lithium insertion mechanism of 
    Wadsley–Roth crystallographic shear phases from first 
    principles."
    Journal of the American Chemical Society 141.38 
    (2019): 15121-15134.

return:
  float: the off-centering distance (in Angstroms)

## `Octahedron.calculate_surface_area`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 951 行。

```python
Octahedron.calculate_surface_area(self)
```

Octahedral surface area is defined by splitting up the surface area
  into a set of triangles and summing the area of these triangles, 
  following the method described in:
    Swanson, Donald K., and R. C. Peterson. "Polyhedral volume 
    calculations."
    The Canadian Mineralogist 18.2 (1980): 153-156.

return:
  float: surface area of the exterior triangles of the Octahedron 
    (in Angstroms squared)

## `Octahedron.calculate_volume`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 986 行。

```python
Octahedron.calculate_volume(self)
```

This method returns the volume of the octahedron. It first attempts
  to split the octahedron into 8 tetrahedra. If this fails, it uses
  the more generalised approach given in:
    Swanson, Donald K., and R. C. Peterson. "Polyhedral volume 
    calculations."
    The Canadian Mineralogist 18.2 (1980): 153-156.

return:
  float: octahedral volume in units of Angstrom cubed

## `Octahedron.is_identical`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 1055 行。

```python
Octahedron.is_identical(self, octahedron)
```

This method checks to see whether a given octahedron object is 
  identical to self.

argument:
  an instance of the Octahedron class, or a subclass

return: 
  boolean: True, if the Octahedron object is 
    identical

## `Octahedron.visualise_pairs`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 1117 行。

```python
Octahedron.visualise_pairs(self, plotted_vectors=None)
```

This method plots a 3D, interactive figure which shows the 
  positions of the pairs relative to the origin.
It can also plot axes alongside the atomic sites if plotted_vectors
  is included.

argument:
  plotted_vectors: must be a python list or None. If list, must be 
    a list with shape (3,3), representing three vectors which may be
    supplied and will be plotted alongside the points. Defaults to 
    None.

## `Octahedron.calculate_off_centering_metric`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 1197 行。

```python
Octahedron.calculate_off_centering_metric(self)
```

This method calculates the off-centering metric defined in 
  page 3588 of:
    Halasyamani, P. Shiv.
    "Asymmetric cation coordination in oxide materials:
    Influence of lone-pair cations on the intra-octahedral 
    distortion in d0 transition metals."
    Chemistry of Materials 16.19 (2004): 3586-3592.

return:
  float: the value of the off-centering metric 

## `Octahedron.find_ligand_opposites`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 1427 行。

```python
Octahedron.find_ligand_opposites(self)
```

Returns 3 arrays each containing two pymatgen PeriodicSite objects. 
  These are the two opposite ligands in the octahedra (i.e. have a 
  bond length 180 degrees through the central atom). It does this 
  by assuming opposite bonds will have the largest angle via the 
  central atom.

return:
  a list of shape (3,2), where the 3 are the three pairs of 
    opposite ligands and the 2 is each ligand within a pair.

## `Octahedron.calculate_van_vleck_distortion_modes`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2373 行。

```python
Octahedron.calculate_van_vleck_distortion_modes(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, output_pairs=False, ignore_angular_distortion=False, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method calculates the magnitude of the Q_1 to Q_6 distortion 
  modes, as defined by Van Vleck:
    Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark
    Splitting for Clusters of the Form XY6." The Journal of 
    Chemical Physics 7.1 (1939): 72-84.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  ignore_angular_distortion: if True, no rotation operations will 
    be performed, and the van Vleck modes will be calculated along 
    bond lengths without regard to their angles. 
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  output_pairs: bool. if True, the code gives as a return the pairs 
    list. Defaults to False.

return:
  -a list containing the 6 Van Vleck modes
  -[if output_pairs]: a 2D list containing the coordinates of all sites after axis transformation

## `Octahedron.calculate_degenerate_Q3_van_vleck_modes`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2499 行。

```python
Octahedron.calculate_degenerate_Q3_van_vleck_modes(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, output_pairs=False, ignore_angular_distortion=False, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method outputs the magnitude of the Q_3 distortion mode 
  and its two symmetrically identical linear combinations of Q_2 
  and Q_3:
    -1/2 Q_3 ± √3/2 Q_2
  where Q_2 and Q_3 are as defined by Van Vleck:
    Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark
      Splitting for Clusters of the Form XY6." The Journal of 
      Chemical Physics 7.1 (1939): 72-84.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  ignore_angular_distortion: if True, no rotation operations will 
    be performed, and the van Vleck modes will be calculated along 
    bond lengths without regard to their angles. 
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. 
  output_pairs: bool. if True, the code gives as a return the pairs 
    list. Defaults to False.

## `Octahedron.calculate_van_vleck_jahn_teller_params`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2591 行。

```python
Octahedron.calculate_van_vleck_jahn_teller_params(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], degrees=False, suppress_warnings=False, automatic_rotation=True, output_pairs=False, ignore_angular_distortion=False, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method outputs sqrt(Q_3^2 + Q_2^2) and arctan(Q_2/Q_3), which 
  can be used to generate a polar plot as in, for example:
    Zhou, J-S., et al. "Jahn–Teller distortion in perovskite 
    KCuF3 under high pressure."
    Journal of Fluorine Chemistry 132.12 (2011): 1117-1121.
    Figure 4.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  ignore_angular_distortion: if True, no rotation operations will 
    be performed, and the van Vleck modes will be calculated along 
    bond lengths without regard to their angles. 
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  output_pairs: bool. if True, the code gives as a return the pairs 
    list. Defaults to False.
  degrees: bool. Defaults to False. If True, phi is returned in 
    degrees, otherwise it is in radians.

Return:
  -a list containing the params: magnitude and angle
  - [if output_pairs]: a list containing the atomic positions 
    within a list with shape (3,2,3)

## `Octahedron.output_sites_for_van_vleck`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2710 行。

```python
Octahedron.output_sites_for_van_vleck(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], automatic_rotation=True, suppress_warnings=False, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method performs rotation of the ligands around a central point 
  and returns their positions. The purpose of this is for checking 
  that the ligand rotation for van Vleck calculation is working
  correctly.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.

## `Octahedron.visualise_sites_for_van_vleck`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2774 行。

```python
Octahedron.visualise_sites_for_van_vleck(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], automatic_rotation=True, output_pairs=False, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD)
```

This method performs rotation of the ligands around a central point
  and plots their positions. The purpose of this is for checking 
  that the ligand rotation for van Vleck calculation is working
  correctly.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  output_pairs: if True, the code gives as a return the pairs 
    list. Defaults to False.

Action:
  plots 3D interactive plot showing the positions of the sites, to 
    check whether they are lose

## `Octahedron.output_and_visualise_sites_for_van_vleck`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2831 行。

```python
Octahedron.output_and_visualise_sites_for_van_vleck(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD)
```

This method performs rotation of the ligands around a central 
  point and plots their positions. The purpose of this is for 
  checking that the ligand rotation for van Vleck calculation 
  is working correctly. It also returns a Python list with 
  shape (3,2,3) containing the coordinates of each site after 
  the rotation.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.

Action:
  plots 3D interactive plot showing the positions of the sites, to 
    check whether they are lose

## `Octahedron.calculate_angular_shear_modes`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 2958 行。

```python
Octahedron.calculate_angular_shear_modes(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False, degrees=False)
```

This method calculates the angular shear modes, Delta ab, 
  Delta ac, and Delta bc.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  degrees: bool. If True, returns units of degrees, otherwise 
    radians. Defaults to False.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.

return:
  list: a list of the three angular shear values, in units of
    degrees or radians depending on user-supplied arguments.

## `Octahedron.calculate_angular_antishear_modes`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 3052 行。

```python
Octahedron.calculate_angular_antishear_modes(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False, degrees=False)
```

This method calculates the angular anti-shear modes, Delta' ab, 
  Delta' ac, and Delta' bc.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  degrees: bool. If True, returns units of degrees, otherwise 
    radians. Defaults to False.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.

return:
  list: a list of the three angular anti-shear values, in units of
    degrees or radians depending on user-supplied arguments.

## `Octahedron.calculate_angular_shear_magnitude`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 3145 行。

```python
Octahedron.calculate_angular_shear_magnitude(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method calculates the Delta_shear parameter.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.

return:
  float: the square root of the sum of the angular shear 
    parameters for the three planes. In units of
    degrees or radians depending on user-supplied arguments.

## `Octahedron.calculate_angular_antishear_magnitude`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 3228 行。

```python
Octahedron.calculate_angular_antishear_magnitude(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method calculates the Delta_antishear parameter.

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.

return:
  float: the square root of the sum of the angular antishear 
    parameters for the three planes. In units of
    degrees or radians depending on user-supplied arguments.

## `Octahedron.calculate_shear_fraction_angular_distortion`

源文件：[van_vleck_calculator.py](van_vleck_calculator.py)，第 3311 行。

```python
Octahedron.calculate_shear_fraction_angular_distortion(self, octahedral_centre='core_atom', specified_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], suppress_warnings=False, automatic_rotation=True, rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD, omit_failed_rotations=False)
```

This method calculates a parameter which represents the fraction 
  of angular distortion which stems from octahedral shear

arguments:
  octahedral_centre (string or list): if "core_atom", the centre 
    of the octahedron is the position of the core atom. If 
    "average_ligand_position" then the octahedral centre
    is the average position of the ligands. If list, must be 
    a list of three floats which are the coordinates to be taken
    as the centre of the octahedron.
  specified_axes must be a list containing three lists, with each 
    sub-list containing three elements. These are three vectors 
    which will be set as the axes for the Van Vleck calculation.  
    They must be perpendicular to eachother.
    For example: [[1,0,0],[0,1,0],[0,0,1]] 
      or [[1,0,1],[1,0,-1],[0,1,0]]
  automatic_rotation: bool. Defaults to True. If True, the 
    automated rotation of the octahedron will occur. If False, it
    will not.
  rotation_tolerance: float, in units of Angstroms. If an atom is
    more than this distance from its assigned axis, the rotation
    is considered to have failed.
  omit_failed_rotations: bool. If True, when rotation fails, 
    returns only NoneType output. Defaults to False.
  suppress_warnings: bool. if True, will suppress warnings to 
    console. Defaults to False.

return:
  float: the shear fraction, eta

## `WyckoffSiteGroup`

源文件：[wyckoff.py](wyckoff.py)，第 9 行。

```python
WyckoffSiteGroup
```

One group of symmetry-equivalent sites.

## `get_wyckoff_sites`

源文件：[wyckoff.py](wyckoff.py)，第 18 行。

```python
get_wyckoff_sites(structure: Structure, symprec: float=0.01, angle_tolerance: float=5.0)
```

Return the element, Wyckoff label, and sites for each equivalent group.

Args:
    structure: Structure to analyze.
    symprec: Cartesian distance tolerance used to determine symmetry, in angstrom.
    angle_tolerance: Angle tolerance used to determine symmetry, in degrees.

Returns:
    A list of dictionaries. Each dictionary describes one symmetry-equivalent
    group and contains ``element``, ``wyckoff``, ``indices``, and ``sites``.
    ``sites`` contains the corresponding sites from the input structure.

Raises:
    ValueError: If the structure is empty or a tolerance is not positive.

