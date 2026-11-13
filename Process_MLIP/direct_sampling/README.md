# DIRECT Sampling

Defaults to the original M3GNet formation-energy structure encoder from `maml.describers.M3GNetStructure`, followed by maml's Kaiser PCA weighted by explained-variance ratio, BIRCH and center-based stratified sampling (k=1). The original M3GNet checkpoint directory must be specified explicitly:

```
python -m Process_MLIP.direct_sampling extract md.traj --model /path/to/m3gnet_checkpoint_directory --cache features.h5
python -m Process_MLIP.direct_sampling sample --cache features.h5 --output selected --n-samples 100
```

Windows py1 setup tested: install tf-keras==2.20.1 separately (TensorFlow 2.20), set TF_USE_LEGACY_KERAS=1 before launching Python, and include the tf-keras installation directory in PYTHONPATH. The existing Keras 3 cannot load the legacy checkpoint. On Windows the module scopes a neighbor-list int64 compatibility wrapper to M3GNet extraction; do not run concurrent M3GNet extraction with unrelated m3gnet users in the same process. TensorFlow controls M3GNet device visibility; the MACE-only --device and --head flags are rejected for M3GNet. Source and model hashes are saved; inputs must remain unchanged. A new output directory is required. Exact requested counts depend on enough distinct features. The center-selection method is deterministic; seed has no effect.

Descriptor fallback: maml 2025.4.3 still reaches the pre-readout features through `describer_model.layers[-2].layers[0]`. With m3gnet 0.2.4 that slot holds a graph-network block, so the call raises `ValueError: setting an array element with a sequence` (the ragged 17-field graph tuple) for both `extract` and `pca`. `M3GNetEmbedding` keeps the maml path whenever it works and otherwise replays the model's own sublayers (`graph_converter` -> `featurizer` -> `n_blocks` x (`three_interactions`, `graph_layers`)) and mean-pools the final 64-dimensional node embeddings over the atoms of the cell. The route is recorded as `descriptor` in `report["extractor"]` and in the encoder metadata (`mean_pooled_node_features`). That vector is a structure-level embedding for coverage and 2D projection, not the readout the paper used, so do not compare its numbers with published formation energies. Verified on the bundled checkpoint: 64 features per structure and `extract` writes a `features (n, 64)` dataset.
MACE alternative: `extract md.traj --encoder mace --model /path/to/model.model --device cuda --cache features.h5`. Its structure vector is a mean of contextual invariant atom descriptors and differs from the paper's M3GNet graph readout. Only load trusted model weights.

Outputs: selected.extxyz, selected_indices.json and sampling_report.json.
PCA projection to a two-dimensional CSV: give a batch of structures and get one CSV row per structure with `pc1`/`pc2`.

```
python -m Process_MLIP.direct_sampling pca frames/*.vasp --model /path/to/m3gnet_checkpoint_directory --output pca.csv
```

The same entry from Python:

```python
from Process_MLIP.direct_sampling import structures_to_pca_csv

report = structures_to_pca_csv(structures, "pca.csv", model="/path/to/checkpoint", labels=groups)
```

`structures` accepts ASE `Atoms`, pymatgen `Structure` and structure file paths; a trajectory path contributes one row per frame, and `source`/`frame_index` in the CSV point back to it. `labels` adds an optional grouping column for colouring a 2D plot. The projection is StandardScaler + sklearn PCA and is deliberately independent of the DIRECT Kaiser PCA used by `sample`, so the two component sets need not agree. The CSV holds `structure_index, source, frame_index, label, pc1, pc2`; explained variance ratios are returned in the report and printed by the command line instead of being replicated on every row. An existing CSV is not replaced unless `overwrite=True`. This is a reporting aid: it neither selects nor rewrites structures, and the full feature matrix is held in memory, so sample a long trajectory first.
 Extraction streams frames, but PCA and clustering load the feature matrix in memory. Million-frame scaling, GPU performance, automatic benchmark and coverage plots have not been validated or implemented. Existing uniform sampling remains `export_traj_frames(..., uniform_count=100)`.

For a historical reference set, extract historical structures first, followed by
new structures, and call `run(cache, output, n_samples, reference_count=N)` with
the number of historical **frames**. DIRECT fits the combined feature space;
clusters containing a reference frame are considered covered and are omitted
from the output. Only new cluster centers are returned, at most 100, and fewer
may be returned. A zero-result run has an empty mapping and no selected.extxyz.
This is feature-space coverage, not a proof of crystallographic identity. The
reference-aware path reports an error if DIRECT cannot form distinct clusters.

Tested 110 synthetic Cu frames to 100 outputs using the original bundled formation-energy checkpoint on CPU; checkpoint variables used by the structural readout were checked after initialization. TensorFlow 2.20 emits complex-to-float cast warnings; their scientific impact has not been independently benchmarked.
