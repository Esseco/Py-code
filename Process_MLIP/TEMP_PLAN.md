# NEPkit layered-oxide initialization plan

Goal: Add a local NEPkit perturbation dataset builder and a separate phase-agent action that can initialize a layered-oxide active-learning round from boundary conditions.

Architecture: Process_MLIP receives structures plus explicit phase/Na/SOC boundary metadata, generates allowed supercells with det(H)<=6 and 36 combined atom/cell perturbations per parent including the parent frame, applies deterministic geometry filtering, and packages structures plus manifest for remote execution. The phase agent exposes a separate initialization action; remote consumes the package, performs NEP relaxation (fmax=0.1 eV/A), geometry filtering, M3GNet features, DIRECT sampling (default 100), DFT, and fine-tuning/QBC through existing lifecycle contracts.

Tech Stack: Python 3, ASE, existing Process_MLIP NEP/relaxation/direct_sampling APIs, phase_agent action/state contracts, JSON/extended XYZ or ASE trajectory manifests.

## Global Constraints
- Default environment is py1.
- Do not alter existing public interfaces or scientific meaning.
- det(H) <= 6.
- Perturbations: a/b +/-1%, c +/-2%; Na/TM/O displacement scales 0.12/0.05/0.08 A.
- Each parent contributes 36 frames including the unperturbed parent.
- Default DIRECT target is 100 and must remain configurable.
- Real project data and HPC actions remain read-only/unsubmitted during development.

## Tasks
1. Add Process_MLIP layered-oxide perturbation module with explicit config, seed, structure metadata, supercell enumeration, 36-frame generation, geometry filter, and manifest/package output.
2. Add unit tests using small synthetic ASE structures for det bound, reproducibility, perturbation counts, parent inclusion, geometry rejection, and manifest fields.
3. Add separate phase_agent initialization action and validation contract that accepts boundary/branch inputs and emits a remote preparation request without invoking existing branch generation.
4. Wire initial configuration so the action can be enabled, while preserving default existing workflow behavior when disabled.
5. Add remote lifecycle adapter for package handoff and downstream stage names, reusing existing relax/feature/sampling/DFT/fine-tune/QBC actions.
6. Update API/README/docs and run targeted tests in both repositories.

## Review Focus
- Invalid or missing boundary conditions must fail explicitly.
- det(H)>6 must never be emitted.
- Re-running with the same seed and inputs must produce identical manifest identities.
- Geometry filtering must reject collapsed cells/bonds without deleting source inputs.
- Remote handoff must preserve branch/phase/x/SOC identity and not bypass approval gates.
