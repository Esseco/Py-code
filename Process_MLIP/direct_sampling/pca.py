"""Standalone PCA projection of a batch of structures to a 2D CSV.

Encodes every structure with a structure encoder (default: the M3GNet
formation-energy descriptor from :mod:`.features`), projects the feature matrix
with StandardScaler + PCA, and writes one CSV row per structure so the result can
be plotted as pc1 vs pc2 without any further processing.

The encoder is injectable, so the projection can be tested or reused with any
callable that maps one ASE Atoms object to a 1D feature vector.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

IDENTITY_COLUMNS = ("structure_index", "source", "frame_index", "label")


def _as_atoms(item):
    from ase import Atoms

    if isinstance(item, Atoms):
        return item
    try:
        from pymatgen.core import Structure
    except ImportError:  # pragma: no cover - pymatgen is a project dependency
        raise TypeError("structures must be ASE Atoms or pymatgen Structure objects") from None
    if isinstance(item, Structure):
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_atoms(item)
    raise TypeError("structures must be ASE Atoms, pymatgen Structure or a structure file path")


def _iter_structures(structures):
    """Yield (source, frame_index, atoms); a file path may contribute many frames."""
    from ase import Atoms
    from ase.io import read

    if isinstance(structures, (str, Path)):
        structures = [structures]
    memory_index = 0
    for item in structures:
        if isinstance(item, (str, Path)):
            path = Path(item).resolve(strict=True)
            frames = read(str(path), index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            for frame_index, atoms in enumerate(frames):
                yield str(path), frame_index, atoms
        else:
            yield "<memory>", memory_index, _as_atoms(item)
            memory_index += 1


def _prepare_output(output_csv, overwrite):
    output = Path(output_csv).resolve()
    if output.is_symlink() or output.is_dir():
        raise FileExistsError(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def structures_to_pca_csv(
    structures,
    output_csv,
    *,
    extractor=None,
    model=None,
    encoder="m3gnet",
    device="cpu",
    head=None,
    n_components=2,
    labels=None,
    overwrite=False,
    progress=None,
):
    """Encode structures, project them on principal components, write one CSV.

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
    """
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if extractor is None:
        if model is None:
            raise ValueError("provide model=... or an explicit extractor=")
        from .features import build_encoder

        extractor = build_encoder(model, encoder, device=device, head=head)

    output = _prepare_output(output_csv, overwrite)
    identity = []
    vectors = []
    for source, frame_index, atoms in _iter_structures(structures):
        vector = np.asarray(extractor(atoms), dtype=np.float64).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Non-finite or empty feature: {source}:{frame_index}")
        if vectors and vector.size != vectors[0].size:
            raise ValueError("inconsistent feature dimension")
        vectors.append(vector)
        identity.append((source, frame_index))
        if progress is not None:
            progress(len(vectors))
    if len(vectors) < 2:
        raise ValueError("PCA needs at least two structures")
    features = np.asarray(vectors, dtype=np.float64)
    if n_components > min(features.shape):
        raise ValueError(
            f"n_components={n_components} exceeds the usable rank of "
            f"{len(features)} structures x {features.shape[1]} features"
        )
    if labels is not None:
        labels = list(labels)
        if len(labels) != len(features):
            raise ValueError("labels must hold one entry per structure")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=int(n_components), random_state=0).fit(scaled)
    projected = pca.transform(scaled)
    if not np.isfinite(projected).all():
        raise ValueError("PCA produced non-finite coordinates; check constant features")

    columns = [f"pc{i + 1}" for i in range(int(n_components))]
    with output.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*IDENTITY_COLUMNS, *columns])
        for index, ((source, frame_index), coordinates) in enumerate(zip(identity, projected)):
            writer.writerow(
                [
                    index,
                    source,
                    frame_index,
                    "" if labels is None else labels[index],
                    *(f"{value:.8g}" for value in coordinates),
                ]
            )

    ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return dict(
        output_csv=str(output),
        n_structures=len(features),
        feature_dim=int(features.shape[1]),
        n_components=int(n_components),
        explained_variance_ratio=ratio.tolist(),
        cumulative_explained_variance_ratio=float(ratio.sum()),
        columns=[*IDENTITY_COLUMNS, *columns],
        extractor=getattr(extractor, "metadata", {"type": "custom"}),
    )
