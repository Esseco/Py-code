"""Unified structure feature extraction for :mod:`Process_Struct`.

Three feature families live behind one entry point:

``geometry``
    Hand-built geometry, bond and octahedra descriptors; delegates to
    :class:`Process_Struct.Get_structfeatures_all.StructureFeatureExtractor`.
``cnn``
    Coordination numbers from pymatgen ``CrystalNN`` (no extra dependency).
``cnn_fp``
    matminer CrystalNN fingerprints (optional dependency, lazily imported).
``mlip``
    Mean-pooled MACE / M3GNet structure embedding
    (:mod:`Process_MLIP.direct_sampling`).

Typical use::

    from Process_Struct.features import extract_features

    result = extract_features(structure, sets=["geometry", "cnn"])
    result.values      # flat dict, keys carry the set prefix
    result.to_frame()  # one-row DataFrame

For batches use :class:`FeatureExtractor`; it loads a model once::

    extractor = FeatureExtractor(sets=["mlip"], model_path=r"E:\\Py-Code\\MLIP_model\\MACE")
    extractor(structure)

Nothing is imported eagerly here: optional packages are checked by
:func:`~Process_Struct.features.registry.check_dependencies` when a set is
selected, so importing this package stays cheap.
"""

from __future__ import annotations

from .api import (
    DEFAULT_SETS,
    FEATURE_SETS,
    FeatureExtractor,
    FeatureResult,
    extract_features,
    extract_many,
    load_structure,
)
from .registry import available_sets

__all__ = [
    "DEFAULT_SETS",
    "FEATURE_SETS",
    "FeatureExtractor",
    "FeatureResult",
    "available_sets",
    "extract_features",
    "extract_many",
    "load_structure",
]
