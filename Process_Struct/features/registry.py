"""Feature-set registry shared by :mod:`Process_Struct.features`.

The registry is deliberately dependency-free: it only records which feature
sets a caller may request, the column prefix each set owns, the extraction
levels it supports and the optional packages it needs.  Availability checks use
:func:`importlib.util.find_spec`, so importing this module never imports a
heavy package.
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_SETS",
    "FEATURE_SETS",
    "LEVELS",
    "FeatureSet",
    "available_sets",
    "resolve_level",
    "resolve_sets",
    "select_names",
]

DEFAULT_SETS = ("geometry",)

LEVELS = ("structure", "site", "both")


@dataclass(frozen=True)
class FeatureSet:
    """One selectable group of feature columns.

    Attributes:
        name: Public name used in ``sets=[...]``.
        prefix: Prefix every column produced by the set carries, e.g. ``geom.``.
        levels: Levels the set can fill: ``structure``, ``site`` or both.
        requires: Importable module names that must be installed.
        description: One-line human description.
    """

    name: str
    prefix: str
    levels: frozenset[str]
    requires: tuple[str, ...]
    description: str


FEATURE_SETS: dict[str, FeatureSet] = {
    "geometry": FeatureSet(
        name="geometry",
        prefix="geom.",
        levels=frozenset({"structure", "site"}),
        requires=("pymatgen",),
        description=(
            "Manually engineered geometry features; delegates to "
            "StructureFeatureExtractor in Process_Struct.Get_structfeatures_all."
        ),
    ),
    "cnn": FeatureSet(
        name="cnn",
        prefix="cnn.",
        levels=frozenset({"structure", "site"}),
        requires=("pymatgen",),
        description="Coordination numbers from pymatgen CrystalNN (native, no extra dependency).",
    ),
    "cnn_fp": FeatureSet(
        name="cnn_fp",
        prefix="cnn_fp.",
        levels=frozenset({"structure", "site"}),
        requires=("pymatgen", "matminer"),
        description="matminer CrystalNN fingerprints (optional dependency, lazily imported).",
    ),
    "mlip": FeatureSet(
        name="mlip",
        prefix="mlip.",
        levels=frozenset({"structure"}),
        requires=("numpy",),
        description=(
            "Mean-pooled MLIP structure embedding (MACE or M3GNet) via "
            "Process_MLIP.direct_sampling.build_encoder."
        ),
    ),
}


def check_dependencies(feature_set: FeatureSet) -> None:
    """Raise :class:`ImportError` when a required package is missing.

    The message names the interpreter so a wrong conda environment is easy to
    spot, which is the most common cause on this machine.
    """

    missing = [module for module in feature_set.requires if find_spec(module) is None]
    if missing:
        raise ImportError(
            f"feature set {feature_set.name!r} requires {', '.join(missing)}; "
            f"not importable from {sys.executable}. "
            f"Install it or drop that set."
        )


def resolve_sets(sets: str | Sequence[str] | None) -> tuple[FeatureSet, ...]:
    """Return the requested feature sets, in request order, without duplicates."""

    if sets is None:
        selected: tuple[str, ...] = DEFAULT_SETS
    elif isinstance(sets, str):
        selected = (sets,)
    else:
        selected = tuple(sets)

    if not selected:
        raise ValueError("select at least one feature set")

    unknown = [name for name in selected if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(
            f"unknown feature set(s) {unknown}; choose from {sorted(FEATURE_SETS)}"
        )

    resolved = []
    for name in dict.fromkeys(selected):
        feature_set = FEATURE_SETS[name]
        check_dependencies(feature_set)
        resolved.append(feature_set)
    return tuple(resolved)


def resolve_level(level: str) -> tuple[str, ...]:
    """Map ``level`` onto the tables that must be filled."""

    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}")
    if level == "both":
        return ("structure", "site")
    return (level,)


def select_names(
    names: Iterable[str],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[str]:
    """Filter full column names with glob patterns.

    Patterns are matched against the prefixed name, e.g. ``"cnn.cn_*"``.
    ``include`` keeps matches only; ``exclude`` then drops matches.
    """

    selected = list(names)
    if include:
        selected = [
            name
            for name in selected
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in include)
        ]
    if exclude:
        selected = [
            name
            for name in selected
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude)
        ]
    return selected


def available_sets() -> dict[str, dict[str, object]]:
    """Describe every registered set, including whether it can run right now."""

    return {
        name: {
            "prefix": feature_set.prefix,
            "levels": sorted(feature_set.levels),
            "requires": list(feature_set.requires),
            "available": all(
                find_spec(module) is not None for module in feature_set.requires
            ),
            "description": feature_set.description,
        }
        for name, feature_set in FEATURE_SETS.items()
    }
