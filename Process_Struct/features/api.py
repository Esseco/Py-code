"""Unified structure feature extraction.

One structure in, a flat table out::

    from Process_Struct.features import extract_features

    result = extract_features("POSCAR", sets=["geometry", "cnn"])
    result.values        # {"geom.a": 5.64, ..., "cnn.cn_mean": 5.8}
    result.to_frame()    # one-row DataFrame

Feature definitions stay in the modules behind :mod:`Process_Struct.features`;
this module only routes inputs, prefixes column names, records warnings and
isolates per-structure failures in batch mode.

Site-level rows are the union of the per-set rows keyed by ``site_index``.  A
set that cannot describe a site (for example geometry features, which only
cover transition-metal sites, on a site outside that set) leaves NaN there, so
downstream code should drop rows or columns before PCA.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pymatgen.core import Structure

from .registry import (
    DEFAULT_SETS,
    FEATURE_SETS,
    FeatureSet,
    resolve_level,
    resolve_sets,
    select_names,
)

__all__ = [
    "DEFAULT_SETS",
    "FEATURE_SETS",
    "FeatureExtractor",
    "FeatureResult",
    "extract_features",
    "extract_many",
    "load_structure",
]

_MAX_WARNINGS = 50

_SITE_ID_COLUMNS = ("site_index", "element")


@dataclass
class FeatureResult:
    """Feature table for one structure.

    Attributes:
        values: Structure-level column name to value, in column order.
        sites: Site-level rows; each row always has ``site_index`` and
            ``element`` plus the prefixed site columns of the enabled sets.
        metadata: Set selection, level, per-set provider metadata, structure
            summary and interpreter details needed to reproduce the run.
        warnings: Warning texts emitted while computing; de-duplicated in
            emission order.
    """

    values: dict[str, float] = field(default_factory=dict)
    sites: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.values)

    @property
    def site_names(self) -> list[str]:
        columns: list[str] = []
        for row in self.sites:
            for name in row:
                if name not in columns:
                    columns.append(name)
        return columns

    def to_vector(self) -> np.ndarray:
        """Structure-level values as a float array in ``names`` order."""

        return np.array([self.values[name] for name in self.values], dtype=float)

    def to_frame(self):
        """One-row DataFrame with structure-level columns."""

        import pandas as pd

        return pd.DataFrame([self.values], columns=self.names)

    def to_site_frame(self):
        """Site-level rows as a DataFrame; no rows give an empty frame."""

        import pandas as pd

        return pd.DataFrame(self.sites, columns=self.site_names)


class FeatureExtractor:
    """Reusable extractor; expensive models are loaded once at construction.

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
    """

    def __init__(
        self,
        sets: str | Sequence[str] | None = None,
        *,
        level: str = "structure",
        encoder: str = "mace",
        model_path: str | None = None,
        device: str = "cpu",
        head: str | None = None,
        mlip_layers: str | int | Sequence[int] = "all",
        mlip_aggregate: str = "atom_mean",
        geometry_options: dict[str, Any] | None = None,
        cnn_options: dict[str, Any] | None = None,
        cnn_site_filter: str = "all",
        cnn_fp_preset: str = "cn",
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> None:
        self.feature_sets: tuple[FeatureSet, ...] = resolve_sets(sets)
        self.level = level
        self.tables = resolve_level(level)
        self.include = list(include or [])
        self.exclude = list(exclude or [])

        self._providers: dict[str, Any] = {}
        for feature_set in self.feature_sets:
            name = feature_set.name
            if name == "geometry":
                from .geometry import GeometryFeatures

                self._providers[name] = GeometryFeatures(geometry_options)
            elif name == "cnn":
                from .crystalnn import CrystalNNCounter

                self._providers[name] = CrystalNNCounter(
                    cnn_options, site_filter=cnn_site_filter
                )
            elif name == "cnn_fp":
                from .crystalnn import CrystalNNFingerprints

                self._providers[name] = CrystalNNFingerprints(preset=cnn_fp_preset)
            elif name == "mlip":
                if not model_path:
                    raise ValueError("sets containing 'mlip' require model_path")
                from .mlip import MLIPFeatures

                self._providers[name] = MLIPFeatures(
                    model_path,
                    encoder=encoder,
                    device=device,
                    head=head,
                    layers=mlip_layers,
                    aggregate=mlip_aggregate,
                )
            else:  # pragma: no cover - resolve_sets already rejects unknown names
                raise ValueError(f"no provider for feature set {name!r}")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "sets": [feature_set.name for feature_set in self.feature_sets],
            "level": self.level,
            "include": self.include,
            "exclude": self.exclude,
            "providers": {
                name: dict(getattr(provider, "metadata", {}) or {})
                for name, provider in self._providers.items()
            },
        }

    def __call__(self, structure_or_path: Any) -> FeatureResult:
        """Alias of :meth:`extract`."""

        return self.extract(structure_or_path)

    def extract(self, structure_or_path: Any) -> FeatureResult:
        """Compute the selected features for one structure."""

        structure = load_structure(structure_or_path)
        values: dict[str, float] = {}
        notes: list[str] = []
        site_tables: dict[str, dict[int, dict[str, Any]]] = {}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for feature_set in self.feature_sets:
                provider = self._providers[feature_set.name]
                prefix = feature_set.prefix
                part, rows = provider.compute(
                    structure, with_sites="site" in self.tables
                )
                for name, value in part.items():
                    values[f"{prefix}{name}"] = _as_float(name, value, notes)
                if "site" in self.tables:
                    if rows:
                        table: dict[int, dict[str, Any]] = {}
                        for row in rows:
                            values_row, skipped = _site_values(row, prefix)
                            table[_site_key(row)] = values_row
                            if skipped:
                                notes.append(
                                    "dropped non-numeric site fields: "
                                    + ", ".join(skipped)
                                )
                        site_tables[feature_set.name] = table
                    elif feature_set.name == "mlip":
                        notes.append("mlip has no site-level rows; skipped")

        filtered = select_names(values, self.include, self.exclude)
        values = {name: values[name] for name in filtered}

        metadata = self.metadata
        metadata["structure"] = _structure_metadata(structure)

        return FeatureResult(
            values=values,
            sites=_merge_site_rows(site_tables, self.include, self.exclude),
            metadata=metadata,
            warnings=_dedupe([*(str(item.message) for item in caught), *notes]),
        )


def _as_float(name: str, value: Any, notes: list[str]) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        notes.append(f"feature {name!r} is not numeric ({value!r}); stored as NaN")
        return float("nan")


def _site_key(row: dict[str, Any]) -> int:
    index = row.get("site_index")
    if index is None:
        raise ValueError("site-level row is missing 'site_index'")
    return int(index)


def _site_values(
    row: dict[str, Any], prefix: str
) -> tuple[dict[str, Any], list[str]]:
    """Rename one provider row to prefixed columns, keeping ``element``.

    Non-numeric fields (neighbour index lists, for example) are dropped so the
    site table stays a numeric matrix; their names are returned for logging.
    """

    values: dict[str, Any] = {}
    skipped: list[str] = []
    for name, value in row.items():
        if name == "site_index":
            continue
        if name == "element":
            values["element"] = value
            continue
        if isinstance(value, (int, float, bool, np.number)):
            values[f"{prefix}{name}"] = value
        else:
            skipped.append(f"{prefix}{name}")
    return values, skipped


def _structure_metadata(structure: Structure) -> dict[str, Any]:
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "a": round(float(structure.lattice.a), 6),
        "b": round(float(structure.lattice.b), 6),
        "c": round(float(structure.lattice.c), 6),
        "alpha": round(float(structure.lattice.alpha), 6),
        "beta": round(float(structure.lattice.beta), 6),
        "gamma": round(float(structure.lattice.gamma), 6),
        "has_oxidation_states": all(
            getattr(site.specie, "oxi_state", None) is not None for site in structure
        ),
    }


def _merge_site_rows(
    site_tables: dict[str, dict[int, dict[str, Any]]],
    include: Sequence[str],
    exclude: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge per-set site rows on ``site_index`` and apply column filters."""

    indices: list[int] = []
    for rows in site_tables.values():
        for index in rows:
            if index not in indices:
                indices.append(index)

    if not indices:
        return []

    merged: list[dict[str, Any]] = []
    for index in sorted(indices):
        row: dict[str, Any] = {"site_index": index}
        for rows in site_tables.values():
            if index in rows:
                for name, value in rows[index].items():
                    row.setdefault(name, value)
        merged.append(row)

    columns = [name for row in merged for name in row if name not in _SITE_ID_COLUMNS]
    columns = select_names(list(dict.fromkeys(columns)), include, exclude)

    return [
        {key: row[key] for key in (*_SITE_ID_COLUMNS, *columns) if key in row}
        for row in merged
    ]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen[:_MAX_WARNINGS]


def load_structure(structure_or_path: Any) -> Structure:
    """Return a pymatgen ``Structure`` from a Structure, ASE Atoms or path."""

    if isinstance(structure_or_path, Structure):
        return structure_or_path
    if hasattr(structure_or_path, "get_chemical_symbols") and hasattr(
        structure_or_path, "get_positions"
    ):
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_structure(structure_or_path)
    if isinstance(structure_or_path, (str, Path)):
        return _read_structure_file(Path(structure_or_path))
    raise TypeError(
        "expected pymatgen Structure, ASE Atoms or a structure path, got "
        f"{type(structure_or_path).__name__}"
    )


def _read_structure_file(path: Path) -> Structure:
    if not path.exists():
        raise FileNotFoundError(f"structure file not found: {path}")
    name = path.name.upper()
    if name in ("POSCAR", "CONTCAR") or path.suffix.lower() in (".vasp", ".poscar"):
        from pymatgen.io.vasp import Poscar

        return Poscar.from_file(str(path)).structure
    return Structure.from_file(str(path))


def extract_features(structure_or_path: Any, **kwargs: Any) -> FeatureResult:
    """One-shot convenience wrapper; rebuilds the extractor on every call.

    Use :class:`FeatureExtractor` for batches so models are loaded once.
    """

    return FeatureExtractor(**kwargs).extract(structure_or_path)


def extract_many(
    structures: Iterable[Any],
    *,
    return_sites: bool = False,
    **kwargs: Any,
):
    """Extract features for many structures; one bad structure does not abort.

    Returns a DataFrame with ``status``, ``error`` and ``formula`` columns in
    addition to the feature columns.  With ``return_sites=True`` returns
    ``(structure_frame, site_frame)``, where the site frame carries a
    ``structure_index`` column.
    """

    import pandas as pd

    extractor = FeatureExtractor(**kwargs)
    rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []

    for position, item in enumerate(structures):
        try:
            result = extractor.extract(item)
        except Exception as exc:  # noqa: BLE001 - reported in the error column
            rows.append(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "formula": None,
                }
            )
            continue

        row: dict[str, Any] = {
            "status": "ok",
            "error": None,
            "formula": result.metadata["structure"]["formula"],
        }
        row.update(result.values)
        rows.append(row)
        for site in result.sites:
            site_rows.append({"structure_index": position, **site})

    frame = pd.DataFrame(rows)
    if not return_sites:
        return frame
    return frame, pd.DataFrame(site_rows)
