"""CrystalNN feature columns.

Two independent layers are provided:

``CrystalNNCounter``
    Coordination numbers from pymatgen's own ``CrystalNN``; no extra package.
``CrystalNNFingerprints``
    matminer ``CrystalNNFingerprint`` and ``SiteStatsFingerprint`` presets; the
    package is imported lazily so the rest of the module keeps working without
    it.

Neither class decides column prefixes; :mod:`Process_Struct.features.api` adds
the ``cnn.`` / ``cnn_fp.`` prefixes recorded in the registry.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Sequence

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

from .registry import FEATURE_SETS

__all__ = ["CrystalNNCounter", "CrystalNNFingerprints"]

_PREFIX = FEATURE_SETS["cnn"].prefix
_FINGERPRINT_PREFIX = FEATURE_SETS["cnn_fp"].prefix

_TM_ELEMENTS = ("Mn", "Fe", "Co", "Ni", "Cu")
_MAX_CN = 12


def _sanitize_label(label: str) -> str:
    """Turn a matminer label into a column-friendly name."""

    cleaned = re.sub(r"[^0-9A-Za-z_.\-]+", "_", str(label)).strip("_")
    return cleaned or "value"


def _unique(names: Sequence[str], prefix: str) -> list[str]:
    seen: set[str] = set()
    duplicates = sorted({name for name in names if name in seen or seen.add(name)})
    if duplicates:
        raise ValueError(f"{prefix} produced duplicate column names: {duplicates}")
    return list(names)


class CrystalNNCounter:
    """Coordination numbers and their distribution from pymatgen ``CrystalNN``.

    Args:
        options: Keyword arguments forwarded to ``CrystalNN``; empty by default,
            which keeps pymatgen's own defaults (no oxidation-state guessing).
        site_filter: ``"all"`` for every site or ``"tm"`` for transition-metal
            sites only.
        tm_elements: Elements treated as transition metals when
            ``site_filter="tm"``.
    """

    prefix = _PREFIX

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        site_filter: str = "all",
        tm_elements: Sequence[str] | None = None,
    ) -> None:
        if site_filter not in ("all", "tm"):
            raise ValueError(f"site_filter must be 'all' or 'tm', got {site_filter!r}")
        self.options = dict(options or {})
        self.site_filter = site_filter
        self.tm_elements = tuple(tm_elements or _TM_ELEMENTS)
        self._cnn = CrystalNN(**self.options)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "options": self.options,
            "site_filter": self.site_filter,
            "tm_elements": list(self.tm_elements) if self.site_filter == "tm" else None,
        }

    def _site_indices(self, structure: Structure) -> list[int]:
        if self.site_filter == "all":
            return list(range(len(structure)))
        return [
            index
            for index, site in enumerate(structure)
            if site.specie.symbol in self.tm_elements
        ]

    def compute(
        self, structure: Structure, *, with_sites: bool = True
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return CrystalNN coordination statistics and optional per-site rows."""

        indices = self._site_indices(structure)
        if not indices:
            warnings.warn(
                "CrystalNN found no site to describe "
                f"(site_filter={self.site_filter!r}); CN statistics are NaN",
                RuntimeWarning,
                stacklevel=2,
            )

        cns: dict[int, float] = {}
        rows: list[dict[str, Any]] = []
        for index in indices:
            try:
                cn = float(len(self._cnn.get_nn_info(structure, index)))
            except Exception as exc:  # one bad site must not kill the batch
                warnings.warn(
                    f"CrystalNN failed for site {index}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                cn = float("nan")
            cns[index] = cn
            if with_sites:
                rows.append(
                    {
                        "site_index": int(index),
                        "element": structure[index].specie.symbol,
                        "cn": cn,
                    }
                )

        values: dict[str, float] = {"n_sites": float(len(indices))}
        valid = np.array([cn for cn in cns.values() if not np.isnan(cn)], dtype=float)
        if valid.size:
            values.update(
                {
                    "cn_mean": float(valid.mean()),
                    "cn_std": float(valid.std()),
                    "cn_min": float(valid.min()),
                    "cn_max": float(valid.max()),
                }
            )
        else:
            values.update(
                {key: float("nan") for key in ("cn_mean", "cn_std", "cn_min", "cn_max")}
            )

        if valid.size:
            for cn_value in range(_MAX_CN + 1):
                values[f"frac_cn_{cn_value}"] = float(
                    np.count_nonzero(valid == cn_value) / valid.size
                )
            values[f"frac_cn_gt{_MAX_CN}"] = float(
                np.count_nonzero(valid > _MAX_CN) / valid.size
            )
        else:
            for cn_value in range(_MAX_CN + 1):
                values[f"frac_cn_{cn_value}"] = float("nan")
            values[f"frac_cn_gt{_MAX_CN}"] = float("nan")

        return values, rows


class CrystalNNFingerprints:
    """matminer CrystalNN fingerprints, imported lazily.

    ``preset="cn"`` gives 24 site-level values and 48 structure-level values
    (mean/std/min/max of the site vector); ``preset="ops"`` gives 61 and 122.
    """

    prefix = _FINGERPRINT_PREFIX

    def __init__(self, preset: str = "cn") -> None:
        from matminer.featurizers.site import CrystalNNFingerprint
        from matminer.featurizers.structure import SiteStatsFingerprint

        self.preset = preset
        self._site = CrystalNNFingerprint.from_preset(preset)
        self._structure = SiteStatsFingerprint.from_preset(
            f"CrystalNNFingerprint_{preset}"
        )
        self._site_names = [
            _sanitize_label(label) for label in self._site.feature_labels()
        ]
        self._structure_names = [
            _sanitize_label(label) for label in self._structure.feature_labels()
        ]
        _unique(self._site_names, f"{self.prefix}site")
        _unique(self._structure_names, f"{self.prefix}structure")

    @property
    def metadata(self) -> dict[str, Any]:
        import matminer

        return {
            "preset": self.preset,
            "matminer_version": getattr(matminer, "__version__", "unknown"),
            "site_dimensions": len(self._site_names),
            "structure_dimensions": len(self._structure_names),
        }

    @property
    def site_names(self) -> list[str]:
        return list(self._site_names)

    @property
    def structure_names(self) -> list[str]:
        return list(self._structure_names)

    def compute(
        self, structure: Structure, *, with_sites: bool = True
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return structure-level statistics and optional per-site rows."""

        raw_structure = self._structure.featurize(structure)
        values = {
            name: float(value)
            for name, value in zip(self._structure_names, np.asarray(raw_structure).reshape(-1))
        }

        rows: list[dict[str, Any]] = []
        if with_sites:
            for index, site in enumerate(structure):
                raw_site = self._site.featurize(structure, index)
                row: dict[str, Any] = {
                    "site_index": int(index),
                    "element": site.specie.symbol,
                }
                row.update(
                    {
                        name: float(value)
                        for name, value in zip(
                            self._site_names, np.asarray(raw_site).reshape(-1)
                        )
                    }
                )
                rows.append(row)

        return values, rows
