"""Manually engineered geometry features.

This module only adapts :class:`Process_Struct.Get_structfeatures_all.StructureFeatureExtractor`
to the flat, prefixed table used by :mod:`Process_Struct.features`.  The
scientific definitions stay in that class; nothing is re-implemented here.
"""

from __future__ import annotations

from typing import Any

from pymatgen.core import Structure

from .registry import FEATURE_SETS

__all__ = ["GeometryFeatures"]

_PREFIX = FEATURE_SETS["geometry"].prefix


class GeometryFeatures:
    """Wrap ``StructureFeatureExtractor`` behind one ``compute`` call."""

    prefix = _PREFIX

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        from ..Get_structfeatures_all import StructureFeatureExtractor

        self.options = dict(options or {})
        self.extractor = StructureFeatureExtractor(**self.options)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "options": self.options,
            "global_groups": sorted(self.extractor.global_groups),
            "local_groups": sorted(self.extractor.local_groups),
        }

    def compute(
        self, structure: Structure, *, with_sites: bool = True
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return ``(structure_values, site_rows)`` for one structure.

        Both tables come from a single ``extract`` call so the neighbour search
        is not repeated.  Keys are *not* prefixed here; :mod:`api` adds prefixes.
        """

        result = self.extractor.extract(structure)

        global_values: dict[str, float] = dict(result.get("global") or {})
        if "Na" in global_values:
            global_values["Na_ratio"] = global_values.pop("Na")

        site_rows: list[dict[str, Any]] = []
        if not with_sites:
            return global_values, site_rows

        for record in result.get("local") or []:
            row: dict[str, Any] = {
                "site_index": record.get("tm_index"),
                "element": record.get("tm_element"),
            }
            for key, value in record.items():
                if key in ("tm_index", "tm_element"):
                    continue
                # The original record calls the Na/O ratio "Na"; the flat table
                # spells it out so it is not mistaken for a sodium count.
                row["Na_ratio" if key == "Na" else key] = value
            site_rows.append(row)

        return global_values, site_rows
