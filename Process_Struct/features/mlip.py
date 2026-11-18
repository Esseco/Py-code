"""MLIP structural embeddings.

The encoders themselves live in :mod:`Process_MLIP.direct_sampling.features`;
this module only converts a pymatgen ``Structure`` into ASE ``Atoms``, selects
the MACE interaction layers and gives the embedding dimensions stable column
names (``d000``, ``d001``, ...; per-element blocks add the element symbol).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import Structure

from .registry import FEATURE_SETS

__all__ = ["MLIPFeatures"]

_PREFIX = FEATURE_SETS["mlip"].prefix

_AGGREGATIONS = ("atom_mean", "element_mean")


class MLIPFeatures:
    """MACE / M3GNet embedding of one structure.

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
    """

    prefix = _PREFIX

    def __init__(
        self,
        model_path: str,
        encoder: str = "mace",
        device: str = "cpu",
        head: str | None = None,
        layers: str | int | tuple[int, ...] | list[int] = "all",
        aggregate: str = "atom_mean",
    ) -> None:
        if aggregate not in _AGGREGATIONS:
            raise ValueError(f"aggregate must be one of {_AGGREGATIONS}, got {aggregate!r}")
        self.model_path = str(model_path)
        self.encoder_name = encoder
        self.aggregate = aggregate
        self._elements: tuple[str, ...] | None = None
        self._dimensions: int | None = None
        self._descriptors: Any = None
        self._encoder: Any = None
        if encoder == "mace":
            from Process_MLIP.direct_sampling import MACEAtomicDescriptors
            from Process_MLIP.direct_sampling.features import _normalize_layers

            self._descriptors = MACEAtomicDescriptors(
                self.model_path, device=device, head=head
            )
            self._layers: tuple[int, ...] | str = _normalize_layers(
                layers, self._descriptors.num_interactions
            )
        else:
            from Process_MLIP.direct_sampling import build_encoder

            if layers not in (None, "all") or aggregate != "atom_mean":
                raise ValueError(
                    "layers and aggregate apply only to the MACE encoder; "
                    "M3GNet exposes one mean-pooled graph readout"
                )
            self._layers = "all"
            self._encoder = build_encoder(
                self.model_path, encoder=encoder, device=device, head=head
            )
        self.layers = self._layers

    @property
    def metadata(self) -> dict[str, Any]:
        if self._descriptors is not None:
            raw = dict(self._descriptors.metadata)
        else:
            raw = dict(getattr(self._encoder, "metadata", {}) or {})
        raw.setdefault("encoder", self.encoder_name)
        raw["layers"] = list(self._layers) if not isinstance(self._layers, str) else self._layers
        raw["aggregate"] = self.aggregate
        raw["dimensions"] = self._dimensions
        if self._elements is not None:
            raw["elements"] = list(self._elements)
        return raw

    @property
    def descriptors(self) -> Any:
        """The loaded :class:`MACEAtomicDescriptors` (MACE only)."""

        if self._descriptors is None:
            raise RuntimeError("the m3gnet encoder does not expose per-atom descriptors")
        return self._descriptors

    @property
    def names(self) -> list[str]:
        if self._dimensions is None:
            raise RuntimeError("column names are only known after the first compute()")
        width = max(3, len(str(self._dimensions - 1)))
        columns = [f"d{index:0{width}d}" for index in range(self._dimensions)]
        if self.aggregate == "atom_mean":
            return columns
        if not self._elements:
            raise RuntimeError("element blocks are only known after the first compute()")
        return [f"{element}.{column}" for element in self._elements for column in columns]

    def compute(
        self, structure: Structure, *, with_sites: bool = False
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return the embedding as ``{"d000": ...}`` or ``{"Na.d000": ...}``."""

        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        if self._descriptors is not None and self.aggregate == "element_mean":
            blocks = self._descriptors.aggregate(atoms, "element_mean", self._layers)
            values: dict[str, float] = {}
            for element in sorted(blocks):
                vector = np.asarray(blocks[element], dtype=float).reshape(-1)
                self._check_dimensions(int(vector.size))
                width = max(3, len(str(vector.size - 1)))
                for index, value in enumerate(vector):
                    values[f"{element}.d{index:0{width}d}"] = float(value)
            self._elements = tuple(sorted(blocks))
            return values, []
        if self._descriptors is not None:
            raw = self._descriptors.aggregate(atoms, "atom_mean", self._layers)
        else:
            raw = self._encoder(atoms)
        vector = np.asarray(raw, dtype=float).reshape(-1)
        self._check_dimensions(int(vector.size))
        return dict(zip(self.names, (float(value) for value in vector))), []

    def _check_dimensions(self, size: int) -> None:
        if self._dimensions is None:
            self._dimensions = size
        elif size != self._dimensions:
            raise ValueError(
                f"embedding dimension changed from {self._dimensions} to {size}"
            )
