"""Optional feature extractors, lazily importing model packages."""
import numpy as np
import hashlib
import threading
from contextlib import contextmanager

_neighbor_lock=threading.RLock()
_m3gnet_import_lock=threading.Lock()

@contextmanager
def _legacy_m3gnet_ase_import():
    """Bridge the obsolete import used by M3GNet, not the relaxation filter."""
    import ase.constraints as constraints
    from ase.filters import ExpCellFilter
    with _m3gnet_import_lock:
        missing = not hasattr(constraints, "ExpCellFilter")
        if missing:
            constraints.ExpCellFilter = ExpCellFilter
        try:
            yield
        finally:
            if missing:
                del constraints.ExpCellFilter

@contextmanager
def _windows_neighbor_compat():
    import sys
    if sys.platform != "win32":
        yield
        return
    import m3gnet.graph._structure as module
    with _neighbor_lock:
        original=module.find_points_in_spheres
        def compatible(*args, **kwargs):
            kwargs["pbc"]=np.asarray(kwargs["pbc"],dtype=np.int64)
            return original(*args,**kwargs)
        module.find_points_in_spheres=compatible
        try: yield
        finally: module.find_points_in_spheres=original
from pathlib import Path


def _sha256_of_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _normalize_layers(layers, num_interactions):
    """Return 1-based interaction indices; ``"all"`` selects every layer."""
    if layers is None or (isinstance(layers, str) and layers.strip().lower() in ("all", "both")):
        return tuple(range(1, num_interactions+1))
    if isinstance(layers, str):
        try: layers=[int(part) for part in layers.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"layers must be interaction numbers or 'all', got {layers!r}") from exc
    if isinstance(layers, int): layers=[layers]
    try: requested=[int(item) for item in layers]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layers must be interaction numbers or 'all', got {layers!r}") from exc
    if not requested: raise ValueError("layers must not be empty")
    out_of_range=sorted({item for item in requested if item<1 or item>num_interactions})
    if out_of_range:
        raise ValueError(f"layers {out_of_range} out of range: checkpoint has {num_interactions} interaction layers")
    return tuple(dict.fromkeys(requested))


class MACEAtomicDescriptors:
    """Per-atom invariant MACE descriptors with selectable interaction layers.

    ``layers`` counts MACE interaction blocks: ``1`` is the first layer, ``2``
    the second, ``"all"`` (default) every layer in the checkpoint.  The public
    ``MACECalculator.get_descriptors`` only returns cumulative layers, so a
    single layer is sliced out of the model's ``node_feats`` (the l=0 block of
    that layer).  The slice reproduces the official cumulative output exactly;
    ``Process_MLIP/tests/test_mace_descriptors.py`` guards that equality.
    """

    def __init__(self, model_path, device="cpu", head=None):
        path=Path(model_path).resolve(strict=True)
        from Process_MLIP import create_mace_calculator
        self.calculator=create_mace_calculator(str(path), device=device, head=head, default_dtype="float32")
        model=self.calculator.models[0]
        from e3nn import o3
        irreps_out=o3.Irreps(str(model.products[0].linear.irreps_out))
        self.num_interactions=int(model.num_interactions)
        self.dim_per_layer=int(irreps_out.dim//(irreps_out.lmax+1)**2)
        widths=[int(irreps_out.dim)]*self.num_interactions
        widths[-1]=self.dim_per_layer
        self._layer_offsets=tuple(int(value) for value in np.cumsum([0]+widths))
        self.metadata=dict(type="mace_atomic", model_path=str(path), model_sha256=_sha256_of_file(path),
                           head=head, device=device, invariants_only=True,
                           num_interactions=self.num_interactions, dim_per_layer=self.dim_per_layer)

    def per_atom(self, atoms, layers="all"):
        """Return ``(n_atoms, dim_per_layer * len(layers))`` invariant descriptors."""
        indices=_normalize_layers(layers, self.num_interactions)
        import torch
        with torch.no_grad():
            batch=self.calculator._atoms_to_batch(atoms)
            node_feats=self.calculator.models[0](batch.to_dict(), compute_force=False)["node_feats"]
        values=node_feats.detach().cpu().numpy()
        if values.shape[0]!=len(atoms): raise ValueError("Unexpected MACE descriptor shape")
        blocks=[]
        for layer in indices:
            start=self._layer_offsets[layer-1]
            blocks.append(values[:, start:start+self.dim_per_layer])
        return np.concatenate(blocks, axis=1)

    def aggregate(self, atoms, mode="atom_mean", layers="all"):
        """Pool per-atom descriptors into one vector or one vector per element."""
        values=self.per_atom(atoms, layers)
        if mode=="atom_mean": return values.mean(axis=0)
        if mode=="element_mean":
            symbols=np.asarray(atoms.get_chemical_symbols())
            return {str(symbol): values[symbols==symbol].mean(axis=0) for symbol in np.unique(symbols)}
        raise ValueError('mode must be "atom_mean" or "element_mean"')


class MACEEmbedding:
    """Mean-pooled MACE descriptors, the structure-level encoder of the CLI."""

    def __init__(self, model_path, device='cpu', head=None, layers="all"):
        self.descriptors=MACEAtomicDescriptors(model_path, device=device, head=head)
        self.calculator=self.descriptors.calculator
        self.layers=_normalize_layers(layers, self.descriptors.num_interactions)
        self.metadata=dict(self.descriptors.metadata)
        self.metadata.update(type="mace", invariants_only=True, aggregation="atom_mean", layers=self.layers)

    def __call__(self, atoms):
        v=self.descriptors.per_atom(atoms, self.layers)
        if v.shape[0]!=len(atoms): raise ValueError('Unexpected MACE descriptor shape')
        return v.reshape(len(atoms),-1).mean(axis=0)

class M3GNetEmbedding:
    """Paper formation-energy structure encoder, provided by maml."""
    def __init__(self, model_path):
        path=Path(model_path).resolve(strict=True)
        if not path.is_dir(): raise ValueError("M3GNet model must be a checkpoint directory")
        files=sorted(p for p in path.rglob("*") if p.is_file())
        if not files: raise ValueError("Empty M3GNet model directory")
        sha=hashlib.sha256()
        for f in files:
            sha.update(f.relative_to(path).as_posix().encode()+b"\0")
            with f.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024),b""): sha.update(chunk)
        self.metadata=dict(type="m3gnet_structure", model_path=str(path), model_sha256=sha.hexdigest(), source="maml.M3GNetStructure")
        from maml.describers import M3GNetStructure
        try:
            with _legacy_m3gnet_ase_import():
                self.encoder=M3GNetStructure(model_path=str(path))
        except ValueError as exc:
            if "File format not supported" in str(exc):
                raise RuntimeError("maml M3GNet checkpoint is incompatible with installed Keras; use a compatible M3GNet/TensorFlow environment") from exc
            raise
    def __call__(self, atoms):
        from pymatgen.io.ase import AseAtomsAdaptor
        structure=AseAtomsAdaptor.get_structure(atoms)
        result=self._describe(structure)
        if not getattr(self, "_weights_checked", False):
            status=self.encoder.describer_model.load_weights(str(Path(self.metadata["model_path"])/"m3gnet"))
            status.assert_existing_objects_matched()
            # The descriptor stops before the prediction head; its unused
            # checkpoint tensors are expected after all built layers match.
            status.expect_partial()
            self._weights_checked=True
            result=self._describe(structure)
        return np.asarray(result).reshape(-1)

    def _describe(self, structure):
        """Return a fixed-length structure vector for one pymatgen structure.

        maml reaches the layer preceding the prediction head through
        describer_model.layers[-2].layers[0].  With current m3gnet releases that
        slot holds a graph-network block, so the indexing returns the graph tuple
        instead of a vector and numpy rejects the ragged result.  The fallback
        replays the same submodel explicitly and mean-pools the final node
        embeddings over the atoms of the cell, a 64-dimensional structure-level
        feature vector that stands in for the documented pre-readout descriptor.
        """
        with _windows_neighbor_compat():
            try:
                raw=np.asarray(self.encoder.transform_one(structure))
                if raw.ndim==2 and raw.shape[0]==len(structure):
                    # Some m3gnet builds return the per-atom graph readout from
                    # that slot instead of a structure vector; it is not a
                    # descriptor, so take the mean-pooled node features too.
                    raise ValueError("per-atom readout")
                return raw.reshape(-1)
            except ValueError as exc:
                if "inhomogeneous" not in str(exc) and "per-atom readout" not in str(exc): raise
                self.metadata["descriptor"]="mean_pooled_node_features"
                return self._mean_pooled_node_features(structure)

    def _mean_pooled_node_features(self, structure):
        from m3gnet.graph import Index, tf_compute_distance_angle
        from m3gnet.layers import polynomial
        model=self.encoder.describer_model
        with _windows_neighbor_compat():
            graph=model.graph_converter.convert(structure).as_list()
            graph=tf_compute_distance_angle(graph)
            three_basis=model.basis_expansion(graph)
            three_cutoff=polynomial(graph[Index.BONDS], model.threebody_cutoff)
            graph=model.feature_adjust(model.featurizer(graph))
            for interaction, layer in zip(model.three_interactions, model.graph_layers):
                graph=interaction(graph, three_basis, three_cutoff)
                graph=layer(graph)
            nodes=np.asarray(graph[Index.ATOMS], dtype="float64")
        if nodes.ndim!=2:
            raise RuntimeError("M3GNet node features are not a two-dimensional matrix")
        return nodes.mean(axis=0)


def build_encoder(model_path, encoder="m3gnet", *, device="cpu", head=None, layers="all"):
    """Return a structure encoder; both encoders import their dependencies lazily.

    ``layers`` selects MACE interaction layers (``1``, ``2``, ``"all"``); the
    M3GNet encoder has a single graph readout and rejects any other value.
    """
    if encoder == "m3gnet":
        if device != "cpu" or head is not None:
            raise ValueError("device and head apply only to MACE; TensorFlow controls M3GNet visibility")
        if layers not in (None, "all"):
            raise ValueError("layers applies only to MACE; M3GNet exposes one mean-pooled graph readout")
        return M3GNetEmbedding(model_path)
    if encoder == "mace":
        return MACEEmbedding(model_path, device=device, head=head, layers=layers)
    raise ValueError("encoder must be m3gnet or mace")
