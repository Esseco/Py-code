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
class MACEEmbedding:
    def __init__(self, model_path, device='cpu', head=None):
        path=Path(model_path).resolve(strict=True)
        sha=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""): sha.update(chunk)
        self.metadata=dict(type="mace", model_path=str(path), model_sha256=sha.hexdigest(), head=head, device=device, invariants_only=True, aggregation="atom_mean")
        from Process_MLIP import create_mace_calculator
        self.calculator=create_mace_calculator(model_path, device=device, head=head, default_dtype="float32")
    def __call__(self, atoms):
        v=np.asarray(self.calculator.get_descriptors(atoms, invariants_only=True))
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
                return np.asarray(self.encoder.transform_one(structure)).reshape(-1)
            except ValueError as exc:
                if "inhomogeneous" not in str(exc): raise
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


def build_encoder(model_path, encoder="m3gnet", *, device="cpu", head=None):
    """Return a structure encoder; both encoders import their dependencies lazily."""
    if encoder == "m3gnet":
        if device != "cpu" or head is not None:
            raise ValueError("device and head apply only to MACE; TensorFlow controls M3GNet visibility")
        return M3GNetEmbedding(model_path)
    if encoder == "mace":
        return MACEEmbedding(model_path, device=device, head=head)
    raise ValueError("encoder must be m3gnet or mace")
