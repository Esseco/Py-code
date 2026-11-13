"""Streaming ASE input, HDF5 feature cache and selected-frame export."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import h5py
from ase.io import iread, write
from .core import sample

def _sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()

def extract(paths, extractor, cache, *, progress=None):
    start=time.perf_counter()
    paths=[str(Path(p).resolve()) for p in paths]
    with h5py.File(cache,'x') as h:
        h.attrs['sources']=json.dumps(paths)
        h.attrs['source_sha256']=json.dumps([_sha256(path) for path in paths])
        h.attrs['extractor']=json.dumps(getattr(extractor,'metadata',{'type':'custom'}))
        data=None; refs=[]
        for source,path in enumerate(paths):
            for frame,atoms in enumerate(iread(path,index=':')):
                vec=np.asarray(extractor(atoms),dtype=np.float32).reshape(-1)
                if not np.isfinite(vec).all(): raise ValueError(f'Non-finite feature: {path}:{frame}')
                if data is None: data=h.create_dataset('features',shape=(0,len(vec)),maxshape=(None,len(vec)),chunks=True,dtype='f4')
                if len(vec)!=data.shape[1]: raise ValueError('inconsistent feature dimension')
                data.resize(data.shape[0]+1,axis=0); data[-1]=vec; refs.append((source,frame))
                if progress is not None: progress(len(refs))
        if data is None: raise ValueError('no frames')
        h.create_dataset('indices',data=np.asarray(refs,dtype=np.int64))
        h.attrs['extraction_seconds']=time.perf_counter()-start

def run(cache, output, n_samples, *, reference_count=0, **options):
    start=time.perf_counter(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    with h5py.File(cache,'r') as h:
        features=h['features'][:]; refs=h['indices'][:]; paths=json.loads(h.attrs['sources']); extractor=json.loads(h.attrs['extractor']); hashes=json.loads(h.attrs['source_sha256']); extraction_seconds=h.attrs.get('extraction_seconds')
    for path,expected in zip(paths,hashes):
        if _sha256(path)!=expected: raise ValueError(f'Source trajectory changed: {path}')
    result=(select_with_reference(features, reference_count, n_samples, **options)
            if reference_count else sample(features,n_samples,**options)); lookup={}
    for pos in result['indices']:
        source,frame=map(int,refs[pos]); lookup[(source,frame)]=int(pos)
    selected={}
    for source,path in enumerate(paths):
        needed={frame for s,frame in lookup if s==source}
        if not needed: continue
        for frame,atoms in enumerate(iread(path,index=':')):
            if frame in needed: selected[lookup[(source,frame)]]=atoms
            if frame>=max(needed): break
    if len(selected)!=len(lookup): raise ValueError('cached frame references no longer match source')
    ids=result['indices']
    if len(ids): write(output/'selected.extxyz',[selected[int(i)] for i in ids],format='extxyz')
    mapping=[dict(source=paths[int(refs[i,0])],trajectory=int(refs[i,0]),frame=int(refs[i,1]),feature_row=int(i)) for i in ids]
    (output/'selected_indices.json').write_text(json.dumps(mapping,indent=2),encoding='utf8')
    labels=result['labels']; report=dict(extractor=extractor,pca_method=options.get('components','kaiser'), pca_weight='explained_variance_ratio',pca_explained_variance_ratio=result['explained_variance_ratio'].tolist(), birch_threshold_used=result['threshold_used'], selection='maml_center',n_frames=len(features),n_samples=len(ids),reference_count=reference_count,seed=options.get('seed',42), options=options,mean_distance=result['mean_distance'],max_distance=result['max_distance'],cluster_counts={str(int(i)):int((labels==i).sum()) for i in np.unique(labels)},selected_counts={str(int(i)):int((labels[ids]==i).sum()) for i in np.unique(labels)},sampling_seconds=time.perf_counter()-start)
    if extraction_seconds is not None: report['feature_extraction_seconds']=float(extraction_seconds)
    (output/'sampling_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

def select_with_reference(features, reference_count, max_samples=100, **options):
    """Select new DIRECT cluster representatives not covered by reference rows.

    Reference rows come first. Any cluster containing a reference is considered
    covered; only the original DIRECT center from remaining clusters is returned.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not 0 <= reference_count < len(features):
        raise ValueError('invalid reference count or features')
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= 100:
        raise ValueError('max_samples must be between 1 and 100')
    distinct = len(np.unique(features, axis=0))
    if distinct < 2:
        raise ValueError('DIRECT requires at least two distinct feature vectors')
    result = sample(features, min(distinct, max_samples + reference_count), **options)
    labels = result['labels']
    covered = set(labels[:reference_count])
    selected = [int(i) for i in result['indices']
                if i >= reference_count and labels[i] not in covered]
    return {**result, 'indices': np.asarray(selected[:max_samples], dtype=int),
            'reference_count': reference_count}
