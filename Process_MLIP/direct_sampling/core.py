"""DIRECT on features using the original maml PCA, BIRCH and stratifier."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors

def sample(features, n_samples, *, components='kaiser', threshold=0.5, clusters=None, k=1, seed=42):
    from maml.sampling.pca import PrincipalComponentAnalysis
    from maml.sampling.clustering import BirchClustering
    from maml.sampling.stratified_sampling import SelectKFromClusters
    x=np.asarray(features,dtype=float)
    if x.ndim!=2 or len(x)<2 or x.shape[1]<1 or not np.isfinite(x).all(): raise ValueError('features must be a finite 2D matrix with at least two rows')
    if isinstance(n_samples,bool) or not isinstance(n_samples,(int,np.integer)) or not 1<=n_samples<=len(x): raise ValueError('invalid n_samples')
    if components!='kaiser' or k!=1: raise ValueError('Original method uses Kaiser PCA and k=1 for exact n_samples')
    if not np.isfinite(threshold) or threshold<=0: raise ValueError('threshold must be positive')
    if clusters is not None and clusters!=n_samples: raise ValueError('clusters must equal n_samples')
    z=StandardScaler().fit_transform(x)
    pca=PrincipalComponentAnalysis(weighting_PCs=True).fit(z)
    projected=pca.transform(z)
    if projected.shape[1]==0 or not np.isfinite(projected).all(): raise ValueError('Kaiser rule selected no usable components')
    current=float(threshold)
    for _ in range(30):
        probe=Birch(threshold=current,n_clusters=None).fit(projected)
        if len(probe.subcluster_centers_)>=n_samples: break
        current*=0.5
    else: raise ValueError('Cannot obtain requested distinct clusters; features may be duplicated')
    # Probe bounds maml's threshold adjustment loop; all actual clustering and selection use maml.
    clustering=BirchClustering(n=n_samples,threshold_init=current)
    data=clustering.transform(projected)
    labels=np.asarray(data['labels'])
    if len(np.unique(labels))!=n_samples: raise ValueError('Could not obtain requested number of nonempty clusters')
    ids=np.asarray(SelectKFromClusters(k=1,selection_criteria='center').transform(data)['selected_indexes'],dtype=int)
    if len(ids)!=n_samples: raise ValueError('maml returned fewer unique points than requested')
    distances=NearestNeighbors(n_neighbors=1).fit(projected[ids]).kneighbors(projected)[0][:,0]
    return dict(indices=ids,labels=labels,projection=projected,explained_variance_ratio=pca.pca.explained_variance_ratio_[:projected.shape[1]],mean_distance=float(distances.mean()),max_distance=float(distances.max()),threshold_used=clustering.threshold_init)
