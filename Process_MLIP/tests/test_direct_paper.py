import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from maml.sampling.pca import PrincipalComponentAnalysis
from maml.sampling.clustering import BirchClustering
from maml.sampling.stratified_sampling import SelectKFromClusters
from Process_MLIP.direct_sampling.core import sample
class OriginalPipelineTest(unittest.TestCase):
 def test_identical_to_maml(self):
  x=np.random.default_rng(8).normal(size=(50,5)); result=sample(x,8)
  z=StandardScaler().fit_transform(x); p=PrincipalComponentAnalysis(weighting_PCs=True).fit(z); projected=p.transform(z)
  clustered=BirchClustering(n=8,threshold_init=result['threshold_used']).transform(projected)
  original=SelectKFromClusters(k=1,selection_criteria='center').transform(clustered)['selected_indexes']
  np.testing.assert_allclose(result['projection'],projected); np.testing.assert_array_equal(result['indices'],original)
 def test_duplicates_fail(self):
  with self.assertRaisesRegex(ValueError,'distinct clusters'): sample(np.array([[0.],[0.],[1.],[1.]]),4)
if __name__=='__main__': unittest.main()
