import unittest,tempfile,json
from pathlib import Path
import numpy as np
import h5py
from ase import Atoms
from ase.io import write,read
from Process_MLIP.direct_sampling.core import sample
from Process_MLIP.direct_sampling.workflow import extract,run
from Process_MLIP.direct_sampling.workflow import select_with_reference
class TestDirect(unittest.TestCase):
 def test_seed_and_counts(self):
  x=np.random.default_rng(2).normal(size=(30,4)); a=sample(x,8,seed=3); b=sample(x,8,seed=3)
  np.testing.assert_array_equal(a['indices'],b['indices']); self.assertEqual(len(set(a['indices'])),8)
  with self.assertRaises(ValueError): sample(x,31)
 def test_sources(self):
  with tempfile.TemporaryDirectory(dir='.') as d:
   d=Path(d); sources=[]
   for j in range(2):
    f=d/f'{j}.extxyz'; write(f,[Atoms('Li',positions=[[j*3+i,0,0]],cell=[20,5,5],pbc=True) for i in range(3)]); sources.append(f)
   cache=d/'features.h5'; extract(sources,lambda a: np.array([a.positions[0,0],len(a)]),cache)
   report=run(cache,d/'out',4,seed=9); self.assertEqual(report['extractor']['type'],'custom')
   mapping=json.loads((d/'out/selected_indices.json').read_text()); frames=read(d/'out/selected.extxyz',index=':')
   for item,atoms in zip(mapping,frames): np.testing.assert_allclose(atoms.positions,read(item['source'],index=item['frame']).positions)
   with sources[0].open('a') as f: f.write('changed')
   with self.assertRaisesRegex(ValueError, 'changed'): run(cache,d/'out2',4)
if __name__=='__main__': unittest.main()

def test_reference_clusters_are_excluded(monkeypatch):
 import Process_MLIP.direct_sampling.workflow as workflow
 monkeypatch.setattr(workflow, 'sample', lambda features, n, **kw: {
  'indices': np.array([0, 3, 4]), 'labels': np.array([0, 1, 0, 1, 2])})
 result=select_with_reference(np.arange(10).reshape(5, 2), reference_count=2)
 assert result['indices'].tolist()==[4]
 assert result['reference_count']==2

def test_extract_reports_progress_and_timing(tmp_path):
 source=tmp_path/'frames.extxyz'
 write(source,[Atoms('Na',positions=[[i,0,0]],cell=[20,5,5],pbc=True) for i in range(3)])
 seen=[]
 cache=tmp_path/'features.h5'
 extract([source],lambda a: np.array([a.positions[0,0],len(a)]),cache,progress=seen.append)
 assert seen==[1,2,3]
 with h5py.File(cache,'r') as handle:
  assert float(handle.attrs['extraction_seconds'])>=0

def test_reference_duplicates_are_not_selected_with_real_direct():
 x=np.random.default_rng(18).normal(size=(20, 5))
 features=np.vstack([x[:2], x[0:1], x[2:]])
 result=select_with_reference(features, reference_count=2, max_samples=10)
 assert len(result['indices'])<=10
 assert all(i>=2 for i in result['indices'])
 assert 2 not in result['indices']

def test_reference_selection_accepts_100_and_rejects_more(monkeypatch):
 import pytest
 import Process_MLIP.direct_sampling.workflow as workflow
 monkeypatch.setattr(workflow, 'sample', lambda features, n, **kw: {
  'indices': np.arange(n), 'labels': np.arange(len(features))})
 features=np.arange(202).reshape(101, 2)
 assert len(select_with_reference(features, 1, max_samples=100)['indices'])==100
 with pytest.raises(ValueError, match='between 1 and 100'):
  select_with_reference(features, 1, max_samples=101)

