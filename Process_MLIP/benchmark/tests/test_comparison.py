import csv
import json
import tempfile
import unittest
from pathlib import Path

from Process_MLIP.benchmark.comparison import plot_comparison, _model_label, _hardware


class ComparisonTest(unittest.TestCase):
    def test_recorded_model_and_hardware_labels(self):
        row = {'model_id': 'nano', 'device': 'cuda',
               'model_config': json.dumps({'type': 'sevennet', 'parameters': {
                   'model_path': '/models/checkpoint_7net_nano_6.0.pth', 'enable_flash': True}}),
               'environment': json.dumps({'gpu': 'Tesla V100-SXM2-32GB, GPU-123, driver'})}
        self.assertEqual(_model_label('nano', [row]), 'SevenNet-Nano6-Flash')
        self.assertEqual(_hardware([row]), 'Hardware: Tesla V100-SXM2-32GB')
        row['model_config'] = json.dumps({'type': 'sevennet', 'parameters': {'model_path': 'different.pth'}})
        self.assertEqual(_model_label('nano', [row]), 'nano')
        configs = [{'model_id': 'mace_omat', 'model_config': json.dumps({'type': 'mace', 'parameters': {
            'model_path': '/models/mace-omat-0-medium.model', 'enable_symmetrix': True,
            'symmetrix_species': species}})} for species in (['Li', 'O'], ['Na', 'O'])]
        self.assertEqual(_model_label('mace_omat', configs), 'MACE-OMAT0-Symmetrix')

    def test_missing_scores_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speed = root / 'result' / 'speed'
            speed.mkdir(parents=True)
            fields = ['model_id', 'structure_id', 'natoms', 'status', 'steps_per_second', 'elapsed_seconds', 'converged']
            for stage in ('md', 'relax'):
                with (speed / f'{stage}.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(dict(model_id='model', structure_id='LCO_756', natoms=756,
                                         status='completed', steps_per_second=2, elapsed_seconds=5, converged='False'))
            cps = root / 'cps.csv'
            cps.write_text('model_id,cps\nmodel,\n', encoding='utf-8')
            result = plot_comparison([root / 'result'], cps, root / 'plots', {'LCO': 'LCO_756'})
            self.assertEqual(result['missing'], [('LCO_756', 'model', 'missing CPS')])
            self.assertTrue((root / 'plots' / 'speed_vs_cps.pdf').exists())
            cps.write_text('model_id,cps\nmodel,0.4\n', encoding='utf-8')
            self.assertEqual(plot_comparison([root / 'result'], cps, root / 'plots', {'LCO': 'LCO_756'})['missing'], [])
            with self.assertRaisesRegex(ValueError, 'Duplicate benchmark row'):
                plot_comparison([root / 'result', root / 'result'], cps, root / 'plots')
