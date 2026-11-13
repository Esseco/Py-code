"""Behavioral tests for endpoint generation and NEB execution without MLIP weights."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from Process_MLIP import generate_neb_pairs, run_mlip_neb, model_config
from Process_MLIP.tests._support import TemporaryDirectoryMixin


class PeriodicBarrier(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions[0, 0]
        self.results['energy'] = float(np.sin(np.pi*x/2)**2)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = -(np.pi/2)*np.sin(np.pi*x)
        self.results['forces'] = forces


class NEBTests(TemporaryDirectoryMixin, unittest.TestCase):
    def setUp(self):
        self.root = self.make_temp_directory()
        self.parent = Atoms('Li2Cu', positions=[[0, 0, 0], [2, 0, 0], [4, 4, 4]], cell=[8, 8, 8], pbc=True)
        self.structure = self.parent[[0, 2]]

    def test_pairs_move_one_mobile_atom_and_reject_bad_parent(self):
        pairs = generate_neb_pairs(self.structure, self.parent, mobile_ion='Li', random_num=1, max_distance=3)
        self.assertEqual(len(pairs), 1)
        initial, final = pairs[0]
        self.assertAlmostEqual(initial[0].distance(final[0]), 2.0)
        bad = self.parent.copy(); bad.positions[-1, 0] += 1.0
        with self.assertRaisesRegex(ValueError, 'Cu sites'):
            generate_neb_pairs(self.structure, bad, mobile_ion='Li', random_num=1)
        bad = self.parent.copy(); bad.cell[0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, 'same lattice'):
            generate_neb_pairs(self.structure, bad, mobile_ion='Li', random_num=1)

    def test_two_stage_neb_computes_known_barrier(self):
        report = run_mlip_neb(self.structure, self.parent, self.root / 'run',
            calculator=PeriodicBarrier(), mobile_ion='Li', random_num=1, max_distance=3,
            relax_endpoints=False, intermediate_images=3, neb_fmax=0.1, cineb_fmax=0.1,
            neb_steps=10, cineb_steps=10)
        row = report['results'][0]
        self.assertEqual(row['status'], 'completed', row.get('error'))
        self.assertTrue(row['neb_converged']); self.assertTrue(row['cineb_converged'])
        self.assertAlmostEqual(row['forward_barrier_eV'], 1.0, places=5)
        self.assertAlmostEqual(row['reverse_barrier_eV'], 1.0, places=5)
        self.assertEqual(len(row['energies_eV']), 5)
        self.assertTrue(Path(report['summary_csv']).is_file())
        self.assertEqual(json.loads((self.root/'run'/'path_001'/'result.json').read_text())['status'], 'completed')
        with self.assertRaises(FileExistsError):
            run_mlip_neb(self.structure, self.parent, self.root/'run', calculator=PeriodicBarrier(), mobile_ion='Li')

    def test_endpoint_nonconvergence_is_reported(self):
        displaced = self.structure.copy(); displaced.positions[0, 0] = 0.25
        report = run_mlip_neb(displaced, self.parent, self.root/'failed',
            calculator=PeriodicBarrier(), mobile_ion='Li', random_num=1, max_distance=3,
            endpoint_steps=0, endpoint_fmax=1e-10)
        self.assertEqual(report['results'][0]['status'], 'failed')
        self.assertIn('did not converge', report['results'][0]['error'])

    def test_existing_model_factories_are_used(self):
        models = [model_config('mace', model_path='/models/mace.model', head=None),
                  model_config('sevennet', model_path='/models/nano.pth', enable_flash=True),
                  model_config('orb', model_path='/models/orb.ckpt')]
        for config in models:
            with self.subTest(model=config['type']), patch('Process_MLIP.neb.create_mlip_calculator', return_value=PeriodicBarrier()) as factory:
                report = run_mlip_neb(self.structure, self.parent, self.root/config['type'],
                    model=config, mobile_ion='Li', random_num=1, max_distance=3,
                    relax_endpoints=False, neb_steps=10, cineb_steps=10)
                self.assertEqual(factory.call_args.args[0]['type'], config['type'])
                self.assertEqual(report['results'][0]['status'], 'completed')

    def test_preselected_pairs_are_reused_and_validated(self):
        pairs = generate_neb_pairs(self.structure, self.parent, mobile_ion='Li', random_num=1, max_distance=3)
        report = run_mlip_neb(self.structure, self.parent, self.root/'selected',
            calculator=PeriodicBarrier(), mobile_ion='Li', endpoint_pairs=pairs,
            relax_endpoints=False, neb_steps=10, cineb_steps=10)
        self.assertAlmostEqual(report['results'][0]['forward_barrier_eV'], 1.0, places=5)
        wrong = pairs[0][1].copy(); wrong.translate_sites([0], [0.05, 0, 0], frac_coords=True)
        with self.assertRaisesRegex(ValueError, 'must start at structure'):
            run_mlip_neb(self.structure, self.parent, self.root/'wrong', calculator=PeriodicBarrier(),
                         mobile_ion='Li', endpoint_pairs=[(wrong, pairs[0][1])])

if __name__ == '__main__':
    unittest.main()
