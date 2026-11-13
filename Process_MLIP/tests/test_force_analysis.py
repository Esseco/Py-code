import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import write

from Process_MLIP import compare_force_results, evaluate_frame_forces


class ConstantForce(Calculator):
    implemented_properties = ['forces']

    def __init__(self, value):
        super().__init__()
        self.value = value

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['forces'] = np.tile(self.value, (len(atoms), 1))


class ForceAnalysisTests(unittest.TestCase):
    def test_compare_and_validation(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            source = root / 'frames.traj'
            atoms = Atoms('Cu2', positions=[[0, 0, 0], [2, 0, 0]])
            atoms.set_constraint(FixAtoms(indices=[0]))
            write(source, [atoms, atoms.copy()])
            outputs = [root/'a.json', root/'b.json']
            for path, value in zip(outputs, ([1, 0, 0], [3, 2, 0])):
                with patch('Process_MLIP.force_analysis.create_mlip_calculator', return_value=ConstantForce(value)):
                    evaluate_frame_forces(source, path, 'chgnet')
            first = json.loads(outputs[0].read_text(encoding='utf-8'))
            self.assertEqual(first['frames'][0]['forces'][0], [1, 0, 0])
            result = compare_force_results(outputs, root/'comparison.json')
            self.assertEqual(len(result['frames']), 2)
            np.testing.assert_allclose(result['frames'][0]['atomic_variance'], [2, 2])
            self.assertAlmostEqual(result['frames'][0]['max'], np.sqrt(2))
            with self.assertRaises(FileExistsError):
                compare_force_results(outputs, root/'comparison.json')
            second = json.loads(outputs[1].read_text(encoding='utf-8'))
            second['frames'][0]['identity'] = 'different'
            outputs[1].write_text(json.dumps(second), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frame mismatch'):
                compare_force_results(outputs, root/'bad.json')
            self.assertFalse((root/'bad.json').exists())
            with self.assertRaises(ValueError):
                compare_force_results([outputs[0], outputs[0]], root/'bad.json')


if __name__ == '__main__':
    unittest.main()
