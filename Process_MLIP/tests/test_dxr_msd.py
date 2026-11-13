"""DXR 全起点 FFT MSD：数值一致性、unwrap、单位与输出保护。"""
import csv
import unittest

from ase import Atoms
from ase.io.trajectory import Trajectory
import numpy as np

from Process_MLIP import DXR_MSD, MP_MSD, export_msd
from Process_MLIP.dxr_msd import _all_origin_msd
from Process_MLIP.tests._support import TemporaryDirectoryMixin


class DXRMSDTests(TemporaryDirectoryMixin, unittest.TestCase):
    def setUp(self):
        self.root = self.make_temp_directory()

    def trajectory(self, positions, cell_length=25.0):
        """Write wrapped fractional frame positions, as GPUMD dumps them."""
        path = self.root / "md.traj"
        cell = np.eye(3) * cell_length
        with Trajectory(path, "w") as output:
            for points in positions:
                wrapped = (np.asarray(points) / cell_length) % 1.0
                atoms = Atoms("Li8Si2", cell=cell, pbc=True)
                atoms.set_scaled_positions(np.vstack([wrapped, [[0.1, 0.2, 0.3],
                                                                [0.6, 0.7, 0.8]]]))
                output.write(atoms)
        return path

    def walk(self, atoms=8, frames=3001, diffusion=0.01, dt_ps=0.05, seed=3):
        rng = np.random.default_rng(seed)
        steps = rng.normal(scale=np.sqrt(2 * diffusion * dt_ps), size=(frames, atoms, 3))
        return np.cumsum(steps, axis=0)

    def test_matches_explicit_multi_origin_sum(self):
        positions = np.random.default_rng(0).normal(size=(120, 5, 3))
        lags, values = _all_origin_msd(positions, 60)
        np.testing.assert_array_equal(lags, np.arange(61))
        for lag in (0, 1, 7, 60):
            explicit = ((positions[lag:] - positions[:120 - lag]) ** 2).mean(axis=(0, 1))
            np.testing.assert_allclose(values[lag], explicit, rtol=1e-12, atol=1e-12)

    def test_wrapped_frames_recover_input_diffusion(self):
        path = self.trajectory(self.walk())
        result = DXR_MSD(path, self.root / "msd.csv", species="Li", frame_interval_fs=50,
                         drift_reference=None, temperature_K=800)
        entry = result["results"][0]
        self.assertEqual(entry["atom_count"], 8)
        # Single wrapped random walk: exact unwrapping must not inflate D.
        self.assertAlmostEqual(entry["diffusion_cm2_s"], 1e-6, delta=6e-7)
        self.assertGreater(entry["r_squared"]["total"], 0.95)
        self.assertEqual(result["method"].split(";")[0], "FFT all-origin MSD")

    def test_conductivity_uses_si_diffusion_and_fixed_volume(self):
        path = self.trajectory(self.walk(frames=1201))
        result = DXR_MSD(path, species="Li", frame_interval_fs=50, drift_reference=None,
                         temperature_K=900, volume_A3=1.5e4)
        entry = result["results"][0]
        factor = (entry["atom_count"] / (1.5e4 * 1e-30) * (1.602176634e-19) ** 2
                  / (1.380649e-23 * 900))
        expected = factor * entry["diffusion_m2_s"]
        self.assertAlmostEqual(entry["conductivity_NE_S_m"], expected, delta=1e-15)
        self.assertAlmostEqual(entry["conductivity_NE_S_cm"], expected / 100, delta=1e-17)
        self.assertAlmostEqual(result["conductivity_NE_sum_S_m"], expected, delta=1e-15)

    def test_csv_columns_and_overwrite_protection(self):
        path = self.trajectory(self.walk(frames=301))
        csv_path = self.root / "msd.csv"
        result = DXR_MSD(path, csv_path, species="Li", frame_interval_fs=50,
                         drift_reference=None)
        with open(result["csv_file"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["segment", "frame_index", "time_fs", "time_ps",
                                        "species", "atom_count", "msd_x_A2", "msd_y_A2",
                                        "msd_z_A2", "msd_total_A2"])
        self.assertAlmostEqual(float(rows[1]["msd_total_A2"]),
                               sum(float(rows[1][key]) for key in
                                   ("msd_x_A2", "msd_y_A2", "msd_z_A2")), places=9)
        with self.assertRaises(FileExistsError):
            DXR_MSD(path, csv_path, species="Li", frame_interval_fs=50)
        self.assertIsNone(result["temperature_K"])
        self.assertEqual(rows[0]["species"], "Li")
        self.assertEqual(len(rows), int(round(result["max_lag_ps"] / 0.05)) + 1)

    def test_validation_and_alias(self):
        path = self.trajectory(self.walk(frames=101))
        with self.assertRaises(ValueError):
            DXR_MSD(path, species="Na", frame_interval_fs=50)
        with self.assertRaises(ValueError):
            DXR_MSD(path, species="Li", frame_interval_fs=50, drift_reference="bogus")
        with self.assertRaises(ValueError):
            DXR_MSD(path, species="Li", frame_interval_fs=50, max_lag_ratio=1.5)
        with self.assertRaises(ValueError):
            DXR_MSD(path, species="Li", frame_interval_fs=50, fit_start_ps=1e6)
        self.assertIs(MP_MSD, export_msd)
        self.assertEqual(MP_MSD.__name__, "export_msd")
