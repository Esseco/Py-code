"""Small offline checks for the native GPUMD timing adapter."""
import unittest

from Process_MLIP.benchmark.config import validate_md_backend
from Process_MLIP.benchmark.nep_native import (
    gpumd_md_parameters, parse_gpumd_completed_steps, parse_gpumd_timing,
)


class NativeNEPTests(unittest.TestCase):
    def test_md_parameters_preserve_requested_nvt_conditions(self):
        params = gpumd_md_parameters({
            "temperature_K": 800,
            "timestep_fs": 2,
            "tdamp_fs": 100,
            "steps": 1000,
            "trajectory_interval": 10,
            "log_interval": 20,
        })
        self.assertEqual(params, [
            ("time_step", 2.0),
            ("ensemble", ("nvt_nhc", 800.0, 800.0, 50.0)),
            ("dump_thermo", 20),
            ("dump_xyz", (10, "movie.xyz", "precision", "double", "velocity")),
            ("run", 1000),
        ])

    def test_parse_native_run_timings_and_throughput(self):
        parsed = parse_gpumd_timing(
            "Time used for this run = 0.125 second.\n"
            "Speed of this run = 672000 atom*step/second.\n",
            natoms=84, steps=1000,
        )
        self.assertAlmostEqual(parsed["elapsed_seconds"], 0.125)
        self.assertAlmostEqual(parsed["atom_steps_per_second"], 672000)
        self.assertAlmostEqual(parsed["steps_per_second"], 8000)

    def test_native_run_must_report_all_requested_steps(self):
        output = "100 steps completed.\n1000 steps completed.\n"
        self.assertEqual(parse_gpumd_completed_steps(output, 1000), 1000)
        with self.assertRaisesRegex(ValueError, "expected 1000"):
            parse_gpumd_completed_steps("900 steps completed.\n", 1000)

    def test_missing_native_timing_is_not_replaced_with_wall_time(self):
        with self.assertRaisesRegex(ValueError, "native GPUMD timing"):
            parse_gpumd_timing("GPUMD exited successfully", natoms=84, steps=1000)

    def test_native_backend_is_restricted_to_gpu_nep(self):
        model = {"type": "nep", "device": "cuda", "parameters": {"backend": "gpu"},
                 "md_backend": "gpumd_native"}
        self.assertIs(validate_md_backend(model), model)
        with self.assertRaisesRegex(ValueError, "requires NEP with backend=gpu"):
            validate_md_backend({"type": "nep", "device": "cpu", "parameters": {"backend": "cpu"},
                                 "md_backend": "gpumd_native"})
        with self.assertRaisesRegex(ValueError, "requires NEP with backend=gpu"):
            validate_md_backend({"type": "mace", "device": "cuda", "parameters": {},
                                 "md_backend": "gpumd_native"})

if __name__ == "__main__":
    unittest.main()
