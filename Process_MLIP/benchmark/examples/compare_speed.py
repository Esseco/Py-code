"""Edit paths and rerun to compare existing benchmark results without loading models."""
from pathlib import Path
from Process_MLIP.benchmark.comparison import plot_comparison

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "speed_pilot_20261027" / "results"
RESULT_DIRS = [RESULTS / name for name in ("chgnet", "mace_mh1", "sevennet_nano")]
RESULT_DIRS += [BASE / "speed_batch2_20261028" / name for name in
                ("mattersim_1m", "orb_direct", "sevennet_omni", "mattersim_5m",
                 "orb_conservative", "mace_omat_LCO", "mace_omat_NVP")]
COMPLETED = BASE / "speed_completed_20261028_1408"
RESULT_DIRS += [COMPLETED / "MLIP_speed_grace_20261028" / "results" / name
                for name in ("grace_2l_full", "grace_3l_full")]
RESULT_DIRS += [COMPLETED / "MLIP_orb_compile_20261028" / "results" / name
                for name in ("orb_direct", "orb_conservative")]
RESULT_DIRS += [BASE / "nep89_native_results_3828109" / "results"]
RESULT_DIRS += [BASE / "nequip_oeq_rerun_results_20261028" / "results" / name
                for name in ("s", "m", "l", "xl")]
CPS_CSV = BASE / "examples" / "matbench_cps.csv"
OUTPUT = RESULTS.parent / "comparison_plots"

if __name__ == "__main__":
    print(plot_comparison(RESULT_DIRS, CPS_CSV, OUTPUT))
