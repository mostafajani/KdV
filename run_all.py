"""
run_all.py - master orchestrator
=================================

Runs every experiment in sequence and dumps a global summary CSV.
Comment out any block you do not want to execute.

Rough wall-clock times on a single CPU (Adam, 5000 epochs, N_col ~ 5000):

    exp1   ~ 2 - 5 min       (single-soliton forward)
    exp2   ~ 3 - 6 min       (double-soliton inverse)
    exp3   ~ 30 - 60 min     (noise sweep = 5 levels x 3 trials x 3000 ep)
    exp4   ~ 30 - 60 min     (sensitivity = 18 configs x 2000 ep)
    exp5   ~ 15 - 30 min     (6 activations x 3000 ep)
    exp6   ~ 15 - 30 min     (6 dt values x 4000 ep)
    exp7   ~  3 -  6 min     (cnoidal)
    exp8   ~  3 -  6 min     (mKdV)
    exp9   ~  3 -  6 min     (two-parameter inverse)
    exp10  ~  3 -  6 min     (conservation-law diagnostic)

Total: ~2 - 4 h on CPU.  Use a GPU or reduce EPOCHS for a quick smoke test.
"""

import subprocess
import sys
import time
import os

EXPERIMENTS = [
    "exp1_forward_single_soliton.py",
    "exp2_inverse_double_soliton.py",
    "exp7_cnoidal_wave.py",
    "exp8_mkdv_extension.py",
    "exp9_inverse_two_params.py",
    "exp10_conservation_laws.py",
    "exp5_activation_ablation.py",     # medium
    "exp6_dt_convergence.py",          # medium
    "exp3_noise_robustness.py",        # slow
    "exp4_sensitivity_analysis.py",    # slow
]


def run(script: str):
    tic = time.time()
    print("\n" + "=" * 78)
    print(f"RUN  {script}")
    print("=" * 78)
    res = subprocess.run([sys.executable, script])
    dt = time.time() - tic
    status = "OK" if res.returncode == 0 else f"FAIL ({res.returncode})"
    print(f"---- {script}: {status} in {dt/60:.1f} min ----")
    return res.returncode == 0, dt


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logs = []
    for s in EXPERIMENTS:
        ok, dt = run(s)
        logs.append({"experiment": s, "ok": ok, "minutes": dt / 60.0})

    print("\n\n===== SUMMARY =====")
    for l in logs:
        print(f"  {l['experiment']:<40s}  ok={l['ok']}  {l['minutes']:.1f} min")
