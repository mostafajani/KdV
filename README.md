# CN-PINN codebase for the KdV manuscript
the code accompanies a manuscript under review
Reference paper under review: "A Crank-Nicolson neural network for simulation of forward and inverse problems in nonlinear wave dynamics"  A. A. Rezapour, M. Jani.

This directory contains ten self-contained experiment scripts plus a shared
utilities module.  Every script produces figures, LaTeX tables and CSV data
under `plots/<experiment_name>/`.

## Requirements

```bash
python -m venv env
source env/bin/activate
pip install torch numpy pandas matplotlib scipy
```

A GPU is optional but strongly recommended (Experiments 3 and 4 do parameter
sweeps).  All code auto-detects CUDA (`common_utils.DEVICE`) and falls back to
CPU gracefully.

## File map

| File                                | Purpose                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------- |
| `common_utils.py`                   | Shared model classes, CN residual, exact solutions, plot style, seeding.    |
| `exp1_forward_single_soliton.py`    | Section 5.1 — forward KdV, single soliton, [-2,2]x[-2,2].                   |
| `exp2_inverse_double_soliton.py`    | Section 5.2 — inverse KdV, double-soliton, identify α.                      |
| `exp3_noise_robustness.py`          |  noise sweep (σ = 0…10 %), mean ± std of identified α.             |
| `exp4_sensitivity_analysis.py`      |  sweeps over depth, width, N_col, Δt.                              |
| `exp5_activation_ablation.py`       | sin vs tanh/GELU/SiLU/softplus/ReLU.                              |
| `exp6_dt_convergence.py`            | verifies observed O(Δt²) temporal convergence order.              |
| `exp7_cnoidal_wave.py`              | additional example: periodic cnoidal wave.                        |
| `exp8_mkdv_extension.py`            | extension to the modified KdV equation (cubic nonlinearity).      |
| `exp9_inverse_two_params.py`        | jointly identify λ₁ *and* α from the same data.                   |
| `exp10_conservation_laws.py`        | mass / momentum / Hamiltonian preservation diagnostic.            |
| `run_all.py`                        | Runs every experiment in order.                                             |

## Which figures / tables go where in the manuscript?
§ Numerical Experiment 1 (single soliton, forward) —
  use figures from `exp1_forward` and the error-evolution + loss-curve plots.
  § Numerical Experiment 2 (double soliton, inverse) —
  use `exp2_inverse` reconstruction, α-convergence and time-slice figures.
§ Robustness to observational noise —
  Table + `fig_alpha_vs_noise.png` + `fig_error_vs_noise.png` from `exp3_noise`.
§ Hyperparameter sensitivity —
  Four bar plots from `exp4_sensitivity` (depth / width / N_col / Δt) as a
  2×2 panel.
§ Activation ablation —
  Table + `fig_activation_bar.png` from `exp5_activation` justifies the sine
  activation choice.
§ Temporal-convergence study —
  `fig_dt_convergence.png` from `exp6_dt_convergence` shows the observed
  slope on a log-log plot; supports the O(Δt²) theoretical claim.
§ Additional worked examples (cnoidal, mKdV) —
  from `exp7_cnoidal` and `exp8_mkdv`.
§ Simultaneous identification of two parameters —
  from `exp9_two_params` — a strictly harder inverse benchmark.
§ Conservation-law preservation —
  `fig_conservation.png` from `exp10_conservation` — physically meaningful
  qualitative validation.

## Reproducibility

* All scripts seed NumPy, PyTorch and Python's `random` module (default seed
  = 42; noise sweep uses derived seeds).

The output structure will be identical; only the accuracy will be lower.
