"""
Experiment 3 - Noise-robustness study for the inverse problem
==============================================================

We repeat the double-soliton inverse identification of alpha but corrupt the
observation data with additive Gaussian noise:

        u_obs^noisy = u_obs * (1 + sigma * xi),   xi ~ N(0, 1)

for sigma in {0%, 1%, 3%, 5%, 10%}. Each noise level is repeated `N_TRIALS`
times with different random seeds and we report mean +/- std of the identified
alpha and the reconstruction L2 error.

Outputs (in ./plots/exp3_noise/):
    * fig_alpha_vs_noise.png   -- boxplot / errorbar of identified alpha
    * fig_error_vs_noise.png   -- L2 error vs noise
    * fig_alpha_trajectories.png -- alpha history for one representative trial
    * noise_results.csv        -- raw numbers
    * table_noise.tex          -- LaTeX summary

Run with:  python exp3_noise_robustness.py
"""

import os, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    InversePINN, cn_residual, u_double_soliton,
    relative_l2, ensure_dir, DEVICE,
)

OUT_DIR   = ensure_dir("plots/exp3_noise")
EPOCHS    = 2000                        # a bit shorter to keep sweep tractable
LR        = 5e-3
LAYERS    = [2, 32, 32, 32, 1]
LAM1      =  6.0
ALPHA_TRUE = 1.0
ALPHA_INIT = 0.5
DT        = 0.01
X_RANGE   = (-10.0, 10.0)
T_RANGE   = (-1.0,   1.0)
N_COL     = 5000
N_OBS     = 2000
N_IC      = 500
IC_WEIGHT = 10.0
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10]   # 0%, 1%, 3%, 5%, 10%
N_TRIALS     = 3

use_double_precision()
set_publication_style()


def run_trial(noise_sigma: float, seed: int):
    set_seed(seed)
    model = InversePINN(LAYERS, "sin", alpha_init=ALPHA_INIT).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_obs = torch.empty(N_OBS, 1).uniform_(*X_RANGE).to(DEVICE)
    t_obs = torch.empty(N_OBS, 1).uniform_(*T_RANGE).to(DEVICE)
    u_obs_clean = u_double_soliton(x_obs, t_obs, alpha=ALPHA_TRUE)

    # multiplicative Gaussian noise
    noise = torch.randn_like(u_obs_clean) * noise_sigma
    u_obs = u_obs_clean * (1.0 + noise)

    x_ic = torch.empty(N_IC, 1).uniform_(*X_RANGE).to(DEVICE)
    t_ic = torch.full((N_IC, 1), T_RANGE[0]).to(DEVICE)
    u_ic = u_double_soliton(x_ic, t_ic, alpha=ALPHA_TRUE)

    x_col = torch.empty(N_COL, 1).uniform_(*X_RANGE).to(DEVICE)
    t_col = torch.empty(N_COL, 1).uniform_(T_RANGE[0], T_RANGE[1] - DT).to(DEVICE)

    alpha_hist = []
    for epoch in range(EPOCHS + 1):
        optim.zero_grad()
        loss_data = torch.mean((model(x_obs, t_obs) - u_obs) ** 2)
        loss_ic   = torch.mean((model(x_ic,  t_ic ) - u_ic) ** 2)
        loss_pde  = cn_residual(model, x_col, t_col, DT, LAM1, model.alpha)
        loss = loss_data + loss_pde + IC_WEIGHT * loss_ic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        alpha_hist.append(model.alpha.item())

    # global L2 error on a fine grid
    x_v = np.linspace(*X_RANGE, 200); t_v = np.linspace(*T_RANGE, 200)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_double_soliton(Xt, Tt, alpha=ALPHA_TRUE).cpu().numpy().reshape(X.shape)
    return {
        "noise":       noise_sigma,
        "seed":        seed,
        "alpha_final": alpha_hist[-1],
        "alpha_err":   abs(alpha_hist[-1] - ALPHA_TRUE),
        "rel_L2":      relative_l2(ut, up),
        "alpha_hist":  alpha_hist,
    }


def sweep():
    tic = time.time()
    all_records = []
    trajectories = {}
    for sigma in NOISE_LEVELS:
        traj_for_sigma = []
        for i in range(N_TRIALS):
            seed = 100 * i + int(sigma * 1000)
            print(f"[noise={sigma:.2%}  trial {i+1}/{N_TRIALS}  seed {seed}] ...")
            rec = run_trial(sigma, seed)
            all_records.append({k: rec[k] for k in
                                ("noise","seed","alpha_final","alpha_err","rel_L2")})
            traj_for_sigma.append(rec["alpha_hist"])
        trajectories[sigma] = traj_for_sigma
    print(f"sweep done in {(time.time()-tic)/60:.1f} min")
    return pd.DataFrame(all_records), trajectories


def visualise(df: pd.DataFrame, trajectories: dict):
    df.to_csv(f"{OUT_DIR}/noise_results.csv", index=False)

    # ---- summary table -----------------------------------------------------
    agg = (df.groupby("noise")
             .agg(alpha_mean=("alpha_final","mean"),
                  alpha_std =("alpha_final","std"),
                  rel_L2_mean=("rel_L2","mean"),
                  rel_L2_std =("rel_L2","std"))
             .reset_index())
    agg["rel_alpha_err_%"] = 100 * (agg["alpha_mean"] - ALPHA_TRUE).abs() / ALPHA_TRUE
    with open(f"{OUT_DIR}/table_noise.tex", "w") as f:
        f.write(agg.to_latex(index=False, float_format="%.4e",
                             caption="Robustness of the inverse CN-PINN to "
                                     "Gaussian observation noise.",
                             label="tab:noise"))
    print(agg)

    # ---- errorbar plot: alpha vs noise -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg["noise"] * 100, agg["alpha_mean"], yerr=agg["alpha_std"],
                fmt="o-", capsize=4, color="tab:red",
                label=r"identified $\alpha$ (mean $\pm$ std)")
    ax.axhline(ALPHA_TRUE, ls="--", color="k",
               label=fr"true $\alpha={ALPHA_TRUE}$")
    ax.set_xlabel("noise level (%)"); ax.set_ylabel(r"$\hat\alpha$")
    ax.set_title("Identified dispersion vs noise")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_alpha_vs_noise.png"); plt.close(fig)

    # ---- errorbar plot: reconstruction L2 vs noise ------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg["noise"] * 100, agg["rel_L2_mean"], yerr=agg["rel_L2_std"],
                fmt="s-", capsize=4, color="tab:blue")
    ax.set_xlabel("noise level (%)"); ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Reconstruction error vs noise")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_error_vs_noise.png"); plt.close(fig)

    # ---- alpha trajectories (first trial per noise level) -----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(NOISE_LEVELS)))
    for c, sigma in zip(colors, NOISE_LEVELS):
        ax.plot(trajectories[sigma][0], color=c, label=f"noise = {sigma:.0%}")
    ax.axhline(ALPHA_TRUE, ls="--", color="k")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"$\alpha$-trajectories under different noise levels")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_alpha_trajectories.png"); plt.close(fig)


if __name__ == "__main__":
    df, trajectories = sweep()
    visualise(df, trajectories)
    print(f"\nAll outputs saved under: {OUT_DIR}")
