"""
Experiment 4 - Hyperparameter sensitivity analysis (forward problem)
=====================================================================

We sweep four architectural / training knobs and report the resulting
relative L2 error on the single-soliton problem (Section 5.1):

    * hidden depth   D  in {2, 3, 4, 5}
    * hidden width   W  in {8, 16, 32, 64}
    * collocation N_col in {250, 500, 1000, 2000, 5000}
    * time step     dt  in {0.005, 0.01, 0.02, 0.05, 0.1}

Each configuration is trained for EPOCHS iterations (kept short for
tractability) and its global relative L2 error is recorded.  A grouped bar
chart is produced for each sweep.

Outputs (in ./plots/exp4_sensitivity/):
    * fig_depth_sensitivity.png
    * fig_width_sensitivity.png
    * fig_ncol_sensitivity.png
    * fig_dt_sensitivity.png
    * sensitivity_results.csv
    * table_sensitivity.tex

Run with:  python exp4_sensitivity_analysis.py
"""

import os, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, cn_residual, u_single_soliton, make_ic_bc_col,
    relative_l2, ensure_dir, DEVICE,
)

OUT_DIR   = ensure_dir("plots/exp4_sensitivity")
EPOCHS    = 2000                # short per config, but enough to rank them
LR        = 1e-3
LAM1      = -6.0
ALPHA     = 1.0
X_RANGE   = (-2.0, 2.0)
T_RANGE   = (-2.0, 2.0)

use_double_precision()
set_publication_style()


def train_one(layers, dt, N_col, N_ic=500, N_bc=500, seed=42):
    set_seed(seed)
    model = PINN(layers, activation="sin").to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_col, t_col, x_ic, t_ic, x_bc, t_bc = make_ic_bc_col(
        X_RANGE, T_RANGE, N_col, N_ic, N_bc, dt
    )
    x_col, t_col = x_col.to(DEVICE), t_col.to(DEVICE)
    x_ic,  t_ic  = x_ic.to(DEVICE),  t_ic.to(DEVICE)
    x_bc,  t_bc  = x_bc.to(DEVICE),  t_bc.to(DEVICE)
    u_ic_true = u_single_soliton(x_ic, t_ic)
    u_bc_true = u_single_soliton(x_bc, t_bc)

    for _ in range(EPOCHS + 1):
        optim.zero_grad()
        loss = (torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
              + torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
              + cn_residual(model, x_col, t_col, dt, LAM1, ALPHA))
        loss.backward(); optim.step()

    # evaluate on fine grid
    x_v = np.linspace(*X_RANGE, 200); t_v = np.linspace(*T_RANGE, 200)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_single_soliton(Xt, Tt).cpu().numpy().reshape(X.shape)
    return relative_l2(ut, up)


# -----------------------------------------------------------------------------
# Sweeps
# -----------------------------------------------------------------------------
def sweep_depth():
    depths = [2, 3, 4, 5]
    rows = []
    for D in depths:
        layers = [2] + [32] * D + [1]
        err = train_one(layers, dt=0.01, N_col=5000)
        rows.append({"depth": D, "rel_L2": err})
        print(f"  depth={D}: rel L2 = {err:.4e}")
    return pd.DataFrame(rows)


def sweep_width():
    widths = [8, 16, 32, 64]
    rows = []
    for W in widths:
        layers = [2, W, W, W, 1]
        err = train_one(layers, dt=0.01, N_col=5000)
        rows.append({"width": W, "rel_L2": err})
        print(f"  width={W}: rel L2 = {err:.4e}")
    return pd.DataFrame(rows)


def sweep_ncol():
    ncols = [250, 500, 1000, 2000, 5000]
    rows = []
    for N in ncols:
        err = train_one([2, 32, 32, 32, 1], dt=0.01, N_col=N)
        rows.append({"N_col": N, "rel_L2": err})
        print(f"  N_col={N}: rel L2 = {err:.4e}")
    return pd.DataFrame(rows)


def sweep_dt():
    dts = [0.005, 0.01, 0.02, 0.05, 0.1]
    rows = []
    for dt in dts:
        err = train_one([2, 32, 32, 32, 1], dt=dt, N_col=5000)
        rows.append({"dt": dt, "rel_L2": err})
        print(f"  dt={dt}: rel L2 = {err:.4e}")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
def bar_plot(df, key, fname, xlabel, log_x=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    ax.bar(x, df["rel_L2"], color="teal", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in df[key]])
    ax.set_yscale("log")
    ax.set_ylabel(r"relative $L^2$ error (log)")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Sensitivity to {xlabel}")
    for i, v in enumerate(df["rel_L2"]):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fname}"); plt.close(fig)


if __name__ == "__main__":
    tic = time.time()
    print(">>> depth sweep");  df_d = sweep_depth()
    print(">>> width sweep");  df_w = sweep_width()
    print(">>> N_col sweep");  df_n = sweep_ncol()
    print(">>> dt sweep");     df_t = sweep_dt()

    bar_plot(df_d, "depth", "fig_depth_sensitivity.png", "hidden depth (layers)")
    bar_plot(df_w, "width", "fig_width_sensitivity.png", "hidden width (neurons)")
    bar_plot(df_n, "N_col", "fig_ncol_sensitivity.png", "collocation points")
    bar_plot(df_t, "dt",    "fig_dt_sensitivity.png",    r"time step $\Delta t$")

    # combined CSV
    df_d["knob"] = "depth"; df_d = df_d.rename(columns={"depth": "value"})
    df_w["knob"] = "width"; df_w = df_w.rename(columns={"width": "value"})
    df_n["knob"] = "N_col"; df_n = df_n.rename(columns={"N_col": "value"})
    df_t["knob"] = "dt";    df_t = df_t.rename(columns={"dt": "value"})
    combined = pd.concat([df_d, df_w, df_n, df_t], ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/sensitivity_results.csv", index=False)
    with open(f"{OUT_DIR}/table_sensitivity.tex", "w") as f:
        f.write(combined.to_latex(index=False, float_format="%.4e",
                                  caption="Hyperparameter sensitivity of the "
                                          "CN-PINN on the KdV forward problem.",
                                  label="tab:sensitivity"))
    print(f"\nsweep total time: {(time.time()-tic)/60:.1f} min")
    print(f"outputs in: {OUT_DIR}")
