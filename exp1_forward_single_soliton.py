"""
Experiment 1 - Forward single-soliton KdV (Section 5.1 of the manuscript)
=========================================================================

Governing equation :  u_t - 6 u u_x + u_xxx = 0
Exact solution     :  u(x,t) = -2 sech^2(x - 4t)
Domain             :  (x, t) in [-2, 2] x [-2, 2],  dt = 0.01
Network            :  [2, 32, 32, 32, 1], sine activation, Xavier init
Optimizer          :  Adam, lr = 1e-3, 5000 epochs

Outputs (all saved to ./plots/exp1_forward/):
    * 3D surface comparison  (exact vs CN-PINN)              -> fig_3d_comparison.png
    * pcolor absolute-error map                              -> fig_error_map.png
    * time-slice line plots  (t = -1, -0.5, 0, 0.5, 1)       -> fig_time_slices.png
    * error time-slice line plots                            -> fig_error_time_slices.png
    * relative-L2 error vs time                              -> fig_error_evolution.png
    * training loss curves                                   -> fig_loss_curve.png
    * 3D spatiotemporal error surface                        -> fig_3d_error.png
    * summary table (LaTeX)                                  -> table_errors.tex
    * summary CSV                                            -> results.csv

Run with:  python exp1_forward_single_soliton.py
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, cn_residual, u_single_soliton,
    make_ic_bc_col, relative_l2, l_inf, rmse, ensure_dir, DEVICE,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUT_DIR   = ensure_dir("plots/exp1_forward")
EPOCHS    = 5000
LR        = 1e-3
LAYERS    = [2, 32, 32, 32, 1]
LAM1      = -6.0
ALPHA     = 1.0
DT        = 0.01
X_RANGE   = (-2.0, 2.0)
T_RANGE   = (-2.0, 2.0)
N_COL     = 5000
N_IC      = 500
N_BC      = 500

set_seed(42)
use_double_precision()
set_publication_style()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train():
    model = PINN(LAYERS, activation="sin").to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_col, t_col, x_ic, t_ic, x_bc, t_bc = make_ic_bc_col(
        X_RANGE, T_RANGE, N_COL, N_IC, N_BC, DT
    )
    x_col, t_col = x_col.to(DEVICE), t_col.to(DEVICE)
    x_ic,  t_ic  = x_ic.to(DEVICE),  t_ic.to(DEVICE)
    x_bc,  t_bc  = x_bc.to(DEVICE),  t_bc.to(DEVICE)

    u_ic_true = u_single_soliton(x_ic, t_ic)
    u_bc_true = u_single_soliton(x_bc, t_bc)

    history = {"total": [], "ic": [], "bc": [], "pde": []}
    print(f"Training CN-PINN (Exp 1) on {DEVICE} ...")
    tic = time.time()

    for epoch in range(EPOCHS + 1):
        optim.zero_grad()

        loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
        loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
        loss_pde = cn_residual(model, x_col, t_col, DT, LAM1, ALPHA)

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        optim.step()

        history["total"].append(loss.item())
        history["ic"   ].append(loss_ic.item())
        history["bc"   ].append(loss_bc.item())
        history["pde"  ].append(loss_pde.item())

        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} | total {loss.item():.4e} | "
                  f"ic {loss_ic.item():.2e}  bc {loss_bc.item():.2e}  "
                  f"pde {loss_pde.item():.2e}  ({time.time()-tic:.1f}s)")

    print(f"training done in {time.time()-tic:.1f}s")
    return model, history


# -----------------------------------------------------------------------------
# Visualisation & tables
# -----------------------------------------------------------------------------
def visualise(model, history):
    x_v = np.linspace(*X_RANGE, 400)
    t_v = np.linspace(*T_RANGE, 400)
    X, T = np.meshgrid(x_v, t_v)
    X_t = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    T_t = torch.tensor(T.flatten()[:, None]).to(DEVICE)

    with torch.no_grad():
        u_pred = model(X_t, T_t).cpu().numpy().reshape(X.shape)
    u_true = u_single_soliton(X_t, T_t).cpu().numpy().reshape(X.shape)
    abs_err = np.abs(u_true - u_pred)

    # ---- 3D surface comparison ---------------------------------------------
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(T, X, u_true, cmap='viridis', edgecolor='none')
    ax1.set_title("Exact solution")
    ax1.set_xlabel("t"); ax1.set_ylabel("x"); ax1.set_zlabel("u")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, u_pred, cmap='viridis', edgecolor='none')
    ax2.set_title("CN-PINN prediction")
    ax2.set_xlabel("t"); ax2.set_ylabel("x"); ax2.set_zlabel("u")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_3d_comparison.png"); plt.close(fig)

    # ---- pcolor error map ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(T, X, abs_err, cmap='jet', shading='auto')
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.set_title(r"Absolute error $|u - \hat{u}|$")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_error_map.png"); plt.close(fig)

    # ---- time-slice solution profiles --------------------------------------
    slices = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(slices)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, ts in zip(colors, slices):
        idx = np.abs(t_v - ts).argmin()
        ax.plot(x_v, u_true[idx],  '--', color=c, label=f"exact  t={ts:+.2f}")
        ax.plot(x_v, u_pred[idx], 'o', ms=3, markevery=8, color=c,
                label=f"pred  t={ts:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x, t)")
    ax.set_title("Solution profiles at selected times")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_time_slices.png"); plt.close(fig)

    # ---- error at same slices ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, ts in zip(colors, slices):
        idx = np.abs(t_v - ts).argmin()
        ax.plot(x_v, abs_err[idx], color=c, label=f"t={ts:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("absolute error")
    ax.set_title("Error profiles at selected times")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_error_time_slices.png"); plt.close(fig)

    # ---- error vs time -----------------------------------------------------
    # NOTE: for a soliton u = -2 sech^2(x - 4t) on x in [-2,2], the wave leaves
    # the visible spatial window near |t| ~ 0.5 (peak at x=4t).  At |t| close to
    # the temporal boundary, ||u_true|| along the x-slice is essentially zero,
    # so a *relative* L^2 error is dominated by machine-precision noise divided
    # by a vanishing denominator and shoots up spuriously.  We therefore plot
    # BOTH the absolute L^2 error and a "safe" relative error where the
    # denominator is floored by max_x ||u_true||.
    dx = x_v[1] - x_v[0]
    abs_L2_t = np.sqrt(np.trapz((u_true - u_pred) ** 2, dx=dx, axis=1))
    denom_t  = np.sqrt(np.trapz(u_true ** 2,           dx=dx, axis=1))
    floor    = 0.05 * denom_t.max()           # 5% of peak norm
    safe_rel = abs_L2_t / np.maximum(denom_t, floor)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    axs[0].plot(t_v, abs_L2_t, 'k-')
    axs[0].set_xlabel("t"); axs[0].set_ylabel(r"absolute $L^2$ error")
    axs[0].set_title("Absolute error evolution")
    axs[1].plot(t_v, safe_rel, 'b-')
    axs[1].set_xlabel("t"); axs[1].set_ylabel(r"relative $L^2$ error (floored)")
    axs[1].set_title("Relative error evolution")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_error_evolution.png"); plt.close(fig)

    # ---- loss curves -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in ("total", "ic", "bc", "pde"):
        ax.semilogy(history[k], label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training loss (log scale)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss_curve.png"); plt.close(fig)

    # ---- 3D spatiotemporal error surface -----------------------------------
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, abs_err, cmap='jet', edgecolor='none')
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_zlabel("|u-u_hat|")
    ax.set_title("3D absolute-error surface")
    fig.colorbar(surf, shrink=0.6)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_3d_error.png"); plt.close(fig)

    # ---- quantitative table -------------------------------------------------
    # NOTE: single-slice relative L2 = ||u_true - u_pred||_slice /
    # ||u_true||_slice can be inflated when the soliton has traveled outside
    # the domain (denominator ~ 0).  We report BOTH the relative L2 and the
    # absolute L2 per time slice, and mark slices where the peak amplitude of
    # the exact solution has dropped below 5 % of its global peak.
    global_peak = float(np.max(np.abs(u_true)))
    rows = []
    for ts in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        idx = np.abs(t_v - ts).argmin()
        slice_peak = float(np.max(np.abs(u_true[idx])))
        rows.append({
            "t":          ts,
            "L2 rel":     relative_l2(u_true[idx], u_pred[idx]),
            "L2 abs":     float(np.sqrt(np.mean((u_true[idx] - u_pred[idx]) ** 2))
                                * np.sqrt(len(u_true[idx]))),
            "L_inf":      l_inf     (u_true[idx], u_pred[idx]),
            "RMSE":       rmse      (u_true[idx], u_pred[idx]),
            "in window":  slice_peak > 0.05 * global_peak,
        })
    rows.append({
        "t": "global",
        "L2 rel": relative_l2(u_true, u_pred),
        "L_inf":  l_inf     (u_true, u_pred),
        "RMSE":   rmse      (u_true, u_pred),
    })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/results.csv", index=False)
    with open(f"{OUT_DIR}/table_errors.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4e",
                            caption="Errors of the CN-PINN for the "
                                    "single-soliton KdV forward problem.",
                            label="tab:exp1_errors"))
    print(df)
    return df


if __name__ == "__main__":
    model, history = train()
    df = visualise(model, history)
    print(f"\nAll outputs saved under: {OUT_DIR}")
