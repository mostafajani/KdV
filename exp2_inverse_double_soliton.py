"""
Experiment 2 - Inverse problem: identify alpha for double-soliton KdV
======================================================================

Governing equation :  u_t + 6 u u_x + alpha u_xxx = 0    (alpha unknown)
True dispersion    :  alpha* = 1.0
Initial guess      :  alpha_0 = 0.5
Exact solution     :  Hirota bilinear double-soliton  (k1=1.0, k2=1.5)
Domain             :  (x, t) in [-10, 10] x [-1, 1],  dt = 0.01
Network            :  [2, 64, 64, 64, 1] + sine + trainable alpha
Optimizer          :  Adam, lr = 5e-4, 5000 epochs

Outputs (in ./plots/exp2_inverse/):
    * alpha convergence trajectory                     -> fig_alpha_convergence.png
    * exact / prediction / abs-error side-by-side      -> fig_reconstruction.png
    * loss-curve panel                                 -> fig_loss_curve.png
    * time-slice comparison                            -> fig_time_slices.png
    * table_alpha.tex + results.csv

Run with:  python exp2_inverse_double_soliton.py
"""

import os, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    InversePINN, cn_residual, u_double_soliton,
    relative_l2, l_inf, rmse, ensure_dir, DEVICE,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUT_DIR   = ensure_dir("plots/exp2_inverse")
EPOCHS    = 2000
LR        = 5e-3
LAYERS    = [2, 32, 32, 32, 1] #was 64
LAM1      =  6.0                # sign convention: u_t + 6 u u_x + alpha u_xxx = 0
ALPHA_TRUE = 1.0
ALPHA_INIT = 0.5
DT        = 0.01
X_RANGE   = (-10.0, 10.0)
T_RANGE   = (-1.0,   1.0)
N_COL     = 6000
N_OBS     = 2000
N_IC      = 500
IC_WEIGHT = 10.0

set_seed(42)
use_double_precision()
set_publication_style()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train():
    model = InversePINN(LAYERS, activation="sin", alpha_init=ALPHA_INIT).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # observation "sensors"
    x_obs = torch.empty(N_OBS, 1).uniform_(*X_RANGE).to(DEVICE)
    t_obs = torch.empty(N_OBS, 1).uniform_(*T_RANGE).to(DEVICE)
    u_obs = u_double_soliton(x_obs, t_obs, alpha=ALPHA_TRUE)

    # initial-condition slice at t = t_range[0]
    x_ic = torch.empty(N_IC, 1).uniform_(*X_RANGE).to(DEVICE)
    t_ic = torch.full((N_IC, 1), T_RANGE[0]).to(DEVICE)
    u_ic = u_double_soliton(x_ic, t_ic, alpha=ALPHA_TRUE)

    # collocation
    x_col = torch.empty(N_COL, 1).uniform_(*X_RANGE).to(DEVICE)
    t_col = torch.empty(N_COL, 1).uniform_(T_RANGE[0], T_RANGE[1] - DT).to(DEVICE)

    hist = {"total": [], "data": [], "ic": [], "pde": [], "alpha": []}
    print(f"Training inverse CN-PINN on {DEVICE} ...")
    tic = time.time()

    for epoch in range(EPOCHS + 1):
        optim.zero_grad()

        loss_data = torch.mean((model(x_obs, t_obs) - u_obs) ** 2)
        loss_ic   = torch.mean((model(x_ic,  t_ic ) - u_ic) ** 2)
        loss_pde  = cn_residual(model, x_col, t_col, DT, LAM1, model.alpha)

        loss = loss_data + loss_pde + IC_WEIGHT * loss_ic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        hist["total"].append(loss.item())
        hist["data" ].append(loss_data.item())
        hist["ic"   ].append(loss_ic.item())
        hist["pde"  ].append(loss_pde.item())
        hist["alpha"].append(model.alpha.item())

        if epoch % 50 == 0:
            print(f"  epoch {epoch:5d} | alpha {model.alpha.item():.4f} | "
                  f"loss {loss.item():.4e} | ({time.time()-tic:.1f}s)")

    print(f"training done in {time.time()-tic:.1f}s")
    return model, hist


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def visualise(model, hist):
    # ---- alpha convergence -------------------------------------------------
    alpha_hist = np.array(hist["alpha"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha_hist, "r-", label=r"identified $\alpha$")
    ax.axhline(ALPHA_TRUE, color="k", ls="--", label=fr"true $\alpha={ALPHA_TRUE}$")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\alpha$")
    ax.set_title("Dispersion parameter convergence")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_alpha_convergence.png"); plt.close(fig)

    # ---- exact / predicted / error side-by-side ---------------------------
    x_v = np.linspace(*X_RANGE, 400)
    t_v = np.linspace(*T_RANGE, 400)
    X, T = np.meshgrid(x_v, t_v)
    X_t = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    T_t = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        u_pred = model(X_t, T_t).cpu().numpy().reshape(X.shape)
    u_true = u_double_soliton(X_t, T_t, alpha=ALPHA_TRUE).cpu().numpy().reshape(X.shape)
    err = np.abs(u_true - u_pred)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap in zip(
        axs, [u_true, u_pred, err],
        ["Exact double-soliton",
         fr"CN-PINN ($\hat\alpha={alpha_hist[-1]:.4f}$)",
         "Absolute error"],
        ["viridis", "viridis", "inferno"],
    ):
        im = ax.pcolormesh(T, X, data, cmap=cmap, shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_reconstruction.png"); plt.close(fig)

    # ---- loss curves -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in ("total", "data", "ic", "pde"):
        ax.semilogy(hist[k], label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Inverse-problem training loss")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss_curve.png"); plt.close(fig)

    # ---- time slices -------------------------------------------------------
    slices = [-0.75, -0.25, 0.0, 0.25, 0.75]
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(slices)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, ts in zip(colors, slices):
        idx = np.abs(t_v - ts).argmin()
        ax.plot(x_v, u_true[idx], '--', color=c, label=f"exact t={ts:+.2f}")
        ax.plot(x_v, u_pred[idx], 'o', ms=3, markevery=6, color=c,
                label=f"pred  t={ts:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title("Double-soliton profiles")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_time_slices.png"); plt.close(fig)

    # ---- summary -----------------------------------------------------------
    final_alpha = alpha_hist[-1]
    df = pd.DataFrame([{
        "true alpha":     ALPHA_TRUE,
        "identified":     final_alpha,
        "abs error":      abs(ALPHA_TRUE - final_alpha),
        "rel error (%)":  100 * abs(ALPHA_TRUE - final_alpha) / ALPHA_TRUE,
        "global L2 rel":  relative_l2(u_true, u_pred),
        "global L_inf":   l_inf(u_true, u_pred),
        "global RMSE":    rmse(u_true, u_pred),
    }])
    df.to_csv(f"{OUT_DIR}/results.csv", index=False)
    with open(f"{OUT_DIR}/table_alpha.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.6f",
                            caption="Inverse identification of the KdV "
                                    "dispersion parameter.",
                            label="tab:exp2_alpha"))
    print(df)


if __name__ == "__main__":
    model, hist = train()
    visualise(model, hist)
    print(f"\nAll outputs saved under: {OUT_DIR}")
