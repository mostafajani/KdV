"""
Experiment 7 - Three-soliton interaction (forward problem, replaces cnoidal)
============================================================================

Governing equation :  u_t + 6 u u_x + u_xxx = 0
Exact solution     :  Hirota bilinear THREE-soliton (k1, k2, k3)
                      u(x,t) = 2 [log f(x,t)]_xx,
                      f = 1 + sum e_i + sum a_ij e_i e_j + a_ijk e_i e_j e_k
                      e_i = exp(k_i x - k_i^3 t)
                      a_ij = ((k_i - k_j)/(k_i + k_j))^2
                      a_ijk = a_12 a_13 a_23

Domain             :  (x, t) in [-10, 10] x [-1, 1]
Wavenumbers        :  k1 = 0.8, k2 = 1.2, k3 = 1.6

This is a genuinely more demanding forward test than the double-soliton: three
distinct waves collide and pass through each other with a phase shift, so the
network has to represent a strongly time-dependent multi-hump profile.

Outputs (in ./plots/exp7_triple/):
    * fig_3d_comparison.png    3D exact vs prediction
    * fig_error_map.png        absolute error pcolor
    * fig_time_slices.png      profiles at t = -0.5, 0, 0.5
    * fig_loss_curve.png       training loss
    * results.csv / table_triple.tex
"""

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, cn_residual, make_ic_bc_col,
    relative_l2, l_inf, rmse, ensure_dir, DEVICE,
)

OUT_DIR = ensure_dir("plots/exp7_triple")
EPOCHS  = 5000
LR      = 1e-3
LAYERS  = [2, 32, 32, 32, 1]
LAM1    = +6.0                # NOTE: opposite sign convention from exp1
ALPHA   =  1.0
DT      = 0.01
X_RANGE = (-10.0, 10.0)
T_RANGE = (-1.0,   1.0)
N_COL   = 6000
N_IC    = 600
N_BC    = 600

K1, K2, K3 = 0.8, 1.2, 1.6

use_double_precision()
set_publication_style()


# -----------------------------------------------------------------------------
# Exact three-soliton via Hirota's tau-function (POSITIVE eta convention)
# -----------------------------------------------------------------------------
def u_triple_soliton(x, t, k1=K1, k2=K2, k3=K3):
    """
    Exact three-soliton of  u_t + 6 u u_x + u_xxx = 0.

    Implementation uses analytic f, f_x, f_xx so that
        u = 2 (f * f_xx - f_x^2) / f^2
    is numerically exact.  Verified to satisfy the PDE to ~1e-5.
    """
    e1 = torch.exp(k1 * x - (k1 ** 3) * t)
    e2 = torch.exp(k2 * x - (k2 ** 3) * t)
    e3 = torch.exp(k3 * x - (k3 ** 3) * t)

    a12 = ((k1 - k2) / (k1 + k2)) ** 2
    a13 = ((k1 - k3) / (k1 + k3)) ** 2
    a23 = ((k2 - k3) / (k2 + k3)) ** 2
    a123 = a12 * a13 * a23

    E12  = e1 * e2
    E13  = e1 * e3
    E23  = e2 * e3
    E123 = e1 * e2 * e3

    f = 1 + e1 + e2 + e3 + a12 * E12 + a13 * E13 + a23 * E23 + a123 * E123

    fx = (k1 * e1 + k2 * e2 + k3 * e3
          + a12 * (k1 + k2) * E12
          + a13 * (k1 + k3) * E13
          + a23 * (k2 + k3) * E23
          + a123 * (k1 + k2 + k3) * E123)

    fxx = (k1 ** 2 * e1 + k2 ** 2 * e2 + k3 ** 2 * e3
           + a12 * (k1 + k2) ** 2 * E12
           + a13 * (k1 + k3) ** 2 * E13
           + a23 * (k2 + k3) ** 2 * E23
           + a123 * (k1 + k2 + k3) ** 2 * E123)

    return 2.0 * (f * fxx - fx ** 2) / f ** 2


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train():
    set_seed(42)
    model = PINN(LAYERS, "sin").to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_col, t_col, x_ic, t_ic, x_bc, t_bc = make_ic_bc_col(
        X_RANGE, T_RANGE, N_COL, N_IC, N_BC, DT
    )
    x_col, t_col = x_col.to(DEVICE), t_col.to(DEVICE)
    x_ic,  t_ic  = x_ic.to(DEVICE),  t_ic.to(DEVICE)
    x_bc,  t_bc  = x_bc.to(DEVICE),  t_bc.to(DEVICE)
    u_ic_true = u_triple_soliton(x_ic, t_ic)
    u_bc_true = u_triple_soliton(x_bc, t_bc)

    hist = {"total": [], "ic": [], "bc": [], "pde": []}
    tic = time.time()
    for epoch in range(EPOCHS + 1):
        optim.zero_grad()
        li = torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
        lb = torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
        lp = cn_residual(model, x_col, t_col, DT, LAM1, ALPHA)
        loss = li + lb + lp
        loss.backward(); optim.step()
        hist["total"].append(loss.item()); hist["ic"].append(li.item())
        hist["bc"].append(lb.item());       hist["pde"].append(lp.item())
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} | total {loss.item():.4e}"
                  f" | ic {li.item():.2e}  bc {lb.item():.2e}"
                  f"  pde {lp.item():.2e}  ({time.time()-tic:.1f}s)")
    return model, hist


def visualise(model, hist):
    x_v = np.linspace(*X_RANGE, 400); t_v = np.linspace(*T_RANGE, 400)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_triple_soliton(Xt, Tt).cpu().numpy().reshape(X.shape)
    err = np.abs(ut - up)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(T, X, ut, cmap="viridis", edgecolor="none")
    ax1.set_title("Exact three-soliton"); ax1.set_xlabel("t"); ax1.set_ylabel("x"); ax1.set_zlabel("u")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, up, cmap="viridis", edgecolor="none")
    ax2.set_title("CN-PINN prediction"); ax2.set_xlabel("t"); ax2.set_ylabel("x"); ax2.set_zlabel("u")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_3d_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(T, X, err, cmap="inferno", shading="auto")
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("Absolute error")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_error_map.png"); plt.close(fig)

    slices = [-0.5, 0.0, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(slices)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, ts in zip(colors, slices):
        i = np.abs(t_v - ts).argmin()
        ax.plot(x_v, ut[i], '--', color=c, label=f"exact t={ts:+.2f}")
        ax.plot(x_v, up[i], 'o', ms=3, markevery=8, color=c, label=f"pred t={ts:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_title("Three-soliton profiles")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_time_slices.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in ("total", "ic", "bc", "pde"):
        ax.semilogy(hist[k], label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("Training loss")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss_curve.png"); plt.close(fig)

    df = pd.DataFrame([{
        "global L2 rel": relative_l2(ut, up),
        "global L_inf":  l_inf(ut, up),
        "global RMSE":   rmse(ut, up),
    }])
    df.to_csv(f"{OUT_DIR}/results.csv", index=False)
    with open(f"{OUT_DIR}/table_triple.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4e",
                            caption="CN-PINN accuracy on the three-soliton "
                                    "interaction of KdV.",
                            label="tab:triple"))
    print(df)


if __name__ == "__main__":
    model, hist = train()
    visualise(model, hist)
    print(f"\nAll outputs saved under: {OUT_DIR}")
