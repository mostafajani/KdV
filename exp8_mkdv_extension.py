"""
Experiment 8 - Extension to the modified KdV (mKdV) equation
=============================================================

The CN-PINN framework is not tied to the classical KdV nonlinearity.  Here we
demonstrate it on the *modified* KdV equation

        u_t + 6 u^2 u_x + u_xxx = 0

whose single-soliton solution is

        u(x, t) = sqrt(c) * sech( sqrt(c) (x - c t) ),   c > 0.

The only change is the residual operator R(u) = 6 u^2 u_x + u_xxx (the
cubic nonlinearity replaces the quadratic one).  We add a dedicated
`mkdv_cn_residual` inline so `common_utils` stays generic.

Outputs (in ./plots/exp8_mkdv/):
    * fig_3d_comparison.png / fig_error_map.png / fig_time_slices.png
    * fig_loss_curve.png / results.csv / table_mkdv.tex
"""

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, u_mkdv_soliton, make_ic_bc_col,
    relative_l2, l_inf, rmse, ensure_dir, DEVICE,
)

OUT_DIR = ensure_dir("plots/exp8_mkdv")
EPOCHS  = 2000
LR      = 1e-3
LAYERS  = [2, 32, 32, 32, 1]
DT      = 0.01
X_RANGE = (-6.0, 6.0)
T_RANGE = (-1.0, 1.0)
N_COL   = 5000
N_IC    = 500
N_BC    = 500
C_SPEED = 1.0             # soliton speed / amplitude parameter

use_double_precision()
set_publication_style()


def mkdv_cn_residual(model, x_col, t_col, dt):
    """CN residual for u_t + 6 u^2 u_x + u_xxx = 0."""
    def R(t_eval):
        u = model(x_col, t_eval)
        ux   = torch.autograd.grad(u.sum(),   x_col, create_graph=True)[0]
        uxx  = torch.autograd.grad(ux.sum(),  x_col, create_graph=True)[0]
        uxxx = torch.autograd.grad(uxx.sum(), x_col, create_graph=True)[0]
        return u, 6.0 * u ** 2 * ux + uxxx
    x_col.requires_grad_(True)
    un,   Rn   = R(t_col)
    unp1, Rnp1 = R(t_col + dt)
    f = (unp1 - un) / dt + 0.5 * (Rn + Rnp1)
    return torch.mean(f ** 2)


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
    u_ic_true = u_mkdv_soliton(x_ic, t_ic, c=C_SPEED)
    u_bc_true = u_mkdv_soliton(x_bc, t_bc, c=C_SPEED)

    hist = {"total": [], "ic": [], "bc": [], "pde": []}
    tic = time.time()
    for epoch in range(EPOCHS + 1):
        optim.zero_grad()
        li = torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
        lb = torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
        lp = mkdv_cn_residual(model, x_col, t_col, DT)
        loss = li + lb + lp
        loss.backward(); optim.step()
        hist["total"].append(loss.item()); hist["ic"].append(li.item())
        hist["bc"].append(lb.item()); hist["pde"].append(lp.item())
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} | {loss.item():.4e} ({time.time()-tic:.1f}s)")
    return model, hist


def visualise(model, hist):
    x_v = np.linspace(*X_RANGE, 400); t_v = np.linspace(*T_RANGE, 300)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_mkdv_soliton(Xt, Tt, c=C_SPEED).cpu().numpy().reshape(X.shape)
    err = np.abs(ut - up)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(T, X, ut, cmap="cividis"); ax1.set_title("Exact mKdV soliton")
    ax1.set_xlabel("t"); ax1.set_ylabel("x"); ax1.set_zlabel("u")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, up, cmap="cividis"); ax2.set_title("CN-PINN")
    ax2.set_xlabel("t"); ax2.set_ylabel("x"); ax2.set_zlabel("u")
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
        ax.plot(x_v, up[i], 'o', ms=3, markevery=6, color=c, label=f"pred t={ts:+.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_title("mKdV soliton profiles")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_time_slices.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in ("total", "ic", "bc", "pde"):
        ax.semilogy(hist[k], label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("Loss")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss_curve.png"); plt.close(fig)

    df = pd.DataFrame([{
        "global L2 rel": relative_l2(ut, up),
        "global L_inf":  l_inf(ut, up),
        "global RMSE":   rmse(ut, up),
    }])
    df.to_csv(f"{OUT_DIR}/results.csv", index=False)
    with open(f"{OUT_DIR}/table_mkdv.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4e",
                            caption="CN-PINN accuracy on the modified KdV "
                                    "single soliton.",
                            label="tab:mkdv"))
    print(df)


if __name__ == "__main__":
    model, hist = train()
    visualise(model, hist)
    print(f"\nAll outputs saved under: {OUT_DIR}")
