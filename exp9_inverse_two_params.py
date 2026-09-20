"""
Experiment 9 - Joint identification of BOTH parameters (lambda_1, alpha)
=========================================================================

We now let *both* the convective coefficient lambda_1 and the dispersion
coefficient alpha be trainable, using the same double-soliton reference
solution.  This is a strictly harder inverse problem than Experiment 2 and
provides a stronger validation of the method.

True parameters      : lambda_1 = 6.0,   alpha = 1.0
Initial guesses      : lambda_1 = 3.0,   alpha = 0.5

Outputs (in ./plots/exp9_two_params/):
    * fig_joint_convergence.png
    * fig_reconstruction.png
    * fig_loss_curve.png
    * results.csv / table_two_params.tex
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, u_double_soliton,
    relative_l2, l_inf, rmse, ensure_dir, DEVICE,
)

OUT_DIR = ensure_dir("plots/exp9_two_params")
EPOCHS  = 2000
LR      = 5e-3
LAYERS  = [2, 32, 32, 32, 1]
LAM1_TRUE, LAM1_INIT   = 6.0, 3.0
ALPHA_TRUE, ALPHA_INIT = 1.0, 0.5
DT      = 0.01
X_RANGE = (-10.0, 10.0)
T_RANGE = (-1.0,   1.0)
N_COL   = 6000
N_OBS   = 2000
N_IC    = 500
IC_WEIGHT = 10.0

use_double_precision()
set_publication_style()


class TwoParamPINN(PINN):
    def __init__(self, layers, lam1_init, alpha_init):
        super().__init__(layers, "sin")
        self.lam1  = nn.Parameter(torch.tensor([lam1_init]))
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))


def cn_residual(model, x_col, t_col, dt):
    def R(t_eval):
        u = model(x_col, t_eval)
        ux   = torch.autograd.grad(u.sum(),   x_col, create_graph=True)[0]
        uxx  = torch.autograd.grad(ux.sum(),  x_col, create_graph=True)[0]
        uxxx = torch.autograd.grad(uxx.sum(), x_col, create_graph=True)[0]
        return u, model.lam1 * u * ux + model.alpha * uxxx
    x_col.requires_grad_(True)
    un, Rn = R(t_col); unp1, Rnp1 = R(t_col + dt)
    return torch.mean(((unp1 - un) / dt + 0.5 * (Rn + Rnp1)) ** 2)


def train():
    set_seed(42)
    model = TwoParamPINN(LAYERS, LAM1_INIT, ALPHA_INIT).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_obs = torch.empty(N_OBS, 1).uniform_(*X_RANGE).to(DEVICE)
    t_obs = torch.empty(N_OBS, 1).uniform_(*T_RANGE).to(DEVICE)
    u_obs = u_double_soliton(x_obs, t_obs, alpha=ALPHA_TRUE)

    x_ic = torch.empty(N_IC, 1).uniform_(*X_RANGE).to(DEVICE)
    t_ic = torch.full((N_IC, 1), T_RANGE[0]).to(DEVICE)
    u_ic = u_double_soliton(x_ic, t_ic, alpha=ALPHA_TRUE)

    x_col = torch.empty(N_COL, 1).uniform_(*X_RANGE).to(DEVICE)
    t_col = torch.empty(N_COL, 1).uniform_(T_RANGE[0], T_RANGE[1] - DT).to(DEVICE)

    hist = {"total": [], "data": [], "ic": [], "pde": [],
            "lam1": [], "alpha": []}
    tic = time.time()
    for epoch in range(EPOCHS + 1):
        optim.zero_grad()
        ld = torch.mean((model(x_obs, t_obs) - u_obs) ** 2)
        li = torch.mean((model(x_ic,  t_ic ) - u_ic ) ** 2)
        lp = cn_residual(model, x_col, t_col, DT)
        loss = ld + lp + IC_WEIGHT * li
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        hist["total"].append(loss.item()); hist["data"].append(ld.item())
        hist["ic"].append(li.item());       hist["pde" ].append(lp.item())
        hist["lam1"].append(model.lam1.item())
        hist["alpha"].append(model.alpha.item())
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} | lam1 {model.lam1.item():.4f} | "
                  f"alpha {model.alpha.item():.4f} | loss {loss.item():.4e}"
                  f" | ({time.time()-tic:.1f}s)")
    return model, hist


def visualise(model, hist):
    # convergence trajectory
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(hist["lam1"], color="tab:red")
    axs[0].axhline(LAM1_TRUE, ls="--", color="k")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel(r"$\lambda_1$")
    axs[0].set_title(fr"$\lambda_1$: true={LAM1_TRUE}, "
                     fr"final={hist['lam1'][-1]:.4f}")

    axs[1].plot(hist["alpha"], color="tab:blue")
    axs[1].axhline(ALPHA_TRUE, ls="--", color="k")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel(r"$\alpha$")
    axs[1].set_title(fr"$\alpha$: true={ALPHA_TRUE}, "
                     fr"final={hist['alpha'][-1]:.4f}")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_joint_convergence.png"); plt.close(fig)

    # reconstruction
    x_v = np.linspace(*X_RANGE, 400); t_v = np.linspace(*T_RANGE, 400)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_double_soliton(Xt, Tt, alpha=ALPHA_TRUE).cpu().numpy().reshape(X.shape)
    err = np.abs(ut - up)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap in zip(
        axs, [ut, up, err],
        ["Exact", fr"CN-PINN ($\hat\lambda_1={hist['lam1'][-1]:.3f},"
                  fr"\ \hat\alpha={hist['alpha'][-1]:.3f}$)", "Abs error"],
        ["viridis", "viridis", "inferno"],
    ):
        im = ax.pcolormesh(T, X, data, cmap=cmap, shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_reconstruction.png"); plt.close(fig)

    # losses
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in ("total", "data", "ic", "pde"):
        ax.semilogy(hist[k], label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss_curve.png"); plt.close(fig)

    df = pd.DataFrame([{
        "param":  "lambda_1", "true": LAM1_TRUE,
        "identified": hist["lam1"][-1],
        "rel_err_%":  100 * abs(hist["lam1"][-1]  - LAM1_TRUE)  / abs(LAM1_TRUE),
    }, {
        "param":  "alpha",    "true": ALPHA_TRUE,
        "identified": hist["alpha"][-1],
        "rel_err_%":  100 * abs(hist["alpha"][-1] - ALPHA_TRUE) / abs(ALPHA_TRUE),
    }])
    df.to_csv(f"{OUT_DIR}/results.csv", index=False)
    with open(f"{OUT_DIR}/table_two_params.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.6f",
                            caption="Joint identification of "
                                    r"$\lambda_1$ and $\alpha$.",
                            label="tab:two_params"))
    print(df)


if __name__ == "__main__":
    model, hist = train()
    visualise(model, hist)
    print(f"\nAll outputs saved under: {OUT_DIR}")
