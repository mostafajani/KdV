"""
Experiment 10 - Conservation-law diagnostic
============================================

The KdV equation  u_t - 6 u u_x + u_xxx = 0  admits infinitely many
conservation laws.  The first three are

    I_1(t) = int_Omega  u             dx      (mass)
    I_2(t) = int_Omega  u^2           dx      (momentum / energy)
    I_3(t) = int_Omega  ( u^3 + 0.5 u_x^2 ) dx (Hamiltonian)

For the single-soliton  u = -2 sech^2(x - 4t)  on a sufficiently wide domain,
these invariants are essentially constant in time.  A well-behaved numerical
scheme should preserve them - so we plot I_k(t) computed on the CN-PINN
solution to demonstrate the qualitative correctness of the learnt dynamics.

Outputs (in ./plots/exp10_conservation/):
    * fig_conservation.png      (I_1, I_2, I_3 vs time - PINN vs exact)
    * conservation.csv
    * table_conservation.tex
"""

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, cn_residual, u_single_soliton, make_ic_bc_col,
    ensure_dir, DEVICE,
)

OUT_DIR = ensure_dir("plots/exp10_conservation")
EPOCHS  = 5000
LR      = 1e-3
LAYERS  = [2, 64, 64, 64, 1]   # keep 64-wide here - conservation is a strict test
LAM1    = -6.0
ALPHA   =  1.0
DT      = 0.01
# Widen domain so the soliton  u = -2 sech^2(x - 4t)  stays fully inside
# for the analysis window.  Peak position = 4 t.  We evaluate invariants
# on |t| <= T_EVAL_MAX = 1.0, so we need L >= 4 * 1.0 + soliton_width ~ 6.
X_RANGE = (-10.0, 10.0)
T_RANGE = (-1.5,   1.5)
T_EVAL_MAX = 1.0                # only integrate invariants over |t| <= 1.0
N_COL   = 8000
N_IC    = 800
N_BC    = 800

use_double_precision()
set_publication_style()


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
    u_ic_true = u_single_soliton(x_ic, t_ic)
    u_bc_true = u_single_soliton(x_bc, t_bc)

    tic = time.time()
    for epoch in range(EPOCHS + 1):
        optim.zero_grad()
        loss = (torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
              + torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
              + cn_residual(model, x_col, t_col, DT, LAM1, ALPHA))
        loss.backward(); optim.step()
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} | loss {loss.item():.4e}"
                  f" | ({time.time()-tic:.1f}s)")
    return model


def compute_invariants(u_field: np.ndarray, ux_field: np.ndarray, dx: float):
    """
    Trapezoidal integration of the three invariants along the x-axis.
    Arrays are shaped (Nt, Nx).
    """
    I1 = np.trapz(u_field,                       dx=dx, axis=1)
    I2 = np.trapz(u_field ** 2,                  dx=dx, axis=1)
    I3 = np.trapz(u_field ** 3 + 0.5 * ux_field ** 2, dx=dx, axis=1)
    return I1, I2, I3


def analyse(model):
    # Evaluate invariants ONLY on the sub-window where the soliton is fully
    # inside the training domain, otherwise the integrals leak mass through
    # the boundary (a domain-truncation effect, not a scheme error).
    x_v = np.linspace(*X_RANGE, 800)
    t_v = np.linspace(-T_EVAL_MAX, T_EVAL_MAX, 200)
    dx  = x_v[1] - x_v[0]
    X, T = np.meshgrid(x_v, t_v)

    # PINN field & spatial derivative
    x_t = torch.tensor(X.flatten()[:, None], requires_grad=True).to(DEVICE)
    t_t = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    u   = model(x_t, t_t)
    ux  = torch.autograd.grad(u.sum(), x_t, create_graph=False)[0]
    up  = u.detach().cpu().numpy().reshape(X.shape)
    upx = ux.detach().cpu().numpy().reshape(X.shape)

    ut  = u_single_soliton(x_t, t_t).detach().cpu().numpy().reshape(X.shape)

    # exact analytic derivative (avoid needing autograd for exact)
    # d/dx [-2 sech^2(x-4t)] = -2 * -2 sech^2 * tanh = 4 sech^2 tanh
    xi = X - 4.0 * T
    sech = 1.0 / np.cosh(xi)
    utx = 4.0 * sech ** 2 * np.tanh(xi)

    I1p, I2p, I3p = compute_invariants(up, upx, dx)
    I1e, I2e, I3e = compute_invariants(ut, utx, dx)

    df = pd.DataFrame({
        "t":  t_v,
        "I1_pinn": I1p, "I1_exact": I1e,
        "I2_pinn": I2p, "I2_exact": I2e,
        "I3_pinn": I3p, "I3_exact": I3e,
    })
    df.to_csv(f"{OUT_DIR}/conservation.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, name in zip(
        axs, ["I1", "I2", "I3"],
        [r"$I_1=\int u\, dx$",
         r"$I_2=\int u^2\, dx$",
         r"$I_3=\int (u^3+\frac{1}{2} u_x^2)\, dx$"],
    ):
        ax.plot(t_v, df[f"{key}_exact"], "k--", label="exact")
        ax.plot(t_v, df[f"{key}_pinn"],  "r-",  label="CN-PINN")
        ax.set_xlabel("t"); ax.set_ylabel(name); ax.set_title(name); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_conservation.png"); plt.close(fig)

    # relative drift summary
    def drift(v):
        return float((v.max() - v.min()) / (abs(v.mean()) + 1e-30))

    summary = pd.DataFrame([{
        "invariant": "I1", "PINN drift": drift(I1p), "exact drift": drift(I1e)
    }, {
        "invariant": "I2", "PINN drift": drift(I2p), "exact drift": drift(I2e)
    }, {
        "invariant": "I3", "PINN drift": drift(I3p), "exact drift": drift(I3e)
    }])
    with open(f"{OUT_DIR}/table_conservation.tex", "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.4e",
                                 caption="Relative drift of the first three "
                                         "KdV invariants under CN-PINN dynamics.",
                                 label="tab:conservation"))
    print(summary)


if __name__ == "__main__":
    model = train()
    analyse(model)
    print(f"\nAll outputs saved under: {OUT_DIR}")
