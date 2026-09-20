"""
Experiment 5 - Activation-function ablation
============================================

Compare six activations on the single-soliton forward problem:
    sin, tanh, gelu, silu, softplus, relu

Justifies the claim in the manuscript that the periodic *sine* activation is
essential for representing high-order dispersion (u_xxx).

Outputs (in ./plots/exp5_activation/):
    * fig_activation_loss_curves.png
    * fig_activation_bar.png
    * activation_results.csv
    * table_activation.tex
"""

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, use_double_precision, set_publication_style,
    PINN, cn_residual, u_single_soliton, make_ic_bc_col,
    relative_l2, ensure_dir, DEVICE,
)

OUT_DIR = ensure_dir("plots/exp5_activation")
EPOCHS  = 2000
LR      = 1e-3
LAYERS  = [2, 32, 32, 32, 1]
LAM1    = -6.0
ALPHA   = 1.0
DT      = 0.01
X_RANGE = (-2.0, 2.0)
T_RANGE = (-2.0, 2.0)
N_COL   = 5000
N_IC    = 500
N_BC    = 500

ACTIVATIONS = ["sin", "tanh", "gelu", "silu", "softplus", "relu"]

use_double_precision()
set_publication_style()


def train(activation: str):
    set_seed(42)
    model = PINN(LAYERS, activation=activation).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_col, t_col, x_ic, t_ic, x_bc, t_bc = make_ic_bc_col(
        X_RANGE, T_RANGE, N_COL, N_IC, N_BC, DT
    )
    x_col, t_col = x_col.to(DEVICE), t_col.to(DEVICE)
    x_ic,  t_ic  = x_ic.to(DEVICE),  t_ic.to(DEVICE)
    x_bc,  t_bc  = x_bc.to(DEVICE),  t_bc.to(DEVICE)
    u_ic_true = u_single_soliton(x_ic, t_ic)
    u_bc_true = u_single_soliton(x_bc, t_bc)

    losses = []
    for _ in range(EPOCHS + 1):
        optim.zero_grad()
        loss = (torch.mean((model(x_ic, t_ic) - u_ic_true) ** 2)
              + torch.mean((model(x_bc, t_bc) - u_bc_true) ** 2)
              + cn_residual(model, x_col, t_col, DT, LAM1, ALPHA))
        loss.backward(); optim.step()
        losses.append(loss.item())

    # evaluate
    x_v = np.linspace(*X_RANGE, 200); t_v = np.linspace(*T_RANGE, 200)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_single_soliton(Xt, Tt).cpu().numpy().reshape(X.shape)
    return losses, relative_l2(ut, up)


if __name__ == "__main__":
    tic = time.time()
    losses_all, errors = {}, {}
    for act in ACTIVATIONS:
        print(f"[activation = {act}] training ...")
        L, e = train(act)
        losses_all[act] = L
        errors[act] = e
        print(f"   -> rel L2 = {e:.4e}")

    # loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for act, L in losses_all.items():
        ax.semilogy(L, label=act)
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss (log)")
    ax.set_title("Loss curves across activation functions")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_activation_loss_curves.png"); plt.close(fig)

    # bar chart of final errors
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ACTIVATIONS))
    vals = [errors[a] for a in ACTIVATIONS]
    ax.bar(x, vals, color=plt.cm.tab10(np.arange(len(x))))
    ax.set_xticks(x); ax.set_xticklabels(ACTIVATIONS)
    ax.set_yscale("log")
    ax.set_ylabel(r"relative $L^2$ error (log)")
    ax.set_title("Final error by activation")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_activation_bar.png"); plt.close(fig)

    df = pd.DataFrame({"activation": ACTIVATIONS,
                       "rel_L2": [errors[a] for a in ACTIVATIONS]})
    df.to_csv(f"{OUT_DIR}/activation_results.csv", index=False)
    with open(f"{OUT_DIR}/table_activation.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4e",
                            caption="Effect of the activation function on "
                                    "the CN-PINN accuracy.",
                            label="tab:activation"))
    print(df)
    print(f"\ndone in {(time.time()-tic)/60:.1f} min. outputs in {OUT_DIR}")
