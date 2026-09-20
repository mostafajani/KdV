"""
Experiment 6 - Temporal convergence order O(dt^2) of the CN-PINN
=================================================================

The Crank-Nicolson scheme is second-order accurate in time.  Here we verify
numerically that the CN-PINN inherits this O(dt^2) behaviour by training the
same architecture for a range of dt values on the single-soliton problem and
plotting log(rel_L2) vs log(dt) on a log-log axis.  A least-squares fit gives
the observed convergence slope.

Outputs (in ./plots/exp6_dt_convergence/):
    * fig_dt_convergence.png (log-log plot with slope annotation)
    * dt_convergence.csv
    * table_dt_convergence.tex
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

OUT_DIR = ensure_dir("plots/exp6_dt_convergence")
EPOCHS  = 2000
LR      = 1e-3
LAYERS  = [2, 32, 32, 32, 1]
LAM1    = -6.0
ALPHA   = 1.0
X_RANGE = (-2.0, 2.0)
T_RANGE = (-2.0, 2.0)
N_COL   = 5000
N_IC    = 500
N_BC    = 500
DT_LIST = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

use_double_precision()
set_publication_style()


def train_dt(dt: float) -> float:
    set_seed(42)
    model = PINN(LAYERS, "sin").to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    x_col, t_col, x_ic, t_ic, x_bc, t_bc = make_ic_bc_col(
        X_RANGE, T_RANGE, N_COL, N_IC, N_BC, dt
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

    x_v = np.linspace(*X_RANGE, 200); t_v = np.linspace(*T_RANGE, 200)
    X, T = np.meshgrid(x_v, t_v)
    Xt = torch.tensor(X.flatten()[:, None]).to(DEVICE)
    Tt = torch.tensor(T.flatten()[:, None]).to(DEVICE)
    with torch.no_grad():
        up = model(Xt, Tt).cpu().numpy().reshape(X.shape)
    ut = u_single_soliton(Xt, Tt).cpu().numpy().reshape(X.shape)
    return relative_l2(ut, up)


if __name__ == "__main__":
    tic = time.time()
    errs = []
    for dt in DT_LIST:
        print(f"[dt = {dt}] training ...")
        e = train_dt(dt)
        errs.append(e)
        print(f"   rel L2 = {e:.4e}")

    dts = np.array(DT_LIST)
    errs = np.array(errs)

    # slope of log(err) vs log(dt)
    slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)
    print(f"observed convergence order  ~  {slope:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(dts, errs, "o-", ms=8, color="tab:red", label="CN-PINN")
    ref = errs[0] * (dts / dts[0]) ** 2
    ax.loglog(dts, ref, "k--", label=r"reference $O(\Delta t^{2})$")
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"relative $L^{2}$ error")
    ax.set_title(fr"Temporal convergence: observed slope $\approx$ {slope:.2f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_dt_convergence.png"); plt.close(fig)

    df = pd.DataFrame({"dt": dts, "rel_L2": errs})
    df["log_dt"]     = np.log(dts)
    df["log_rel_L2"] = np.log(errs)
    df.to_csv(f"{OUT_DIR}/dt_convergence.csv", index=False)
    with open(f"{OUT_DIR}/table_dt_convergence.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4e",
                            caption=f"Temporal convergence of the CN-PINN "
                                    f"(fitted slope = {slope:.2f}).",
                            label="tab:dt_convergence"))
    print(df)
    print(f"\ndone in {(time.time()-tic)/60:.1f} min. outputs in {OUT_DIR}")
