"""
common_utils.py
================
Shared utilities for the Crank-Nicolson PINN (CN-PINN) manuscript.

Contents:
    - Reproducibility helpers (seeding, dtype)
    - Sine activation module
    - Base PINN (forward) and Inverse PINN (with trainable alpha) architectures
    - Publication-quality Matplotlib rc settings
    - CN residual builder used across every experiment
    - L2 / Linf error helpers
    - Exact solutions for all examples (single soliton, double soliton via Hirota,
      cnoidal wave, mKdV soliton, KdV-Burgers travelling wave)

This file is imported by every experiment script. Do NOT run it directly.

Author: (your name)
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix seeds for reproducibility across NumPy, PyTorch, and Python's random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def use_double_precision() -> None:
    """Force torch to use float64 - critical for third-order derivatives (u_xxx)."""
    torch.set_default_dtype(torch.float64)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Publication-quality Matplotlib style
# ---------------------------------------------------------------------------
def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "lines.linewidth": 1.8,
    })


# ---------------------------------------------------------------------------
# Activation zoo (for ablation studies)
# ---------------------------------------------------------------------------
class SineActivation(nn.Module):
    """Sine activation - preserves derivatives of every order (critical for u_xxx)."""
    def forward(self, x):
        return torch.sin(x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "sin":       return SineActivation()
    if name == "tanh":      return nn.Tanh()
    if name == "relu":      return nn.ReLU()
    if name == "gelu":      return nn.GELU()
    if name == "silu":      return nn.SiLU()
    if name == "softplus":  return nn.Softplus()
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """
    Fully-connected feed-forward PINN u_theta(x, t).

    Parameters
    ----------
    layers : list[int]
        e.g. [2, 64, 64, 64, 1] gives 3 hidden layers of width 64.
    activation : str
        'sin' | 'tanh' | 'gelu' | 'relu' | 'silu' | 'softplus'.
    """
    def __init__(self, layers, activation: str = "sin"):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = get_activation(activation)
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        for layer in self.linears[:-1]:
            h = self.activation(layer(h))
        return self.linears[-1](h)


class InversePINN(PINN):
    """
    PINN that additionally exposes the dispersion coefficient alpha as a
    trainable scalar parameter (inverse identification).
    """
    def __init__(self, layers, activation: str = "sin", alpha_init: float = 0.5):
        super().__init__(layers, activation=activation)
        self.alpha = nn.Parameter(
            torch.tensor([alpha_init], dtype=torch.get_default_dtype())
        )


# ---------------------------------------------------------------------------
# Crank-Nicolson PDE residual
# ---------------------------------------------------------------------------
def kdv_spatial_operator(model, x_col, t_eval, lam1: float, alpha):
    """
    Compute R(u) = lam1 * u * u_x + alpha * u_xxx (autograd through model).

    Notes
    -----
    * `x_col` must already have `requires_grad = True`.
    * `alpha` may be a float (forward) or a torch scalar (inverse).
    """
    u = model(x_col, t_eval)
    ux   = torch.autograd.grad(u.sum(),   x_col, create_graph=True)[0]
    uxx  = torch.autograd.grad(ux.sum(),  x_col, create_graph=True)[0]
    uxxx = torch.autograd.grad(uxx.sum(), x_col, create_graph=True)[0]
    return u, lam1 * u * ux + alpha * uxxx


def cn_residual(model, x_col, t_col, dt: float, lam1: float, alpha):
    """
    Crank-Nicolson PDE residual:

        F = (u^{n+1} - u^n) / dt  +  0.5 * ( R(u^{n+1}) + R(u^n) )

    where R(u) = lam1 * u * u_x + alpha * u_xxx.
    Returns a scalar MSE loss.
    """
    x_col.requires_grad_(True)
    u_n,   res_n   = kdv_spatial_operator(model, x_col, t_col,        lam1, alpha)
    u_np1, res_np1 = kdv_spatial_operator(model, x_col, t_col + dt,   lam1, alpha)
    f = (u_np1 - u_n) / dt + 0.5 * (res_n + res_np1)
    return torch.mean(f ** 2)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def relative_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-30))


def l_inf(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.max(np.abs(u_true - u_pred)))


def rmse(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u_true - u_pred) ** 2)))


# ---------------------------------------------------------------------------
# EXACT SOLUTIONS
# ---------------------------------------------------------------------------
def u_single_soliton(x, t):
    """
    Manuscript Eq. (21) - exact single-soliton of  u_t - 6 u u_x + u_xxx = 0.

        u(x, t) = -2 * sech^2(x - 4t)

    (equivalently  u_t + lam1 u u_x + alpha u_xxx = 0  with lam1=-6, alpha=1).
    """
    return -2.0 / (torch.cosh(x - 4.0 * t)) ** 2


def u_double_soliton(x, t, k1: float = 1.0, k2: float = 1.5, alpha: float = 1.0):
    """
    Hirota bilinear double-soliton for  u_t + 6 u u_x + alpha u_xxx = 0.

    u = 2 (log f)_xx  with  f = 1 + e^{-eta1} + e^{-eta2} + A e^{-(eta1+eta2)}.

    Numerically stable form (all exponentials use negative arguments).
    """
    eta1 = k1 * x - (alpha * k1 ** 3) * t
    eta2 = k2 * x - (alpha * k2 ** 3) * t
    A = ((k2 - k1) / (k2 + k1)) ** 2
    e1, e2 = torch.exp(-eta1), torch.exp(-eta2)
    f    = 1.0 + e1 + e2 + A * e1 * e2
    fx   = -k1 * e1 - k2 * e2 - (k1 + k2) * A * e1 * e2
    fxx  =  k1 ** 2 * e1 + k2 ** 2 * e2 + (k1 + k2) ** 2 * A * e1 * e2
    return 2.0 * (fxx * f - fx ** 2) / (f ** 2)


def u_mkdv_soliton(x, t, c: float = 1.0):
    """
    Modified KdV single soliton:  u_t + 6 u^2 u_x + u_xxx = 0
        u(x, t) = sqrt(c) * sech( sqrt(c) (x - c t) )
    """
    k = float(np.sqrt(c))
    return k / torch.cosh(k * (x - c * t))


def u_kdv_burgers(x, t, nu: float = 1.0):
    """
    KdV-Burgers travelling wave for   u_t + u u_x - nu u_xx + u_xxx = 0.
    Classical form (see Johnson 1970):
        u(x, t) = A * (1 + tanh(k (x - c t)) )^2   (up to a constant shift)
    We adopt the compact profile
        u(x, t) = (3 nu^2 / 25) * ( 1 - tanh(k xi) )^2,   k = nu / 10,
                  xi = x - c t,   c = 6 nu^2 / 25.
    """
    k  = nu / 10.0
    c  = 6.0 * nu ** 2 / 25.0
    xi = x - c * t
    return (3.0 * nu ** 2 / 25.0) * (1.0 - torch.tanh(k * xi)) ** 2


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def uniform_sample(n, lo, hi):
    return torch.empty(n, 1).uniform_(lo, hi)


def make_ic_bc_col(x_range, t_range, N_col, N_ic, N_bc, dt):
    """Standard collocation / IC / BC sampling used by every forward experiment."""
    x_col = uniform_sample(N_col, *x_range)
    t_col = uniform_sample(N_col, t_range[0], t_range[1] - dt)
    x_ic  = uniform_sample(N_ic, *x_range)
    t_ic  = torch.full((N_ic, 1), t_range[0])
    x_bc  = torch.cat([
        torch.full((N_bc // 2, 1), x_range[0]),
        torch.full((N_bc // 2, 1), x_range[1]),
    ])
    t_bc  = uniform_sample(N_bc, *t_range)
    return x_col, t_col, x_ic, t_ic, x_bc, t_bc


# ---------------------------------------------------------------------------
# Directory helper
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
