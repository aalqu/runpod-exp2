"""
Sanity checks for fd_4d_core.py — spec §11.

Zero-stochasticity limit: when xi_v = xi_r = 0 and theta_v/theta_r land
exactly on grid points, the 4D solver must reproduce the 2D fd_solve result
at the (theta_v, theta_r) slice to within 1e-3.

Grid design: v_max=0.20, Nv=10  →  v[2] = 0.04 = theta_v  ✓
             r_max=0.20, Nr=10  →  r[2] = 0.04 = theta_r  ✓
             Nt=100 reduces Strang splitting error (two W(dt/2) vs one W(dt))
             below the 1e-3 tolerance while keeping the test under ~60 s.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve()
while not (ROOT / "fd_core.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from fd_core import asymp_goalreach, fd_solve
from fd_4d_core import fd_solve_4d

# ── Shared parameters ─────────────────────────────────────────────────────────

MU       = 0.08
THETA_V  = 0.04          # variance = sigma^2  →  sigma = 0.20
THETA_R  = 0.04
T        = 1.0
A        = 2.5
GOAL     = 1.10
D, U     = -5.0, 3.0

# Grid: theta_v and theta_r land exactly on grid nodes
GRID = dict(Nw=40, Nv=10, Nr=10, Nt=100, A=A, v_max=0.20, r_max=0.20)

SIGMA = float(np.sqrt(THETA_V))


def _2d_reference():
    """2D fd_solve with sigma=sqrt(theta_v), r=theta_r."""
    w, V, Pi = fd_solve(
        mu=MU, r=THETA_R, sigma=SIGMA,
        T=T, A=A,
        Nw=GRID["Nw"], Nt=GRID["Nt"],
        d=D, u=U,
        utility_fn=lambda w: (np.asarray(w) >= GOAL).astype(float),
        asymptotic_fn=lambda w, tau: asymp_goalreach(
            w, tau, sigma=SIGMA, d=D, u=U, goal=GOAL
        ),
    )
    return w, V, Pi


def _4d_zero_stoch():
    """4D fd_solve_4d with xi_v=xi_r=0."""
    return fd_solve_4d(
        mu=MU, r0=THETA_R, v0=THETA_V,
        T=T, goal_mult=GOAL, w0=1.0,
        kappa_v=5.0,  theta_v=THETA_V, xi_v=0.0,
        kappa_r=0.50, theta_r=THETA_R, xi_r=0.0,
        rho_Wv=0.0, rho_Wr=0.0, rho_vr=0.0,
        **GRID,
        d=D, u=U,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_zero_stoch_value_function_matches_2d():
    """
    At (theta_v, theta_r) the 4D value function must match the 2D reference
    within 1e-3 (spec §11).
    """
    _, V2d, _ = _2d_reference()
    w4d, v4d, r4d, V4d, _ = _4d_zero_stoch()

    iv0 = int(np.argmin(np.abs(v4d - THETA_V)))
    ir0 = int(np.argmin(np.abs(r4d - THETA_R)))

    # Confirm theta_v/theta_r land exactly on a grid node
    assert abs(v4d[iv0] - THETA_V) < 1e-12, f"theta_v not on grid: v[{iv0}]={v4d[iv0]}"
    assert abs(r4d[ir0] - THETA_R) < 1e-12, f"theta_r not on grid: r[{ir0}]={r4d[ir0]}"

    err = float(np.max(np.abs(V4d[:, iv0, ir0] - V2d)))
    assert err < 1e-3, (
        f"Max V error at (theta_v, theta_r) slice: {err:.2e} ≥ 1e-3\n"
        f"  iv0={iv0}, v4d[iv0]={v4d[iv0]}\n"
        f"  ir0={ir0}, r4d[ir0]={r4d[ir0]}\n"
        f"  V4d[20:25, iv0, ir0] = {V4d[20:25, iv0, ir0]}\n"
        f"  V2d[20:25]           = {V2d[20:25]}"
    )


def test_zero_stoch_policy_matches_2d():
    """
    At (theta_v, theta_r) the 4D optimal policy must match the 2D reference
    within 0.05 (policies converge together with V, but more sensitive to
    the kink in V near the goal).
    """
    _, _, Pi2d = _2d_reference()
    _, v4d, r4d, _, Pi4d = _4d_zero_stoch()

    iv0 = int(np.argmin(np.abs(v4d - THETA_V)))
    ir0 = int(np.argmin(np.abs(r4d - THETA_R)))

    err = float(np.max(np.abs(Pi4d[:, iv0, ir0] - Pi2d)))
    assert err < 0.05, (
        f"Max Pi error at (theta_v, theta_r) slice: {err:.2e} ≥ 0.05"
    )
