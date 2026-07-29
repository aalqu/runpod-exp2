"""
TDD tests for _gbm_mc_eval and the nn_policy batch-inference path.

Covers the shape-mismatch bug reported for n_assets=5:
  "operands could not be broadcast together with shapes (2000,2000) (4000000,2000)"
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_experiment import _gbm_mc_eval, _extract_policy_fn, make_config


class _FakeMarketData:
    """Minimal MarketData stand-in for testing."""
    def __init__(self, n_assets):
        rng = np.random.default_rng(0)
        self.n       = n_assets
        self.tickers = [f"A{i}" for i in range(n_assets)]
        self.mu_ann  = 0.03 + rng.uniform(0.04, 0.08, n_assets)
        self.r       = 0.03
        sig          = rng.uniform(0.15, 0.25, n_assets)
        rho          = np.eye(n_assets) + 0.3 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        self.omega   = np.outer(sig, sig) * rho
        T            = 252
        L            = np.linalg.cholesky(self.omega + 1e-10 * np.eye(n_assets))
        Z            = rng.standard_normal((T, n_assets))
        dt           = 1 / 252
        self.log_ret = (self.mu_ann - 0.5 * np.diag(self.omega)) * dt + (Z @ L.T) * np.sqrt(dt)


def _const_policy(n_assets):
    """Returns a policy function that outputs equal weights regardless of input."""
    w = np.full(n_assets, 1.0 / n_assets)
    def policy_fn(w_norm, tau):
        batch = np.asarray(w_norm)
        if batch.ndim == 0:
            return w
        return np.tile(w, (len(batch), 1))
    return policy_fn


# ── Scalar policy (works for both n=1 and n>1) ───────────────────────────────

def test_gbm_mc_eval_n1():
    """n=1: output dict has expected keys and goal_prob in [0,1]."""
    mkt = _FakeMarketData(1)
    pfn = _const_policy(1)
    result = _gbm_mc_eval(pfn, mkt, 1.0, 1.10, n_mc=200, seed=0)
    assert "mc_goal_prob" in result
    assert 0.0 <= result["mc_goal_prob"] <= 1.0
    assert "mc_mean_wealth" in result
    assert result["mc_mean_wealth"] > 0


def test_gbm_mc_eval_n5_shape():
    """n=5: policy must return (n_mc, 5) per step — no broadcast error."""
    mkt = _FakeMarketData(5)
    pfn = _const_policy(5)
    result = _gbm_mc_eval(pfn, mkt, 1.0, 1.10, n_mc=200, seed=0)
    assert "mc_goal_prob" in result
    assert 0.0 <= result["mc_goal_prob"] <= 1.0


def test_gbm_mc_eval_n5_torch_policy():
    """
    n=5: the nn_policy closure returned by _extract_policy_fn must produce
    shape (n_mc, 5) when called with a (n_mc,) w_norm batch — this is the
    exact call pattern used by _gbm_mc_eval.
    """
    import torch
    from comparisons.core.torch_nn_models import TorchPolicyNet, policy_weights

    n_assets = 5
    n_mc     = 200

    net = TorchPolicyNet(n_assets=n_assets, hidden_layers=(16, 16))
    net.eval()

    # Simulate the nn_policy closure from _extract_policy_fn
    total_steps = 252
    goal_val    = 1.10
    cfg = make_config(quick=True)  # minimal config for weight bounds

    def nn_policy(w_norm, tau):
        w_arr    = np.atleast_1d(np.asarray(w_norm, dtype=float))
        scalar   = np.ndim(w_norm) == 0
        step_idx = max(0, int(round(total_steps * (1.0 - tau))))
        weights  = np.asarray(
            policy_weights(net, w_arr * goal_val, goal_val,
                           step_idx=step_idx, total_steps=total_steps),
            dtype=float)
        from comparisons.core.evaluation import apply_leverage_constraint
        out = apply_leverage_constraint(
            np.atleast_2d(weights), cfg.weight_lower_bound,
            cfg.weight_upper_bound, cfg.max_long_leverage, cfg.max_short_leverage)
        return out[0] if scalar else out

    w_batch = np.ones(n_mc) * 1.0  # (n_mc,)
    tau_val = 0.5
    result  = nn_policy(w_batch, tau_val)
    result  = np.asarray(result, dtype=float)

    assert result.shape == (n_mc, n_assets), (
        f"Expected ({n_mc}, {n_assets}), got {result.shape}"
    )


def _run_arch_n5(arch_name):
    """Train a tiny model for arch_name with n=5 and run the full MC pipeline."""
    from comparisons.core.torch_nn_models import train_torch_policy_net

    n_assets = 5
    mkt      = _FakeMarketData(n_assets)
    cfg      = make_config(quick=True)

    train_kwargs = dict(
        mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
        architecture_name=arch_name,
        w0=1.0, goal_mult=1.10,
        n_paths=16, n_iters=5, n_steps=8,
        pretrain_iters=0, patience=9999, seed=1,
    )
    if arch_name == 'nn_historical_replay':
        train_kwargs['historical_returns'] = np.exp(mkt.log_ret) - 1.0

    net, _ = train_torch_policy_net(**train_kwargs)

    fake_result = {
        "method_family": "nn",
        "n_assets": n_assets,
        "target_wealth": 1.10,
        "_model_artifact": {"model": net, "metadata": {"n_steps": 8}},
    }

    pfn = _extract_policy_fn(fake_result, mkt, cfg)
    assert pfn is not None, f"_extract_policy_fn returned None for {arch_name}"

    mc = _gbm_mc_eval(pfn, mkt, 1.0, 1.10, n_mc=200, seed=0)
    assert "mc_goal_prob" in mc, f"{arch_name}: missing mc_goal_prob"
    assert 0.0 <= mc["mc_goal_prob"] <= 1.0, f"{arch_name}: goal_prob out of range"
    return mc


def test_gbm_mc_eval_n5_mlp_small():
    _run_arch_n5("nn_mlp_small")


def test_gbm_mc_eval_n5_mlp_deep():
    _run_arch_n5("nn_mlp_deep")


def test_gbm_mc_eval_n5_policy_net():
    _run_arch_n5("nn_policy_net")


def test_gbm_mc_eval_n5_ste_goalreach():
    _run_arch_n5("nn_ste_goalreach")


def test_gbm_mc_eval_n5_policy_long_only():
    _run_arch_n5("nn_policy_long_only")


def test_gbm_mc_eval_n5_historical_replay():
    _run_arch_n5("nn_historical_replay")


def test_gbm_mc_eval_n5_deep_bsde():
    _run_arch_n5("deep_bsde")


def test_gbm_mc_eval_n5_actor_critic():
    _run_arch_n5("actor_critic")


def test_gbm_mc_eval_n5_lstm():
    _run_arch_n5("lstm")


def test_gbm_mc_eval_n5_transformer():
    _run_arch_n5("transformer")
