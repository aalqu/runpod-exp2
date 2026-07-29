import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_experiment import build_summary


def _result(**overrides):
    """Minimal valid result dict that build_summary accepts."""
    base = {
        "method_name":   "test_method",
        "method_family": "nn",
        "n_assets":      1,
        "seed":          1,
        "goal_mult":     1.10,
        "target_wealth": 1.10,
        "wealth_path":   np.array([1.0, 1.05, 1.10]),
        "weight_path":   np.array([[0.8], [0.9], [1.0]]),
        "goal_hit":      np.array([True]),
        "drawdown_path": np.array([-0.05, -0.02, 0.0]),
    }
    base.update(overrides)
    return base


def test_drawdown_normal():
    """hist_max_drawdown is the minimum of drawdown_path."""
    result = _result(drawdown_path=np.array([-0.10, -0.05, 0.0]))
    df = build_summary([result])
    assert df["hist_max_drawdown"].iloc[0] == pytest.approx(-0.10)


def test_drawdown_empty_array():
    """Empty drawdown_path does not crash and returns 0.0."""
    result = _result(drawdown_path=np.array([]))
    df = build_summary([result])
    assert df["hist_max_drawdown"].iloc[0] == pytest.approx(0.0)


def test_drawdown_key_missing():
    """Missing drawdown_path key does not crash and returns 0.0."""
    result = _result()
    del result["drawdown_path"]
    df = build_summary([result])
    assert df["hist_max_drawdown"].iloc[0] == pytest.approx(0.0)
