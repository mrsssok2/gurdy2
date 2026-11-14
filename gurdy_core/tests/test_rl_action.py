import numpy as np
import pytest

from gurdy_core.rl_action import ActionMapper
from gurdy_core.robot import load_robot

SCALE = {"hip_yaw": 0.3, "hip_pitch": 0.7, "knee": 0.6}


def test_no_filter_maps_and_clips():
    m = ActionMapper(load_robot(), SCALE, alpha=1.0)
    assert m(np.zeros(18)) == pytest.approx(m.q_stance)
    q = m(np.full(18, 50.0))
    assert np.all(q <= m.q_hi) and np.all(q >= m.q_lo) and np.all(q[2::3] >= 0)


def test_ema_filter_and_reset():
    m = ActionMapper(load_robot(), SCALE, alpha=0.5)
    a = np.full(18, 0.4)
    m(a)
    assert m.a_filt == pytest.approx(np.full(18, 0.2))
    m(a)
    assert m.a_filt == pytest.approx(np.full(18, 0.3))
    for _ in range(40):
        m(a)
    assert m.a_filt == pytest.approx(a, abs=1e-9)
    m.reset()
    assert np.all(m.a_filt == 0)


def test_white_noise_attenuated():
    m = ActionMapper(load_robot(), SCALE, alpha=0.5)
    rng = np.random.default_rng(0)
    out = np.array([m(rng.normal(0, 0.1, 18)) - m.q_stance for _ in range(20000)]) / m.scale
    # EMA of white noise: std ratio sqrt(alpha / (2 - alpha)) = 0.577
    assert out.std() == pytest.approx(0.1 * np.sqrt(0.5 / 1.5), rel=0.03)


def test_alpha_validated():
    with pytest.raises(ValueError):
        ActionMapper(load_robot(), SCALE, alpha=0.0)
