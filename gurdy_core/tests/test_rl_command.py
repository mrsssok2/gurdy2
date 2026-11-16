import numpy as np

from gurdy_core.rl_command import ANGULAR_DEADBAND, LINEAR_DEADBAND, apply_deadband


def test_values():
    assert LINEAR_DEADBAND == 0.015 and ANGULAR_DEADBAND == 0.05


def test_small_linear_zeroed_by_norm():
    assert np.allclose(apply_deadband((0.01, 0.01, 0.3)), (0, 0, 0.3))      # |v| 0.0141 < 0.015
    assert np.allclose(apply_deadband((0.012, 0.012, 0.3)), (0.012, 0.012, 0.3))   # |v| 0.017


def test_small_angular_zeroed():
    assert np.allclose(apply_deadband((0.1, 0, 0.04)), (0.1, 0, 0))
    assert np.allclose(apply_deadband((0.1, 0, -0.06)), (0.1, 0, -0.06))


def test_does_not_modify_input():
    c = np.array([0.001, 0.0, 0.01])
    apply_deadband(c)
    assert c[0] == 0.001


# ---------------------------------------------------------------- minimum speed (lift, not zero)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gurdy_core.rl_command import (ANGULAR_DEADBAND, LINEAR_DEADBAND, MIN_ANGULAR_SPEED, MIN_LINEAR_SPEED,  # noqa: E402
                                   apply_deadband, lift_to_min_speed)


def test_min_speed_zero_and_epsilon_are_exact_zero():
    assert np.all(lift_to_min_speed([0, 0, 0]) == 0)
    assert np.all(lift_to_min_speed([LINEAR_DEADBAND, 0, 0.0]) == 0)           # at the deadband -> 0
    assert lift_to_min_speed([0, 0, ANGULAR_DEADBAND])[2] == 0.0
    out = lift_to_min_speed([LINEAR_DEADBAND * 1.1, 0, ANGULAR_DEADBAND * 1.1])  # just above -> lifted
    assert out[0] == pytest.approx(MIN_LINEAR_SPEED) and out[2] == pytest.approx(MIN_ANGULAR_SPEED)


def test_min_speed_lifts_linear_keeping_direction():
    out = lift_to_min_speed([0.024, 0.004, 0.0])
    assert np.hypot(out[0], out[1]) == pytest.approx(MIN_LINEAR_SPEED)
    assert out[1] / out[0] == pytest.approx(0.004 / 0.024)             # direction kept
    out = lift_to_min_speed([0.0, -0.02, 0.0])
    assert out[0] == 0.0 and out[1] == pytest.approx(-MIN_LINEAR_SPEED)  # sign kept


def test_min_speed_wz_independent_and_signed():
    out = lift_to_min_speed([0.0, 0.0, -0.08])
    assert tuple(out) == (0.0, 0.0, -MIN_ANGULAR_SPEED)
    out = lift_to_min_speed([0.1, 0.0, 0.06])                           # real translation, small turn -> turn lifted
    assert out[0] == 0.1 and out[2] == pytest.approx(MIN_ANGULAR_SPEED)
    out = lift_to_min_speed([0.02, 0.0, 0.3])                           # small translation, real turn -> only linear lifted
    assert out[0] == pytest.approx(MIN_LINEAR_SPEED) and out[2] == 0.3


def test_min_speed_leaves_large_commands_alone_and_is_idempotent():
    for c in ([0.1, 0.0, 0.0], [0.06, 0.03, -0.4], [0.0, 0.0, 0.15], [0.05, 0.0, 0.0]):
        assert np.allclose(lift_to_min_speed(c), c)
    for c in ([0.024, 0.004, 0.06], [0.0, 0.03, 0.1], [0.001, 0.0, 0.0]):
        once = lift_to_min_speed(c)
        assert np.allclose(lift_to_min_speed(once), once)


def test_lifted_commands_clear_the_policy_deadband_and_dead_zone():
    # whatever the lift outputs, the policy's own deadband must not zero it again
    assert MIN_LINEAR_SPEED > LINEAR_DEADBAND and MIN_ANGULAR_SPEED > ANGULAR_DEADBAND
    for c in ([0.02, 0.0, 0.0], [0.0, 0.0, 0.03], [0.011, 0.011, 0.0]):
        out = lift_to_min_speed(c)
        assert np.allclose(apply_deadband(out), out)
    # MPPI's first step from standstill (ax_max 0.4 * model_dt 0.1 = 0.04) is lifted, not lost
    assert np.hypot(*lift_to_min_speed([0.04, 0.0, 0.0])[:2]) == pytest.approx(MIN_LINEAR_SPEED)
