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
