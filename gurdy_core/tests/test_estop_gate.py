from gurdy_core.estop_gate import EstopGate


def test_passes_until_estop_then_zero_then_passes_again():
    g = EstopGate()
    assert g.filter(0.1, 0.0, 0.2) == (0.1, 0.0, 0.2)
    g.set_estop(True)
    assert g.filter(0.1, 0.05, 0.0) == (0.0, 0.0, 0.0)
    assert g.filter(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)            # zero commands are not counted as blocked
    assert g.blocked == 1
    assert g.set_estop(False) == 1 and g.blocked == 0
    assert g.filter(0.1, 0.0, 0.0) == (0.1, 0.0, 0.0)


def test_release_without_stop_blocks_nothing():
    g = EstopGate()
    assert g.set_estop(False) == 0
