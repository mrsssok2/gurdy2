"""Step 2.3 reward tests: each term in isolation, then behaviour in the simulator
(standing vs tripod walker vs half-speed walker under fixed commands)."""
import json
import os

import numpy as np
import pytest

from envs import reward as R
from envs.gurdy_env import GurdyEnv, load_config
from gurdy_core.gait import TripodGait

CFG = load_config("configs/flat.yaml")
RC = CFG["reward"]
FLOOR_V, FLOOR_W = RC["sigma_v_floor"], RC["sigma_w_floor"]


def trk(v_xy, wz, cmd, kappa=RC["kappa"]):
    return R.tracking(v_xy, wz, cmd, kappa, FLOOR_V, FLOOR_W)


# ====================================================================== tracking term
MOVE_CMDS = [(0.1, 0, 0), (-0.1, 0, 0), (0, 0.08, 0), (0, -0.08, 0), (0.05, 0, 0), (0.15, 0, 0),
             (0, 0, 0.3), (0, 0, -0.3), (0, 0, 0.6), (0, 0, 0.1), (0.08, 0.04, 0.2)]


def test_tracking_perfect_is_one():
    for c in MOVE_CMDS + [(0, 0, 0)]:
        assert trk(c[:2], c[2], c) == pytest.approx(1.0)


@pytest.mark.parametrize("cmd", MOVE_CMDS)
def test_standing_scores_the_same_low_value_for_every_move_command(cmd):
    """Command-relative sigma: standing under ANY move command (incl. pure turning) gets
    exp(-1/kappa^2) (unless a floor is active for a very small command)."""
    k = RC["kappa"]
    expected = np.exp(-1 / k ** 2)
    got = trk((0, 0), 0.0, cmd)
    assert got <= expected + 1e-12
    assert got < 0.25


@pytest.mark.parametrize("cmd", MOVE_CMDS)
def test_half_speed_beats_standing_clearly(cmd):
    c = np.array(cmd, float)
    assert trk(0.5 * c[:2], 0.5 * c[2], c) >= 3.0 * trk((0, 0), 0.0, c)


def test_zero_command_floors():
    assert trk((0, 0), 0.0, (0, 0, 0)) == 1.0                    # no division by zero
    assert trk((0.02, 0), 0.0, (0, 0, 0)) == pytest.approx(np.exp(-1))      # drift = floor
    assert trk((0, 0), 0.05, (0, 0, 0)) == pytest.approx(np.exp(-1))
    assert trk((0.005, 0), 0.01, (0, 0, 0)) > 0.9                # tiny drift: mild


def test_pure_turn_uses_linear_floor():
    # v_cmd = 0: sigma_v = floor; a 1 cm/s sway while turning costs exp(-0.25)
    assert trk((0.01, 0), 0.3, (0, 0, 0.3)) == pytest.approx(np.exp(-0.25))
    assert trk((0, 0), 0.3, (0, 0, 0.3)) == pytest.approx(1.0)


def test_annealing_kappa_sharpens():
    c = (0.1, 0, 0)
    assert trk((0, 0), 0, c, kappa=0.5) == pytest.approx(np.exp(-4))
    assert trk((0, 0), 0, c, kappa=0.5) < trk((0, 0), 0, c, kappa=0.8)


# ====================================================================== other terms
def _terms(**kw):
    base = dict(v_body=np.zeros(3), w_body=np.zeros(3), track_vel=np.zeros(3), gravity_body=np.array([0, 0, -1.0]),
                command=np.array([0.1, 0, 0]), kappa=0.8, cfg=R.DEFAULTS, joint_vel=np.zeros(18),
                joint_vel_prev=np.zeros(18), dt=0.02, torque=np.zeros(18), action=np.zeros(18),
                action_prev=np.zeros(18), foot_contact=np.ones(6, bool), foot_vel_xy=np.zeros((6, 2)),
                air_bonus=0.0, q=np.zeros(18), q_stance=np.zeros(18), fallen=False)
    base.update(kw)
    return R.terms(**base)


def test_penalty_terms():
    t = _terms(v_body=np.array([0.1, 0, 0.2]), w_body=np.array([0.3, 0.4, 0.0]),
               gravity_body=np.array([0.1, 0.2, -0.97]), torque=np.full(18, 2.0),
               joint_vel=np.full(18, 0.5), joint_vel_prev=np.full(18, 0.3),
               action=np.full(18, 0.5), action_prev=np.full(18, 0.4), fallen=True)
    assert t["vz"] == pytest.approx(0.04)
    assert t["ang_xy"] == pytest.approx(0.25)
    assert t["tilt"] == pytest.approx(0.05)
    assert t["energy"] == pytest.approx(18 * 1.0)
    assert t["torque"] == pytest.approx(18 * 4.0)
    assert t["joint_acc"] == pytest.approx(18 * 100.0)
    assert t["action_rate"] == pytest.approx(18 * 0.01)
    assert t["fall"] == 1.0 and t["alive"] == 1.0


def test_foot_slip_only_counts_stance_feet():
    vel = np.zeros((6, 2))
    vel[0] = [0.1, 0.0]
    vel[1] = [0.0, 0.2]
    contact = np.array([True, False, True, True, True, True])
    assert _terms(foot_vel_xy=vel, foot_contact=contact)["foot_slip"] == pytest.approx(0.01)


def test_air_time_and_stand_still_gating():
    assert _terms(air_bonus=0.3)["air_time"] == 0.3
    assert _terms(air_bonus=0.3, command=np.zeros(3))["air_time"] == 0.0
    q = np.full(18, 0.1)
    assert _terms(q=q, command=np.zeros(3))["stand_still"] == pytest.approx(1.8)
    assert _terms(q=q)["stand_still"] == 0.0


def test_combine_clips_and_scales():
    w = {"track": 0.3, "alive": 0.05, "energy": -0.5, "joint_acc": -0.4, "fall": 0.0, "air_time": -0.1}
    r, applied, clipped = R.combine(w, 1.0, True)
    assert r == 0.0 and clipped                                   # -0.65 clipped to 0
    r, applied, clipped = R.combine(w, 0.1, True)
    assert applied["energy"] == pytest.approx(-0.05) and applied["joint_acc"] == pytest.approx(-0.04)
    assert applied["track"] == 0.3 and applied["air_time"] == -0.1   # not penalty-scaled
    assert r == pytest.approx(0.3 + 0.05 - 0.05 - 0.04 - 0.1) and not clipped
    r, _, _ = R.combine({**w, "fall": -10.0}, 1.0, True)
    assert r == -10.0                                             # fall kept outside the clip
    r, _, _ = R.combine(w, 1.0, False)
    assert r == pytest.approx(-0.65)


def test_falling_never_beats_surviving():
    """With only_positive, any surviving step is >= 0, a fall is -10 once: an episode that
    falls at step k returns <= the same episode surviving."""
    surviving = [R.combine({"track": 0.0, "alive": 0.05, "energy": -3.0, "fall": 0.0}, 1.0, True)[0]] * 100
    falling = surviving[:10] + [R.combine({"track": 0.0, "alive": 0.05, "energy": -3.0, "fall": -10.0}, 1.0, True)[0]]
    assert sum(falling) < sum(surviving)


def test_ema_filter():
    f = np.zeros(3)
    for _ in range(100):
        f = R.ema_update(f, (0.1, 0, 0), 0.02, 0.2)
    assert f == pytest.approx([0.1, 0, 0], abs=1e-5)
    assert R.ema_update(np.zeros(3), (0.1, 0, 0), 0.02, 0.0) == pytest.approx([0.1, 0, 0])
    assert R.ema_update(np.zeros(3), (0.1, 0, 0), 0.02, 0.2)[0] == pytest.approx(0.01)


def test_air_timer():
    t = R.FootAirTimer(2, 0.02)
    assert t.update([True, True], 0.2) == 0.0
    for _ in range(14):                                          # foot 0 in the air 0.28 s
        t.update([False, True], 0.2)
    assert t.update([True, True], 0.2) == pytest.approx(0.28 + 0.02 - 0.2)
    assert t.update([True, True], 0.2) == 0.0


# ====================================================================== behaviour in the simulator
SIM_CMDS = {
    "forward": (0.10, 0, 0), "backward": (-0.10, 0, 0), "strafe_left": (0, 0.08, 0),
    "strafe_right": (0, -0.08, 0), "turn_left": (0, 0, 0.3), "turn_right": (0, 0, -0.3),
    "combined": (0.08, 0.04, 0.2), "slow": (0.05, 0, 0), "fast_turn": (0, 0, 0.6),
}
EPISODE_S = 6.0


def _rollout(cmd, policy: str, seed=0):
    """Return of one fixed-command episode. policy: stand | tripod | half."""
    env = GurdyEnv({**CFG["env"], "episode_s": EPISODE_S, "resample_command": False},
                   reward_config=RC)
    env.reset(seed=seed, options={"command": cmd})
    gait = TripodGait(env.robot)
    gait_cmd = np.array(cmd) * (0.5 if policy == "half" else 1.0)
    ret, max_a, terms, done = 0.0, 0.0, {}, False
    while not done:
        if policy == "stand":
            a = np.zeros(18)
        else:
            a = (gait.step(gait_cmd) - env.q_stance) / env.scale
            max_a = max(max_a, float(np.max(np.abs(a))))
        _, r, term, trunc, info = env.step(a)
        ret += r
        for k, v in info["reward_terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        done = term or trunc
    return ret, max_a, term, terms


@pytest.fixture(scope="module")
def sim_results():
    out = {}
    for name, cmd in {**SIM_CMDS, "zero": (0, 0, 0)}.items():
        out[name] = {p: _rollout(cmd, p) for p in ("stand", "tripod", "half")}
    # record the tripod reference returns (step 2.3 deliverable)
    path = os.environ.get("GURDY_REWARD_REF", "../docs/results/phase2_reward_reference.json")
    rec = {n: {p: {"return": r[0], "max_abs_action": r[1], "fell": bool(r[2]),
                   "terms": {k: round(v, 4) for k, v in r[3].items()}}
               for p, r in res.items()} for n, res in out.items()}
    rec["_meta"] = {"episode_s": EPISODE_S, "kappa": RC["kappa"],
                    "sigma_floors": [FLOOR_V, FLOOR_W], "weights": RC["weights"]}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return out


def test_tripod_is_representable_in_action_space(sim_results):
    for name, res in sim_results.items():
        assert res["tripod"][1] <= 1.0, (name, res["tripod"][1])


def test_nobody_falls(sim_results):
    for name, res in sim_results.items():
        for p, r in res.items():
            assert not r[2], (name, p)


@pytest.mark.parametrize("name", list(SIM_CMDS))
def test_tripod_beats_standing_by_2_5x(sim_results, name):
    stand, tripod = sim_results[name]["stand"][0], sim_results[name]["tripod"][0]
    assert tripod >= 2.5 * stand, (name, tripod, stand)


@pytest.mark.parametrize("name", list(SIM_CMDS))
def test_half_speed_walker_clearly_beats_standing(sim_results, name):
    stand, half = sim_results[name]["stand"][0], sim_results[name]["half"][0]
    assert half >= 1.5 * stand, (name, half, stand)


def test_standing_wins_at_zero_command(sim_results):
    z = sim_results["zero"]
    # tripod/half with a zero command also stand (gait stops) -> compare against walking anyway
    walk_under_zero = _rollout((0, 0, 0), "stand")[0]
    assert z["stand"][0] >= walk_under_zero - 1e-9
    moving_when_told_to_stand = _walk_under(cmd=(0, 0, 0), gait_cmd=(0.1, 0, 0))
    assert z["stand"][0] > 2.0 * moving_when_told_to_stand, (z["stand"][0], moving_when_told_to_stand)


def _walk_under(cmd, gait_cmd):
    env = GurdyEnv({**CFG["env"], "episode_s": EPISODE_S, "resample_command": False}, reward_config=RC)
    env.reset(seed=0, options={"command": cmd})
    gait = TripodGait(env.robot)
    ret, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step((gait.step(gait_cmd) - env.q_stance) / env.scale)
        ret += r
        done = term or trunc
    return ret
