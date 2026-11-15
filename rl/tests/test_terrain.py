"""Step 2.8: terrain heightfield, curriculum, small commands, deadband, eval isolation."""
import mujoco
import numpy as np
import pytest

from envs import terrain as T
from envs.gurdy_env import GurdyEnv, load_config
from gurdy_core.rl_command import ANGULAR_DEADBAND, LINEAR_DEADBAND
from train import load_cfg

CFG = load_cfg("configs/terrain.yaml")


def make(**kw):
    return GurdyEnv({**CFG["env"], "domain_randomization": {"enabled": False}, **kw}, reward_config=CFG["reward"])


def test_flat_config_has_no_heightfield():
    e = GurdyEnv(load_config("configs/flat.yaml")["env"])
    assert e.model.nhfield == 0 and e.terrain_geom == -1


@pytest.mark.parametrize("tt,val", [("rough", 0.02), ("slopes", 12.0), ("steps", 0.03)])
def test_heightfield_matches_generated_heights(tt, val):
    """mj_ray onto the uploaded heightfield == the env's own height map (layout, offset, scale)."""
    e = make(terrain_fixed={"type": tt, "value": val})
    e.reset(seed=3)
    gid, n = np.zeros(1, np.int32), 0
    for x, y in np.random.default_rng(0).uniform(-3, 3, (300, 2)):
        dist = mujoco.mj_ray(e.model, e.data, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
        if gid[0] == e.terrain_geom:
            n += 1
            assert 3.0 - dist == pytest.approx(e.tgen.height_at(e.heights, x, y), abs=5e-4)
    assert n > 20


def test_terrain_generators_amplitudes():
    g, rng = T.TerrainGenerator({}), np.random.default_rng(0)
    h = g.generate("rough", 0.02, rng)
    assert 0.012 < np.ptp(h) <= 0.02 + 1e-9
    h = g.generate("steps", 0.03, rng)
    assert set(np.round(np.unique(h), 6)) == {0.0, 0.03}
    h = g.generate("slopes", 10.0, rng)
    dy, dx = np.gradient(h, g.cfg["res"])
    assert np.degrees(np.arctan(np.hypot(dx, dy))).mean() == pytest.approx(10.0, abs=0.01)


def test_flat_episodes_use_plane_and_terrain_switches_geoms():
    e = make(terrain_fixed={"type": "flat", "value": 0.0})
    e.reset(seed=0)
    assert e.heights is None and e.ground_geom == e.floor_geom
    assert e.model.geom_contype[e.terrain_geom] == 0 and e.model.geom_contype[e.floor_geom] == 1
    e = make(terrain_fixed={"type": "steps", "value": 0.03})
    e.reset(seed=0)
    assert e.ground_geom == e.terrain_geom and e.model.geom_contype[e.floor_geom] == 0


@pytest.mark.parametrize("tt,val", [("rough", 0.04), ("slopes", 15.0), ("steps", 0.05)])
def test_stands_on_hardest_terrain_with_feet_down(tt, val):
    """Spawn + settle on the hardest level with stance targets: no fall, feet touch the terrain,
    fall height is measured from the ground under the feet."""
    e = make(terrain_fixed={"type": tt, "value": val})
    for seed in range(3):
        e.reset(seed=seed, options={"command": (0, 0, 0)})
        for _ in range(100):
            _, _, term, _, info = e.step(np.zeros(18))
            assert not term
        assert info["foot_contact"].sum() >= 3          # rigid stance on blocks: at least a tripod
        assert 0.08 < e.data.qpos[2] - e._ground_ref() < 0.14


def test_curriculum_rules():
    cfg, rng = T.merged({"enabled": True}), np.random.default_rng(0)
    c = T.TerrainCurriculum(cfg, rng)
    assert all(0 <= v <= cfg["init_max_level"] for v in c.levels.values())
    c.levels["rough"] = 4
    assert c.update("rough", False, 0.1, rng) == 1 and c.levels["rough"] == 5
    assert c.update("rough", True, 0.1, rng) == -1                    # fall -> down
    assert c.update("rough", False, 0.9, rng) == -1                   # poor tracking -> down
    assert c.update("rough", False, 0.5, rng) == 0                    # in between -> stay
    assert c.update("rough", False, None, rng) == 0                   # only zero commands -> stay
    c.levels["rough"] = 0
    assert c.update("rough", True, None, rng) == 0                    # floor
    c.levels["steps"] = cfg["levels"] - 1
    c.update("steps", False, 0.1, rng)
    assert 0 <= c.levels["steps"] < cfg["levels"]                     # top: random level
    assert T.difficulty_value(cfg, "steps", 0) == cfg["easy"]["steps"]
    assert T.difficulty_value(cfg, "steps", cfg["levels"] - 1) == cfg["hard"]["steps"]


def test_curriculum_moves_level_in_env():
    e = make()
    e.reset(seed=0)
    e.curriculum.levels = {k: 5 for k in e.curriculum.levels}
    e.terrain_type, e.terrain_level = "rough", 5
    e.data.qpos[2] = -1.0                                              # force a fall on the next step
    _, _, term, _, info = e.step(np.zeros(18))
    assert term and info["terrain_level_change"] == -1 and e.curriculum.levels["rough"] == 4


def test_terrain_types_sampled_with_config_probabilities():
    e = make()
    seen = {}
    for s in range(200):
        e.reset(seed=s)
        seen[e.terrain_type] = seen.get(e.terrain_type, 0) + 1
    assert set(seen) == set(T.TYPES)
    assert seen["steps"] > seen["flat"]


def test_small_commands_and_deadband():
    e = make(zero_command_prob=0.0, small_command_prob=1.0, axis_aligned_prob=0.0)
    e.reset(seed=0)
    lim = e.cmd_lim
    for _ in range(500):
        c = e._sample_command()
        frac = np.max(np.abs(c) / lim)
        assert frac <= 0.40 + 1e-9
        assert np.hypot(c[0], c[1]) == 0 or np.hypot(c[0], c[1]) >= LINEAR_DEADBAND
        assert c[2] == 0 or abs(c[2]) >= ANGULAR_DEADBAND
    e = make(zero_command_prob=0.0, small_command_prob=0.0, axis_aligned_prob=0.0)
    e.reset(seed=0)
    for _ in range(2000):
        c = e._sample_command()
        assert np.hypot(c[0], c[1]) == 0 or np.hypot(c[0], c[1]) >= LINEAR_DEADBAND
        assert c[2] == 0 or abs(c[2]) >= ANGULAR_DEADBAND


def test_eval_env_ignores_training_only_options():
    """Eval inputs must not depend on the evaluated run's training config (2.7b lesson)."""
    import eval as E
    env = E._env(CFG, 20.0, fixed=False)
    assert not env.terrain_on and env.cfg["zero_command_prob"] == 0.15
    assert env.cfg["small_command_prob"] == 0 and not env.cfg["command_deadband"]
    env = E._env(CFG, 20.0, fixed=False, terrain={"type": "rough", "value": 0.01})
    env.reset(seed=0)
    assert env.terrain_type == "rough" and env.terrain_value == 0.01 and env.curriculum is None


def test_start_gate_has_small_commands():
    import eval as E
    assert len(E.START_CMDS) == 18
    for k, (c, before) in E.START_CMDS.items():
        if k[0] == "s":
            assert not any(before)
        if k[0] == "w":
            assert any(before) and np.argmax(np.abs(before)) != np.argmax(np.abs(c))
