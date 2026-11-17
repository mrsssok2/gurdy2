"""Random mock worlds + scenarios. The room geometry is flat_lab's (the planner, doorway and table are the same); the
OBJECT SET differs per seed (class, colour, count, positions, duplicates), so train / val / test do not share worlds."""
from __future__ import annotations

import random

from ..eval.harness import build_world
from ..mock.backend import MockBackend
from ..mock.world import GtObject, World, flat_lab
from .lexicon import COLORS_BY_CLASS, LEGS, SAVED_PLACE_NAMES

RADIUS = {"box": 0.2, "barrel": 0.2, "fire_extinguisher": 0.1, "traffic_cone": 0.15, "toolbox": 0.2, "mannequin": 0.25, "chair": 0.22}
CLASS_P = {"box": 4, "barrel": 2, "fire_extinguisher": 1.5, "traffic_cone": 1.5, "toolbox": 1.5, "mannequin": 1, "chair": 1.5}


def _spot(rng: random.Random, taken: list[tuple[float, float]]) -> tuple[float, float] | None:
    for _ in range(60):
        room_b = rng.random() < 0.5
        x = rng.uniform(2.9, 6.1) if room_b else rng.uniform(-1.6, 2.1)
        y = rng.uniform(-2.6, 2.6)
        if abs(x) < 0.9 and abs(y) < 0.9:            # keep the start free
            continue
        if 0.3 < x < 1.5 and -1.9 < y < -0.9:          # the table
            continue
        if 2.0 < x < 3.0 and -0.3 < y < 1.3:           # doorway
            continue
        if all((x - tx) ** 2 + (y - ty) ** 2 > 0.8 ** 2 for tx, ty in taken):
            return round(x, 2), round(y, 2)
    return None


def random_world(rng: random.Random) -> World:
    w = flat_lab()
    objs: list[GtObject] = []
    taken: list[tuple[float, float]] = []
    classes = list(CLASS_P)
    for gid in range(1, rng.randint(3, 7) + 1):
        cls = rng.choices(classes, [CLASS_P[c] for c in classes])[0]
        spot = _spot(rng, taken)
        if spot is None:
            break
        taken.append(spot)
        objs.append(GtObject(gid, cls, rng.choice(COLORS_BY_CLASS[cls]), spot[0], spot[1], RADIUS[cls]))
    w.objects = objs
    return w


def base_scenario(rng: random.Random, world: World) -> dict:
    """Random robot / world state. A family may override single keys afterwards."""
    spec: dict = {}
    r = rng.random()
    if r < 0.68:
        spec["explored"] = True
    else:
        spec["explored"] = False
        if r < 0.93 and world.objects:                 # partly known world: some objects seen, some not
            k = rng.randint(1, len(world.objects))
            spec["seen"] = sorted(o.gid for o in rng.sample(world.objects, k))
    if rng.random() < 0.35:
        if rng.random() < 0.5:
            spec["pose"] = {"place": rng.choice(["home", "room A", "room B", "doorway"]), "yaw_deg": rng.choice([0, 90, 180, -90])}
        else:
            for _ in range(40):
                x, y = round(rng.uniform(-1.6, 6.1), 2), round(rng.uniform(-2.6, 2.6), 2)
                if world.is_free(x, y):
                    spec["pose"] = {"x": x, "y": y, "yaw_deg": rng.choice([0, 45, 90, 135, 180, -45, -90, -135])}
                    break
    r = rng.random()
    spec["battery"] = round(rng.uniform(55, 100), 0) if r < 0.88 else round(rng.uniform(12, 29), 0) if r < 0.96 else round(rng.uniform(3, 9), 0)
    if rng.random() < 0.03:
        spec["fail_legs"] = [rng.choice(LEGS)]
    if rng.random() < 0.08:
        spec["zones"] = [{"place": rng.choice(["room B", "doorway", "room A"]), "radius_m": rng.choice([0.8, 1.0, 1.5])}]
    return spec


def build(spec: dict, world: World, seed: int) -> MockBackend:
    return build_world(spec, seed, world=world)


def with_saved_places(b: MockBackend, rng: random.Random) -> None:
    """Sometimes the user saved a place earlier ('workbench'): extra named places to go to."""
    if rng.random() < 0.15:
        x, y = b.x, b.y
        b.x, b.y = round(rng.uniform(-1.5, 2.0), 2), round(rng.uniform(-2.0, 2.0), 2)
        if b.world.is_free(b.x, b.y):
            b.save_place(rng.choice(SAVED_PLACE_NAMES))
        b.x, b.y = x, y
