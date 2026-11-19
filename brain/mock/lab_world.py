"""The mock world that matches the REAL Gazebo flat_lab, generated from its sources (not a copy):

- walls, doorway, objects (positions, sizes, colours): `scripts/gen_world_flat_lab.py` (the SDF generator: walls(), OBJECTS, constants)
- rooms and named places: `maps/flat_lab.places.yaml` (the same file RosBackend uses)

`brain.mock.world.flat_lab()` is NOT this world: it is the richer world the evaluation sets (testset_v1 ...) were written for (barrel,
fire extinguisher, mannequin) and stays untouched. Use `gazebo_flat_lab()` for the web demo / dashboard, so what the page shows is what
the robot would find in Gazebo.

Object classes: the Gazebo lab has plain coloured boxes and one cylinder. `*_box` -> detector class `box` with its colour, the green
cylinder -> `barrel` (the nearest detector class; the real detector often calls it `box`), the table is an obstacle only (not a class).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from .world import GtObject, Rect, World

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "gen_world_flat_lab.py"
PLACES = ROOT / "maps" / "flat_lab.places.yaml"
_CLASS_OF = {"cylinder": "barrel"}           # geometry -> detector class when the name has no class word
_NOT_OBJECTS = {"table"}                     # obstacles that are not detector classes


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_world_flat_lab", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _centre_rect(cx: float, cy: float, sx: float, sy: float) -> Rect:
    return Rect(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)


def _colour_of(name: str) -> str:
    return name.split("_")[0]


def gazebo_flat_lab() -> World:
    g = _load_generator()
    walls = [_centre_rect(c[0], c[1], s[0], s[1]) for _, c, s in g.walls()]
    obstacles: list[Rect] = []
    objects: list[GtObject] = []
    gid = 0
    for name, kind, (x, y), size, _rgb in g.OBJECTS:
        sx, sy = (size[0], size[1]) if kind == "box" else (2 * size[0], 2 * size[0])
        if name in _NOT_OBJECTS:
            obstacles.append(_centre_rect(x, y, sx, sy))
            continue
        gid += 1
        cls = "box" if name.endswith("_box") else _CLASS_OF.get(kind, kind)
        objects.append(GtObject(gid, cls, _colour_of(name), x, y, max(sx, sy) / 2))
    spec = yaml.safe_load(PLACES.read_text())
    rooms: dict[str, Rect] = {}
    places: dict[str, tuple[float, float]] = {}
    for pname, v in spec["places"].items():
        if "rect" in v:
            r = Rect(*[float(c) for c in v["rect"]])
            rooms[pname] = r
            places[pname] = r.centre
        else:
            places[pname] = (float(v["x"]), float(v["y"]))
    pad = 0.2
    bounds = Rect(g.X_MIN - pad, g.Y_MIN - pad, g.X_MAX + pad, g.Y_MAX + pad)
    return World(bounds=bounds, rooms=rooms, walls=walls, obstacles=obstacles, objects=objects, places=places,
                 map_id=spec.get("map_id", "flat_lab"))
