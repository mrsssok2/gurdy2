"""Dataset generator pieces that need no Gazebo: objects, scenes, SDF validity, labels, splits, augmentation."""
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gurdy_core"))

from gurdy_core.vision.classes import CLASSES  # noqa: E402
from vision import objects as O  # noqa: E402
from vision import scene as S  # noqa: E402
from vision.augment import augment_rgb, motion_blur  # noqa: E402
from vision.textures import make_texture, write_textures  # noqa: E402


@pytest.mark.parametrize("cls", CLASSES)
def test_every_class_builds_with_sane_size(cls):
    rng = np.random.default_rng(1)
    var = O.random_variant(cls, rng)
    w, d, h = O.footprint(cls, 1.0, var)
    assert 0.1 < max(w, d) < 1.0 and 0.15 < h < 2.0
    assert all(p["pos"][2] >= -1e-9 for p in O.parts(cls, 1.0, var))
    assert cls in O.LICENSES and "MIT" in O.LICENSES[cls]


def test_scale_scales_footprint():
    a, b = O.footprint("chair", 1.0), O.footprint("chair", 1.5)
    assert b[2] == pytest.approx(1.5 * a[2])


def test_model_sdf_is_valid_xml_with_label(tmp_path):
    sdf = O.model_sdf("c", O.parts("traffic_cone"), 1, 2, 0.5, label=4, mesh_dir=str(tmp_path))
    root = ET.fromstring(sdf)
    assert root.find("plugin").get("filename") == "gz-sim-label-system" and root.find("plugin/label").text == "4"
    assert (tmp_path / "c_p0.obj").exists() and (tmp_path / "c_p0.obj").read_text().startswith("# generated")
    assert ET.fromstring(O.model_sdf("d", O.parts("box"), 0, 0, 0)).find("plugin") is None     # distractor: no label


def test_splits_are_70_15_15_and_scene_disjoint():
    names = [S.split_for_scene(i) for i in range(200)]
    assert names.count("train") == 140 and names.count("val") == 30 and names.count("test") == 30


def test_scene_is_deterministic_and_json_serialisable():
    a, b = S.sample_scene(5), S.sample_scene(5)
    assert json.dumps(a) == json.dumps(b) and json.dumps(a) != json.dumps(S.sample_scene(6))


def test_scene_randomisation_covers_ranges():
    scs = [S.sample_scene(i, 8) for i in range(60)]
    assert {o["cls"] for s in scs for o in s["objects"]} == set(CLASSES)
    heights = [v["pos"][2] for s in scs for v in s["views"]]
    assert min(heights) >= 0.10 and max(heights) <= 0.30 and max(heights) - min(heights) > 0.12
    assert len({s["floor"]["texture"] for s in scs}) > 5 and any(s["lamps"] for s in scs) and any(s["shadows"] for s in scs)
    assert all(1 <= len(s["objects"]) <= 6 for s in scs) and any(s["distractors"] for s in scs)
    assert all(len(s["views"]) >= 6 for s in scs)


def test_objects_and_views_stay_inside_room_and_apart():
    for i in range(30):
        s = S.sample_scene(i, 10)
        W, D = s["room"]["w"], s["room"]["d"]
        items = [(o["x"], o["y"]) for o in s["objects"]] + [(d["x"], d["y"]) for d in s["distractors"]]
        assert all(abs(x) < W / 2 and abs(y) < D / 2 for x, y in items)
        assert all(abs(v["pos"][0]) < W / 2 - 0.29 and abs(v["pos"][1]) < D / 2 - 0.29 for v in s["views"])
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                assert math.dist(items[a], items[b]) > 0.2
        assert all(min(math.dist(v["pos"][:2], p) for p in items) >= 0.39 for v in s["views"]) if items else True


def test_optical_projection_conventions():
    # camera at origin looking along +x (yaw 0, no pitch): a point 2 m ahead is on the optical axis
    p = S.optical_from_world([2, 0, 0], [0, 0, 0], [0, 0, 0])
    assert p == pytest.approx([0, 0, 2])
    assert S.project(p, 1.047, 320, 240) == pytest.approx((160, 120))
    left = S.optical_from_world([2, 0.5, 0], [0, 0, 0], [0, 0, 0])       # +y world = left -> negative optical x
    assert left[0] < 0 and S.project(left, 1.047, 320, 240)[0] < 160
    up = S.optical_from_world([2, 0, 0.5], [0, 0, 0], [0, 0, 0])
    assert up[1] < 0                                                        # above axis -> smaller v
    yawed = S.optical_from_world([0, 2, 0], [0, 0, 0], [0, 0, math.pi / 2])
    assert yawed == pytest.approx([0, 0, 2], abs=1e-9)
    down = S.optical_from_world([2, 0, 0], [0, 0, 0.2], [0, 0.17, 0])       # pitching down: a point at camera height is ABOVE the axis
    assert down[1] < 0
    assert S.project([0, 0, -1], 1.047, 320, 240) is None


def test_gt_for_view_matches_pixel_of_object():
    sc = S.sample_scene(3, 6)
    for v in sc["views"]:
        for g in S.gt_for_view(sc, v):
            assert 0 <= g["uv"][0] < 320 and 0 <= g["uv"][1] < 240 and g["p_cam"][2] > 0


def test_world_sdf_valid_and_has_labelled_objects(tmp_path):
    sc = S.sample_scene(11, 5)
    write_textures(str(tmp_path / "tex"), S.N_TEXTURES)
    xml = S.world_sdf(sc, str(tmp_path), str(tmp_path / "tex"))
    root = ET.fromstring(xml)
    world = root.find("world")
    assert world.get("name") == S.WORLD_NAME
    labelled = [m for m in world.findall("model") if m.find("plugin") is not None]
    assert len(labelled) == len(sc["objects"])
    assert {int(m.find("plugin/label").text) for m in labelled} <= set(range(1, len(CLASSES) + 1))
    cam = [m for m in world.findall("model") if m.get("name") == "cam"][0]
    types = {s.get("type") for s in cam.iter("sensor")}
    assert types == {"rgbd_camera", "boundingbox_camera"} and cam.find(".//box_type").text == "visible_2d"
    assert cam.find(".//rgbd_camera") is None and cam.find(".//image/width").text == "320"
    assert "gz-sim-sensors-system" in xml and "gz-sim-user-commands-system" in xml


def test_camera_matches_robot_yaml_intrinsics():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "ros2_ws/src/gurdy_description/config/robot.yaml"))["sensors"]["depth_camera"]
    for k in ("width", "height", "hfov", "pitch_down"):
        assert S.DEFAULT_CAMERA[k] == cfg[k]
    assert S.DEFAULT_CAMERA["range"] == cfg["range"]


def test_yolo_lines_normalise_clip_and_drop_tiny():
    lines = S.yolo_lines([(2, 100, 60, 140, 180), (0, 300, 200, 330, 250), (1, 10, 10, 14, 40), (3, 50, 50, 56, 56)], 320, 240)
    assert lines[0] == "2 0.375000 0.500000 0.125000 0.500000"
    cid, cx, cy, bw, bh = lines[1].split()                   # clipped at the right/bottom edge
    assert cid == "0" and float(cx) * 320 + float(bw) * 320 / 2 == pytest.approx(320) and float(cy) * 240 + float(bh) * 240 / 2 == pytest.approx(240)
    assert len(lines) == 2                                      # 4x30 px and 6x6 px boxes dropped


def test_textures_deterministic_and_distinct(tmp_path):
    assert np.array_equal(make_texture("wood", 3), make_texture("wood", 3))
    assert not np.array_equal(make_texture("wood", 3), make_texture("wood", 4))
    assert len(write_textures(str(tmp_path), 6)) == 6 and len(list(tmp_path.glob("*.png"))) == 6


def test_augment_keeps_shape_dtype_and_changes_pixels():
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), np.uint8)
    out = augment_rgb(img, np.random.default_rng(1), {"blur_prob": 1.0})
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)
    flat = np.full((20, 20, 3), 128, np.uint8)
    assert np.array_equal(motion_blur(flat, 5, 0.3), flat)
    stripe = np.zeros((20, 20, 3), np.uint8)
    stripe[:, 10] = 255
    assert motion_blur(stripe, 5, 0.0)[:, 10].mean() < 255


def test_plan_command_writes_worlds_and_labels_never_cross_splits(tmp_path):
    out = tmp_path / "ds"
    r = subprocess.run([sys.executable, str(ROOT / "vision/generate_dataset.py"), "plan", "--out", str(out), "--scenes", "20", "--views", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    scenes = [json.loads(p.read_text()) for p in sorted((out / "worlds").glob("scene_*.json"))]
    assert len(scenes) == 20 and len(list((out / "worlds").glob("*.sdf"))) == 20
    assert [s["split"] for s in scenes].count("test") == 3 and len({s["seed"] for s in scenes}) == 20
    ET.parse(out / "worlds" / "scene_0000.sdf")


def test_test_world_is_current_and_covers_all_classes():
    r = subprocess.run([sys.executable, str(ROOT / "vision/make_test_world.py"), "--check"], capture_output=True)
    assert r.returncode == 0, "run python3 vision/make_test_world.py"
    gt = json.loads((ROOT / "docs/results/data/phase4/semantic_lab_gt.json").read_text())["objects"]
    assert {o["class"] for o in gt} == set(CLASSES)
    world = ET.parse(ROOT / "ros2_ws/src/gurdy_gazebo/worlds/semantic_lab.sdf").getroot().find("world")
    names = {m.get("name") for m in world.findall("model")}
    assert {o["name"] for o in gt} <= names and world.get("name") == "semantic_lab"


def test_light_colours_in_range_and_gz_accepts_worlds(tmp_path):
    import re
    import shutil
    for seed in range(40):
        xml = S.world_sdf(S.sample_scene(seed, 3), str(tmp_path))
        for m in re.finditer(r"<(?:diffuse|specular|ambient)>([\d. -]+)</", xml):
            assert all(0.0 <= float(v) <= 1.0 for v in m.group(1).split()), m.group(0)
    if shutil.which("gz"):                                           # only inside the ROS image
        f = tmp_path / "w.sdf"
        f.write_text(S.world_sdf(S.sample_scene(1, 3), str(tmp_path)))
        r = subprocess.run(["gz", "sdf", "-k", str(f)], capture_output=True, text=True)
        assert "Valid" in r.stdout, r.stdout + r.stderr


def test_compare_semantic_map():
    from vision.compare_semantic_map import compare
    gt = [{"name": "a", "class": "box", "x": 0, "y": 0, "color": "red"}, {"name": "b", "class": "chair", "x": 3, "y": 0, "color": ""}]
    found = [{"class": "box", "x": 0.1, "y": 0, "color": "red"}, {"class": "box", "x": 0.1, "y": 0.1}, {"class": "chair", "x": 5, "y": 0}]
    r = compare(found, gt)
    assert r["recall"] == 0.5 and r["false_or_duplicate"] == 2 and r["rows"][0]["color_ok"] is True and r["median_err_m"] == pytest.approx(0.1)


def test_rpy_to_quat_matches_pose_convention():
    from vision.generate_dataset import rpy_to_quat
    assert rpy_to_quat(0, 0, 0) == pytest.approx((0, 0, 0, 1))
    assert rpy_to_quat(0, 0, math.pi) == pytest.approx((0, 0, 1, 0), abs=1e-9)
    q = rpy_to_quat(0.1, 0.2, 0.3)
    assert sum(v * v for v in q) == pytest.approx(1.0)
