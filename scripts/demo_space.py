#!/usr/bin/env python3
"""Public demo -> Hugging Face Docker Space.

  python3 scripts/demo_space.py stage [--out build/demo_space]      copy exactly the files the demo needs (+ Dockerfile, Space README) and print the size
  python3 scripts/demo_space.py deploy [--space USER/gurdy-demo]    stage, create the Space if needed, upload, wait until it runs, check /api/health

`deploy` needs `pip install huggingface_hub` and the access token in the ENVIRONMENT variable HF_TOKEN (never written to a file, never printed).
No git history is pushed: the Space gets a clean snapshot of the staged folder each time.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Files / folders copied into the Space (relative to the repo root). Checked by building the image and running the demo from it.
FILES = [
    "brain/__init__.py", "brain/eval/__init__.py", "brain/eval/keyword_parser.py", "brain/data_gen/__init__.py", "brain/data_gen/lexicon.py",
    "brain/voice/__init__.py", "brain/voice/audio.py", "brain/voice/host.py", "brain/voice/pipeline.py", "brain/voice/stt.py", "brain/voice/tts.py", "brain/voice/vad.py",
    "gurdy_core/gurdy_core/__init__.py", "gurdy_core/gurdy_core/vision/__init__.py", "gurdy_core/gurdy_core/vision/classes.py",
    "scripts/gen_world_flat_lab.py", "maps/flat_lab.places.yaml",
    "web/__init__.py",
]
DIRS = {"brain/agent": ("*.py", "*.yaml"), "brain/mock": ("*.py",), "brain/tools": ("*.py",),
        "web/api": ("*.py",), "web/app": ("**/*",)}
SKIP = {"ros_extras.py", "ros_telemetry.py", "__pycache__"}      # the real-robot telemetry needs ROS: not part of the demo
MAX_FILE_MB = 5


def stage(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied: list[Path] = []

    def put(src: Path) -> None:
        rel = src.relative_to(ROOT)
        if any(p in SKIP for p in rel.parts):
            return
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    for f in FILES:
        put(ROOT / f)
    for d, pats in DIRS.items():
        for pat in pats:
            for src in sorted((ROOT / d).glob(pat)):
                if src.is_file():
                    put(src)
    shutil.copy2(ROOT / "docker/demo/Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "docker/demo/requirements.txt", out / "requirements.txt")
    shutil.copy2(ROOT / "docker/demo/README_SPACE.md", out / "README.md")
    big = [p for p in out.rglob("*") if p.is_file() and p.stat().st_size > MAX_FILE_MB * 1e6]
    if big:
        sys.exit(f"refusing: files over {MAX_FILE_MB} MB in the Space folder: {big}")
    for p in out.rglob("*"):                          # nothing secret may ever be staged
        if p.is_file() and p.suffix in (".env", ".pem", ".key") or p.name in (".env", "memory.db"):
            sys.exit(f"refusing: {p} looks like a secret / local data")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    kb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024
    print(f"staged {n} files, {kb:.0f} KiB in {out}")
    return copied


def deploy(out: Path, space: str | None, wait_s: int) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a token (docs/phase7_demo_deploy.md, step 2) and run:  read -s HF_TOKEN; export HF_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub   (in a virtualenv, see docs/phase7_demo_deploy.md)")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = space or f"{user}/gurdy-demo"
    stage(out)
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="space", commit_message="Update the Gurdy 2.0 demo",
                      delete_patterns=["web/**", "brain/**", "gurdy_core/**", "scripts/**", "maps/**", "Dockerfile", "requirements.txt"])
    url = f"https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print(f"uploaded to https://huggingface.co/spaces/{repo}; building (the first build takes a few minutes) ...")
    t0, stage_name = time.time(), ""
    while time.time() - t0 < wait_s:
        stage_name = str(api.get_space_runtime(repo).stage)
        print(f"  {int(time.time() - t0):>4d} s  {stage_name}", flush=True)
        if stage_name.endswith(("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR")):
            break
        time.sleep(15)
    if not stage_name.endswith("RUNNING"):
        sys.exit(f"the Space is not running ({stage_name}). Open https://huggingface.co/spaces/{repo} -> Logs.")
    for _ in range(10):
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=10) as r:
                print("health:", r.read().decode())
                print(f"LIVE: {url}")
                return
        except OSError as e:
            print("  waiting for the URL ...", e)
            time.sleep(6)
    sys.exit(f"the Space says RUNNING but {url}/api/health does not answer yet; try again in a minute")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "deploy"])
    ap.add_argument("--out", default=str(ROOT / "build" / "demo_space"))
    ap.add_argument("--space", help="USER/NAME (default: <your user>/gurdy-demo)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the build (deploy)")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(Path(a.out))
    else:
        deploy(Path(a.out), a.space, a.wait)


if __name__ == "__main__":
    main()
