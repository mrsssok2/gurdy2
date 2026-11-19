#!/usr/bin/env python3
"""Public demo -> a STATIC Hugging Face Space (free for everyone; Docker Spaces on free CPU need PRO, so the demo runs in the browser instead).

  python3 scripts/demo_space.py stage [--out build/demo_space]     build the static demo (scripts/build_static_demo.py) + the Space README (sdk: static)
  python3 scripts/demo_space.py deploy [--space USER/gurdy-demo]   stage, create the Space if needed, upload, wait until it runs, fetch the page

`deploy` needs `pip install huggingface_hub` and the access token in the ENVIRONMENT variable HF_TOKEN (never written to a file, never printed).
The same build is what the GitHub Pages workflow publishes (.github/workflows/pages.yml).
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
sys.path.insert(0, str(ROOT / "scripts"))
MAX_FILE_MB = 5


def stage(out: Path) -> Path:
    import build_static_demo
    build_static_demo.build(out)
    shutil.copy2(ROOT / "web/static_demo/README_SPACE.md", out / "README.md")
    (out / ".nojekyll").unlink(missing_ok=True)                 # GitHub Pages marker, meaningless on a Space
    big = [p for p in out.rglob("*") if p.is_file() and p.stat().st_size > MAX_FILE_MB * 1e6]
    if big:
        sys.exit(f"refusing: files over {MAX_FILE_MB} MB in the Space folder: {big}")
    bad = [p for p in out.rglob("*") if p.is_file() and (p.suffix in (".env", ".pem", ".key", ".db") or p.name in (".env", "memory.db"))]
    if bad:
        sys.exit(f"refusing: {bad} look like secrets / local data")
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1024:.0f} KiB in {out}")
    return out


def deploy(out: Path, space: str | None, wait_s: int) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a token (docs/phase7_demo_deploy.md) and run:  read -s HF_TOKEN; export HF_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub   (in a virtualenv, see docs/phase7_demo_deploy.md)")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = space or f"{user}/gurdy-demo"
    stage(out)
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="space", commit_message="Update the Gurdy 2.0 demo",
                      delete_patterns=["*.js", "*.css", "*.html", "*.json", "*.zip", "fonts/*", "icons/*", "Dockerfile", "requirements.txt", "web/**", "brain/**"])
    url = f"https://{repo.replace('/', '-').replace('_', '-').lower()}.static.hf.space/"
    print(f"uploaded to https://huggingface.co/spaces/{repo}; waiting for it to run ...")
    t0, stage_name = time.time(), ""
    while time.time() - t0 < wait_s:
        stage_name = str(api.get_space_runtime(repo).stage)
        print(f"  {int(time.time() - t0):>4d} s  {stage_name}", flush=True)
        if stage_name.endswith(("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR")):
            break
        time.sleep(8)
    if not stage_name.endswith("RUNNING"):
        sys.exit(f"the Space is not running ({stage_name}). Open https://huggingface.co/spaces/{repo}")
    for _ in range(10):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                body = r.read().decode("utf-8", "replace")
                if "Gurdy" in body and "static_backend.js" in body:
                    print(f"LIVE: {url}   (also https://huggingface.co/spaces/{repo})")
                    return
        except OSError as e:
            print("  waiting for the URL ...", e)
        time.sleep(6)
    sys.exit(f"the Space says RUNNING but {url} does not serve the demo yet; try again in a minute")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "deploy"])
    ap.add_argument("--out", default=str(ROOT / "build" / "demo_space"))
    ap.add_argument("--space", help="USER/NAME (default: <your user>/gurdy-demo)")
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for the Space (deploy)")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(Path(a.out))
    else:
        deploy(Path(a.out), a.space, a.wait)


if __name__ == "__main__":
    main()
