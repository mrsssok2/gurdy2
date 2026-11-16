# Phase 4 — 5-scene render trial (2026-11-16)

Runbook step 1 (`docs/results/phase4_runbook.md`): scenes 0–4 of `datasets/synth_v1` rendered headless in Gazebo Harmonic (sim image), 52 frames kept of 60 views
(8 dropped: views without a labelled object are kept with probability 0.1). About 8 s per scene (12 views).

**Camera assumptions (all verified)**
| assumption | result |
|---|---|
| `/world/synth/set_pose` moves the *static* `cam` model | yes — service returns true in ≤ 0.1 s, and the boxes match ground truth projected from the *planned* pose (below) |
| boundingbox camera publishes on `/cam/bbox` | yes (messages received on that name; `/cam/bbox/set_rate` service exists) |
| RGB, depth and bbox share the same sim stamp | yes — identical stamp lists on all three topics (10 Hz, e.g. 3.9 / 4.0 / 4.1 s) |
| boxes cover only the visible part (`visible_2d`) | looks right in the samples (e.g. the chair partly behind a distractor in `s0004_v05`); not measured numerically |

**Label check** (`vision/check_labels.py`, numbers in `label_check.json`): 71 boxes in 52 frames; 60 boxes contain the independently projected centre of an object of the same class;
the other 11 are objects whose centre is outside the image or hidden (partly visible objects) — not inspected one by one. 60 of 68 ground-truth centres in view have a box (the rest occluded / too small).
Depth at the box centre minus ground-truth centre depth: median −0.08 m (the surface is nearer than the centre), 90 % within 0.25 m.
Only 5 of 7 classes occur in these 5 scenes (no `box`, no `mannequin`; one `barrel`) — those are still unchecked visually.

Samples (3× upscaled, box = label, white cross = ground-truth centre): `s0000_v01_boxes.png` … `s0004_v05_boxes.png`.

**Problem found:** on the first run scenes 0 and 1 produced 0 frames (the Gazebo server stopped answering `set_pose`, 12 × 3 s timeouts) and were still marked as done.
Re-rendering them worked (11/12 each). Cause not found. `render` now retries a scene once and does not mark a 0-frame scene as done.

Full render started afterwards: log in `datasets/synth_v1/render.log` (container `gurdy2-render`, ends with `finalize`).
