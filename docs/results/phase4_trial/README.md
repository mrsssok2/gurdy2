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

**Problem found: scenes with 0 frames.** First run: scenes 0 and 1; first full render: scene 6 once, then scenes 16–19 in a row (render stopped by hand).
What was observed while it was failing: every scene already had its own Gazebo server (no left-over servers, the current server alive at 59 % CPU), RAM fine (container 890 MiB, GPU 410 MiB),
`set_pose` timing out (3 s × 12 views per attempt), and the single long-lived Python render process at 747 MiB RSS with one gz-transport node (3 subscriptions) left over from every earlier scene.
The same scenes 16–19 rendered without any failure as soon as each scene ran in a fresh process. So the fault follows the long-lived process, not the scene or the server —
**the exact mechanism was not established** (the old servers' logs were discarded, and scenes 0/1 failed with only one or two nodes alive).

Fix (`vision/generate_dataset.py render`): every scene and every retry runs in its own worker process (`render-scene`) with a fresh Gazebo server and its own `GZ_PARTITION`; the whole process group is
killed afterwards; hard limit 150 s per attempt, 3 s per service call, 15 s per frame; 3 attempts; frames of a broken attempt are deleted; a scene counts only if no view was lost; finished scenes are
kept (`rendered.txt`); the run aborts with exit code 2 if more than 5 % of the scenes fail or 3 fail in a row. Server and worker logs per scene: `datasets/synth_v1/logs/`.
A periodic restart every N scenes is not needed because the server is restarted for every scene (about 1 s extra per scene: 9 s instead of 8 s).

Full render restarted 2026-11-16 16:48 UTC in container `gurdy2-render` (log `datasets/synth_v1/render.log`, ends with `finalize`); first 6 scenes OK, 0 retries.
