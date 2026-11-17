# Phase 6 design — Robot Incident Graph

Status (2026-11-17): **design only, nothing built, nothing measured.** Every RAM / time / count number below is an
**estimate** until milestone 6.1 measures it. ROADMAP Phase 6 holds the checklist and acceptance criteria (AC); this file is the how.
Scope: English only. One heavy workload at a time: the ops stack never runs together with Gazebo, RL training or SFT (hard rule 2);
it runs on **recorded** telemetry, so none of those need to be on.

## 1. What counts as an incident

An **event** is one thing that happened (an alert, a log line, a metric crossing a threshold). An **incident** is a *group of related events
with one underlying cause*, with a start, an end, a severity and a status. The graph part: incident → events (members), incident → suspected
cause → evidence (config diff, metric window, log lines), incident → similar past incidents.

Incident sources, in order of how much already exists:

| Source | Exists today | Event types |
|---|---|---|
| `alert_monitor_node` (`gurdy_core/ops_monitors.py`) | yes (5.7 code) | `leg_failure`, `battery_low/critical`, `fall`, `stuck`, `path_blocked`, `object_*` |
| `leg_health_node` | yes | per-leg tracking-error state `ok / degraded / failed` |
| Brain event stream `/gurdy/events` + `logs/events/*.jsonl` | designed in Phase 5 (`phase5_design.md` 4.2) | `task_failed`, `estop`, `refusal`, lifecycle |
| Nav2 / controller logs | ROS logs | recoveries, plan failures, controller timeouts |
| Metric windows (new, derived) | no | joint-velocity high-frequency energy (chatter, scripts/ chatter metric), tracking error per leg, commanded vs measured speed, effort saturation fraction, tilt, CoT |

Rules (all deterministic, no LLM):
- **Open** an incident when a warning/critical event arrives or a metric rule fires for longer than its hold time; **merge** later events into
  it when they are within a time gap (default 30 s sim time) and the clustering (section 4) says they belong together; **close** after 60 s with no new member.
- **Not** an incident: info events (`object_new`), a single transient spike below hold time, an operator e-stop (kept as context, it explains
  other events but is not a failure), known-benign events during an active no-go refusal.
- Severity = max of member severities; `fall` and `leg_failure` are critical.

## 2. Data flow

```
 ROS 2 (sim or rosbag replay)                      ops stack (compose profile `ops`)
 /gurdy/events ─┐                                  ┌─────────────────────────────────────────────┐
 /gurdy/alerts ─┤                                  │ Redpanda topics: raw.events, raw.metrics     │
 /joint_states ─┼─ gurdy_exporter ─ JSONL ─ file ─►│   │                                          │
 /odom, /imu   ─┤   (ROS node, 1–10 Hz metrics,    │   ▼                                          │
 /rosout (logs)─┘    window features)               │ normalizer (schema, fingerprint, features)   │
                                                    │   ├─► PostgreSQL: events, incidents, runs    │
 recorded datasets (JSONL / rosbag) ──replay────────►│   ├─► Qdrant: event/incident embeddings      │
   (no simulator needed)                            │   ▼                                          │
                                                    │ Celery workers (Redis): cluster, link, investigate
                                                    │   ▼                                          │
                                                    │ FastAPI: /incidents, /incidents/{id}, /search │
                                                    └───────────────▲─────────────────────────────┘
                                                                    │ HTTP (brain tool `incidents`, Phase 7 web page)
```

- **Exporter** (`ops/exporter`, ROS node + offline `replay.py`): writes the *same* JSONL the Phase 5 event bus already writes, plus a metric
  stream (windowed features every 1 s). From a rosbag, `replay.py` produces identical records, so every experiment is reproducible without Gazebo.
- **Event schema** (extends `gurdy_interfaces/Event`, no second format): `run_id, stamp, type, severity, source, dedup_key, text, data{}` and for metrics
  `run_id, stamp, window_s, features{name: value}`. `run_id` ties records to a recorded run, its robot.yaml hash and controller (needed for config diffs).
- **Fingerprint**: normalized text (numbers/ids replaced by placeholders) + `type` + `source` → stable hash; identical fingerprints count up instead of
  creating rows (same idea as log-template mining).
- **Replay speed**: any recorded run can be pushed at 1×–50× through the exporter path; the AC "runs on recorded telemetry" is tested exactly this way.

## 3. Storage

| Store | Holds | Why this one |
|---|---|---|
| Redpanda (Kafka API) | raw event / metric stream, 24 h retention | roadmap choice; decouples exporter from workers, replayable |
| PostgreSQL | `runs`, `events`, `incidents`, `incident_events`, `causes`, `evidence`, `investigations` | relational joins for "what happened today"; incident graph = foreign keys |
| Qdrant | one vector per event fingerprint and per incident summary (payload: run_id, type, severity, time) | nearest-neighbour "have we seen this before" + clustering input |
| Redis | Celery broker / results, dedup cooldown keys | minimal |

Graph model (tables, not a graph DB): `incident(id, run_id, start, end, severity, status, title)`, `incident_event(incident_id, event_id, role)`,
`cause(id, incident_id, label, confidence, source{rule|agent|label})`, `evidence(cause_id, kind{metric|log|config_diff|event}, ref, summary)`,
`similar(incident_id, other_id, score)`. A graph database is not needed at this size (hundreds of incidents).

**Fallback if the stack misses the RAM budget (decided now, so it is not a surprise later):** run the same code against SQLite + a local
numpy/FAISS-style index and an in-process queue (`ops --lite`). The full stack remains the roadmap deliverable; lite is the CI / low-RAM path.

## 4. Clustering and root-cause linking

**Event features** (all computable offline): text embedding of the normalized message (small sentence-embedding model on CPU,
22M-class, ~384 dims — proposed, to be chosen in 6.3), event type / severity one-hots, time since run start, and the metric-window features
at the event time (per-leg tracking error, high-frequency joint-velocity energy, effort saturation fraction, commanded-vs-measured speed ratio, tilt).
Metrics matter more than text: different faults emit similar text ("stuck") but separate clearly in metric space.

**Clustering**: HDBSCAN (min_cluster_size 3–5, tuned on the labelled fault runs), run per time window and over the last N hours; noise points
stay unclustered events. Incident = cluster ∪ time-gap merge rule. Re-clustering only changes membership of *open* incidents.

**Root-cause linking** (in this order, each step adds evidence, never overwrites):
1. **Rule hypotheses (no LLM)** from metric signatures, e.g. one leg's tracking error ≫ others + `leg_failure` → *disabled / weak leg*; all legs'
   tracking error up + effort saturation → *weak motors / low torque limit*; high-frequency joint-velocity energy without tracking-error rise →
   *controller gain / chatter*; commanded speed high, odometry low, **no** leg tracking error → *low friction / slipping*; battery events first → *power*.
2. **Config diff**: compare the run's `robot.yaml` / gains / controller model hash against the nominal run (and the previous run); changed keys become evidence
   (`kd 0.05 → 0.5` explains chatter). Changes are listed, never guessed.
3. **Similar-incident lookup** in Qdrant: top-k past incidents with their confirmed causes.
4. **LangGraph investigator** (section 6) reads the evidence tools and writes a report with a ranked cause list and the evidence it used. It may *only* cite
   evidence returned by its tools (same grounding rule as the brain, reusing `brain/agent/grounding.py` ideas).

## 5. Fault injection and ground truth

Four faults from the roadmap, each an overlay on a nominal run (parameters in `ops/fault_injection/*.yaml`, applied in the simulator that already exists;
MuJoCo first because it is fast and headless, a Gazebo subset for the demo):

| Fault | How injected | Expected signature |
|---|---|---|
| High D-gain chatter | `kd` ×10 (config) | high-frequency joint velocity, no speed loss |
| Weak motors | effort limit ×0.5 on all (or one) joints | tracking error on all legs, effort saturation, speed deficit |
| Slippery floor | floor friction ×0.2 | commanded vs measured speed gap, no leg error, possible `stuck` |
| Disabled leg | existing `limp` / `locked` leg failure (Exp B env, `leg_failure_node`) | one-leg tracking error, `leg_failure` alert |

Ground truth = the injection config, stored with the run (`runs.fault_label`). Dataset plan (estimate): 4 faults × 3 severities × 5 seeds + 20 healthy
runs ≈ 80 runs of 60–120 s, MuJoCo generation ~minutes each in parallel; plus 4 Gazebo recordings for the demo. Datasets live in `datasets/ops_v1/` (JSONL, small).
The hard pair is *weak motors vs disabled leg* (similar symptoms; they differ in which legs); expect confusion there and report it honestly (confusion matrix, not only ARI).

## 6. Queries for the brain: "what went wrong today?"

New read-only brain tools (added to the tool layer; backend = FastAPI client, `RobotBackend`-style fake for tests; the existing 24 tools stay):

| Tool | Args | Returns |
|---|---|---|
| `incidents` | `since?` (today / last_run / ISO), `severity?`, `status?` | list: id, title, start, end, severity, top cause + confidence, n events |
| `incident_detail` | `id` | members (counts by type), metric highlights, causes with evidence, similar past incidents |
| `explain_incident` | `id` | investigator report (cached, else queued; returns "investigating" with an ETA) |
| `compare_runs` | `run_a`, `run_b` | config diff + metric deltas |

Example flow: *"what went wrong today?"* → `incidents(since=today)` → grounded template: "Two incidents today. 14:02 — critical, front-left leg failure, 3 events,
cause: leg tracking error 9× the others (confidence 0.8). 15:40 — warning, slipping on the way to room B." Follow-up *"why did the first one happen?"* → `explain_incident`.
Numbers and names in the reply must come from the tool result (existing grounding check). Test cases go into the brain eval as a new category (10 cases, scripted incident DB) so the
brain's use of these tools is measured, not assumed. The brain keeps working without the ops stack: tools return `UNAVAILABLE`, as for the other missing backends.

## 7. Minimal stack for the laptop (8 GB WSL)

Estimates (to be measured in 6.1), steady state while idle / during replay:

| Component | RAM (est.) | Notes |
|---|---|---|
| Redpanda | 0.5–1.0 GiB | `--smp 1 --memory 512M --overprovisioned`; the heaviest part |
| PostgreSQL | ~0.15 GiB | small `shared_buffers` |
| Qdrant | 0.15–0.3 GiB | few thousand vectors |
| Redis | ~0.03 GiB | |
| FastAPI + 2 Celery workers | 0.4–0.6 GiB | embedding model on CPU adds ~0.3 GiB in the normalizer worker |
| Investigator LLM | **GPU**: Qwen3-1.7B Q4 via Ollama, 1.7 GiB VRAM (measured, 5.1); the SFT/base brain model can be reused | only during investigations |
| **Total CPU RAM** | **~1.5–2.5 GiB** | fits 8 GB with nothing else heavy; ops compose profile only |

Choices that keep it small: single-broker Redpanda, one Postgres DB, no Kafka Connect / Schema Registry / Grafana in the core (Phase 7 web page is the UI),
replay instead of live sim, batch clustering every few seconds instead of streaming ML. If measured RAM > 3 GiB or startup > 60 s, switch the default to `--lite` (section 3).

## 8. Milestones and time estimates

Estimates are working days of focused work for one person, plus machine time; they assume the Phase 5 event stream exists (5.7). They are guesses until 6.1 is done.

| # | Milestone | Output / test | Work | Machine |
|---|---|---|---|---|
| 6.1 | **Stack + budget**: `docker/ops`, compose profile `ops`, Redpanda/Postgres/Qdrant/Redis up, FastAPI hello, RAM measured idle + under load | measured RAM / start time; decision on `--lite` | 1.5 d | minutes |
| 6.2 | **Schema + exporter + replay**: event/metric JSONL schema, `gurdy_exporter` node, `replay.py`, fingerprinting, tests without ROS | one recorded run replays into Redpanda → Postgres | 2 d | minutes |
| 6.3 | **Metric features + embeddings → Qdrant**; choose embedding model | features unit-tested on MuJoCo runs; embeddings stored | 2 d | minutes |
| 6.4 | **Fault injection + datasets** (`ops_v1`: 4 faults, healthy, seeds) in MuJoCo; Gazebo recordings for 4 faults | labelled runs on disk; chatter metric reproduces the Gurdy 1.0 failure mode | 2.5 d | ~2 h generation |
| 6.5 | **Clustering → incidents** (HDBSCAN, merge/close rules) + purity / ARI + confusion matrix | AC: each fault forms its own cluster, numbers reported | 2.5 d | minutes |
| 6.6 | **Root-cause linking**: rule hypotheses, config diff, similar-incident search | rule-only accuracy on the 4 faults reported (baseline before the LLM) | 2 d | minutes |
| 6.7 | **Investigator agent** (LangGraph, evidence tools, grounding of its report) + eval on all fault runs | AC: ≥ 3 of 4 root causes correct, vs rule baseline; hallucinated-evidence count = 0 | 3 d | ~1 h GPU eval |
| 6.8 | **FastAPI + brain tools** (`incidents`, `incident_detail`, `explain_incident`, `compare_runs`), brain eval category | "what went wrong today?" works on a scripted day; 10 new eval cases | 2 d | minutes |
| 6.9 | **Demo + write-up**: recorded day with two injected faults, incident view (CLI/JSON; web page is Phase 7), results doc, ROADMAP/README | video + `docs/results/phase6_*.md` | 1.5 d | Gazebo run ~1 h |
| | **Total** | | **~19 days** (range 15–25) | |

Ordering notes: 6.1 first (RAM budget decides lite vs full), 6.4 early (datasets drive everything after), 6.5 before 6.7 (clustering quality bounds what the investigator can do).
The Phase 7 incidents page only needs the FastAPI endpoints from 6.8.

## 9. Risks and open questions

- **Metric-first separation may be too easy** with MuJoCo-only faults (clean signals); the Gazebo recordings and noise injection are the honest check. Report both.
- **Weak motors vs disabled leg** may merge; the AC asks for separate clusters per fault, so severity levels and one-vs-all-legs variants must be chosen so this is a real test, not tuned to pass.
- **Leg-health thresholds are not calibrated** (5.7 note); calibrate against healthy walking first (false-alarm rate) in 6.3/6.6.
- **Investigator evaluation leakage**: the config diff contains the injected change by construction; evaluate also with config-diff hidden (metrics + logs only) and report both.
- **Qdrant + HDBSCAN on a few hundred events is overkill**; they stay because they are the roadmap stack and the interview story, but the lite path must give the same incidents (tested).
- Decisions for the user: keep full Redpanda stack as default vs lite as default; embedding model (CPU small vs reuse Ollama embeddings); whether `explain_incident` may block the voice loop (proposed: no, returns "investigating").
