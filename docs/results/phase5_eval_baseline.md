# Phase 5 — test set v1 and keyword baseline (milestone 5.3 part 1)

Date 2026-11-17. Everything below was run in the mock flat_lab world (virtual clock) in the `brain` image
(`docker compose run --rm brain python3 -m brain.eval.run --agent keyword --failures`). Raw output:
`phase5_eval_keyword_v1.txt`.

## What was built
- `brain/eval/testset_v1.yaml`: 100 English commands, the 9 categories of `docs/phase5_design.md` 5.1
  (20 / 12 / 14 / 8 / 10 / 8 / 8 / 10 / 10). Each case = scenario + turns + allowed tool traces + reply facts + end-state predicates.
- `brain/eval/harness.py` (+ `agents.py`, `run.py`): runs a case, classifies what the agent *did* per turn
  (execute / answer / clarify / refuse / confirm / none), checks planned calls, reply facts and the mock-world end state.
- Metrics: success, tool-sequence match, argument accuracy, direction errors (own column), clarification asked / unnecessary,
  refusals / over-refusals / confirmation violations, grounding (replies the planner wrote; rejected + final violations),
  latency p50/p95, LLM calls per turn, tool calls per executed turn.
- **Oracle agent** (plans the first allowed trace of every turn): 100 / 100 cases pass. This proves the cases are solvable and
  the scoring consistent; it is pinned by `brain/tests/test_eval.py`. It is an upper bound, not a competitor.

## Keyword baseline (Gurdy-1.0-style regex parser as the planner, inside the same agent graph)
Success **55 / 100**.

| # | Category | Pass | % |
|---|---|---|---|
| 1 | single motion / navigation | 15/20 | 75.0 |
| 2 | search and inspection | 9/12 | 75.0 |
| 3 | spatial questions | 7/14 | 50.0 |
| 4 | status, why, memory | 3/8 | 37.5 |
| 5 | multi-step missions | 8/10 | 80.0 |
| 6 | multi-turn dialogues | 1/8 | 12.5 |
| 7 | ambiguous requests | 1/8 | 12.5 |
| 8 | unsafe / risky | 8/10 | 80.0 |
| 9 | noisy STT (synthetic) | 3/10 | 30.0 |

| Metric | Keyword baseline |
|---|---|
| tool-sequence match / argument accuracy (when the sequence matched) | 74.8 % / 91.5 % |
| direction errors (left/right, forward/back swapped) | 0 |
| clarification asked when needed | 2 / 10 (20 %); unnecessary clarifications on clear commands: 1 / 95 |
| refused when needed | 6 / 7; over-refusals 0 |
| confirmation asked when needed | 3 / 4 |
| risky action executed without confirmation / despite refusal | **1** of 11 risky turns (`patrol ... five times`: the parser never reads the loop count) |
| grounding violations in final replies | 0 of 45 checked (replies are templates) |
| planner latency p50 / p95 | 3.3 ms / 48 ms per turn (a regex; says nothing about an LLM) |

Why it fails (read from the per-case output): no number words ("two", "forty five" -> default 1 m / 90 degrees, silently wrong),
no map points, no pronouns / ellipsis ("go there", "and again", "how far is it"), no asking back (7 of 8 ambiguous cases end in
"Sorry, I did not understand"; `turn` with no angle is *guessed* as left 90 degrees), no question words beyond a few phrases,
and `why did you stop` is parsed as the command stop. The 80 % in category 8 is mostly the shared safety layer, not the parser.

## Caveats (read before quoting numbers)
- The baseline runs inside our agent graph, so the safety checker, confirmation gate and grounding apply to it as well. A
  Gurdy-1.0 system had none of these; its safety numbers would be worse.
- Category 9 transcripts are **hand-written** Whisper-style errors (marked `noise: synthetic`), not recorded Whisper output.
  They are to be replaced with real transcripts when voice exists (5.5 / 5.8).
- The mock world is a model: nominal speeds, a camera cone, no detector errors, no Nav2. Success here means "right tool calls,
  right end state in the mock", not a Gazebo result.
- Latency is the planner only. LLM latency, STT and TTS get measured in 5.1 / 5.3 part 2.
- Single seed, deterministic world; the keyword run is bit-for-bit repeatable (tested).
- Phrasings are hand-written, not sampled from users; keeping them held out from the training generator (5.4) still has to be enforced there.

## Agent bugs the test set found (fixed, with tests)
1. `why did you stop` / `when did you stop` triggered the **e-stop fast path** (answered "Stopped." and cancelled the mission).
2. `disable the emergency stop` triggered the e-stop fast path instead of the refusal. The unsafe-request rule now runs first and
   a question about stopping is not a stop command.
3. `TurnResult.calls` was emptied when safety refused or asked back, hiding what the planner had proposed. It is now kept
   (the verdict says what happened).
Also: three `find` cases started in room A with the default 300 s timeout and never reached room B in the mock's search tour;
they now start in room B (a test-set flaw, not an agent bug).

## Re-plan after a failed mission step (also in this step)
`Agent._on_mission_failed`: when the executor reports `mission_failed` (not a user cancel), the LLM is asked **once**
(`llm.replan(system, state, tools, failure, tool_results, feedback)`; `failure` = every step with status + the failed step and
reason). Its calls go through the safety checker like any plan (refuse / clarify -> the reason is spoken, confirm -> held in the
gate). A text reply passes the grounding check (one regeneration with feedback, then the plain failure sentence). A re-plan's own
failure is only reported (at most one re-plan per user request); the keyword planner returns `{}` (no recovery). Tested with
`ScriptedLLM` (5 tests). The test set has no case for it yet: a "path blocked mid-way" scenario hook belongs in testset v2.
