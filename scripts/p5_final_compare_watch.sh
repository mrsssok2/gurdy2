#!/bin/bash
# Wait for a final-eval run to finish, then write the three-column comparison doc and commit ONLY that file.
#   nohup scripts/p5_final_compare_watch.sh brain/runs/final_eval/<fresh run> <eval script pid> > /tmp/p5_compare.out 2>&1 &
# Waits until the run's status.txt says DONE, or the eval script (pid) has died without it (then the doc is written from what exists, marked partial), max 5 h.
cd "$(dirname "$0")/.."
RUN=$1; PID=${2:-0}
OLD=${OLD_RUN:-brain/runs/final_eval/20261118_181226}
DOC=docs/results/phase5_final_eval.md
end=$((SECONDS + 18000))
until grep -q '^DONE' "$RUN/status.txt" 2>/dev/null; do
  [ "$PID" != 0 ] && ! kill -0 "$PID" 2>/dev/null && { echo "eval script $PID is gone without DONE"; break; }
  [ $SECONDS -gt $end ] && { echo "gave up after 5 h"; break; }
  sleep 30
done
sleep 5
python3 -m brain.eval.final_compare --first-pass "$OLD/first_pass" --rerun "$OLD" --fresh "$RUN" --notes docs/results/phase5_final_eval_fixes.md --doc "$DOC"
if grep -q '^DONE' "$RUN/status.txt" 2>/dev/null; then note=""; else note=" (PARTIAL: the run did not finish)"; fi
git add "$DOC" && git commit -q -m "5.11 final eval: three-column results (first pass / after setup re-runs / fresh run after the fixes)$note

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git log --oneline -1
