cd ~/gurdy2; export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco
M=models/terrain_v1_60M.onnx
for opt in "--latency-ms 0" "--latency-ms 10" "--latency-ms 20" "--latency-ms 30" "--friction-scale 0.6"; do
  echo "== $opt"
  python3 scripts/policy_track_mujoco.py --model $M $opt | python3 -c "
import json,sys; r=json.load(sys.stdin)
print(' '.join(f\"{s['segment'][:6]}:{s['main_axis_error_pct']:+.0f}\" for s in r['segments']), '| fell', r['fell'])"
done
