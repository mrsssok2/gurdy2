#!/bin/bash
# usage: sample_resources.sh OUT.csv SECONDS  -> every 2 s: gpu_used_mib, ram_used_mib, ram_avail_mib, swap_used_mib (WSL view)
echo "t,gpu_mib,ram_used_mib,ram_avail_mib,swap_used_mib" > "$1"; end=$((SECONDS + $2)); t0=$SECONDS
while [ $SECONDS -lt $end ]; do
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  read -r _ _ used _ _ _ avail < <(free -m | awk '/Mem:/{print}'); sw=$(free -m | awk '/Swap:/{print $3}')
  echo "$((SECONDS-t0)),$g,$used,$avail,$sw" >> "$1"; sleep 2
done
