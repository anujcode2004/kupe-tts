#!/usr/bin/env bash
# Stop OmniVoice / uvicorn leftovers and ask the NVIDIA driver to drop idle contexts.
set -euo pipefail

echo "Stopping OmniVoice-related Python servers …"
pkill -TERM -f "OmniVoice/server\.py" 2>/dev/null || true
pkill -TERM -f "uvicorn.*server:app" 2>/dev/null || true
sleep 1
pkill -KILL -f "OmniVoice/server\.py" 2>/dev/null || true
pkill -KILL -f "uvicorn.*server:app" 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU compute apps (before):"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true
  mapfile -t _pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk -F',' '{gsub(/ /,"",$1); print $1}')
  if ((${#_pids[@]})); then
    echo "Sending SIGTERM to GPU PIDs: ${_pids[*]}"
    kill -TERM "${_pids[@]}" 2>/dev/null || true
    sleep 2
    mapfile -t _pids2 < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk -F',' '{gsub(/ /,"",$1); print $1}')
    if ((${#_pids2[@]})); then
      echo "Sending SIGKILL to remaining GPU PIDs: ${_pids2[*]}"
      kill -KILL "${_pids2[@]}" 2>/dev/null || true
    fi
  fi
  echo "GPU status:"
  nvidia-smi 2>/dev/null || true
else
  echo "nvidia-smi not found — skipping GPU PID cleanup."
fi

echo "Done."
