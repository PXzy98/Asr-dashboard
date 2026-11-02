#!/usr/bin/env bash
# Start a vLLM server exposing Qwen3-ASR for ASR Dashboard (Linux / WSL2).
#   /v1/realtime          (WebSocket streaming, used by "实时转录")
#   /v1/chat/completions  (used by the "openai_chat_audio" API engine)
#   /v1/audio/transcriptions
#
# Usage: ./start_realtime.sh [1.7b|0.6b] [port] [gpu_memory_utilization]
#   ./start_realtime.sh 0.6b 8000 0.5
#
# Requirements: NVIDIA/AMD GPU driver, Python 3.11/3.12, vLLM >= 0.32 with audio extras:
#   uv venv && source .venv/bin/activate && uv pip install "vllm[audio]>=0.32"
#
# Why --hf-overrides: vLLM only enables the /v1/realtime endpoint for models that implement its realtime
# interface. For Qwen3-ASR that is the architecture "Qwen3ASRRealtimeGeneration" (vllm/model_executor/models/
# qwen3_asr_realtime.py); the checkpoints declare "Qwen3ASRForConditionalGeneration", so we override it.
set -euo pipefail

VARIANT="${1:-1.7b}"
PORT="${2:-8000}"
GPU_UTIL="${3:-0.8}"

case "$VARIANT" in
  1.7b|1.7B) MODEL="Qwen/Qwen3-ASR-1.7B" ;;
  0.6b|0.6B) MODEL="Qwen/Qwen3-ASR-0.6B" ;;
  *) echo "variant must be 1.7b or 0.6b" >&2; exit 1 ;;
esac

REALTIME="${REALTIME:-1}"   # REALTIME=0 -> plain serving without the realtime architecture override
EXTRA=()
if [ "$REALTIME" = "1" ]; then
  EXTRA+=(--hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}')
fi
if [ -n "${API_KEY:-}" ]; then
  EXTRA+=(--api-key "$API_KEY")
fi

echo "Serving $MODEL on 0.0.0.0:$PORT (gpu_memory_utilization=$GPU_UTIL, realtime=$REALTIME)"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "${EXTRA[@]}"
