#!/usr/bin/env bash
# serve_dgx_spark.sh - Production Turnkey Server Launcher for NVIDIA DGX Spark (GB10 / SM121)
# Optimally tuned for DeepSeek-V4-Flash: 256K Context, 2 Resident Sessions, DSpark 40 tok/s, NVMe KV Cache.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${DS4_MODEL:-./ds4flash.gguf}"
DRAFTER="${DS4_DRAFTER:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}"
PORT="${DS4_PORT:-8000}"
HOST="${DS4_HOST:-0.0.0.0}"
CTX="${DS4_CTX:-262144}"
SESSIONS="${DS4_SESSIONS:-2}"
PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-8192}"
DCONF="${DS4_DCONF:-0.4}"
KV_DIR="${DS4_KV_DIR:-/tmp/ds4-kv-cache}"
KV_MB="${DS4_KV_MB:-65536}"

# 1. Verify model file exists
if [ ! -f "$MODEL" ]; then
    echo "Errore: Modello base non trovato in $MODEL"
    echo "Assicurati che il symlink ds4flash.gguf punti al file GGUF."
    exit 1
fi

mkdir -p "$KV_DIR"

# 2. Configure Grace CPU performance core affinity (Cortex-X925)
TASKSET_CMD=""
if command -v taskset >/dev/null 2>&1; then
    TASKSET_CMD="taskset -c 5-9,15-19"
fi

# 3. Enable Blackwell SM121 runtime optimizations
export DS4_CUDA_FP4_INDEX=1
export DS4_CONT_CAPTURE=1

# 4. Check drafter for DSpark Speculation
DRAFTER_FLAG=""
if [ -f "$DRAFTER" ]; then
    DRAFTER_FLAG="--dspark --mtp-model $DRAFTER --dspark-confidence $DCONF"
fi

echo "=========================================================="
echo "  NVIDIA DGX Spark (GB10 / SM121) Production Server"
echo "  Model:           $MODEL"
[ -n "$DRAFTER_FLAG" ] && echo "  Drafter:         $DRAFTER (DSpark conf=$DCONF, ~40 tok/s)"
echo "  Context:         $CTX tokens"
echo "  Sessions:        $SESSIONS residenti (--batched-session)"
echo "  Prefill Chunk:   $PREFILL_CHUNK tokens (>900 tok/s)"
echo "  KV Disk Offload: $KV_DIR (${KV_MB} MB)"
echo "  API Endpoint:    http://${HOST}:${PORT}/v1"
echo "=========================================================="

# 5. Launch optimized server
exec $TASKSET_CMD ./ds4-server --cuda \
    -m "$MODEL" \
    $DRAFTER_FLAG \
    -c "$CTX" \
    --batched-session "$SESSIONS" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --host "$HOST" \
    --port "$PORT" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    "$@"
