#!/usr/bin/env bash
# serve_dgx_spark.sh - Production Turnkey Server Launcher for NVIDIA DGX Spark (GB10 / SM121)
# Optimally tuned for DeepSeek-V4-Flash: 256K Context, DSpark Speculative Decoding (~40 tok/s), NVMe KV Cache.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${DS4_MODEL:-./ds4flash.gguf}"
DRAFTER="${DS4_DRAFTER:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}"
PORT="${DS4_PORT:-8000}"
HOST="${DS4_HOST:-0.0.0.0}"
CTX="${DS4_CTX:-262144}"
PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-4096}"
DCONF="${DS4_DCONF:-0.4}"
KV_DIR="${DS4_KV_DIR:-/tmp/ds4-kv-cache}"
KV_MB="${DS4_KV_MB:-65536}"

# 1. Verify the server binary and model file exist
if [ ! -x ./ds4-server ]; then
    echo "Errore: binario ./ds4-server non trovato. Esegui prima: make cuda-spark"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Errore: Modello base non trovato in $MODEL"
    echo "Assicurati che il symlink ds4flash.gguf punti al file GGUF."
    exit 1
fi

mkdir -p "$KV_DIR"

# 2. Best-effort GPU/driver availability check (nvidia-smi may fail on GB10)
if [ ! -e /dev/nvidia0 ] || ! lsmod | grep -q '^nvidia '; then
    echo "Attenzione: GPU NVIDIA non rilevata (/dev/nvidia0 o driver nvidia assenti)."
    echo "Il server CUDA potrebbe non avviarsi. Verifica il driver."
fi

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
echo "  Prefill Chunk:   $PREFILL_CHUNK tokens (~700 tok/s)"
echo "  KV Disk Offload: $KV_DIR (${KV_MB} MB)"
echo "  API Endpoint:    http://${HOST}:${PORT}/v1"
echo "=========================================================="

# 5. Optional host tuning (best-effort, needs root): disable background page
#    compaction stalls that add 4-5s jitter on GB10. Non-fatal if not writable.
if [ -w /proc/sys/vm/compaction_proactiveness ]; then
    echo 0 > /proc/sys/vm/compaction_proactiveness 2>/dev/null || true
    echo "  Host tuning:    vm.compaction_proactiveness=0 (active)"
fi

# 6. Launch optimized server (Single-stream / DSpark mode). The server handles
#    SIGTERM/SIGINT itself (logs the signal name and flushes the KV cache).
exec $TASKSET_CMD ./ds4-server --cuda \
    -m "$MODEL" \
    $DRAFTER_FLAG \
    -c "$CTX" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --host "$HOST" \
    --port "$PORT" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    "$@"
