#!/usr/bin/env bash
# bench_advanced_sweep.sh - Advanced DGX Spark (SM121) Configuration Benchmarks
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="./ds4flash.gguf"
DRAFTER_PATH="./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf"
PROMPT_FILE="/tmp/dgx_spark_bench_prompt.txt"

if [ ! -f "$PROMPT_FILE" ] || [ $(wc -c < "$PROMPT_FILE") -lt 100000 ]; then
    cat README.md docs/*.md Makefile > "$PROMPT_FILE" 2>/dev/null || true
fi

TASKSET_CMD=""
if command -v taskset >/dev/null 2>&1; then
    TASKSET_CMD="taskset -c 5-9,15-19"
fi

export DS4_CUDA_FP4_INDEX=1
export DS4_CONT_CAPTURE=1

echo "=========================================================="
echo "  DGX Spark (SM121) Advanced Configuration Benchmarks"
echo "=========================================================="

echo
echo ">>> [TEST 1] Prefill Chunk Size Comparison: 4096 vs 8192 <<<"
echo "--- Prefill Chunk = 4096 (Baseline) ---"
$TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --prefill-chunk 4096 \
    --ctx-start 8192 --ctx-max 8192 \
    --gen-tokens 0

echo "--- Prefill Chunk = 8192 (Large Tile on Blackwell) ---"
$TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --prefill-chunk 8192 \
    --ctx-start 8192 --ctx-max 8192 \
    --gen-tokens 0

echo
echo ">>> [TEST 2] DSpark Confidence Threshold Sweep (Tuning Speculation) <<<"
for conf in 0.4 0.5 0.6; do
    echo "--- DSpark Confidence = $conf ---"
    $TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
        --dspark --mtp-model "$DRAFTER_PATH" \
        --dspark-confidence "$conf" \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start 4096 --ctx-max 4096 \
        --gen-tokens 128
done

echo
echo ">>> [TEST 3] Long-Context Scaling (16k -> 32k tokens) with FP4 Indexer <<<"
$TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --ctx-start 16384 --ctx-max 32768 --step-incr 16384 \
    --gen-tokens 64

echo
echo "Tutti i benchmark avanzati sono stati completati!"
