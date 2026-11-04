#!/usr/bin/env bash
# bench_dgx_spark.sh - Turnkey DGX Spark (SM121) Performance Benchmark
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1. Locate base model
MODEL_PATH=""
for p in "./ds4flash.gguf" \
         "./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
         "$HOME/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"; do
    if [ -f "$p" ]; then
        MODEL_PATH="$p"
        break
    fi
done

# 2. Locate drafter support model
DRAFTER_PATH=""
for p in "./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
         "$HOME/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
         "./gguf/DSpark-drafter-Q2K-Q8-0731.gguf" \
         "$HOME/gguf/DSpark-drafter-Q2K-Q8-0731.gguf"; do
    if [ -f "$p" ]; then
        DRAFTER_PATH="$p"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "Errore: Modello DeepSeek V4 Flash non trovato."
    echo "Scaricalo con: ./download_model.sh ds4f-q2"
    exit 1
fi

echo "=========================================================="
echo "  NVIDIA DGX Spark (SM121) Benchmark Suite"
echo "  Base Model: $MODEL_PATH"
[ -n "$DRAFTER_PATH" ] && echo "  Drafter:    $DRAFTER_PATH (DSpark attivo)"
echo "=========================================================="

# Taskset on Grace Cortex performance cores if available
TASKSET_CMD=""
if command -v taskset >/dev/null 2>&1; then
    TASKSET_CMD="taskset -c 5-9,15-19"
fi

# Enable Blackwell GB10 fast-path optimizations
export DS4_CUDA_FP4_INDEX=1
export DS4_CONT_CAPTURE=1

# Prepare a representative benchmark prompt file if needed
PROMPT_FILE="/tmp/dgx_spark_bench_prompt.txt"
if [ ! -f "$PROMPT_FILE" ] || [ $(wc -c < "$PROMPT_FILE") -lt 100000 ]; then
    cat README.md docs/*.md Makefile > "$PROMPT_FILE" 2>/dev/null || true
fi

# --- 1. Pure Prefill Sweep (D2R MoE Throughput) ---
echo
echo ">>> [1/3] Benchmark PREFILL (D2R MoE Tensor Cores) <<<"
$TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --ctx-start 2048 --ctx-max 16384 --step-incr 4096 \
    --gen-tokens 0

# --- 2. Plain Single-Token Decode (Bandwidth-bound roofline) ---
echo
echo ">>> [2/3] Benchmark PLAIN DECODE (Memoria LPDDR5X) <<<"
$TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --ctx-start 2048 --ctx-max 8192 --step-incr 2048 \
    --gen-tokens 128

# --- 3. DSpark Speculative Decode (se il drafter è presente) ---
if [ -n "$DRAFTER_PATH" ]; then
    echo
    echo ">>> [3/3] Benchmark DSPARK SPECULATIVE DECODE <<<"
    $TASKSET_CMD ./ds4-bench --cuda -m "$MODEL_PATH" \
        --dspark --mtp-model "$DRAFTER_PATH" \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start 2048 --ctx-max 4096 --step-incr 2048 \
        --gen-tokens 128
fi

echo
echo "Benchmark completato con successo!"
