#!/usr/bin/env bash
# bench_all_configs_matrix.sh - Comprehensive Configuration Sweep on DGX Spark (GB10)
# Evaluates Prefill Throughput, Decode Throughput, TTFT, and RAM Occupancy across configurations.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="./ds4flash.gguf"
DRAFTER="./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf"
PROMPT_FILE="/tmp/dgx_spark_bench_prompt.txt"
CSV_OUT="/tmp/benchmark_results_matrix.csv"

if [ ! -f "$PROMPT_FILE" ] || [ $(wc -c < "$PROMPT_FILE") -lt 100000 ]; then
    cat README.md docs/*.md Makefile > "$PROMPT_FILE" 2>/dev/null || true
fi

TASKSET_CMD=""
if command -v taskset >/dev/null 2>&1; then
    TASKSET_CMD="taskset -c 5-9,15-19"
fi

export DS4_CUDA_DIRECT_MODEL=1

echo "=========================================================="
echo "  DGX Spark (SM121) Full Matrix Benchmark Sweep"
echo "  Model: $MODEL"
echo "  Drafter: $DRAFTER"
echo "  Results CSV: $CSV_OUT"
echo "=========================================================="

echo "config_id,description,ctx_frontier,prefill_tps,decode_tps,ttft_ms,steady_tps" > "$CSV_OUT"

run_bench_case() {
    local cfg_id="$1"
    local desc="$2"
    local env_vars="$3"
    local extra_args="$4"
    local ctx="$5"
    local gen_tok="$6"

    echo
    echo "----------------------------------------------------------"
    echo ">>> Running [$cfg_id]: $desc (Context: $ctx, Gen: $gen_tok) <<<"
    echo "    Env: $env_vars"
    echo "    Args: $extra_args"
    echo "----------------------------------------------------------"

    # Execute and capture CSV output lines
    local raw_out
    raw_out=$(env $env_vars $TASKSET_CMD ./ds4-bench --cuda -m "$MODEL" \
        $extra_args \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start "$ctx" --ctx-max "$ctx" \
        --gen-tokens "$gen_tok" 2>&1)

    echo "$raw_out" | grep -E "planned|context buffers|ctx_tokens|[0-9]+,[0-9]+" || true

    local data_line
    data_line=$(echo "$raw_out" | grep -E "^[0-9]+,[0-9]+,[0-9.]+" | tail -n 1 || true)

    if [ -n "$data_line" ]; then
        # Format: ctx,prefill_toks,prefill_tps,gen_toks,gen_tps,first_ms,steady_toks,steady_tps,kv_bytes
        local p_tps=$(echo "$data_line" | cut -d',' -f3)
        local d_tps=$(echo "$data_line" | cut -d',' -f5)
        local ttft=$(echo "$data_line" | cut -d',' -f6)
        local s_tps=$(echo "$data_line" | cut -d',' -f8)
        echo "$cfg_id,\"$desc\",$ctx,$p_tps,$d_tps,$ttft,$s_tps" >> "$CSV_OUT"
    fi
}

# ==============================================================================
# DIMENSIONE 1: Prefill Chunk Size Comparison (Tensor Core Scaling)
# ==============================================================================
run_bench_case "PREFILL_CHUNK_2048" "Prefill Chunk 2048 (Baseline)" \
    "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
    "--prefill-chunk 2048" 8192 0

run_bench_case "PREFILL_CHUNK_4096" "Prefill Chunk 4096 (Standard)" \
    "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
    "--prefill-chunk 4096" 8192 0

run_bench_case "PREFILL_CHUNK_8192" "Prefill Chunk 8192 (Large Tile GB10)" \
    "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
    "--prefill-chunk 8192" 8192 0

# ==============================================================================
# DIMENSIONE 2: DSpark Confidence Threshold Grid (Speculation Sweep)
# ==============================================================================
for conf in 0.3 0.4 0.5 0.6 0.7; do
    run_bench_case "DSPARK_CONF_${conf}" "DSpark Speculative Decode (Confidence ${conf})" \
        "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
        "--dspark --mtp-model $DRAFTER --dspark-confidence $conf --prefill-chunk 8192" 4096 128
done

# ==============================================================================
# DIMENSIONE 3: FP4 Compressed Indexer vs Full Precision Indexer
# ==============================================================================
run_bench_case "INDEXER_FP4_ON" "FP4 Compressed Indexer (Memory Optimized)" \
    "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
    "--prefill-chunk 8192" 16384 64

run_bench_case "INDEXER_FP4_OFF" "Full Precision Indexer" \
    "DS4_CUDA_FP4_INDEX=0 DS4_CONT_CAPTURE=1" \
    "--prefill-chunk 8192" 16384 64

# ==============================================================================
# DIMENSIONE 4: High-Context Frontier Scaling (16k -> 32k -> 64k)
# ==============================================================================
for ctx_frontier in 16384 32768 65536; do
    run_bench_case "HIGH_CTX_${ctx_frontier}_PLAIN" "High Context Plain Decode (${ctx_frontier})" \
        "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
        "--prefill-chunk 8192" "$ctx_frontier" 64

    run_bench_case "HIGH_CTX_${ctx_frontier}_DSPARK" "High Context DSpark Decode (${ctx_frontier})" \
        "DS4_CUDA_FP4_INDEX=1 DS4_CONT_CAPTURE=1" \
        "--dspark --mtp-model $DRAFTER --dspark-confidence 0.4 --prefill-chunk 8192" "$ctx_frontier" 64
done

echo
echo "=========================================================="
echo "  BENCHMARK MATRIX COMPLETED! Risultati salvati in $CSV_OUT"
echo "=========================================================="
column -s, -t < "$CSV_OUT" || cat "$CSV_OUT"
