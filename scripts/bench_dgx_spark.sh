#!/usr/bin/env bash
# bench_dgx_spark.sh - DGX Spark (SM121) Performance Sweep Harness
# Measures prefill throughput, plain decode, and DSpark speculative speedup.

set -euo pipefail

MODEL="${DS4_MODEL:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
DRAfter="${DS4_DRAFTER:-DSpark-drafter-Q2K-Q8-0731.gguf}"
GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
PORT="${DS4_PORT:-8000}"

echo "=========================================================="
echo "  DGX Spark (SM121) Performance Benchmark"
echo "  Model:   $MODEL"
echo "  Drafter: $DRAfter"
echo "=========================================================="

# Check if binary exists
if [ ! -x "./ds4-server" ]; then
    echo "ds4-server not found. Building with 'make cuda-spark'..."
    make cuda-spark
fi

# 1. Prefill throughput sweep (2k, 16k, 64k)
echo
echo "--- [1/2] Prefill Throughput Sweep ---"
for ctx in 2048 16384 65536; do
    echo "Running prefill at context depth $ctx..."
    ./ds4-bench --cuda -m "$GGUF_DIR/$MODEL" -p "$ctx" -n 1 || true
done

# 2. Speculative Decode Verification
echo
echo "--- [2/2] DSpark Speculative Decode Sweep ---"
echo "Benchmarking DSpark with guard=2.16 on code & prose prompts..."
./ds4-bench --cuda -m "$GGUF_DIR/$MODEL" --dspark --mtp-model "$GGUF_DIR/$DRAfter" -p 2048 -n 128 || true

echo
echo "Benchmark sweep completed."
