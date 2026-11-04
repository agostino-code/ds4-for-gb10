#!/usr/bin/env bash
# test_prefix_cache_and_server.sh - Multi-turn Warm Prefix Cache & Server Verification
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="./ds4flash.gguf"
DRAFTER_PATH="./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf"
SERVER_PORT=8008

echo "=========================================================="
echo "  [TEST 1] Warm Prefix Cache & Multi-Turn TTFT Verification"
echo "  Port: $SERVER_PORT | Model: $MODEL_PATH"
echo "=========================================================="

# Cleanup any previous server instance on this port
pkill -f "ds4-server.*$SERVER_PORT" || true
sleep 1

export DS4_CUDA_FP4_INDEX=1
export DS4_CONT_CAPTURE=1

# Start ds4-server in background
echo ">>> Avvio ds4-server in background..."
taskset -c 5-9,15-19 ./ds4-server \
  --cuda \
  -m "$MODEL_PATH" \
  --dspark \
  --mtp-model "$DRAFTER_PATH" \
  --dspark-confidence 0.4 \
  --prefill-chunk 8192 \
  --ctx 65536 \
  --port "$SERVER_PORT" \
  --host 127.0.0.1 > /tmp/ds4_server_test.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo ">>> Arresto ds4-server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo ">>> Attesa inizializzazione modelli e artefatti CUDA (circa 30s)..."
for i in $(seq 1 60); do
    if grep -q "listening on http" /tmp/ds4_server_test.log 2>/dev/null; then
        echo ">>> ds4-server PRONTO su http://127.0.0.1:$SERVER_PORT"
        break
    fi
    sleep 1
done

if ! grep -q "listening on http" /tmp/ds4_server_test.log 2>/dev/null; then
    echo "Errore: ds4-server non si e' avviato entro 60s. Log:"
    cat /tmp/ds4_server_test.log
    exit 1
fi

echo
echo "=========================================================="
echo "  Turno 1: Prompt Lungo (Cold Cache / Full Prefill)"
echo "=========================================================="
T1_START=$(date +%s%N)
RES1=$(curl -s "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "Sei un ingegnere esperto di sistemi Linux e architetture GPU Blackwell."},
      {"role": "user", "content": "Spiega in sintesi i vantaggi architetturali dell unione fisica di CPU e GPU su bus LPDDR5X ad alta velocita nel chip NVIDIA Grace Blackwell GB10."}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }')
T1_END=$(date +%s%N)
T1_MS=$(( (T1_END - T1_START) / 1000000 ))

echo "Risposta Turno 1 ricevuta in: ${T1_MS} ms"
echo "Contenuto Turno 1:"
echo "$RES1" | grep -o '"content":"[^"]*"' | head -n 1 || echo "$RES1"

echo
echo "=========================================================="
echo "  Turno 2: Stessa Conversazione + Domanda Aggiuntiva (Warm Prefix Cache)"
echo "=========================================================="
T2_START=$(date +%s%N)
RES2=$(curl -s "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "Sei un ingegnere esperto di sistemi Linux e architetture GPU Blackwell."},
      {"role": "user", "content": "Spiega in sintesi i vantaggi architetturali dell unione fisica di CPU e GPU su bus LPDDR5X ad alta velocita nel chip NVIDIA Grace Blackwell GB10."},
      {"role": "assistant", "content": "Il chip NVIDIA Grace Blackwell GB10 offre memoria unificata ad alta banda (LPDDR5X, circa 273 GB/s) condivisa coerentemente tra CPU ARM e GPU."},
      {"role": "user", "content": "Quale impatto ha questa memoria unificata sulla latenza di decode nei modelli Mixture-of-Experts come DeepSeek-V4?"}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }')
T2_END=$(date +%s%N)
T2_MS=$(( (T2_END - T2_START) / 1000000 ))

echo "Risposta Turno 2 ricevuta in: ${T2_MS} ms"
echo "Contenuto Turno 2:"
echo "$RES2" | grep -o '"content":"[^"]*"' | head -n 1 || echo "$RES2"

echo
echo "=========================================================="
echo "  Metriche di Caching e Timing dai Log del Server"
echo "=========================================================="
grep -E "chat ctx|live kv cache|prompt done" /tmp/ds4_server_test.log || true

echo
echo "Test completato con successo!"
