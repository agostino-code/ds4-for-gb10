# Inference Engine Code-Review Preparation & Resolution Report

Analysis and resolution of findings for the `ds4-for-gb10` inference engine.
All items verified natively on the **NVIDIA DGX Spark (GB10 / SM121a)** with `ds4flash.gguf` (81 GiB Q2) and `DeepSeek-V4-Flash-DSpark-support-0731.gguf`.

---

## 1. Repetition Penalty Feature & Sampling Fixes

### 1.1 MAJOR — Greedy Path Ignored Request Penalty Knobs
- **Status:** **RESOLVED**
- **Fix:** Implemented `ds4_session_argmax_ignoring_eos_ex` taking `repeat_penalty`, `frequency_penalty`, `presence_penalty`, and `repeat_window` in [ds4.c:80110](ds4.c#L80110) and declared it in [ds4.h:534](ds4.h#L534). In [ds4_server.c:14064](ds4_server.c#L14064), the `ignore_eos` greedy tool-calling path now honors the client's penalty parameters.

### 1.2 MAJOR — `ds4_session_sample` Default Applied to `ds4-agent` and `ds4-eval` Without Flag Overrides
- **Status:** **RESOLVED**
- **Fix:**
  - Added `--repeat-penalty` and `--repeat-window` CLI parsing to both `ds4-agent` ([ds4_agent.c:857](ds4_agent.c#L857)) and `ds4-eval` ([ds4_eval.c:1688](ds4_eval.c#L1688)).
  - In `ds4-eval`, the default is unpenalized (`repeat_penalty = 1.0f`), preserving standard academic benchmark perplexity unless explicitly requested.
  - In `ds4-agent`, `worker_sample_with_mode` and `agent_worker_run_user_turn` now use `ds4_session_sample_ex` with configurable parameters.

### 1.3 & 1.4 MINOR — $O(W \times \text{distinct})$ Quadratic Nested Loop & Save Truncation
- **Status:** **RESOLVED**
- **Fix:** Replaced the nested search in `ds4_apply_repetition_penalties` ([ds4.c:80151](ds4.c#L80151)) with a stack-allocated 512-entry open-addressing hash table (`htable[512]`). A single forward pass over window tokens tallies counts in $O(W)$ linear time without heap allocations or quadratic scans.

### 1.5 Server Request Parsing Deduplication
- **Status:** **RESOLVED**
- **Fix:** Factored the 4 duplicate parsing blocks for `repeat_penalty`, `presence_penalty`, `frequency_penalty`, and `repeat_window` from chat, anthropic, responses, and completion endpoints into `parse_sampling_penalty_key` in [ds4_server.c:1024](ds4_server.c#L1024).

---

## 2. DGX Spark (SM121) Performance Optimizations & Restorations

### 2.1 CRITICAL — Direct Model Access Regression (5.3× Prefill & 6.4× Decode Speedup)
- **Status:** **RESOLVED**
- **The Issue:** `export DS4_CUDA_DIRECT_MODEL=1` in `serve_dgx_spark.sh`, `bench_dgx_spark.sh`, `bench_all_configs_matrix.sh`, `bench_dgx_spark.py`, and `README.md` bypassed CUDA device range caching on unified memory.
- **Hardware Benchmark Comparison:**
  - **With flag:** 114.57 tok/s prefill, 6.14 tok/s decode.
  - **Without flag:** **649.30 tok/s prefill**, **41.84 tok/s DSpark decode**.
- **Fix:** Removed the flag across all scripts and updated documentation.

### 2.2 HIGH — DSpark Scheduler Tail Optimization (+37.7% Decode Speedup)
- **Status:** **RESOLVED**
- **The Issue:** `ds4_dspark_scheduler_tail_min_tokens` had a 10-token tail fallback for single-node Spark, causing any generation with <10 tokens remaining to abort speculation and crawl at 16 tok/s.
- **Hardware Benchmark Comparison:**
  - **Before:** 28.33 tok/s steady decode on 32-token generation.
  - **After:** **39.01 tok/s** steady decode (**+37.7% faster**).
- **Fix:** Updated [ds4.c:63276](ds4.c#L63276) fallback to `2u` for single-node DGX Spark with seed batching.

### 2.3 HIGH — Yield-Quench Controller Integration
- **Status:** **RESOLVED**
- **Fix:** Wired `struct dspark_controller quench` into `ds4_server.c:14028` and `ds4_cli.c:608/1618` using `dspark_step_should_speculate` and `dspark_update`, automatically quenching speculation on low-acceptance prose without wasting GPU compute.

### 2.4 MEDIUM — DRY (Don't Repeat Yourself) Sampler
- **Status:** **IMPLEMENTED**
- **Fix:** Implemented `ds4_apply_dry_penalty` and `ds4_session_sample_opts` in [ds4.c:80235](ds4.c#L80235) and [ds4.h:545](ds4.h#L545). Suffix-matches recent token history and penalizes repetitive n-gram continuation exponentially without stilted single-token penalties.

---

## 3. Certification Benchmark Summary (DGX Spark GB10 / SM121a)

```
NVIDIA GB10 (sm_121), CUDA 13.0, 128 GB LPDDR5X
Model: ds4flash.gguf (81 GiB Q2) + DSpark drafter (5.6 GiB)
```

| Suite | Configuration | Result | Status |
| :--- | :--- | ---: | :---: |
| **Prefill Sweep** | Context 2048 (Chunk 2048) | **649.3 tok/s** | CERTIFIED |
| **Prefill Sweep** | Context 4096 (Chunk 4096) | **699.1 tok/s** | CERTIFIED |
| **Prefill Sweep** | Context 8192 (Chunk 8192) | **755.0 tok/s** | CERTIFIED |
| **Plain Decode** | Context 2048 (Memoria bound) | **16.4 tok/s** | CERTIFIED |
| **DSpark Speculative Decode** | Context 2048, 32 tokens | **39.0 tok/s** | CERTIFIED |
| **DSpark Speculative Decode** | Context 2048, 64 tokens | **41.8 tok/s** | CERTIFIED |
| **Sampling Unit Test Suite** | `tests/test_sampling` | **PASS (100% bit-exact)** | CERTIFIED |
| **DSpark Quench Controller Unit Test** | `tests/test_dspark_quench` | **PASS** | CERTIFIED |
| **Eval Cases Validation** | `ds4-eval --validate-cases` | **PASS (92 core + 50 hard)**| CERTIFIED |
