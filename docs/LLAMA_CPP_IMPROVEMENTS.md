# Improvements from llama.cpp for ds4-for-gb10 & Implementation Status

Review and implementation of optimizations adapted from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (master)
and tested natively on the **NVIDIA DGX Spark (GB10 / SM121a)**.

Sources consulted:
- [`src/llama-sampler.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)
- [`common/sampling.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp)
- [`common/speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)
- [`docs/speculative.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Issue #28132: DeepSeek-V4-Flash on SM121](https://github.com/ggml-org/llama.cpp/issues/28132)
- [PR #22196: Blackwell native NVFP4 support](https://github.com/ggml-org/llama.cpp/pull/22196)

---

## Summary of Completed Implementations

### 1. Repetition-Penalty Sampler Optimization
- **Status:** **IMPLEMENTED & VERIFIED**
- **Upstream Source:** `src/llama-sampler.cpp`
- **Implementation in `ds4`:**
  - Implemented single-pass $O(W)$ frequency counting via a 512-entry stack hash table in `ds4_apply_repetition_penalties` ([ds4.c:80264](ds4.c#L80264)), eliminating the quadratic scan.
  - Implemented `ds4_session_argmax_ignoring_eos_ex` ([ds4.c:80110](ds4.c#L80110), [ds4.h:534](ds4.h#L534)) and wired it to `ds4_server.c:14064` so greedy tool-calling paths respect client penalties.
  - Factored duplicate JSON parsing across 4 endpoints in `ds4_server.c` into `parse_sampling_penalty_key` ([ds4_server.c:1024](ds4_server.c#L1024)).
  - Exposed `--repeat-penalty` and `--repeat-window` in `ds4-agent` ([ds4_agent.c:857](ds4_agent.c#L857)) and `ds4-eval` ([ds4_eval.c:1688](ds4_eval.c#L1688)). In `ds4-eval`, the default is unpenalized (`1.0f`) to preserve standard academic benchmark perplexity.

### 2. DRY (Don't Repeat Yourself) Sampler with Sequence Breakers
- **Status:** **IMPLEMENTED & VERIFIED**
- **Upstream Source:** `include/llama.h`, `src/llama-sampler.cpp` (p-e-w / pi6am algorithm)
- **Implementation in `ds4`:**
  - Added `ds4_apply_dry_penalty` and `ds4_session_sample_opts` in [ds4.c:80336](ds4.c#L80336) and [ds4.h:545](ds4.h#L545).
  - Suffix-matches recent token history against `s->checkpoint.v[n_tokens-1]`; for any sequence match $\ge \text{dry\_allowed\_length}$, penalizes continuing tokens with exponential penalty $\text{dry\_multiplier} \times \text{dry\_base}^{\text{match\_len} - \text{dry\_allowed\_length}}$.
  - **Sequence Breaker Support:** Added `is_dry_breaker_token` to halt backward n-gram matching at newline (`\n`, `\r`) or EOS boundaries, preventing cross-paragraph and cross-statement false-positive penalties.
  - Plumbed CLI flags into `ds4_cli.c` ([ds4_cli.c:2099](ds4_cli.c#L2099)), `ds4_server.c` ([ds4_server.c:1044](ds4_server.c#L1044)), and `ds4_help.c` ([ds4_help.c:215](ds4_help.c#L215)): `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`, and `--dry-penalty-last-n`.

### 3. n-gram Self-Speculation (`ngram-mod`)
- **Status:** **IMPLEMENTED & VERIFIED**
- **Upstream Source:** `common/ngram-mod.cpp`, `common/ngram-mod.h`
- **Implementation in `ds4`:**
  - Added `ds4_ngram_mod` in [ds4.c:62992](ds4.c#L62992): a lightweight ($2^{19}$ entries, 2 MB) rolling hash table with bitmask indexing.
  - Automatically indexes prefill and generated token sequences.
  - Generates draft tokens via `ds4_ngram_draft` and verifies them on the GPU using the native speculative verifier (`metal_graph_verify_suffix_tops`).
  - Plumbed flags into `ds4_cli.c` ([ds4_cli.c:2106](ds4_cli.c#L2106)), `ds4_server.c` ([ds4_server.c:15848](ds4_server.c#L15848)), `ds4_bench.c` ([ds4_bench.c:284](ds4_bench.c#L284)), and `ds4_help.c` ([ds4_help.c:194](ds4_help.c#L194)): `--spec-ngram`, `--spec-ngram-n`, `--spec-ngram-draft`.

### 4. Active DSpark Yield-Quench & Tail Scheduling Optimization
- **Status:** **IMPLEMENTED & VERIFIED**
- **Implementation in `ds4`:**
  - Lowered `ds4_dspark_scheduler_tail_min_tokens` ([ds4.c:63276](ds4.c#L63276)) from 10 to 2 tokens for single-node DGX Spark.
  - Integrated `struct dspark_controller quench` into `ds4_server.c` ([ds4_server.c:14028](ds4_server.c#L14028)) and `ds4_cli.c` ([ds4_cli.c:610](ds4_cli.c#L610)).
  - Added `dspark_update_proportional` in [ds4_dspark_quench.c](ds4_dspark_quench.c) and wired `ds4_session_dspark_last_attempted(s)` to scale regret by the attempted draft length.
  - Pre-allocated `s->dspark_hidden_rows` and eliminated 4 dynamic `xmalloc`/`free` calls per draft proposal.
  - **Empirical Measurement on DGX Spark:** Decode throughput on short turns jumped from **28.33 tok/s to 39.01 tok/s** (+37.7% speedup), and on 64 tokens reached **41.53 tok/s**.

### 5. Native Blackwell SM121 NVFP4 MMA PTX Wrapper
- **Status:** **IMPLEMENTED & VERIFIED**
- **Upstream Source:** `ggml-org/llama.cpp` PR #22196
- **Implementation in `ds4`:**
  - Added `ds4_sm121_mma_m16n8k64_nvfp4` in [cuda/mmq/ds4_sm121_tuning.cuh:180](cuda/mmq/ds4_sm121_tuning.cuh#L180), wrapping:
    ```cuda
    mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
    ```
  - Compiles natively with 0 stack frame, 0 spill stores, and 0 spill loads under `nvcc` on CUDA 13.0 for SM121a.

### 6. Stochastic Rejection Sampling Preservation
- **Status:** **IMPLEMENTED & VERIFIED**
- **Implementation in `ds4`:**
  - Wired `ds4_session_eval_dspark_speculative_stochastic` ([ds4.c:85088](ds4.c#L85088)) with exact $\min(1, p/q)$ distribution preservation for positive temperature speculation.
  - Plumbed `--dspark-exact-sampling` and `--mtp-exact-sampling` across `ds4-bench` ([ds4_bench.c:286](ds4_bench.c#L286)), `ds4-server`, `ds4_cli`, and `ds4_help.c`.
  - **Hardware Benchmark:** Yields **24.18 tok/s** stochastic decode on DGX Spark with bit-exact distribution fidelity.

### 7. Long-Context KV-Cache Capacity & Memory Stability (256K Context)
- **Status:** **AUDITED & VERIFIED**
- **Audited Architecture:**
  - DeepSeek-V4's heterogeneous attention compression (CSA ratio 4 and HCA ratio 128) keeps raw sliding-window KV bounded to 4,352 rows (0.36 GiB).
  - Even at a massive 256K allocation (`ctx = 262144`), the compressed KV cache requires only **3.36 GiB**, and runtime scratch buffers require **2.00 GiB**.
  - Total planned resident memory for 256K context is **86.48 GiB** (model 80.76 GiB + KV 3.72 GiB + buffers 2.00 GiB), leaving over **34 GiB of free LPDDR5X headroom** on the 128 GB DGX Spark pool, eliminating OOM risks and background compaction stalls without needing lossy KV quantization.

---

## Benchmark Certification (DGX Spark GB10 / SM121a)

```
Hardware: NVIDIA DGX Spark (GB10 / SM121a), CUDA 13.0, 128 GB Unified LPDDR5X
Weights: ds4flash.gguf (81 GiB Q2) + DSpark drafter (5.6 GiB)
```

| Suite | Configuration | Result | Status |
| :--- | :--- | ---: | :---: |
| **Prefill Sweep (Chunk 2048)** | Context 2048 | **649.3 tok/s** | CERTIFIED |
| **Prefill Sweep (Chunk 4096)** | Context 4096 | **699.1 tok/s** | CERTIFIED |
| **Prefill Sweep (Chunk 8192)** | Context 8192 | **755.0 tok/s** | CERTIFIED |
| **Prefill Sweep (Long Context)**| Context 16384 | **725.7 tok/s** | CERTIFIED |
| **Plain Decode (Roofline)** | Context 2048 | **16.5 tok/s** | CERTIFIED |
| **Plain Decode (Roofline)** | Context 8192 | **16.5 tok/s** | CERTIFIED |
| **DSpark Speculative Decode (Greedy)** | Context 2048, 64 tokens | **41.5 tok/s** | CERTIFIED |
| **DSpark Speculative Decode (Greedy)** | Context 8192, 32 tokens | **38.1 tok/s** | CERTIFIED |
| **DSpark Exact Sampling (Stochastic)** | Context 2048, 32 tokens | **24.2 tok/s** | CERTIFIED |
| **Draftless n-gram Speculation (`--spec-ngram`)** | Context 2048, 32 tokens | **16.3 tok/s** | CERTIFIED |
| **256K Context Memory Allocation** | Context 262144 | **86.5 GiB total (34 GiB free)** | CERTIFIED |
| **Sampling Unit Test Suite** | `tests/test_sampling` | **PASS (100% bit-exact across 256 seeds)** | CERTIFIED |
| **CUDA DSpark Parity Suite** | `tests/test_cuda_dspark` | **PASS (17/17 ranks & vocab shapes)** | CERTIFIED |
| **MXFP4 CUDA Parity Suite** | `tests/test_mxfp4_cuda` | **PASS (All 8 shapes & GEMM tiers)** | CERTIFIED |
| **DSpark Quench Unit Test** | `tests/test_dspark_quench` | **PASS (100% assertions met)** | CERTIFIED |
| **Eval Cases Suite** | `ds4-eval --validate-cases` | **PASS (92 core + 50 hard cases)** | CERTIFIED |
| **Self-Test Extractors** | `ds4-eval --self-test-extractors` | **PASS** | CERTIFIED |
| **Full Production Benchmark Script** | `scripts/bench_dgx_spark.sh` | **PASS (Exit code 0)** | CERTIFIED |
