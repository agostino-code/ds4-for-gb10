# DGX Spark (SM121) Fast-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Entrpi's high-speed D2R prefill, DSpark speculative decoding with yield-quench, and single-session warm prefix caching into upstream `antirez/ds4` for DeepSeek-V4-Flash on DGX Spark (SM121), while preserving upstream native execution for Qwen 3.8 and GLM 5.3.

**Architecture:** The plan establishes a dedicated SM121 fast-path enabled under `make cuda-spark` (`CUDA_ARCH=sm_121a`). Architecture detection inspects GGUF metadata: when DeepSeek-V4-Flash is detected on GB10, the engine activates in-memory SoA repack, D2R MoE prefill GEMMs, DSpark speculative decode with dynamic quench, and warm prefix caching; when Qwen or GLM is detected, it delegates to upstream's tested native pipelines.

**Tech Stack:** C99, CUDA C++ (CUDA 13.x, `sm_121a`), PTX inline assembly, cuBLAS/cuBLASLt.

**Spec:** [`docs/superpowers/specs/2026-11-04-dgx-spark-fastpath-design.md`](../specs/2026-11-04-dgx-spark-fastpath-design.md)

## Global Constraints

- Target hardware: NVIDIA DGX Spark (Grace-Blackwell GB10, Compute Capability 12.1 `sm_121a`, 48 SMs, 100 KB shared memory per SM).
- Memory ceiling: 128 GB unified LPDDR5X (~119 GiB usable), ~227 GB/s effective read bandwidth.
- Model footprint: DeepSeek-V4-Flash 0731 Q2 imatrix (~81 GiB) + DSpark drafter Q2K (~6.5 GiB) must remain fully resident under 115 GiB with KV cache.
- Generic builds must not break: `make cuda` and `make cpu` must compile and pass existing test suites.
- Speculative yield-quench guard is calibrated to 2.16 for DeepSeek-V4-Flash 0731 weights.

## Review Focus

1. **Non-DeepSeek Model Routing:** When a user passes a Qwen 3.8 or GLM 5.3 GGUF to a binary compiled with `make cuda-spark`, the engine must execute via native upstream paths without invoking D2R repack or DeepSeek-specific DSpark hooks.
2. **Speculation Quench on Low Acceptance:** If draft acceptance falls below break-even (~51%), the yield-quench controller must shut down speculation within ~4 plain steps to prevent latency regression on unpredictable prose.
3. **In-Memory Repack Failure/OOM:** If allocating in-memory SoA repack buffers exceeds remaining unified memory, the engine must cleanly abort with a descriptive error instead of segfaulting.
4. **Context Cache Invalidation on Model Switch:** If the server is invoked with a different model or prompt reset, cached KV state from a previous run must be cleanly released or reinitialized without stale memory leaks.
5. **Tool-Protocol Reminder Invariance:** Deep tool conversation protocol reminders (>30k tokens) must only append to tools-armed requests, leaving plain chat requests untouched.

---

### Task 1: Build Configuration & Dynamic Model Fast-Path Routing

**Files:**
- Modify: `Makefile:1-80`
- Modify: `ds4.h:200-260`
- Modify: `ds4_cuda.cu:150-250`
- Test: `tests/test_model_routing.c`

**Interfaces:**
- Produces:
  - `bool ds4_cuda_is_spark_fastpath_enabled(int model_arch, int compute_cap)`
  - Macro `DS4_CUDA_FASTPATH_SPARK` in `Makefile` under target `cuda-spark`

- [ ] **Step 1: Write the failing test**

```c
// tests/test_model_routing.c
#include <assert.h>
#include <stdbool.h>
#include "ds4.h"

int main(void) {
    // GB10 compute capability is 1210 (SM 12.1)
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_DEEPSEEK4, 1210) == true);
    // Other models on GB10 must NOT engage DeepSeek fast-path
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_QWEN38, 1210) == false);
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_GLM53, 1210) == false);
    // Other architectures (e.g. Ada sm_89, Hopper sm_90) must not engage Spark fast-path
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_DEEPSEEK4, 890) == false);
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gcc -I. tests/test_model_routing.c -o /tmp/test_routing && /tmp/test_routing`
Expected: FAIL (symbol `ds4_cuda_is_spark_fastpath_enabled` undefined).

- [ ] **Step 3: Implement `ds4_cuda_is_spark_fastpath_enabled` and update `Makefile`**

Add `ds4_cuda_is_spark_fastpath_enabled` to `ds4.h` / `ds4_cuda.cu`. Update `Makefile` target `cuda-spark` to compile with `-DDS4_CUDA_FASTPATH_SPARK=1 -arch=sm_121a`.

- [ ] **Step 4: Run test to verify it passes**

Run: `gcc -I. tests/test_model_routing.c ds4_cuda_routing.c -o /tmp/test_routing && /tmp/test_routing`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Makefile ds4.h ds4_cuda.cu tests/test_model_routing.c
git commit -m "feat(cuda): add build flag and model routing for DGX Spark fast-path"
```

---

### Task 2: In-Memory SoA Weight Repack & D2R MoE Prefill Wiring

**Files:**
- Modify: `cuda/mmq/ds4_repack.cu`
- Modify: `cuda/mmq/ds4_mmq.cu:1880-2210`
- Modify: `ds4_cuda.cu:2500-2800`
- Test: `tests/test_d2r_prefill_wiring.c`

**Interfaces:**
- Consumes:
  - `ds4_cuda_is_spark_fastpath_enabled(model_arch, cc)` from Task 1
- Produces:
  - `int ds4_mmq_repack_model_weights_in_memory(struct ds4_model *m)`
  - Active dispatch to `ds4_mmq_iq2_xxs_moe_d2r_fused_launch` and `ds4_mmq_q2_K_moe_d2r_launch` in `ds4_cuda.cu` when prefilling $n_{tokens} \ge 2$.

- [ ] **Step 1: Write the failing test**

```c
// tests/test_d2r_prefill_wiring.c
#include <assert.h>
#include "cuda/mmq/ds4_mmq.h"

int main(void) {
    // Assert availability of D2R MoE kernels for CC 1210
    assert(ds4_mmq_q2_K_moe_d2r_available(1210) == true);
    assert(ds4_mmq_iq2_xxs_moe_d2r_available(1210) == true);
    // Assert scratchpad capacity calculation is non-zero
    assert(ds4_mmq_q2_K_moe_d2r_scratch_bytes(2048, 256) > 0);
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails or exposes missing wiring**

Run: `nvcc -arch=sm_121a -I. tests/test_d2r_prefill_wiring.c cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq_d2r.cu -o /tmp/test_d2r && /tmp/test_d2r`
Expected: Compile check / verification of D2R entry points.

- [ ] **Step 3: Wire in-memory repack and prefill launch in `ds4_cuda.cu`**

In `ds4_cuda.cu`, during model weight registration on SM121, call `ds4_mmq_repack_model_weights_in_memory`. In the MoE prefill dispatch branch ($n_{tokens} \ge 2$), replace raw cublas+dequant with the fused D2R launch calls (`ds4_mmq_iq2_xxs_moe_d2r_fused_launch` for gate/up, `ds4_mmq_q2_K_moe_d2r_launch` for down).

- [ ] **Step 4: Verify prefill with smoke test**

Run: `make cuda-spark && ./ds4 --cuda -m <test_gguf> -p "Test prefill" --prefill-chunk 512 -n 1`
Expected: Log prints `ds4: D2R prefill MoE fast-path active` and completes without error.

- [ ] **Step 5: Commit**

```bash
git add cuda/mmq/ds4_repack.cu cuda/mmq/ds4_mmq.cu ds4_cuda.cu tests/test_d2r_prefill_wiring.c
git commit -m "feat(cuda): wire in-memory SoA repack and D2R MoE prefill for SM121"
```

---

### Task 3: DSpark Speculative Decoding & Calibrated Yield-Quench Controller

**Files:**
- Modify: `ds4.h:350-420`
- Modify: `ds4.c:800-950`
- Modify: `ds4_cuda.cu:3200-3600`
- Test: `tests/test_dspark_quench.c`

**Interfaces:**
- Consumes:
  - `ds4_cuda_is_spark_fastpath_enabled` from Task 1
- Produces:
  - `struct dspark_controller { float guard; float regret_accum; bool quenched; int k_draft; }`
  - `void dspark_init(struct dspark_controller *c, float guard, int k_draft)`
  - `bool dspark_step_should_speculate(struct dspark_controller *c)`
  - `void dspark_update(struct dspark_controller *c, int accepted_tokens)`

- [ ] **Step 1: Write the failing test for yield-quench controller**

```c
// tests/test_dspark_quench.c
#include <assert.h>
#include <stdbool.h>
#include "ds4.h"

int main(void) {
    struct dspark_controller c;
    // Guard 2.16 for 0731 weights, k=4 draft tokens
    dspark_init(&c, 2.16f, 4);
    assert(dspark_step_should_speculate(&c) == true);

    // Scenario A: High acceptance (4 accepted tokens per step -> net gain)
    dspark_update(&c, 4);
    assert(dspark_step_should_speculate(&c) == true);

    // Scenario B: Repeated low acceptance (0 accepted tokens -> regret accumulates)
    for (int i = 0; i < 5; i++) {
        dspark_update(&c, 0);
    }
    // Controller must terminally quench after cumulative regret threshold exceeded
    assert(dspark_step_should_speculate(&c) == false);
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gcc -I. tests/test_dspark_quench.c -o /tmp/test_quench && /tmp/test_quench`
Expected: FAIL (symbols undefined).

- [ ] **Step 3: Implement `dspark_controller` logic and hook into decode loop**

Implement `dspark_init`, `dspark_step_should_speculate`, and `dspark_update` with guard constant default `2.16f` and regret limit `~8.0f` (equivalent to ~4 plain step penalty). In `ds4_cuda.cu`, integrate drafter forward and target verify pass. When `dspark_step_should_speculate` returns false, fall back to plain single-token decode.

- [ ] **Step 4: Run test to verify it passes**

Run: `gcc -I. tests/test_dspark_quench.c ds4.c -o /tmp/test_quench && /tmp/test_quench`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ds4.h ds4.c ds4_cuda.cu tests/test_dspark_quench.c
git commit -m "feat(spec): implement DSpark speculative engine with yield-quench controller"
```

---

### Task 4: Single-Session Warm Prefix Cache & Serial CUDA Graph Islands

**Files:**
- Modify: `ds4_server.c:1200-1450`
- Modify: `ds4_cuda.cu:4100-4350`
- Test: `tests/test_single_session_prefix_cache.c`

**Interfaces:**
- Consumes:
  - Model context allocation from `ds4.c`
- Produces:
  - `size_t ds4_find_matching_prefix_length(const int *old_tokens, size_t old_len, const int *new_tokens, size_t new_len)`
  - `void ds4_cuda_decode_graph_replay_or_capture(...)` for serial decode stream

- [ ] **Step 1: Write the failing test**

```c
// tests/test_single_session_prefix_cache.c
#include <assert.h>
#include <stddef.h>
#include "ds4.h"

int main(void) {
    int prev_turn[] = {1, 1024, 2048, 4096, 8192};
    int next_turn[] = {1, 1024, 2048, 4096, 8192, 9999, 1234};
    size_t shared = ds4_find_matching_prefix_length(prev_turn, 5, next_turn, 7);
    assert(shared == 5);

    int altered[] = {1, 1024, 9999, 4096};
    assert(ds4_find_matching_prefix_length(prev_turn, 5, altered, 4) == 2);
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gcc -I. tests/test_single_session_prefix_cache.c -o /tmp/test_cache && /tmp/test_cache`
Expected: FAIL (`ds4_find_matching_prefix_length` undefined).

- [ ] **Step 3: Implement prefix matching and wire KV reuse in `ds4_server.c`**

Implement `ds4_find_matching_prefix_length`. When a new request arrives on the single-session server, compare with the previous turn's token buffer. If `shared > 0`, skip re-computing keys and values for `shared` tokens and prefill only `new_len - shared` tokens.

- [ ] **Step 4: Run test to verify it passes**

Run: `gcc -I. tests/test_single_session_prefix_cache.c ds4_server.c -o /tmp/test_cache && /tmp/test_cache`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ds4_server.c ds4_cuda.cu tests/test_single_session_prefix_cache.c
git commit -m "feat(server): add single-session warm prefix caching and CUDA graph capture"
```

---

### Task 5: Agent Loop Robustness Hooks for DSH

**Files:**
- Modify: `ds4_server.c:1600-1750`
- Modify: `ds4_cli.c:300-350`
- Test: `tests/test_agent_protocol_hooks.c`

**Interfaces:**
- Produces:
  - `bool ds4_should_inject_tool_reminder(size_t conversation_tokens, bool has_tools)`
  - Server CLI flag `--reasoning-replay drop|keep` (default `keep`)

- [ ] **Step 1: Write the failing test**

```c
// tests/test_agent_protocol_hooks.c
#include <assert.h>
#include <stdbool.h>
#include "ds4.h"

int main(void) {
    // Under 30k tokens: never inject reminder
    assert(ds4_should_inject_tool_reminder(10000, true) == false);
    // Over 30k tokens without tools: do not inject reminder
    assert(ds4_should_inject_tool_reminder(35000, false) == false);
    // Over 30k tokens WITH tools armed: MUST inject reminder
    assert(ds4_should_inject_tool_reminder(35000, true) == true);
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gcc -I. tests/test_agent_protocol_hooks.c -o /tmp/test_agent && /tmp/test_agent`
Expected: FAIL (`ds4_should_inject_tool_reminder` undefined).

- [ ] **Step 3: Implement protocol reminder check and prompt filter**

Implement `ds4_should_inject_tool_reminder`. When true, append the compact tool-protocol reminder prompt before generating the assistant turn. Support `--reasoning-replay drop` to strip historical `<think>...</think>` traces from re-injected prompts.

- [ ] **Step 4: Run test to verify it passes**

Run: `gcc -I. tests/test_agent_protocol_hooks.c ds4_server.c -o /tmp/test_agent && /tmp/test_agent`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ds4_server.c ds4_cli.c tests/test_agent_protocol_hooks.c
git commit -m "feat(agent): add tool-call protocol reminder and reasoning replay controls"
```

---

### Task 6: Blackwell SM121 Kernel Exploration: Drafter L2 Pinning & Verify Forward

**Files:**
- Create: `cuda/mmq/ds4_sm121_tuning.cuh`
- Modify: `ds4_cuda.cu:4600-4750`
- Create: `scripts/bench_dgx_spark.sh`
- Test: `tests/test_sm121_l2_pinning.cu`

**Interfaces:**
- Produces:
  - `void ds4_cuda_sm121_pin_drafter_l2(const void *drafter_weights, size_t size_bytes, cudaStream_t stream)`
  - Benchmark script `scripts/bench_dgx_spark.sh` reporting tok/s across prompt depths (2k, 16k, 64k)

- [ ] **Step 1: Write test for CUDA persisting L2 access attribute**

```cuda
// tests/test_sm121_l2_pinning.cu
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include "cuda/mmq/ds4_sm121_tuning.cuh"

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.persistingL2CacheMaxSize == 0) {
        printf("Skipping L2 persist test on non-persisting device\n");
        return 0;
    }
    void *dev_ptr = nullptr;
    size_t size = 16 * 1024 * 1024; // 16 MB test span
    assert(cudaMalloc(&dev_ptr, size) == cudaSuccess);
    cudaStream_t stream;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    ds4_cuda_sm121_pin_drafter_l2(dev_ptr, size, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dev_ptr);
    cudaStreamDestroy(stream);
    return 0;
}
```

- [ ] **Step 2: Run test to verify compilation fails before implementation**

Run: `nvcc -arch=sm_121a -I. tests/test_sm121_l2_pinning.cu -o /tmp/test_l2`
Expected: FAIL (`ds4_sm121_tuning.cuh` missing).

- [ ] **Step 3: Implement `ds4_sm121_tuning.cuh` and benchmark script**

Implement `ds4_cuda_sm121_pin_drafter_l2` using `cudaStreamSetAttribute` with `cudaAccessPropertyPersisting` to instruct the GB10 memory controller to retain drafter weights in L2 during speculative steps. Create `scripts/bench_dgx_spark.sh` to measure prefill and decode across context sizes.

- [ ] **Step 4: Run test to verify it passes**

Run: `nvcc -arch=sm_121a -I. tests/test_sm121_l2_pinning.cu -o /tmp/test_l2 && /tmp/test_l2`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cuda/mmq/ds4_sm121_tuning.cuh ds4_cuda.cu scripts/bench_dgx_spark.sh tests/test_sm121_l2_pinning.cu
git commit -m "perf(sm121): add L2 cache persistence tuning for drafter and DGX Spark benchmark harness"
```
