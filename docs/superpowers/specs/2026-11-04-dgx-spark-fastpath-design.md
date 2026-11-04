# DGX Spark (SM121) Fast-Path Design: Entrpi Optimizations & Multi-Model Architecture

**Author:** DeepSeek Harness & User  
**Date:** 2026-11-04  
**Status:** Proposed  
**Target:** `antirez/ds4` on NVIDIA DGX Spark (GB10 / SM121)

---

## 1. Intent and Context

The objective is to achieve maximum single-session inference and agent loop speed on an NVIDIA DGX Spark (GB10, SM121, 128 GB unified memory) using DeepSeek-V4-Flash, while maintaining upstream `antirez/ds4` support for additional architectures and models (Qwen 3.8 Flash Next, GLM 5.3, SSD streaming).

Upstream `antirez/ds4` provides wide model support and architectural experiments, but only partially merged early commits from `Entrpi/ds4`. It lacks the complete D2R (*dequant-to-register*) MoE prefill pipeline, the DSpark speculative decoding engine with calibrated yield-quench, and single-session warm prefix caching. Conversely, the `Entrpi/ds4` fork stripped out non-DeepSeek models to focus strictly on serving Blackwell CUDA.

This project bridges that gap by implementing a dedicated, modular **"DGX Spark Fast-Path"** inside the upstream codebase, activated when compiling for `sm_121` (`make cuda-spark`), and exploring next-generation SM121-specific kernel enhancements.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **Prefill Throughput:** Achieve 2.4×–3.3× prefill speedup on DGX Spark (~950–1,050 tok/s across 2k–64k context) by activating the full D2R SoA MoE pipeline.
2. **Decode Speedup via Speculation:** Deliver 28–35 tok/s (1.3×–1.7× plain decode) on code and structured reasoning tasks using the DSpark 3-layer block drafter with calibrated yield-quench (break-even ~2.16 on 0731 weights).
3. **Low-Latency Agent Loops:** Maintain single-session warm prefix cache for multi-turn DSH agent interactions, reducing secondary turn TTFT from tens of seconds to <0.7s.
4. **Preserve Multi-Model Support:** Keep existing upstream dispatch and pipelines for Qwen 3.8 Flash Next, GLM 5.3, and SSD streaming completely functional when those models are requested.
5. **Explore Blackwell SM121 Optimizations:** Investigate and prototype SM121-specific speedups:
   - L2 cache residency hints for the compact ~6.5 GiB drafter model.
   - Fused multi-token verify forward to reduce verify overhead from ~2.05 to <1.75 plain decode steps.

### Non-Goals
- Multi-user continuous batching scheduler with high-concurrency bank eviction (priority is minimum latency for a single active session).
- Porting D2R or DSpark to ROCm, Metal, or CPU backends.
- Rewriting Qwen or GLM kernels in the initial phase (they continue running their tested upstream paths).

### Anti-Goals
- **No speculative latency regressions:** Speculation must not degrade throughput on unpredictable prose. The yield-quench controller must shut down speculation if realized draft acceptance fails to cover verify costs.
- **No breaking changes to generic builds:** Standard `make cuda`, `make cpu`, and existing tests must continue to build and pass cleanly.

---

## 3. Constraints

- **Hardware:** NVIDIA DGX Spark with Grace-Blackwell GB10, Compute Capability 12.1 (`sm_121a`), 48 SMs, 100 KB shared memory per SM, 64K 32-bit registers per SM.
- **Memory Ceiling:** 128 GB physical unified LPDDR5X (~119 GiB usable by OS/processes). Measured memory bandwidth: ~215–227 GB/s read.
- **Model Footprint:** DeepSeek-V4-Flash 0731 Q2 imatrix (~81 GiB) + DSpark drafter Q2K (~6.5 GiB) + KV cache buffer must comfortably fit under 115 GiB total resident memory.
- **Toolchain:** CUDA Toolkit 13.x with `nvcc` targeting `sm_121a`.

---

## 4. Approaches Considered

### Approach 1: Modular Upstream Fast-Path Integration (*Selected*)
Maintain `antirez/ds4` main as the canonical tree. Integrate Entrpi's D2R repack and SoA pipeline, DSpark speculation engine, and single-session warm prefix cache as a dedicated fast-path enabled under `CUDA_ARCH=sm_121` (`make cuda-spark`).
- *Pros:* Retains 100% of upstream multi-model features (Qwen 3.8, GLM 5.3, SSD streaming), isolates SM121-specific optimizations cleanly, allows incremental verification.
- *Cons:* Requires disciplined surgical patching in `ds4_cuda.cu` and `cuda/mmq/ds4_mmq.cu`.

### Approach 2: Rebase Entrpi Fork and Re-import Upstream Models
Start from `Entrpi/ds4` v0.6.5 and re-introduce the Qwen 3.8 and GLM 5.3 model definitions and kernels that Entrpi had stripped out.
- *Pros:* Entrpi's serving stack and Memory Governor are already pre-assembled.
- *Cons:* High maintenance overhead; Entrpi aggressively gutted multi-model abstractions, so re-introducing Qwen/GLM would require adapting them to Entrpi's custom server architecture and continuous batch buffers.

*Decision:* Proceed with **Approach 1**.

---

## 5. Detailed Architecture and Subsystems

### Component 1: D2R MoE Prefill & Repack Pipeline
- **Mechanism:** In `cuda/mmq/`:
  - Upstream already contains the raw kernel files (`ds4_mmq_d2r.cu`, `ds4_repack.cu`), but lacks the active wiring for aligned-SoA artifacts.
  - Wire `ds4_repack` to transform the IQ2_XXS and Q2_K expert weights into aligned Structure-of-Arrays (SoA) layout during model load (or zero-copy attach via weight server).
  - Activate `ds4_mmq_q2_K_moe_d2r_launch` and `ds4_mmq_iq2_xxs_moe_d2r_fused_launch` inside `ds4_cuda.cu` for prefill ($n_{tokens} \ge 2$).
- **Fallback:** Guarded by `DS4_MMQ_D2R=1` (default on for SM121, disableable via env).

### Component 2: DSpark Speculative Decoding & Yield-Quench
- **Architecture:**
  - Load the secondary lightweight drafter GGUF (`DSpark-drafter-Q2K-Q8-0731.gguf`, ~6.5 GiB).
  - Execute 3-layer target-fused drafting: propose $K$ tokens (typically $K=4$ or $5$).
  - Target model executes a single verify step evaluating the $1$ committed + $K$ draft tokens in parallel.
  - Implement the **terminal yield-quench controller**: track cumulative regret `guard - tokens_per_step` per request. If regret exceeds threshold (~4 plain steps), quench speculation for the rest of the generation and revert to plain decode.
  - Guard calibrated to 2.16 for DeepSeek-V4-Flash 0731 weights.

### Component 3: Single-Session Warm Prefix Cache & CUDA Graph Capture
- **Context Management:**
  - For single-session DSH execution, allocate a dedicated virtual KV cache with demand mapping, defaulting to a context limit of 512k (`-c 524288`).
  - Model architecture detection uses the existing GGUF metadata (`general.architecture`); when architecture is `deepseek4`/`deepseek2` and CUDA compute capability is 12.1 (`sm_121a`), the Spark Fast-Path is automatically engaged.
  - For Qwen 3.8 and GLM 5.3, the server transparently routes to the tested native upstream execution paths.
  - Repack of weights into aligned SoA format is performed in-memory during initial model attach (~1.5s on Grace-Blackwell unified memory), avoiding mandatory pre-conversion disk artifacts.
  - On successive turns with matching prompt prefixes (tool calls, conversation history), preserve existing KV tokens without re-prefill.
  - Extend CUDA Graph capture across the full serial decode sequence: capture static graph islands per layer to eliminate host driver launch overhead.

### Component 4: Agent Robustness Hooks (for DSH)
- Port the v0.6.5 tool-protocol reminder logic: for tools-armed requests with conversation depth > 30k tokens, append the compact protocol prompt to prevent 2-bit quantization drift towards plain prose.
- Provide flag `--reasoning-replay drop` to strip historical reasoning traces and prevent context bloat during multi-step reasoning.

### Component 5: SM121 Kernel Research & Next-Level Optimizations
- **L2 Cache Pinning for Drafter:**
  - Utilize `cudaStreamSetAttribute` with `cudaAccessPropertyPersisting` to pin drafter model weights into the GB10 L2 cache during active decode sessions.
- **Fused Verify Forward:**
  - Profile and fuse the multi-token verify forward GEMM in `ds4_cuda.cu` to minimize kernel launch barriers on SM121.

---

## 6. Verification and Acceptance Criteria

1. **Build Verification:**
   - `make cuda-spark` builds `ds4`, `ds4-server`, and `ds4-bench` cleanly with `sm_121a` SASS.
   - `make cuda` and `make cpu` continue to build without regressions.
2. **Model Parity:**
   - Smoke test against DeepSeek-V4-Flash 0731 returns exact expected answers (e.g., "Paris" smoke test).
   - Qwen 3.8 Flash Next loads and runs inference using its existing native path.
3. **Performance Metrics on GB10:**
   - DeepSeek-V4-Flash Prefill ($N=2048$): $\ge 900\text{ tok/s}$.
   - Plain Decode ($N=1$): $\ge 18.5\text{ tok/s}$.
   - DSpark Speculative Decode (Code/Math): $\ge 28.0\text{ tok/s}$.
   - Multi-turn Turn-2 TTFT on 32k prompt prefix: $< 1.0\text{ s}$.

---

## 7. What is Left to the Builder

- Exact layout of data structures in `ds4_cuda.cu` for managing the dual model handles (base + drafter).
- Fine-tuning the dynamic scratchpad buffer sizes in `ds4_mmq_d2r.cu` for GB10's 100 KB shared memory limit.
- Implementation of the CLI and server flags (`--dspark`, `--mtp-model`, `--reasoning-replay`).
