# ds4-for-gb10

**High-Performance, Stripped & Tuned LLM Inference Engine for NVIDIA DGX Spark (Grace-Blackwell GB10 / SM121)**

`ds4-for-gb10` is an architecture-specialized fork of [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4), specifically streamlined and tuned for maximum single-node and multi-node throughput on the **NVIDIA DGX Spark** (GB10 SoC, Compute Capability `sm_121a`, 128 GB LPDDR5X unified memory).

It synthesizes state-of-the-art community optimizations into a clean, maintainable C/CUDA codebase while stripping legacy non-NVIDIA architectures (Metal, ROCm).

---

## 🚀 Performance on NVIDIA DGX Spark (GB10)

Measured directly on hardware with **DeepSeek-V4-Flash (0731 imatrix Q2, ~80.8 GiB)**:

| Metric | Upstream `antirez/ds4` | `ds4-for-gb10` (This Repo) | Improvement |
| :--- | :--- | :--- | :--- |
| **Prefill Throughput (Prompt)** | ~305–350 tok/s | **~916.8 tok/s** | **~3× Faster** (D2R MoE Tensor Cores) |
| **Decode Throughput (Generation)** | ~19.5 tok/s (plain) | **~36.7 tok/s** (DSpark) | **+85% Faster** (Lossless Speculation) |
| **Agent Multi-Turn TTFT (Turn 2+)** | ~35 seconds | **< 0.5 seconds** | **> 30× Faster** (Warm Prefix Cache) |
| **Dual 256k Resident Sessions** | Crashed (OOM) | **Stable** (~81 GiB base, 38 GiB free) | Zero duplicate memory allocation |

---

## 🛠️ Optimizations Applied & Upstream Citations

This repository unifies and integrates insights, research papers, and kernels from across the Blackwell DGX Spark engineering community:

### 1. D2R MoE Prefill & Repack Pipeline
* **Source:** [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) / [`Entrpi/ds4`](https://github.com/Entrpi/ds4)
* **What changed:** MoE gate/up (`IQ2_XXS`) and down (`Q2_K`) expert matrices are repacked into aligned Structure-of-Arrays (SoA) format. Tensor Core GEMM kernels dequantize directly into MMA registers (*Dequant-to-Register*), completely bypassing intermediate FP16 memory allocations.
* **Why:** Boosts prefill throughput from ~300 to >915 tok/s.

### 2. DSpark Lossless Speculation with Calibrated Yield-Quench
* **Source:** [`Entrpi/ds4`](https://github.com/Entrpi/ds4) & [`deepseek-ai/deepinfer`](https://github.com/deepseek-ai/deepinfer)
* **What changed:** Integrated a 3-layer target-fused block drafter (~5.6 GiB support model) evaluated in parallel (5 draft tokens). Added a terminal **Yield-Quench Controller** calibrated with a guard of 2.16 for the 0731 weights: if draft acceptance drops below break-even on unpredictable prose, speculation is terminally quenched for that request.
* **Why:** Delivers 36.7 tok/s on code and structured reasoning without ever regressing below plain decode speeds on creative prose.

### 3. Hardware Persisting L2 Cache on Blackwell GB10
* **Source:** NVIDIA Blackwell Architecture Manual & [`cuda/mmq/ds4_sm121_tuning.cuh`](cuda/mmq/ds4_sm121_tuning.cuh)
* **What changed:** Explicitly unlocks and sets hardware L2 cache reservation via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size)` and binds drafter weights to `cudaAccessPropertyPersisting`.
* **Why:** The GB10 L2 cache default limit is 0; activating it pins the hot drafter projections in on-chip L2, eliminating LPDDR5X DRAM traffic.

### 4. DRAM Channel Anti-Aliasing
* **Source:** Blake Ledden ([`blake-snc/sm121-kernels`](https://github.com/blake-snc/sm121-kernels))
* **What changed:** Added `ds4_cuda_sm121_pad_vt_stride()`: power-of-two row strides ($S=2048, 4096, 8192$) are padded by 128 bytes to prevent memory channel aliasing on LPDDR5X unified memory.
* **Why:** Unlocks up to 109.9 TFLOPS in Flash Attention without memory bus stalls.

### 5. Single-Session Warm Prefix Cache & Agent Robustness Hooks
* **Source:** Developed for DeepSeek Harness (DSH), Claude Code, and agentic loops.
* **What changed:** 
  * `ds4_find_matching_prefix_length()` detects existing token histories and skips re-prefill.
  * Injects a compact tool-protocol reminder for tools-armed conversations exceeding 30,000 tokens.
* **Why:** Drops secondary turn TTFT from 35s to <0.5s and eliminates "tool slips" where 2-bit models drift into narrative prose instead of invoking tools.

### 6. Host Tuning & CPU Core Pinning
* **Source:** Christopher Owen ([`christopherowen/spark-ds41f`](https://github.com/christopherowen/spark-ds41f)) & [`myllmbox/qwen38-flash-next-recipe`](https://github.com/myllmbox/qwen38-flash-next-recipe)
* **What changed:** Binds CUDA host execution threads to Grace Performance Cores (`taskset -c 5-9,15-19`) and documents `vm.compaction_proactiveness=0`.
* **Why:** Removes host scheduling jitter and eliminates 4–5 second background page compaction stalls.

---

## 🧹 Codebase Stripping: Clean & Maintainable

Upstream `antirez/ds4` is broad and cross-platform. To reduce technical debt and maintenance burden, **59 obsolete files (>126,000 lines of dead code)** were stripped:
* ❌ **Apple Metal Backend:** `metal/` (26 shader files), `ds4_metal.m`, and all Darwin-specific benchmarks.
* ❌ **AMD ROCm Backend:** `rocm/` (24 header files), `ds4_rocm.*`, and Strix Halo build targets.
* ✅ **Retained:** Full support for **DeepSeek-V4-Flash**, **Qwen 3.8 Flash Next**, **GLM 5.3 Flash**, and **multi-node RDMA network tensor parallelism** (`ds4_tp.c`, `ds4_distributed.c`).

---

## 📦 Quick Start

### Build
Compile natively for DGX Spark / GB10:
```bash
make cuda-spark -j$(nproc)
```

### Download Weights
Download the DeepSeek-V4-Flash 0731 base model and matching DSpark drafter:
```bash
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
```

### Run Benchmarks
Run the automated turnkey benchmark sweep:
```bash
./scripts/bench_dgx_spark.sh
```

### Start Production Server (for DeepSeek Harness / DSH)
Launch the server with 2 resident sessions of 256k context, DSpark, and NVMe KV cache offload:
```bash
./scripts/serve_dgx_spark.sh
```
The server exposes OpenAI and Anthropic compatible endpoints at `http://127.0.0.1:8000/v1`.

---

## 📜 Citations & Acknowledgments

- **Salvatore Sanfilippo ([antirez](https://github.com/antirez))**: Author of the upstream [`ds4`](https://github.com/antirez/ds4) inference engine.
- **Entrpi ([Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark))**: D2R prefill kernels, DSpark integration, and batched serving foundation.
- **Blake Ledden ([blake-snc/sm121-kernels](https://github.com/blake-snc/sm121-kernels))**: GB10 PTX kernel insights and DRAM channel anti-aliasing techniques.
- **Christopher Owen ([christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f))**: DGX Spark performance core pinning and multi-node cluster recipes.
- **Shresth Mishra (COLM 2026)**: *Deployable W4A4 LLM Inference on Consumer Blackwell (SM121)* for INT4 Tensor Core instruction designs.
- **MyLLMBox ([myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe))**: DGX Spark memory compaction tuning.
- **NVIDIA ([NVIDIA/spark-local-inference](https://github.com/NVIDIA/spark-local-inference))**: Local inference architectural blueprints (Project Flare).

## License
MIT.
