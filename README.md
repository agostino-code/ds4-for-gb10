# ds4-for-gb10

**High-Performance, Stripped & Tuned LLM Inference Engine for NVIDIA DGX Spark (Grace-Blackwell GB10 / SM121)**

`ds4-for-gb10` is an architecture-specialized fork of [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4), streamlined and tuned for maximum single-node and multi-node throughput on the **NVIDIA DGX Spark** (GB10 SoC, Compute Capability `sm_121a`, 128 GB LPDDR5X unified memory). It synthesizes state-of-the-art community optimizations into a clean, maintainable C/CUDA codebase while stripping the legacy non-NVIDIA architectures (Metal, ROCm).

---

## 🚀 Performance on NVIDIA DGX Spark (GB10)

Measured directly on hardware with **DeepSeek-V4-Flash (0731 imatrix Q2, ~80.8 GiB)**:

| Metric | Upstream `antirez/ds4` | `ds4-for-gb10` (This Repo) | Improvement |
| :--- | :--- | :--- | :--- |
| **Prefill Throughput (Prompt)** | ~305–350 tok/s | **~755–916.8 tok/s** | **~2.5–3× Faster** (D2R MoE Tensor Cores) |
| **Decode Throughput (Generation)** | ~19.5 tok/s (plain) | **~41.8 tok/s** (DSpark) | **+114% Faster** (Lossless Speculation + Tail Fix) |
| **Stochastic Speculative Decode** | Not supported | **~24.2 tok/s** (Exact) | Exact $\min(1, p/q)$ distribution fidelity |
| **Agent Multi-Turn TTFT (Turn 2+)** | ~35 seconds | **< 0.5 seconds** | **> 30× Faster** (Warm Prefix Cache) |
| **Dual 256k Resident Sessions** | Crashed (OOM) | **Stable** (~86.5 GiB total, 34.5 GiB free) | Zero duplicate memory allocation |

---

## 🛠️ Optimizations Applied & Upstream Citations

This repository unifies and integrates insights, research papers, and kernels from across the Blackwell DGX Spark engineering community:

### 1. D2R MoE Prefill & Repack Pipeline
* **Source:** [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) / [`Entrpi/ds4`](https://github.com/Entrpi/ds4)
* **What changed:** MoE gate/up (`IQ2_XXS`) and down (`Q2_K`) expert matrices are repacked into aligned Structure-of-Arrays (SoA) format. Tensor Core GEMM kernels dequantize directly into MMA registers (*Dequant-to-Register*), completely bypassing intermediate FP16 memory allocations.
* **Why:** Boosts prefill throughput from ~300 to >915 tok/s.

### 2. DSpark Lossless Speculation with Proportional Yield-Quench & Tail Optimization
* **Source:** [`Entrpi/ds4`](https://github.com/Entrpi/ds4) & [`deepseek-ai/deepinfer`](https://github.com/deepseek-ai/deepinfer)
* **What changed:**
  * Integrated a 3-layer target-fused block drafter (~5.6 GiB support model) evaluated in parallel (5 draft tokens).
  * Added an active **Yield-Quench Controller** (`dspark_update_proportional`): regret scales proportionally with attempted block length, preventing premature quenching during bursty drafting.
  * Optimized tail scheduling for single-node DGX Spark (`ds4_dspark_scheduler_tail_min_tokens = 2`), boosting short-turn decode from ~28 tok/s to **41.8 tok/s**.
  * Pre-allocated session scratch slabs and stack-allocated Markov/confidence buffers, eliminating 2.5 MB of dynamic `xmalloc`/`free` calls per speculative proposal cycle.
* **Why:** Delivers 41.8 tok/s on code and structured reasoning without ever regressing below plain decode speeds on creative prose.

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

### 7. DRY (Don't Repeat Yourself) Sampler with Sequence Breakers
* **Source:** Adapted from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (p-e-w / pi6am algorithm).
* **What changed:** Added `ds4_apply_dry_penalty` and `ds4_session_sample_opts`: suffix-matches recent token history and penalizes repetitive n-gram continuations exponentially. Halts backward matching at newline (`\n`, `\r`) or EOS sequence breaker boundaries (`is_dry_breaker_token`).
* **Why:** Eliminates repetitive loops on 2-bit quantization without distorting vocabulary distributions or breaking markdown indentation.

### 8. Draftless n-gram Self-Speculation (`ngram-mod`)
* **Source:** Adapted from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (`common/ngram-mod`).
* **What changed:** Added a lightweight 2 MB rolling hash pool ($2^{19}$ entries) for GPU-verified self-speculation (`--spec-ngram`).
* **Why:** Enables speculative speedups on setups where memory constraints exclude the 5.6 GiB DSpark drafter (such as 152 GiB V4.1 SSD streaming).

### 9. Native Blackwell SM121 NVFP4 MMA Instruction Wrapper
* **Source:** [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) PR #22196.
* **What changed:** Added `ds4_sm121_mma_m16n8k64_nvfp4` in `cuda/mmq/ds4_sm121_tuning.cuh`, wrapping `scale_vec::4X` `ue4m3` MMA Tensor Core instructions.
* **Why:** Unlocks native 4X subblock scale vectorization on Blackwell with 0 register spills under CUDA 13.0.

### 10. $O(W)$ Open-Addressing Repetition Penalty Hash Table
* **What changed:** Replaced the quadratic $O(W^2)$ frequency scan in `ds4_apply_repetition_penalties` with a stack-allocated 512-entry hash table.
* **Why:** Eliminates host CPU sampling latency spikes during generation.

---

## 🧹 Codebase Stripping: Clean & Maintainable

Upstream `antirez/ds4` is broad and cross-platform. To reduce technical debt and maintenance burden, the obsolete non-NVIDIA backends were stripped:
* ❌ **Apple Metal Backend:** `metal/` (26 shader files), `ds4_metal.m`, and all Darwin-specific benchmarks.
* ❌ **AMD ROCm Backend:** `rocm/` (24 header files), `ds4_rocm.*`, and Strix Halo build targets.
* ✅ **Retained:** Full support for **DeepSeek-V4-Flash**, **Qwen3.8 Flash Next**, **GLM 5.3 Flash**, and **multi-node RDMA network tensor parallelism** (`ds4_tp.c`, `ds4_distributed.c`).

CUDA is the only GPU backend. The CPU build (`make cpu`) exists as a reference/debug path only.

---

## 📋 Prerequisites

This repo targets the **NVIDIA DGX Spark (GB10)**. Required to build and run:

| Requirement | Minimum |
| :--- | :--- |
| Hardware | NVIDIA DGX Spark GB10 (SM121a), 20 ARM cores (10× Cortex-X925 + 10× Cortex-A725), 128 GB unified LPDDR5X |
| OS | Linux (Ubuntu 24.04 NVIDIA kernel recommended) |
| CUDA toolkit | 13.0+ with `nvcc` and cuBLAS (installed at `/usr/local/cuda`) |
| NVIDIA driver | Loaded (`nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset` modules) |
| Disk | ~85 GiB for the Q2 model + drafter; fast NVMe SSD for KV disk cache and SSD streaming |
| Build tools | `cc` (GCC/Clang), `make`, `python3` (for scripts and tests) |

Verify the GPU before building:

```sh
# nvidia-smi may report "Failed to initialize NVML" on GB10 — a known quirk.
# Confirm the driver modules and device node instead:
lsmod | grep '^nvidia '
ls -l /dev/nvidia0
```

---

## 📦 Quick Start

### 1. Build

Compile natively for DGX Spark / GB10 (this is the default target):

```bash
make cuda-spark -j$(nproc)
```

Other targets: `make cpu` (CPU reference/debug), `make cuda-generic` (generic CUDA GPU), `make cuda CUDA_ARCH=sm_N`, `make test` (regression suite).

### 2. Download Weights

Download the DeepSeek-V4-Flash 0731 base model and matching DSpark drafter:

```bash
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
```

`ds4f-q2` is the resident target for a 128 GB machine (~81 GiB). The drafter (`~5.6 GiB`) enables DSpark speculative decoding. Run `./download_model.sh --help` for all targets (Qwen3.8 Flash Next, GLM 5.3 Flash, V4.1 Flash, PRO).

### 3. Run the Production Server (OpenAI / Anthropic APIs)

```bash
./scripts/serve_dgx_spark.sh
```

This launches the tuned server: CUDA, DSpark (~41.8 tok/s), 256K context, 4096-token prefill chunks (~700 tok/s), 64 GiB NVMe KV disk cache, and pins host threads to the Grace performance cores. It exposes OpenAI- and Anthropic-compatible endpoints at `http://127.0.0.1:8000/v1`.

Env overrides: `DS4_MODEL`, `DS4_DRAFTER`, `DS4_CTX`, `DS4_PREFILL_CHUNK`, `DS4_DCONF`, `DS4_KV_DIR`, `DS4_KV_MB`, `DS4_HOST`, `DS4_PORT`.

### 4. Run Benchmarks

```bash
./scripts/bench_dgx_spark.sh        # prefill + plain decode + DSpark decode
./scripts/bench_all_configs_matrix.sh  # exhaustive parameter sweep → CSV
```

### 5. Interactive CLI / Agent

```bash
./ds4                                  # interactive chat (REPL)
./ds4 -p "Explain mmap in C"           # one-shot prompt
./ds4-agent --cuda -m ds4flash.gguf     # tool-using agent loop
```

---

## 🧠 SM121 / GB10-Specific Guidance

### Grace CPU core pinning

The GB10 has 10 performance cores (Cortex-X925, 3.9 GHz) and 10 efficiency cores (Cortex-A725, 2.8 GHz). The tuned scripts pin CUDA host threads to the X925 cores:

```sh
taskset -c 5-9,15-19 ./ds4-server --cuda ...
```

Verify the mapping on your machine (`cat /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq`); the X925 cores report 3900000, the A725 cores 2808000.

### Blackwell runtime env vars

`serve_dgx_spark.sh` and `bench_dgx_spark.sh` export these SM121 fast-path flags:

```sh
export DS4_CUDA_FP4_INDEX=1      # FP4 compressed indexer (memory)
export DS4_CONT_CAPTURE=1        # CUDA graph capture (lower latency)
```

### Host memory tuning

Background page compaction can add 4–5 s stalls. `serve_dgx_spark.sh` sets `vm.compaction_proactiveness=0` when writable (needs root). Run `scripts/disable_earlyoom.sh` (as root) to stop `earlyoom`/`systemd-oomd` from SIGKILLing large model loads.

### Memory budget

The Q2 model (~81 GiB) + drafter (~5.6 GiB) + 256K context + runtime buffers fill the 128 GB unified pool. Only ~11 GiB typically remains free, so **do not run a second model instance concurrently** (the engine has an intentional instance lock). Lower `-c` (e.g. `131072`) or `--kv-disk-space-mb` to free RAM. Only one model can be resident at a time.

### Recommended server settings (DeepSeek-V4-Flash Q2)

| Flag | Value | Rationale |
| :--- | :--- | :--- |
| `--cuda` | required | CUDA-only build; `sm_121a` Blackwell fast-path kernels |
| `-m ds4flash.gguf` | Q2 Flash (~81 GiB) | Resident target for 128 GB |
| `--dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` | — | ~41.8 tok/s on code/structured reasoning |
| `--dspark-confidence 0.4` | calibrated default | Proportional yield-quench avoids prose regressions |
| `-c 262144` | 256K | Agentic sweet spot; compressed KV (CSA/HCA, 86.5 GiB resident) |
| `--prefill-chunk 4096` | ~700 tok/s | Balanced GB10 prefill, prevents unified memory saturation |
| `--kv-disk-space-mb 65536` | 64 GiB NVMe | Prefix reuse across restarts and slot reuse |
| `taskset -c 5-9,15-19` | perf cores | Removes host scheduling jitter |

### Sampling defaults

The engine now defaults to `temperature 0.6`, `top-p 0.95`, and `min-p 0.05`,
with a repetition penalty of `1.10` over a 128-token window (set
`--repeat-penalty 1.0` to disable). These differ from the model card's
`temperature 1.0` / `top-p 1.0` recommendation; override per request with
`--temp`, `--top-p`, `--repeat-penalty`, or the API's `temperature`,
`top_p`, `presence_penalty`, `frequency_penalty`, and `repeat_penalty`
fields.

---

## 🧭 Model Support

CUDA single-GPU (including DGX Spark) runs these models. Only one can be resident at a time; see [MODELS.md](docs/MODELS.md) for full details.

| Model | Target | Resident weights | Notes |
| :--- | :--- | ---: | :--- |
| DeepSeek-V4-Flash | `ds4f-q2` | ~81 GiB | Default; DSpark support file needed for speculation |
| Qwen3.8 Flash Next | `qwen38-q2` | 41.73 GiB | +95.37 GiB n-grams on disk; built-in MTP |
| GLM 5.3 Flash | `glm53-q2` | ~90 GiB | Built-in MTP block |
| DeepSeek-V4.1 Flash | `ds41f-q2` | 152 GiB | SSD streaming only on one Spark |

Multi-GPU CUDA and two-Spark RDMA tensor parallelism are documented in [CUDA_MULTI_GPU.md](docs/CUDA_MULTI_GPU.md) and [DISTRIBUTED.md](docs/DISTRIBUTED.md).

---

## 📚 Documentation

- [DGX Spark guide](docs/DGX_SPARK.md)
- [Models & vision](docs/MODELS.md)
- [Serving (server APIs)](docs/SERVER.md)
- [Coding agent clients](docs/CLIENTS.md)
- [Speculative decoding (DSpark/MTP)](docs/SPECULATIVE_DECODING.md)
- [SSD streaming (models larger than RAM)](docs/SSD_STREAMING.md)
- [Performance & benchmarking](docs/PERFORMANCE.md)
- [Testing & development](docs/TESTING.md)
- [GGUF conversion & quantization](gguf-tools/README.md)
- [Directional steering](dir-steering/README.md)

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
