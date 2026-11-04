# Plan: DGX Spark Stripped Architecture (CUDA SM121 Focused)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an ultra-lean, maintainable version of `ds4` stripped of legacy architectures (Metal for Apple Silicon, ROCm for AMD, multi-GPU network TP over RDMA, CPU fallbacks), while retaining native multi-model support (DeepSeek-V4-Flash, Qwen 3.8 Flash Next, GLM 5.3) fully optimized for NVIDIA DGX Spark (GB10 / SM121).

**Architecture:** Purge obsolete backend directories (`metal/`, `rocm/`, and unused platform headers), clean `Makefile` to target CUDA Blackwell directly, and eliminate multi-thousand lines of dead `#ifdef` branches. The runtime is focused on single-node DGX Spark unified memory execution.

**Tech Stack:** C99, CUDA C++ (CUDA 13.x, `sm_121a`), cuBLAS/cuBLASLt.

**Spec:** [`docs/superpowers/specs/2026-11-04-dgx-spark-fastpath-design.md`](../specs/2026-11-04-dgx-spark-fastpath-design.md)

## Global Constraints

- Target hardware: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory).
- Models supported: DeepSeek-V4-Flash, Qwen 3.8 Flash Next, GLM 5.3 Flash.
- Backends retained: CUDA (native SM121).
- Backends removed: Apple Metal (`.metal`, `.m`), AMD ROCm (`rocm/*.cuh`, `ds4_rocm.*`), multi-node network TP (`ds4_tp.c`, `ds4_distributed.c` remote RDMA links).
- Resulting binary must compile cleanly with `make cuda-spark -j$(nproc)` without missing symbol errors.

## Review Focus

1. **Clean Model Decoupling:** Removing ROCm/Metal headers must not break Qwen 3.8 or GLM 5.3 model definitions, tokenizers, or CUDA kernels.
2. **Zero Dead Targets:** Targets like `make strix-halo`, `make test-rocm`, and `METAL_LDLIBS` must be pruned from the `Makefile` without impacting `make cuda-spark` and `make clean`.
3. **Compile Time & Binary Size:** The build process must run noticeably faster (~40% less compile overhead) and produce exact binary artifacts (`ds4`, `ds4-server`, `ds4-bench`).

---

### Task 1: Clean Platform Subsystems (Purge Metal & ROCm trees)

**Files:**
- Remove: `metal/` (entire directory: 26 files)
- Remove: `rocm/` (entire directory: 24 files)
- Remove: `ds4_metal.m`, `ds4_metal_device.h`
- Remove: `ds4_rocm.cu`, `ds4_rocm.h`, `ds4_rocm_compat.cu`, `ds4_rocm_memory.h`, `ds4_rocm_unavailable.cu`
- Remove: `cuda/mmq/vendors/hip.h`
- Modify: `Makefile` (remove Metal/ROCm targets and compiler variables)

- [ ] **Step 1: Check git status and create branch**

Run: `git checkout -b chore/spark-stripped-clean`
Expected: Switched to branch.

- [ ] **Step 2: Remove obsolete backend directories and files**

Run: `git rm -r metal/ rocm/ ds4_metal.m ds4_metal_device.h ds4_rocm* cuda/mmq/vendors/hip.h`
Expected: Files staged for deletion.

- [ ] **Step 3: Clean `Makefile` to prune Darwin/Metal and ROCm rules**

Remove `METAL_SRCS`, `ROCM_SRCS`, `HIPCC`, `ROCM_ARCH`, `strix-halo`, and `test-rocm` rules from `Makefile`.

- [ ] **Step 4: Verify test build**

Run: `make cuda-spark -j$(nproc)`
Expected: Build succeeds without missing backend dependencies.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "chore: remove Metal and ROCm backends to streamline DGX Spark codebase"
```

---

### Task 2: Streamline Multi-GPU Network TP (Focus on Single-Node UMA)

**Files:**
- Modify: `ds4_tp.c` / `ds4_tp.h` (stub or prune remote RDMA/TCP socket logic while keeping in-node tensor splits)
- Modify: `ds4_distributed.c` / `ds4_distributed.h`
- Modify: `Makefile`

- [ ] **Step 1: Strip unused network-TP tests and RDMA coordinator hooks**

Prune `tests/test_tp_rdma.c`, `tests/test_tp_tcp.c`, `tests/test_tp_link.c`.

- [ ] **Step 2: Verify `ds4-server` and `ds4` build cleanly**

Run: `make cuda-spark -j$(nproc)`
Expected: Compilation produces `ds4`, `ds4-server`, and `ds4-bench`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: streamline single-node execution and prune remote RDMA network dependencies"
```

---

### Task 3: Regression Verification & Performance Re-validation

**Files:**
- Test: `scripts/bench_dgx_spark.sh`
- Test: `tests/test_model_routing.c`

- [ ] **Step 1: Run model routing and fast-path tests**

Run: `gcc -I. tests/test_model_routing.c ds4_cuda_routing.c -o /tmp/t1 && /tmp/t1`
Expected: PASS.

- [ ] **Step 2: Run benchmark suite on DGX Spark**

Run: `./scripts/bench_dgx_spark.sh`
Expected: Prefill throughput ~900+ tok/s and DSpark decode ~36 tok/s verified on stripped tree.

- [ ] **Step 3: Merge back to main**

Run: `git checkout main && git merge chore/spark-stripped-clean`
Expected: Fast-forward clean merge.
