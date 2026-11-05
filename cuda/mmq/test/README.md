# MMQ & D2R Micro-Benchmark Prototypes

This directory contains standalone CUDA prototypes and micro-benchmarks developed during the bring-up of the **D2R (Dequant-to-Register)** and **SoA Tile Repack** pipelines for NVIDIA DGX Spark (GB10 / SM121).

## Scope & Isolation
* **Standalone execution:** None of the files in this directory are linked into production binaries (`ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`).
* **Production Code:** The production implementations reside in:
  * `cuda/mmq/ds4_mmq_d2r.cu`: Core D2R GEMM / GEMV execution kernels.
  * `cuda/mmq/ds4_repack.cu`: Aligned SoA repack builders for IQ2_XXS, Q2_K, and Q8_0.
  * `cuda/mmq/ds4_sm121_tuning.cuh`: Blackwell SM121 hardware constants, launch bounds, and MMA wrappers.

## Prototype Inventory
- `proto_m2_*`: Micro-benchmarks investigating warp decomposition for small batch sizes ($M=2..6$) in speculative decode.
- `proto_gemm_*`: Register allocation experiments comparing shared memory staging against direct register dequantization.
- `proto_indexer_score.cu`: Prototype low-precision Tensor Core scoring for the DeepSeek-V4 attention indexer.
- `test_mmq_parity.cu`: Offline bit-exact numerical parity validation against cuBLAS.
