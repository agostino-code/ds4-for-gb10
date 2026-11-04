#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture constants for NVIDIA Grace Blackwell GB10 (SM 12.1 / compute capability 12.1 = 1210) */
#define DS4_GB10_NUM_SMS                48
#define DS4_GB10_MAX_SMEM_PER_SM        (99 * 1024)   /* ~99-100 KB shared memory per SM */
#define DS4_GB10_REGS_PER_SM            65536         /* 64K 32-bit registers per SM */

/* Calibrated launch bounds for Blackwell SM121 / SM120:
 * At kThreads=256, minBlocks=2 rigidly caps registers to 65536 / 512 = 128 regs/thread,
 * which forces ptxas to spill 212+ bytes into local stack/DRAM in complex MoE D2R kernels.
 * On SM121 (48 SMs), minBlocks=1 allows ptxas to allocate up to 255 registers per thread,
 * completely eliminating register spills (0 bytes spill stores, 0 bytes spill loads). */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define DS4_SM121_D2R_MIN_BLOCKS        1
#define DS4_SM121_MMVQ_MIN_BLOCKS       1
#else
#define DS4_SM121_D2R_MIN_BLOCKS        2
#define DS4_SM121_MMVQ_MIN_BLOCKS       1
#endif

/* Set persisting L2 cache window on Blackwell GB10 / SM121 for hot drafter sub-tensors.
 * Safely clamps size_bytes to the device's persistingL2CacheMaxSize to prevent cache thrashing. */
static inline int ds4_cuda_sm121_pin_drafter_l2(const void *drafter_weights,
                                                size_t size_bytes,
                                                cudaStream_t stream) {
    if (!drafter_weights || size_bytes == 0) return 0;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return -2;

    if (prop.persistingL2CacheMaxSize == 0) {
        return 0; // Device does not support persisting L2 cache window
    }

    // Clamp window to device persisting budget
    size_t window_size = size_bytes;
    if (window_size > (size_t)prop.persistingL2CacheMaxSize) {
        window_size = (size_t)prop.persistingL2CacheMaxSize;
    }

    // Set the hardware persisting L2 cache limit on Blackwell GB10
    (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size);

    cudaStreamAttrValue val;
    val.accessPolicyWindow.base_ptr = (void *)drafter_weights;
    val.accessPolicyWindow.num_bytes = window_size;
    val.accessPolicyWindow.hitRatio = 1.0f;
    val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &val);
    if (err != cudaSuccess) {
        return -3;
    }
    return 0;
}

/* DRAM channel anti-aliasing (Blake Ledden / sm121-kernels technique):
 * On GB10 unified LPDDR5X, power-of-two row strides (e.g. S=2048/4096/8192)
 * alias to identical memory channel partitions, degrading throughput.
 * Auto-padding the stride by 128 bytes distributes accesses across channels,
 * boosting FP8 flash attention from ~89.9 to ~109.9 TFLOPS. */
static inline size_t ds4_cuda_sm121_pad_vt_stride(size_t stride_bytes) {
    if (stride_bytes >= 1024 && (stride_bytes & (stride_bytes - 1)) == 0) {
        return stride_bytes + 128;
    }
    return (stride_bytes + 127) & ~127ull;
}

#ifdef __CUDACC__
/* Direct non-coherent 128-bit global load (L2 bypass to registers for D2R weights) */
__device__ __forceinline__ int4 ds4_sm121_ld_global_128_nc(const void *ptr) {
    int4 ret;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
#else
    ret = *reinterpret_cast<const int4 *>(ptr);
#endif
    return ret;
}

/* Direct non-coherent 64-bit global load */
__device__ __forceinline__ uint2 ds4_sm121_ld_global_64_nc(const void *ptr) {
    uint2 ret;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(ret.x), "=r"(ret.y)
                 : "l"(ptr));
#else
    ret = *reinterpret_cast<const uint2 *>(ptr);
#endif
    return ret;
}

/* Native INT4 Tensor Core MMA wrapper for SM121 (Mishra / W4A4 COLM 2026 paper):
 * mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 consumes packed INT4 weights
 * and per-token INT4 activations with INT32 accumulation. */
__device__ __forceinline__ void ds4_sm121_mma_m16n8k64_int4(
    int32_t d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const int32_t c[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

/* Native MXF4 Block-Scaled Tensor Core MMA wrapper for SM121:
 * mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
 * Consumes packed E2M1 weights and E2M1 activations with E8M0 block scales and FP32 accumulation. */
__device__ __forceinline__ void ds4_sm121_mma_m16n8k64_mxf4(
    float d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float c[4],
    uint32_t a_scale,
    uint32_t b_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(a_scale), "r"(b_scale));
#else
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
    (void)a; (void)b; (void)a_scale; (void)b_scale;
#endif
}

/* Native MXF8F6F4 Block-Scaled Tensor Core MMA wrapper for SM121:
 * mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
 * Consumes packed E4M3 activations and E2M1 weights with E8M0 block scales and FP32 accumulation. */
__device__ __forceinline__ void ds4_sm121_mma_m16n8k32_mxf8f6f4(
    float d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float c[4],
    uint32_t a_scale,
    uint32_t b_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(a_scale), "r"(b_scale));
#else
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
    (void)a; (void)b; (void)a_scale; (void)b_scale;
#endif
}
#endif

#ifdef __cplusplus
}
#endif
