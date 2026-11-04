#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
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
#endif

#ifdef __cplusplus
}
#endif
