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

#ifdef __cplusplus
}
#endif
