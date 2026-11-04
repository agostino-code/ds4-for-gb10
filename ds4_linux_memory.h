#ifndef DS4_LINUX_MEMORY_H
#define DS4_LINUX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

/* Linux Meminfo structure for tracking page cache pressure and anti-thrashing */
typedef struct ds4_linux_meminfo {
    uint64_t mem_total;
    uint64_t mem_available;
    uint64_t mem_free;
    uint64_t buffers;
    uint64_t cached;
    uint64_t active_file;
    uint64_t inactive_file;
    uint64_t cma_free;
} ds4_linux_meminfo;

/* DGX Spark GB10 Unified Memory Constants (128 GB physical, ~119 GiB usable) */
#define DS4_SPARK_GB10_PHYSICAL_RAM_BYTES  (128ull * 1024ull * 1024ull * 1024ull)
#define DS4_SPARK_GB10_USABLE_RAM_BYTES    (119ull * 1024ull * 1024ull * 1024ull)
#define DS4_SPARK_GB10_MIN_HEADROOM_BYTES  (8ull * 1024ull * 1024ull * 1024ull)   /* 8 GiB minimum OS/transient reserve */
#define DS4_SPARK_GB10_MAX_SAFE_FOOTPRINT  (111ull * 1024ull * 1024ull * 1024ull)  /* 111 GiB safe allocation ceiling */

/* Parse full meminfo snapshot from /proc/meminfo */
static inline bool ds4_linux_meminfo_parse(FILE *fp, ds4_linux_meminfo *info) {
    if (!fp || !info) return false;
    memset(info, 0, sizeof(*info));
    char line[256];
    bool found_avail = false;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib = 0;
        if (sscanf(line, "MemTotal: %llu kB", &kib) == 1) {
            info->mem_total = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            info->mem_available = (uint64_t)kib * 1024u;
            found_avail = true;
        } else if (sscanf(line, "MemFree: %llu kB", &kib) == 1) {
            info->mem_free = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "Buffers: %llu kB", &kib) == 1) {
            info->buffers = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "Cached: %llu kB", &kib) == 1) {
            info->cached = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "Active(file): %llu kB", &kib) == 1) {
            info->active_file = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "Inactive(file): %llu kB", &kib) == 1) {
            info->inactive_file = (uint64_t)kib * 1024u;
        } else if (sscanf(line, "CmaFree: %llu kB", &kib) == 1) {
            info->cma_free = (uint64_t)kib * 1024u;
        }
    }
    return found_avail && !ferror(fp);
}

static inline bool ds4_linux_meminfo_query(ds4_linux_meminfo *info) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return false;
    const bool ok = ds4_linux_meminfo_parse(fp, info);
    fclose(fp);
    return ok;
}

/* CMA pages can back movable allocations, but not ordinary pinned GPU pages. */
static inline bool ds4_linux_nonmovable_memory_parse(FILE *fp, uint64_t *bytes) {
    char line[256];
    uint64_t available = 0, cma = 0;
    bool found = false;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib;
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            available = (uint64_t)kib * 1024u;
            found = true;
        } else if (sscanf(line, "CmaFree: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            cma = (uint64_t)kib * 1024u;
        }
    }
    if (!found || ferror(fp)) return false;
    *bytes = available > cma ? available - cma : 0;
    return true;
}

static inline bool ds4_linux_nonmovable_memory(uint64_t *bytes) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return false;
    const bool ok = ds4_linux_nonmovable_memory_parse(fp, bytes);
    fclose(fp);
    return ok;
}

/* Anti-thrashing: evict page cache entries on Linux to prevent kernel page cache from filling 128 GB UMA */
static inline int ds4_linux_fadvise_dontneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || bytes == 0) return 0;
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
    return posix_fadvise(fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)fd; (void)offset; (void)bytes;
    return 0;
#endif
}

/* 4096-byte page-aligned allocation for Zero-Copy DMA and direct I/O (O_DIRECT) */
static inline void *ds4_linux_aligned_alloc(size_t size) {
    if (size == 0) return NULL;
    void *ptr = NULL;
    if (posix_memalign(&ptr, 4096, size) != 0) return NULL;
    return ptr;
}

static inline void ds4_linux_aligned_free(void *ptr) {
    if (ptr) free(ptr);
}

/* Anti-thrashing check for GB10 Unified Memory:
 * Returns true if requested_bytes can be safely resident while preserving the minimum OS headroom. */
static inline bool ds4_linux_spark_anti_thrashing_check(uint64_t requested_bytes, uint64_t *safe_budget) {
    ds4_linux_meminfo info;
    uint64_t avail = 0;
    if (ds4_linux_meminfo_query(&info)) {
        avail = info.mem_available;
    } else {
        avail = DS4_SPARK_GB10_USABLE_RAM_BYTES;
    }
    uint64_t limit = (avail > DS4_SPARK_GB10_MIN_HEADROOM_BYTES)
        ? (avail - DS4_SPARK_GB10_MIN_HEADROOM_BYTES)
        : (avail / 2);
    if (safe_budget) *safe_budget = limit;
    return requested_bytes <= limit;
}

#endif
