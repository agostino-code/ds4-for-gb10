#include <assert.h>
#include <string.h>
#include "ds4_linux_memory.h"

static void memory_case(const char *text, bool valid, uint64_t expected) {
    FILE *fp = tmpfile();
    assert(fp);
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    rewind(fp);
    uint64_t bytes = 123;
    assert(ds4_linux_nonmovable_memory_parse(fp, &bytes) == valid);
    if (valid) assert(bytes == expected);
    fclose(fp);
}

int main(void) {
    memory_case("MemAvailable: 100 kB\n", true, 100 * 1024);
    memory_case("MemAvailable: 100 kB\nCmaFree: 30 kB\n", true, 70 * 1024);
    memory_case("CmaTotal: 0 kB\nCmaFree: 30 kB\nMemAvailable: 100 kB\n",
                true, 70 * 1024);
    memory_case("MemAvailable: 10 kB\nCmaFree: 30 kB\n", true, 0);
    memory_case("MemAvailable: 0 kB\n", true, 0);
    memory_case("CmaFree: 30 kB\n", false, 0);
    memory_case("MemAvailable: 18446744073709551615 kB\n", false, 0);

    // Meminfo parsing test
    FILE *mfp = tmpfile();
    assert(mfp);
    const char *minfo_text = "MemTotal: 131072000 kB\nMemAvailable: 125000000 kB\nMemFree: 10000000 kB\nBuffers: 500000 kB\nCached: 20000000 kB\nActive(file): 15000000 kB\nInactive(file): 5000000 kB\nCmaFree: 1000000 kB\n";
    assert(fwrite(minfo_text, 1, strlen(minfo_text), mfp) == strlen(minfo_text));
    rewind(mfp);
    ds4_linux_meminfo minfo;
    assert(ds4_linux_meminfo_parse(mfp, &minfo) == true);
    assert(minfo.mem_total == 131072000ull * 1024u);
    assert(minfo.mem_available == 125000000ull * 1024u);
    assert(minfo.active_file == 15000000ull * 1024u);
    fclose(mfp);

    // 4096-byte aligned allocation for Zero-Copy / O_DIRECT
    void *ptr = ds4_linux_aligned_alloc(16384);
    assert(ptr != NULL);
    assert(((uintptr_t)ptr & 4095u) == 0);
    ds4_linux_aligned_free(ptr);

    // Anti-thrashing check
    uint64_t safe_budget = 0;
    assert(ds4_linux_spark_anti_thrashing_check(10ull * 1024ull * 1024ull * 1024ull, &safe_budget) == true);
    assert(safe_budget > 0);

    puts("Linux nonmovable memory: PASS");
    return 0;
}
