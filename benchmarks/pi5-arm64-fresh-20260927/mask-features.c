// Test-only feature masking. This is not an emulator or unsupported-hardware proof.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/auxv.h>
unsigned long getauxval(unsigned long type) {
    unsigned long (*real_getauxval)(unsigned long) = dlsym(RTLD_NEXT, "getauxval");
    unsigned long value = real_getauxval(type);
    if (type == AT_HWCAP) {
        // FPHP, ASIMDHP, ASIMDDP and SVE (Linux arm64 uapi bits).
        value &= ~((1UL << 9) | (1UL << 10) | (1UL << 20) | (1UL << 22));
    }
    if (type == AT_HWCAP2) value = 0;
    return value;
}
