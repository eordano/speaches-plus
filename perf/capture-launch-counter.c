#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
typedef struct { unsigned x, y, z; } dim3_t;
static _Atomic long in_capture = 0;
static _Atomic long k = 0, mc = 0, ms = 0;
static void* real(const char* n) { return dlsym(RTLD_NEXT, n); }
int cudaStreamBeginCapture(void* s, int mode) {
    static int (*r)(void*, int) = 0; if (!r) r = real("cudaStreamBeginCapture");
    atomic_store(&in_capture, 1); atomic_store(&k, 0); atomic_store(&mc, 0); atomic_store(&ms, 0);
    return r(s, mode);
}
int cudaStreamEndCapture(void* s, void** g) {
    static int (*r)(void*, void**) = 0; if (!r) r = real("cudaStreamEndCapture");
    static int (*getn)(void*, void**, unsigned long*) = 0;
    static int (*gett)(void*, int*) = 0;
    if (!getn) getn = real("cudaGraphGetNodes");
    if (!gett) gett = real("cudaGraphNodeGetType");
    int rc = r(s, g);
    unsigned long n = 0;
    long hist[16] = {0};
    if (rc == 0 && g && *g && getn && getn(*g, 0, &n) == 0 && n > 0 && n < 100000) {
        void** nodes = malloc(n * sizeof(void*));
        unsigned long m = n;
        if (nodes && getn(*g, nodes, &m) == 0) {
            for (unsigned long i = 0; i < m; i++) {
                int t = -1;
                if (gett && gett(nodes[i], &t) == 0 && t >= 0 && t < 16) hist[t]++;
            }
        }
        free(nodes);
    }
    fprintf(stderr, "[countlaunch] capture ended: graph_nodes=%lu kernel=%ld memcpy=%ld "
            "memset=%ld host=%ld child=%ld empty=%ld runtime_launch_seen=%ld\n",
            n, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], atomic_load(&k));
    atomic_store(&in_capture, 0);
    return rc;
}
int cudaLaunchKernel(const void* f, dim3_t g, dim3_t b, void** a, unsigned long sh, void* st) {
    static int (*r)(const void*, dim3_t, dim3_t, void**, unsigned long, void*) = 0;
    if (!r) r = real("cudaLaunchKernel");
    if (atomic_load(&in_capture)) atomic_fetch_add(&k, 1);
    return r(f, g, b, a, sh, st);
}
int cudaMemcpyAsync(void* d, const void* s2, unsigned long n, int kind, void* st) {
    static int (*r)(void*, const void*, unsigned long, int, void*) = 0;
    if (!r) r = real("cudaMemcpyAsync");
    if (atomic_load(&in_capture)) atomic_fetch_add(&mc, 1);
    return r(d, s2, n, kind, st);
}
int cudaMemsetAsync(void* d, int v, unsigned long n, void* st) {
    static int (*r)(void*, int, unsigned long, void*) = 0;
    if (!r) r = real("cudaMemsetAsync");
    if (atomic_load(&in_capture)) atomic_fetch_add(&ms, 1);
    return r(d, v, n, st);
}
