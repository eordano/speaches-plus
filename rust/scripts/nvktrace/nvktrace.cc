#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <vector>
#include <algorithm>

namespace {
struct Agg { double ms; unsigned long n; };
std::map<std::string, Agg> g_agg;
std::mutex g_mu;
unsigned long g_records = 0;

void CUPTIAPI buf_req(uint8_t** buf, size_t* size, size_t* max) {
  *size = 8 << 20;
  *buf = (uint8_t*)aligned_alloc(8, *size);
  *max = 0;
}

void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  std::lock_guard<std::mutex> g(g_mu);
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = (CUpti_ActivityKernel9*)rec;
      double ms = (double)(k->end - k->start) * 1e-6;
      auto& a = g_agg[k->name ? k->name : "?"];
      a.ms += ms; a.n += 1; g_records++;
    }
  }
  free(buf);
}

void dump() {
  cuptiActivityFlushAll(1);
  std::lock_guard<std::mutex> g(g_mu);
  std::vector<std::pair<std::string, Agg>> v(g_agg.begin(), g_agg.end());
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.ms > b.second.ms; });
  double total = 0; for (auto& e : v) total += e.second.ms;
  fprintf(stderr, "NVKTRACE total_kernel_ms=%.1f records=%lu names=%zu\n", total, g_records, v.size());
  size_t top = std::min<size_t>(v.size(), 60);
  for (size_t i = 0; i < top; ++i)
    fprintf(stderr, "NVKTRACE %9.1f ms %8lu calls %7.3f ms/call  %s\n", v[i].second.ms, v[i].second.n,
            v[i].second.ms / v[i].second.n, v[i].first.substr(0, 140).c_str());
}

struct Init {
  Init() {
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    atexit(dump);
  }
} g_init;
}
