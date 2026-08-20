// Minimal CUPTI activity-trace injection library.
// Build as a .so, point CUDA_INJECTION64_PATH at it; CUDA calls InitializeInjection().
// Dumps a TSV of every kernel / memcpy / memset with GPU start+end timestamps.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <unistd.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#define BUF_SIZE (32 * 1024 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

namespace {

struct Rec {
  uint64_t start;
  uint64_t end;
  uint32_t nameId;
  uint32_t streamId;
  uint32_t graphId;
  uint32_t correlationId;
  int32_t gridX, gridY, gridZ;
  int32_t blockX, blockY, blockZ;
  uint64_t bytes;
  uint8_t kind; // 0 kernel, 1 memcpy, 2 memset
  uint8_t copyKind;
};

std::mutex g_mu;
std::vector<Rec> g_recs;
std::unordered_map<std::string, uint32_t> g_names;
std::vector<std::string> g_nameList;
bool g_done = false;

uint32_t internName(const char *n) {
  if (!n) n = "(null)";
  std::string s(n);
  auto it = g_names.find(s);
  if (it != g_names.end()) return it->second;
  uint32_t id = (uint32_t)g_nameList.size();
  g_names.emplace(s, id);
  g_nameList.push_back(s);
  return id;
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *raw = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  *buffer = ALIGN_BUFFER(raw, ALIGN_SIZE);
  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t *buffer,
                              size_t /*size*/, size_t validSize) {
  CUpti_Activity *record = NULL;
  std::lock_guard<std::mutex> lk(g_mu);
  if (validSize > 0) {
    do {
      CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        Rec r;
        memset(&r, 0, sizeof(r));
        switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_KERNEL: {
          CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)record;
          r.kind = 0;
          r.start = k->start;
          r.end = k->end;
          r.nameId = internName(k->name);
          r.streamId = k->streamId;
          r.graphId = k->graphId;
          r.correlationId = k->correlationId;
          r.gridX = k->gridX; r.gridY = k->gridY; r.gridZ = k->gridZ;
          r.blockX = k->blockX; r.blockY = k->blockY; r.blockZ = k->blockZ;
          g_recs.push_back(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)record;
          r.kind = 1;
          r.start = m->start;
          r.end = m->end;
          r.streamId = m->streamId;
          r.graphId = m->graphId;
          r.correlationId = m->correlationId;
          r.bytes = m->bytes;
          r.copyKind = m->copyKind;
          r.nameId = internName("memcpy");
          g_recs.push_back(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
          CUpti_ActivityMemset4 *m = (CUpti_ActivityMemset4 *)record;
          r.kind = 2;
          r.start = m->start;
          r.end = m->end;
          r.streamId = m->streamId;
          r.graphId = m->graphId;
          r.correlationId = m->correlationId;
          r.bytes = m->bytes;
          r.nameId = internName("memset");
          g_recs.push_back(r);
          break;
        }
        default:
          break;
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        break;
      }
    } while (1);
  }
  free(buffer);
}

const char *copyKindStr(uint8_t k) {
  switch (k) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
  default: return "other";
  }
}

void dumpAll() {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_done) return;
    g_done = true;
  }
  cuptiActivityFlushAll(1);
  std::lock_guard<std::mutex> lk(g_mu);
  const char *out = getenv("NVTRACE_OUT");
  if (!out) out = "nvtrace.tsv";
  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "[nvtrace] cannot open %s\n", out);
    return;
  }
  fprintf(f, "#kind\tstart\tend\tdur_ns\tstream\tgraph\tcorr\tgrid\tblock\tbytes\tcopy\tname\n");
  for (const Rec &r : g_recs) {
    const char *kindS = r.kind == 0 ? "kernel" : (r.kind == 1 ? "memcpy" : "memset");
    fprintf(f, "%s\t%llu\t%llu\t%llu\t%u\t%u\t%u\t%dx%dx%d\t%dx%dx%d\t%llu\t%s\t%s\n",
            kindS, (unsigned long long)r.start, (unsigned long long)r.end,
            (unsigned long long)(r.end - r.start), r.streamId, r.graphId,
            r.correlationId, r.gridX, r.gridY, r.gridZ, r.blockX, r.blockY,
            r.blockZ, (unsigned long long)r.bytes,
            r.kind == 1 ? copyKindStr(r.copyKind) : "-",
            g_nameList[r.nameId].c_str());
  }
  fclose(f);
  fprintf(stderr, "[nvtrace] wrote %zu records to %s\n", g_recs.size(), out);
}

void atExitHandler(void) { dumpAll(); }

// A server host dies by SIGTERM, which skips atexit and would drop the whole
// trace. dumpAll is idempotent (g_done); not strictly async-signal-safe, but
// the only use is a controlled teardown of a quiesced process.
void sigHandler(int) {
  dumpAll();
  _exit(0);
}

} // namespace

extern "C" int InitializeInjection(void) {
  static bool inited = false;
  if (inited) return 1;
  inited = true;
  g_recs.reserve(4 << 20);
  cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
  atexit(&atExitHandler);
  signal(SIGTERM, sigHandler);
  signal(SIGINT, sigHandler);
  fprintf(stderr, "[nvtrace] injection active\n");
  return 1;
}
