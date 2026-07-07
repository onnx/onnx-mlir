// Perf hardware counters to measure, selected at runtime via the
// ONNX_MLIR_INSTRUMENT_PERF_COUNTER environment variable as a space-separated
// list (e.g. "INSTRUCTIONS DTLB2_MISSES"). Each entry accepts a raw numeric
// PERF_TYPE_HARDWARE config value (decimal or 0x-prefixed hex) as a fallback
// for counters not in the name table below. When the variable is not set,
// perf counter collection is disabled entirely: no perf_event_open call is
// ever made, so the same binary keeps working for users without the
// privileges perf_event_open requires.
#define MAX_PERF_COUNTERS 16
static uint64_t perfCounterConfigs[MAX_PERF_COUNTERS];
static char perfCounterNames[MAX_PERF_COUNTERS][64];
static struct perf_event_attr perfCounterAttrs[MAX_PERF_COUNTERS];
static int perfCounterFds[MAX_PERF_COUNTERS];

// s390 cpum_cf PMU (type=10) hardware counter events: name-to-code table,
static const struct {
  const char *name;
  uint64_t config;
} perfCounterTable[] = {
    {"CPU_CYCLES", 0x0000},
    {"INSTRUCTIONS", 0x0001},
};

