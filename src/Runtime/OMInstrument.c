/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- OMInstrument.c - C Neutral Instrumentation Implementation -------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of the OMInstrument calls.
//
//===----------------------------------------------------------------------===//

#include <assert.h>

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "onnx-mlir/Runtime/OMInstrument.h"

// Define global time variables.
#ifdef _WIN32
#include "windows.h"
// The windows.h include must go first.
#include "psapi.h"

static LARGE_INTEGER globalTime, initTime;
static LARGE_INTEGER perfFrequency;
#else
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

static struct timeval globalTimeVal, initTimeVal;
static pid_t mypid;
#endif

// Global variables for help.
static bool instrumentReportDisabled = false;
static bool instrumentReportTimeDisabled = false;
static bool instrumentReportMemoryDisabled = false;
static int instrumentCounter = 0; // For ticks; default without time/mem errors.
static int psErrorCount = 0;      // For counting memory errors.
// For string processing
static char instrumentReportOpName[INSTRUMENT_OP_NAME_MASK + 1];
static char instrumentReportNodeName[INSTRUMENT_NODE_NAME_MASK + 1];
static FILE *instrumentFout = 0; // For file output; none: undef; stdout.
static bool instrumentInitialized = false;
bool startReportPrinted = false;
// Compilation info passed by new-protocol model .so via OMInstrumentPointInit.
// NULL means old protocol, don't print any info.
static const char *instrumentCompilationInfo = NULL;

// Buffer data structure and array.
struct TimeRecord {
  const char *opName;
  const char *nodeName;
  uint64_t tag;
#ifdef _WIN32
  LARGE_INTEGER beforeTime;
  LARGE_INTEGER afterTime;
#else
  struct timeval beforeTime;
  struct timeval afterTime;
#endif
};

// Granite 3.1 needs 3K entries, safe with 8K.
#define MAX_TIME_RECORD_BUFFER (8 * 1024)
static struct TimeRecord timeRecordBuffer[MAX_TIME_RECORD_BUFFER];
int64_t bufferIndex = 0;

// =============================================================================
// Global state reset

// Maintain consistent with above initialization.
static void resetLocalState() {
  instrumentReportDisabled = false;
  instrumentReportTimeDisabled = false;
  instrumentReportMemoryDisabled = false;
  instrumentCounter = 0;
  psErrorCount = 0;
  instrumentFout = 0;
  instrumentInitialized = false;
  startReportPrinted = false;
  bufferIndex = 0;
  instrumentCompilationInfo = NULL;
}

// =============================================================================
// Time and memory error support.

// Global variable to help OMInstrumentHelper.h to keep track of nesting level
// of timing operations.
int timing_nest_level = 0;

#ifdef __MVS__
#define timersub(a, b, result)                                                 \
  do {                                                                         \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                              \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                           \
    if ((result)->tv_usec < 0) {                                               \
      --(result)->tv_sec;                                                      \
      (result)->tv_usec += 1000000;                                            \
    }                                                                          \
  } while (0);
#endif

#ifdef _WIN32
static void TimeInit() {
  QueryPerformanceFrequency(&perfFrequency);
  QueryPerformanceCounter(&globalTime);
  initTime = globalTime;
}
#else
static void TimeInit() {
  gettimeofday(&globalTimeVal, NULL);
  initTimeVal = globalTimeVal;
}
#endif

#ifdef _WIN32
static inline void WinTimerSub(LARGE_INTEGER newTime, LARGE_INTEGER prevTime,
    LONGLONG *resultSeconds, LONGLONG *resultMicroseconds) {
  LONGLONG elapsed = newTime.QuadPart - prevTime.QuadPart;
  *resultSeconds = elapsed / perfFrequency.QuadPart;
  *resultMicroseconds =
      ((elapsed * 1000000) / perfFrequency.QuadPart) % 1000000;
}
static inline void GetTime(LARGE_INTEGER *newTime) {
  QueryPerformanceCounter(newTime);
}
static inline void PrintTime(LARGE_INTEGER *newTime, int isBefore) {
  LONGLONG resultSeconds1, resultMicroseconds1;
  LONGLONG resultSeconds2, resultMicroseconds2;
  WinTimerSub(newTime, globalTime, &resultSeconds1, &resultMicroseconds1);
  WinTimerSub(newTime, initTime, &resultSeconds2, &resultMicroseconds2);
  // Print header and data for time.
  fprintf(instrumentFout,
      "==PERF-REPORT==, %s, %s, %s, %lld.%06lld, %lld.%06lld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), resultSeconds1, resultMicroseconds1,
      resultSeconds2, resultMicroseconds2);
  globalTime = newTime;
}
#else
static inline void GetTime(struct timeval *newTimeValue) {
  gettimeofday(newTimeValue, NULL);
}
static inline void PrintTime(struct timeval *newTimeValue, int isBefore) {
  struct timeval result1, result2;
  timersub(newTimeValue, &globalTimeVal, &result1);
  timersub(newTimeValue, &initTimeVal, &result2);
  // Print header and data for time.
  fprintf(instrumentFout, "==PERF-REPORT==, %s, %s, %s, %ld.%06ld, %ld.%06ld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), (long int)result1.tv_sec,
      (long int)result1.tv_usec, (long int)result2.tv_sec,
      (long int)result2.tv_usec);
  globalTimeVal = *newTimeValue;
}
#endif

#ifdef _WIN32
static void ReportMemory() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(
      GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
  SIZE_T vMemSizeKB = pmc.PrivateUsage / 1024;
  fprintf(instrumentFout, "%zu\n", vMemSizeKB);
}
#else
static void ReportMemory() {
  char memCommand[200];
  char memOutput[200];
  FILE *memPipe;
  mypid = getpid();
  int num_chars_written =
      snprintf(memCommand, sizeof(memCommand), "ps -o vsz='' -p %d", mypid);
  assert(num_chars_written >= 0); // Error:"snprintf write error to memCommand".
  memPipe = popen(memCommand, "r");
  if (!memPipe) {
    fprintf(instrumentFout, ", error-failed-to-execute-ps\n");
    psErrorCount++;
    return;
  }
  (void)fgets(memOutput, 200, memPipe);
  (void)fgetc(memPipe);
  memOutput[strcspn(memOutput, "\n")] = 0;
  if (!feof(memPipe)) {
    fprintf(instrumentFout, ", error-unexpected-output-from-pipe\n");
    psErrorCount++;
  } else {
    // No error, print data.
    fprintf(instrumentFout, ", %s\n", memOutput);
  }
  pclose(memPipe);
}
#endif

static void ProcessName(
    const char *opName, uint64_t tag, const char *nodeName) {
  // Unfortunately, the op and node names passed at runtime have sometimes an
  // incorrect length, and as a result, garbage is printed. To avoid this, a
  // (possibly temporary) fix is to encode the string lengths in the tag
  // (which are correct at compile time) so that we only print the intended
  // info here.
  uint64_t opNameLen = GET_INSTRUMENT_OP_NAME_LEN(tag);
  uint64_t nodeNameLen = GET_INSTRUMENT_NODE_NAME_LEN(tag);
  assert(opNameLen <= INSTRUMENT_OP_NAME_MASK &&
         nodeNameLen <= INSTRUMENT_NODE_NAME_MASK);
  // Safe copy of op and node names.
  strncpy(instrumentReportOpName, opName, opNameLen);
  instrumentReportOpName[opNameLen] = '\0';
  strncpy(instrumentReportNodeName, nodeName, nodeNameLen);
  instrumentReportNodeName[nodeNameLen] = '\0';
}

// =============================================================================
// Perf Utilities

static bool perfCounterEnabled = false;
static int perfCounterCount = 0;

#ifdef __s390x__
#define _GNU_SOURCE
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>

static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
    int group_fd, unsigned long flags) {
  return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

#include "PerfData.inc"

// Resolve a single counter token (name or raw numeric config) into slot
// `perfCounterCount`. Returns true and bumps perfCounterCount on success.
static bool addPerfCounter(const char *token) {
  for (size_t i = 0; i < sizeof(perfCounterTable) / sizeof(perfCounterTable[0]);
       ++i) {
    if (strcmp(token, perfCounterTable[i].name) == 0) {
      perfCounterConfigs[perfCounterCount] = perfCounterTable[i].config;
      strncpy(perfCounterNames[perfCounterCount], perfCounterTable[i].name,
          sizeof(perfCounterNames[0]) - 1);
      perfCounterCount++;
      return true;
    }
  }
  // Fall back: allow a raw numeric config value. Resolve it back to its
  // symbolic name (if known) so reports print a name instead of a number.
  char *end;
  long val = strtol(token, &end, 0);
  if (end != token && *end == '\0') {
    perfCounterConfigs[perfCounterCount] = (uint64_t)val;
    const char *resolvedName = token;
    for (size_t i = 0;
         i < sizeof(perfCounterTable) / sizeof(perfCounterTable[0]); ++i) {
      if (perfCounterTable[i].config == (uint64_t)val) {
        resolvedName = perfCounterTable[i].name;
        break;
      }
    }
    strncpy(perfCounterNames[perfCounterCount], resolvedName,
        sizeof(perfCounterNames[0]) - 1);
    perfCounterCount++;
    return true;
  }
  fprintf(stderr,
      "Warning: unknown ONNX_MLIR_INSTRUMENT_PERF_COUNTER entry '%s', "
      "skipping\n",
      token);
  return false;
}

// Starts (opens, resets, enables) the perf counter for slot `index` of
// perfCounterConfigs/perfCounterAttrs, storing its fd in perfCounterFds. The
// counter then runs continuously (it is not reset or disabled per op); use
// read_perf_counter() to sample it and end_perf_counter() to close it.
static inline void start_perf_counter(int index) {
  struct perf_event_attr *pe = &perfCounterAttrs[index];
  memset(pe, 0, sizeof(*pe));
  pe->type = 10; // cpum_cf PMU type for s390
  pe->size = sizeof(*pe);

  pe->config = perfCounterConfigs[index];

  pe->disabled = 1;

  pe->exclude_kernel = 0;
  pe->exclude_hv = 0;

  int fd = perf_event_open(pe, 0, -1, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
    exit(-1);
  }

  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  perfCounterFds[index] = fd;
}

// Disables and closes the perf counter for slot `index`.
static inline void end_perf_counter(int index) {
  int fd = perfCounterFds[index];
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  close(fd);
}

static void initPerfCounterConfig() {
  // Close any counters still open from a previous run before reopening.
  for (int i = 0; i < perfCounterCount; ++i)
    end_perf_counter(i);
  perfCounterCount = 0;
  perfCounterEnabled = false;

  const char *env = getenv("ONNX_MLIR_INSTRUMENT_PERF_COUNTER");
  if (!env)
    return; // Not set: leave perf counter collection disabled.
  // Tokenize a private copy of the env value on spaces; strtok_r needs a
  // mutable buffer and getenv's result must not be modified.
  char buf[256];
  strncpy(buf, env, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  char *saveptr;
  char *token = strtok_r(buf, " ", &saveptr);
  while (token && perfCounterCount < MAX_PERF_COUNTERS) {
    addPerfCounter(token);
    token = strtok_r(NULL, " ", &saveptr);
  }
  if (token)
    fprintf(stderr,
        "Warning: ONNX_MLIR_INSTRUMENT_PERF_COUNTER has more than %d "
        "counters, extra entries ignored\n",
        MAX_PERF_COUNTERS);
  perfCounterEnabled = perfCounterCount > 0;
  for (int i = 0; i < perfCounterCount; ++i)
    start_perf_counter(i);
}

// Reads the current (cumulative, since start_perf_counter) value of the perf
// counter for slot `index` and prints it; the counter is left running.
static inline void read_perf_counter(int index, bool isBefore) {
  uint64_t count;
  if (read(perfCounterFds[index], &count, sizeof(count)) != sizeof(count)) {
    perror("read");
    exit(-1);
  }
  fprintf(instrumentFout, "==CNTR-REPORT==, %s, %s, %s, %s, %lu\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), perfCounterNames[index], count);
}
#else
// Empty functions for systems not implemented for perf
static void initPerfCounterConfig() {}
static inline void read_perf_counter(int index, bool isBefore) {}
#endif

// =============================================================================
// Buffer management

static inline void printStartReport() {
  if (startReportPrinted)
    return;
  assert(instrumentFout); // Error: "expected instrumentFout for reporting".
  fprintf(instrumentFout, "==START-REPORT==\n");
  if (instrumentCompilationInfo)
    fprintf(instrumentFout, "==COMPILE-INFO-REPORT==, %s\n",
        instrumentCompilationInfo);
  startReportPrinted = true;
}

static void flushRecordBuffer() {
  if (instrumentReportTimeDisabled || bufferIndex <= 0)
    return;
  // We have entries, print them now.
  printStartReport();
  for (int64_t i = 0; i < bufferIndex; ++i) {
    uint64_t tag = timeRecordBuffer[i].tag;
    ProcessName(timeRecordBuffer[i].opName, tag, timeRecordBuffer[i].nodeName);
    bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
    if (isBefore)
      PrintTime(&timeRecordBuffer[i].beforeTime, /*before*/ true);
    bool isAfter = IS_INSTRUMENT_BEFORE_OP(tag);
    if (isAfter)
      PrintTime(&timeRecordBuffer[i].afterTime, /*before*/ false);
  }
  fflush(instrumentFout);
  bufferIndex = 0;
}

static void updateRecordBuffer(
    const char *opName, uint64_t tag, const char *nodeName) {
  int64_t i = bufferIndex - 1;
  if (i >= 0 && timeRecordBuffer[i].opName == opName &&
      timeRecordBuffer[i].nodeName == nodeName) {
    // Can reuse entry, or the tags so we have bits for both.
    timeRecordBuffer[i].tag = timeRecordBuffer[i].tag | tag;
  } else {
    // Need a new entry; flush if full, then initialize entry.
    if (bufferIndex >= MAX_TIME_RECORD_BUFFER)
      flushRecordBuffer();
    i = bufferIndex++;
    timeRecordBuffer[i].opName = opName;
    timeRecordBuffer[i].nodeName = nodeName;
    timeRecordBuffer[i].tag = tag;
  }
  // Record time.
  bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
  if (isBefore)
    GetTime(&timeRecordBuffer[i].beforeTime);
  else
    GetTime(&timeRecordBuffer[i].afterTime);
}

// =============================================================================
// Support for initialization

// Initialize instrumentFout on the first call (as instrumentFout is statically
// instrumentInitialized to null). If defined, instrumentFout will target the
// value in ONNX_MLIR_INSTRUMENT_FILE. Otherwise, instrumentFout default to
// standard out.

FILE *getInstrumentFile(bool withPrintStartReport) {
  if (!instrumentFout) {
    instrumentFout = stdout;
    if (getenv("ONNX_MLIR_INSTRUMENT_FILE")) {
      char *fileName = getenv("ONNX_MLIR_INSTRUMENT_FILE");
      FILE *newFileHandle = fopen(fileName, "a");
      if (newFileHandle) {
        instrumentFout = newFileHandle;
      }
    }
    assert(instrumentFout);
  }
  if (withPrintStartReport)
    printStartReport();
  return instrumentFout;
}

// Called only from OMInstrumentPoint, which is only generated by the compiler
// into the model .so.
static void startInstrumentation() {

  // New instrumentation, set printed start report to false.  Buffer is also
  // reset between runs.
  startReportPrinted = false;
  bufferIndex = 0;
  if (!instrumentInitialized) {
    instrumentInitialized = true;
    // Read environment variables.
    if (getenv("ONNX_MLIR_NO_INSTRUMENT_TIME"))
      instrumentReportTimeDisabled = true;
    if (getenv("ONNX_MLIR_NO_INSTRUMENT_MEMORY"))
      instrumentReportMemoryDisabled = true;
    if (getenv("ONNX_MLIR_NO_INSTRUMENT")) {
      instrumentReportDisabled = true;
      instrumentReportTimeDisabled = true;
      instrumentReportMemoryDisabled = true;
    }
    initPerfCounterConfig();
    // Open the output file only when reporting is active, so that
    // ONNX_MLIR_NO_INSTRUMENT never touches the output file at all.
    // Still open when only time or only memory is disabled, since the
    // other path may still need to write (e.g. memory-only reporting).
    if (!instrumentReportDisabled)
      getInstrumentFile(
          /*print report will be on demand in flush buffer*/ false);
  }

  // Init as appropriate.
  if (!instrumentReportDisabled) {
    TimeInit();
  }
}

// =============================================================================
// Support OM interface.

/// @brief  Print the instrumentation records on standard out or in a file,
/// depending on whether `ONNX_MLIR_INSTRUMENT_FILE` is defined or not. It only
/// print the instrumentation of the last run. If no instrumentation was
/// generated, this call does nothing. Note that printing may occur prior to
/// this call in the unlikely event that the instrumentation buffer became full.
/// So essentially this call make sure that all of the timeing info is printed.
void omInstrumentPrint() {
  flushRecordBuffer();
  resetLocalState();
}

// Single-slot snapshot of the most recent instrument call. Written
// unconditionally at the top of every OMInstrumentPoint entry so an
// external statistical sampler (e.g. profile-model.py's in-process
// SIGPROF handler) can read "what op is the model currently on?"
// directly at sample time. The pointers point into the model .so's
// `.rodata` (the global string literals the compiler emitted for each
// op/node name) and are stable for the program's lifetime, so it's
// safe to dereference them later from a non-interrupt context. The
// `tag` carries the begin/end bit (IS_INSTRUMENT_BEFORE_OP) so the
// reader can tell whether sampling caught the op body or a brief gap
// between ops. Three pointer-sized writes are negligible relative to
// the existing OMInstrumentPoint cost; they happen even when
// `instrumentReportDisabled` is set so the sampler keeps working
// under `ONNX_MLIR_NO_INSTRUMENT=1` (which is how profile-model
// silences the per-call CSV output without losing the markers).
const char *OMCurrentOpName = NULL;
const char *OMCurrentNodeName = NULL;
int64_t OMCurrentTag = 0;

// External call, part of the OM interface, called once per inference run by
// new-protocol model .so files before the first OMInstrumentPoint. Stores the
// compilation info string and initializes the runtime for this run.
void OMInstrumentPointInit(int64_t iTag, const char *compilationInfo) {
  instrumentCompilationInfo = compilationInfo;
  startInstrumentation();
}

// External call, part of the OM interface, calls generated by compiler.
void OMInstrumentPoint(const char *opName, int64_t iTag, const char *nodeName) {
  OMCurrentOpName = opName;
  OMCurrentNodeName = nodeName;
  OMCurrentTag = iTag;
  if (instrumentReportDisabled)
    return;

  // Process init.
  uint64_t tag = iTag;
  bool isInit = IS_INSTRUMENT_INIT(tag);
  if (isInit) {
    startInstrumentation();
  }
  // Report.
  bool reportTime =
      !instrumentReportTimeDisabled && IS_INSTRUMENT_REPORT_TIME(tag);
  bool reportMem =
      !instrumentReportMemoryDisabled && IS_INSTRUMENT_REPORT_MEMORY(tag);

  if (reportTime)
    updateRecordBuffer(opName, tag, nodeName);
  if (reportMem && psErrorCount < 20) {
    // Print header and data for memory.
    printStartReport();
    ProcessName(opName, tag, nodeName);
    bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
    fprintf(instrumentFout, "==MEM-REPORT==, %s, %s, %s",
        instrumentReportOpName, instrumentReportNodeName,
        (isBefore ? "before" : "after"));
    ReportMemory();
  }
  // Code for perf counter. Only touches perf_event_open when the user opted
  // in via ONNX_MLIR_INSTRUMENT_PERF_COUNTER, so users without permission to
  // call perf_event_open are unaffected. Counters are started once (see
  // initPerfCounterConfig) and simply sampled here at each op boundary.
  if (perfCounterEnabled) {
    printStartReport();
    ProcessName(opName, tag, nodeName);
    bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
    for (int i = 0; i < perfCounterCount; ++i)
      read_perf_counter(i, isBefore);
  }
}
