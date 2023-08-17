/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMTensor.inc - C/C++ Neutral OMTensor Implementation--------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of OMTensor data structures
// and helper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>
#else
#include <assert.h>
#endif

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMInstrument.h"

#ifdef _WIN32
#include "psapi.h"
#include "windows.h"

static LARGE_INTEGER globalTime, initTime;
static LARGE_INTEGER perfFrequency;
#else
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

static struct timeval globalTimeVal, initTimeVal;
static pid_t mypid;
static int psErrorCount = 0;
#endif

static bool instrumentReportDisabled = false;
static bool instrumentReportTimeDisabled = false;
static bool instrumentReportMemoryDisabled = false;
static int instrumentCounter = 0;

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
void TimeInit() {
  QueryPerformanceFrequency(&perfFrequency);
  QueryPerformanceCounter(&globalTime);
  initTime = globalTime;
}
#else
void TimeInit() {
  gettimeofday(&globalTimeVal, NULL);
  initTimeVal = globalTimeVal;
}
#endif

#ifdef _WIN32
inline void WinTimerSub(LARGE_INTEGER newTime, LARGE_INTEGER prevTime,
    LONGLONG *resultSeconds, LONGLONG *resultMicroseconds) {
  LONGLONG elapsed = newTime.QuadPart - prevTime.QuadPart;
  *resultSeconds = elapsed / perfFrequency.QuadPart;
  *resultMicroseconds =
      ((elapsed * 1000000) / perfFrequency.QuadPart) % 1000000;
}
void ReportTime() {
  LARGE_INTEGER newTime;
  LONGLONG resultSeconds, resultMicroseconds;
  QueryPerformanceCounter(&newTime);
  WinTimerSub(newTime, globalTime, &resultSeconds, &resultMicroseconds);
  fprintf(stderr, ", %lld.%06lld", resultSeconds, resultMicroseconds);
  WinTimerSub(newTime, initTime, &resultSeconds, &resultMicroseconds);
  fprintf(stderr, ", %lld.%06lld\n", resultSeconds, resultMicroseconds);
  globalTime = newTime;
}
#else
void ReportTime() {
  struct timeval newTimeValue, result;
  gettimeofday(&newTimeValue, NULL);
  timersub(&newTimeValue, &globalTimeVal, &result);
  fprintf(stderr, ", %ld.%06ld", (long int)result.tv_sec,
      (long int)result.tv_usec);
  timersub(&newTimeValue, &initTimeVal, &result);
  fprintf(stderr, ", %ld.%06ld\n", (long int)result.tv_sec,
      (long int)result.tv_usec);
  globalTimeVal = newTimeValue;
}
#endif

#ifdef _WIN32
void ReportMemory() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(
      GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
  SIZE_T vMemSizeKB = pmc.PrivateUsage / 1024;
  fprintf(stderr, "%zu\n", vMemSizeKB);
}
#else
void ReportMemory() {
  char memCommand[200];
  char memOutput[200];
  FILE *memPipe;
  mypid = getpid();
  snprintf(memCommand, sizeof(memCommand), "ps -o vsz='' -p %d", mypid);
  memPipe = popen(memCommand, "r");
  if (!memPipe) {
    fprintf(stderr, ", error-failed-to-execute-ps\n");
    psErrorCount++;
    return;
  }
  (void)fgets(memOutput, 200, memPipe);
  (void)fgetc(memPipe);
  memOutput[strcspn(memOutput, "\n")] = 0;
  if (!feof(memPipe)) {
    fprintf(stderr, ", error-unexpected-output-from-pipe\n");
    psErrorCount++;
  } else {
    // No error, print data/
    printf(", %s\n", memOutput);
  }
  pclose(memPipe);
}
#endif

enum InstrumentActions {
  InstrumentBeforeOp,
  InstrumentAfterOp,
  InstrumentReportTime,
  InstrumentReportMemory
};

void OMInstrumentInit() {
  if (getenv("NOOMINSTRUMENTTIME")) {
    instrumentReportTimeDisabled = true;
  }
  if (getenv("NOOMINSTRUMENTMEMORY")) {
    instrumentReportMemoryDisabled = true;
  }
  if (getenv("NOOMINSTRUMENT")) {
    instrumentReportDisabled = true;
  }

  if (!instrumentReportDisabled) {
    TimeInit();
  }
}

void OMInstrumentPoint(const char *opName, int64_t tag, const char *nodeName) {
  if (instrumentReportDisabled)
    return;

  bool isBefore = tag & (1 << (int)InstrumentBeforeOp);
  bool reportTime =
      tag & (1 << (int)InstrumentReportTime) && !instrumentReportTimeDisabled;
  bool reportMem = tag & (1 << (int)InstrumentReportMemory) &&
                   !instrumentReportMemoryDisabled;

  if (reportTime) {
    // Print header and data for time.
    fprintf(stderr, "==PERF-REPORT==, %s, %s, %s", opName,
        (strncmp(nodeName, "NOTSET", 6) != 0 ? nodeName : "no-node-name"),
        (isBefore ? "before" : "after"));
    ReportTime();
  }
  if (reportMem && psErrorCount < 20) {
    // Print header and data for memory.
    fprintf(stderr, "==MEM-REPORT==, %s, %s, %s", opName,
        (strncmp(nodeName, "NOTSET", 6) != 0 ? nodeName : "no-node-name"),
        (isBefore ? "before" : "after"));
    ReportMemory();
  }
  // Not sure if needed, remove if that is not the case.
  instrumentCounter++;
}
