/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- OMInstrument.inc - C/C++ Neutral Instrumentation Implementation---===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of the OMInstrument calls.
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

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "onnx-mlir/Runtime/OMInstrument.h"

#ifdef __cplusplus
using namespace onnx_mlir;
#endif

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

static bool instrumentReportDisabled = false;
static bool instrumentReportTimeDisabled = false;
static bool instrumentReportMemoryDisabled = false;
static int instrumentCounter = 0;
static int psErrorCount = 0;
static char instrumentReportOpName[INSTRUMENT_OP_NAME_MASK + 1];
static char instrumentReportNodeName[INSTRUMENT_NODE_NAME_MASK + 1];
static FILE *fout = 0;

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
static void ReportTime(char *instrumentReportOpName,
    char *instrumentReportNodeName, int isBefore) {
  LARGE_INTEGER newTime;
  LONGLONG resultSeconds1, resultMicroseconds1;
  LONGLONG resultSeconds2, resultMicroseconds2;
  QueryPerformanceCounter(&newTime);
  WinTimerSub(newTime, globalTime, &resultSeconds1, &resultMicroseconds1);
  WinTimerSub(newTime, initTime, &resultSeconds2, &resultMicroseconds2);
  // Print header and data for time.
  fprintf(fout, "==PERF-REPORT==, %s, %s, %s, %lld.%06lld, %lld.%06lld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), resultSeconds1, resultMicroseconds1,
      resultSeconds2, resultMicroseconds2);
  globalTime = newTime;
}
#else
static void ReportTime(char *instrumentReportOpName,
    char *instrumentReportNodeName, int isBefore) {
  struct timeval newTimeValue, result1, result2;
  gettimeofday(&newTimeValue, NULL);
  timersub(&newTimeValue, &globalTimeVal, &result1);
  timersub(&newTimeValue, &initTimeVal, &result2);
  // Print header and data for time.
  fprintf(fout, "==PERF-REPORT==, %s, %s, %s, %ld.%06ld, %ld.%06ld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), (long int)result1.tv_sec,
      (long int)result1.tv_usec, (long int)result2.tv_sec,
      (long int)result2.tv_usec);
  globalTimeVal = newTimeValue;
}
#endif

#ifdef _WIN32
static void ReportMemory() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(
      GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
  SIZE_T vMemSizeKB = pmc.PrivateUsage / 1024;
  fprintf(fout, "%zu\n", vMemSizeKB);
}
#else
static void ReportMemory() {
  char memCommand[200];
  char memOutput[200];
  FILE *memPipe;
  mypid = getpid();
  int num_chars_written =
      snprintf(memCommand, sizeof(memCommand), "ps -o vsz='' -p %d", mypid);
  assert(num_chars_written >= 0 && "snprintf write error to memCommand");
  memPipe = popen(memCommand, "r");
  if (!memPipe) {
    fprintf(fout, ", error-failed-to-execute-ps\n");
    psErrorCount++;
    return;
  }
  (void)fgets(memOutput, 200, memPipe);
  (void)fgetc(memPipe);
  memOutput[strcspn(memOutput, "\n")] = 0;
  if (!feof(memPipe)) {
    fprintf(fout, ", error-unexpected-output-from-pipe\n");
    psErrorCount++;
  } else {
    // No error, print data.
    fprintf(fout, ", %s\n", memOutput);
  }
  pclose(memPipe);
}
#endif

FILE *getInstrumentFile() {
  if (fout)
    return fout;
  fout = stdout;
  if (getenv("ONNX_MLIR_INSTRUMENT_FILE")) {
    char *fileName = getenv("ONNX_MLIR_INSTRUMENT_FILE");
    FILE *newFileHandle = fopen(fileName, "w+");
    if (newFileHandle) {
      fout = newFileHandle;
    }
  }
  assert(fout);
  return fout;
}

void OMInstrumentInit() {
  // Read environment variables.
  if (getenv("ONNX_MLIR_NO_INSTRUMENT_TIME")) {
    instrumentReportTimeDisabled = true;
    return;
  }
  if (getenv("ONNX_MLIR_NO_INSTRUMENT_MEMORY")) {
    instrumentReportMemoryDisabled = true;
  }
  if (getenv("ONNX_MLIR_NO_INSTRUMENT")) {
    instrumentReportDisabled = true;
  }
  // Handle redirection to file if requested.
  getInstrumentFile();

  // Init as appropriate.
  if (!instrumentReportDisabled) {
    TimeInit();
  }

  fprintf(fout, "==START-REPORT==\n");
}

void OMInstrumentPoint(const char *opName, int64_t iTag, const char *nodeName) {
  if (instrumentReportDisabled)
    return;

  // Detect which reporting we have to do here.
  uint64_t tag = iTag;
  bool initInstrument = IS_INSTRUMENT_INIT(tag);
  bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
  bool reportTime =
      !instrumentReportTimeDisabled && IS_INSTRUMENT_REPORT_TIME(tag);
  bool reportMem =
      !instrumentReportMemoryDisabled && IS_INSTRUMENT_REPORT_MEMORY(tag);

  if (initInstrument)
    OMInstrumentInit();

  if (!reportTime && !reportMem) {
    fprintf(fout, "==TICK-REPORT==, %i\n", instrumentCounter++);
    return;
  }

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

  if (reportTime) {
    ReportTime(instrumentReportOpName, instrumentReportNodeName, isBefore);
  }
  if (reportMem && psErrorCount < 20) {
    // Print header and data for memory.
    fprintf(fout, "==MEM-REPORT==, %s, %s, %s", instrumentReportOpName,
        instrumentReportNodeName, (isBefore ? "before" : "after"));
    ReportMemory();
  }
}
