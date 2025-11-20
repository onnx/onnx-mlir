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
static char instrumentReportOpName[INSTRUMENT_OP_NAME_MASK + 1];
static char instrumentReportNodeName[INSTRUMENT_NODE_NAME_MASK + 1];
static FILE *fout = 0;         // For file output; when none, stdout.
static bool useBuffer = false; // True if ONNX_MLIR_FAST_INSTRUMENT is defined.
static int initCounter = 0;

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

static inline void PrintTime(LARGE_INTEGER *newTime,
    char *instrumentReportOpName, char *instrumentReportNodeName,
    int isBefore) {
  LONGLONG resultSeconds1, resultMicroseconds1;
  LONGLONG resultSeconds2, resultMicroseconds2;
  WinTimerSub(newTime, globalTime, &resultSeconds1, &resultMicroseconds1);
  WinTimerSub(newTime, initTime, &resultSeconds2, &resultMicroseconds2);
  // Print header and data for time.
  fprintf(fout, "==PERF-REPORT==, %s, %s, %s, %lld.%06lld, %lld.%06lld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), resultSeconds1, resultMicroseconds1,
      resultSeconds2, resultMicroseconds2);
  globalTime = newTime;
}

static void ReportTime(char *instrumentReportOpName,
    char *instrumentReportNodeName, int isBefore) {
  LARGE_INTEGER newTime;
  GetTime(&newTime);
  PrintTime(
      &newTime, instrumentReportOpName, instrumentReportNodeName, isBefore);
}
#else
static inline void GetTime(struct timeval *newTimeValue) {
  gettimeofday(newTimeValue, NULL);
}

static inline void PrintTime(struct timeval *newTimeValue,
    char *instrumentReportOpName, char *instrumentReportNodeName,
    int isBefore) {
  struct timeval result1, result2;
  timersub(newTimeValue, &globalTimeVal, &result1);
  timersub(newTimeValue, &initTimeVal, &result2);
  // Print header and data for time.
  fprintf(fout, "==PERF-REPORT==, %s, %s, %s, %ld.%06ld, %ld.%06ld\n",
      instrumentReportOpName, instrumentReportNodeName,
      (isBefore ? "before" : "after"), (long int)result1.tv_sec,
      (long int)result1.tv_usec, (long int)result2.tv_sec,
      (long int)result2.tv_usec);
  globalTimeVal = *newTimeValue;
}

static void ReportTime(char *instrumentReportOpName,
    char *instrumentReportNodeName, int isBefore) {
  struct timeval newTimeValue;
  GetTime(&newTimeValue);
  PrintTime(&newTimeValue, instrumentReportOpName, instrumentReportNodeName,
      isBefore);
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

void ProcessName(const char *opName, uint64_t tag, const char *nodeName) {
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
// Buffer management

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

#define MAX_TIME_RECORD_BUFFER (16 * 1024)
static struct TimeRecord timeRecordBuffer[MAX_TIME_RECORD_BUFFER];
static int64_t bufferIndex = 0;

void flushRecordBuffer() {
  for (int64_t i = 0; i < bufferIndex; ++i) {
    uint64_t tag = timeRecordBuffer[i].tag;
    ProcessName(timeRecordBuffer[i].opName, tag, timeRecordBuffer[i].nodeName);
    bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
    if (isBefore)
      PrintTime(&timeRecordBuffer[i].beforeTime, instrumentReportOpName,
          instrumentReportNodeName, /*before*/ true);
    bool isAfter = IS_INSTRUMENT_BEFORE_OP(tag);
    if (isAfter)
      PrintTime(&timeRecordBuffer[i].afterTime, instrumentReportOpName,
          instrumentReportNodeName, /*before*/ false);
  }
  fflush(fout);
  bufferIndex = 0;
}

void updateRecordBuffer(
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

// Initialize fout on the first call (as fout is statically initialized to
// null). If defined, fout will target the value in ONNX_MLIR_INSTRUMENT_FILE.
// Otherwise, fout default to standard out.
//
// Now this initialization is also used by omTensorPrint, for example to print
// each tensor's signatures (when omTensorPrint start with the %i substring).
// Now because we want to disable signature printing when using buffers to print
// (so as to have low overheads), we thus disable printing by returning the 0
// value. The useBuffer variable is defined by ONNX_MLIR_FAST_INSTRUMENT.
FILE *getInstrumentFile() {
  if (!fout) {
    fout = stdout;
    if (getenv("ONNX_MLIR_INSTRUMENT_FILE")) {
      char *fileName = getenv("ONNX_MLIR_INSTRUMENT_FILE");
      FILE *newFileHandle = fopen(fileName, "w+");
      if (newFileHandle) {
        fout = newFileHandle;
      }
    }
    assert(fout);
    if (getenv("ONNX_MLIR_FAST_INSTRUMENT")) {
      useBuffer = true;
    }
  }
  return useBuffer ? 0 : fout;
}

void OMInstrumentStart() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    // First: read environment variables.
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
    // Then handle the possible redirection of output to file, and determine if
    // buffered output is requested..
    getInstrumentFile();
  }

  // Init as appropriate.
  if (!instrumentReportDisabled) {
    TimeInit();
  }
  fprintf(fout, "==START-REPORT==\n");
}

void OMInstrumentStop() {
  if (useBuffer)
    flushRecordBuffer();
}

// =============================================================================
// Support OMInstrumentPoint.

void OMInstrumentPoint(const char *opName, int64_t iTag, const char *nodeName) {
  if (instrumentReportDisabled)
    return;

  // Process init (aka start/stop instrumentation).
  uint64_t tag = iTag;
  bool startOrStopInstrument = IS_INSTRUMENT_START_STOP(tag);
  if (startOrStopInstrument) {
    if ((++initCounter) % 2 == 1) {
      OMInstrumentStart();
      if (instrumentReportDisabled)
        return;
    } else {
      OMInstrumentStop();
      return;
    }
  }
  // Determine if we need to report on time or memory, and quick shortcut for
  // buffered output.
  bool reportTime =
      !instrumentReportTimeDisabled && IS_INSTRUMENT_REPORT_TIME(tag);
  bool reportMem =
      !instrumentReportMemoryDisabled && IS_INSTRUMENT_REPORT_MEMORY(tag);
  if (useBuffer && reportTime && !reportMem) {
    updateRecordBuffer(opName, tag, nodeName);
    return;
  }
  if (!reportTime && !reportMem) {
    fprintf(fout, "==TICK-REPORT==, %i\n", instrumentCounter++);
    return;
  }

  // Process non-buffered reporting.
  ProcessName(opName, tag, nodeName);
  bool isBefore = IS_INSTRUMENT_BEFORE_OP(tag);
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
