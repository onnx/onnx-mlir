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
// TO BE FIXED
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

#ifdef _WIN32
void TimeInit() {}
#else
void TimeInit() {
  gettimeofday(&globalTimeVal, NULL);
  initTimeVal = globalTimeVal;
}
#endif

#ifdef _WIN32
void ReportTime() {}
#else
void ReportTime() {
  struct timeval newTimeValue, result;
  gettimeofday(&newTimeValue, NULL);
  timersub(&newTimeValue, &globalTimeVal, &result);
  printf("Time elapsed: %ld.%06ld ", (long int)result.tv_sec,
      (long int)result.tv_usec);
  timersub(&newTimeValue, &initTimeVal, &result);
  printf("accumulated: %ld.%06ld", (long int)result.tv_sec,
      (long int)result.tv_usec);
  globalTimeVal = newTimeValue;
}
#endif

#ifdef _WIN32
void ReportMemory() {}
#else
void ReportMemory() {
  char memCommand[200];
  mypid = getpid();
  sprintf(memCommand, "ps -o vsz --noheader -p %d", mypid);
  system(memCommand);
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

void OMInstrumentPoint(int64_t id, int64_t tag) {
  if (instrumentReportDisabled)
    return;

  // Print header
  printf("#%3d) %s op=%8s ", instrumentCounter,
      tag & (1 << (int)InstrumentBeforeOp) ? "before" : "after ", (char *)&id);
  instrumentCounter++;

  bool localReportTime =
      tag & (1 << (int)InstrumentReportTime) && !instrumentReportTimeDisabled;

  if (localReportTime) {
    ReportTime();
  }

  bool localReportMemory = tag & (1 << (int)InstrumentReportMemory) &&
                           !instrumentReportMemoryDisabled;
  if (localReportMemory) {
    ReportMemory();
  }
  printf("\n");
}
