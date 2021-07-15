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

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "onnx-mlir/Runtime/OMInstrument.h"

#ifdef _WIN32
// TO BE FIXED
#else
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

static struct timeval globalTimeVal, initTimeVal;
static pid_t mypid;
#endif

static bool timeEnabled = true;
static bool virtualMemEnabled = true;

#ifdef _WIN32
void TimeInit() {}
#else
void TimeInit() {
  gettimeofday(&globalTimeVal, NULL);
  initTimeVal = globalTimeVal;
}
#endif

#ifdef __WIN32
void ReportTime() {}
#else
void ReportTime() {
  struct timeval newTimeValue, result;
  gettimeofday(&newTimeValue, NULL);
  timersub(&newTimeValue, &globalTimeVal, &result);
  printf("Time elapsed: %ld.%06ld ", (long int)result.tv_sec,
      (long int)result.tv_usec);
  timersub(&newTimeValue, &initTimeVal, &result);
  printf("accumulated: %ld.%06ld\n", (long int)result.tv_sec,
      (long int)result.tv_usec);
  globalTimeVal = newTimeValue;
}
#endif

#ifdef __WIN32
void ReportMemory() {}
#else
void ReportMemory() {
  char memCommand[200];
  mypid = getpid();
  sprintf(memCommand, "ps -o vsz --noheader -p %d", mypid);
  system(memCommand);
}
#endif

void OMInstrumentInit() {
  if (getenv("OMINSTRUMENTTIME")) {
    timeEnabled = !timeEnabled;
  }
  if (getenv("OMINSTRUMENTMEMORY")) {
    virtualMemEnabled = !virtualMemEnabled;
  }

  if (timeEnabled) {
    TimeInit();
  }
}

void OMInstrumentPoint(int64_t id, int64_t tag) {
  // Print header
  if (timeEnabled || virtualMemEnabled)
    printf("ID=%s TAG=%ld ", (char *)&id, tag);

  if (timeEnabled) {
    ReportTime();
  }

  if (virtualMemEnabled) {
    ReportMemory();
  }
}
