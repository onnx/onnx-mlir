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

#include "onnx-mlir/Runtime/OMInstrument.h"

#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

static struct timeval globalTimeVal, initTimeVal;
static bool timeEnabled = true;
static bool virtualMemEnabled = true;
static pid_t mypid;
static char memCommand[200];

void OMInstrumentInit() {
  if (getenv("OMINSTRUMENTTIME")) {
	  timeEnabled = !timeEnabled;
  }
  if (getenv("OMINSTRUMENTTIME")) {
	  virtualMemEnabled = !virtualMemEnabled;
  }

  if (timeEnabled) {
    gettimeofday(&globalTimeVal, NULL);
    initTimeVal = globalTimeVal;
  }

  if (virtualMemEnabled) {
     mypid = getpid();
     sprintf(memCommand, "ps -o vsz --noheader -p %d", mypid);
  }
}

void OMInstrumentPoint(int64_t id, int64_t tag) {
  if (timeEnabled) {
    struct timeval newTimeValue, result;
    gettimeofday(&newTimeValue, NULL);
    printf("ID=%s TAG=%ld ",(char *)&id, tag);
    timersub(&newTimeValue, &globalTimeVal, &result);
    printf("Time elapsed: %ld.%06ld ", (long int)result.tv_sec, (long int)result.tv_usec);
    timersub(&newTimeValue, &initTimeVal, &result);
    printf("accumulated: %ld.%06ld\n", (long int)result.tv_sec, (long int)result.tv_usec);
    globalTimeVal = newTimeValue;
  }

  if (virtualMemEnabled) {
    system(memCommand);
  }

}

