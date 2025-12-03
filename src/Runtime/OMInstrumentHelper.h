/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMInstrumentHelper.h - Helper for Instrumentation ----------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helpers for gathering timing, enabled by setting:
//
// #define OM_DRIVER_TIMING 1
// #include "src/Runtime/OMInstrumentHelper.h"
//
// When not defined, all macros are empty, aka generate no code, so no
// overheads.
//
// In Linux: uses gettimeofday and timersub.
//
// TIMING_INIT(var) defines a timing var (context must hold for all timing
//   operations).
// TIMING_START(var) starts the timer named var. TIMING_STOP(var)
//   adds to the timer named var the difference between now and the last start.
// TIMING_PRINT(var) prints the cumulative time of timer named var.
//
// TIMING_INIT_START does both init and start.
// TIMING_STOP_PRINT does both stop and print.
//
//===----------------------------------------------------------------------===//

#ifndef OM_INSTRUMENT_HELPER_H
#define OM_INSTRUMENT_HELPER_H 1

// Set to 1 to disable all timing regardless of other flags.
#define OM_DRIVER_TIMING_DISABLE_ALL 1 /* 1 (unless when debugging perf) */

//===----------------------------------------------------------------------===//
// Timing support for MVS

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

//===----------------------------------------------------------------------===//
// Timing functions

#if OM_DRIVER_TIMING && !OM_DRIVER_TIMING_DISABLE_ALL
#include <stdio.h>
#include <sys/time.h>

// Global variable to help OMInstrumentHelper.h to keep track of nesting level
// of timing operations.
extern int timing_nest_level;

#define TIMING_INIT(_var_name)                                                 \
  /* Define variable in current scope. */                                      \
  struct timeval _var_name, _var_name##_tmp;                                   \
  int _var_name##_nest_level = -1;                                             \
  _var_name.tv_sec = 0;                                                        \
  _var_name.tv_usec = 0;

#define TIMING_START(_var_name)                                                \
  _var_name##_nest_level = timing_nest_level;                                  \
  ++timing_nest_level;                                                         \
  gettimeofday(&_var_name##_tmp, NULL);

#define TIMING_STOP(_var_name)                                                 \
  { /* Define variables in their own scope */                                  \
    struct timeval start_time, stop_time, diff_time;                           \
    start_time = _var_name##_tmp;                                              \
    gettimeofday(&stop_time, NULL);                                            \
    timersub(&stop_time, &start_time, &diff_time);                             \
    _var_name.tv_sec += diff_time.tv_sec;                                      \
    _var_name.tv_usec += diff_time.tv_usec;                                    \
    --timing_nest_level;                                                       \
  }

#define TIMING_PRINT(_var_name)                                                \
  if (_var_name##_nest_level >= 0) { /* was started at least once */           \
    int l = _var_name##_nest_level;                                            \
    fprintf(stderr, "@OM_DRIVER, %*s%s, %ld.%06ld\n", l, " ", #_var_name,      \
        (long int)_var_name.tv_sec, (long int)_var_name.tv_usec);              \
  }

#else
#define TIMING_INIT(_var_name)
#define TIMING_START(_var_name)
#define TIMING_STOP(_var_name)
#define TIMING_PRINT(_var_name)
#endif

// Combined calls.
#define TIMING_INIT_START(_var_name)                                           \
  TIMING_INIT(_var_name) TIMING_START(_var_name)
#define TIMING_STOP_PRINT(_var_name)                                           \
  TIMING_STOP(_var_name) TIMING_PRINT(_var_name)

#endif
