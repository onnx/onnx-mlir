/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- PerfHelper.hpp - Helper for perf tests -=============//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper macro and functions for repetitive Benchmark
// actions.
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

void perf_recordFlops(benchmark::State &state, float f);
int perf_main(int argc, char **argv);

// Pass f as a (double) number of FLOP in the measurement and report it as the
// actual number (FLOP) and as a rate per seconds (FLOPS).
#define PERF_RECORD_FLOPS(_s, _f) perf_recordFlops(_s, _f)

// Define performance main, with default opt level of 3, and scan PERF_ARGS to
// override default onnx-mlir compiler options.
#define PERF_MAIN()                                                            \
  int main(int argc, char **argv) { return perf_main(argc, argv); }
