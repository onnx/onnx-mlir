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

// Pass f as a (double) number of FLOP in the measurement and report it as the
// actual number (FLOP) and as a rate per seconds (FLOPS).
#define PERF_RECORD_FLOPS(_f)                                                  \
  {                                                                            \
    state.counters["FLOPS"] = benchmark::Counter((_f),                         \
        benchmark::Counter::kIsRate |                                          \
            benchmark::Counter::kIsIterationInvariant,                         \
        benchmark::Counter::OneK::kIs1000);                                    \
    state.counters["FLOP"] = benchmark::Counter((_f),                          \
        benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1000);     \
  }

// Define performance main, with default opt level of 3, and scan PERF_ARGS to
// override default onnx-mlir compiler options.
#define PERF_MAIN()                                                            \
  int main(int argc, char **argv) {                                            \
    ::benchmark::Initialize(&argc, argv);                                      \
    int onnxMlirArgc = 2;                                                      \
    const char *onnxMlirArgv[onnxMlirArgc];                                    \
    onnxMlirArgv[0] = argv[0];                                                 \
    onnxMlirArgv[1] = "-O3";                                                   \
    if (!llvm::cl::ParseCommandLineOptions(onnxMlirArgc, onnxMlirArgv,         \
            "set options for perf-algo", nullptr, /*env var*/ "PERF_ARGS"))    \
      return 2;                                                                \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))                  \
      return 1;                                                                \
    ::benchmark::RunSpecifiedBenchmarks();                                     \
    ::benchmark::Shutdown();                                                   \
    return 0;                                                                  \
  }
