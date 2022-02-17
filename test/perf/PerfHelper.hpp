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
  const std::string envArgName("PERF_ARGS");                                   \
  const std::string O3("3");                                                   \
                                                                               \
  int main(int argc, char **argv) {                                            \
    ::benchmark::Initialize(&argc, argv);                                      \
    onnx_mlir::omSetCompilerOption(                                            \
        onnx_mlir::OptionKind::CompilerOptLevel, O3.c_str());                  \
    if (onnx_mlir::omSetCompilerOptionsFromEnv(envArgName.c_str()) != 0)       \
      return 2;                                                                \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))                  \
      return 1;                                                                \
    ::benchmark::RunSpecifiedBenchmarks();                                     \
    ::benchmark::Shutdown();                                                   \
    return 0;                                                                  \
  }
