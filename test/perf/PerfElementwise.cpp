/*
 * SPDX-License-Identifier: Apache-2.0
 */

//=================-- PerfElementwise.cpp - Simple performance tests
//-=========//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains tests for simple test cases for an arbitrary small
// set of parameters.
//   * Time is set to report in miliseconds (ms)
//   * Complexity is calculated in the original nanoseconds.
//   * Default opt level is O3, options found in PERF_ARGS override default.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

#include "include/OnnxMlirCompiler.h"
#include "test/modellib/ModelLib.hpp"
#include "test/perf/PerfHelper.hpp"

const std::string modelName("./perfelemenmtwise");

static void BM_Add(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXAddOp", 2, I, J);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 1.0 * I * J);
}
BENCHMARK(BM_Add)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

static void BM_HardSigmoid(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXHardSigmoidOp", 1, I, J);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 1.0 * I * J);
}
BENCHMARK(BM_HardSigmoid)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// Will set opt at -O3.
PERF_MAIN()
