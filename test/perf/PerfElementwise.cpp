/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===============-- PerfElementwise.cpp - Simple performance tests -==========//
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

//===----------------------------------------------------------------------===//
// Nice SIMD opportunities for backend compiler by having large tiles in both
// dimensions.
//===----------------------------------------------------------------------===//

static void BM_Add_nice(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXAddOp", I, J);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 1.0 * I * J); // Add.
}
BENCHMARK(BM_Add_nice)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

//===----------------------------------------------------------------------===//
// Harder SIMD opportunities for backend compiler by having small tiles in the
// last dimension.
//===----------------------------------------------------------------------===//

static void BM_Add_harder(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int tot = I * J;
  int inner = 7;
  int outer = tot / inner;
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXAddOp", outer, inner);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 1.0 * outer * inner); // Add.
}
BENCHMARK(BM_Add_harder)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

//===----------------------------------------------------------------------===//
// Nice SIMD opportunities for backend compiler by having large tiles in both
// dimensions.
//===----------------------------------------------------------------------===//

static void BM_HardSigmoid_nice(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXHardSigmoidOp", I, J);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 4.0 * I * J); // FMA plus 2 float compare.
}
BENCHMARK(BM_HardSigmoid_nice)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

//===----------------------------------------------------------------------===//
// Harder SIMD opportunities for backend compiler by having small tiles in the
// last dimension.
//===----------------------------------------------------------------------===//

static void BM_HardSigmoid_harder(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int tot = I * J;
  int inner = 7;
  int outer = tot / inner;
  onnx_mlir::test::Elementwise2DLibBuilder model(
      modelName, "ONNXHardSigmoidOp", outer, inner);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed elementwise add");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 4.0 * outer * inner); // FMA plus 2 float compare.
}
BENCHMARK(BM_HardSigmoid_harder)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// Will set opt at -O3.
PERF_MAIN()
