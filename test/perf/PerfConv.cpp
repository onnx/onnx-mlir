/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==================-- PerfConv.cpp - Simple Conv performance tests -=========//
//
// Copyright 2022 The IBM Research Authors.
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

const std::string modelName("./perfconv");
const onnx_mlir::CompilerOptionList opts{
    {onnx_mlir::OptionKind::CompilerOptLevel, "3"}};

static void BM_Conv2D_C16_K1(benchmark::State &state) {
  int N = state.range(0);
  int C = 16;
  int H = state.range(1);
  int W = state.range(1);
  int K = 1;
  int P = 0;
  int S = 1;
  int D = 1;
  onnx_mlir::test::Conv2DLibBuilder model(modelName, N, C, C, H, W, K, K,
      onnx_mlir::test::ConvAutoPad::VALID, P, P, P, P, S, D, false);
  assert(model.build() && model.compileAndLoad(opts) && model.prepareInputs() &&
         "failed conv");
  for (auto _ : state)
    model.run();
  // FLOPS assume D=1, S=1.
  perf_recordFlops(state, 2.0 * N * C * C * H * W * K * K);
}
BENCHMARK(BM_Conv2D_C16_K1)
    ->ArgsProduct({{1, 16, 64}, {16, 64, 256}})
    ->Unit(benchmark::kMillisecond);

static void BM_Conv2D_C16_K3(benchmark::State &state) {
  int N = state.range(0);
  int C = 16;
  int H = state.range(1);
  int W = state.range(1);
  int K = 3;
  int P = 0;
  int S = 1;
  int D = 1;
  onnx_mlir::test::Conv2DLibBuilder model(modelName, N, C, C, H, W, K, K,
      onnx_mlir::test::ConvAutoPad::VALID, P, P, P, P, S, D, false);
  assert(model.build() && model.compileAndLoad(opts) && model.prepareInputs() &&
         "failed conv");
  for (auto _ : state)
    model.run();
  // FLOPS assume D=1, S=1.
  perf_recordFlops(state, 2.0 * N * C * C * H * W * K * K);
}
BENCHMARK(BM_Conv2D_C16_K3)
    ->ArgsProduct({{1, 16, 64}, {16, 64, 256}})
    ->Unit(benchmark::kMillisecond);

PERF_MAIN()
