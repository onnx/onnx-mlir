/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- PerfGemm.cpp - Simple performance tests -=============//
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

const std::string modelName("./perfgemm");

static void BM_MatrixVectorProduct(benchmark::State &state) {
  int I = state.range(0);
  int J = 1;
  int K = state.range(0);
  onnx_mlir::test::MatMul2DLibBuilder model(modelName, I, J, K);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed matmul");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 2.0 * I * J * K);
}
BENCHMARK(BM_MatrixVectorProduct)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

static void BM_MatmulSquare(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int K = state.range(0);
  onnx_mlir::test::MatMul2DLibBuilder model(modelName, I, J, K);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed matmul");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, 2.0 * I * J * K);
}
BENCHMARK(BM_MatmulSquare)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// Broadcast same matrices as above, but with 4x broadcast on B
static void BM_MatmulSquareBroadcastB4x(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int K = state.range(0);
  onnx_mlir::test::MatMulSingleBroadcastLibBuilder model(modelName,
      /*broadcast B*/ true, /*same static broadcast*/ false, {4}, I, J, K);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed matmul");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  perf_recordFlops(state, /*broadcast 4x*/ 4.0 /*matmul*/ * 2.0 * I * J * K);
}
BENCHMARK(BM_MatmulSquareBroadcastB4x)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

static void BM_MatMulWithGemmSquare(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int K = state.range(0);
  onnx_mlir::test::GemmLibBuilder model(
      modelName, I, J, K, false, false, 1, 1.0, 0.0);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed gemm");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  // Because alpha is 1, its not counted; beta is zero, sum of B is ignored.
  perf_recordFlops(state, 1.0 * I * J * (2.0 * K - 1.0));
}
BENCHMARK(BM_MatMulWithGemmSquare)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

static void BM_GemmSquare(benchmark::State &state) {
  int I = state.range(0);
  int J = state.range(0);
  int K = state.range(0);
  onnx_mlir::test::GemmLibBuilder model(
      modelName, I, J, K, false, false, 1, 1.0, 1.0);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed gemm");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(I);
  // Because alpha is 1, its not counted; beta is 1, sum of B is counted.
  perf_recordFlops(state, 1.0 * I * J * (2.0 * K - 1.0) + I * K);
}
BENCHMARK(BM_GemmSquare)
    ->RangeMultiplier(2)
    ->Range(16, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// Will set opt at -O3.
PERF_MAIN()
