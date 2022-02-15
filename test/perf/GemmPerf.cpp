/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- GemmPerf.hpp - Simple performance tests -===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains tests for simple test cases, like for an arbitrary small
// set of parameters.
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>

#include "test/modellib/ModelLib.hpp"

using namespace std;

const std::string modelName("./gemmperf");
const CompilerOptionList opts{{onnx_mlir::OptionKind::CompilerOptLevel, "3"}};

static void BM_MatmulSquare(benchmark::State &state) {
  int IJK = state.range(0);
  MatMul2DLibBuilder model(modelName, IJK, IJK, IJK);
  assert(model.build() && model.compileAndLoad(opts) && model.prepareInputs() &&
         "failed matmul");

  for (auto _ : state)
    model.run();
  state.SetComplexityN(IJK);
}
BENCHMARK(BM_MatmulSquare)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

static void BM_GemmSquare(benchmark::State &state) {
  int IJK = state.range(0);
  GemmLibBuilder model(modelName, IJK, IJK, IJK, false, false, 1, 1.0, 1.0);
  assert(model.build() && model.compileAndLoad(opts) && model.prepareInputs() &&
         "failed gemm");

  for (auto _ : state)
    model.run();
  state.SetComplexityN(IJK);
}
BENCHMARK(BM_GemmSquare)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
