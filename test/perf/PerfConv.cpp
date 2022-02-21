/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==================-- PerfConv.hpp - Simple Conv performance tests -=========//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains tests for simple test cases,  for an arbitrary small set
// of parameters.
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>

#include "test/modellib/ModelLib.hpp"

using namespace std;

const std::string modelName("./perfconv");
const CompilerOptionList opts{{onnx_mlir::OptionKind::CompilerOptLevel, "3"}};

static void BM_Conv2D_C16_K3(benchmark::State &state) {
  int N = state.range(0);
  int C = 16;
  int HW = state.range(1);
  int K = 3;
  int P = 0;
  int S = 1;
  int D = 1;
  Conv2DLibBuilder model(
      modelName, N, C, HW, HW, K, K, AUTO_PAD_VALID, P, P, P, P, S, D, false);
  assert(model.build() && model.compileAndLoad(opts) && model.prepareInputs() &&
         "failed conv");
  for (auto _ : state)
    model.run();
  state.SetComplexityN(N * (C) * (HW) * (K));
}
BENCHMARK(BM_Conv2D_C16_K3)
    ->ArgsProduct({{1, 16, 64}, {16, 64, 256}})
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
