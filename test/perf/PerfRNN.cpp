/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==================-- PerfRNN.cpp - Simple RNN performance tests -===========//
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

#include <cassert>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>

#include "include/OnnxMlirCompiler.h"
#include "test/modellib/ModelLib.hpp"
#include "test/perf/PerfHelper.hpp"

using namespace std;

const std::string modelName("./perfrnn");
const onnx_mlir::CompilerOptionList opts{
    {onnx_mlir::OptionKind::CompilerOptLevel, "3"}};

static void CommonArgs(benchmark::internal::Benchmark *b) {
  // clang-format off
  // Input and hidden sizes are divisible by 8.
  b->ArgsProduct({
    {1},                                            // unidirectional
    {10},                                           // timestep
    benchmark::CreateRange(512, 2048, /*multi=*/2), // batch size
    benchmark::CreateRange(128, 512, /*multi=*/2),  // input size
    benchmark::CreateRange(128, 512, /*multi=*/2)   // hidden size
  })
  // Input and hidden sizes are indivisible by 8.
  ->ArgsProduct({
      {1},                                            // unidirectional
      {10},                                           // timestep
      benchmark::CreateRange(512, 2048, /*multi=*/2), // batch size
      {100, 300, 500},                                // input size
      {100, 300, 500},                                // hidden size
  })
  // Some concrete examples.
  ->Args({1, 7, 2000, 200, 200})
  ->Args({1, 7, 2000, 204, 200});
  // clang-format on
}

static void BM_LSTM(benchmark::State &state) {
  int D = state.range(0);
  int S = state.range(1);
  int B = state.range(2);
  int I = state.range(3);
  int H = state.range(4);

  onnx_mlir::test::LSTMLibBuilder lstm(
      modelName, D, S, B, I, H, /*isDynamicS=*/false, /*isDynamicB=*/false);
  assert(lstm.build() && lstm.compileAndLoad(opts) && lstm.prepareInputs() &&
         "failed lstm");
  for (auto _ : state)
    lstm.run();
  // FLOPS for LSTM: ignore activations, assume static S and B.
  // Eight matrix-matrix multiplications are combined into two
  // matrix-matrix multiplications: [B,I]x[I,4*H] and [B,H]x[H,4*H].
  perf_recordFlops(state,
      D * S * (4.0 * B * H * (2.0 * I - 1.0) + 4.0 * B * H * (2.0 * H - 1.0)));
}
BENCHMARK(BM_LSTM)->Apply(CommonArgs)->Unit(benchmark::kMillisecond);

static void BM_GRU_LINEAR_BEFORE_RESET(benchmark::State &state) {
  int D = state.range(0);
  int S = state.range(1);
  int B = state.range(2);
  int I = state.range(3);
  int H = state.range(4);

  onnx_mlir::test::GRULibBuilder gru(modelName, D, S, B, I, H,
      /*linearBeforeReset=*/true,
      /*isDynamicS=*/false, /*isDynamicB=*/false);
  assert(gru.build() && gru.compileAndLoad(opts) && gru.prepareInputs() &&
         "failed gru");
  for (auto _ : state)
    gru.run();
  // FLOPS for GRU: ignore activations, assume static S and B.
  // Six matrix-matrix multiplications are combined into two
  // matrix-matrix multiplications: [B,I]x[I,3*H] and [B,H]x[H,3*H].
  perf_recordFlops(state,
      D * S * (3.0 * B * H * (2.0 * I - 1.0) + 3.0 * B * H * (2.0 * H - 1.0)));
}
BENCHMARK(BM_GRU_LINEAR_BEFORE_RESET)
    ->Apply(CommonArgs)
    ->Unit(benchmark::kMillisecond);

static void BM_GRU_LINEAR_AFTER_RESET(benchmark::State &state) {
  int D = state.range(0);
  int S = state.range(1);
  int B = state.range(2);
  int I = state.range(3);
  int H = state.range(4);

  onnx_mlir::test::GRULibBuilder gru(modelName, D, S, B, I, H,
      /*linearBeforeReset=*/false,
      /*isDynamicS=*/false, /*isDynamicB=*/false);
  assert(gru.build() && gru.compileAndLoad(opts) && gru.prepareInputs() &&
         "failed gru");
  for (auto _ : state)
    gru.run();
  // FLOPS for GRU: ignore activations, assume static S and B.
  // Six matrix-matrix multiplications are combined into two
  // matrix-matrix multiplications: [B,I]x[I,3*H] and [B,H]x[H,3*H].
  perf_recordFlops(state,
      D * S * (3.0 * B * H * (2.0 * I - 1.0) + 3.0 * B * H * (2.0 * H - 1.0)));
}
BENCHMARK(BM_GRU_LINEAR_AFTER_RESET)
    ->Apply(CommonArgs)
    ->Unit(benchmark::kMillisecond);

static void BM_RNN(benchmark::State &state) {
  int D = state.range(0);
  int S = state.range(1);
  int B = state.range(2);
  int I = state.range(3);
  int H = state.range(4);

  onnx_mlir::test::RNNLibBuilder rnn(modelName, D, S, B, I, H,
      /*isDynamicS=*/false,
      /*isDynamicB=*/false);
  assert(rnn.build() && rnn.compileAndLoad(opts) && rnn.prepareInputs() &&
         "failed rnn");
  for (auto _ : state)
    rnn.run();
  // FLOPS for RNN: ignore activations, assume static S and B.
  // Two matrix-matrix multiplications: [B,I]x[I,H] and [B,H]x[H,H].
  perf_recordFlops(
      state, D * S * (B * H * (2.0 * I - 1.0) + B * H * (2.0 * H - 1.0)));
}
BENCHMARK(BM_RNN)->Apply(CommonArgs)->Unit(benchmark::kMillisecond);

PERF_MAIN()
