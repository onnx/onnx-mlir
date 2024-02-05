/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestRNN.cpp - test RNN code -========================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test RNN code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestRNN_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled RNN is producing the same results as a
// naive implementation of RNN for a specific set of RNN
// parameters/configuration.
bool isOMRNNTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, bool isDynamicS = false, bool isDynamicB = false,
    int layout = 0) {

  RNNLibBuilder rnn(
      SHARED_LIB_BASE.str(), direction, S, B, I, H, isDynamicS, isDynamicB);
  return rnn.build() && rnn.compileAndLoad() &&
         rnn.prepareInputsFromEnv("TEST_DATARANGE") &&
         rnn.checkInstructionFromEnv("TEST_INSTRUCTION") && rnn.run() &&
         rnn.verifyOutputs();
}

} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      onnx_mlir::getTargetFilename(SHARED_LIB_BASE.str(), onnx_mlir::EmitLib));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestRNN\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  // RapidCheck test case generation.
  bool success = rc::check("RNN implementation correctness", []() {
    // The number of directions.
    // 1: forward, -1: reverse, 2: bidirectional
    const int D = *rc::gen::element(1, -1, 2);
    // Sequence length.
    const int S = *rc::gen::inRange(1, 5);
    // Batch size.
    const int B = *rc::gen::inRange(5, 10);
    // Input size.
    const int I = *rc::gen::inRange(5, 10);
    // Hidden size.
    const int H = *rc::gen::inRange(5, 10);
    // Layout.
    const int layout = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for sequence.
    const int isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const int isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(isOMRNNTheSameAsNaiveImplFor(
        D, S, B, I, H, isDynS == 0, isDynB == 0, layout));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++) {
          // Static dimensions.
          // forward
          assert(isOMRNNTheSameAsNaiveImplFor(1, s, b, i, h));
          // reverse
          assert(isOMRNNTheSameAsNaiveImplFor(-1, s, b, i, h));
          // bidirectional
          assert(isOMRNNTheSameAsNaiveImplFor(2, s, b, i, h));

          // Dynamic dimensions for sequence, batch size.
          // forward
          assert(isOMRNNTheSameAsNaiveImplFor(1, s, b, i, h, true, true));
          // reverse
          assert(isOMRNNTheSameAsNaiveImplFor(-1, s, b, i, h, true, true));
          // bidirectional
          assert(isOMRNNTheSameAsNaiveImplFor(2, s, b, i, h, true, true));
        }
  return 0;
}
