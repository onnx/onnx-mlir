/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestGRU.cpp - test GRU code -========================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test GRU code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestGRU_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled GRU is producing the same results as a
// naive implementation of GRU for a specific set of GRU
// parameters/configuration.
bool isOMGRUTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, const int linearBeforeReset,
    bool isDynamicS = false, bool isDynamicB = false, int layout = 0) {

  static int testNum = 0;
  printf("attempt %d with direction %d, S %d, B %d, I %d, H %d, "
         "linearBeforeReset %d, isDynS %d, isDynB %d, layout %d\n",
      ++testNum, direction, S, B, I, H, linearBeforeReset, isDynamicS,
      isDynamicB, layout);
  GRULibBuilder gru(SHARED_LIB_BASE.str(), direction, S, B, I, H,
      linearBeforeReset, isDynamicS, isDynamicB, layout);
  return gru.build() && gru.compileAndLoad() &&
         gru.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         gru.prepareInputsFromEnv("TEST_DATARANGE") && gru.run() &&
         gru.verifyOutputs();
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
      argc, argv, "TestGRU\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::string target = getCompilerOption(OptionKind::TargetAccel);
  std::cout << "Target options: \"" << target << "\"\n";
  // Set default configurations
  int minL = 0; // Lower bound for L
  // Update configurations from an environment variable or target
  std::map<std::string, std::string> opts =
      ModelLibBuilder::getTestConfigFromEnv("TEST_CONFIG");
  if (target == "--maccel=NNPA" || opts["-linearBeforeReset"] == "1") {
    std::cout << "linear_before_reset: \"always set\"\n";
    minL = 1;
  }

  // RapidCheck test case generation.
  bool success = rc::check("GRU implementation correctness", [&]() {
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
    // LinearBeforeReset.
    const int L = *rc::gen::inRange(minL, 2);
    // Whether test dynamic dimension for sequence.
    const int isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const int isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(isOMGRUTheSameAsNaiveImplFor(
        D, S, B, I, H, L, isDynS == 0, isDynB == 0, layout));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++)
          for (int64_t l = minL; l < 2; l++) {
            // Static dimensions.
            // forward
            assert(isOMGRUTheSameAsNaiveImplFor(1, s, b, i, h, l));
            // reverse
            assert(isOMGRUTheSameAsNaiveImplFor(-1, s, b, i, h, l));
            // bidirectional
            assert(isOMGRUTheSameAsNaiveImplFor(2, s, b, i, h, l));
            // Dynamic dimensions for sequence, batch size.
            // forward
            assert(isOMGRUTheSameAsNaiveImplFor(1, s, b, i, h, l, true, true));
            // reverse
            assert(isOMGRUTheSameAsNaiveImplFor(-1, s, b, i, h, l, true, true));
            // bidirectional
            assert(isOMGRUTheSameAsNaiveImplFor(2, s, b, i, h, l, true, true));
          }
  return 0;
}
