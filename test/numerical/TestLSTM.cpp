/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestLSTM.cpp - test matmul with broadcast -=========+++++++++++++====//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test LSTM code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestLSTM_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled LSTM is producing the same results as a
// naive implementation of LSTM for a specific set of LSTM
// parameters/configuration.
bool isOMLSTMTheSameAsNaiveImplFor(const int direction, const int S,
    const int B, const int I, const int H, bool isDynamicS = false,
    bool isDynamicB = false, bool isNoneH = false, bool isNoneC = false,
    bool isNoneP = false, int layout = 0) {

  static int testNum = 0;
  printf("attempt %d with direction %d, S %d, B %d, I %d, H %d, isDynS %d, "
         "isDynB %d, isNoneH %d, isNoneC %d, isNoneP %d, layout %d\n",
      ++testNum, direction, S, B, I, H, isDynamicS, isDynamicB, isNoneH,
      isNoneC, isNoneP, layout);
  LSTMLibBuilder lstm(SHARED_LIB_BASE.str(), direction, S, B, I, H, isDynamicS,
      isDynamicB, isNoneH, isNoneC, isNoneP, layout);
  return lstm.build() && lstm.compileAndLoad() &&
         lstm.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         lstm.prepareInputsFromEnv("TEST_DATARANGE") && lstm.run() &&
         lstm.verifyOutputs();
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
      argc, argv, "TestLSTM\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::string target = getCompilerOption(OptionKind::TargetAccel);
  std::cout << "Target options: \"" << target << "\"\n";
  // Set default configuration
  int minNoneP = 0; // Peephole is tested by default
  // Update configurations from an environment variable or target
  std::map<std::string, std::string> opts =
      ModelLibBuilder::getTestConfigFromEnv("TEST_CONFIG");
  if (target == "--maccel=NNPA" || opts["-peephole"] == "0") {
    std::cout << "Peephole: \"not tested\"" << std::endl;
    minNoneP = 1; // peephole not tested
  }

  // RapidCheck test case generation.
  bool success = rc::check("LSTM implementation correctness", [&]() {
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
    // Whether initial value of the hidden(initial_h) is specified.
    const int isNoneH = *rc::gen::inRange(0, 2);
    // Whether initial value of the cell(initial_c) is specified.
    const int isNoneC = *rc::gen::inRange(0, 2);
    // Whether the weight tensor for peepholes(P) is specified.
    const int isNoneP = *rc::gen::inRange(minNoneP, 2);

    RC_ASSERT(isOMLSTMTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0,
        isDynB == 0, isNoneH == 1, isNoneC == 1, isNoneP == 1, layout));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++)
          for (int64_t dyns = 0; dyns < 2; dyns++)
            for (int64_t dynb = 0; dynb < 2; dynb++)
              for (int64_t noneh = 0; noneh < 2; noneh++)
                for (int64_t nonec = 0; nonec < 2; nonec++)
                  for (int64_t nonep = minNoneP; nonep < 2; nonep++) {
                    // forward
                    assert(isOMLSTMTheSameAsNaiveImplFor(
                        1, s, b, i, h, dyns, dynb, noneh, nonec, nonep));
                    // reverse
                    assert(isOMLSTMTheSameAsNaiveImplFor(
                        -1, s, b, i, h, dyns, dynb, noneh, nonec, nonep));
                    // bidirectional
                    assert(isOMLSTMTheSameAsNaiveImplFor(
                        2, s, b, i, h, dyns, dynb, noneh, nonec, nonep));
                  }
  return 0;
}
