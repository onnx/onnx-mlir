/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidcheck.h>

#include "llvm/Support/FileSystem.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

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
    bool isNoneP = false) {

  LSTMLibBuilder lstm(SHARED_LIB_BASE.str(), direction, S, B, I, H, isDynamicS,
      isDynamicB, isNoneH, isNoneC, isNoneP);
  return lstm.build() && lstm.compileAndLoad() && lstm.prepareInputs() &&
         lstm.run() && lstm.verifyOutputs();
}

} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestLSTM\n", nullptr, "TEST_ARGS");
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  // RapidCheck test case generation.
  bool success = rc::check("LSTM implementation correctness", []() {
    // The number of directions.
    // 1: forward, -1: reverse, 2: bidirectional
    const auto D = *rc::gen::element(1, -1, 2);
    // Sequence length.
    const auto S = *rc::gen::inRange(1, 5);
    // Batch size.
    const auto B = *rc::gen::inRange(5, 10);
    // Input size.
    const auto I = *rc::gen::inRange(5, 10);
    // Hidden size.
    const auto H = *rc::gen::inRange(5, 10);
    // Whether test dynamic dimension for sequence.
    const auto isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const auto isDynB = *rc::gen::element(0, 1);
    // Whether initial value of the hidden(initial_h) is specified.
    const auto isNoneH = *rc::gen::element(0, 1);
    // Whether initial value of the cell(initial_c) is specified.
    const auto isNoneC = *rc::gen::element(0, 1);
    // Whether the weight tensor for peepholes(P) is specified.
    const auto isNoneP = *rc::gen::element(0, 1);

    RC_ASSERT(isOMLSTMTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0,
        isDynB == 0, isNoneH == 0, isNoneC == 0, isNoneP == 0));
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
                  for (int64_t nonep = 0; nonep < 2; nonep++) {
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
