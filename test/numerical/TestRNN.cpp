/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidcheck.h>

#include "llvm/Support/FileSystem.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestRNN_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled RNN is producing the same results as a
// naive implementation of RNN for a specific set of RNN
// parameters/configuration.
bool isOMRNNTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, bool isDynamicS = false,
    bool isDynamicB = false) {

  RNNLibBuilder rnn(
      SHARED_LIB_BASE.str(), direction, S, B, I, H, isDynamicS, isDynamicB);
  return rnn.build() && rnn.compileAndLoad() && rnn.prepareInputs() &&
         rnn.run() && rnn.verifyOutputs();
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
      argc, argv, "TestRNN\n", nullptr, "TEST_ARGS");

  // RapidCheck test case generation.
  bool success = rc::check("RNN implementation correctness", []() {
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

    RC_ASSERT(
        isOMRNNTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0, isDynB == 0));
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
