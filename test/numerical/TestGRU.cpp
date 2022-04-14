/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidcheck.h>

#include "llvm/Support/FileSystem.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestGRU_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled GRU is producing the same results as a
// naive implementation of GRU for a specific set of GRU
// parameters/configuration.
bool isOMGRUTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, const int linearBeforeReset,
    bool isDynamicS = false, bool isDynamicB = false) {

  GRULibBuilder gru(SHARED_LIB_BASE.str(), direction, S, B, I, H,
      linearBeforeReset, isDynamicS, isDynamicB);
  return gru.build() && gru.compileAndLoad() && gru.prepareInputs() &&
         gru.run() && gru.verifyOutputs();
}

} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  setCompilerOptions({{OptionKind::CompilerOptLevel, "3"}});
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestGRU\n", nullptr, "TEST_ARGS");

  // RapidCheck test case generation.
  bool success = rc::check("GRU implementation correctness", []() {
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
    // LinearBeforeReset.
    const auto L = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for sequence.
    const auto isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const auto isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(isOMGRUTheSameAsNaiveImplFor(
        D, S, B, I, H, L, isDynS == 0, isDynB == 0));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++)
          for (int64_t l = 0; l < 2; l++) {
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
