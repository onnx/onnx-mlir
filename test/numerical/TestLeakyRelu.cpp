/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestLeakyRelu.cpp - test GEMM code -======================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test LeakyRelu code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly surpress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestLeakyRelu_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

static bool isOMLeakyReluTheSameAsNaiveImplFor(const int N, const float alphaVal) {
  static int testNum = 0;
  printf("attempt %d with N %d, alpha %7.3f\n", ++testNum, N, (double)alphaVal);

  LeakyReluLibBuilder leakyRelu( SHARED_LIB_BASE.str(), N, alphaVal);
  return leakyRelu.build() && leakyRelu.compileAndLoad() &&
         leakyRelu.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         leakyRelu.prepareInputsFromEnv("TEST_DATARANGE") && leakyRelu.run() &&
         leakyRelu.verifyOutputs();
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
      argc, argv, "TestLeakyRelu\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::string target = getCompilerOption(OptionKind::TargetAccel);
  std::cout << "Target options: \"" << target << "\"\n";
  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("LeakyRelu implementation correctness", [&]() {
      const int maxRange = 50;
      const int N = *rc::gen::inRange(1, maxRange);
      float alpha = *rc::gen::inRange(-10, 10) / 10.0;
      RC_ASSERT(isOMLeakyReluTheSameAsNaiveImplFor(N, alpha));
    });
    if (!success)
      return 1;
  }
  return 0;
}
