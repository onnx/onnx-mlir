/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestSoftplus.cpp - test GEMM code -======================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test Softplus code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly surpress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestSoftplus_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

static bool isOMSoftplusTheSameAsNaiveImplFor(const int N) {
  static int testNum = 0;
  printf("attempt %d with N %d\n", ++testNum, N);

  SoftplusLibBuilder softplus( SHARED_LIB_BASE.str(), N);
  return softplus.build() && softplus.compileAndLoad() &&
         softplus.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         softplus.prepareInputsFromEnv("TEST_DATARANGE") && softplus.run() &&
         softplus.verifyOutputs();
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
      argc, argv, "TestSoftplus\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::string target = getCompilerOption(OptionKind::TargetAccel);
  std::cout << "Target options: \"" << target << "\"\n";
  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Softplus implementation correctness", [&]() {
      const int maxRange = 50;
      const int N = *rc::gen::inRange(1, maxRange);
      RC_ASSERT(isOMSoftplusTheSameAsNaiveImplFor(N));
    });
    if (!success)
      return 1;
  }
  return 0;
}
