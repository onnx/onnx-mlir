/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestElementwise.cpp - test elementwise code =========================//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test some elementwise code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly surpress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestElementwise_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

void *omTensorGetAllocatedPtr(OMTensor *tensor);

// Returns whether onnx-mlir compiled Gemm is producing the same results
// as a naive implementation of Gemm for a specific set of Gemm
// parameters/configuration. Gemm: A[IxK] * B[KxJ] = C[IxJ]
static bool isOMElementwiseTheSameAsNaiveImplFor(
    const std::string &elementwiseOpName, const int I, const int J) {

  static int testNum = 0;
  printf("attempt %d %s with i %d, j %d\n", ++testNum,
      elementwiseOpName.c_str(), I, J);

  Elementwise2DLibBuilder elementwise(
      SHARED_LIB_BASE.str(), elementwiseOpName, I, J);
  return elementwise.build() && elementwise.compileAndLoad() &&
         elementwise.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         elementwise.prepareInputsFromEnv("TEST_DATARANGE") &&
         elementwise.run() && elementwise.verifyOutputs();
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
      argc, argv, "TestElementwise\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::string target = getCompilerOption(OptionKind::TargetAccel);
  std::cout << "Target options: \"" << target << "\"\n";
  bool success;

  printf("RapidCheck test Add case generation.\n");
  success = rc::check("Gemm implementation correctness", [&]() {
    const int maxRange = 128;
    const int I = *rc::gen::inRange(1, maxRange);
    const int J = *rc::gen::inRange(1, maxRange);
    RC_ASSERT(isOMElementwiseTheSameAsNaiveImplFor("ONNXAddOp", I, J));
  });
  if (!success)
    return 1;

  printf("RapidCheck test DIV case generation.\n");
  success = rc::check("Gemm implementation correctness", [&]() {
    const int maxRange = 128;
    const int I = *rc::gen::inRange(1, maxRange);
    const int J = *rc::gen::inRange(1, maxRange);
    RC_ASSERT(isOMElementwiseTheSameAsNaiveImplFor("ONNXDivOp", I, J));
  });
  if (!success)
    return 1;

  printf("RapidCheck test HardSigmoid case generation.\n");
  success = rc::check("Gemm implementation correctness", [&]() {
    const int maxRange = 128;
    const int I = *rc::gen::inRange(1, maxRange);
    const int J = *rc::gen::inRange(1, maxRange);
    RC_ASSERT(isOMElementwiseTheSameAsNaiveImplFor("ONNXHardSigmoidOp", I, J));
  });
  if (!success)
    return 1;

  printf("RapidCheck test Erf case generation.\n");
  success = rc::check("Gemm implementation correctness", [&]() {
    const int maxRange = 128;
    const int I = *rc::gen::inRange(1, maxRange);
    const int J = *rc::gen::inRange(1, maxRange);
    RC_ASSERT(isOMElementwiseTheSameAsNaiveImplFor("ONNXErfOp", I, J));
  });
  if (!success)
    return 1;

  return 0;
}
