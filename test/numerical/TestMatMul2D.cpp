/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestMastMul2D.cpp - test matmul without broadcast -==================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test 2D matrix multiply code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestMatmul2D_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled Matmul is producing the same results
// as a naive implementation of Matmul for a specific set of Matmul
// parameters/configuration. Matmul: A[IxK] * B[KxJ] = C[IxJ]
static bool isOMMatmulTheSameAsNaiveImplFor(
    const int I, const int J, const int K) {
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d\n", ++testNum, I, J, K);
  MatMul2DLibBuilder matmul(SHARED_LIB_BASE.str(), I, J, K);
  return matmul.build() && matmul.compileAndLoad() &&
         matmul.checkInstructionFromEnv("TEST_INSTRUCTION") &&
         matmul.prepareInputsFromEnv("TEST_DATARANGE") && matmul.run() &&
         matmul.verifyOutputs();
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
      argc, argv, "TestMatMul2D\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  printf("RapidCheck Matrix-Vector test case generation.\n");
  bool success =
      rc::check("Matrix-Vector Matmul implementation correctness", []() {
        const int I = *rc::gen::inRange(4, 50);
        const int K = *rc::gen::inRange(4, 14);

        RC_ASSERT(isOMMatmulTheSameAsNaiveImplFor(I, 1, K));
      });
  if (!success)
    return 1;

  printf("RapidCheck Matrix-Matrix test case generation.\n");
  success = rc::check("Matrix-Matrix Matmul implementation correctness", []() {
    const int I = *rc::gen::inRange(1, 50);
    const int J = *rc::gen::inRange(1, 50);
    const int K = *rc::gen::inRange(1, 50);

    RC_ASSERT(isOMMatmulTheSameAsNaiveImplFor(I, J, K));
  });
  if (!success)
    return 1;

  printf("\n\nExhaustive test case generation.\n");
  for (int I = 1; I < 9; I++)
    for (int J = 1; J < 9; J++)
      for (int K = 1; K < 9; K++)
        assert(isOMMatmulTheSameAsNaiveImplFor(I, J, K));

  return 0;
}
