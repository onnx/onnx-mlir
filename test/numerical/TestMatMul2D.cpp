/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <rapidcheck.h>
#include <string>

#include "llvm/Support/FileSystem.h"

#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestMatmul2D_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Returns whether onnx-mlir compiled Matmul is producing the same results
// as a naive implementation of Matmul for a specific set of Matmul
// parameters/configuration. Matmul: A[IxK] * B[KxJ] = C[IxJ]
static bool isOMMatmulTheSameAsNaiveImplFor(
    const int I, const int J, const int K) {
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d\n", ++testNum, I, J, K);
  MatMul2DLibBuilder matmul(SHARED_LIB_BASE.str(), I, J, K);
  return matmul.build() && matmul.compileAndLoad() && matmul.prepareInputs() &&
         matmul.run() && matmul.verifyOutputs();
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestMatMul2D\n", nullptr, "TEST_ARGS");

  printf("RapidCheck test case generation.\n");
  bool success = rc::check("Matmul implementation correctness", []() {
    const auto I = *rc::gen::inRange(1, 50);
    const auto J = *rc::gen::inRange(1, 50);
    const auto K = *rc::gen::inRange(1, 50);

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
