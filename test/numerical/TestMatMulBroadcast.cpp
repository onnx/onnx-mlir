/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestMatMulBroadcast.cpp - test matmul with broadcast -===============//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test matrix multiply where one of the
// dimension is broadcasted to the other. Currently only test the case where
// only A or B is broadcasted, not both.
//
//===----------------------------------------------------------------------===//

#include <rapidcheck.h>

#include "llvm/Support/FileSystem.h"

#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE(
    "./TestMatmulBroadcast_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled Matmul is producing the same results
// as a naive implementation of Matmul for a specific set of Matmul
// parameters/configuration. Matmul: A[IxK] * B[KxJ] = C[IxJ].
// Include broadcasting in either A or B.
static bool isOMMatmulTheSameAsNaiveImplFor(bool broadcastingB,
    std::vector<int64_t> &broadcastDims, const int I, const int J,
    const int K) {
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d and broadcasting %c with rank %i "
         "and sizes (",
      ++testNum, I, J, K, broadcastingB ? 'B' : 'A', (int)broadcastDims.size());
  for (unsigned int i = 0; i < broadcastDims.size(); ++i)
    printf("%s%lld", i > 0 ? ", " : "", (long long int)broadcastDims[i]);
  printf(")\n");
  MatMulSingleBroadcastLibBuilder matmul(
      SHARED_LIB_BASE.str(), broadcastingB, broadcastDims, I, J, K);
  return matmul.build() && matmul.compileAndLoad() && matmul.prepareInputs() &&
         matmul.run() && matmul.verifyOutputs();
}
} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      onnx_mlir::getTargetFilename(SHARED_LIB_BASE.str(), onnx_mlir::EmitLib));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestMatMulBroadcast\n", nullptr, "TEST_ARGS");
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  bool success;
  printf("RapidCheck Matrix-Vector with broadcast test case generation.\n");
  success =
      rc::check("Matrix-Vector Matmul implementation correctness", []() {
        const int I = *rc::gen::inRange(4, 50);
        const int K = *rc::gen::inRange(4, 14);
        const int bRank = *rc::gen::inRange(1, 3);
        std::vector<int64_t> bDims;
        for (int i = 0; i < bRank; ++i)
          bDims.emplace_back(*rc::gen::inRange(1, 5));
        RC_ASSERT(isOMMatmulTheSameAsNaiveImplFor(
            /*broadcastB*/ true, bDims, I, 1, K));
      });
  if (!success)
    return 1;

  printf("RapidCheck Matrix-Matrix with broadcast test case generation.\n");
  success = rc::check("Matrix-Matrix Matmul implementation correctness", []() {
    const int I = *rc::gen::inRange(2, 20);
    const int J = *rc::gen::inRange(2, 20);
    const int K = *rc::gen::inRange(2, 8);
    const int bRank = *rc::gen::inRange(1, 4); // Additional rank for broadcast.
    std::vector<int64_t> bDims;
    for (int i = 0; i < bRank; ++i)
      bDims.emplace_back((int64_t)*rc::gen::inRange(1, 20));

    RC_ASSERT(
        isOMMatmulTheSameAsNaiveImplFor(/*broadcastB*/ true, bDims, I, J, K));
  });
  if (!success)
    return 1;

  return 0;
}
