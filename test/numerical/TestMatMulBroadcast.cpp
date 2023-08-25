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

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

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
    bool sameStaticBroadcast, std::vector<int64_t> &broadcastDims, const int I,
    const int J, const int K) {
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d and broadcasting %s with rank %i "
         "and sizes (",
      ++testNum, I, J, K,
      sameStaticBroadcast ? "both" : (broadcastingB ? "B" : "A"),
      (int)broadcastDims.size());
  for (unsigned int i = 0; i < broadcastDims.size(); ++i)
    printf("%s%lld", i > 0 ? ", " : "", (long long int)broadcastDims[i]);
  printf(")\n");
  MatMulSingleBroadcastLibBuilder matmul(SHARED_LIB_BASE.str(), broadcastingB,
      sameStaticBroadcast, broadcastDims, I, J, K);
  return matmul.build() && matmul.compileAndLoad() && matmul.prepareInputs() &&
         matmul.run() && matmul.verifyOutputs();
}
} // namespace test
} // namespace onnx_mlir

bool broadcastB;          // Indicates if we broadcast on B (true) or A (false).
bool sameStaticBroadcast; // Indicates if both A and B have the same static
                          // broadcast dims.
int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      onnx_mlir::getTargetFilename(SHARED_LIB_BASE.str(), onnx_mlir::EmitLib));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestMatMulBroadcast\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  bool success;
  for (int i = 0; i < 3; ++i) {
    // Define same static, broadcast A, or broadcast B.
    broadcastB = sameStaticBroadcast = false;
    if (i == 0)
      sameStaticBroadcast = true; // Same static broadcast.
    else if (i == 1)              // Broadcast A.
      broadcastB = false;
    else if (i == 2) // Broadcast B.
      broadcastB = true;
    printf("RapidCheck Matrix-vector/Matrix with broadcast %s test case "
           "generation.\n",
        sameStaticBroadcast ? "same static" : (broadcastB ? "B" : "A"));
    success = rc::check(
        "Matrix-vector/Matrix Matmul implementation correctness", []() {
          // There is some tiling up to 64, so pick up a large UB number .
          const int I = *rc::gen::inRange(2, 80);
          const int J = *rc::gen::inRange(1, 80); // From 1 for mat-vect.
          const int K = *rc::gen::inRange(2, 80);
          const int broadcastRank =
              *rc::gen::inRange(1, 5); // Additional rank for broadcast.
          std::vector<int64_t> broadcastDims;
          for (int i = 0; i < broadcastRank; ++i)
            // No need for large dims, as broadcast of 2 is enough to uncover
            // bugs.
            broadcastDims.emplace_back((int64_t)*rc::gen::inRange(2, 4));

          RC_ASSERT(isOMMatmulTheSameAsNaiveImplFor(
              broadcastB, sameStaticBroadcast, broadcastDims, I, J, K));
        });
    if (!success)
      return 1;
  }

  return 0;
}
