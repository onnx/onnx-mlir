/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "llvm/Support/FileSystem.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestMatmul2D_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

// Returns whether onnx-mlir compiled Matmul is producing the same results
// as a naive implementation of Matmul for a specific set of Matmul
// parameters/configuration. Matmul: A[IxK] * B[KxJ] = C[IxJ]
bool isOMMatmulTheSameAsNaiveImplFor(const int I, const int J, const int K) {
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d\n", ++testNum, I, J, K);
  if (!genMatMul2DModelAndCompile(
          /*compiler options */
          SHARED_LIB_BASE.str(),
          /* GEMM param in*/
          I, J, K))
    return false;

  onnx_mlir::ExecutionSession sess(getSharedLibName(SHARED_LIB_BASE.str()));

  std::vector<OMTensorUniquePtr> inputs;
  auto aOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>({I, K}), omTensorDestroy);
  inputs.emplace_back(move(aOmt));
  auto bOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>({K, J}), omTensorDestroy);
  inputs.emplace_back(move(bOmt));

  auto ref = omTensorCreateWithShape<float>({I, J});
  auto &a = inputs.at(0);
  auto &b = inputs.at(1);
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(ref, {i, j}) = 0;
      for (int64_t k = 0; k < K; k++) {
        omTensorGetElem<float>(ref, {i, j}) +=
            omTensorGetElem<float>(a.get(), {i, k}) *
            omTensorGetElem<float>(b.get(), {k, j});
      }
    }
  }

  auto outputs = sess.run(move(inputs));
  auto &Matmul = outputs.at(0);

  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  return omTensorAreTwoOmtsClose<float>(Matmul.get(), ref, rtol, atol);
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE.str()));

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
