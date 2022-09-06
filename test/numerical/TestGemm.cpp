/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <rapidcheck.h>
#include <string>

#include "llvm/Support/FileSystem.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestGemm_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

void *omTensorGetAllocatedPtr(OMTensor *tensor);
template <typename TYPE>
void omPrintAsPython(OMTensor *tensor, std::string name) {
  int rank = omTensorGetRank(tensor);
  int64_t *shape = omTensorGetShape(tensor);
  if (false) {
    printf("# tensor 0x%llx, allocated addr 0x%llx, data addr 0x%llx\n",
        (long long)tensor, (long long)omTensorGetAllocatedPtr(tensor),
        (long long)omTensorGetDataPtr(tensor));
  }
  if (rank == 2) {
    std::cout << name << " = np.array([";
    for (int64_t i = 0; i < shape[0]; ++i) {
      if (i)
        std::cout << ", ";
      std::cout << "[";
      for (int64_t j = 0; j < shape[1]; ++j) {
        if (j)
          std::cout << ", ";
        std::cout << omTensorGetElem<TYPE>(tensor, {i, j});
      }
      std::cout << "]";
    }
    std::cout << "])\n";
  }
}

// Returns whether onnx-mlir compiled Gemm is producing the same results
// as a naive implementation of Gemm for a specific set of Gemm
// parameters/configuration. Gemm: A[IxK] * B[KxJ] = C[IxJ]
static bool isOMGemmTheSameAsNaiveImplFor(const int I, const int J, const int K,
    const int aTrans, const int bTrans, const int cRank, const float alphaVal,
    const float betaVal) {

  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d%s%s, cRank %d, alpha %7.3f, beta "
         "%7.3f\n",
      ++testNum, I, J, K, (aTrans ? ", aTrans" : ""),
      (bTrans ? ", bTrans" : ""), cRank, (double)alphaVal, (double)betaVal);

  GemmLibBuilder gemm(
      SHARED_LIB_BASE.str(), I, J, K, aTrans, bTrans, cRank, alphaVal, betaVal);
  return gemm.build() && gemm.compileAndLoad() &&
         gemm.checkInstructionFromEnv("TestGemmNNPA_INSTRUCTION") &&
         gemm.prepareInputs() && gemm.run() && gemm.verifyOutputs();
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
      argc, argv, "TestGemm\n", nullptr, "TEST_ARGS");
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Gemm implementation correctness", []() {
      const int maxRange = 50;
      const auto I = *rc::gen::inRange(1, maxRange);
      const auto J = *rc::gen::inRange(1, maxRange);
      const auto K = *rc::gen::inRange(1, maxRange);
      const auto aTrans = *rc::gen::inRange(0, 2);
      const auto bTrans = *rc::gen::inRange(0, 2);
      const auto cRank = *rc::gen::inRange(1, 3);
#ifdef TEST_GEMM_ALPHA_BETA_1
      float alpha = 1.0;
      float beta = 1.0;
#else
      const auto hasAlpha = *rc::gen::inRange(0, 2);
      const auto hasBeta = *rc::gen::inRange(0, 2);
      float alpha = hasAlpha ? 1.2 : 1.0;
      float beta = hasBeta ? 0.8 : 1.0;
#endif
      RC_ASSERT(isOMGemmTheSameAsNaiveImplFor(
          I, J, K, aTrans, bTrans, cRank, alpha, beta));
    });
    if (!success)
      return 1;
  }

  if (false) {
    // Was too slow on some machines, disable test.
    printf("\n\nIndividual test case generation (benchmarks).\n");
#ifndef TEST_GEMM_ALPHA_BETA_1
    assert(isOMGemmTheSameAsNaiveImplFor(3, 5, 4, 0, 0, 2, 0.25, 0.35));
#endif
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 1024, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 2048, 0, 1, 2, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 25088, 0, 1, 1, 1.0, 1.0));
    // vcg 19
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 25088, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 4096, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 4096, 0, 1, 1, 1.0, 1.0));
  }
  return 0;
}
