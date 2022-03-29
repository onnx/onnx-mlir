/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <rapidcheck.h>
#include <string>

#include "llvm/Support/FileSystem.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestGemm_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

void *omTensorGetAllocatedPtr(OMTensor *tensor);
template <typename TYPE>
void omPrintAsPython(OMTensor *tensor, string name) {
  int rank = omTensorGetRank(tensor);
  int64_t *shape = omTensorGetShape(tensor);
  if (false) {
    printf("# tensor 0x%llx, allocated addr 0x%llx, data addr 0x%llx\n",
        (long long)tensor, (long long)omTensorGetAllocatedPtr(tensor),
        (long long)omTensorGetDataPtr(tensor));
  }
  if (rank == 2) {
    cout << name << " = np.array([";
    for (int64_t i = 0; i < shape[0]; ++i) {
      if (i)
        cout << ", ";
      cout << "[";
      for (int64_t j = 0; j < shape[1]; ++j) {
        if (j)
          cout << ", ";
        cout << omTensorGetElem<TYPE>(tensor, {i, j});
      }
      cout << "]";
    }
    cout << "])\n";
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
  return gemm.build() && gemm.compileAndLoad() && gemm.prepareInputs() &&
         gemm.run() && gemm.verifyOutputs();
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestGemm\n", nullptr, "TEST_ARGS");

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
      const auto hasAlpha = *rc::gen::inRange(0, 2);
      const auto hasBeta = *rc::gen::inRange(0, 2);
      float alpha = hasAlpha ? 1.2 : 1.0;
      float beta = hasBeta ? 0.8 : 1.0;
      RC_ASSERT(isOMGemmTheSameAsNaiveImplFor(
          I, J, K, aTrans, bTrans, cRank, alpha, beta));
    });
    if (!success)
      return 1;
  }

  if (false) {
    // Was too slow on some machines, disable test.
    printf("\n\nIndividual test case generation (benchmarks).\n");
    assert(isOMGemmTheSameAsNaiveImplFor(3, 5, 4, 0, 0, 2, 0.25, 0.35));

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
