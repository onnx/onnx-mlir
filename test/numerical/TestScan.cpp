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

static const llvm::StringRef SHARED_LIB_BASE("./TestScan_main_graph");

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

// Returns whether onnx-mlir compiled Scan is producing the same results
// as a naive implementation of Scan for a specific set of Scan
// parameters/configuration.
static bool isOMScanTheSameAsNaiveImplFor(
    const int B, const int S, const int I) {

  static int testNum = 0;
  printf("attempt %d with B=%d, S=%d, I=%d\n", ++testNum, B, S, I);

  ScanLibBuilder scan(SHARED_LIB_BASE.str(), B, S, I);
  return scan.build() && scan.compileAndLoad() && scan.prepareInputs() &&
         scan.run() && scan.verifyOutputs();
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
      argc, argv, "TestScan\n", nullptr, "TEST_ARGS");
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Scan implementation correctness", []() {
      const int maxRange = 50;
#if 0
      const auto B = *rc::gen::inRange(1, maxRange);
#else
      const auto B = *rc::gen::inRange(1, 2);
#endif
      const auto S = *rc::gen::inRange(1, maxRange);
      const auto I = *rc::gen::inRange(1, maxRange);
      RC_ASSERT(isOMScanTheSameAsNaiveImplFor(B, S, I));
    });
    if (!success)
      return 1;
  }
  return 0;
}
