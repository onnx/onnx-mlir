/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestScan.cpp - test Scan code -======================================//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test Scan code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestScan_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

void *omTensorGetAllocatedPtr(OMTensor *tensor);
template <typename TYPE>
void omPrintAsPython(OMTensor *tensor, std::string name) {
  int rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
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
    const int S, const int I, const int B = 1, const bool is_v8 = false) {

  static int testNum = 0;
  if (is_v8)
    printf("attempt %d with S=%d, I=%d, B=%d, is_v8=%d\n", ++testNum, S, I, B,
        is_v8);
  else
    printf("attempt %d with S=%d, I=%d, is_v8=%d\n", ++testNum, S, I, is_v8);

  ScanLibBuilder scan(SHARED_LIB_BASE.str(), S, I, B, is_v8);
  return scan.build() && scan.compileAndLoad() &&
         scan.prepareInputsFromEnv("TEST_DATARANGE") && scan.run() &&
         scan.verifyOutputs();
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
      argc, argv, "TestScan\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Scan implementation correctness", []() {
      const int maxRange = 50;
      const int S = *rc::gen::inRange(1, maxRange);
      const int I = *rc::gen::inRange(1, maxRange);
      RC_ASSERT(isOMScanTheSameAsNaiveImplFor(S, I));
    });
    if (!success)
      return 1;
  }
  return 0;
}
