/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestUnique.cpp - test Unique code
//-======================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test Unique code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestUnique_main_graph");

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

// Returns whether onnx-mlir compiled Unique is producing the same results
// as a naive implementation of Unique for a specific set of Unique
// parameters/configuration.
  static bool isOMUniqueTheSameAsNaiveImplFor(const int rank, const int I,
      const int J, const int K, const int axis, const int sorted = 0,
      const int isNoneAxis = 0, const int isNoneIndexOutput = 0,
      const int useExample = 0) {

    UniqueLibBuilder unique(SHARED_LIB_BASE.str(), rank, I, J, K, axis,
        sorted, isNoneAxis, isNoneIndexOutput, useExample);
  return unique.build() && unique.compileAndLoad() &&
         unique.prepareInputs(0.0, 4.0) &&
         unique.run() && unique.verifyOutputs();
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
      argc, argv, "TestUnique\n", nullptr, "TEST_ARGS");
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  // test with existing example cases in the onnx github
  // Example 1: _not_sorted_without_axis
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/1, /*I=*/6, /*J=*/1, /*K=*/-1, /*axis=*/0, /*sorted=*/0,
      /*isNoneAxis=*/1, /*isNoneIndexOutput=*/0, /*example=*/1));
  // Example 2: _not_sorted_without_axis
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/2, /*I=*/2, /*J=*/2, /*K=*/-1, /*axis=*/0, /*sorted=*/1,
      /*isNoneAxis=*/1, /*isNoneIndexOutput=*/0, /*example=*/2));
  // Example 3: _sorted_with_axis
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/2, /*I=*/3, /*J=*/3, /*K=*/-1, /*axis=*/0, /*sorted=*/1,
      /*isNoneAxis=*/0, /*isNoneIndexOutput=*/0, /*example=*/3));
  // Example 4: _sorted_with_axis_3d
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/3, /*I=*/2, /*J=*/4, /*K=*/2, /*axis=*/1, /*sorted=*/1,
      /*isNoneAxis=*/0, /*isNoneIndexOutput=*/0, /*example=*/4));
  // Example 5: _sorted_with_negative_axis
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/2, /*I=*/3, /*J=*/3, /*K=*/-1, /*axis=*/-1, /*sorted=*/1,
      /*isNoneAxis=*/0, /*isNoneIndexOutput=*/0, /*example=*/5));
  return 0;
#if 0
  // Example 6: _sorted_with_axis_3d (data modified)
  RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
      /*rank=*/3, /*I=*/2, /*J=*/4, /*K=*/2, /*axis=*/1, /*sorted=*/1,
      /*isNoneAxis=*/0, /*isNoneIndexOutput=*/0, /*example=*/6));
#endif
  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Unique implementation correctness", []() {
      const int rank = 2; // *rc::gen::inRange(1, maxRank);
      const int I = 2;    // *rc::gen::inRange(1, maxRank);
      const int J = 3;    // *rc::gen::inRange(1, maxRank);
      const int K = -1;   // *rc::gen::inRange(1, maxRank);
      const int axis = *rc::gen::inRange(0, rank);
      const int sorted = *rc::gen::inRange(0, 2);
      const int isNoneAxis = *rc::gen::inRange(0, 2);
      const int isNoneIndexOutput = 0; // *rc::gen::inRange(0, 1);
      const int useExample = 0;
      RC_ASSERT(isOMUniqueTheSameAsNaiveImplFor(
          rank, I, J, K, axis, sorted, isNoneAxis, isNoneIndexOutput, useExample));
    });
    if (!success)
      return 1;
  }
  return 0;
}
