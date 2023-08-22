/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- TestCategoryMapper.cpp - Test CategoryMapper Implementation -----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains end-to-end tests for the CategoryMapper operator.
//
//===----------------------------------------------------------------------===//

#include "test/modellib/ModelLib.hpp"
#include "llvm/Support/FileSystem.h"

static const std::string SharedLibBaseName("./TestCategoryMapper_main_graph");

static bool testInt64ToStr(int inputRank) {
  using CategoryMapperBuilder =
      onnx_mlir::test::CategoryMapperLibBuilder<int64_t, const char *>;

  const CategoryMapperBuilder::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};
  const llvm::SmallVector<int64_t, 6> input = {1, 2, 3, 6, 4, 5};
  const llvm::SmallVector<const char *, 6> expResult = {
      "cat", "dog", "human", "unknown", "tiger", "beaver"};
  CategoryMapperBuilder categoryMapper(
      SharedLibBaseName, attributes, input, expResult, inputRank);
  bool passed = categoryMapper.build() && categoryMapper.compileAndLoad() &&
                categoryMapper.prepareInputs() && categoryMapper.run() &&
                categoryMapper.verifyOutputs();
  if (!passed)
    llvm::errs() << __func__ << " failed\n";

  return passed;
}

static bool testStrToInt64(int inputRank) {
  using CategoryMapperBuilder =
      onnx_mlir::test::CategoryMapperLibBuilder<const char *, int64_t>;

  const CategoryMapperBuilder::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};
  const llvm::SmallVector<const char *, 6> input = {
      "dog", "human", "cat", "beaver", "tiger", "bird"};
  const llvm::SmallVector<int64_t, 6> expResult = {2, 3, 1, 5, 4, -1};

  CategoryMapperBuilder categoryMapper(
      SharedLibBaseName, attributes, input, expResult, inputRank);

  bool passed = categoryMapper.build() && categoryMapper.compileAndLoad() &&
                categoryMapper.prepareInputs() && categoryMapper.run() &&
                categoryMapper.verifyOutputs();
  if (!passed)
    llvm::errs() << __func__ << " failed\n";

  return passed;
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      onnx_mlir::getTargetFilename(SharedLibBaseName, onnx_mlir::EmitLib));

  setCompilerOption(onnx_mlir::OptionKind::CompilerOptLevel, "3");
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestCategoryMapper\n", nullptr, "TEST_ARGS");
  onnx_mlir::initCompilerConfig();
  std::cout << "Target options: \""
            << onnx_mlir::getCompilerOption(onnx_mlir::OptionKind::TargetAccel)
            << "\"\n";

  bool rc = testInt64ToStr(/*inputRank=*/1);
  rc &= testStrToInt64(/*inputRank=*/1);
  rc &= testInt64ToStr(/*inputRank=*/2);
  rc &= testStrToInt64(/*inputRank=*/2);
  rc &= testInt64ToStr(/*inputRank=*/3);
  rc &= testStrToInt64(/*inputRank=*/3);
  return rc ? 0 : 1;
}
