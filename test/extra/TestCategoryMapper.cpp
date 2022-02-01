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

#include "test/extra/TestHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tests"

using namespace onnx_mlir;

static const string SharedLibBase("./TestCategoryMapper_main_graph");

namespace {

class CategoryMapperTester {
  using TestHelper = Tests::TestHelper;
  TestHelper testHelper;

public:
  // Aggregate CategoryMapper attributes.
  struct CMAttributes {
    ArrayRef<int64_t> cat_int64s;
    ArrayRef<StringRef> cat_strings;
    int64_t default_int;
    StringRef default_string;
  };

  CategoryMapperTester(MLIRContext &ctx) : testHelper(TestHelper(ctx)) {}

  // Test CategoryMapper (with an input tensor of int64_t numbers).
  bool testInt64ToStr(const CMAttributes &attributes, ArrayRef<int64_t> input,
      ArrayRef<const char *> expectedOutput) {
    assert(input.size() == expectedOutput.size() &&
           "Expecting input/output to have the same size");

    int64_t inputShape[] = {static_cast<int64_t>(input.size())};
    auto inputType =
        RankedTensorType::get(inputShape, testHelper.getBuilder().getI64Type());
    auto outputType = RankedTensorType::get(
        inputShape, onnxmlir::StringType::get(&testHelper.getContext()));

    // Create the test code.
    FuncOp funcOp = testHelper.createEmptyTestFunction(inputType, outputType);
    createCategoryMapper(outputType, attributes, funcOp);
    testHelper.createEntryPoint(funcOp);

    // Compile the test.
    if (!testHelper.compileTest(SharedLibBase)) {
      llvm::errs() << "Failed to compile test case\n";
      return false;
    }

    // Run the test and verify the result.
    using OMTensorPtr = unique_ptr<OMTensor, decltype(&omTensorDestroy)>;

    std::vector<OMTensorPtr> inputOMTs, expectedOutputOMTs;
    auto inputOMT = OMTensorPtr(omTensorCreate((void *)input.data(), inputShape,
                                    1 /*rank*/, ONNX_TYPE_INT64),
        omTensorDestroy);
    auto expectedOutputOMT =
        OMTensorPtr(omTensorCreate((void *)expectedOutput.data(), inputShape,
                        1 /*rank*/, ONNX_TYPE_STRING),
            omTensorDestroy);

    LLVM_DEBUG({
      llvm::dbgs() << "input: ";
      int64_t *inputDataPtr = (int64_t *)omTensorGetDataPtr(inputOMT.get());
      for (int i = 0; i < omTensorGetNumElems(inputOMT.get()); ++i)
        llvm::dbgs() << inputDataPtr[i] << " ";
      llvm::dbgs() << "\n";

      llvm::errs() << "expectedOutput: ";
      const char **outputDataPtr =
          (const char **)omTensorGetDataPtr(expectedOutputOMT.get());
      for (int i = 0; i < omTensorGetNumElems(expectedOutputOMT.get()); ++i)
        llvm::dbgs() << outputDataPtr[i] << " ";
      llvm::dbgs() << "\n";
    });

    inputOMTs.emplace_back(move(inputOMT));
    expectedOutputOMTs.emplace_back(move(expectedOutputOMT));

    return testHelper.runAndVerifyTest(
        inputOMTs, expectedOutputOMTs, SharedLibBase);
  }

  // Prepare for a new test.
  void reset() { testHelper.reset(); }

private:
  // Create the category mapper test code into the given function, and add the
  // function into the given module.
  void createCategoryMapper(
      Type outputType, const CMAttributes &attributes, FuncOp &funcOp) {
    ModuleOp &module = testHelper.getModule();
    Location &loc = testHelper.getLocation();
    OpBuilder &builder = testHelper.getBuilder();

    Block &entryBlock = funcOp.getBody().front();
    auto input = entryBlock.getArgument(0);
    auto categoryMapperOp = builder.create<ONNXCategoryMapperOp>(loc,
        outputType, input, builder.getI64ArrayAttr(attributes.cat_int64s),
        builder.getStrArrayAttr(attributes.cat_strings), attributes.default_int,
        builder.getStringAttr(attributes.default_string));

    llvm::SmallVector<Value, 1> results = {categoryMapperOp.getResult()};
    builder.create<ReturnOp>(loc, results);

    module.push_back(funcOp);
  }
};

} // namespace

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(Tests::TestHelper::getSharedLibName(SharedLibBase));

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestCategoryMapper\n", nullptr, "TEST_ARGS");

  MLIRContext ctx;
  CategoryMapperTester categoryMapperTester(ctx);
  const CategoryMapperTester::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};

  if (!categoryMapperTester.testInt64ToStr(attributes, {1, 2, 3, 4, 5},
          {"cat", "dog", "human", "tiger", "beaver"}))
    return 1;

  return 0;
}