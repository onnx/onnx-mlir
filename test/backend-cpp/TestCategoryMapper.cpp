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

#include "test/backend-cpp/ModelBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "backend-cpp"

static const string SharedLibBaseName("./TestCategoryMapper_main_graph");

namespace {

class CategoryMapperTester {
  using ModelBuilder = BackendCppTests::ModelBuilder;
  ModelBuilder modelBuilder;

public:
  // Aggregate CategoryMapper attributes.
  struct CMAttributes {
    ArrayRef<int64_t> cat_int64s;
    ArrayRef<StringRef> cat_strings;
    int64_t default_int;
    StringRef default_string;
  };

  CategoryMapperTester(MLIRContext &ctx)
      : modelBuilder(ModelBuilder(ctx, SharedLibBaseName)) {}

  // Test CategoryMapper (with an input tensor of int64_t numbers).
  bool testInt64ToStr(const CMAttributes &attributes, ArrayRef<int64_t> input,
      ArrayRef<const char *> expectedOutput) {
    assert(input.size() == expectedOutput.size() &&
           "Expecting input/output to have the same size");

    int64_t inputShape[] = {static_cast<int64_t>(input.size())};
    auto inputType = RankedTensorType::get(
        inputShape, modelBuilder.getBuilder().getI64Type());
    auto outputType = RankedTensorType::get(
        inputShape, ONNXStringType::get(&modelBuilder.getContext()));

    // Create the test code.
    llvm::SmallVector<Type, 1> inputsType{inputType}, outputsType{outputType};
    FuncOp funcOp =
        modelBuilder.createEmptyTestFunction(inputsType, outputsType);
    createCategoryMapper(outputType, attributes, funcOp);
    modelBuilder.createEntryPoint(funcOp);

    // Compile the test.
    if (!modelBuilder.compileTest(
            {{onnx_mlir::OptionKind::CompilerOptLevel, "3"}})) {
      llvm::errs() << "Failed to compile the test case\n";
      return false;
    }

    // Run the test and verify the result.
    std::vector<onnx_mlir::OMTensorUniquePtr> inputOMTs, expectedOutputOMTs;
    auto inputOMT = onnx_mlir::OMTensorUniquePtr(
        omTensorCreate(static_cast<void *>(const_cast<int64_t *>(input.data())),
            inputShape, 1 /*rank*/, ONNX_TYPE_INT64),
        omTensorDestroy);
    auto expectedOutputOMT = onnx_mlir::OMTensorUniquePtr(
        omTensorCreate(static_cast<void *>(
                           const_cast<const char **>(expectedOutput.data())),
            inputShape, 1 /*rank*/, ONNX_TYPE_STRING),
        omTensorDestroy);

    LLVM_DEBUG({
      llvm::dbgs() << "input: ";
      int64_t *inputDataPtr =
          static_cast<int64_t *>(omTensorGetDataPtr(inputOMT.get()));
      for (int i = 0; i < omTensorGetNumElems(inputOMT.get()); ++i)
        llvm::dbgs() << inputDataPtr[i] << " ";
      llvm::dbgs() << "\n";

      llvm::errs() << "expectedOutput: ";
      const char **outputDataPtr = static_cast<const char **>(
          omTensorGetDataPtr(expectedOutputOMT.get()));
      for (int i = 0; i < omTensorGetNumElems(expectedOutputOMT.get()); ++i)
        llvm::dbgs() << outputDataPtr[i] << " ";
      llvm::dbgs() << "\n";
    });

    inputOMTs.emplace_back(move(inputOMT));
    expectedOutputOMTs.emplace_back(move(expectedOutputOMT));

    return modelBuilder.runAndVerifyTest(
        inputOMTs, expectedOutputOMTs, verifyFunction);
  }

  // Prepare for a new test.
  void reset() { modelBuilder.reset(); }

private:
  // Create the category mapper test code into the given function, and add the
  // function into the given module.
  void createCategoryMapper(
      Type outputType, const CMAttributes &attributes, FuncOp &funcOp) {
    ModuleOp &module = modelBuilder.getModule();
    Location &loc = modelBuilder.getLocation();
    OpBuilder &builder = modelBuilder.getBuilder();

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

  static bool verifyFunction(OMTensor *out, OMTensor *expected) {
    // Verify that the output tensor has the expected rank/extents.
    if (omTensorGetRank(out) != omTensorGetRank(expected)) {
      llvm::errs() << "Output tensor has rank " << omTensorGetRank(out)
                   << ", expecting " << omTensorGetRank(expected) << "\n";
      return false;
    }
    if (omTensorGetNumElems(out) != omTensorGetNumElems(expected)) {
      llvm::errs() << "Output tensor has " << omTensorGetNumElems(out)
                   << "elements, expecting " << omTensorGetNumElems(expected)
                   << "\n";
      return false;
    }

    // Verify that the output tensor contains the expected result.
    auto outDataPtr = (const char **)omTensorGetDataPtr(out);
    auto expectedDataPtr = (const char **)(omTensorGetDataPtr(expected));

    LLVM_DEBUG(llvm::dbgs() << "Result Verification:\n");
    for (int i = 0; i < omTensorGetNumElems(out); i++) {
      const char *str = outDataPtr[i];
      const char *expectedStr = expectedDataPtr[i];
      LLVM_DEBUG(llvm::dbgs()
                 << "str: " << str << ", expectedStr: " << expectedStr << "\n");

      if (strcmp(str, expectedStr) != 0) {
        llvm::errs() << "Output tensor contains \"" << str
                     << "\" at index = " << i << ", expecting \"" << expectedStr
                     << "\"\n";
        return false;
      }
    }

    return true;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      BackendCppTests::ModelBuilder::getSharedLibName(SharedLibBaseName));

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
