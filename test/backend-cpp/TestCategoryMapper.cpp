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
      ArrayRef<const char *> expOutput) {
    assert(input.size() == expOutput.size() &&
           "Expecting input/expOutput to have the same size");

    // Create the test function.
    int64_t shape[1] = {static_cast<int64_t>(input.size())};
    auto inputType =
        RankedTensorType::get(shape, modelBuilder.getBuilder().getI64Type());
    auto outputType = RankedTensorType::get(
        shape, ONNXStringType::get(&modelBuilder.getContext()));
    createTestFunction(inputType, outputType, attributes);

    // Compile the test.
    if (!modelBuilder.compileTest(
            {{onnx_mlir::OptionKind::CompilerOptLevel, "3"}})) {
      llvm::errs() << "Failed to compile the test case\n";
      return false;
    }

    // Run the test and verify the result.
    auto inputOMT = onnx_mlir::OMTensorUniquePtr(
        createOMTensor<int64_t>(input, shape, 1, ONNX_TYPE_INT64),
        omTensorDestroy);
    auto expOutputOMT = onnx_mlir::OMTensorUniquePtr(
        createOMTensor<const char *>(expOutput, shape, 1, ONNX_TYPE_STRING),
        omTensorDestroy);
    LLVM_DEBUG({
      llvm::dbgs() << "input: ";
      printTensorData<int64_t>(inputOMT.get());
      llvm::dbgs() << "expected output: ";
      printTensorData<const char *>(expOutputOMT.get());
    });

    std::vector<onnx_mlir::OMTensorUniquePtr> inputOMTs, expOutputOMTs;
    inputOMTs.emplace_back(move(inputOMT));
    expOutputOMTs.emplace_back(move(expOutputOMT));

    return modelBuilder.runAndVerifyTest(
        inputOMTs, expOutputOMTs, verifyResults<const char *>);
  }

  // Test CategoryMapper (with an input tensor of strings).
  bool testStrToInt64(const CMAttributes &attributes,
      ArrayRef<const char *> input, ArrayRef<int64_t> expOutput) {
    assert(input.size() == expOutput.size() &&
           "Expecting input/expOutput to have the same size");

    // Create the test function.
    int64_t shape[1] = {static_cast<int64_t>(input.size())};
    auto inputType = RankedTensorType::get(
        shape, ONNXStringType::get(&modelBuilder.getContext()));
    auto outputType =
        RankedTensorType::get(shape, modelBuilder.getBuilder().getI64Type());
    createTestFunction(inputType, outputType, attributes);

    // Compile the test.
    if (!modelBuilder.compileTest(
            {{onnx_mlir::OptionKind::CompilerOptLevel, "3"}})) {
      llvm::errs() << "Failed to compile the test case\n";
      return false;
    }

    // Run the test and verify the result.
    auto inputOMT = onnx_mlir::OMTensorUniquePtr(
        createOMTensor<const char *>(input, shape, 1, ONNX_TYPE_STRING),
        omTensorDestroy);
    auto expOutputOMT = onnx_mlir::OMTensorUniquePtr(
        createOMTensor<int64_t>(expOutput, shape, 1, ONNX_TYPE_INT64),
        omTensorDestroy);
    LLVM_DEBUG({
      llvm::dbgs() << "input: ";
      printTensorData<int64_t>(inputOMT.get());
      llvm::dbgs() << "expected output: ";
      printTensorData<const char *>(expOutputOMT.get());
    });

    std::vector<onnx_mlir::OMTensorUniquePtr> inputOMTs, expOutputOMTs;
    inputOMTs.emplace_back(move(inputOMT));
    expOutputOMTs.emplace_back(move(expOutputOMT));

    return modelBuilder.runAndVerifyTest(
        inputOMTs, expOutputOMTs, verifyResults<const char *>);
  }

  // Prepare for a new test.
  void reset() { modelBuilder.reset(); }

private:
  // Create the function to test.
  void createTestFunction(
      Type inputType, Type outputType, const CMAttributes &attributes) {
    llvm::SmallVector<Type, 1> inputsType{inputType}, outputsType{outputType};
    FuncOp funcOp =
        modelBuilder.createEmptyTestFunction(inputsType, outputsType);
    createCategoryMapper(outputType, attributes, funcOp);
    modelBuilder.createEntryPoint(funcOp);
  }

  // Create the category mapper operator, and insert it into the test function.
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

  // Verify that the output tensor has the expected rank.
  static bool verifyRank(const OMTensor &out, int64_t rank) {
    if (omTensorGetRank(&out) != rank) {
      llvm::errs() << "Output tensor has rank " << omTensorGetRank(&out)
                   << ", expecting " << rank << "\n";
      return false;
    }
    return true;
  }

  // Verify that the output tensor has the expected number of elements.
  static bool verifyNumElements(const OMTensor &out, int64_t numElems) {
    if (omTensorGetNumElems(&out) != numElems) {
      llvm::errs() << "Output tensor has " << omTensorGetNumElems(&out)
                   << " elements, expecting " << numElems << "\n";
      return false;
    }
    return true;
  }

  template <typename T>
  static bool compareEqual(T val, T expectedVal) {
    return val == expectedVal;
  }

  // Verification function.
  // This function will be called back by the ModelBuilder.
  template <typename T>
  static bool verifyResults(const OMTensor *out, const OMTensor *expected) {
    if (!verifyRank(*out, omTensorGetRank(expected)))
      return false;
    if (!verifyNumElements(*out, omTensorGetNumElems(expected)))
      return false;

    // Verify that the output tensor contains the expected result.
    const auto *outDataPtr = static_cast<T *>(omTensorGetDataPtr(out));
    const auto *expDataPtr = static_cast<T *>(omTensorGetDataPtr(expected));

    LLVM_DEBUG(llvm::dbgs() << "Result Verification:\n");
    for (int64_t i = 0; i < omTensorGetNumElems(out); i++) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Got: " << outDataPtr[i] << ", expected: " << expDataPtr[i]
                 << "\n");

      if (!compareEqual(outDataPtr[i], expDataPtr[i])) {
        llvm::errs() << "Output tensor contains \"" << outDataPtr[i]
                     << "\" at index = " << i << ", expecting \""
                     << expDataPtr[i] << "\"\n";
        return false;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Result is OK.\n");

    return true;
  }

  // Utility function used to create an OMTensor.
  template <typename T>
  static OMTensor *createOMTensor(
      ArrayRef<T> array, int64_t shape[], int64_t rank, OM_DATA_TYPE dtype) {
    return omTensorCreate(
        static_cast<void *>(const_cast<T *>(array.data())), shape, rank, dtype);
  }

  // Print the data pointed to by the given OMtensor.
  template <typename T>
  static void printTensorData(const OMTensor *omt) {
    T *dataPtr = static_cast<T *>(omTensorGetDataPtr(omt));
    for (int64_t i = 0; i < omTensorGetNumElems(omt); ++i)
      llvm::dbgs() << dataPtr[i] << " ";
    llvm::dbgs() << "\n";
  }
};

template <>
bool CategoryMapperTester::compareEqual(
    const char *str, const char *expectedStr) {
  return strcmp(str, expectedStr) == 0;
}

} // namespace

bool testInt64ToStr() {
  MLIRContext ctx;
  CategoryMapperTester categoryMapperTester(ctx);
  const CategoryMapperTester::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};

  return categoryMapperTester.testInt64ToStr(
      attributes, {1, 2, 3, 4, 5}, {"cat", "dog", "human", "tiger", "beaver"});
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      BackendCppTests::ModelBuilder::getSharedLibName(SharedLibBaseName));

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestCategoryMapper\n", nullptr, "TEST_ARGS");

  if (!testInt64ToStr())
    return 1;

  return 0;
}
