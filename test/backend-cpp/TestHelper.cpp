/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- TestHelper.cpp - Test Helper Implementation ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the implementation of the TestHelper class.
//
//===----------------------------------------------------------------------===//

#include "test/backend-cpp/TestHelper.hpp"
#include "llvm/Support/Debug.h"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "backend-cpp"

namespace BackendCppTests {

FuncOp TestHelper::createEmptyTestFunction(Type inputType, Type outputType) {
  llvm::SmallVector<Type, 1> inputsType{inputType};
  llvm::SmallVector<Type, 1> outputsType{outputType};
  auto funcType = builder.getFunctionType(inputsType, outputsType);

  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp = builder.create<FuncOp>(loc, "main_graph", funcType, attrs);

  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return funcOp;
}

void TestHelper::createEntryPoint(
    FuncOp &funcOp, int numInputs, int numOutputs) {
  auto entryPoint =
      ONNXEntryPointOp::create(loc, funcOp, numInputs, numOutputs, "");
  module.push_back(entryPoint);
}

bool TestHelper::compileTest(const string &sharedLibBase) {
  assert(!module.getBody()->empty() &&
         "Expecting the module to contain the test code");
  assert(module.verify().succeeded() && "Malformed module");

  OwningModuleRef modRef(module);
  setCompileContext(ctx, {{OptionKind::CompilerOptLevel, "3"}});
  return (compileModule(modRef, ctx, sharedLibBase, onnx_mlir::EmitLib) == 0);
}

bool TestHelper::runAndVerifyTest(
    std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> &inputs,
    std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>>
        &expectedOutputs,
    const string &sharedLibBase) {
  onnx_mlir::ExecutionSession execSession(
      getSharedLibName(sharedLibBase), "run_main_graph");
  auto outputs = execSession.run(move(inputs));

  assert(
      outputs.size() == expectedOutputs.size() && "Should have the same size");

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &output = outputs.at(i);
    auto &expectedOutput = expectedOutputs.at(i);
    OMTensor *out = output.get();
    OMTensor *expected = expectedOutput.get();

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
  }

  return true;
}

ModuleOp TestHelper::createEmptyModule() const {
  setCompileContext(ctx, {{OptionKind::CompilerOptLevel, "3"}});
  return ModuleOp::create(loc);
}

void TestHelper::reset() { module = createEmptyModule(); }

} // namespace BackendCppTests
