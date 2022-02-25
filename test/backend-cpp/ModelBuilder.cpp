/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ModelBuilder.cpp - Test Helper Implementation -------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the implementation of the ModelBuilder class.
//
//===----------------------------------------------------------------------===//

#include "test/backend-cpp/ModelBuilder.hpp"
#include "llvm/Support/Debug.h"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "backend-cpp"

namespace BackendCppTests {

FuncOp ModelBuilder::createEmptyTestFunction(
    const llvm::SmallVectorImpl<Type> &inputsType,
    const llvm::SmallVectorImpl<Type> &outputsType) {
  assert(!inputsType.empty() && "Expecting inputsTypes to be non-empty");
  assert(!outputsType.empty() && "Expecting outputsTypes to be non-empty");

  FunctionType funcType = builder.getFunctionType(inputsType, outputsType);

  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp = builder.create<FuncOp>(loc, "main_graph", funcType, attrs);

  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  return funcOp;
}

void ModelBuilder::createEntryPoint(FuncOp &funcOp) {
  FunctionType funcType = funcOp.getType();
  auto entryPoint = ONNXEntryPointOp::create(
      loc, funcOp, funcType.getNumInputs(), funcType.getNumResults(), "");
  module.push_back(entryPoint);
}

bool ModelBuilder::compileTest(const CompilerOptionList &compileOptions) {
  assert(!module.getBody()->empty() &&
         "Expecting the module to contain the test code");

  OwningOpRef<ModuleOp> modRef(module);
  setCompilerOptions(compileOptions);
  return (
      compileModule(modRef, ctx, sharedLibBaseName, onnx_mlir::EmitLib) == 0);
}

bool ModelBuilder::runAndVerifyTest(std::vector<OMTensorUniquePtr> &inputs,
    std::vector<OMTensorUniquePtr> &expectedOutputs,
    std::function<bool(OMTensor *, OMTensor *)> verifyFunction) {
  assert(!inputs.empty() && "Expecting valid inputs");

  // Run the test code.
  onnx_mlir::ExecutionSession execSession(getSharedLibName(sharedLibBaseName));
  auto outputs = execSession.run(move(inputs));
  assert(
      outputs.size() == expectedOutputs.size() && "Should have the same size");

  // Verify the result(s).
  for (size_t i = 0; i < outputs.size(); ++i) {
    OMTensorUniquePtr &output = outputs.at(i);
    OMTensorUniquePtr &expectedOutput = expectedOutputs.at(i);
    if (!verifyFunction(output.get(), expectedOutput.get()))
      return false;
  }

  return true;
}

ModuleOp ModelBuilder::createEmptyModule() const {
  registerDialects(ctx);
  setCompilerOptions({{OptionKind::CompilerOptLevel, "3"}});
  return ModuleOp::create(loc);
}

void ModelBuilder::reset() { module = createEmptyModule(); }

} // namespace BackendCppTests
