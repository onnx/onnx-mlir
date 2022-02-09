/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========-- ModelLib.cpp - Helper function for building models -==========//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for all the models that can be built.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

ModelLibBuilder::ModelLibBuilder(const string &name)
    : sharedLibBaseName(name), ctx(), loc(UnknownLoc::get(&ctx)),
      module(ModuleOp::create(loc)), builder(&ctx) {
  registerDialects(ctx);
}

bool ModelLibBuilder::build() {
  llvm_unreachable("subclass must overload this");
}

bool ModelLibBuilder::compile(const CompilerOptionList &options) {
  // hi alex, set options
  OwningModuleRef moduleRef(module);
  int rc = compileModule(moduleRef, ctx, sharedLibBaseName, onnx_mlir::EmitLib);
  return rc == 0;
}

FuncOp ModelLibBuilder::createEmptyTestFunction(
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

void ModelLibBuilder::createEntryPoint(FuncOp &funcOp) {
  FunctionType funcType = funcOp.getType();
  auto entryPoint = ONNXEntryPointOp::create(
      loc, funcOp, funcType.getNumInputs(), funcType.getNumResults(), "");
  module.push_back(entryPoint);
}

mlir::ONNXConstantOp ModelLibBuilder::buildONNXConstantOp(
    OMTensor *omt, mlir::RankedTensorType resultType) {
  int64_t numElems = omTensorGetNumElems(omt);
  auto bufferPtr = omTensorGetDataPtr(omt);
  float *arrayPtr = reinterpret_cast<float *>(bufferPtr);
  auto array = std::vector<float>(arrayPtr, arrayPtr + numElems);
  auto denseAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(array));
  return builder.create<ONNXConstantOp>(loc, resultType, Attribute(), denseAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}
