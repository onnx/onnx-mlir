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
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

ModelLibBuilder::ModelLibBuilder(const string &name)
    : sharedLibBaseName(name), ctx(), loc(UnknownLoc::get(&ctx)), builder(&ctx),
      module(ModuleOp::create(loc)), inputs(nullptr), outputs(nullptr),
      exec(nullptr) {
  registerDialects(ctx);
}

ModelLibBuilder::~ModelLibBuilder() {
  if (inputs)
    omTensorListDestroy(inputs);
  if (outputs)
    omTensorListDestroy(outputs);
  if (exec)
    delete exec;
}

bool ModelLibBuilder::compileAndLoad() {
  OwningModuleRef moduleRef(module);
  int rc = compileModule(moduleRef, ctx, sharedLibBaseName, onnx_mlir::EmitLib);
  if (rc != 0)
    return false;
#ifdef _WIN32
  std::string fullName(sharedLibBaseName + ".dll");
#else
  std::string fullName(sharedLibBaseName + ".so");
#endif
  exec = new ExecutionSession(fullName);
  return exec != nullptr;
}

bool ModelLibBuilder::run() {
  assert(inputs && exec && "expected successful compile and load");
  if (outputs) {
    omTensorListDestroy(outputs);
    outputs = nullptr;
  }
  outputs = exec->run(inputs);
  return outputs != nullptr;
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

ONNXConstantOp ModelLibBuilder::buildONNXConstantOp(
    const OMTensor *omt, const RankedTensorType resultType) {
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

bool ModelLibBuilder::areCloseFloat(const OMTensor *res, const OMTensor *ref) {
  if (!res || !ref)
    return false;
  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;
  return omTensorAreTwoOmtsClose<float>(res, ref, rtol, atol);
}
