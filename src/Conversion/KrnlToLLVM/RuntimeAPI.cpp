/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ RuntimeAPI.cpp - Implementation of Runtime API ----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the Runtime API.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// RuntimeAPI
//===----------------------------------------------------------------------===//

void RuntimeAPI::declareAPI(ModuleOp &module, OpBuilder &builder) {
  auto funcType = LLVM::LLVMFunctionType::get(outputTy, inputTys,
      /*isVarArg=*/false);
  symbolRef = getOrInsertExternFunc(name, module, funcType, builder);
}

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
Value RuntimeAPI::callApi(OpBuilder &builder, Location loc,
    const RuntimeAPIRegistry &registry, API apiId, ArrayRef<Value> params) {
  // To be used as parameters in LLVM::CallOp, voidTy must be converted
  // to empty list to avoid emission of an SSA value with voidTy. However,
  // we still keep using LLVM voidTy (as opposed to empty list) when recording
  // API function signatures in API registry because when declaring API
  // functions in LLVM IR, the correct way to indicate an output type for
  // "void" is still LLVM voidTy. Relevant discussion thread:
  // https://github.com/onnx/onnx-mlir/issues/255.
  SmallVector<Type, 1> outputTys;
  const RuntimeAPI &runtimeAPI = registry.getAPI(apiId);
  auto outputTy = runtimeAPI.outputTy;
  if (!outputTy.isa<LLVM::LLVMVoidType>())
    outputTys.emplace_back(outputTy);
  auto returnVals = builder.create<LLVM::CallOp>(loc, ArrayRef<Type>(outputTys),
      registry.getAPI(apiId).symbolRef, ArrayRef<Value>(params));
  if (returnVals.getNumResults() == 1)
    return returnVals.getResult(0);
  return nullptr;
}

FlatSymbolRefAttr RuntimeAPI::getOrInsertExternFunc(StringRef funcName,
    ModuleOp module, mlir::Type funcType, OpBuilder &builder) {
  MLIRContext *context = module.getContext();
  if (auto sym = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    assert(sym.getType() == funcType && "wrong symbol type");
    return SymbolRefAttr::get(context, funcName);
  }

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  return SymbolRefAttr::get(context, funcName);
}

//===----------------------------------------------------------------------===//
// RuntimeAPIRegistry
//===----------------------------------------------------------------------===//

RuntimeAPIRegistry *RuntimeAPIRegistry::instance = nullptr;

RuntimeAPIRegistry::~RuntimeAPIRegistry() {
  // To support multiple modules in the same compilation process we need to
  // reset the singleton instance.
  instance = nullptr;
}

const RuntimeAPIRegistry RuntimeAPIRegistry::build(
    ModuleOp &module, OpBuilder &builder) {
  if (!instance)
    instance = new RuntimeAPIRegistry(module, builder);

  return *instance;
}

RuntimeAPIRegistry::RuntimeAPIRegistry(ModuleOp &module, OpBuilder &builder)
    : registry() {
  MLIRContext *context = module.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto int8Ty = IntegerType::get(context, 8);
  auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
  auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
  auto int64Ty = IntegerType::get(context, 64);
  auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);

  // Declare API type as an enum value, its string name and an LLVM Type
  // specifying its signature.
  // clang-format off
  using API = RuntimeAPI::API;  
  std::vector<RuntimeAPI> RuntimeAPISpecs = {
    RuntimeAPI(API::CREATE_OMTENSOR_LIST, "omTensorListCreateWithOwnership", opaquePtrTy, {opaquePtrPtrTy, int64Ty, int64Ty}),
    RuntimeAPI(API::CREATE_OMTENSOR, "omTensorCreateUntyped", opaquePtrTy, {int64Ty}),
    RuntimeAPI(API::GET_DATA, "omTensorGetDataPtr", opaquePtrTy, {opaquePtrTy}),
    RuntimeAPI(API::SET_DATA, "omTensorSetDataPtr", voidTy, {opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_SHAPE, "omTensorGetShape", int64PtrTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_STRIDES, "omTensorGetStrides", int64PtrTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_TYPE, "omTensorGetDataType", int64Ty, {opaquePtrTy}),
    RuntimeAPI(API::SET_DATA_TYPE, "omTensorSetDataType", voidTy, {opaquePtrTy, int64Ty}),
    RuntimeAPI(API::GET_OMT_ARRAY, "omTensorListGetOmtArray", opaquePtrPtrTy, {opaquePtrTy}),
    RuntimeAPI(API::PRINT_OMTENSOR, "omTensorPrint", voidTy, {opaquePtrTy, opaquePtrTy}),
  };
  // clang-format on

  // Declare APIs in the current module and build an API registry mapping api
  // identities to a symbol reference to the API function.
  for (auto &apiSpec : RuntimeAPISpecs) {
    apiSpec.declareAPI(module, builder);
    registry.emplace(apiSpec.id, apiSpec);
  }
}
