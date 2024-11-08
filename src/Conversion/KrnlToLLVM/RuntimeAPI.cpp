/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ RuntimeAPI.cpp - Implementation of Runtime API ----------------===//
//
// Copyright 2022,2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the Runtime API.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// RuntimeAPI
//===----------------------------------------------------------------------===//

void RuntimeAPI::declareAPI(ModuleOp &module, OpBuilder &builder) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::LLVMBuilder> create(
      builder, module.getLoc());
  symbolRef =
      create.llvm.getOrInsertSymbolRef(module, name, outputTy, inputTys);
}

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
Value RuntimeAPI::callApi(OpBuilder &builder, Location loc,
    const RuntimeAPIRegistry &registry, API apiId, ArrayRef<Value> params) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::LLVMBuilder> create(builder, loc);
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
  if (!mlir::isa<LLVM::LLVMVoidType>(outputTy))
    outputTys.emplace_back(outputTy);
  return create.llvm.call(ArrayRef<Type>(outputTys),
      registry.getAPI(apiId).symbolRef, ArrayRef<Value>(params));
}

//===----------------------------------------------------------------------===//
// RuntimeAPIRegistry
//===----------------------------------------------------------------------===//

RuntimeAPIRegistry::~RuntimeAPIRegistry() {}

RuntimeAPIRegistry::RuntimeAPIRegistry(
    ModuleOp &module, OpBuilder &builder, const LLVMTypeConverter &typeConvert)
    : registry() {
  MLIRContext *context = module.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(context);
  Type int1Ty = IntegerType::get(context, 1);
  auto int8Ty = IntegerType::get(context, 8);
  auto opaquePtrTy = onnx_mlir::krnl::getPointerType(context, int8Ty);
  auto opaquePtrPtrTy = onnx_mlir::krnl::getPointerType(context, opaquePtrTy);
  auto int64Ty = IntegerType::get(context, 64);
  auto int64PtrTy = onnx_mlir::krnl::getPointerType(context, int64Ty);

  // Declare API type as an enum value, its string name and an LLVM Type
  // specifying its signature.
  // clang-format off
  using API = RuntimeAPI::API;
  std::vector<RuntimeAPI> RuntimeAPISpecs = {
    RuntimeAPI(API::CREATE_OMTENSOR_LIST, "omTensorListCreate", opaquePtrTy, {opaquePtrPtrTy, int64Ty}),
    RuntimeAPI(API::CREATE_OMTENSOR, "omTensorCreateUntyped", opaquePtrTy, {int64Ty}),
    RuntimeAPI(API::DESTROY_OMTENSOR, "omTensorDestroy", voidTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA, "omTensorGetDataPtr", opaquePtrTy, {opaquePtrTy}),
    RuntimeAPI(API::SET_DATA, "omTensorSetDataPtr", voidTy, {opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_RANK, "omTensorGetRank", int64Ty, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_SHAPE, "omTensorGetShape", int64PtrTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_STRIDES, "omTensorGetStrides", int64PtrTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA_TYPE, "omTensorGetDataType", int64Ty, {opaquePtrTy}),
    RuntimeAPI(API::SET_DATA_TYPE, "omTensorSetDataType", voidTy, {opaquePtrTy, int64Ty}),
    RuntimeAPI(API::GET_OMT_ARRAY, "omTensorListGetOmtArray", opaquePtrPtrTy, {opaquePtrTy}),
    RuntimeAPI(API::PRINT_OMTENSOR, "omTensorPrint", voidTy, {opaquePtrTy, opaquePtrTy}),
    RuntimeAPI(API::GET_OMTENSOR_LIST_SIZE, "omTensorListGetSize", int64Ty, {opaquePtrTy}),
    RuntimeAPI(API::MMAP_BINARY_FILE, "omMMapBinaryFile", int1Ty, {opaquePtrPtrTy, opaquePtrTy, int64Ty, int64Ty}),
    RuntimeAPI(API::GET_EXTERNAL_CONSTANT_ADDR, "omGetExternalConstantAddr", voidTy, {opaquePtrPtrTy, opaquePtrPtrTy, int64Ty}),
  };
  // clang-format on

  // Declare APIs in the current module and build an API registry mapping api
  // identities to a symbol reference to the API function.
  for (auto &apiSpec : RuntimeAPISpecs) {
    apiSpec.declareAPI(module, builder);
    registry.emplace(apiSpec.id, apiSpec);
  }
}
