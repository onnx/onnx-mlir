/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVMCommon.hpp - Lowering from ZLow to LLVM ---------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in lowering ZLow to LLVM
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zdnn.h"

namespace onnx_mlir {
namespace zlow {

enum class API {
  NULL_API,
  // Tensor functions
  ZDNN_INIT_PRE_TRANSFORMED_DESC,
  ZDNN_GENERATE_TRANSFORMED_DESC,
  ZDNN_GENERATE_TRANSFORMED_DESC_CONCATENATED,
  ZDNN_GETSIZE_ZTENSOR,
  ZDNN_TRANSFORM_ZTENSOR,
  ZDNN_TRANSFORM_ORIGTENSOR,
  // Elementwise operations
  ZDNN_ADD,
  ZDNN_SUB,
  ZDNN_MUL,
  ZDNN_DIV,
  ZDNN_MIN,
  ZDNN_MAX,
  ZDNN_LOG,
  ZDNN_EXP,
  // Activation operations
  ZDNN_RELU,
  ZDNN_TANH,
  ZDNN_SIGMOID,
  ZDNN_SOFTMAX,
  // RNN operations
  ZDNN_LSTM,
  ZDNN_GRU,
  // Other operations
  ZDNN_MATMUL_OP,
  ZDNN_MATMUL_BCAST_OP,
  ZDNN_CONV2D,
  ZDNN_AVGPOOL2D,
  ZDNN_MAXPOOL2D,
  ZDNN_MEANREDUCE2D,
  ZDNN_BATCHNORM
};

// Obtain a zDNN API for an elementwise ZLow operation.
template <typename ZLowOp>
API APIFor() {
  return API::NULL_API;
}

// API specs to declare external function types.
struct ApiSpec {
  API id;
  std::string name;
  mlir::Type outputTy;
  mlir::SmallVector<mlir::Type, 4> inputTys;
  bool isVarArg;

  ApiSpec(API id, const std::string &name, mlir::Type outputTy,
      mlir::ArrayRef<mlir::Type> inputTys, const bool isVarArg)
      : id(id), name(name), outputTy(outputTy),
        inputTys(inputTys.begin(), inputTys.end()), isVarArg(isVarArg) {}

  mlir::Type funcTy() {
    return mlir::LLVM::LLVMFunctionType::get(outputTy, inputTys,
        /*isVarArg=*/isVarArg);
  }
};

using ApiRegistry = std::map<API, ApiSpec>;
ApiRegistry RegisterAllApis(mlir::MLIRContext *context);

/// A struct to hold pointers to a zTensor.
struct ZTensor {
  mlir::Value val;
  // zTensor's members.
  mlir::Value preTransformedDescPtr;
  mlir::Value transformedDescPtr;
  mlir::Value bufferSize;
  mlir::Value bufferPtr;
  bool isTransformed;
};

/// A helper class to create a zTensor.
class ZTensorHelper {
public:
  // Constructor.
  ZTensorHelper(mlir::PatternRewriter &rewriter, mlir::Location loc,
      mlir::ModuleOp module, ApiRegistry apiRegistry);

  // Get a pre-transformed descriptor.
  mlir::Value getPreTransformedDescPtr(zdnn_data_types zDNNDataType,
      zdnn_data_layouts zDNNDataLayout, mlir::ArrayRef<mlir::Value> dims);
  // Get a transformed descriptor.
  mlir::Value getTransformedDescPtr(mlir::Value preTransformedDescPtr,
      bool isConcat = false,
      zdnn_concat_info concatInfo = RNN_TYPE_GRU | USAGE_WEIGHTS |
                                    PREV_LAYER_NONE);
  // Get the pointer to memref.
  mlir::Value getAlignedI8Ptr(mlir::Value memRef);
  // Get buffer size from a transformed descriptor.
  mlir::Value getBufferSize(mlir::Value transformedDescPtr);
  // Create a zTensor.
  ZTensor getZTensor(mlir::Value bufferPtr, zdnn_data_types dataType,
      zdnn_data_layouts layout, mlir::ArrayRef<mlir::Value> originalDims,
      bool isTransformed, bool isConcat = false,
      zdnn_concat_info concatInfo = RNN_TYPE_GRU | USAGE_WEIGHTS |
                                    PREV_LAYER_NONE);
  // Create a zTensor from existing descriptors.
  ZTensor getZTensor(mlir::Value preTransformedDescPtr,
      mlir::Value transformedDescPtr, mlir::Value bufferSize,
      mlir::Value bufferPtr, bool isTransformed);

private:
  mlir::PatternRewriter &rewriter;
  mlir::Location loc;
  mlir::ModuleOp module;
  // API registry to call external functions.
  ApiRegistry apiRegistry;
};

/// Search for a function reference. Insert a new function reference if not
/// found.
mlir::FlatSymbolRefAttr getOrInsertExternFuncRef(
    mlir::PatternRewriter &rewriter, mlir::ModuleOp module,
    mlir::StringRef funcName, mlir::Type funcType);

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
mlir::Value callApi(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::ModuleOp module, ApiRegistry registry, API apiId,
    mlir::ArrayRef<mlir::Value> params);

/// Get rank of a memref.
size_t getRankFromMemRefType(mlir::LLVM::LLVMStructType memRefTy);

/// Get a vector of 'size' dimensions from a 1D DenseElementsAttr.
std::vector<mlir::Value> getDimsFromDenseElementsAttr(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::ModuleOp module,
    mlir::DenseElementsAttr valueAttr, unsigned size);

/// Get a vector of 'size' dimensions from a 1D MemRef of shape.
std::vector<mlir::Value> getDimsFromShapeMemRefBySize(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::ModuleOp module,
    mlir::Value shapeMemRef, unsigned size);

/// Get a vector of dimensions from a 1D MemRef of shape, using layout.
std::vector<mlir::Value> getDimsFromShapeMemRef(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::ModuleOp module, mlir::Value shapeMemRef,
    unsigned layout);

/// Get dimensions from a MemRef value.
void getDimsFromMemRef(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::ModuleOp module, mlir::Value memRef,
    mlir::SmallVectorImpl<mlir::Value> &dims);

/// Type conversion from LLVMType to zDNNType.
/// TODO: fill in the complete list of the zDNN types.
zdnn_data_types llvmTypeToZDNNType(mlir::Type elemType);

/// Function to create a zTensor descriptor struct type.
mlir::Type getZTensorDescStructTy(mlir::MLIRContext *context);

/// Function to return the size (in bytes) of a zTensor descriptor struct.
size_t getZTensorDescStructSizeInBytes(
    mlir::MLIRContext *context, mlir::Type descTy);

/// Function to create a zTensor struct type.
mlir::Type getZTensorStructTy(mlir::MLIRContext *context);

/// Function to cast an LLVM pointer to an opaque LLVM pointer.
mlir::Value toOpaquePtr(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::ModuleOp module, mlir::Value ptr);

/// Function to fill in members of a zTensor.
void fillInZTensor(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::ModuleOp module, mlir::Value zTensor,
    mlir::Value preTransformedDescPtr, mlir::Value transformedDescPtr,
    bool isTransformed, mlir::Value bufferSize, mlir::Value alignedBuffer);

} // namespace zlow
} // namespace onnx_mlir
