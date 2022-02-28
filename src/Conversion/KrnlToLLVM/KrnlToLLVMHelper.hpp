/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToLLVMHelper.hpp ------------------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Declare utility functions for the Krnl to LLVM dialect conversion.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "onnx/onnx_pb.h"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"

namespace onnx_mlir {
namespace krnl {

/// Get the rank of the given tensor (represented as a memref).
int64_t getRankFromMemRefType(mlir::LLVM::LLVMStructType memRefTy);

/// Get the ONNX type corresponding to an MLIR type.
onnx::TensorProto::DataType mlirTypeToOnnxType(mlir::Type elemType);

/// Create an OMTensor from a memref.
void fillOMTensorWithMemRef(mlir::Value &outMemRef, mlir::Value &outOMTensor,
    int64_t outOwning, mlir::PatternRewriter &rewriter,
    const mlir::Location &loc, const RuntimeAPIRegistry &apiRegistry,
    mlir::ModuleOp &module);

/// Return the GlobalOp for the given string, creating one if not found.
mlir::LLVM::GlobalOp getOrCreateGlobalString(llvm::StringRef str,
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::LLVMTypeConverter *typeConverter);

/// Return a pointer to the first character in a global string.
mlir::Value getPtrToGlobalString(const mlir::LLVM::GlobalOp &global,
    mlir::Location loc, mlir::OpBuilder &builder);

/// If the operation has a valid alignment attribute use it, otherwise attempt
/// to set the alignment based on the module datalayout (if it exists).
void setAlignment(mlir::LLVM::GlobalOp &global, mlir::IntegerAttr alignmentAttr,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    mlir::LLVMTypeConverter &typeConverter);

/// Retrieve the declaration of a function in the given module.
llvm::Optional<mlir::FlatSymbolRefAttr> getFunctionDeclaration(
    mlir::ModuleOp module, llvm::StringRef funcName);

/// Return a symbol reference to the strncmp function, inserting it into the
/// module if necessary.
mlir::FlatSymbolRefAttr getOrInsertStrncmp(
    mlir::OpBuilder &builder, mlir::ModuleOp module);

} // namespace krnl
} // namespace onnx_mlir
