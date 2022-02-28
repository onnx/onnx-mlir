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

using namespace mlir;

namespace onnx_mlir {

/// Get the rank of the given tensor (represented as a memref).
int64_t getRankFromMemRefType(LLVM::LLVMStructType memRefTy);

/// Get the ONNX type corresponding to an MLIR type.
onnx::TensorProto::DataType mlirTypeToOnnxType(Type elemType);

/// Create an OMTensor from a memref.
void fillOMTensorWithMemRef(Value &outMemRef, Value &outOMTensor,
    int64_t outOwning, PatternRewriter &rewriter, const Location &loc,
    const RuntimeAPIRegistry &apiRegistry, ModuleOp &module);

/// Return the GlobalOp for the given string, creating one if not found.
LLVM::GlobalOp getOrCreateGlobalString(StringRef str, Location loc,
    OpBuilder &builder, ModuleOp module, LLVMTypeConverter *typeConverter);

/// Return a pointer to the first character in a global string.
Value getPtrToGlobalString(
    const LLVM::GlobalOp &global, Location loc, OpBuilder &builder);

/// If the operation has a valid alignment attribute use it, otherwise attempt
/// to set the alignment based on the module datalayout (if it exists).
void setAlignment(LLVM::GlobalOp &global, IntegerAttr alignmentAttr,
    ModuleOp module, OpBuilder &builder, LLVMTypeConverter &typeConverter);

/// Convert a string from ASCII to EBCDIC.
/// This is not in-place conversion and a new string in EBCDIC is returned.
std::string a2e_s(std::string a_s);

} // namespace onnx_mlir
