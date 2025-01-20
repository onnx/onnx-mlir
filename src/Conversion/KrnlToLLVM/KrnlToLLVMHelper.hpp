/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToLLVMHelper.hpp ------------------------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Declare utility functions for the Krnl to LLVM dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_TO_LLVM_H
#define ONNX_MLIR_KRNL_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

namespace onnx_mlir {
namespace krnl {

/// Get the rank of the given tensor (represented as a memref).
int64_t getRankFromMemRefType(mlir::LLVM::LLVMStructType memRefTy);

/// Get the ONNX type corresponding to an MLIR type.
int64_t mlirTypeToOnnxType(mlir::Type elemType);

/// Create an OMTensor from a memref.
void fillOMTensorWithMemRef(mlir::Value &outMemRef, mlir::Type elemTy,
    mlir::Value &outOMTensor, int64_t outOwning,
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const RuntimeAPIRegistry &apiRegistry, mlir::ModuleOp &module);

/// Return the GlobalOp for the given string, creating one if not found.
mlir::LLVM::GlobalOp getOrCreateGlobalString(llvm::StringRef str,
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    const mlir::LLVMTypeConverter *typeConverter);

/// Return a pointer to the first character in a global string.
mlir::Value getPtrToGlobalString(const mlir::LLVM::GlobalOp &global,
    mlir::Location loc, mlir::OpBuilder &builder);

/// If the operation has a valid alignment attribute use it, otherwise attempt
/// to set the alignment based on the module datalayout (if it exists).
void setAlignment(mlir::LLVM::GlobalOp &global, mlir::IntegerAttr alignmentAttr,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const mlir::LLVMTypeConverter &typeConverter);

/// Return a symbol reference to the strncmp function, inserting it into the
/// module if necessary.
mlir::FlatSymbolRefAttr getOrInsertStrncmp(
    mlir::OpBuilder &builder, mlir::ModuleOp module);

/// Convert a string from ASCII to EBCDIC IBM-1047.
/// This is not in-place conversion and a new string in EBCDIC is returned.
std::string a2e_s(std::string a_s);

/// Convert a string from EBCDIC IBM-1047 to ASCII.
/// This is not in-place conversion and a new string in ASCII is returned.
std::string e2a_s(std::string e_s);

/// Generate LLVM code to set errno to the given value.
void emitErrNo(mlir::ModuleOp module, mlir::OpBuilder &builder,
    mlir::Location loc, int err);

/// Emit code for `IF lhs != rhs THEN return null ELSE do nothing`.
void equalOrFailed(mlir::ModuleOp &module, mlir::OpBuilder &rewriter,
    mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
    std::string errorMsg = "", bool appendRHS = true);

/// Emit code for `IF lhs != rhs THEN return retVal ELSE do nothing`.
void equalOrReturn(mlir::ModuleOp &module, mlir::OpBuilder &rewriter,
    mlir::Location loc, mlir::Value lhs, mlir::Value rhs, mlir::Value retVal,
    std::string errorMsg = "");

/// Creates an LLVM pointer type with the given element type and address space.
/// This function is meant to be used in code supporting both typed and opaque
/// pointers, as it will create an opaque pointer with the given address space
/// if opaque pointers are enabled in the lowering options. This function is
/// obtained from LLVMTypeConverter. Put it here so that there is no need to
/// construct an LLVMTypeConverter.
mlir::LLVM::LLVMPointerType getPointerType(mlir::MLIRContext *context,
    mlir::Type elementType, unsigned addressSpace = 0);

mlir::LLVM::LLVMPointerType getI8PointerType(
    mlir::MLIRContext *context, unsigned addressSpace = 0);

/// Get the entry point function that locate first among other entry points in
/// the same block.
mlir::Operation *getFirstEntryOpInBlock(mlir::ModuleOp &module,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps);

/// Get rawData from a DenseElementsAttr or a DenseResourceElementsAttr.
llvm::ArrayRef<char> getRawData(mlir::KrnlGlobalOp &op);

/// Check if the module is for z/OS or not.
bool isZOS(mlir::ModuleOp module);

} // namespace krnl
} // namespace onnx_mlir
#endif
