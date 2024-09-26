/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- KrnlTypes.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of krnl types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_TYPES_H
#define ONNX_MLIR_KRNL_TYPES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace onnx_mlir {
namespace krnl {

class LoopType
    : public mlir::Type::TypeBase<LoopType, mlir::Type, mlir::TypeStorage> {

public:
  using Base::Base;

  static constexpr const char *name = "loop.type";

  // Support type inquiry through isa, cast and dyn_cast.

  // Get a unique instance of Loop type.
  static LoopType get(mlir::MLIRContext *context) { return Base::get(context); }
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage,
          mlir::MemRefElementTypeInterface::Trait> {

public:
  using Base::Base;

  static constexpr const char *name = "string.type";

  // Get a unique instance of StringType.
  static StringType get(mlir::MLIRContext *context) {
    return Base::get(context);
  }

  // Return the LLVM dialect type for a string with unknown value.
  Type getLLVMType(mlir::MLIRContext *context) const {
    // This should really be an i8*, however a ptr type is *not* a valid element
    // type for a memref, so we use an i64 (that type has the same length as a
    // pointer).
    // TODO: change when memref accept aptr types as elements.
    //    SmallVector<int64_t> shape(1, ShapedType::kDynamic);
    //    return MemRefType::get(
    //  shape, LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
    return mlir::IntegerType::get(context, 64);
  }

  // Return the LLVM dialect type for a string with a know value (a string
  // literal). In LLVM a string literal is represented by an array of i8.
  Type getLLVMType(mlir::MLIRContext *context, mlir::StringRef value) const {
    return mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(context, 8), value.size());
  }

  // Return the size in bits for the underlying element type (i64).
  int32_t getElementSize() const { return 64; }
};

/// Add custom type conversions to convert krnl types to the given \p
/// typeConverter.
void customizeTypeConverter(mlir::LLVMTypeConverter &typeConverter);

} // namespace krnl
} // namespace onnx_mlir
#endif
