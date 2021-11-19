/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- KrnlTypes.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of krnl types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {

class LoopType
    : public mlir::Type::TypeBase<LoopType, mlir::Type, mlir::TypeStorage> {

public:
  using Base::Base;

  // Support type inquiry through isa, cast and dyn_cast.

  // Get a unique instance of Loop type.
  static LoopType get(mlir::MLIRContext *context) { return Base::get(context); }
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage,
          mlir::MemRefElementTypeInterface::Trait> {

public:
  using Base::Base;

  // Get a unique instance of StringType.
  static StringType get(mlir::MLIRContext *context) {
    return Base::get(context);
  }

  // Return the LLVM dialect type for a string with unknown value.
  Type getLLVMType(mlir::MLIRContext *context) const {
    // This should really be an i8* so a
    // LLVM::PointerType::get(IntegerType::get(context, 8)); but a ptr type is
    // not a valid element type for a memref, so we represents the string as a
    // memref<?xi8>.
    SmallVector<int64_t> shape(1, -1);
    return MemRefType::get(shape, IntegerType::get(context, 8));
  }

  // Return the LLVM dialect type for a string with a know value (a string
  // literal). In LLVM a string literal is represented by an array of i8.
  Type getLLVMType(mlir::MLIRContext *context, StringRef value) const {
    return LLVM::LLVMArrayType::get(IntegerType::get(context, 8), value.size());
  }

  // Return the size in bits for the underlying element type (i8).
  int32_t getElementSize() const { return 8; }
};

} // namespace mlir
