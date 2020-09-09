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

#include <mlir/IR/Types.h>

namespace mlir {
class LoopType
    : public mlir::Type::TypeBase<LoopType, mlir::Type, mlir::TypeStorage> {

public:
  using Base::Base;

  // Support type inquiry through isa, cast and dyn_cast.

  // Get a unique instance of Loop type.
  static LoopType get(mlir::MLIRContext *context) { return Base::get(context); }
};
} // namespace mlir
