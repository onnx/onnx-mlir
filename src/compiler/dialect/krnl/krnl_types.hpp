//===--------------------- krnl_types.hpp - MLIR Operations ---------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Types.h>

namespace mlir {

namespace KrnlTypes {
enum Kinds {
  // A krnl.loop is simply a reference to a for loop and will be used to:
  // - Indicate the presence of a for loop in krnl.iterate.
  // - Identify the loop in optimization intrinsics.
  Loop = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
}

class LoopType : public mlir::Type::TypeBase<LoopType, mlir::Type> {
 public:
  using Base::Base;

  // Support type inquiry through isa, cast and dyn_cast.
  static bool kindof(unsigned kind) { return kind == KrnlTypes::Loop; }

  // Get a unique instance of Loop type.
  static LoopType get(mlir::MLIRContext* context) {
    return Base::get(context, KrnlTypes::Loop);
  }
};
}  // namespace mlir
