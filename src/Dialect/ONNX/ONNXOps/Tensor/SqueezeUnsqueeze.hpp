/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SqueezeUnsqueeze.hpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides helper functions for Squeeze and Unsqueeze operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Update axes attribute so that it contains only positive values.
// Helper functions for both Unsqueeze and Squeeze Ops
template <typename Op>
void updateNegativeAxis(Op *op, ArrayRef<int64_t> axes) {
  OpBuilder builder(op->getContext());
  if (auto axesConstOp = getONNXConstantOp(op->axes())) {
    auto tensorType = axesConstOp.getType().template cast<RankedTensorType>();
    auto constDenseAttr = mlir::DenseElementsAttr::get(tensorType, axes);
    builder.setInsertionPoint(*op);
    auto constOp = builder.create<mlir::ONNXConstantOp>(
        op->getLoc(), mlir::Attribute(), constDenseAttr);
    mlir::Value constRes = constOp.output();
    op->setOperand(1, constRes);
  } else {
    llvm_unreachable("cannot update axes for non-constant Op");
  }
}

template <typename Op>
void updateNegativeAxisV11(Op *op, ArrayRef<int64_t> axes) {
  auto builder = mlir::Builder(op->getContext());
  ArrayRef<int64_t> defaultRefs(axes);
  op->axesAttr(builder.getI64ArrayAttr(defaultRefs));
}

} // namespace
