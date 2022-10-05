/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXToZHigh.hpp - Common functions in ONNXToZHigh ---------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in ONNX to ZHigh lowering.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"

template <typename OP_TYPE>
void addDynamicallyLegalOpFor(mlir::ConversionTarget *target,
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  target->addDynamicallyLegalOp<OP_TYPE>([execNodesOnCpu](OP_TYPE op) {
    // Check operations to be forced to run on CPU.
    mlir::Operation *genericOp = op.getOperation();
    mlir::StringAttr nodeName =
        genericOp->getAttrOfType<mlir::StringAttr>("onnx_node_name");
    if (nodeName) {
      bool exists =
          llvm::any_of(execNodesOnCpu, [nodeName](llvm::StringRef val) {
            return nodeName.getValue().equals_insensitive(val);
          });
      if (exists)
        return true;
    }

    // Check zDNN limitations
    // TODO: Check tensor size NNPA_MAXIMUM_TENSOR_SIZE of another limitation
    bool exceedLimit =
        llvm::any_of(genericOp->getOperands(), [](mlir::Value operand) {
          if (auto valueType = operand.getType().dyn_cast<mlir::ShapedType>()) {
            // Check if static dimension size exceeds zDNN limitations
            llvm::ArrayRef<int64_t> valueShape = valueType.getShape();
            if (llvm::any_of(valueShape, [](int64_t dim) {
                  return (dim != -1) &&
                         (dim > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE);
                }))
              return true;
          }
          return false;
        });
    if (exceedLimit)
      return true;

    return !isSuitableForZDNN<OP_TYPE>(op);
  });
}

/// Get transposed tensor by using a permutation array.
mlir::Value emitONNXTranspose(mlir::Location loc,
    mlir::PatternRewriter &rewriter, mlir::Value x,
    mlir::ArrayRef<int64_t> perms);

/// Get transposed tensor by using a permutation array and a result type.
mlir::Value emitONNXTransposeWithType(mlir::Location loc,
    mlir::PatternRewriter &rewriter, mlir::Type transposedType, mlir::Value x,
    mlir::ArrayRef<int64_t> perms);
