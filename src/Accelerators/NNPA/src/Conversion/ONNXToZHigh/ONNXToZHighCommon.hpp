//===---------- ONNXToZHigh.hpp - Common functions in ONNXToZHigh ---------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in ONNX to ZHigh lowering.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Support/LayoutHelper.hpp"

template <typename OP_TYPE>
void addDynamicallyLegalOpFor(mlir::ConversionTarget *target,
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  target->addDynamicallyLegalOp<OP_TYPE>([execNodesOnCpu](OP_TYPE op) {
    // Check operations to be forced to run on CPU.
    Operation *genericOp = op.getOperation();
    StringAttr nodeName =
        genericOp->getAttrOfType<::mlir::StringAttr>("onnx_node_name");
    if (nodeName) {
      bool exists = llvm::any_of(execNodesOnCpu, [nodeName](StringRef val) {
        return nodeName.getValue().equals_insensitive(val);
      });
      if (exists)
        return true;
    }
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
