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

#pragma once

#include "llvm/ADT/STLExtras.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

const std::string CPU_DEVICE = "cpu";
const std::string NNPA_DEVICE = "nnpa";

template <typename OP_TYPE>
void addDynamicallyLegalOpFor(mlir::ConversionTarget *target,
    const onnx_mlir::DimAnalysis *dimAnalysis,
    llvm::function_ref<bool(OP_TYPE, const DimAnalysis *)> checkLegalityFn =
        nullptr) {
  target->addDynamicallyLegalOp<OP_TYPE>([dimAnalysis, checkLegalityFn](
                                             OP_TYPE op) {
    mlir::Operation *genericOp = op.getOperation();
    mlir::StringAttr device =
        genericOp->getAttrOfType<mlir::StringAttr>("device");
    assert((!device ||
               (device &&
                   (device.getValue().equals_insensitive("") ||
                       device.getValue().equals_insensitive(CPU_DEVICE) ||
                       device.getValue().equals_insensitive(NNPA_DEVICE)))) &&
           "Invalid device name");

    // If device is CPU, force to run the op on CPU.
    if (device && device.getValue().equals_insensitive(CPU_DEVICE))
      return true;

    // If not CPU, check if the op is legal for NNPA.
    bool isLegalForNNPA = false;
    if (checkLegalityFn)
      isLegalForNNPA = !checkLegalityFn(op, dimAnalysis);
    else {
      // Check zDNN limitations for each input tensors.
      // TODO: Check tensor size NNPA_MAXIMUM_TENSOR_SIZE of another limitation
      bool exceedLimit =
          llvm::any_of(genericOp->getOperands(), [](mlir::Value operand) {
            if (auto valueType =
                    operand.getType().dyn_cast<mlir::ShapedType>()) {
              // Check if static dimension size exceeds zDNN limitations
              llvm::ArrayRef<int64_t> valueShape = valueType.getShape();
              if (llvm::any_of(valueShape, [](int64_t dim) {
                    return (!mlir::ShapedType::isDynamic(dim)) &&
                           (dim > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE);
                  }))
                return true;
            }
            return false;
          });
      isLegalForNNPA =
          !exceedLimit && isSuitableForZDNN<OP_TYPE>(op, dimAnalysis);
    }

    // Users specified NNPA device of an op, but the compiler found the op is
    // not legal for NNPA, e.g. in case of dynamic shape.  In this case, print
    // out a warning message.
    if (device && device.getValue().equals_insensitive(NNPA_DEVICE) &&
        !isLegalForNNPA) {
      llvm::outs() << "Warning: though the following operation was specified "
                      "to run on NNPA, the compiler found that NNPA did not "
                      "support that operation. It's potentially that the "
                      "compiler was not able to check broadcasting in case of "
                      "dynamic shape so that it thought the operation was not "
                      "legal for NNPA.\n";
      op.dump();
      return false;
    }

    return !isLegalForNNPA;
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

/// Split a tensor along an axis in which each chunk has a size of
/// NNPA_MAXIMUM_DIMENSION_INDEX_SIZE and the last chucnk can be smaller.
mlir::ValueRange splitAlongAxis(
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> &create,
    mlir::Value X, int64_t axis);

} // namespace onnx_mlir
