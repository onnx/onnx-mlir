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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.h"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

#include "mlir/Dialect/Async/IR/Async.h"

namespace onnx_mlir {

const std::string DEVICE_ATTRIBUTE = "device";
const std::string CPU_DEVICE = "cpu";
const std::string NNPA_DEVICE = "nnpa";

template <typename OP_TYPE>
void addDynamicallyLegalOpFor(mlir::ConversionTarget *target,
    const onnx_mlir::DimAnalysis *dimAnalysis,
    llvm::function_ref<bool(OP_TYPE, const DimAnalysis *, int, int)>
        checkLegalityFn = nullptr,
    int nnpaParallelNdev = 0, int nnpaParallelMinimumDimThreshold = 0) {
  target->addDynamicallyLegalOp<OP_TYPE>(
      [dimAnalysis, checkLegalityFn, nnpaParallelNdev,
          nnpaParallelMinimumDimThreshold](OP_TYPE op) {
        mlir::Operation *genericOp = op.getOperation();
        mlir::StringAttr device =
            genericOp->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
        assert(
            (!device ||
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
          isLegalForNNPA = !checkLegalityFn(op, dimAnalysis, nnpaParallelNdev,
              nnpaParallelMinimumDimThreshold);
        else {
          // Check zDNN limitations for each input tensors.
          // TODO: Check tensor size NNPA_MAXIMUM_TENSOR_SIZE of another
          // limitation
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

/// Split a tensor along an axis by chunkSize. The last chunk becomes smaller
/// than it. The default chunkSize is NNPA_MAXIMUM_DIMENSION_INDEX_SIZE.
mlir::ValueRange splitAlongAxis(
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> &create,
    mlir::Value X, int64_t axis,
    int64_t chunkSize = NNPA_MAXIMUM_DIMENSION_INDEX_SIZE);

// Check if a value is a constant tensor of a single f32 value or not.
bool isF32ScalarConstantTensor(mlir::Value v);

// Get FloatAttr from a constant tensor of a single f32 value.
mlir::FloatAttr getScalarF32AttrFromConstant(mlir::Value v);

// Emit ONNX Concat to store the shape of the input x.
mlir::Value getDynShape(
    mlir::Location loc, mlir::PatternRewriter &rewriter, mlir::Value x);

} // namespace onnx_mlir
