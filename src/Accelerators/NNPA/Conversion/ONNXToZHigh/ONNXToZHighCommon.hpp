/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ONNXToZHighCommon.hpp - Common functions in ONNXToZHigh ---------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in ONNX to ZHigh lowering.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_COMMON_H
#define ONNX_MLIR_ZHIGH_COMMON_H

#include "llvm/ADT/STLExtras.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

const std::string DEVICE_ATTRIBUTE = "device";
const std::string CPU_DEVICE = "cpu";
const std::string NNPA_DEVICE = "nnpa";

bool isEnableScalarBcastBinary();

// Populated by configureONNXToZHighLoweringPass().
struct ONNXToZHighLoweringConfiguration {
  static int optReportNNPAUnsupportedOps;
  static int reportOnNNPAUnsupportedOps;
  static bool isDynQuant;
  struct Quant {
    static bool isActivationSym;
    static bool isWeightSym;
    static llvm::SmallVector<std::string> opTypes;
  };
};

template <typename OP_TYPE>
void addDynamicallyLegalOpFor(mlir::ConversionTarget *target,
    const onnx_mlir::DimAnalysis *dimAnalysis,
    llvm::function_ref<bool(OP_TYPE, const DimAnalysis *)> checkLegalityFn =
        nullptr) {
  target->addDynamicallyLegalOp<OP_TYPE>([dimAnalysis, checkLegalityFn](
                                             OP_TYPE op) {
    mlir::Operation *genericOp = op.getOperation();
    mlir::StringAttr device =
        genericOp->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
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
                    mlir::dyn_cast<mlir::ShapedType>(operand.getType())) {
              // Check if static dimension size exceeds zDNN limitations
              llvm::ArrayRef<int64_t> valueShape = valueType.getShape();
              int64_t valueRank = valueShape.size();
              for (int64_t i = 0; i < valueRank; ++i) {
                int64_t dim = valueShape[i];
                if (!mlir::ShapedType::isDynamic(dim) &&
                    dim > NNPAGetMaxForDim(i, valueRank))
                  return true;
              }
            }
            return false;
          });
      if (exceedLimit)
        onnxToZHighUnsupportedReport(
            op.getOperation(), "Exceed maximum dimension index size.");
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

/// Split a tensor along an axis in which each chunk has a size of
/// NNPAGetMaxForDim and the last chuck can be smaller.
mlir::ValueRange splitAlongAxis(
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> &create,
    mlir::Value X, int64_t axis);

// Check if a value is a constant tensor of a single f32 value or not.
bool isF32ScalarConstantTensor(mlir::Value v);

// Get FloatAttr from a constant tensor of a single f32 value.
mlir::FloatAttr getScalarF32AttrFromConstant(mlir::Value v);

// Emit ONNX Concat to store the shape of the input x.
mlir::Value getDynShape(
    mlir::Location loc, mlir::PatternRewriter &rewriter, mlir::Value x);

} // namespace onnx_mlir
#endif
