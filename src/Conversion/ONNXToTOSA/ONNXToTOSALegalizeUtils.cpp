/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.cpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp" // from @llvm-project
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;
namespace onnx_mlir {
namespace tosa {

int64_t convertNegativeAxis(int64_t axis, int64_t inputRank) {
  if (axis < 0)
    axis += inputRank;

  // Check if axis is in correct range.
  assert(
      (axis >= 0 && axis < inputRank) && "axis attribute not in correct range");

  return axis;
}

llvm::SmallVector<int64_t> createInt64VectorFromIndexExpr(
    llvm::ArrayRef<IndexExpr> indexVector) {
  llvm::SmallVector<int64_t, 4> literalVector(indexVector.size());
  llvm::transform(indexVector, literalVector.begin(),
      [](const auto &indexExpr) { return indexExpr.getLiteral(); });
  return literalVector;
}

mlir::RankedTensorType reduceAxisToOne(llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType, mlir::Attribute encoding) {
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), 1), elementType, encoding);
}

mlir::ElementsAttr getElementsAttrFromConst(mlir::Value &val) {
    if (auto source = val.getDefiningOp<mlir::ONNXConstantOp>()) {
      if (source.value())
        return source.value().value(); 
    }
    // if the constant is not an onnx.const it has to be a tosa.const
    assert(val.getDefiningOp<mlir::tosa::ConstOp>());
    return tosa::getValueFromTosaConst<ElementsAttr>(val);
}

// Create a TOSA rescale op from input framework tensor, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
    ShapedType output_type, Value input_val, double scale, int64_t input_zp,
    int64_t output_zp, bool double_round, bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  mlir::tosa::computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  auto rescale_op = CreateOpAndInfer<mlir::tosa::RescaleOp>(rewriter,
      op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getI32ArrayAttr({multiplier}), rewriter.getI32ArrayAttr({shift}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
    Value input_val, double input_scale, int64_t input_zp) {
  // Output is always int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
      input_zp, 0, false, true);
}

} // namespace tosa
} // namespace onnx_mlir
