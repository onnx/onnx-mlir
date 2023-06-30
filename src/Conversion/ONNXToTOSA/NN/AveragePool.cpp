/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXAveragePoolOp.cpp - ONNXAveragePoolOp --------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXAveragePoolOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

void handleIncludePadAttr(
    ConversionPatternRewriter &rewriter, Operation *op, Value input) {
  mlir::Location loc = op->getLoc();

  // Get shape.
  IndexExprBuilderForTosa createTosaIE(rewriter, loc);
  ONNXGenericPoolOpShapeHelper<ONNXAveragePoolOp> shapeHelper(
      op, {}, &createTosaIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  // Get onnx padding. Ignore ceil mode since it is handled later.
  mlir::ArrayAttr pads = tosa::createOrderedPadAttr(rewriter,
      input.getType().cast<mlir::TensorType>().getShape(), shapeHelper,
      /*ceilMode*/ 0, {0, 1, 2, 3});

  // Build an array with padding.
  llvm::SmallVector<int64_t, 4> intValues;
  llvm::for_each(pads.getAsRange<IntegerAttr>(),
      [&](mlir::IntegerAttr n) { intValues.push_back(n.getInt()); });

  // Create Padding and ConstPad tosa::ConstOp's
  TosaBuilder tosaBuilder(rewriter, loc);
  Value padding = tosa::buildOnnxToTosaPaddingConstOp(
      rewriter, intValues, loc, {0, 0, 0, 0}, {});
  auto constTosaTensor = tosaBuilder.getSplattedConst(0.0);

  auto inputType = input.getType().cast<mlir::TensorType>();
  auto padOp = tosa::CreateOpAndInfer<mlir::tosa::PadOp>(rewriter, loc,
      mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 4>(
              inputType.getShape().size(), ShapedType::kDynamic),
          inputType.getElementType()),
      input, padding, constTosaTensor);

  // In-place update of AveragePool by setting operand to PadOp
  // and pads attribute to {0, 0, 0, 0}.
  op->setOperand(0, padOp);
  op->setAttr("pads", rewriter.getI32ArrayAttr({0, 0, 0, 0}));
}

class ONNXAveragePoolOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXAveragePoolOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXAveragePoolOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXAveragePoolOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto averagePoolOp = llvm::cast<ONNXAveragePoolOp>(op);
    OpAdaptor adaptor(operands, op->getAttrDictionary());

    const int64_t includePad = adaptor.getCountIncludePad();

    if (includePad != 0) {
      // When attribute include_pad is set, create a tosa::PadOp before lowering
      // AveragePool to TOSA. We use ONNX format for it so that the AveragePool
      // lowering still generates transposes between ONNX and TOSA formats, and
      // implementation doesn't diverge much.
      handleIncludePadAttr(rewriter, op, adaptor.getX());
    }

    std::optional<Value> newAveragePoolOp =
        tosa::convertPoolOp<ONNXAveragePoolOp, mlir::tosa::AvgPool2dOp>(
            rewriter, op);

    if (!newAveragePoolOp) {
      return rewriter.notifyMatchFailure(
          averagePoolOp, "Could not create averagepool op.");
    }

    rewriter.replaceOp(op, newAveragePoolOp.value());
    return success();
  }
};

} // namespace

void populateLoweringONNXAveragePoolOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXAveragePoolOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
