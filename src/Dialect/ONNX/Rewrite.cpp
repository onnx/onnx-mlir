/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Rewrite.cpp - ONNX High Level Optimizer ----------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include <math.h>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

// =============================================================================
// Rewrite pattern for elementwise binary ops.
// =============================================================================

// Rewrite the following pattern
//
// %0 = onnx.Constant
// %1 = onnx.ConstantOfShape
// %2 = "onnx.Sub"(%0, %1) : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//
// into
// %0 = onnx.ConstantOfShape
//
// This pattern only handles the case where one of the operand is a scalar
// constant. For such a case, we can easily infer the shape operand for the
// resulting ConstantOfShape.
template <typename OP_TYPE>
class ReplaceBinaryOpByConstantOfShapePattern
    : public OpRewritePattern<OP_TYPE> {
public:
  using OpRewritePattern<OP_TYPE>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OP_TYPE binaryOp, PatternRewriter &rewriter) const override {
    Operation *op = binaryOp.getOperation();
    Location loc = binaryOp.getLoc();

    assert(op->getNumOperands() == 2 && "op must be binary");
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type outputType = op->getResult(0).getType();

    // Match
    // lhs is ConstantOp of a single scalar value and rhs is ConstantOfShapeOp,
    // or vice versa.
    ONNXConstantOp constantOp;
    ONNXConstantOfShapeOp constantOfShapeOp;
    auto matchValue = [&](Value v) -> ElementsAttr {
      if (definedBy<ONNXConstantOp>(v)) {
        int64_t rank = getRank(v.getType());
        ArrayRef<int64_t> shape = getShape(v.getType());
        if (rank == 0 || (rank == 1 && shape[0] == 1)) {
          auto cOp = cast<ONNXConstantOp>(v.getDefiningOp());
          if (cOp.getValue().has_value()) {
            constantOp = cOp;
            return dyn_cast_or_null<ElementsAttr>(cOp.getValue().value());
          }
        }
      } else if (definedBy<ONNXConstantOfShapeOp>(v)) {
        auto cosOp = cast<ONNXConstantOfShapeOp>(v.getDefiningOp());
        if (cosOp.getValue().has_value()) {
          constantOfShapeOp = cosOp;
          return dyn_cast_or_null<ElementsAttr>(cosOp.getValue().value());
        }
      }
      return nullptr;
    };
    ElementsAttr lhsAttr = matchValue(lhs);
    ElementsAttr rhsAttr = matchValue(rhs);
    if (!lhsAttr || !rhsAttr || !constantOp || !constantOfShapeOp)
      return failure();

    // Rewrite
    // Get scalar values from ConstantOp and ConstantOfShape.
    DenseElementsAttr resAttr;
    Type elementType = lhsAttr.getType().cast<ShapedType>().getElementType();
    if (isa<IntegerType>(elementType)) {
      int64_t lhs = getScalarValue<int64_t>(lhsAttr, elementType);
      int64_t rhs = getScalarValue<int64_t>(rhsAttr, elementType);
      int64_t res = ElementWiseBinaryOpImpl<OP_TYPE, int64_t>::eval(lhs, rhs);
      resAttr = DenseElementsAttr::get(
          RankedTensorType::get({1}, elementType), ArrayRef<int64_t>({res}));
    } else if (isa<FloatType>(elementType)) {
      double lhs = getScalarValue<double>(lhsAttr, elementType);
      double rhs = getScalarValue<double>(rhsAttr, elementType);
      float res =
          (float)ElementWiseBinaryOpImpl<OP_TYPE, double>::eval(lhs, rhs);
      resAttr = DenseElementsAttr::get(
          RankedTensorType::get({1}, elementType), ArrayRef<float>({res}));
    } else
      llvm_unreachable("Unexpected type.");

    Value res =
        OnnxBuilder(rewriter, loc)
            .constantOfShape(outputType, resAttr, constantOfShapeOp.getInput());

    rewriter.replaceOp(op, {res});
    return success();
  }
};

// =============================================================================
// Rewrite pattern for unsqueeze.
// =============================================================================

// Rewrite a pattern like the following:
//
// %shape = onnx.Concat(%dim1, %dim2)
// %axes = onnx.Constant {value = 2}
// %data = onnx.ConstantOfShape(%shape) {value = 1.0}
// %u = "onnx.Unsqueeze"(%data, %axes)
//
// into
//
// %new_shape = onnx.Concat(%dim1, %dim2, 1)
// %u = onnx.ConstantOfShape(%new_shape) {value = 1.0}
class ReplaceUnsqueezeOfConstantOfShapeRewritePattern
    : public OpRewritePattern<ONNXUnsqueezeOp> {
public:
  using OpRewritePattern<ONNXUnsqueezeOp>::OpRewritePattern;

  ReplaceUnsqueezeOfConstantOfShapeRewritePattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(
      ONNXUnsqueezeOp unsqueezeOp, PatternRewriter &rewriter) const override {
    Operation *op = unsqueezeOp.getOperation();
    Location loc = unsqueezeOp.getLoc();
    Value data = unsqueezeOp.getData();
    Value axes = unsqueezeOp.getAxes();

    // Match
    // 1. data is from ConstantOfShapeOp, axes is from ConstantOp.
    if (!definedBy<ONNXConstantOfShapeOp>(data) ||
        !definedBy<ONNXConstantOp>(axes))
      return failure();
    auto constantOfShapeOp = cast<ONNXConstantOfShapeOp>(data.getDefiningOp());
    // 2. ConstantOfShapeOp has value.
    if (!constantOfShapeOp.getValue().has_value())
      return failure();
    // 3. ConstantOfShapeOp's shape is defined by dimensions.
    if (!areDims(constantOfShapeOp.getInput()))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    // Get the old shape.
    SmallVector<Value, 4> oldDims;
    getDims(constantOfShapeOp.getInput(), oldDims);
    int64_t oldRank = oldDims.size();
    // Get unsqueeze axes.
    ElementsAttr axesAttrs = getElementAttributeFromONNXValue(axes);
    SmallVector<int64_t> axesI64(axesAttrs.getValues<int64_t>());
    for (unsigned int i = 0; i < axesI64.size(); ++i)
      if (axesI64[i] < 0)
        axesI64[i] += oldRank;

    // Construct a new shape.
    SmallVector<Value, 4> newDims;
    int64_t newRank = oldRank + axesI64.size();
    Value one = create.onnx.constantInt64(ArrayRef<int64_t>({1}));
    for (int64_t i = 0, j = 0; i < newRank || j < oldRank; ++i)
      if (std::find(axesI64.begin(), axesI64.end(), i) != axesI64.end())
        // found i in unsqueeze axes.
        newDims.emplace_back(one);
      else
        // original axes.
        newDims.emplace_back(oldDims[j++]);
    Value newShape = create.onnx.concat(
        RankedTensorType::get({newRank}, rewriter.getI64Type()), newDims, 0);

    Value res = create.onnx.constantOfShape(op->getResult(0).getType(),
        constantOfShapeOp.getValue().value(), newShape);
    rewriter.replaceOp(op, {res});
    return success();
  };
};

namespace {

struct RewriteONNXToONNXPass
    : public PassWrapper<RewriteONNXToONNXPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteONNXToONNXPass)

  StringRef getArgument() const override { return "onnx-rewrite"; }

  StringRef getDescription() const override {
    return "Perform rewriting ONNX operators into a better IR";
  }

  void runOnOperation() final;
};

void RewriteONNXToONNXPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  // Define patterns.
  RewritePatternSet patterns(context);
  patterns.insert<ReplaceBinaryOpByConstantOfShapePattern<ONNXAddOp>>(context);
  patterns.insert<ReplaceBinaryOpByConstantOfShapePattern<ONNXDivOp>>(context);
  patterns.insert<ReplaceBinaryOpByConstantOfShapePattern<ONNXMulOp>>(context);
  patterns.insert<ReplaceBinaryOpByConstantOfShapePattern<ONNXSubOp>>(context);
  patterns.insert<ReplaceUnsqueezeOfConstantOfShapeRewritePattern>(context);

  // Apply patterns.
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a RewriteONNXToONNX pass.
 */
std::unique_ptr<mlir::Pass> createRewriteONNXToONNXPass() {
  return std::make_unique<RewriteONNXToONNXPass>();
}

} // namespace onnx_mlir
