/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to Recompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the Recomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
// TODO: This file is quite busy as the number of decomposing op is increasing.
// It is better to move decomposition of each operation into a separate file.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/Recompose.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "recompose"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
// #include "src/Transform/ONNX/ONNXRecompose.inc"

// Support for recognizing patterns. Detects if the operation "op" has an input
// operand number "matchThisOperandIndex" that is defined by an operation of
// type "OP". If that is the case, "matchOperand" will be set to that operand,
// and "matchOp" will be set to that op. For unary operations; write matches
// only on success.
//
// This call is formatted so that it mimic the operations that we are trying to
// match. For example:
//
// %norm = "onnx.Div"(%d, %stdDev)
// %normScaled = "onnx.Mul"(%norm, %scale)
//
// We can have a match like this (version below is for binary op):
//
//  if (!operandOfOpDefinedBy<ONNXDivOp>(mulOp, norm, scale, divOp, 0))
//
// Namely, test if the mul op has input operands 0 that is defined by a divide
// op. If it does, then set norm, scale, and divOp to their appropriate values.

template <typename OP>
bool operandOfOpDefinedBy(Operation *op, Value &matchOperand,
    Operation *&matchOp, int64_t matchThisOperandIndex = 0) {
  assert(matchThisOperandIndex >= 0 &&
         matchThisOperandIndex < op->getNumOperands() &&
         "bad match operand index");
  Value operand = op->getOperand(matchThisOperandIndex);
  // operand.dump();
  //  Check for a match with definition of operand.
  if (!operand.isa<BlockArgument>() && isa<OP>(operand.getDefiningOp())) {
    matchOperand = operand;
    matchOp = operand.getDefiningOp();
    return true;
  }
  return false;
}

// Similar as above, for binary operation; write matches only on success.
template <typename OP>
bool operandOfOpDefinedBy(Operation *op, Value &matchOperand0,
    Value &matchOperand1, Operation *&matchOp, int64_t matchThisOperandIndex) {
  Value dummy;
  if (operandOfOpDefinedBy<OP>(op, dummy, matchOp, matchThisOperandIndex)) {
    matchOperand0 = op->getOperand(0);
    matchOperand1 = op->getOperand(1);
    return true;
  }
  return false;
}

struct RecomposeLayerNormFromAddPattern : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Location loc = addOp.getLoc();
    // Match
    Value x, scale, bias;
    FloatAttr epsilon;
    int64_t axis;
    if (!matchLayerNormPattern(addOp, x, scale, bias, axis, epsilon))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Type xType = x.getType();
    Value res = create.onnx.layerNorm(xType, x, scale, bias, axis, epsilon);
    rewriter.replaceOp(addOp, res);
    return success();
  }

  static bool matchLayerNormPattern(ONNXAddOp layerNormOp, Value &x,
      Value &scale, Value &bias, int64_t &axis, FloatAttr &epsilonAttr) {
    Value y;
    Operation *yLayerNormOp, *ywbLayerNormOp;
    ywbLayerNormOp = layerNormOp;
    // %noBias = "onnx.NoValue"()
    // %y, %mean, %invStdDev = "onnx.LayerNormalization"(%x, %scale, %noBias)
    //     {axis = 2 : si64, epsilon = 9.994E-6 : f32, stash_type = 1 : si64}
    // %yWithBias = "onnx.Add"(%y, %bias)
    if (!operandOfOpDefinedBy<ONNXLayerNormalizationOp>(
            ywbLayerNormOp, y, bias, yLayerNormOp, 0) &&
        !operandOfOpDefinedBy<ONNXLayerNormalizationOp>(
            ywbLayerNormOp, bias, y, yLayerNormOp, 1))
      return reportFailure("missing y, layer norm op");
    // Study layer norm op; make sure its used only one and that bias is not
    // used.
    if (!yLayerNormOp->hasOneUse())
      return reportFailure("y/layer norm has too many uses");
    auto lnOp = cast<ONNXLayerNormalizationOp>(yLayerNormOp);
    if (!onnx_mlir::isNoneValue(lnOp.getB()))
      return reportFailure("layer norm already has a bias");
    // We are fine.
    x = lnOp.getX();
    scale = lnOp.getScale();
    epsilonAttr = lnOp.getEpsilonAttr();
    axis = lnOp.getAxis();
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm from add, axis : " << axis << "\n");
    return true;
  }

private:
  static bool reportFailure(std::string msg) {
    // Can disable line below if not needed.
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm failure:" << msg << "\n");
    return false;
  }
};

struct RecomposeLayerNormFromMulPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Location loc = mulOp.getLoc();
    // Match
    Value x, scale;
    FloatAttr epsilon;
    int64_t axis;
    if (!matchLayerNormPattern(mulOp, x, scale, axis, epsilon))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Type xType = x.getType();
    Value noneVal = create.onnx.none();
    Value res = create.onnx.layerNorm(xType, x, scale, noneVal, axis, epsilon);
    rewriter.replaceOp(mulOp, res);
    return success();
  }

  static bool matchLayerNormPattern(ONNXMulOp LayerNormOp, Value &x,
      Value &scale, int64_t &axis, FloatAttr &epsilonAttr) {
    // Values that will be gathered and only kept locally.
    Value norm, stdDev, varEps, var, epsilon, dd, d, mean;
    // Replicate of values, check that they are identical to originals.
    Value d1, d2;
    // Operations that will be gathered and kept locally.
    Operation *nsMulOp, *ddMulOp, *nDivOp, *sdSqrtOp, *veAddOp, *vReduceOp,
        *mReduceOp, *dSubOp;
    nsMulOp = LayerNormOp.getOperation();
    // %norm = "onnx.Div"(%d, %stdDev)
    // %normScaled = "onnx.Mul"(%norm, %scale)
    if (!operandOfOpDefinedBy<ONNXDivOp>(nsMulOp, norm, scale, nDivOp, 0) &&
        !operandOfOpDefinedBy<ONNXDivOp>(nsMulOp, scale, norm, nDivOp, 1))
      return reportFailure("missing norm, div op");
    // %stdDev = "onnx.Sqrt"(%varEps)
    // %norm = "onnx.Div"(%d, %stdDev)
    if (!operandOfOpDefinedBy<ONNXSqrtOp>(nDivOp, d, stdDev, sdSqrtOp, 1))
      return reportFailure("missing std dev, sqrt op");
    // %varEps = "onnx.Add"(%var, %eps)
    // %stdDev = "onnx.Sqrt"(%varEps)
    if (!operandOfOpDefinedBy<ONNXAddOp>(sdSqrtOp, varEps, veAddOp))
      return reportFailure("missing var + eps, add op");
    // %var = "onnx.ReduceMeanV13"(%dd)
    // %varEps = "onnx.Add"(%var, %eps)
    if (!operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
            veAddOp, var, epsilon, vReduceOp, 0) &&
        !operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
            veAddOp, epsilon, var, vReduceOp, 1))
      return reportFailure("missing var, reduce mean op");
    // %dd = "onnx.Mul"(%d, %d)
    // %var = "onnx.ReduceMeanV13"(%dd)
    if (!operandOfOpDefinedBy<ONNXMulOp>(vReduceOp, dd, ddMulOp))
      return reportFailure("missing DD, mul op");
    // %d = "onnx.Sub"(%X, %mean)
    // %dd = "onnx.Mul"(%d, %d)
    if (!operandOfOpDefinedBy<ONNXSubOp>(ddMulOp, d1, d2, dSubOp, 0))
      return reportFailure("missing D, sub op");
    if (!operandOfOpDefinedBy<ONNXSubOp>(ddMulOp, d1, d2, dSubOp, 1))
      return reportFailure("missing D, sub op");
    if (d != d1 || d != d2)
      return reportFailure("Various versions of d do not match");
    // %mean = "onnx.ReduceMeanV13"(%x)
    // %d = "onnx.Sub"(%X, %mean)
    if (!operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
            dSubOp, x, mean, mReduceOp, 1))
      return reportFailure("missing mean, reduce mean op");
    // Verify that the mReduceOp uses x as well.
    auto lnOp = cast<ONNXReduceMeanV13Op>(mReduceOp);
    Value x2 = lnOp.getData();
    if (x != x2)
      return reportFailure("input x to mean/ReduceMean and sub are different");
    // Check the number of uses, for now only the 1 uses.
    if (!mReduceOp->hasOneUse())
      return reportFailure("mean reduce has too many uses");
    // d = sub has 3 uses from 2 distinct ops, ignore test for now.
    if (!ddMulOp->hasOneUse())
      return reportFailure("dd/mul has too many uses");
    if (!vReduceOp->hasOneUse())
      return reportFailure("var reduce has too many uses");
    if (!veAddOp->hasOneUse())
      return reportFailure("var eps add has too many uses");
    if (!sdSqrtOp->hasOneUse())
      return reportFailure("std dev sqrt has too many uses");
    if (!nDivOp->hasOneUse())
      return reportFailure("norm div has too many uses");
    if (!nsMulOp->hasOneUse())
      return reportFailure("norm scale mul has too many uses");

    // Now check values epsilon.
    if (!onnx_mlir::isScalarTensor(epsilon))
      return reportFailure("epsilon is expected to be scalar");
    ONNXConstantOp epsilonOp =
        dyn_cast<ONNXConstantOp>(epsilon.getDefiningOp());
    if (!epsilonOp)
      return reportFailure("epsilon needs to be a constant");
    epsilonAttr = epsilonOp.getValueFloatAttr();
    // Check axes.
    if (!onnx_mlir::hasShapeAndRank(x))
      return reportFailure("need rank and shape for input x");
    int64_t xRank = x.getType().cast<ShapedType>().getRank();
    int64_t meanAxis, varAxis;
    if (!suitableAxis(mReduceOp, xRank, meanAxis))
      return reportFailure("unsuitable mean reduce axes");
    if (!suitableAxis(vReduceOp, xRank, varAxis))
      return reportFailure("unsuitable var reduce axes");
    if (meanAxis != varAxis)
      return reportFailure("mean and var axes must be the same");
    // Axis is fine
    axis = meanAxis;
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm from mult, axis : " << axis << "\n");
    return true;
  }

private:
  static bool suitableAxis(Operation *op, int64_t xRank, int64_t &axis) {
    ONNXReduceMeanV13Op reduceOp = cast<ONNXReduceMeanV13Op>(op);
    if (reduceOp.getKeepdims() != 1)
      return reportFailure("need keepdims = 1");
    ArrayAttr axesAttr = reduceOp.getAxesAttr();
    int64_t axesSize = axesAttr.size();
    // Record axes value in bit vector.
    llvm::SmallBitVector reduceAxes(xRank, false);
    for (int64_t i = 0; i < axesSize; ++i) {
      int64_t a = onnx_mlir::getAxisInRange(
          onnx_mlir::ArrayAttrIntVal(axesAttr, i), xRank);
      reduceAxes[a] = true;
    }
    // Check that we have a "false"* "true"+ pattern.
    bool foundFirstAxis = false;
    for (int64_t i = 0; i < xRank; ++i) {
      if (!foundFirstAxis) {
        if (reduceAxes[i]) {
          foundFirstAxis = true;
          axis = i;
        }
      } else if (!reduceAxes[i]) {
        // Once we found an axis, we must reduce all subsequent dimensions.
        return false;
      }
    }
    // Ensure we had at least one reduction.
    return foundFirstAxis;
  }

  static bool reportFailure(std::string msg) {
    // Can disable line below if not needed.
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm failure:" << msg << "\n");
    return false;
  }
};

struct RecomposeONNXToONNXPass
    : public PassWrapper<RecomposeONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecomposeONNXToONNXPass)

  RecomposeONNXToONNXPass(const std::string &target) { this->target = target; }
  RecomposeONNXToONNXPass(const RecomposeONNXToONNXPass &pass)
      : mlir::PassWrapper<RecomposeONNXToONNXPass,
            OperationPass<func::FuncOp>>() {
    this->target = pass.target.getValue();
  }

  StringRef getArgument() const override { return "recompose-onnx"; }

  StringRef getDescription() const override {
    return "Recompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  Option<std::string> target{*this, "target",
      llvm::cl::desc("Target Dialect to Recompose into"), ::llvm::cl::init("")};

  void runOnOperation() final;

  typedef PassWrapper<RecomposeONNXToONNXPass, OperationPass<func::FuncOp>>
      BaseType;
};

void RecomposeONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

  // These ops will be Recomposed into other ONNX ops. Hence, they will not be
  // available after this pass.

#ifndef SKIP_RECOMPOSE_LAYER_NORM
  // Recompose LayerNorm, starting from scale/mul op
  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    Value x, scale;
    FloatAttr epsilon;
    int64_t axis;
    return !RecomposeLayerNormFromMulPattern::matchLayerNormPattern(
        op, x, scale, axis, epsilon);
  });
  // Recompose LayerNorm, starting from bias/add op
  target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
    Value x, scale, bias;
    FloatAttr epsilon;
    int64_t axis;
    return !RecomposeLayerNormFromAddPattern::matchLayerNormPattern(
        op, x, scale, bias, axis, epsilon);
  });
#endif

  RewritePatternSet patterns(context);
  onnx_mlir::getRecomposeONNXToONNXPatterns(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getRecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
#ifndef SKIP_RECOMPOSE_LAYER_NORM
  patterns.insert<RecomposeLayerNormFromMulPattern>(context);
  patterns.insert<RecomposeLayerNormFromAddPattern>(context);
#endif

  // TODO: consider whether to include SoftmaxPattern here
}

/*!
 * Create a RecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createRecomposeONNXToONNXPass(
    const std::string &target) {
  return std::make_unique<RecomposeONNXToONNXPass>(target);
}
