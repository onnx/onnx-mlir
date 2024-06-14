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

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/Recompose.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "recompose"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
// #include "src/Dialect/ONNX/Transforms/ONNXRecompose.inc"

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
    bool isRMSLayerNorm;
    if (!matchLayerNormPattern(mulOp, x, scale, axis, epsilon, isRMSLayerNorm))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Type xType = x.getType();
    Value noneVal = create.onnx.none();
    Value res;
    if (isRMSLayerNorm)
      res = create.onnx.RMSLayerNorm(xType, x, scale, noneVal, axis, epsilon);
    else
      res = create.onnx.layerNorm(xType, x, scale, noneVal, axis, epsilon);
    rewriter.replaceOp(mulOp, res);
    return success();
  }

  /*
   * Primary LayerNormalization pattern being matched:

     mean = reduceMean(X)
     D = X - mean
     var = reduceMean(D*D)
     stdDev = sqrt(var + eps)
     y = mul(scale, D / stdDev)


  * Secondary pattern associated with RMSLayerNormalization:

    var = reduceMean(X * X)
    stdDev = sqrt(var + eps)
    Y = mul(scale, X / stdDev)

  As it can be seen here, the RMS LN pattern matches the traditional LN for
  the bottom 3 statements. In RMS LN, X is the raw input, whereas in the
  traditional LN, the input to the lower 3 statements are D = X - mean(X).

  * Variations around the div (for both patterns):

     D / stdDev
     D * (1 / stdDev)
     D * recip(stdDev)

  */
  static bool matchLayerNormPattern(ONNXMulOp LayerNormOp, Value &x,
      Value &scale, int64_t &axis, FloatAttr &epsilonAttr,
      bool &isRMSLayerNorm) {
    using namespace onnx_mlir;
    Location loc = LayerNormOp.getLoc();
    isRMSLayerNorm = false;

    // 1: Start first to detect if we have the common layer norm pattern.
    // Values that will be gathered and only kept locally.
    Value norm, invStdDev, stdDev, varEps, var, epsilon, dd, d, mean, x1;
    // Replicate of values, check that they are identical to originals.
    Value d1, d2;
    // Operations that will be gathered and kept locally.
    Operation *nsMulOp, *ddMulOp, *nDivOp, *nMulOp, *isdRecipOp, *sdSqrtOp,
        *veAddOp, *vReduceOp, *mReduceOp, *dSubOp;
    nsMulOp = LayerNormOp.getOperation();
    // after this group, we have defined norm, scale, d, and sdSqrtOp.
    nDivOp = nMulOp = isdRecipOp = nullptr;
    if (operandOfOpDefinedBy<ONNXDivOp>(nDivOp, nsMulOp, norm, scale, 0) ||
        operandOfOpDefinedBy<ONNXDivOp>(nDivOp, nsMulOp, scale, norm, 1)) {
      // Matched norm = d / stdDev.
      // %norm = "onnx.Div"(%d, %stdDev)
      // %normScaled = "onnx.Mul"(%norm, %scale)

      // Now search for the sqrt.
      // %stdDev = "onnx.Sqrt"(%varEps)
      // %norm = "onnx.Div"(%d, %stdDev)
      if (!operandOfOpDefinedBy<ONNXSqrtOp>(sdSqrtOp, nDivOp, d, stdDev, 1))
        return reportFailure("RMS missing std dev (via div), sqrt op");

    } else if (operandOfOpDefinedBy<ONNXMulOp>(
                   nMulOp, nsMulOp, norm, scale, 0) ||
               operandOfOpDefinedBy<ONNXMulOp>(
                   nMulOp, nsMulOp, scale, norm, 1)) {
      // Matched norm = d * (invStdDev).
      // %Norm = "onnx.Mul"(%d, %InvStdDev)
      // %NormScaled = "onnx.Mul"(%Norm, %scale)

      if (operandOfOpDefinedBy<ONNXReciprocalOp>(
              isdRecipOp, nMulOp, invStdDev, d, 0) ||
          operandOfOpDefinedBy<ONNXReciprocalOp>(
              isdRecipOp, nMulOp, d, invStdDev, 1)) {
        // Now matched the reciprocal.
        // %InvStdDev = "onnx.Reciprocal"(%StdDev)
        // %Norm = "onnx.Mul"(%d, %InvStdDev)

        // Now search for the sqrt.
        // %StdDev = "onnx.Sqrt"(%varEps)
        // %InvStdDev = "onnx.Reciprocal"(%StdDev)
        if (!operandOfOpDefinedBy<ONNXSqrtOp>(sdSqrtOp, isdRecipOp, stdDev)) {
          return reportFailure("RMS missing std dev (via reciprocal), sqrt op");
        }
      } else if (operandOfOpDefinedBy<ONNXDivOp>(
                     isdRecipOp, nMulOp, invStdDev, d, 0) ||
                 operandOfOpDefinedBy<ONNXDivOp>(
                     isdRecipOp, nMulOp, d, invStdDev, 1)) {
        // Now matched the div(1/stddev).
        // %InvStdDev = "onnx.Div"(%one, %StdDev)
        // %Norm = "onnx.Mul"(%d, %InvStdDev)

        // Now search for the sqrt.
        // %StdDev = "onnx.Sqrt"(%varEps)
        // %InvStdDev = "onnx.Reciprocal"(%StdDev)
        Value one;
        if (operandOfOpDefinedBy<ONNXSqrtOp>(
                sdSqrtOp, isdRecipOp, one, stdDev, 1)) {
          // Has a match, check that the value one is a scalar/tensor of size
          // one with value 1.
          IndexExprScope scope(nullptr, loc);
          IndexExprBuilderForAnalysis createIE(loc);
          if (createIE.hasShapeAndRank(one) &&
              createIE.getArraySize(one, /*static only*/ true) == 1) {
            IndexExpr floatVal = createIE.getFloatAsNonAffine(one);
            if (!floatVal.isLiteral() || (floatVal.getFloatLiteral() != 1.0))
              return reportFailure(
                  "RMS missing std dev (via 1/x), not div of 1.0");
          } else {
            return reportFailure(
                "missing std dev (via 1/x), not div of scalar");
          }
        } else {
          return reportFailure("RMS missing std dev (via 1/x), sqrt op");
        }
      } else {
        return reportFailure("RMS missing inv std dev, reciprocal op");
      }
    } else {
      return reportFailure("RMS missing norm, div or reciprocal op");
    }
    // %varEps = "onnx.Add"(%var, %eps)
    // %stdDev = "onnx.Sqrt"(%varEps)
    if (!operandOfOpDefinedBy<ONNXAddOp>(veAddOp, sdSqrtOp, varEps))
      return reportFailure("RMS missing var + eps, add op");
    // %var = "onnx.ReduceMeanV13"(%dd)
    // %varEps = "onnx.Add"(%var, %eps)
    if (!operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
            vReduceOp, veAddOp, var, epsilon, 0) &&
        !operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
            vReduceOp, veAddOp, epsilon, var, 1))
      return reportFailure("RMS missing var, reduce mean op");
    // %dd = "onnx.Mul"(%d, %d)
    // %var = "onnx.ReduceMeanV13"(%dd)
    if (!operandOfOpDefinedBy<ONNXMulOp>(ddMulOp, vReduceOp, dd))
      return reportFailure("RMS missing DD, mul op");

    // 2: Now we have the common pattern, make additional checks
    // Make sure that all of the d's matches.
    d1 = ddMulOp->getOperand(0);
    d2 = ddMulOp->getOperand(1);
    if (d != d1 || d != d2)
      return reportFailure("Various versions of d do not match");
    // Check one usages of the key computations.
    if (!ddMulOp->hasOneUse())
      return reportFailure("RMS dd/mul has too many uses");
    if (!vReduceOp->hasOneUse())
      return reportFailure("RMS var reduce has too many uses");
    if (!veAddOp->hasOneUse())
      return reportFailure("RMS var eps add has too many uses");
    if (!sdSqrtOp->hasOneUse())
      return reportFailure("RMS std dev sqrt has too many uses");
    // Gate the next 3 ops by being nonnull, as there are multiple paths.
    if (nDivOp && !nDivOp->hasOneUse())
      return reportFailure("RMS norm div has too many uses");
    if (nMulOp && !nMulOp->hasOneUse())
      return reportFailure("RMS norm mul has too many uses");
    if (isdRecipOp && !isdRecipOp->hasOneUse())
      return reportFailure("RMS norm recip has too many uses");
    if (!nsMulOp->hasOneUse())
      return reportFailure("RMS norm scale mul has too many uses");
    // Now check values epsilon.
    if (!isScalarTensor(epsilon))
      return reportFailure("RMS epsilon is expected to be scalar");
    ONNXConstantOp epsilonOp =
        dyn_cast<ONNXConstantOp>(epsilon.getDefiningOp());
    if (!epsilonOp)
      return reportFailure("RMS epsilon needs to be a constant");
    epsilonAttr = epsilonOp.getValueFloatAttr();
    // Check axes.
    if (!hasShapeAndRank(dd))
      return reportFailure("RMS need rank and shape for input dd");
    int64_t ddRank = mlir::cast<ShapedType>(dd.getType()).getRank();
    int64_t varAxis;
    if (!suitableAxis(vReduceOp, ddRank, varAxis))
      return reportFailure("RMS unsuitable var reduce axes");

    // 3: All the conditions are now correct for having an RMS pattern.
    // Now check if we can extend the pattern to a full LM pattern.

    bool hasFullPattern = true;
    // %d = "onnx.Sub"(%X, %mean)
    // %dd = "onnx.Mul"(%d, %d)
    if (!operandOfOpDefinedBy<ONNXSubOp>(dSubOp, ddMulOp, d1, d2, 0))
      hasFullPattern = reportFailure("LN missing D, sub op");
    if (hasFullPattern &&
        !operandOfOpDefinedBy<ONNXSubOp>(dSubOp, ddMulOp, d1, d2, 1))
      hasFullPattern = reportFailure("LN missing D, sub op");
    // %mean = "onnx.ReduceMeanV13"(%x)
    // %d = "onnx.Sub"(%X, %mean)
    if (hasFullPattern && !operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
                              mReduceOp, dSubOp, x1, mean, 1))
      hasFullPattern = reportFailure("LN missing mean, reduce mean op");

    // 4: We have the ops for a traditional LM pattern, now check a few more
    // things.

    // d = sub has 3 uses from 2 distinct ops, ignore test for now.

    if (hasFullPattern) {
      // Verify that the mReduceOp uses x as well.
      auto lnOp = cast<ONNXReduceMeanV13Op>(mReduceOp);
      Value x2 = lnOp.getData();
      if (x1 != x2)
        hasFullPattern = reportFailure(
            "LN input x to mean/ReduceMean and sub are different");
    }
    // Check the number of uses, for now only the 1 uses.
    if (hasFullPattern && !mReduceOp->hasOneUse())
      hasFullPattern = reportFailure("LN mean reduce has too many uses");
    if (hasFullPattern) {
      if (!hasShapeAndRank(x1))
        return reportFailure("LN need rank and shape for input x");
      int64_t x1Rank = mlir::cast<ShapedType>(x1.getType()).getRank();
      int64_t meanAxis;
      if (!suitableAxis(mReduceOp, x1Rank, meanAxis))
        hasFullPattern = reportFailure("LN unsuitable mean reduce axes");
      else if (meanAxis != varAxis)
        hasFullPattern = reportFailure("LN mean and var axes must be the same");
    }

    // We have now success, either with the shorter RMS LN pattern or the
    // full/traditional LN pattern. Set the last params and report success.
    axis = varAxis;
    if (hasFullPattern) {
      isRMSLayerNorm = false;
      x = x1;
      LLVM_DEBUG(llvm::dbgs() << "LayerNorm from mult, axis " << axis << "\n");
    } else {
      isRMSLayerNorm = true;
      x = d;
      LLVM_DEBUG(
          llvm::dbgs() << "RMSLayerNorm from mult, axis " << axis << "\n");
    }
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

  // Recompose LayerNorm, starting from scale/mul op
  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    Value x, scale;
    FloatAttr epsilon;
    int64_t axis;
    bool isRMSLayerNorm;
    return !RecomposeLayerNormFromMulPattern::matchLayerNormPattern(
        op, x, scale, axis, epsilon, isRMSLayerNorm);
  });

  RewritePatternSet patterns(context);
  onnx_mlir::getRecomposeONNXToONNXPatterns(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getRecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<RecomposeLayerNormFromMulPattern>(context);
}

/*!
 * Create a RecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createRecomposeONNXToONNXPass(
    const std::string &target) {
  return std::make_unique<RecomposeONNXToONNXPass>(target);
}
