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

#include <numeric>

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

namespace onnx_mlir {
// splits a tensor along a static axis into multiple outputs based on specified
// channel sizes using the ONNX Split operation
ValueRange emitSplitByChannels(PatternRewriter &rewriter, Location loc,
    Value input, ArrayRef<int64_t> splitSizes, int64_t axis) {

  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Ensure the axis is within bounds and is a static dimension
  assert(axis < static_cast<int64_t>(inputShape.size()) && axis >= 0 &&
         "Axis out of bounds for input shape.");

  assert(!inputType.isDynamicDim(axis) &&
         "Channel dimension for input tensor must be static.");
  // Validate split sizes
  int64_t totalChannels = inputShape[axis];
  int64_t sumSplitSizes =
      std::accumulate(splitSizes.begin(), splitSizes.end(), 0);

  assert(totalChannels == sumSplitSizes &&
         "Split sizes must sum up to the total number of elements along the "
         "axis.");

  // Create Split Constant
  Value splitConstant = create.onnx.constantInt64(splitSizes);

  // Create output types for each split part
  SmallVector<Type, 4> resultTypes;
  for (int64_t size : splitSizes) {
    SmallVector<int64_t> splitShape(inputShape.begin(), inputShape.end());
    splitShape[axis] = size;
    resultTypes.push_back(RankedTensorType::get(splitShape, elementType));
  }
  // Perform Split Operation
  ValueRange results =
      create.onnx.split(ArrayRef(resultTypes), input, splitConstant, axis);

  return results;
}

} // namespace onnx_mlir

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
    // Now check values epsilon.
    if (!isScalarTensor(epsilon))
      return reportFailure("RMS epsilon is expected to be scalar");
    ONNXConstantOp epsilonOp =
        mlir::dyn_cast<ONNXConstantOp>(epsilon.getDefiningOp());
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
      auto lnOp = mlir::cast<ONNXReduceMeanV13Op>(mReduceOp);
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
    ONNXReduceMeanV13Op reduceOp = mlir::cast<ONNXReduceMeanV13Op>(op);
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

struct RecomposeGeluFromMulPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Location loc = mulOp.getLoc();
    // Match:
    // - for exact gelu
    // gelu(x) = 0.5 * x * (1 + erf(x/1.41421354))
    // where 1.41421354 is sqrt(2).
    //
    // or
    //
    // - for approximate gelu
    // gelu(x) = 0.5 * x * (1 + tanh[0.797884583 * (x + 0.044715 * x^3)])
    // where 0.797884583 is sqrt(2/pi).
    Value x;
    bool isExactGelu = false;
    if (!matchGeluPattern(mulOp, x, isExactGelu))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    StringAttr approximateAttr =
        rewriter.getStringAttr(isExactGelu ? "none" : "tanh");
    Value res = create.onnx.gelu(x, approximateAttr);
    rewriter.replaceOp(mulOp, res);
    return success();
  }

  static bool matchGeluPattern(ONNXMulOp mulOp, Value &x, bool &isExactGelu) {
    using namespace onnx_mlir;
    // Subgraph to match:
    // - for exact gelu
    // gelu(x) = 0.5 * x * (1 + erf(x/1.41421354))
    // where 1.41421354 is sqrt(2).
    //
    // or
    //
    // - for approximate gelu
    // gelu(x) = 0.5 * x * (1 + tanh[0.797884583 * (x + 0.044715 * x^3)])
    // where 0.797884583 is sqrt(2/pi).
    //
    // Associcative and communitative properties are handled.

    // Helper function.
    auto constOf = [](Value v, double n) {
      return isDenseONNXConstant(v) && isConstOf(v, n);
    };

    // Match 0.5 * a * b
    // Two associative cases depending on which Mul 0.5 belongs to:
    // - 0.5 * (a * b)
    // - (0.5 * a) * b
    // For each case, we have four communitive cases: 2 for the outer Mul and 2
    // for the inner Mul. In total, we handle 8 cases.
    Value lhs = mulOp.getOperand(0);
    Value rhs = mulOp.getOperand(1);

    Value fstMulVal, sndMulVal;
    bool foundHalf = false;

    ONNXMulOp innerMulOp;
    if (matchConstAndOp<ONNXMulOp>(lhs, rhs, 0.5, innerMulOp)) {
      // - 0.5 * (a * b) or (a * b) * 0.5
      fstMulVal = innerMulOp.getOperand(0);
      sndMulVal = innerMulOp.getOperand(1);
      foundHalf = true;
    }
    if (!foundHalf && !constOf(lhs, 0.5) && !constOf(rhs, 0.5)) {
      if (auto lhsMulOp = lhs.getDefiningOp<ONNXMulOp>()) {
        // - (0.5 * a) * b
        Value l = lhsMulOp.getOperand(0);
        Value r = lhsMulOp.getOperand(1);
        if (constOf(l, 0.5)) {
          fstMulVal = r;
          sndMulVal = rhs;
          foundHalf = true;
        } else if (constOf(r, 0.5)) {
          fstMulVal = l;
          sndMulVal = rhs;
          foundHalf = true;
        }
      }
      if (!foundHalf) {
        if (auto rhsMulOp = rhs.getDefiningOp<ONNXMulOp>()) {
          // - b * (0.5 * a)
          Value l = rhsMulOp.getOperand(0);
          Value r = rhsMulOp.getOperand(1);
          if (constOf(l, 0.5)) {
            fstMulVal = lhs;
            sndMulVal = r;
            foundHalf = true;
          } else if (constOf(r, 0.5)) {
            fstMulVal = lhs;
            sndMulVal = l;
            foundHalf = true;
          }
        }
      }
    }
    if (!foundHalf)
      return reportFailure("missing 0.5 * a * b");

    // Exact gelu.
    // Match 1 + erf()
    bool foundErf = false;
    ONNXErfOp erfOp;
    // Try the first operand.
    if (auto add1Op = fstMulVal.getDefiningOp<ONNXAddOp>()) {
      foundErf = matchConstAndOp<ONNXErfOp>(
          add1Op.getOperand(0), add1Op.getOperand(1), 1.0, erfOp);
      if (foundErf)
        x = sndMulVal;
    }
    if (!foundErf) {
      // Try the second operand.
      if (auto add1Op = sndMulVal.getDefiningOp<ONNXAddOp>()) {
        foundErf = matchConstAndOp<ONNXErfOp>(
            add1Op.getOperand(0), add1Op.getOperand(1), 1.0, erfOp);
        if (foundErf)
          x = fstMulVal;
      }
    }
    if (foundErf) {
      // gelu(x) = 0.5 * x * (1 + erf(x/1.41421354))
      Value erfInput = erfOp.getOperand();
      auto divOp = erfInput.getDefiningOp<ONNXDivOp>();
      if (!divOp)
        return reportFailure("[Exact] missing div op");
      if (divOp.getOperand(0) != x)
        return reportFailure("[Exact] missing x in x/1.41421354");
      if (!constOf(divOp.getOperand(1), 1.41421354))
        return reportFailure("[Exact] missing 1.41421354");
      isExactGelu = true;
      return true;
    } else {
      // Do not return here, we still check the approximate case.
      reportFailure("[Exact] missing (1 + erf)");
    }

    // Approximate gelu.
    // gelu(x) = 0.5 * x * (1 + tanh[0.797884583 * (x + 0.044715 * x^3)])
    // Match 1 + tanh()
    bool foundTanh = false;
    ONNXTanhOp tanhOp;
    // Try the first operand.
    if (auto add1Op = fstMulVal.getDefiningOp<ONNXAddOp>()) {
      foundTanh = matchConstAndOp<ONNXTanhOp>(
          add1Op.getOperand(0), add1Op.getOperand(1), 1.0, tanhOp);
      if (foundTanh)
        x = sndMulVal;
    }
    if (!foundTanh) {
      // Try the second operand.
      if (auto add1Op = sndMulVal.getDefiningOp<ONNXAddOp>()) {
        foundTanh = matchConstAndOp<ONNXTanhOp>(
            add1Op.getOperand(0), add1Op.getOperand(1), 1.0, tanhOp);
        if (foundTanh)
          x = fstMulVal;
      }
    }
    if (!foundTanh)
      return reportFailure("[Approximate] missing (1 + tanh)");

    // Match 0.797884583 * (x + 0.044715 * x^3)
    auto mul1Op = tanhOp.getOperand().getDefiningOp<ONNXMulOp>();
    if (!mul1Op)
      return reportFailure("[Approximate] missing mul op for (0.797884583 *)");
    ONNXAddOp add2Op;
    if (!matchConstAndOp<ONNXAddOp>(
            mul1Op.getOperand(0), mul1Op.getOperand(1), 0.797884583, add2Op))
      return reportFailure(
          "[Approximate] missing add op for (x + 0.044715*x^3))");

    // Match x + 0.044715 * x^3
    ONNXMulOp mul2Op;
    if (!matchValueAndOp<ONNXMulOp>(
            add2Op.getOperand(0), add2Op.getOperand(1), x, mul2Op))
      return reportFailure("[Approximate] missing mul op for 0.044715 * x^3");

    // Match 0.044715 * x^3
    ONNXPowOp powOp;
    if (!matchConstAndOp<ONNXPowOp>(
            mul2Op.getOperand(0), mul2Op.getOperand(1), 0.044715, powOp))
      return reportFailure("[Approximate] missing 0.044715 and/or pow op");

    // Match x^3
    lhs = powOp.getOperand(0);
    rhs = powOp.getOperand(1);
    if (lhs == x && constOf(rhs, 3.0))
      return true;

    return reportFailure("subgraph not found");
  }

  static bool reportFailure(std::string msg) {
    // Can disable line below if not needed.
    LLVM_DEBUG(llvm::dbgs() << "Gelu failure: " << msg << "\n");
    return false;
  }
};

struct RecomposeQLinearMatMulFromQuantizeLinearPattern
    : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qlOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Location loc = qlOp.getLoc();
    // Match
    Value a, aScale, aZeroPoint, b, bScale, bZeroPoint, outScale, outZeroPoint;
    if (!matchQLinearMatMulPattern(qlOp, a, aScale, aZeroPoint, b, bScale,
            bZeroPoint, outScale, outZeroPoint))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value res = create.onnx.qlinearMatMul(qlOp.getY().getType(), a, aScale,
        aZeroPoint, b, bScale, bZeroPoint, outScale, outZeroPoint);

    rewriter.replaceOp(qlOp, res);
    return success();
  }

  // Recompose QLinearMatMul, starting from QuantizeLinear.
  // Pattern: DequanizeLinear + MatMul + QuantizeLinear.
  static bool matchQLinearMatMulPattern(ONNXQuantizeLinearOp op, Value &a,
      Value &aScale, Value &aZeroPoint, Value &b, Value &bScale,
      Value &bZeroPoint, Value &outScale, Value &outZeroPoint) {
    Operation *quantizeOp = op.getOperation();
    outScale = op.getYScale();
    outZeroPoint = op.getYZeroPoint();
    // Matching MatMul.
    Value qlX, matA, matB;
    Operation *matmulOp;
    bool matchMatMul = onnx_mlir::operandOfOpDefinedBy<ONNXMatMulOp>(
        matmulOp, quantizeOp, qlX, 0);
    if (!matchMatMul)
      return false;
    matA = mlir::cast<ONNXMatMulOp>(matmulOp).getA();
    matB = mlir::cast<ONNXMatMulOp>(matmulOp).getB();
    // Matching input A of MatMul.
    auto dlOpA = matA.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dlOpA)
      return false;
    a = dlOpA.getX();
    aScale = dlOpA.getXScale();
    aZeroPoint = dlOpA.getXZeroPoint();
    // Matching input B of MatMul.
    auto dlOpB = matB.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dlOpB)
      return false;
    b = dlOpB.getX();
    bScale = dlOpB.getXScale();
    bZeroPoint = dlOpB.getXZeroPoint();
    // Matched the pattern.
    return true;
  }
};

struct CombineParallelConv2DPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp1, PatternRewriter &rewriter) const final {
    Value input = convOp1.getX();
    if (!onnx_mlir::isRankedShapedType(input.getType()) ||
        !mlir::cast<ShapedType>(input.getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp1, "input must be a ranked tensor with static shape");

    if (!cast<ShapedType>(convOp1.getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(
          convOp1, "output type must be a ranked tensor with static shape");

    // Collect all ONNXConvOps using this input.
    SmallVector<ONNXConvOp> candidateConvs;
    for (auto user : input.getUsers()) {
      if (auto conv = dyn_cast<ONNXConvOp>(user))
        candidateConvs.push_back(conv);
    }

    // Must have at least two convs to combine.
    if (candidateConvs.size() < 2)
      return rewriter.notifyMatchFailure(
          convOp1, "not enough conv ops to combine");

    // Ensure all candidate convs are compatible (including bias check).
    for (size_t i = 1; i < candidateConvs.size(); ++i) {
      if (!areCompatible(candidateConvs[0], candidateConvs[i]))
        return rewriter.notifyMatchFailure(
            convOp1, "conv ops are not compatible for combining");
    }

    auto totalUses = static_cast<size_t>(
        std::distance(input.getUsers().begin(), input.getUsers().end()));
    if (candidateConvs.size() != totalUses)
      return rewriter.notifyMatchFailure(
          convOp1, "number of candidate convs does not match input uses");

    SmallVector<ONNXConvOp> parallelConvs = candidateConvs;

    SmallVector<Value> weightValues;
    int64_t totalOutputChannels = 0;
    for (auto conv : parallelConvs) {
      auto weightType = mlir::cast<ShapedType>(conv.getW().getType());
      if (!weightType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            conv, "weight must be a ranked tensor with static shape");
      if (!cast<ShapedType>(conv.getType()).hasStaticShape())
        return rewriter.notifyMatchFailure(
            conv, "output type must be a ranked tensor with static shape");
      weightValues.push_back(conv.getW());
      totalOutputChannels += weightType.getShape()[0];
    }

    auto *latestConv =
        llvm::max_element(parallelConvs, [](ONNXConvOp a, ONNXConvOp b) {
          return a->isBeforeInBlock(b.getOperation());
        });

    const auto checkIfOtherConvsReachable = [&](ONNXConvOp conv) {
      SmallVector<Operation *> worklist;
      DenseSet<Operation *> visited;
      worklist.push_back(conv.getOperation());
      while (!worklist.empty()) {
        Operation *current = worklist.back();
        worklist.pop_back();

        for (auto *user : current->getUsers()) {
          if (auto otherConv = dyn_cast<ONNXConvOp>(user)) {
            if (llvm::is_contained(parallelConvs, otherConv)) {
              // Found another conv that is part of the parallel convs.
              return true;
            }
          }
          if (visited.insert(user).second &&
              user->isBeforeInBlock(*latestConv)) {
            worklist.push_back(user);
          }
        };
      }
      return false;
    };
    // Ensure all convolutions are really parallel, none of then can be part of
    // the input of another convolution
    if (llvm::any_of(parallelConvs, checkIfOtherConvsReachable)) {
      return rewriter.notifyMatchFailure(
          convOp1, "conv ops are not parallel (reachable from each other)");
    }

    bool allHaveBias = !mlir::isa<NoneType>(parallelConvs[0].getB().getType());

    Location loc = convOp1.getLoc();
    for (auto conv : parallelConvs) {
      loc = rewriter.getFusedLoc({loc, conv.getLoc()});
    }
    auto inputType = mlir::cast<ShapedType>(input.getType());
    Type elementType = inputType.getElementType();
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(*latestConv);

    int64_t concatAxis = 1;

    auto firstWeightType =
        mlir::cast<ShapedType>(parallelConvs[0].getW().getType());
    SmallVector<int64_t> newWeightShape(
        firstWeightType.getShape().begin(), firstWeightType.getShape().end());
    newWeightShape[0] = totalOutputChannels;
    Type newWeightType =
        RankedTensorType::get(newWeightShape, firstWeightType.getElementType());
    Value newWeight = create.onnx.concat(newWeightType, weightValues, 0);

    Value newBias;
    if (allHaveBias) {
      SmallVector<Value> biasValues;
      for (auto conv : parallelConvs) {
        biasValues.push_back(conv.getB());
      }
      SmallVector<int64_t> newBiasShape = {totalOutputChannels};
      Type newBiasType = RankedTensorType::get(newBiasShape, elementType);
      newBias = create.onnx.concat(newBiasType, biasValues, 0);
    } else {
      newBias = parallelConvs[0].getB();
    }

    SmallVector<int64_t> newOutputShape(
        mlir::cast<ShapedType>(convOp1.getResult().getType())
            .getShape()
            .begin(),
        mlir::cast<ShapedType>(convOp1.getResult().getType()).getShape().end());
    newOutputShape[concatAxis] = totalOutputChannels;
    auto newOutputType = RankedTensorType::get(newOutputShape, elementType);

    auto newConv =
        ONNXConvOp::create(rewriter, loc, newOutputType, input, newWeight,
            newBias, convOp1.getAutoPadAttr(), convOp1.getDilationsAttr(),
            convOp1.getGroupAttr(), convOp1.getKernelShapeAttr(),
            convOp1.getPadsAttr(), convOp1.getStridesAttr());

    ONNXConcatOp commonConcatOp = nullptr;
    bool allOutputsUsedInCommonConcat = true;

    for (auto conv : parallelConvs) {
      bool usedInCommonConcat = false;
      for (auto user : conv.getResult().getUsers()) {
        if (auto concatOp = dyn_cast<ONNXConcatOp>(user)) {
          if (!commonConcatOp) {
            commonConcatOp = concatOp;
          }
          if (concatOp != commonConcatOp) {
            allOutputsUsedInCommonConcat = false;
            break;
          }
          usedInCommonConcat = true;
        } else {
          allOutputsUsedInCommonConcat = false;
          break;
        }
      }
      if (!usedInCommonConcat || !allOutputsUsedInCommonConcat) {
        allOutputsUsedInCommonConcat = false;
        break;
      }
    }

    if (allOutputsUsedInCommonConcat && commonConcatOp &&
        commonConcatOp.getAxis() == 1) {
      rewriter.replaceOp(commonConcatOp, newConv);
    } else {
      SmallVector<int64_t> splitSizesVec;
      for (auto conv : parallelConvs) {
        int64_t channels = mlir::cast<ShapedType>(conv.getResult().getType())
                               .getShape()[concatAxis];
        splitSizesVec.push_back(channels);
      }

      ValueRange splitResults = onnx_mlir::emitSplitByChannels(
          rewriter, loc, newConv.getResult(), splitSizesVec, concatAxis);
      for (size_t i = 0; i < parallelConvs.size(); ++i) {
        rewriter.replaceAllOpUsesWith(parallelConvs[i], splitResults[i]);
      }
      // Sort the block topological, as the operations after the split may be in
      // the wrong place otherwise
      mlir::sortTopologically(newConv->getBlock());
    }
    for (auto conv : parallelConvs) {
      rewriter.eraseOp(conv);
    }

    return success();
  }

  static bool areCompatible(ONNXConvOp a, ONNXConvOp b) {
    if (a.getAutoPad() != b.getAutoPad() ||
        a.getDilations() != b.getDilations() || a.getGroup() != b.getGroup() ||
        a.getKernelShape() != b.getKernelShape() ||
        a.getPads() != b.getPads() || a.getStrides() != b.getStrides())
      return false;

    auto shapeA = mlir::cast<ShapedType>(a.getW().getType()).getShape();
    auto shapeB = mlir::cast<ShapedType>(b.getW().getType()).getShape();
    if (shapeA != shapeB)
      return false;

    bool hasBiasA = !mlir::isa<NoneType>(a.getB().getType());
    bool hasBiasB = !mlir::isa<NoneType>(b.getB().getType());
    if (hasBiasA != hasBiasB)
      return false;

    if (hasBiasA) {
      auto biasShapeA = mlir::cast<ShapedType>(a.getB().getType()).getShape();
      auto biasShapeB = mlir::cast<ShapedType>(b.getB().getType()).getShape();
      if (biasShapeA != biasShapeB)
        return false;
    }
    return true;
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

  RewritePatternSet patterns(context);
  onnx_mlir::getRecomposeONNXToONNXPatterns(patterns);

  if (failed(applyPatternsGreedily(function, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getRecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<RecomposeGeluFromMulPattern>(context);
  patterns.insert<RecomposeLayerNormFromMulPattern>(context);
  patterns.insert<RecomposeQLinearMatMulFromQuantizeLinearPattern>(context);
  patterns.insert<CombineParallelConv2DPattern>(context);
}

/*!
 * Create a RecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createRecomposeONNXToONNXPass(
    const std::string &target) {
  return std::make_unique<RecomposeONNXToONNXPass>(target);
}
