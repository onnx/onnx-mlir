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
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

// There could be scenarios that the pattern can be matched as layernorm but the
// axis is not suitable. However, there is a possibility to make the axis
// suitable by transposing the original input tensor and axis. In the following
// example, since the second dimension is being reduced, it's not a layernorm
// but if that dimension is transposed to the last dimension and and change the
// axis of layernorm to 3 and transpose the result back, the transform IR is
// semantically and mathematically correct and legal.
//
// Before:
// %9 = "onnx.ReduceMeanV13"(%8) {axes = [1]} : (<1x4x128xf32>) -> <1x1x128xf32>
// %10 = "onnx.Sub"(%8, %9) (<1x4x128xf32>, <1x1x128xf32>) -> <1x4x128xf32>
// %11 = "onnx.Mul"(%10, %10) (<1x4x128xf32>, <1x4x128xf32>) -> <1x4x128xf32>
// %12 = "onnx.ReduceMeanV13"(%11) {axes = [1]} : (<1x4x128xf32>) ->
// <1x1x128xf32> %13 = "onnx.Add"(%12, %4) : (<1x1x128xf32>, <f32>) ->
// <1x1x128xf32> %14 = "onnx.Sqrt"(%13): (<1x1x128xf32>) -> <1x1x128xf32> %15 =
// "onnx.Div"(%10, %14) : (<1x4x128xf32>, <1x1x128xf32>) -> <1x4x128xf32> %16 =
// "onnx.Mul"(%15, %5) : (<1x4x128xf32>, <4x1x1xf32>) -> <1x4x128xf32>
//
// After:
// %9 = "onnx.Transpose"(%6)
//      {perm = [0, 2, 3, 1]} : (<1x4x128x128xf32>) -> <1x128x128x4xf32>
// %Y = "onnx.LayerNormalization"(%9, %8, %1)
//      {axis = 3 : si64} : (<1x128x128x4xf32>, <1x1x1x4xf32>, none) ->
//      (<1x128x128x4xf32>)
// %10 = "onnx.Transpose"(%Y)
//      {perm = [0, 3, 1, 2]} : (<1x128x128x4xf32>) -> <1x4x128x128xf32>

template <bool RecomposeLayernormByTranspose = false>
struct RecomposeLayerNormFromMulPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  FailureOr<mlir::Value> createScaleConstOp(const Type &xType,
      const onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> &builder)
      const {
    mlir::Value scale;
    if (!isa<ShapedType>(xType))
      return failure();

    auto xShape = mlir::cast<ShapedType>(xType).getShape();
    auto xInnerMostDim = xShape[xShape.size() - 1];

    // Inner most dim is required
    if (xInnerMostDim <= 0)
      return failure();

    auto scaleType = mlir::RankedTensorType::get(
        {xInnerMostDim}, getElementTypeOrSelf(xType));

    // For now only float32 type supported
    mlir::DenseElementsAttr attr;
    if (onnx_mlir::getEltSizeInBytes(scaleType) == sizeof(float)) {
      auto scaleVal = SmallVector<float>(xInnerMostDim, 1.0f);
      attr = mlir::DenseElementsAttr::get(scaleType, ArrayRef(scaleVal));
    } else {
      return failure();
    }

    scale = builder.onnx.constant(attr);
    return success(scale);
  }

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    // Match
    Value x, scale;
    FloatAttr epsilon;
    int64_t axis;
    bool isRMSLayerNorm;
    SmallVector<Location> layerNormLocations;

    // This will be filled up with the permutation that helps us by transposing
    // the original tensor and adapting the axis, we make unsuitable axis for
    // layer suitable.
    SmallVector<int64_t> permutation;
    if (!matchLayerNormPattern(mulOp, x, scale, axis, epsilon,
            layerNormLocations, isRMSLayerNorm, permutation))
      return failure();

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(
        rewriter, rewriter.getFusedLoc(layerNormLocations));
    Value noneVal = create.onnx.none();
    Value res;

    if constexpr (RecomposeLayernormByTranspose) {
      if (!hasShapeAndRank(scale))
        return rewriter.notifyMatchFailure(
            mulOp, "the scale doesn't have shape or rank");
      // if the permutation is empty, nothing is needed to be permuted.
      // Otherwise, both input and scale must be transposed.
      if (!permutation.empty()) {
        if (scale) {
          // Layernorm supports broadcasting and in order to transpose, the rank
          // should be equalized.
          scale = create.onnx.upRank(scale, getRank(x.getType()));
          scale = create.onnx.transposeInt64(scale, permutation);
        }
        x = create.onnx.transposeInt64(x, permutation);
      }
    }

    if (isRMSLayerNorm) {
      if (!scale) {
        // set scale to unity if there is no scale value present in the pattern
        // This is required because RMSLayerNorm mandates passing a scale value

        // NOTE: The scale is being passed as a tensor instead of a scalar
        // because the pass that generates RMSNormAdf templated graph requires
        // the scale to be a tensor of dimension same as the inner most
        // dimension of input tensor
        auto result = this->createScaleConstOp(x.getType(), create);
        if (failed(result))
          return failure();
        scale = result.value();
      }
      res = create.onnx.RMSLayerNorm(
          x.getType(), x, scale, noneVal, axis, epsilon);
    } else {
      res =
          create.onnx.layerNorm(x.getType(), x, scale, noneVal, axis, epsilon);
    }

    if constexpr (RecomposeLayernormByTranspose) {
      // Transpose back the result if the input and scale got transposed.
      if (!permutation.empty()) {
        res = create.onnx.transposeInt64(
            res, invertPermutationVector(permutation));
      }
    }

    copySingleResultType(mulOp, res);
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


  * Second pattern associated with RMSLayerNormalization:

    var = reduceMean(X * X)
    stdDev = sqrt(var + eps)
    Y = mul(scale, X / stdDev)

  * Third pattern associated with RMSLayerNormalization

    var = reduceMean(X * X)
    invStdDev = pow(var + eps, -0.5)
    Y = mul(X, invStdDev)

  As it can be seen here, the second RMS LN pattern matches the traditional
  LN for the bottom 3 statements. In RMS LN, X is the raw input, whereas in the
  traditional LN, the input to the lower 3 statements are D = X - mean(X).

  * Variations around the div (for both patterns):

     D / stdDev
     D * (1 / stdDev)
     D * recip(stdDev)

  * About third pattern: In the third pattern, sqrt and div/recip ops are
  replaced by a single Pow op with exponent as -0.5. Everything other than the
  last two lines follows a logic common to primary and secondary pattern.
  */
  static bool matchLayerNormPattern(ONNXMulOp LayerNormOp, Value &x,
      Value &scale, int64_t &axis, FloatAttr &epsilonAttr,
      SmallVectorImpl<Location> &layerNormLocations, bool &isRMSLayerNorm,
      SmallVectorImpl<int64_t> &perm) {
    using namespace onnx_mlir;
    Location loc = LayerNormOp.getLoc();
    isRMSLayerNorm = false;

    // 1: Start first to detect if we have the common layer norm pattern.
    // Values that will be gathered and only kept locally.
    Value norm, invStdDev, stdDev, varEps, var, epsilon, dd, d, mean, x1;
    // Replicate of values, check that they are identical to originals.
    Value d1, d2;
    // Operations that will be gathered and kept locally.
    Operation *nsMulOp = LayerNormOp.getOperation();
    Operation *ddMulOp = nullptr;
    Operation *nDivOp = nullptr;
    Operation *nMulOp = nullptr;
    Operation *isdRecipOp = nullptr;
    Operation *sdSqrtOp = nullptr;
    Operation *veAddOp = nullptr;
    Operation *vReduceOp = nullptr;
    Operation *mReduceOp = nullptr;
    Operation *dSubOp = nullptr;
    Operation *powOp = nullptr;
    // after this group, we have defined norm, scale, d, and sdSqrtOp.
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
    } else if (operandOfOpDefinedBy<ONNXPowOp>(
                   powOp, nsMulOp, invStdDev, d, 0) ||
               operandOfOpDefinedBy<ONNXPowOp>(
                   powOp, nsMulOp, d, invStdDev, 1)) {
      // The following pattern is now matched
      // %invStdDev = "onnx.Pow"(%varEps, %negHalf)
      // %norm = "onnx.Mul"(%d, %invStdDev)

      // Now verify the value of exponent for pow op
      mlir::Value negHalf;
      if (operandOfOpDefinedBy<ONNXAddOp>(veAddOp, powOp, varEps, negHalf, 0)) {
        // Verify if the exponent is floating point scalar with value -0.5
        IndexExprScope scope(nullptr, loc);
        IndexExprBuilderForAnalysis createIE(loc);
        if (createIE.hasShapeAndRank(negHalf) &&
            createIE.getArraySize(negHalf, /*static only*/ true) == 1) {
          IndexExpr floatVal = createIE.getFloatAsNonAffine(negHalf);
          if (!floatVal.isLiteral() || (floatVal.getFloatLiteral() != -0.5)) {
            return reportFailure("RMS missing std dev (via pow(var + eps, "
                                 "-0.5)), exp is not -0.5");
          }
        } else {
          return reportFailure(
              "missing std dev (via pow(var + eps, -0.5)), not exp of scalar");
        }
      } else {
        return reportFailure("RMS missing std dev (via pow(var + eps, -0.5)), "
                             "input to pow is not an Add");
      }
    } else {
      return reportFailure("RMS missing norm, div, reciprocal or pow op");
    }

    // extract veAddOp and varEps if sqrt pattern is present instead of pow.
    // In case of pow these values have already been extracted above
    if (!powOp) {
      // %varEps = "onnx.Add"(%var, %eps)
      // %stdDev = "onnx.Sqrt"(%varEps)
      if (!operandOfOpDefinedBy<ONNXAddOp>(veAddOp, sdSqrtOp, varEps))
        return reportFailure("RMS missing var + eps, add op");
    }
    // %var = "onnx.ReduceMean(V13)"(%dd)
    // %varEps = "onnx.Add"(%var, %eps)
    if ((!operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
             vReduceOp, veAddOp, var, epsilon, 0) &&
            !operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
                vReduceOp, veAddOp, epsilon, var, 1)) &&
        (!operandOfOpDefinedBy<ONNXReduceMeanOp>(
             vReduceOp, veAddOp, var, epsilon, 0) &&
            !operandOfOpDefinedBy<ONNXReduceMeanOp>(
                vReduceOp, veAddOp, epsilon, var, 1)))
      return reportFailure("RMS missing var, reduce mean op");
    // %dd = "onnx.Mul"(%d, %d)
    // %var = "onnx.ReduceMean(V13)"(%dd)
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
    if (sdSqrtOp && !sdSqrtOp->hasOneUse())
      return reportFailure("RMS std dev sqrt has too many uses");
    if (powOp && !powOp->hasOneUse())
      return reportFailure("RMS std dev pow has too many uses");
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
    const auto epsilonValue = getScalarValue<double>(epsilonOp);
    epsilonAttr =
        FloatAttr::get(Float32Type::get(epsilonOp->getContext()), epsilonValue);
    // Check axes.
    if (!hasShapeAndRank(dd))
      return reportFailure("RMS need rank and shape for input dd");
    int64_t ddRank = mlir::cast<ShapedType>(dd.getType()).getRank();
    int64_t varAxis;
    if constexpr (!RecomposeLayernormByTranspose) {
      if (!suitableAxis(vReduceOp, ddRank, varAxis))
        return reportFailure("RMS unsuitable var reduce axes");
    } else {
      if (failed(isAxisSuitableWithTranspose(vReduceOp, ddRank, varAxis)))
        return reportFailure("RMS unsuitable var reduce axes that cannot be "
                             "handled even with transposition");
    }

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
    // %mean = "onnx.ReduceMean(V13)"(%x)
    // %d = "onnx.Sub"(%X, %mean)
    if (hasFullPattern && (!operandOfOpDefinedBy<ONNXReduceMeanV13Op>(
                               mReduceOp, dSubOp, x1, mean, 1) &&
                              !operandOfOpDefinedBy<ONNXReduceMeanOp>(
                                  mReduceOp, dSubOp, x1, mean, 1)))
      hasFullPattern = reportFailure("LN missing mean, reduce mean op");

    // 4: We have the ops for a traditional LM pattern, now check a few more
    // things.

    // d = sub has 3 uses from 2 distinct ops, ignore test for now.

    if (hasFullPattern) {
      // Verify that the mReduceOp uses x as well.
      Value x2 = [](Operation *op) {
        if (auto rmOp = mlir::dyn_cast<ONNXReduceMeanOp>(op)) {
          return rmOp.getData();
        }
        if (auto rmV13Op = mlir::dyn_cast<ONNXReduceMeanV13Op>(op)) {
          return rmV13Op.getData();
        }
        llvm_unreachable("Expected ONNXReduceMeanOp or ONNXReduceMeanV13Op");
      }(mReduceOp);
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
      if constexpr (!RecomposeLayernormByTranspose) {
        if (!suitableAxis(mReduceOp, x1Rank, meanAxis))
          hasFullPattern = reportFailure("LN unsuitable mean reduce axes");
        else if (meanAxis != varAxis)
          hasFullPattern =
              reportFailure("LN mean and var axes must be the same");
      } else {
        auto permutation =
            isAxisSuitableWithTranspose(mReduceOp, x1Rank, meanAxis);
        if (failed(permutation))
          hasFullPattern =
              reportFailure("LN unsuitable mean reduce axes that cannot be "
                            "handled even with transposition");
        else if (meanAxis != varAxis)
          hasFullPattern =
              reportFailure("LN mean and var axes must be the same");
        perm = permutation.value();
      }
    }

    // We have now success, either with the shorter RMS LN pattern or the
    // full/traditional LN pattern. Set the last params and report success.
    axis = varAxis;
    if (hasFullPattern) {
      isRMSLayerNorm = false;
      x = x1;

    } else {
      isRMSLayerNorm = true;
      x = d;
    }

    static const std::string layerNormKind =
        isRMSLayerNorm ? "RMSLayerNorm" : "LayerNorm";
    if (perm.empty()) {
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "{0} from mult, axis {1}", layerNormKind, axis));
    } else {
      std::string msg;
      llvm::raw_string_ostream rso(msg);
      llvm::interleaveComma(perm, rso);
      LLVM_DEBUG(llvm::dbgs()
                 << llvm::formatv("{0} from mult with transpose {1} of the "
                                  "original input and new axis {2} \n",
                        layerNormKind, msg, axis));
    }

    // Collect the locations of the recomposed ops
    if (mReduceOp)
      layerNormLocations.push_back(mReduceOp->getLoc());
    if (dSubOp)
      layerNormLocations.push_back(dSubOp->getLoc());
    layerNormLocations.push_back(ddMulOp->getLoc());
    layerNormLocations.push_back(vReduceOp->getLoc());
    layerNormLocations.push_back(veAddOp->getLoc());
    if (sdSqrtOp)
      layerNormLocations.push_back(sdSqrtOp->getLoc());
    else
      layerNormLocations.push_back(powOp->getLoc());
    if (isdRecipOp)
      layerNormLocations.push_back(isdRecipOp->getLoc());
    if (nMulOp)
      layerNormLocations.push_back(nMulOp->getLoc());
    if (nDivOp)
      layerNormLocations.push_back(nDivOp->getLoc());
    layerNormLocations.push_back(loc);

    return true;
  }

private:
  // Return the reduced dimensions as bit vector.
  static FailureOr<llvm::SmallBitVector> getReducedAxis(
      Operation *op, int64_t xRank, int64_t &axis) {
    SmallVector<int64_t> axes; // The axes attribute/operand of the ReduceMeanOp
    if (auto reduceOpV13 = mlir::dyn_cast<ONNXReduceMeanV13Op>(op)) {
      if (reduceOpV13.getKeepdims() != 1)
        return success(reportFailure("need keepdims = 1"));
      ArrayAttr axesAttr = reduceOpV13.getAxesAttr();
      for (size_t i = 0; i < axesAttr.size(); ++i) {
        axes.emplace_back(onnx_mlir::ArrayAttrIntVal(axesAttr, i));
      }
    } else if (auto reduceOp = mlir::dyn_cast<ONNXReduceMeanOp>(op)) {
      if (reduceOp.getKeepdims() != 1)
        return success(reportFailure("need keepdims = 1"));
      Value axesValue = reduceOp.getAxes();
      if (isa<NoneType>(axesValue.getType())) {
        if (reduceOp.getNoopWithEmptyAxes()) {
          // No reduction
          return success(
              reportFailure("needs a reduction on at least one dimension"));
        } else {
          // Reduction on all dimensions
          axis = 0;
          return llvm::SmallBitVector(xRank, true);
        }
      }
      if (!onnx_mlir::getI64ValuesFromONNXConstantOp(axesValue, axes)) {
        return success(reportFailure("only static axes are supported"));
      }
    } else {
      llvm_unreachable("ReduceMean is the only supported op");
    }

    // Record axes value in bit vector.
    llvm::SmallBitVector reduceAxes(xRank, false);
    for (int64_t axe : axes) {
      int64_t a = onnx_mlir::getAxisInRange(axe, xRank);
      reduceAxes[a] = true;
    }
    return reduceAxes;
  }

  // Check if the axis is suitable for Layernorm.
  static bool suitableAxis(Operation *op, int64_t xRank, int64_t &axis) {
    auto reduceAxes = getReducedAxis(op, xRank, axis);
    if (failed(reduceAxes))
      return false;

    // Check that we have a "false"* "true"+ pattern.
    bool foundFirstAxis = false;
    for (int64_t i = 0; i < xRank; ++i) {
      if (!foundFirstAxis) {
        if (reduceAxes.value()[i]) {
          foundFirstAxis = true;
          axis = i;
        }
      } else if (!reduceAxes.value()[i]) {
        // Once we found an axis, we must reduce all subsequent dimensions.
        return false;
      }
    }
    // Ensure we had at least one reduction.
    return foundFirstAxis;
  }

  // Checks if the axes with transpose op could be suitable for Layernorm.
  // normalized_axes is [axis, ..., rank of X - 1] in layernorm but the
  // reduce_mean ops in the decomposition have axes = [1,3] in the 4d tensor,
  // that's not suitable for layernorm. However, we transpose the dimensions 1
  // and 3 to the innermost dimension with perm = [0,2,1,3], this can be
  // represented as Layernorm with axis = 2.
  static FailureOr<SmallVector<int64_t>> isAxisSuitableWithTranspose(
      Operation *op, int64_t xRank, int64_t &axis) {
    auto reduceAxes = getReducedAxis(op, xRank, axis);
    if (failed(reduceAxes))
      return failure();

    SmallVector<int64_t> reducedIdx;
    SmallVector<int64_t> nonReducedIdx;
    for (int64_t i = 0; i < xRank; ++i) {
      auto &array = reduceAxes.value()[i] ? reducedIdx : nonReducedIdx;
      array.push_back(i);
    }

    SmallVector<int64_t> perm;
    perm.append(nonReducedIdx);
    perm.append(reducedIdx);
    axis = nonReducedIdx.size();

    return perm;
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
    copySingleResultType(mulOp, res);
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

// Result of attempting recomposing DepthToSpace. Contains useful information
// for the matching
struct DepthToSpaceRecompositionResult {
  Value input;
  int64_t blockSize;
  std::string mode;
  Location fusedLocation;
};

template <typename Derived>
struct RecomposeDepthToSpace : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    std::optional<DepthToSpaceRecompositionResult> result =
        Derived::matchDepthToSpacePattern(reshapeOp);
    if (!result) {
      return failure();
    }

    MultiDialectBuilder<OnnxBuilder> create(rewriter, result->fusedLocation);
    rewriter.replaceOp(
        reshapeOp, create.onnx.createOpAndInferShapes<ONNXDepthToSpaceOp>(
                       reshapeOp.getType(), result->input, result->blockSize,
                       result->mode));
    return success();
  }

  static std::optional<DepthToSpaceRecompositionResult>
  matchDepthToSpacePattern(ONNXReshapeOp reshapeOp,
      function_ref<bool(ArrayRef<int64_t>, ArrayRef<int64_t>)> fstReshapePred,
      function_ref<bool(ArrayRef<int64_t>)> transposePred) {
    using namespace onnx_mlir;
    ONNXReshapeOp r0;
    ONNXTransposeOp t;
    ONNXReshapeOp r1 = reshapeOp;

    t = r1->getOperand(0).getDefiningOp<ONNXTransposeOp>();
    if (!t) {
      return reportFailure("missing transpose");
    }
    r0 = t->getOperand(0).getDefiningOp<ONNXReshapeOp>();
    if (!r0) {
      return reportFailure("missing first reshape");
    }

    auto hasShapedStaticType = [](Type ty) {
      auto shapedType = dyn_cast<ShapedType>(ty);
      return shapedType && shapedType.hasStaticShape();
    };

    const bool haveOperationsValidTy =
        llvm::all_of(TypeRange{r0.getOperand(0).getType(), r0.getType(),
                         t.getType(), r1.getType()},
            hasShapedStaticType);
    if (!haveOperationsValidTy) {
      return reportFailure(
          "pattern operations have no shaped static tensor types");
    }

    auto fstReshapeInTy = cast<ShapedType>(r0->getOperand(0).getType());
    ArrayRef<int64_t> fstReshapeInShape = fstReshapeInTy.getShape();
    const size_t fstReshapeInRank = fstReshapeInTy.getRank();
    if (fstReshapeInRank != 4) {
      return reportFailure("input rank is not 4D ");
    }

    auto fstReshapeOutTy = cast<ShapedType>(r0.getType());
    ArrayRef<int64_t> fstReshapeOutShape = fstReshapeOutTy.getShape();
    const size_t fstReshapeOutRank = fstReshapeOutTy.getRank();
    if (fstReshapeOutRank != 6) {
      return reportFailure("output rank of first reshape is not 6D");
    }

    // Blocksize can be found in both working modes in dimension 2
    // CRD:
    //   reshape %x NxCxHxW -> NxC//(B*B)xBxBxHxW
    //                         0 1        2 3 4 5
    // DCR:
    //   reshape %x NxCxHxW -> NxBxBxC//(B*B)xHxW
    //                         0 1 2 3        4 5
    const int64_t blocksize = fstReshapeOutShape[2];

    if (!fstReshapePred(fstReshapeInShape, fstReshapeOutShape))
      return std::nullopt;

    // Check for concrete permutation pattern:
    //   transpose %r0 perm=[0, 1, 4, 2, 5, 3]
    std::optional<ArrayAttr> permOpt = t.getPerm();
    if (!permOpt) {
      return reportFailure("missing permutation on transpose");
    }

    // Get transpose permutation
    SmallVector<int64_t, 6> perms;
    ArrayAttrIntVals(*permOpt, perms);

    // Check for transpose permutation
    if (!transposePred(perms))
      return std::nullopt;

    // Check for concrete reshape pattern. This pattern applies to both DCR and
    // CRD modes:
    //   reshape NxC//(B*B)xHxBxWxB -> NxC//(B*B)x(HxB)x(WxB)
    auto sndReshapeInTy = cast<ShapedType>(t.getType());
    ArrayRef<int64_t> sndReshapeInShape = sndReshapeInTy.getShape();

    auto sndReshapeOutTy = cast<ShapedType>(r1.getType());
    ArrayRef<int64_t> sndReshapeOutShape = sndReshapeOutTy.getShape();
    const size_t sndReshapeOutRank = sndReshapeOutTy.getRank();
    if (sndReshapeOutRank != 4) {
      return reportFailure("out rank of second reshape is not 4D");
    }

    if (sndReshapeInShape[0] != sndReshapeOutShape[0] ||
        sndReshapeInShape[1] != sndReshapeOutShape[1] ||
        sndReshapeInShape[2] * sndReshapeInShape[3] != sndReshapeOutShape[2] ||
        sndReshapeInShape[4] * sndReshapeInShape[5] != sndReshapeOutShape[3]) {
      return reportFailure("unexpected second reshape result shape");
    }

    Location fusedLocation = FusedLoc::get(
        reshapeOp->getContext(), {r0->getLoc(), t->getLoc(), r1->getLoc()});

    return DepthToSpaceRecompositionResult{/*input=*/r0.getOperand(0),
        blocksize, /*mode=*/Derived::mode.str(), fusedLocation};
  }

  static std::nullopt_t reportFailure(
      StringRef msg, StringRef depthToSpaceMode = "") {
    // Can disable line below if not needed.
    LLVM_DEBUG(llvm::dbgs()
               << "DepthToSpace "
               << (depthToSpaceMode.empty()
                          ? std::string("")
                          : llvm::formatv("[{0}]", depthToSpaceMode))
               << " failure: " << msg << "\n");
    return std::nullopt;
  }
};

struct RecomposeDepthToSpaceDCR
    : public RecomposeDepthToSpace<RecomposeDepthToSpaceDCR> {
  using RecomposeDepthToSpace::RecomposeDepthToSpace;

  static std::optional<DepthToSpaceRecompositionResult>
  matchDepthToSpacePattern(ONNXReshapeOp reshapeOp) {
    auto fstReshapePredForDCR = [](ArrayRef<int64_t> fstReshapeInShape,
                                    ArrayRef<int64_t> fstReshapeOutShape) {
      // Check for concrete reshape pattern:
      //   reshape %x NxCxHxW -> NxBxBxC//(B*B)xHxW
      const int64_t blocksize = fstReshapeOutShape[2];
      if (blocksize != fstReshapeOutShape[1]) {
        reportFailureForDCRMode("blocksize do not match in dim 1 and 2");
        return false;
      }

      if (fstReshapeInShape[0] != fstReshapeOutShape[0] ||
          fstReshapeInShape[1] !=
              fstReshapeOutShape[3] * blocksize * blocksize ||
          fstReshapeInShape[2] != fstReshapeOutShape[4] ||
          fstReshapeInShape[3] != fstReshapeOutShape[5]) {
        reportFailureForDCRMode("unexpected first reshape result shape");
        return false;
      }

      return true;
    };

    auto transposePredForDCR = [&](ArrayRef<int64_t> transposePerms) {
      constexpr std::array<int64_t, 6> expectedPerms = {0, 3, 4, 1, 5, 2};
      if (transposePerms != ArrayRef(expectedPerms)) {
        reportFailureForDCRMode("unexpected permutations");
        return false;
      }
      return true;
    };

    return RecomposeDepthToSpace::matchDepthToSpacePattern(
        reshapeOp, fstReshapePredForDCR, transposePredForDCR);
  }

  static constexpr StringLiteral mode = "DCR";

  static std::nullopt_t reportFailureForDCRMode(StringRef msg) {
    return reportFailure(msg, mode);
  }
};

struct RecomposeDepthToSpaceCRD
    : public RecomposeDepthToSpace<RecomposeDepthToSpaceCRD> {
  using RecomposeDepthToSpace::RecomposeDepthToSpace;

  static std::optional<DepthToSpaceRecompositionResult>
  matchDepthToSpacePattern(ONNXReshapeOp reshapeOp) {
    auto fstReshapePredForCRD = [](ArrayRef<int64_t> fstReshapeInShape,
                                    ArrayRef<int64_t> fstReshapeOutShape) {
      // Check for concrete reshape pattern:
      //   reshape %x NxCxHxW -> NxC//(B*B)xBxBxHxW
      const int64_t blocksize = fstReshapeOutShape[2];
      if (blocksize != fstReshapeOutShape[3]) {
        reportFailureForCRDMode("blocksize do not match in dim 2 and 3");
        return false;
      }

      if (fstReshapeInShape[0] != fstReshapeOutShape[0] ||
          fstReshapeInShape[1] !=
              fstReshapeOutShape[1] * blocksize * blocksize ||
          fstReshapeInShape[2] != fstReshapeOutShape[4] ||
          fstReshapeInShape[3] != fstReshapeOutShape[5]) {
        reportFailureForCRDMode("unexpected first reshape result shape");
        return false;
      }

      return true;
    };

    auto transposePredForCRD = [&](ArrayRef<int64_t> transposePerms) {
      constexpr std::array<int64_t, 6> expectedPerms = {0, 1, 4, 2, 5, 3};
      if (transposePerms != ArrayRef(expectedPerms)) {
        reportFailureForCRDMode("unexpected permutations");
        return false;
      }
      return true;
    };

    return RecomposeDepthToSpace::matchDepthToSpacePattern(
        reshapeOp, fstReshapePredForCRD, transposePredForCRD);
  }

  static constexpr StringLiteral mode = "CRD";

  static std::nullopt_t reportFailureForCRDMode(StringRef msg) {
    return reportFailure(msg, mode);
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
    copySingleResultType(qlOp, res);
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
        rewriter.create<ONNXConvOp>(loc, newOutputType, input, newWeight,
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

  RecomposeONNXToONNXPass(
      const std::string &target, const bool &recomposeLayernormByTranspose) {
    this->target = target;
    this->recomposeLayernormByTranspose = recomposeLayernormByTranspose;
  }
  RecomposeONNXToONNXPass(const RecomposeONNXToONNXPass &pass)
      : mlir::PassWrapper<RecomposeONNXToONNXPass,
            OperationPass<func::FuncOp>>() {
    this->target = pass.target.getValue();
    this->recomposeLayernormByTranspose =
        pass.recomposeLayernormByTranspose.getValue();
  }

  StringRef getArgument() const override { return "recompose-onnx"; }

  StringRef getDescription() const override {
    return "Recompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  Option<std::string> target{*this, "target",
      llvm::cl::desc("Target Dialect to Recompose into"), ::llvm::cl::init("")};

  Option<bool> recomposeLayernormByTranspose{*this,
      "recompose-layernorm-by-transpose",
      llvm::cl::desc("Use transpose operator to make unsuitable axes suitable "
                     "for matching layernorm"),
      ::llvm::cl::init(false)};

  void runOnOperation() final;

  typedef PassWrapper<RecomposeONNXToONNXPass, OperationPass<func::FuncOp>>
      BaseType;
};

void RecomposeONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  onnx_mlir::getRecomposeONNXToONNXPatterns(
      patterns, recomposeLayernormByTranspose);

  if (failed(applyPatternsGreedily(function, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getRecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns, bool recomposeLayernormByTranspose) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<RecomposeGeluFromMulPattern>(context);
  patterns.insert<RecomposeLayerNormFromMulPattern<false>>(context);
  if (recomposeLayernormByTranspose)
    patterns.insert<RecomposeLayerNormFromMulPattern<true>>(context);
  patterns.insert<RecomposeDepthToSpaceCRD>(context);
  patterns.insert<RecomposeDepthToSpaceDCR>(context);
  // AMD Disabled as downstream has no special support for it
  // patterns.insert<RecomposeQLinearMatMulFromQuantizeLinearPattern>(context);
  // patterns.insert<CombineParallelConv2DPattern>(context);
}

/*!
 * Create a RecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createRecomposeONNXToONNXPass(
    const std::string &target, const bool &recomposeLayernormByTranspose) {
  return std::make_unique<RecomposeONNXToONNXPass>(
      target, recomposeLayernormByTranspose);
}
