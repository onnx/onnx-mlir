/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForZHigh.cpp - Rewrite ONNX ops for ZHigh lowering ----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements pass for rewriting of ONNX operations to generate
// combination of ONNX and ZHigh operations.
//
// - `ONNXBatchNormalizationInferenceModeOp`
// In this pass, `ONNXBatchNormalizationInferenceModeOp` is converted into
// `ZHigh.BatchNorm`, generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
// and `ONNX.Sqrt` to calculate inputs(`a` and `b`) for `ZHigh.BatchNorm`.
// `ONNXToZHighLoweringPass`(`--convert-onnx-to-zhigh`) is also able to generate
// the ONNX ops, but,they are lowered to ZHigh ops. So, constant
// propagation(`--constprop-onnx`) doesn't work. To enable to work it, this
// separate pass is needed. By using this pass, constant propagation works by
// running it just after this pass.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Calculate sqrt(var + epsilon) for batchnorm op A.
/// A = scale / sqrt(var + epsilon)
Value getSqrtResultBatchNormA(
    Location loc, PatternRewriter &rewriter, Value var, FloatAttr epsilon) {
  Type elementType = var.getType().cast<ShapedType>().getElementType();

  // epsilon
  RankedTensorType epsilonType = RankedTensorType::get({1}, elementType);
  DenseElementsAttr epsilonConstAttr =
      DenseElementsAttr::get<float>(epsilonType, epsilon.getValueAsDouble());
  Value epsilonConst = rewriter.create<ONNXConstantOp>(loc, epsilonType,
      nullptr, epsilonConstAttr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);

  // sqrt(var + epsilon)
  Value var_plus_epsilon = rewriter.create<ONNXAddOp>(loc, var, epsilonConst);
  Value sqrtResult =
      rewriter.create<ONNXSqrtOp>(loc, var.getType(), var_plus_epsilon);

  return sqrtResult;
}

// Reshape: B1xB2x...xBkxMxN to BxMxN
Value reshapeTo3D(PatternRewriter &rewriter, Location loc, Value val) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  int64_t rank = getRank(val.getType());
  assert(rank > 3 && "Require rank > 3");
  ArrayRef<int64_t> shape = getShape(val.getType());
  Type elementType = getElementType(val.getType());
  Type shapeType = RankedTensorType::get({rank}, rewriter.getI64Type());
  Type twoI64Type = RankedTensorType::get({2}, rewriter.getI64Type());
  Type threeI64Type = RankedTensorType::get({3}, rewriter.getI64Type());

  Value shapeVal = create.onnx.shape(shapeType, val);

  Value zero =
      create.onnx.constant(rewriter.getI64TensorAttr(ArrayRef<int64_t>({0})));
  Value one =
      create.onnx.constant(rewriter.getI64TensorAttr(ArrayRef<int64_t>({1})));
  Value minusOne =
      create.onnx.constant(rewriter.getI64TensorAttr(ArrayRef<int64_t>({-1})));
  Value r2Const = create.onnx.constant(
      rewriter.getI64TensorAttr(ArrayRef<int64_t>({rank - 2})));
  Value rConst = create.onnx.constant(
      rewriter.getI64TensorAttr(ArrayRef<int64_t>({rank})));
  Value lastTwoDimVal =
      create.onnx.slice(twoI64Type, shapeVal, r2Const, rConst, zero, one);

  IntegerAttr concatAxis =
      IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 0, /*isSigned=*/true));
  // newShapeVal is [-1, M, N] where M and N are the last dims in the input.
  Value newShapeVal = create.onnx.concat(
      threeI64Type, ValueRange({minusOne, lastTwoDimVal}), concatAxis);

  // Shape inference will infer the correct shape later.
  return create.onnx.reshape(
      RankedTensorType::get(
          {-1, shape[rank - 2], shape[rank - 1]}, elementType),
      val, newShapeVal);
}

// Get a value that store the shape of the matmul result.
Value getMatMulResultShape(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  int64_t lhsRank = getRank(lhs.getType());
  int64_t rhsRank = getRank(rhs.getType());
  assert((lhsRank >= 2 && rhsRank >= 2) && "Input rank must be >= 2");
  // lhs shape: B1xB2x...xBkxMxN or MxN
  // rhs shape: B1xB2x...xBkxNxP or NxP

  int64_t rank = std::max(lhsRank, rhsRank);
  IntegerAttr concatAxisAttr =
      IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 0, /*isSigned=*/true));

  Type rI64Type = RankedTensorType::get({rank}, rewriter.getI64Type());
  Type lhsRType = RankedTensorType::get({lhsRank}, rewriter.getI64Type());
  Type lhsR1Type = RankedTensorType::get({lhsRank - 1}, rewriter.getI64Type());
  Type rhsRType = RankedTensorType::get({rhsRank}, rewriter.getI64Type());
  Type rhsR2Type = RankedTensorType::get({rhsRank - 2}, rewriter.getI64Type());
  Type oneI64Type = RankedTensorType::get({1}, rewriter.getI64Type());

  Value lhsShape = create.onnx.shape(lhsRType, lhs);
  Value rhsShape = create.onnx.shape(rhsRType, rhs);

  Value zero =
      create.onnx.constant(rewriter.getI64TensorAttr(ArrayRef<int64_t>({0})));
  Value one =
      create.onnx.constant(rewriter.getI64TensorAttr(ArrayRef<int64_t>({1})));
  Value lhsR1Const = create.onnx.constant(
      rewriter.getI64TensorAttr(ArrayRef<int64_t>({lhsRank - 1})));
  Value rhsRConst = create.onnx.constant(
      rewriter.getI64TensorAttr(ArrayRef<int64_t>({rhsRank})));
  Value rhsR1Const = create.onnx.constant(
      rewriter.getI64TensorAttr(ArrayRef<int64_t>({rhsRank - 1})));

  // if lhsRank >= rhsRank:
  //   - get B1xB2x...xBkxM from lhs shape, then append P from rhs shape.
  // else
  //   - get B1xB2x...xBk from rhs shape, then append M from lhs and append P
  //   from rhs shape.
  Value shapeVal;
  if (lhsRank >= rhsRank) {
    Value bmVal =
        create.onnx.slice(lhsR1Type, lhsShape, zero, lhsR1Const, zero, one);
    Value pVal = create.onnx.slice(
        oneI64Type, rhsShape, rhsR1Const, rhsRConst, zero, one);
    shapeVal =
        create.onnx.concat(rI64Type, ValueRange({bmVal, pVal}), concatAxisAttr);
  } else {
    Value lhsR2Const = create.onnx.constant(
        rewriter.getI64TensorAttr(ArrayRef<int64_t>({lhsRank - 2})));
    Value rhsR2Const = create.onnx.constant(
        rewriter.getI64TensorAttr(ArrayRef<int64_t>({rhsRank - 2})));
    Value bVal =
        create.onnx.slice(rhsR2Type, rhsShape, zero, rhsR2Const, zero, one);
    Value mVal = create.onnx.slice(
        oneI64Type, lhsShape, lhsR2Const, lhsR1Const, zero, one);
    Value pVal = create.onnx.slice(
        oneI64Type, rhsShape, rhsR1Const, rhsRConst, zero, one);
    shapeVal = create.onnx.concat(
        rI64Type, ValueRange({bVal, mVal, pVal}), concatAxisAttr);
  }
  return shapeVal;
}

// Get result type of matmul.
Type getMatMulResultType(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  Type elementType = getElementType(lhs.getType());
  int64_t lhsRank = getRank(lhs.getType());
  int64_t rhsRank = getRank(rhs.getType());
  assert((lhsRank >= 2 && rhsRank >= 2) && "Input rank must be >= 2");
  // lhs shape: B1xB2x...xBkxMxN or MxN
  // rhs shape: B1xB2x...xBkxNxP or NxP

  int64_t rank = std::max(lhsRank, rhsRank);
  ArrayRef<int64_t> lhsShape = getShape(lhs.getType());
  ArrayRef<int64_t> rhsShape = getShape(rhs.getType());

  // if lhsRank >= rhsRank:
  //   - get B1xB2x...xBkxM from lhs shape, then append P from rhs shape.
  // else
  //   - get B1xB2x...xBk from rhs shape, then append M from lhs and append P
  //   from rhs shape.
  if (lhsRank >= rhsRank) {
    SmallVector<int64_t, 4> resultShape(lhsShape.begin(), lhsShape.end());
    resultShape[rank - 1] = rhsShape[rhsRank - 1];
    return RankedTensorType::get(resultShape, elementType);
  }

  SmallVector<int64_t, 4> resultShape(rhsShape.begin(), rhsShape.end());
  resultShape[rank - 2] = lhsShape[lhsRank - 2];
  resultShape[rank - 1] = rhsShape[rhsRank - 1];
  return RankedTensorType::get(resultShape, elementType);
}

/// Check if A is unidirectionally broadcastable to B, e.g.
/// A: [256], B: [128x256]
/// A: [1], B: [128x256]
/// More info about unidirectional broadcasting:
/// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
/// Note: being differenct from ONNX broadcasting, we return false if A and B
/// have exactly the same static shape.
bool isUniBroadcatableFirstToSecond(Value A, Value B) {
  if (!hasStaticShape(A.getType()) || !hasStaticShape(B.getType()))
    return false;
  ArrayRef<int64_t> aDims = getShape(A.getType());
  ArrayRef<int64_t> bDims = getShape(B.getType());
  // A and B have exactly the same static shape.
  if (aDims == bDims)
    return false;
  // aDims size > bDims size: not unidirectional broadcasting from A to B, but B
  // to A.
  if (aDims.size() > bDims.size())
    return false;
  // Pre-pad A's shape with dims 1 so that two shapes have the same size.
  SmallVector<int64_t> paddedADims(bDims.size(), 1);
  for (unsigned i = 0; i < aDims.size(); ++i)
    paddedADims[i + bDims.size() - aDims.size()] = aDims[i];
  // Check unidirectional broadcasting.
  return llvm::all_of(llvm::zip(paddedADims, bDims), [](auto v) {
    return ((std::get<0>(v) == 1 && std::get<1>(v) != 1) ||
            (std::get<0>(v) == std::get<1>(v)));
  });
}

/// Check a value is defined by ONNXConstantOp or not.
bool isDefinedByONNXConstantOp(Value v) {
  if (v.isa<BlockArgument>())
    return false;
  return isa<ONNXConstantOp>(v.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// Rewrite ONNX ops to ZHigh ops and ONNX ops for ZHigh.
//===----------------------------------------------------------------------===//

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXRewriteONNXForZHigh.inc"

struct RewriteONNXForZHighPass
    : public PassWrapper<RewriteONNXForZHighPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "rewrite-onnx-for-zhigh"; }

  StringRef getDescription() const override {
    return "Rewrite ONNX ops for ZHigh.";
  }

  RewriteONNXForZHighPass() = default;
  RewriteONNXForZHighPass(mlir::ArrayRef<std::string> execNodesOnCpu)
      : execNodesOnCpu(execNodesOnCpu) {}
  void runOnOperation() final;

public:
  mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>();
};

void RewriteONNXForZHighPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, func::FuncDialect>();

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);

  // `ONNXBatchNormalizationInferenceModeOp` to `ZHigh.BatchNorm`,
  // generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
  // and `ONNX.Sqrt` to calculate inputs(`a` and `b`)
  addDynamicallyLegalOpFor<ONNXBatchNormalizationInferenceModeOp>(
      &target, execNodesOnCpu);

  // Legalize BinaryOp if one of the two inputs is a constant and unidirectional
  // broadcastable to the other input. Rewrite patterns will be added to turn a
  // broadcasting op into a non-broadcasting op.
  //
  // This is preferred for NNPA because NNPA BinaryOp does not support
  // broadcasting.
  target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXDivOp>([](ONNXDivOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXSubOp>([](ONNXSubOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  // Legalize MatMulOp if
  // - both inputs are *the same* N-D, N > 3, or
  // - one input is N-D, N > 3 and the other is 2-D.
  // Rewrite patterns will be added to turn this MatMulOp into the one where N-D
  // will become 3-D.
  target.addDynamicallyLegalOp<ONNXMatMulOp>([](ONNXMatMulOp op) {
    Type aType = op.A().getType();
    Type bType = op.B().getType();
    if (!isRankedShapedType(aType) || !isRankedShapedType(bType))
      return true;

    int64_t aRank = getRank(aType);
    int64_t bRank = getRank(bType);
    if (aRank == 2 && bRank > 3)
      return false;
    if (bRank == 2 && aRank > 3)
      return false;
    if (aRank > 3 && (aRank == bRank))
      return false;

    return true;
  });

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass() {
  return std::make_unique<RewriteONNXForZHighPass>();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  return std::make_unique<RewriteONNXForZHighPass>(execNodesOnCpu);
}

} // namespace onnx_mlir
