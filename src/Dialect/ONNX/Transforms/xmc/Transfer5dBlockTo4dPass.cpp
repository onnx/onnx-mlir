// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass transforms 5D block operations to 4D equivalents for more
// efficient execution. Supports multiple patterns involving reshape, transpose,
// concat, and elementwise operations.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "transfer-5d-block-to-4d"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Creates a DenseElementsAttr constant from shape values
DenseElementsAttr getShapeAttr(MLIRContext *ctx, ArrayRef<int64_t> shape) {
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(shape.size())}, IntegerType::get(ctx, 64));
  return DenseElementsAttr::get(tensorType, shape);
}

/// Converts a vector of integers into an MLIR ArrayAttr
ArrayAttr getI64ArrayAttr(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  for (auto val : values)
    attrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), val));
  return ArrayAttr::get(ctx, attrs);
}

/// Check if reshape converts from 4D to 5D
bool isReshape4Dto5D(ONNXReshapeOp reshapeOp) {
  auto inputType = dyn_cast<RankedTensorType>(reshapeOp.getData().getType());
  auto outputType = dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!inputType || !outputType)
    return false;
  return inputType.getRank() == 4 && outputType.getRank() == 5;
}

/// Get transpose perm as vector
SmallVector<int64_t, 5> getTransposePerm(ONNXTransposeOp transposeOp) {
  SmallVector<int64_t, 5> perm;
  if (auto permAttr = transposeOp.getPermAttr()) {
    for (auto attr : permAttr)
      perm.push_back(cast<IntegerAttr>(attr).getInt());
  }
  return perm;
}

/// Check if perm matches expected pattern {0, 3, 4, 1, 2}
bool isTransposePerm03412(ONNXTransposeOp transposeOp) {
  auto perm = getTransposePerm(transposeOp);
  return perm == SmallVector<int64_t, 5>{0, 3, 4, 1, 2};
}

/// Create shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape) {
  auto *ctx = rewriter.getContext();
  auto shapeAttr = getShapeAttr(ctx, shape);
  return rewriter.create<ONNXConstantOp>(loc, shapeAttr.getType(), Attribute(),
      shapeAttr, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(),
      StringAttr(), ArrayAttr());
}

//===----------------------------------------------------------------------===//
// Generic Elementwise Op Helpers
//===----------------------------------------------------------------------===//

/// Enum for supported binary elementwise operations
enum class EltwiseOpKind { Add, Sub, Mul, Div, Unknown };

/// Get the kind of binary elementwise op from an operation
EltwiseOpKind getEltwiseOpKind(Operation *op) {
  if (!op)
    return EltwiseOpKind::Unknown;
  if (isa<ONNXAddOp>(op))
    return EltwiseOpKind::Add;
  if (isa<ONNXSubOp>(op))
    return EltwiseOpKind::Sub;
  if (isa<ONNXMulOp>(op))
    return EltwiseOpKind::Mul;
  if (isa<ONNXDivOp>(op))
    return EltwiseOpKind::Div;
  return EltwiseOpKind::Unknown;
}

/// Try to get a binary elementwise op from a value
Operation *getEltwiseOp(Value v) {
  auto *op = v.getDefiningOp();
  if (getEltwiseOpKind(op) != EltwiseOpKind::Unknown)
    return op;
  return nullptr;
}

/// Get the first operand (A) of a binary elementwise op
Value getEltwiseOpA(Operation *op) {
  if (getEltwiseOpKind(op) != EltwiseOpKind::Unknown)
    return op->getOperand(0);
  return nullptr;
}

/// Get the second operand (B) of a binary elementwise op
Value getEltwiseOpB(Operation *op) {
  if (getEltwiseOpKind(op) != EltwiseOpKind::Unknown)
    return op->getOperand(1);
  return nullptr;
}

/// Create a new binary elementwise op of the specified kind
Value createEltwiseOp(PatternRewriter &rewriter, Location loc,
    EltwiseOpKind kind, Type resultType, Value lhs, Value rhs) {
  switch (kind) {
  case EltwiseOpKind::Add:
    return rewriter.create<ONNXAddOp>(loc, resultType, lhs, rhs).getResult();
  case EltwiseOpKind::Sub:
    return rewriter.create<ONNXSubOp>(loc, resultType, lhs, rhs).getResult();
  case EltwiseOpKind::Mul:
    return rewriter.create<ONNXMulOp>(loc, resultType, lhs, rhs).getResult();
  case EltwiseOpKind::Div:
    return rewriter.create<ONNXDivOp>(loc, resultType, lhs, rhs).getResult();
  default:
    llvm_unreachable("Unknown elementwise op kind");
  }
}

//===----------------------------------------------------------------------===//
// Pattern 1: Reshape + Add + Concat(axis=4) + Transpose{0,3,4,1,2} + Reshape
//===----------------------------------------------------------------------===//

class Transfer5dBlockWithTransposePattern
    : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOut, PatternRewriter &rewriter) const override {
    // Check output reshape: should have 5D input and produce final output
    auto reshapeOutInputType =
        dyn_cast<RankedTensorType>(reshapeOut.getData().getType());
    if (!reshapeOutInputType || reshapeOutInputType.getRank() != 5)
      return rewriter.notifyMatchFailure(
          reshapeOut, "expected 5D input to output reshape");

    // Input to output reshape should be transpose
    auto transposeOp = reshapeOut.getData().getDefiningOp<ONNXTransposeOp>();
    if (!transposeOp)
      return failure();

    // Check transpose has perm {0, 3, 4, 1, 2}
    if (!isTransposePerm03412(transposeOp))
      return rewriter.notifyMatchFailure(
          reshapeOut, "transpose perm must be {0, 3, 4, 1, 2}");

    // Input to transpose should be concat
    auto concatOp = transposeOp.getData().getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return failure();

    // Check concat axis is 4
    auto axis = concatOp.getAxis();
    if (axis != 4)
      return rewriter.notifyMatchFailure(reshapeOut, "concat axis must be 4");

    // Concat should have exactly 2 inputs, each from an element-wise op
    auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() != 2)
      return rewriter.notifyMatchFailure(
          reshapeOut, "concat must have exactly 2 inputs");

    // Get the two eltwise ops
    auto *eltwise0 = getEltwiseOp(concatInputs[0]);
    auto *eltwise1 = getEltwiseOp(concatInputs[1]);
    if (!eltwise0 || !eltwise1)
      return rewriter.notifyMatchFailure(
          reshapeOut, "concat inputs must be from supported elementwise ops");

    auto eltwiseKind0 = getEltwiseOpKind(eltwise0);
    auto eltwiseKind1 = getEltwiseOpKind(eltwise1);

    // Get reshape ops feeding the eltwise ops
    auto reshape0 = getEltwiseOpA(eltwise0).getDefiningOp<ONNXReshapeOp>();
    auto reshape1 = getEltwiseOpA(eltwise1).getDefiningOp<ONNXReshapeOp>();
    if (!reshape0 || !reshape1)
      return failure();

    // Check reshapes are 4D->5D
    if (!isReshape4Dto5D(reshape0) || !isReshape4Dto5D(reshape1))
      return rewriter.notifyMatchFailure(
          reshapeOut, "input reshapes must be 4D->5D");

    // Transform: merge dimensions 1,2 for 4D operations
    auto loc = reshapeOut.getLoc();
    auto *ctx = rewriter.getContext();

    // Get 5D shapes and element types
    auto reshape0OutputType =
        cast<RankedTensorType>(reshape0.getResult().getType());
    auto reshape1OutputType =
        cast<RankedTensorType>(reshape1.getResult().getType());
    auto shape0_5D = reshape0OutputType.getShape();
    auto shape1_5D = reshape1OutputType.getShape();
    auto elemTy0 = reshape0OutputType.getElementType();
    auto elemTy1 = reshape1OutputType.getElementType();
    auto eltwise0ElemTy =
        cast<RankedTensorType>(eltwise0->getResult(0).getType())
            .getElementType();
    auto eltwise1ElemTy =
        cast<RankedTensorType>(eltwise1->getResult(0).getType())
            .getElementType();
    auto concatElemTy =
        cast<RankedTensorType>(concatOp.getResult().getType()).getElementType();
    auto transposeElemTy =
        cast<RankedTensorType>(transposeOp.getResult().getType())
            .getElementType();

    // Compute 4D shapes by merging dims 1 and 2: [N, C*D, H, W]
    SmallVector<int64_t, 4> shape0_4D = {
        shape0_5D[0], shape0_5D[1] * shape0_5D[2], shape0_5D[3], shape0_5D[4]};
    SmallVector<int64_t, 4> shape1_4D = {
        shape1_5D[0], shape1_5D[1] * shape1_5D[2], shape1_5D[3], shape1_5D[4]};

    // Create new 4D reshape ops
    auto reshape0Type4D = RankedTensorType::get(shape0_4D, elemTy0);
    auto reshape1Type4D = RankedTensorType::get(shape1_4D, elemTy1);

    auto shape0Const4D = createShapeConstant(rewriter, loc, shape0_4D);
    auto shape1Const4D = createShapeConstant(rewriter, loc, shape1_4D);

    auto newReshape0 = rewriter.create<ONNXReshapeOp>(
        loc, reshape0Type4D, reshape0.getData(), shape0Const4D);
    auto newReshape1 = rewriter.create<ONNXReshapeOp>(
        loc, reshape1Type4D, reshape1.getData(), shape1Const4D);

    // Create new 4D eltwise ops
    auto eltwise0Type4D = RankedTensorType::get(shape0_4D, eltwise0ElemTy);
    auto eltwise1Type4D = RankedTensorType::get(shape1_4D, eltwise1ElemTy);
    auto newEltwise0 = createEltwiseOp(rewriter, loc, eltwiseKind0,
        eltwise0Type4D, newReshape0.getResult(), getEltwiseOpB(eltwise0));
    auto newEltwise1 = createEltwiseOp(rewriter, loc, eltwiseKind1,
        eltwise1Type4D, newReshape1.getResult(), getEltwiseOpB(eltwise1));

    // Create new 4D concat (axis=3 instead of 4)
    SmallVector<int64_t, 4> concatShape4D = {
        shape0_4D[0], shape0_4D[1], shape0_4D[2], shape0_4D[3] + shape1_4D[3]};
    auto concatType4D = RankedTensorType::get(concatShape4D, concatElemTy);
    auto newConcat = rewriter.create<ONNXConcatOp>(
        loc, concatType4D, ValueRange{newEltwise0, newEltwise1}, /*axis=*/3);

    // Create new 4D transpose with perm {0, 2, 3, 1}
    SmallVector<int64_t, 4> transposeShape4D = {
        concatShape4D[0], concatShape4D[2], concatShape4D[3], concatShape4D[1]};
    auto transposeType4D =
        RankedTensorType::get(transposeShape4D, transposeElemTy);
    auto newTranspose = rewriter.create<ONNXTransposeOp>(loc, transposeType4D,
        newConcat.getResult(), getI64ArrayAttr(ctx, {0, 2, 3, 1}));

    // Create final reshape back to original output shape
    auto outputType = cast<RankedTensorType>(reshapeOut.getResult().getType());
    auto outputShape = outputType.getShape();
    auto outputShapeConst = createShapeConstant(rewriter, loc, outputShape);
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, newTranspose.getResult(), outputShapeConst);

    rewriter.replaceOp(reshapeOut, finalReshape.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Two-branch concat without transpose
//===----------------------------------------------------------------------===//

class Transfer5dBlockWithConcatPattern
    : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOut, PatternRewriter &rewriter) const override {
    // Check output reshape: should have 5D input
    auto reshapeOutInputType =
        dyn_cast<RankedTensorType>(reshapeOut.getData().getType());
    if (!reshapeOutInputType || reshapeOutInputType.getRank() != 5)
      return rewriter.notifyMatchFailure(
          reshapeOut, "expected 5D input to output reshape");

    // Input to output reshape should be concat directly (no transpose)
    auto concatOp = reshapeOut.getData().getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return failure();

    // Check concat axis is 2
    auto axis = concatOp.getAxis();
    if (axis != 2)
      return rewriter.notifyMatchFailure(reshapeOut, "concat axis must be 2");

    // Concat should have exactly 2 inputs
    auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() != 2)
      return rewriter.notifyMatchFailure(
          reshapeOut, "concat must have exactly 2 inputs");

    // Get the two eltwise ops
    auto *eltwise0 = getEltwiseOp(concatInputs[0]);
    auto *eltwise1 = getEltwiseOp(concatInputs[1]);
    if (!eltwise0 || !eltwise1)
      return rewriter.notifyMatchFailure(
          reshapeOut, "concat inputs must be from supported elementwise ops");

    auto eltwiseKind0 = getEltwiseOpKind(eltwise0);
    auto eltwiseKind1 = getEltwiseOpKind(eltwise1);

    // Get reshape ops feeding the eltwise ops
    auto reshape0 = getEltwiseOpA(eltwise0).getDefiningOp<ONNXReshapeOp>();
    auto reshape1 = getEltwiseOpA(eltwise1).getDefiningOp<ONNXReshapeOp>();
    if (!reshape0 || !reshape1)
      return failure();

    // Check reshapes are 4D->5D
    if (!isReshape4Dto5D(reshape0) || !isReshape4Dto5D(reshape1))
      return rewriter.notifyMatchFailure(
          reshapeOut, "input reshapes must be 4D->5D");

    // Transform: merge dimensions 3,4 for 4D operations
    auto loc = reshapeOut.getLoc();

    // Get 5D shapes and element types
    auto reshape0OutputType =
        cast<RankedTensorType>(reshape0.getResult().getType());
    auto reshape1OutputType =
        cast<RankedTensorType>(reshape1.getResult().getType());
    auto shape0_5D = reshape0OutputType.getShape();
    auto shape1_5D = reshape1OutputType.getShape();
    auto elemTy0 = reshape0OutputType.getElementType();
    auto elemTy1 = reshape1OutputType.getElementType();
    auto eltwise0ElemTy =
        cast<RankedTensorType>(eltwise0->getResult(0).getType())
            .getElementType();
    auto eltwise1ElemTy =
        cast<RankedTensorType>(eltwise1->getResult(0).getType())
            .getElementType();
    auto concatElemTy =
        cast<RankedTensorType>(concatOp.getResult().getType()).getElementType();

    // Compute 4D shapes by merging dims 3 and 4: [N, C, D, H*W]
    SmallVector<int64_t, 4> shape0_4D = {
        shape0_5D[0], shape0_5D[1], shape0_5D[2], shape0_5D[3] * shape0_5D[4]};
    SmallVector<int64_t, 4> shape1_4D = {
        shape1_5D[0], shape1_5D[1], shape1_5D[2], shape1_5D[3] * shape1_5D[4]};

    // Create new 4D reshape ops
    auto reshape0Type4D = RankedTensorType::get(shape0_4D, elemTy0);
    auto reshape1Type4D = RankedTensorType::get(shape1_4D, elemTy1);

    auto shape0Const4D = createShapeConstant(rewriter, loc, shape0_4D);
    auto shape1Const4D = createShapeConstant(rewriter, loc, shape1_4D);

    auto newReshape0 = rewriter.create<ONNXReshapeOp>(
        loc, reshape0Type4D, reshape0.getData(), shape0Const4D);
    auto newReshape1 = rewriter.create<ONNXReshapeOp>(
        loc, reshape1Type4D, reshape1.getData(), shape1Const4D);

    // Create new 4D eltwise ops
    auto eltwise0Type4D = RankedTensorType::get(shape0_4D, eltwise0ElemTy);
    auto eltwise1Type4D = RankedTensorType::get(shape1_4D, eltwise1ElemTy);
    auto newEltwise0 = createEltwiseOp(rewriter, loc, eltwiseKind0,
        eltwise0Type4D, newReshape0.getResult(), getEltwiseOpB(eltwise0));
    auto newEltwise1 = createEltwiseOp(rewriter, loc, eltwiseKind1,
        eltwise1Type4D, newReshape1.getResult(), getEltwiseOpB(eltwise1));

    // Create new 4D concat (axis=2 stays same since we merged 3,4)
    SmallVector<int64_t, 4> concatShape4D = {
        shape0_4D[0], shape0_4D[1], shape0_4D[2] + shape1_4D[2], shape0_4D[3]};
    auto concatType4D = RankedTensorType::get(concatShape4D, concatElemTy);
    auto newConcat = rewriter.create<ONNXConcatOp>(
        loc, concatType4D, ValueRange{newEltwise0, newEltwise1}, /*axis=*/2);

    // Create final reshape back to original output shape
    auto outputType = cast<RankedTensorType>(reshapeOut.getResult().getType());
    auto outputShape = outputType.getShape();
    auto outputShapeConst = createShapeConstant(rewriter, loc, outputShape);
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, newConcat.getResult(), outputShapeConst);

    rewriter.replaceOp(reshapeOut, finalReshape.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 3: Two-branch eltwise with transpose
//===----------------------------------------------------------------------===//

class Transfer5dEltwiseBlockPattern : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOut, PatternRewriter &rewriter) const override {
    // Check output reshape: should have 5D input
    auto reshapeOutInputType =
        dyn_cast<RankedTensorType>(reshapeOut.getData().getType());
    if (!reshapeOutInputType || reshapeOutInputType.getRank() != 5)
      return rewriter.notifyMatchFailure(
          reshapeOut, "expected 5D input to output reshape");

    // Input to output reshape should be a binary elementwise op
    auto *eltwiseOp = getEltwiseOp(reshapeOut.getData());
    if (!eltwiseOp)
      return rewriter.notifyMatchFailure(
          reshapeOut, "expected supported elementwise op before reshape");

    auto eltwiseKind = getEltwiseOpKind(eltwiseOp);

    // One input to eltwise should be from transpose
    ONNXTransposeOp transposeOp = nullptr;
    ONNXReshapeOp reshape1 = nullptr;
    Value otherEltwiseInput;
    bool transposeWasFirstOperand = false;

    if (auto t = getEltwiseOpA(eltwiseOp).getDefiningOp<ONNXTransposeOp>()) {
      transposeOp = t;
      otherEltwiseInput = getEltwiseOpB(eltwiseOp);
      transposeWasFirstOperand = true;
    } else if (auto t =
                   getEltwiseOpB(eltwiseOp).getDefiningOp<ONNXTransposeOp>()) {
      transposeOp = t;
      otherEltwiseInput = getEltwiseOpA(eltwiseOp);
    }

    if (!transposeOp)
      return rewriter.notifyMatchFailure(
          reshapeOut, "one eltwise input must be from transpose op");

    // Check transpose has perm {0, 3, 4, 1, 2}
    if (!isTransposePerm03412(transposeOp))
      return rewriter.notifyMatchFailure(
          reshapeOut, "transpose perm must be {0, 3, 4, 1, 2}");

    // Input to transpose should be reshape
    auto reshape0 = transposeOp.getData().getDefiningOp<ONNXReshapeOp>();
    if (!reshape0)
      return failure();

    // Other eltwise input should be from reshape
    reshape1 = otherEltwiseInput.getDefiningOp<ONNXReshapeOp>();
    if (!reshape1)
      return failure();

    // Check reshapes are 4D->5D
    if (!isReshape4Dto5D(reshape0) || !isReshape4Dto5D(reshape1))
      return rewriter.notifyMatchFailure(
          reshapeOut, "input reshapes must be 4D->5D");

    // Transform to 4D
    auto loc = reshapeOut.getLoc();
    auto *ctx = rewriter.getContext();

    // Get 5D shapes and element types
    auto reshape0OutputType =
        cast<RankedTensorType>(reshape0.getResult().getType());
    auto reshape1OutputType =
        cast<RankedTensorType>(reshape1.getResult().getType());
    auto shape0_5D = reshape0OutputType.getShape();
    auto shape1_5D = reshape1OutputType.getShape();
    auto elemTy0 = reshape0OutputType.getElementType();
    auto elemTy1 = reshape1OutputType.getElementType();
    auto transposeElemTy =
        cast<RankedTensorType>(transposeOp.getResult().getType())
            .getElementType();
    auto eltwiseElemTy =
        cast<RankedTensorType>(eltwiseOp->getResult(0).getType())
            .getElementType();

    // For reshape0 (before transpose): merge dims 1,2: [N, C*D, H, W]
    SmallVector<int64_t, 4> shape0_4D = {
        shape0_5D[0], shape0_5D[1] * shape0_5D[2], shape0_5D[3], shape0_5D[4]};
    // For reshape1: merge dims 3,4: [N, C, D, H*W]
    SmallVector<int64_t, 4> shape1_4D = {
        shape1_5D[0], shape1_5D[1], shape1_5D[2], shape1_5D[3] * shape1_5D[4]};

    // Create new 4D reshape ops
    auto reshape0Type4D = RankedTensorType::get(shape0_4D, elemTy0);
    auto reshape1Type4D = RankedTensorType::get(shape1_4D, elemTy1);

    auto shape0Const4D = createShapeConstant(rewriter, loc, shape0_4D);
    auto shape1Const4D = createShapeConstant(rewriter, loc, shape1_4D);

    auto newReshape0 = rewriter.create<ONNXReshapeOp>(
        loc, reshape0Type4D, reshape0.getData(), shape0Const4D);
    auto newReshape1 = rewriter.create<ONNXReshapeOp>(
        loc, reshape1Type4D, reshape1.getData(), shape1Const4D);

    // Create new 4D transpose with perm {0, 2, 3, 1}
    SmallVector<int64_t, 4> transposeShape4D = {
        shape0_4D[0], shape0_4D[2], shape0_4D[3], shape0_4D[1]};
    auto transposeType4D =
        RankedTensorType::get(transposeShape4D, transposeElemTy);
    auto newTranspose = rewriter.create<ONNXTransposeOp>(loc, transposeType4D,
        newReshape0.getResult(), getI64ArrayAttr(ctx, {0, 2, 3, 1}));

    // Create new 4D eltwise with computed broadcasted shape
    Value lhs = transposeWasFirstOperand ? newTranspose.getResult()
                                         : newReshape1.getResult();
    Value rhs = transposeWasFirstOperand ? newReshape1.getResult()
                                         : newTranspose.getResult();

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    SmallVector<int64_t> eltwiseShape;
    if (!OpTrait::util::getBroadcastedShape(
            lhsType.getShape(), rhsType.getShape(), eltwiseShape))
      return failure();

    auto newEltwiseType = RankedTensorType::get(eltwiseShape, eltwiseElemTy);
    auto newEltwise =
        createEltwiseOp(rewriter, loc, eltwiseKind, newEltwiseType, lhs, rhs);

    // Create final reshape directly to original output shape
    auto outputType = cast<RankedTensorType>(reshapeOut.getResult().getType());
    auto outputShape = outputType.getShape();
    auto outputShapeConst = createShapeConstant(rewriter, loc, outputShape);
    auto finalReshape = rewriter.create<ONNXReshapeOp>(
        loc, outputType, newEltwise, outputShapeConst);

    rewriter.replaceOp(reshapeOut, finalReshape.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer 5D block operations to 4D equivalents
struct Transfer5dBlockTo4dPass
    : public PassWrapper<Transfer5dBlockTo4dPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "transfer-5d-block-to-4d"; }
  StringRef getDescription() const override {
    return "Transfer 5D block operations to 4D equivalents";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Add all patterns
    patterns.add<Transfer5dBlockWithTransposePattern>(ctx);
    patterns.add<Transfer5dBlockWithConcatPattern>(ctx);
    patterns.add<Transfer5dEltwiseBlockPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransfer5dBlockTo4dPass() {
  return std::make_unique<Transfer5dBlockTo4dPass>();
}

} // namespace onnx_mlir
