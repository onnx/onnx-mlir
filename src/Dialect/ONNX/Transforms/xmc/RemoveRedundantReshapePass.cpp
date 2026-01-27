// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// RemoveRedundantReshapePass
//
// This pass removes redundant reshape operations around element-wise ops.
// Patterns:
// 1. Reshape -> Sigmoid -> Reshape  =>  Sigmoid -> Reshape
// 2. Reshape -> BinaryOp <- Reshape -> Reshape  =>  BinaryOp -> Reshape
//    (Add, Mul, Sub)
//
// Note: Q/DQ nodes are not handled here as they are replaced with quant types.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a value is a constant
bool isConstant(Value val) {
  return val.getDefiningOp<ONNXConstantOp>() != nullptr;
}

/// Check if a value is NOT a constant
bool isNotConstant(Value val) {
  return !isConstant(val);
}

/// Check if two reshape operations have equivalent shapes (same reshape action)
bool equalReshapeAction(ONNXReshapeOp reshape1, ONNXReshapeOp reshape2) {
  // Get shape constants
  auto shape1Op = reshape1.getShape().getDefiningOp<ONNXConstantOp>();
  auto shape2Op = reshape2.getShape().getDefiningOp<ONNXConstantOp>();
  
  if (!shape1Op || !shape2Op)
    return false;
  
  auto shape1Attr = mlir::dyn_cast<DenseElementsAttr>(shape1Op.getValueAttr());
  auto shape2Attr = mlir::dyn_cast<DenseElementsAttr>(shape2Op.getValueAttr());
  
  if (!shape1Attr || !shape2Attr)
    return false;
  
  // Compare shapes
  auto shape1Values = shape1Attr.getValues<int64_t>();
  auto shape2Values = shape2Attr.getValues<int64_t>();
  
  if (shape1Attr.size() != shape2Attr.size())
    return false;
  
  auto it1 = shape1Values.begin();
  auto it2 = shape2Values.begin();
  while (it1 != shape1Values.end()) {
    if (*it1 != *it2)
      return false;
    ++it1;
    ++it2;
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// Pattern 1: Reshape -> Sigmoid -> Reshape
//===----------------------------------------------------------------------===//

struct RemoveReshapeAroundSigmoid : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReshapeOp reshape2,
                                PatternRewriter &rewriter) const override {
    // Check if shape is constant
    if (!isConstant(reshape2.getShape()))
      return failure();

    // Check if input comes from Sigmoid
    auto sigmoidOp = reshape2.getData().getDefiningOp<ONNXSigmoidOp>();
    if (!sigmoidOp)
      return failure();

    // Check if Sigmoid input comes from Reshape
    auto reshape1 = sigmoidOp.getX().getDefiningOp<ONNXReshapeOp>();
    if (!reshape1)
      return failure();

    // Check reshape1 has constant shape and non-constant data
    if (!isConstant(reshape1.getShape()) || !isNotConstant(reshape1.getData()))
      return failure();

    // ============== Pattern matched! Now perform transformation ==============
    Location loc = reshape2.getLoc();

    // Create new Sigmoid with original input (bypassing first reshape)
    auto newSigmoid = rewriter.create<ONNXSigmoidOp>(
        sigmoidOp.getLoc(), reshape1.getData().getType(), reshape1.getData());

    // Create new reshape with Sigmoid output
    auto newReshape = rewriter.create<ONNXReshapeOp>(
        loc, reshape2.getType(), newSigmoid.getResult(), reshape2.getShape(),
        reshape2.getAllowzeroAttr());

    rewriter.replaceOp(reshape2, newReshape.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Reshape -> BinaryOp <- Reshape -> Reshape (Add/Mul/Sub)
//===----------------------------------------------------------------------===//

template <typename BinaryOpTy>
struct RemoveReshapeAroundBinaryOp : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReshapeOp reshape3,
                                PatternRewriter &rewriter) const override {
    // Check if shape is constant
    if (!isConstant(reshape3.getShape()))
      return failure();
    
    // Check if input comes from binary op
    auto binaryOp = reshape3.getData().getDefiningOp<BinaryOpTy>();
    if (!binaryOp)
      return failure();
    
    // Get both inputs of the binary op
    Value lhs = binaryOp.getA();
    Value rhs = binaryOp.getB();
    
    // Check if both inputs come from Reshape
    auto reshape1 = lhs.getDefiningOp<ONNXReshapeOp>();
    auto reshape2 = rhs.getDefiningOp<ONNXReshapeOp>();
    if (!reshape1 || !reshape2)
      return failure();
    
    // Check reshape1 has constant shape and non-constant data
    if (!isConstant(reshape1.getShape()) || !isNotConstant(reshape1.getData()))
      return failure();
    
    // Check reshape2 has constant shape and non-constant data
    if (!isConstant(reshape2.getShape()) || !isNotConstant(reshape2.getData()))
      return failure();
    
    // Check that both reshapes have equivalent shape (equal_reshape_action)
    if (!equalReshapeAction(reshape1, reshape2))
      return failure();
    
    // ============== Pattern matched! Now perform transformation ==============
    Location loc = reshape3.getLoc();
    
    // Get original inputs (bypassing reshapes)
    Value origLhs = reshape1.getData();
    Value origRhs = reshape2.getData();
    
    // Compute output type for new binary op
    auto origLhsType = mlir::dyn_cast<RankedTensorType>(origLhs.getType());
    auto origRhsType = mlir::dyn_cast<RankedTensorType>(origRhs.getType());
    if (!origLhsType || !origRhsType)
      return failure();
    
    auto binaryOutputType = mlir::dyn_cast<RankedTensorType>(binaryOp.getType());
    if (!binaryOutputType)
      return failure();
    
    // New binary op output type
    auto newBinaryType = RankedTensorType::get(
        origLhsType.getShape(), binaryOutputType.getElementType());
    
    // Create new binary op with original inputs (bypassing reshapes)
    auto newBinaryOp = rewriter.create<BinaryOpTy>(
        binaryOp.getLoc(), newBinaryType, origLhs, origRhs);
    
    // Create new reshape with binary op output
    auto newReshape = rewriter.create<ONNXReshapeOp>(
        loc, reshape3.getType(), newBinaryOp.getResult(), reshape3.getShape(),
        reshape3.getAllowzeroAttr());
    
    rewriter.replaceOp(reshape3, newReshape.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct RemoveRedundantReshapePass
    : public PassWrapper<RemoveRedundantReshapePass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "remove-redundant-reshape"; }
  StringRef getDescription() const override {
    return "Remove redundant reshape operations around element-wise ops";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Add all patterns
    patterns.add<RemoveReshapeAroundSigmoid>(context);
    patterns.add<RemoveReshapeAroundBinaryOp<ONNXAddOp>>(context);
    patterns.add<RemoveReshapeAroundBinaryOp<ONNXMulOp>>(context);
    patterns.add<RemoveReshapeAroundBinaryOp<ONNXSubOp>>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createRemoveRedundantReshapePass() {
  return std::make_unique<RemoveRedundantReshapePass>();
}

} // namespace onnx_mlir
