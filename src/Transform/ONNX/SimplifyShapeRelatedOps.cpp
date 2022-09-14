/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- SimplifyShapeRelatedOps.cpp - ONNX high level Optimizations -----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of optimizations to eliminate as many as possible
// shape-related operations that are often generated by ONNX converters.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

Value createONNXConcatOp(
    MultiDialectBuilder<OnnxBuilder> create, ValueRange inputs) {
  if (inputs.size() == 1)
    return inputs[0];
  OpBuilder b = create.onnx.getBuilder();
  int64_t rank = inputs.size();
  Type elementType = getElementType(inputs[0].getType());
  Type resultType = RankedTensorType::get({rank}, elementType);
  Value concatOutput = create.onnx.concat(resultType, inputs, /*axis=*/0);
  // Annotate ONNXConcatOp with an attribute telling this op is storing
  // dimensions. This attribute will be removed before finishing this pass.
  concatOutput.getDefiningOp()->setAttr("for_dims", b.getBoolAttr(true));
  return concatOutput;
}

template <typename OP>
bool definedBy(Value v) {
  return !v.isa<BlockArgument>() && isa<OP>(v.getDefiningOp());
}

bool definedByDimConcatOp(Value val) {
  return (definedBy<ONNXConcatOp>(val) &&
          (val.getDefiningOp()->getAttrOfType<::mlir::Attribute>("for_dims")));
}

OperandRange getDims(Value val) {
  assert(definedByDimConcatOp(val) && "Not defined by ONNXConcatOp");
  return val.getDefiningOp()->getOperands();
}

void getDimsInt64(Value val, SmallVectorImpl<int64_t> &result) {
  for (Value v : getDims(val)) {
    if (definedBy<ONNXConstantOp>(v)) {
      auto constOp = dyn_cast<ONNXConstantOp>(v.getDefiningOp());
      auto valueAttr = constOp.valueAttr().cast<DenseElementsAttr>();
      int64_t dim = valueAttr.getSplatValue<int64_t>();
      result.emplace_back(dim);
    } else {
      result.emplace_back(-1);
    }
  }
}

/// Update the function signature if the op's output is the return value.
void updateFunctionSignature(Operation *op) {
  assert(op->getResults().size() == 1 && "Only support single result ops");
  Operation *parentOp = op->getParentOp();
  if (auto f = dyn_cast<func::FuncOp>(parentOp)) {
    if (!f.back().empty() && f.back().back().hasTrait<OpTrait::IsTerminator>())
      if (auto terminator = f.getBody().back().getTerminator()) {
        for (Operation *user : op->getResults()[0].getUsers()) {
          if (user == terminator) {
            auto results = user->getOperandTypes();
            f.setType(FunctionType::get(f.getContext(),
                f.getFunctionType().getInputs(),
                std::vector<Type>(results.begin(), results.end())));
          }
        }
      }
  }
}

/// Rewrite onnx.Shape into onnx.Dim and onnx.Concat.
//
/// For example, the folowing onnx.Shape:
/// %0 = "onnx.Shape"(%arg0) : (tensor<?x256xi64>) -> tensor<2xi64>
///
/// will be rewritten into:
///
/// %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} :
///           (tensor<?x256xi64>) -> tensor<1xi64>
/// %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} :
///           () -> tensor<1xi64>
/// %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64, for_dims = true} :
///           (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
class ShapeToDimConcatPattern : public OpRewritePattern<ONNXShapeOp> {
public:
  using OpRewritePattern<ONNXShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXShapeOp shapeOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = shapeOp.getLoc();
    Value data = shapeOp.data();

    // Match
    if (!onnx_mlir::isRankedShapedType(data.getType()))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    ArrayRef<int64_t> dims = onnx_mlir::getShape(data.getType());
    int64_t rank = onnx_mlir::getRank(data.getType());

    SmallVector<Value, 4> dimValues;
    for (unsigned i = 0; i < rank; ++i) {
      if (dims[i] != -1)
        dimValues.emplace_back(create.onnx.constantInt64({dims[i]}));
      else
        dimValues.emplace_back(create.onnx.dim(data, i));
    }
    Value replacedValue = createONNXConcatOp(create, dimValues);

    rewriter.replaceOp(shapeOp, replacedValue);
    return success();
  }
};

/// This pattern rewrites
///   ONNXCastOp(ONNXConcatOp)
/// into
///   ONNXConcatOp(ONNXCastOp, ONNXCastOp, ..., ONNXCastOp)
class PassThroughCastPattern : public OpRewritePattern<ONNXCastOp> {
public:
  using OpRewritePattern<ONNXCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXCastOp castOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = castOp.getLoc();
    Value input = castOp.input();
    TypeAttr toType = castOp.toAttr();

    // Match
    if (!definedByDimConcatOp(input))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    OperandRange dims = getDims(input);
    SmallVector<Value> castedDims;
    for (Value d : dims)
      castedDims.emplace_back(create.onnx.cast(d, toType));

    Value replacedValue = createONNXConcatOp(create, castedDims);

    rewriter.replaceOp(castOp, replacedValue);
    return success();
  }
};

// Rewrite ONNXSliceOp into ONNXConcatOp.
class PassThroughSlicePattern : public OpRewritePattern<ONNXSliceOp> {
public:
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = sliceOp.getLoc();
    Value input = sliceOp.data();
    Value starts = sliceOp.starts();
    Value ends = sliceOp.ends();
    Value steps = sliceOp.steps();

    // Match
    // Input is defined by Concat of dims, so it has rank of 1.
    if (!definedByDimConcatOp(input))
      return failure();
    // Starts and ends are constants.
    if (!definedBy<ONNXConstantOp>(starts) ||
        !definedBy<ONNXConstantOp>(ends) || !definedBy<ONNXConstantOp>(steps))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    // Get starts, ends, axes and steps via ShapeHelper.
    ONNXSliceOpShapeHelper shapeHelper(&sliceOp);
    ONNXSliceOpAdaptor operandAdaptor(sliceOp);
    if (failed(shapeHelper.computeShape(operandAdaptor))) {
      sliceOp.emitError("Failed to scan " + ONNXSliceOp::getOperationName() +
                        " parameters successfully");
      return failure();
    }

    // Compute indices of interest.
    SmallVector<int64_t, 4> indices;
    int64_t start = shapeHelper.starts[0].getLiteral();
    int64_t end = shapeHelper.ends[0].getLiteral();
    int64_t step = shapeHelper.steps[0].getLiteral();
    if (step > 0)
      for (int64_t i = start; i < end; i += step)
        indices.emplace_back(i);
    else if (step < 0)
      for (int64_t i = start; i > end; i += step)
        indices.emplace_back(i);
    else
      // step = 0 (invalid).
      return failure();

    // Replace SliceOp by ConcatOp of specific dimensions.
    OperandRange dims = getDims(input);
    SmallVector<Value> slicedDims;
    for (int64_t i : indices)
      slicedDims.emplace_back(dims[i]);

    Value replacedValue = createONNXConcatOp(create, slicedDims);

    rewriter.replaceOp(sliceOp, replacedValue);
    return success();
  }
};

class PassThroughConcatPattern : public OpRewritePattern<ONNXConcatOp> {
public:
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = concatOp.getLoc();
    Operation *genericOp = concatOp.getOperation();
    ValueRange inputs = concatOp.inputs();

    // Match condition:
    // - This ConcatOp does not have "for_dims" attribute, and
    // - Inputs are of 1D Dim or Cast or Constant.

    // Does not have "for_dims" attribute.
    if (genericOp->getAttrOfType<::mlir::Attribute>("for_dims"))
      return failure();

    // Inputs are of 1D Dim or Cast or Constant, and
    SmallVector<Value, 4> dims;
    for (Value v : inputs) {
      // Must be defined by ONNXDimOp or ONNXCastOp or ONNXConstantOp.
      if (!(definedBy<ONNXDimOp>(v) || definedBy<ONNXCastOp>(v) ||
              definedBy<ONNXConstantOp>(v) || definedBy<ONNXConcatOp>(v)))
        return failure();
      // Need a further check for Cast.
      if (definedBy<ONNXCastOp>(v)) {
        Value castInput = v.getDefiningOp()->getOperands()[0];
        if (!(definedBy<ONNXDimOp>(castInput) ||
                definedBy<ONNXConstantOp>(castInput)))
          return failure();
      }
      // Need a further check for Concat.
      if (definedBy<ONNXConcatOp>(v) && !definedByDimConcatOp(v))
        return failure();

      // Check type.
      Type vType = v.getType();
      // Must be ranked.
      if (!isRankedShapedType(vType))
        return failure();
      // Must be 1-D of one element.
      if (getRank(vType) != 1 && getShape(vType)[0] != 1)
        return failure();

      // Store this value for rewrite.
      if (definedByDimConcatOp(v))
        for (Value dim : getDims(v))
          dims.emplace_back(dim);
      else
        dims.emplace_back(v);
    }

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value replacedValue = createONNXConcatOp(create, dims);
    rewriter.replaceOp(concatOp, replacedValue);
    return success();
  }
};

// Update Reshape's output shape if possible.
class UpdateReshapePattern : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXReshapeOp reshapeOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Value shape = reshapeOp.shape();
    Value output = reshapeOp.reshaped();
    int64_t allowZero = reshapeOp.allowzero();

    // Match
    // Does not support allowzero.
    if (allowZero != 0)
      return failure();
    // Shape is defined by Concat of dims, so it has rank of 1.
    if (!definedByDimConcatOp(shape))
      return failure();
    // Get dimensions in the given shape.
    SmallVector<int64_t, 4> userDims;
    getDimsInt64(shape, userDims);
    // Does not support 0 in the given shape.
    if (llvm::any_of(userDims, [](int64_t d) { return d == 0; }))
      return failure();
    // Rewrite only if the given shape is different from the output shape.
    ArrayRef<int64_t> outputDims = getShape(output.getType());
    if (userDims == outputDims)
      return failure();

    // Rewrite
    updateType(output, userDims);
    // Update the function signature.
    updateFunctionSignature(reshapeOp.getOperation());

    return success();
  }
};

// Update ConstantOfShape's output shape if possible.
class UpdateConstantOfShapePattern
    : public OpRewritePattern<ONNXConstantOfShapeOp> {
public:
  using OpRewritePattern<ONNXConstantOfShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConstantOfShapeOp cosOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Value input = cosOp.input();
    Value output = cosOp.output();

    // Match
    // Input is defined by Concat of dims, so it has rank of 1.
    if (!definedByDimConcatOp(input))
      return failure();
    // Rewrite only if the given shape is different from the output shape.
    SmallVector<int64_t, 4> userDims;
    getDimsInt64(input, userDims);
    ArrayRef<int64_t> outputDims = getShape(output.getType());
    if (userDims == outputDims)
      return failure();

    // Rewrite
    updateType(output, userDims);
    // Update the function signature.
    updateFunctionSignature(cosOp.getOperation());

    return success();
  }
};

} // namespace onnx_mlir

namespace {

struct SimplifyShapeRelatedOpsPass
    : public PassWrapper<SimplifyShapeRelatedOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyShapeRelatedOpsPass)

  StringRef getArgument() const override {
    return "simplify-shape-related-ops-onnx";
  }

  StringRef getDescription() const override {
    return "Perform ONNX to ONNX optimizations for shape-related operations";
  }

  void runOnOperation() final;

private:
  void topDownShapeSimplification(MLIRContext *context, ModuleOp moduleOp);
};

void SimplifyShapeRelatedOpsPass::topDownShapeSimplification(
    MLIRContext *context, ModuleOp moduleOp) {
  RewritePatternSet patterns(context);

  // Rewrite onnx.Shape into onnx.Dim and onnx.Concat. By this way, all
  // dimensions are visible to other operations. Thus, it is easy to propagate
  // the dimensions.
  patterns.insert<onnx_mlir::ShapeToDimConcatPattern>(context);

  // Pass the dimensions through operations of interest.
  patterns.insert<onnx_mlir::PassThroughCastPattern>(context);
  patterns.insert<onnx_mlir::PassThroughConcatPattern>(context);
  patterns.insert<onnx_mlir::PassThroughSlicePattern>(context);

  // Update Reshape's output shape using inferred dimensions.
  patterns.insert<onnx_mlir::UpdateReshapePattern>(context);
  patterns.insert<onnx_mlir::UpdateConstantOfShapePattern>(context);

  // Canonicalize shape-related ops during this pass to simplify rules in this
  // pass.
  ONNXCastOp::getCanonicalizationPatterns(patterns, context);
  ONNXDimOp::getCanonicalizationPatterns(patterns, context);
  ONNXReshapeOp::getCanonicalizationPatterns(patterns, context);
  ONNXSliceOp::getCanonicalizationPatterns(patterns, context);
  ONNXSqueezeOp::getCanonicalizationPatterns(patterns, context);
  ONNXSqueezeV11Op::getCanonicalizationPatterns(patterns, context);
  ONNXUnsqueezeOp::getCanonicalizationPatterns(patterns, context);
  ONNXUnsqueezeV11Op::getCanonicalizationPatterns(patterns, context);

  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;

  // Simplify shape-related ops.
  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns))))
    signalPassFailure();
}

void SimplifyShapeRelatedOpsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  // Repeat `shape simplification -> constant propagation -> shape inference` so
  // that all ops' shape are updated.
  for (unsigned i = 0; i < 3; ++i) {
    topDownShapeSimplification(context, moduleOp);
    OpPassManager pm("builtin.module");
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, moduleOp)))
      return signalPassFailure();
  }

  // Clean up added attributes in ConcatOp.
  moduleOp.walk([&](ONNXConcatOp concatOp) {
    Operation *op = concatOp.getOperation();
    if (op->getAttrOfType<::mlir::Attribute>("for_dims"))
      op->removeAttr("for_dims");
  });
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a SimplifyShapeRelatedOps pass.
 */
std::unique_ptr<mlir::Pass> createSimplifyShapeRelatedOpsPass() {
  return std::make_unique<SimplifyShapeRelatedOpsPass>();
}

} // namespace onnx_mlir
