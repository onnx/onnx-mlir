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

// clang-format off
/*

Shape-related operations that are often generated by onnx converters and
complicate our shape inference for unknown dimensions. Hence, it is better to
simplify them as many as possible.

Usually shape-related calculation starts with an ONNXShapeOp to get the shape of
a tensor, e.g.

```mlir
%0 = "onnx.Shape"(%arg0) : (tensor<?x256x1xf32>) -> tensor<3xi64>
%1 = "onnx.Reshape"(%arg1, %0) : (tensor<?x256xf32>,  tensor<3xi64>) -> tensor<?x?x?xf32>
```

The problem is that ONNXShapeOp returns a tensor whose values are unknown at
compile time, so the value `256` in <?x256> cannot be propagated further after
this ONNXShapeOp. Hence, the output shape of Reshape is totally unknown.

The essential idea here to solve the problem is representing ONNXShapeOp by
ONNXDimOp and ONNXConcatOp. By that way, each invidual dimension of a tensor is
exposed and easily propagated through other ops. For example, the above
"onnx.Shape" will rewritten into:

```mlir
%0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256x1xi64>) -> tensor<1xi64>
%1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
%2 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
%3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
%4 = "onnx.Reshape"(%arg0, %3) : (tensor<?x256xf32>,  tensor<3xi64>) -> tensor<?x?x?xf32>
```

Now, it's straighforward to update the output shape of Reshape from
`<?x?x?xf32>` to `<?x256x1xf32>` by looking at the inputs of Concat.

*/
// clang-format on

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "simplify_shape_related_ops"

using namespace mlir;

namespace onnx_mlir {

/// Get all dimensions in I64 (-1 for unknown) that are stored by the value.
void getDimsInt64(Value val, SmallVectorImpl<int64_t> &result) {
  SmallVector<Value, 4> dims;
  getDims(val, dims);
  for (Value v : dims) {
    if (auto constOp = dyn_cast<ONNXConstantOp>(v.getDefiningOp())) {
      auto valueAttr = constOp.valueAttr().cast<DenseElementsAttr>();
      int64_t dim = valueAttr.getSplatValue<int64_t>();
      result.emplace_back(dim);
    } else {
      result.emplace_back(-1);
    }
  }
}

/// Create a ConcatOp to concat the list of tensors.
Value emitConcatOpForDims(MultiDialectBuilder<OnnxBuilder> create,
    ValueRange inputs, Type outputType) {
  int64_t rank = inputs.size();
  if (rank == 1) {
    // Input is tensor<1xf32>, squeeze it if the output type is scalar i.e.
    // tensor<f32>
    if (auto tensorType = outputType.dyn_cast<RankedTensorType>()) {
      if (tensorType.getRank() == 0) {
        Value zero = create.onnx.constantInt64({0});
        return create.onnx.squeeze(outputType, inputs[0], zero);
      }
    }
    return inputs[0];
  }
  Type elementType = getElementType(inputs[0].getType());
  Type resultType = RankedTensorType::get({rank}, elementType);
  Value concatOutput = create.onnx.concat(resultType, inputs, /*axis=*/0);
  return concatOutput;
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

/// Update the output shape using userDims.
/// Return success if the output shape is updated. Otherwise, return failure.
LogicalResult updateOutputType(
    Value output, const SmallVectorImpl<int64_t> &userDims) {
  // Try to update the output shape using userDims.
  Type oldOutputType = output.getType();
  updateType(output, userDims);
  Type newOutputType = output.getType();

  // No new output type is inferred.
  if (newOutputType == oldOutputType)
    return failure();

  // Update the function signature if the output type changed.
  updateFunctionSignature(output.getDefiningOp());
  return success();
}

/// Rewrite onnx.Shape into onnx.Dim and onnx.Concat.
//
/// For example, the following onnx.Shape:
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
    Type outputType = shapeOp.shape().getType();

    // Match: Input is ranked.
    if (!onnx_mlir::isRankedShapedType(data.getType()))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    ArrayRef<int64_t> dims = onnx_mlir::getShape(data.getType());

    // Get start and end values.
    NewONNXShapeOpShapeHelper shapeHelper(shapeOp.getOperation(), {});
    LogicalResult shapeComputed = shapeHelper.computeShape();
    if (failed(shapeComputed)) {
      shapeOp.emitError("Failed to scan " + ONNXShapeOp::getOperationName() +
                        " parameters successfully");
      return failure();
    }
    int64_t start = shapeHelper.start;
    int64_t end = shapeHelper.end;

    SmallVector<Value, 4> dimValues;
    for (unsigned i = start; i < end; ++i) {
      Value dimVal = (ShapedType::isDynamic(dims[i]))
                         ? create.onnx.dim(data, i)
                         : create.onnx.constantInt64({dims[i]});
      dimValues.emplace_back(dimVal);
    }
    Value replacedValue = emitConcatOpForDims(create, dimValues, outputType);

    rewriter.replaceOp(shapeOp, replacedValue);
    return success();
  }
};

/// This pattern rewrites
///   ONNXCastOp(ONNXConcatOp)
/// into
///   ONNXConcatOp(ONNXCastOp, ONNXCastOp, ..., ONNXCastOp)
/// so that Cast is close to Dim or Constant.
class PassThroughCastPattern : public OpRewritePattern<ONNXCastOp> {
public:
  using OpRewritePattern<ONNXCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXCastOp castOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = castOp.getLoc();
    Value input = castOp.input();
    TypeAttr toType = castOp.toAttr();
    Type outputType = castOp.output().getType();

    // Match
    if (!areDimsFromConcat(input))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    SmallVector<Value, 4> dims;
    getDims(input, dims);
    SmallVector<Value> castedDims;
    for (Value d : dims)
      castedDims.emplace_back(create.onnx.cast(d, toType));

    Value replacedValue = emitConcatOpForDims(create, castedDims, outputType);

    rewriter.replaceOp(castOp, replacedValue);
    return success();
  }
};

/// Simplify ONNXGatherOp into ONNXConcatOp.
class PassThroughGatherPattern : public OpRewritePattern<ONNXGatherOp> {
public:
  using OpRewritePattern<ONNXGatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = gatherOp.getLoc();
    Value input = gatherOp.data();
    Value indices = gatherOp.indices();
    int64_t axis = gatherOp.axis();
    Type outputType = gatherOp.output().getType();

    // Match
    // Gather on axis 0.
    if (axis != 0)
      return failure();
    // Input is defined by Concat of dims, so it has rank of 1.
    if (!areDimsFromConcat(input))
      return failure();

    // Indices are constants.
    if (!definedBy<ONNXConstantOp>(indices))
      return failure();
    DenseElementsAttr indicesAttr =
        getDenseElementAttributeFromONNXValue(indices);
    if (!indicesAttr)
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    int64_t inputRank = getRank(input.getType());

    // Compute integer indices.
    SmallVector<int64_t, 4> indicesI64;
    for (auto element : indicesAttr.getValues<IntegerAttr>()) {
      int64_t axis = element.getInt();
      axis = (axis < 0) ? (axis + inputRank) : axis;
      indicesI64.emplace_back(axis);
    }

    // Replace GatherOp by ConcatOp of specific dimensions.
    SmallVector<Value, 4> dims;
    getDims(input, dims);
    SmallVector<Value> castedDims;
    SmallVector<Value> gatherDims;
    for (int64_t i : indicesI64)
      gatherDims.emplace_back(dims[i]);

    Value replacedValue = emitConcatOpForDims(create, gatherDims, outputType);

    rewriter.replaceOp(gatherOp, replacedValue);
    return success();
  }
};

/// Simplify ONNXSliceOp into ONNXConcatOp.
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
    Type outputType = sliceOp.output().getType();

    // Match
    // Input is defined by Concat of dims, so it has rank of 1.
    if (!areDimsFromConcat(input))
      return failure();
    // Starts and ends are constants.
    if (!definedBy<ONNXConstantOp>(starts) ||
        !definedBy<ONNXConstantOp>(ends) || !definedBy<ONNXConstantOp>(steps))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

// Get starts, ends, axes and steps via ShapeHelper.
    NewONNXSliceOpShapeHelper shapeHelper(sliceOp.getOperation(), {});
    if (failed(shapeHelper.computeShape())) {
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
    SmallVector<Value, 4> dims;
    getDims(input, dims);
    SmallVector<Value> castedDims;
    SmallVector<Value> slicedDims;
    for (int64_t i : indices)
      slicedDims.emplace_back(dims[i]);

    Value replacedValue = emitConcatOpForDims(create, slicedDims, outputType);

    rewriter.replaceOp(sliceOp, replacedValue);
    return success();
  }
};

// clang-format off
/// Flatten Concat of N-D tensors in to Concat of 1-D tensors.
/// For example:
/// ```mlir
/// %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
/// %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
/// %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
/// %3 = "onnx.Concat"(%2, %0) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
/// ```
/// The last Concat will be flattened into
///
/// ```
/// %0 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
/// %1 = "onnx.Constant"() {value = dense<256> : tensor<1xi64>} : () -> tensor<1xi64>
/// %2 = "onnx.Concat"(%0, %1, %0) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
/// ```
// clang-format off
class PassThroughConcatPattern : public OpRewritePattern<ONNXConcatOp> {
public:
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Location loc = concatOp.getLoc();
    ValueRange inputs = concatOp.inputs();
    Type outputType = concatOp.concat_result().getType();

    // Match: inputs are dimensions but not all of them are of rank 1.
    if (!llvm::all_of(inputs, [](Value v) { return areDims(v); }))
      return failure();
    if (llvm::all_of(inputs, [](Value v) {
          Type vType = v.getType();
          return (isRankedShapedType(vType) && (getRank(vType) == 1) &&
                  (getShape(vType)[0] == 1));
        }))
      return failure();

    // Rewrite: flatten Concat of N-D tensors in to Concat of 1-D tensors.
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    SmallVector<Value, 4> dims;
    for (Value v : inputs) {
      SmallVector<Value, 4> dimsV;
      getDims(v, dimsV);
      for (Value dim : dimsV)
        dims.emplace_back(dim);
    }
    Value replacedValue = emitConcatOpForDims(create, dims, outputType);
    rewriter.replaceOp(concatOp, replacedValue);
    return success();
  }
};

/// Update Reshape's output shape if possible.
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
    // Shape is defined by Concat of dims.
    if (!areDimsFromConcat(shape))
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
    return updateOutputType(output, userDims);
  }
};

/// Update ConstantOfShape's output shape if possible.
class UpdateConstantOfShapePattern
    : public OpRewritePattern<ONNXConstantOfShapeOp> {
public:
  using OpRewritePattern<ONNXConstantOfShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConstantOfShapeOp cosOp, PatternRewriter &rewriter) const override {
    // Get basic op info.
    Value input = cosOp.input();
    Value output = cosOp.output();

    // Match: Input is defined by Concat of dims.
    if (!areDimsFromConcat(input))
      return failure();
    // Rewrite only if the given shape is different from the output shape.
    SmallVector<int64_t, 4> userDims;
    getDimsInt64(input, userDims);
    ArrayRef<int64_t> outputDims = getShape(output.getType());
    if (userDims == outputDims)
      return failure();

    // Rewrite
    return updateOutputType(output, userDims);
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
  patterns.insert<onnx_mlir::PassThroughGatherPattern>(context);
  patterns.insert<onnx_mlir::PassThroughSlicePattern>(context);

  // Update the output shape of the followring ops using inferred dimensions.
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

  // Repeat `shape simplification -> constant propagation -> shape inference`
  // so that all ops' shape are updated.
  for (unsigned i = 0; i < 3; ++i) {
    topDownShapeSimplification(context, moduleOp);
    OpPassManager pm("builtin.module");
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, moduleOp)))
      return signalPassFailure();
  }
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
