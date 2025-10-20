/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
// TODO: This file is quite busy as the number of decomposing op is increasing.
// It is better to move decomposition of each operation into a separate file.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <numeric>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/Decompose.hpp"
#include "src/Dialect/ONNX/Transforms/DecomposeEinsum.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose"

using namespace mlir;

namespace onnx_mlir {

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  assert(origAttrs && "handle EXISTING ArrayAttr only");

  if (mlir::dyn_cast<FloatAttr>(origAttrs.getValue()[0])) {
    Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] =
          mlir::cast<FloatAttr>(origAttrs.getValue()[i]).getValueAsDouble();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::ArrayRef(wrapper));
  }

  if (mlir::dyn_cast<IntegerAttr>(origAttrs.getValue()[0])) {
    Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] = mlir::cast<IntegerAttr>(origAttrs.getValue()[i]).getInt();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::ArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

/// Create an Scalar DenseElementsAttr from FloatAttr or IntegerAttr.
/// This is used to create an ONNXConstant of rank 0, e.g. tensor<f32>.
DenseElementsAttr createScalarDenseAttr(
    PatternRewriter &rewriter, Attribute attr) {
  if (mlir::dyn_cast<FloatAttr>(attr)) {
    Type elementType = rewriter.getF32Type();
    SmallVector<float, 1> wrapper;
    wrapper.emplace_back(mlir::cast<FloatAttr>(attr).getValueAsDouble());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::ArrayRef(wrapper));
  }

  if (mlir::dyn_cast<IntegerAttr>(attr)) {
    Type elementType = rewriter.getIntegerType(64);
    SmallVector<int64_t, 1> wrapper;
    wrapper.emplace_back(mlir::cast<IntegerAttr>(attr).getSInt());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::ArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

// Create an DenseElementsAttr of ArrayAttr.
// When ArrayAttr is Null, an empty Integer DenseElementAttr is returned
DenseElementsAttr createDenseArrayAttrOrEmpty(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  if (origAttrs)
    return createDenseArrayAttr(rewriter, origAttrs);

  Type elementType = rewriter.getIntegerType(64);
  int nElements = 0;
  SmallVector<int64_t, 4> wrapper(nElements, 0);
  for (int i = 0; i < nElements; ++i)
    wrapper[i] = i;

  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::ArrayRef(wrapper));
}

Value createSequenceConstructOp(
    PatternRewriter &rewriter, Value seq, OperandRange inputs) {
  Type resType = seq.getType();
  Location loc = seq.getLoc();
  Value position = rewriter.create<ONNXNoneOp>(loc);

  for (auto input : inputs)
    seq = rewriter.create<ONNXSequenceInsertOp>(
        loc, resType, seq, input, position);

  return seq;
}

// Reverse all elements of the first or second dimension of `input`.
Value reverseAllElements(
    PatternRewriter &rewriter, Location loc, Value input, int64_t dimension) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t, 4> sLens;
  assert((dimension == 0 or dimension == 1) &&
         "Reversed dimension need to be 0 or 1.");
  // Create `sequence_lengths`, `batch_axis` and `time_axis` to reverse all
  // elements. When reversing the first dim of input(d0 x d1), set `batch_axis`
  // = 1, and `time_axis` = 0 and create [d0, d0,...,d0] as `sequence_lengths`
  // whose the number of elements are d1.
  // Example:
  //   input(d0 x d1) = (4 x 3)) then, `sequence_lengths` is [4, 4, 4].
  // When reverse the second dim of input(d0 x d1), set `batch_axis` = 0,
  // and `time_axis` = 1 and create [d1, d1,...,d1] as `sequence_lengths`
  // whose the number of elements are d0.
  // Example:
  // input(d0 x d1) = (4 x 3)) then, `sequence_lengths` is [3, 3, 3, 3].
  int64_t batchAxis = dimension == 0 ? 1 : 0;
  int64_t timeAxis = dimension == 0 ? 0 : 1;
  for (int i = 0; i < inputShape[batchAxis]; ++i)
    sLens.emplace_back(inputShape[timeAxis]);
  Value sLensVal = create.onnx.constantInt64(sLens);
  Type resultType = mlir::cast<RankedTensorType>(input.getType());
  Value result = create.onnx.reverseSequence(
      resultType, input, sLensVal, batchAxis, timeAxis);
  return result;
}

// Reverse elements in weight tensor of ConvTranspose op. The reversed weight
// tensor are used as weight tensor of Conv op generated by rewriting.
// 1. Transpose weight tensor from NxCxD0xD1xD2x... to D0xD1xD2x ... xNxC to
//    reverse elements by using ReverseSequence op.
//    The ReverseSequence op can reverse elements in the first and second
//    dimensions. So, spatial dimensions are moved using Transpose op.
// 2. Reverse The first two dimensions by two ReverseSequence ops.
//    Reverse D0 by the first ReverseSequence op, then reverse D1 by the second
//    ReverseSequence op. Reverse D0 and D1 and move them to last
//    (D0xD1xD2xD3x... to D2xD3x...xD0xD1) to reverse D2 and D3. Continue this
//    to reverse all spatial dimensions.
// 3. Reverse the last spatial dimension (Dn) using single ReverseSequence if
//    rank is odd.
// 4. Reverse non-spatial dimensions (N and C).
//    Transpose "N x C x D0 x D1 x D2 x ... x Dn" to "C x N x D0 x D1 x D2 x
//    ...x Dn".
Value reverseWeightTensor(
    PatternRewriter &rewriter, Location loc, Value input) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  assert(inputType.hasRank() && "Need rank to reverse weight tensor.");
  // 1. Transpose NxCxD0xD1xD2x... to D0xD1xD2x ... xNxC.
  int64_t spatialOffset = 2; // for N and C
  int64_t spatialRank = inputType.getRank() - spatialOffset;
  SmallVector<int64_t, 4> permsVal;
  for (int i = 0; i < spatialRank; ++i)
    permsVal.emplace_back(spatialOffset + i);
  for (int i = 0; i < spatialOffset; ++i)
    permsVal.emplace_back(i);
  ArrayRef<int64_t> perms(permsVal);
  Value transposedInput = create.onnx.transposeInt64(input, perms);
  // 2. Reverse the first and second spatial dimensions.
  ShapedType tInputType = mlir::cast<ShapedType>(transposedInput.getType());
  for (int i = 0; i < spatialRank / 2; i += 2) {
    // TODO: Support dynamic dim in reverseAllElements().
    assert((!tInputType.isDynamicDim(0) && !tInputType.isDynamicDim(1)) &&
           "Spatial dimensions for weight tensor need to be static.");
    Value reverse0 =
        reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
    Value reverse1 =
        reverseAllElements(rewriter, loc, reverse0, /*dimension*/ 1);
    // Move two reversed dimensions to the last for next reverse.
    SmallVector<int64_t, 4> permsVal0;
    for (int j = 0; j < inputType.getRank() - 2; ++j)
      permsVal0.emplace_back(j + 2);
    for (int j = 0; j < 2; ++j)
      permsVal0.emplace_back(j);
    ArrayRef<int64_t> perms(permsVal0);
    transposedInput = create.onnx.transposeInt64(reverse1, permsVal0);
  }
  // 3. Reverse the rest of dimension if spatial rank is odd.
  if (spatialRank % 2 != 0) {
    ShapedType tInType = mlir::cast<ShapedType>(transposedInput.getType());
    ArrayRef<int64_t> tInShape = tInType.getShape();
    Value reverse0;
    if (tInShape[1] == ShapedType::kDynamic) {
      // When N is unknown dim,
      // reshape "Dn x N x C x D0 x D1 x D2 x ... x Dn-1"
      // to "Dn x 1 x N x C x D0 x D1 x D2 x ... x Dn-1",
      // then, reshape back to original shape after reversed.
      // TODO: Support dynamic dim in reverseAllElements(). If supported, this
      // code becomes much simpler.
      int64_t tInRank = tInShape.size();
      Type tInShapeType =
          RankedTensorType::get({tInRank}, rewriter.getI64Type());
      Value tInShapeVals = create.onnx.shape(tInShapeType, transposedInput);
      SmallVector<int64_t, 6> reshapedShapeVec;
      reshapedShapeVec.emplace_back(tInShape[0]);
      reshapedShapeVec.emplace_back(1);
      for (int i = 1; i < tInType.getRank(); ++i)
        reshapedShapeVec.emplace_back(tInShape[i]);
      Type reshapedType = RankedTensorType::get(reshapedShapeVec, elementType);
      Type firstShapeType = RankedTensorType::get({1}, rewriter.getI64Type());
      Type otherShapeType =
          RankedTensorType::get({tInRank - 1}, rewriter.getI64Type());
      Value oneVal = create.onnx.constantInt64(ArrayRef<int64_t>({1}));
      Value firstShapeVal = create.onnx.slice(
          firstShapeType, tInShapeVals, /* starts */ 0, /* ends */ 1);
      Value otherShapeVals = create.onnx.slice(
          otherShapeType, tInShapeVals, /* starts */ 1, /* ends */ tInRank);
      Type reshapeShapeType =
          RankedTensorType::get({tInRank + 1}, rewriter.getI64Type());
      Value shape = create.onnx.concat(reshapeShapeType,
          ValueRange{firstShapeVal, oneVal, otherShapeVals}, 0);
      transposedInput =
          create.onnx.reshape(reshapedType, transposedInput, shape);
      reverse0 =
          reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
      reverse0 = create.onnx.reshape(tInType, reverse0, tInShapeVals);
    } else {
      reverse0 =
          reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
    }

    // Move reversed one dimension to the last.
    SmallVector<int64_t, 4> permsVal1;
    for (int j = 0; j < inputType.getRank() - 1; ++j)
      permsVal1.emplace_back(j + 1);
    permsVal1.emplace_back(0);
    ArrayRef<int64_t> perms(permsVal1);
    transposedInput = create.onnx.transposeInt64(reverse0, permsVal1);
  }
  // 4. Reverse non-spatial dimensions.
  SmallVector<int64_t, 4> permsVal2;
  for (int i = 0; i < spatialOffset; ++i)
    permsVal2.emplace_back(spatialOffset - 1 - i);
  for (int i = 0; i < spatialRank; ++i)
    permsVal2.emplace_back(spatialOffset + i);
  ArrayRef<int64_t> perms2(permsVal2);
  Value result = create.onnx.transposeInt64(transposedInput, perms2);
  return result;
}

// The convOutputs are adjusted to add an extra dimension at the innermost
// level. The outputs of conv1 and conv3 are concatenated at this innermost
// level, resulting in concat1_output. Similarly, the outputs of conv4 and conv2
// are concatenated at the innermost level, creating concat2_output. These
// concatenated outputs are then reshaped to modify the two innermost levels,
// ensuring the second innermost level is set to 1.
// Finally, a concatenation is performed on the two reshaped outputs at the
// second innermost level, after which the result is reshaped back to match the
// original convtranspose output dimensions.

Value getFinalOutputFromFourConvOutput(PatternRewriter &rewriter, Location loc,
    ONNXConvOp convOp, Value conv1Output, Value conv2Output, Value conv3Output,
    Value conv4Output) {

  auto int64Type = mlir::IntegerType::get(rewriter.getContext(), 64);

  ONNXConvOpShapeHelper convShapeHelper(convOp.getOperation(), {});
  Type elementType = getElementType(conv1Output.getType());
  (void)convShapeHelper.computeShapeAndUpdateType(elementType);
  int outputRank = convShapeHelper.getOutputDims().size();
  SmallVector<int64_t, 4> convOutputShape;
  for (int i = 0; i < outputRank; ++i) {
    int64_t d = convShapeHelper.getOutputDims()[i].isLiteral()
                    ? convShapeHelper.getOutputDims()[i].getLiteral()
                    : ShapedType::kDynamic;
    convOutputShape.emplace_back(d);
  }

  auto getOnnxConstOpForReshape = [&](SmallVector<int64_t, 4> outputShape) {
    SmallVector<mlir::Attribute, 4> elements;
    for (auto val : outputShape) {
      elements.push_back(mlir::IntegerAttr::get(int64Type, val));
    }
    auto constTypeForReshape =
        RankedTensorType::get(outputShape.size(), int64Type);

    return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(),
        DenseElementsAttr::get(constTypeForReshape, llvm::ArrayRef(elements)));
  };

  // The four convOutputs are adjusted to add an extra dimension at the
  // innermost level.
  SmallVector<int64_t, 4> outputShapePlusOneDim(convOutputShape);
  outputShapePlusOneDim.push_back(1);
  auto onnxConstForReshapeAddOneDim =
      getOnnxConstOpForReshape(outputShapePlusOneDim);

  auto reshapeOutputType =
      RankedTensorType::get(outputShapePlusOneDim, elementType);

  auto reshapeOutputAddOneDimConv1 = rewriter.create<ONNXReshapeOp>(
      loc, reshapeOutputType, conv1Output, onnxConstForReshapeAddOneDim);
  auto reshapeOutputAddOneDimConv2 = rewriter.create<ONNXReshapeOp>(
      loc, reshapeOutputType, conv2Output, onnxConstForReshapeAddOneDim);
  auto reshapeOutputAddOneDimConv3 = rewriter.create<ONNXReshapeOp>(
      loc, reshapeOutputType, conv3Output, onnxConstForReshapeAddOneDim);
  auto reshapeOutputAddOneDimConv4 = rewriter.create<ONNXReshapeOp>(
      loc, reshapeOutputType, conv4Output, onnxConstForReshapeAddOneDim);

  SmallVector<int64_t, 4> outputShapeFirstConcat(outputShapePlusOneDim);
  outputShapeFirstConcat[outputShapeFirstConcat.size() - 1] = 2;
  auto firstConcatOutputType =
      RankedTensorType::get(outputShapeFirstConcat, elementType);

  // Below concats result will have the innermost dim as 2.
  auto firstConcat = rewriter.create<ONNXConcatOp>(loc, firstConcatOutputType,
      ValueRange{reshapeOutputAddOneDimConv1, reshapeOutputAddOneDimConv3}, -1);
  auto secondConcat = rewriter.create<ONNXConcatOp>(loc, firstConcatOutputType,
      ValueRange{reshapeOutputAddOneDimConv4, reshapeOutputAddOneDimConv2}, -1);

  // Reshaping to modify the two innermost levels,ensuring the second innermost
  // level is set to 1
  SmallVector<int64_t, 4> outputShapeForDimAdjust(convOutputShape);
  auto dimValueAtLastIndex = convOutputShape[convOutputShape.size() - 1] * 2;
  outputShapeForDimAdjust[outputShapeForDimAdjust.size() - 1] = 1;
  outputShapeForDimAdjust.push_back(dimValueAtLastIndex);

  auto onnxConstForReshapeDimAdjust =
      getOnnxConstOpForReshape(outputShapeForDimAdjust);

  auto reshapeOutputForDimAdjustType =
      RankedTensorType::get(outputShapeForDimAdjust, elementType);

  auto reshapeOutputDimAdjustOfFirstConcat = rewriter.create<ONNXReshapeOp>(loc,
      reshapeOutputForDimAdjustType, firstConcat, onnxConstForReshapeDimAdjust);
  auto reshapeOutputDimAdjustOfSecondConcat =
      rewriter.create<ONNXReshapeOp>(loc, reshapeOutputForDimAdjustType,
          secondConcat, onnxConstForReshapeDimAdjust);

  SmallVector<int64_t, 4> outputShapeForFinalConcat(outputShapeForDimAdjust);
  outputShapeForFinalConcat[outputShapeForFinalConcat.size() - 2] = 2;

  auto finalConcatOutputType =
      RankedTensorType::get(outputShapeForFinalConcat, elementType);

  // Final Concat is performed on the two reshaped outputs at the
  // second innermost level
  auto finalConcat = rewriter.create<ONNXConcatOp>(loc, finalConcatOutputType,
      ValueRange{reshapeOutputDimAdjustOfFirstConcat,
          reshapeOutputDimAdjustOfSecondConcat},
      -2);
  SmallVector<int64_t, 4> outputShapeForResult(convOutputShape);
  dimValueAtLastIndex = convOutputShape[convOutputShape.size() - 1] * 2;
  auto dimValueAtSecondLastIndex =
      convOutputShape[convOutputShape.size() - 2] * 2;
  outputShapeForResult[outputShapeForResult.size() - 2] =
      dimValueAtSecondLastIndex;
  outputShapeForResult[outputShapeForResult.size() - 1] = dimValueAtLastIndex;

  auto onnxConstForLastReshape = getOnnxConstOpForReshape(outputShapeForResult);

  auto finalOutputType =
      RankedTensorType::get(outputShapeForResult, elementType);
  // Result is reshaped back to match the original convtranspose output
  // dimensions
  auto finalOutput = rewriter.create<ONNXReshapeOp>(
      loc, finalOutputType, finalConcat, onnxConstForLastReshape);
  return finalOutput;
}
Value sliceOfWeightTensorForPhase(
    PatternRewriter &rewriter, Location loc, Value input, int phase) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
  assert(inputType.hasRank() && "Need rank to reverse weight tensor.");
  auto shape = inputType.getShape();
  MLIRContext *context = rewriter.getContext();

  auto int64Type = mlir::IntegerType::get(context, 64);
  auto getONNXConstOpForSlice =
      [&](SmallVector<int64_t> values) -> ONNXConstantOp {
    SmallVector<mlir::Attribute, 4> elements;
    for (auto val : values) {
      elements.push_back(mlir::IntegerAttr::get(int64Type, val));
    }
    auto constType = RankedTensorType::get(values.size(), int64Type);
    return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(),
        DenseElementsAttr::get(constType, llvm::ArrayRef(elements)));
  };

  ONNXConstantOp startOnnxConst;
  llvm::SmallVector<int64_t> startVector;
  switch (phase) {
  case 1:
    startVector = {0, 0, 1, 1};
    break;
  case 2:
    startVector = {0, 0, 0, 0};
    break;
  case 3:
    startVector = {0, 0, 1, 0};
    break;
  case 4:
    startVector = {0, 0, 0, 1};
    break;
  }
  startOnnxConst = getONNXConstOpForSlice(startVector);
  llvm::SmallVector<int64_t> newShape = {
      shape[0], shape[1], shape[2] / 2, shape[3] / 2};

  auto endOnnxConst = getONNXConstOpForSlice(SmallVector<int64_t, 4>(shape));
  llvm::SmallVector<int64_t> stepVector = {1, 1, 2, 2};
  auto stepOnnxConst = getONNXConstOpForSlice(stepVector);
  llvm::SmallVector<int64_t> axisVector = {0, 1, 2, 3};
  auto axisOnnxConst = getONNXConstOpForSlice(axisVector);
  auto newOuputShapedType = inputType.get(newShape, inputType.getElementType());
  auto sliceOp = create.onnx.slice(newOuputShapedType, input, startOnnxConst,
      endOnnxConst, axisOnnxConst, stepOnnxConst);
  return sliceOp;
}
Value ph1WeightTensor(PatternRewriter &rewriter, Location loc, Value input) {
  return sliceOfWeightTensorForPhase(rewriter, loc, input, 1);
}
Value ph2WeightTensor(PatternRewriter &rewriter, Location loc, Value input) {
  return sliceOfWeightTensorForPhase(rewriter, loc, input, 2);
}
Value ph3WeightTensor(PatternRewriter &rewriter, Location loc, Value input) {
  return sliceOfWeightTensorForPhase(rewriter, loc, input, 3);
}
Value ph4WeightTensor(PatternRewriter &rewriter, Location loc, Value input) {
  return sliceOfWeightTensorForPhase(rewriter, loc, input, 4);
}
ArrayAttr getAttrForPhaseConv(
    PatternRewriter &rewriter, Location loc, ArrayAttr valAttr) {
  assert(mlir::dyn_cast<IntegerAttr>(valAttr.getValue()[0]) &&
         "Attribute must be integer");
  int nElements = valAttr.getValue().size();
  SmallVector<int64_t, 4> wrapper(nElements, 0);
  for (int i = 0; i < nElements; ++i)
    wrapper[i] = mlir::cast<IntegerAttr>(valAttr.getValue()[i]).getInt() / 2;
  return rewriter.getI64ArrayAttr(wrapper);
}
// Calculate padding size used in Conv op from pads for ConvTranspose op.
ArrayAttr getPadsConvTranspose(
    PatternRewriter &rewriter, Location loc, ONNXConvTransposeOp op) {
  // Calculate pads for generated Conv op.
  // new_pads = kernel -  pads - 1
  // Reference: Dumoulin, Vincent, and Francesco Visin. "A guide to convolution
  // arithmetic for deep learning." arXiv preprint arXiv:1603.07285 (2016).
  ONNXConvTransposeOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
  SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
  DimsExpr pads = shapeHelper.pads;
  assert(IndexExpr::isLiteral(kernelShape) && IndexExpr::isLiteral(pads) &&
         "Currently only static dims are supported in spatial dims.");

  SmallVector<int64_t, 4> newPads;
  SmallVector<int64_t, 2> newKernel;
  // If `dilations` is not default [1, 1], `kernel` is updated by inserting
  // spaces in kernel elements.
  //   ex. kernel [2, 3] and dilation [2, 2], then new `kernel` is [3, 4]
  for (unsigned int i = 0; i < kernelShape.size(); ++i)
    newKernel.emplace_back(
        kernelShape[i].getLiteral() +
        (kernelShape[i].getLiteral() - 1) * (dilations[i] - 1));
  // Calculate new pads. `kernel` size is doubled for the calculation.
  for (unsigned int i = 0; i < kernelShape.size() * 2; ++i)
    newPads.emplace_back(
        newKernel[i % kernelShape.size()] - pads[i].getLiteral() - 1);
  return rewriter.getI64ArrayAttr(newPads);
}

// Check if strides is unit strides.
bool hasUnitStrides(ArrayAttr strides) {
  // Default is unit strides
  if (strides == nullptr)
    return true;
  SmallVector<int64_t, 3> vStrides;
  for (unsigned int i = 0; i < ArrayAttrSize(strides); ++i)
    vStrides.emplace_back(ArrayAttrIntVal(strides, i));
  return llvm::all_of(vStrides, [](int64_t s) { return s == 1; });
}

// Check if v's shape N x C x D1 x D2 ... x Dn has static dims D1 ... Dn.
bool hasStaticSpatialDims(Value v) {
  ShapedType type = mlir::cast<ShapedType>(v.getType());
  if (!type.hasRank())
    return false;
  // Shape has the form N x C x D1 x D2 ... x Dn.
  ArrayRef<int64_t> NxCxDs = type.getShape();
  // Remove leading batch size N and channels C dims,
  // so we're left with D1 x D2 ... x Dn.
  ArrayRef<int64_t> Ds = NxCxDs.drop_front(2);
  // These must all be static for decomposition to work.
  return llvm::none_of(Ds, ShapedType::isDynamic);
}

// In the following pattern, a SequenceAt can be replaced with Split
//   %seq = onnx.SplitToSequence(%input, %split) {%axis : }
//   %res = onnx.SequenceAt(%seq, %position)
// We just try to avoid using the sequence related ops, which are less
// optimized, or even not implemented in onnx-mlir.
// In the targeted use case, %split and %position are constant scalar and the
// tensor of %input and %res have static shape.
// This condition greatly reduces the complexity of code generation to replace
// SequenceAt with split op
//   %res = onnx.Split(%input, onnx.expand(%split, %input.shape()[%axis]))
//   {%axis : } : %position
// onnx.expand(%split, %input.shape()[%axis]) can be a constant under the
// assumed condition.
// Here %position has to be compiler time constant.
// For multiple SequenceAt from the same SplitToSequence result, the onnx.split
// for different SequenceAt are expected to be merged by optimization.
// Alternatively, Slice can be used
//   %res = onnx.Slice(%input, %start, %end, %step)
// The start, and end for slice will be onnx.constant:
//   start: %position*%split for %axis, 0 for other dimensionis
//   end: (%positiion+1)*%split for %axis, upper bound for other dimension
//   step: 1 for all dimensions
// The split approach may have better performance than the alternative slice
// approach,  because the slicing is done separately.

bool canSequenceAtBeReplaced(Value sequenceAtResult) {
  if (!hasStaticShape(sequenceAtResult.getType()))
    return false;

  ONNXSequenceAtOp op = sequenceAtResult.getDefiningOp<ONNXSequenceAtOp>();

  Value inputSequence = op.getInputSequence();
  Value position = op.getPosition();

  if (!isDenseONNXConstant(position))
    return false;

  // Input sequence should be defined with SplitToSequence
  ONNXSplitToSequenceOp splitToSequence =
      inputSequence.getDefiningOp<ONNXSplitToSequenceOp>();
  if (!splitToSequence)
    return false;

  // Check the pattern of the SplitToSequence op
  Value input = splitToSequence.getInput();
  if (!hasStaticShape(input.getType()))
    return false;
  Value split = splitToSequence.getSplit();
  if (!isScalarConstantTensor(split))
    return false;

  return true;
}

Attribute upgradeGridSampleV16Mode(PatternRewriter &rewriter, Attribute mode) {
  const auto stringMode = mlir::cast<StringAttr>(mode);
  if (stringMode.strref() == "bilinear") {
    return rewriter.getStringAttr("linear");
  }
  if (stringMode.strref() == "bicubic") {
    return rewriter.getStringAttr("cubic");
  }
  assert(stringMode.strref() == "nearest");
  return mode;
}

Value replaceSequenceAt(
    PatternRewriter &rewriter, Location loc, Value sequenceAtResult) {
  ONNXSequenceAtOp op = sequenceAtResult.getDefiningOp<ONNXSequenceAtOp>();

  Value inputSequence = op.getInputSequence();
  Value position = op.getPosition();

  ONNXConstantOp positionConstant =
      mlir::cast<ONNXConstantOp>(position.getDefiningOp());
  int64_t positionInt = getScalarValue<int64_t>(positionConstant);

  ONNXSplitToSequenceOp splitToSequence =
      mlir::cast<ONNXSplitToSequenceOp>(inputSequence.getDefiningOp());

  Value input = splitToSequence.getInput();
  Value split = splitToSequence.getSplit();

  ONNXConstantOp splitConstant =
      mlir::cast<ONNXConstantOp>(split.getDefiningOp());
  int64_t splitInt = getScalarValue<int64_t>(splitConstant);
  int64_t axisInt = splitToSequence.getAxis();

  auto shape = getShape(input.getType());

  OnnxBuilder create(rewriter, loc);

  Type sequenceElementType =
      mlir::cast<SeqType>(inputSequence.getType()).getElementType();
  mlir::SmallVector<mlir::Type, 4> outputTypes(
      shape[axisInt] / splitInt, sequenceElementType);
  auto numSplit = create.constantInt64(
      mlir::SmallVector<int64_t, 4>(shape[axisInt] / splitInt, splitInt));
  auto resultRange = create.split(outputTypes, input, numSplit, axisInt);
  auto rawResult = resultRange[positionInt];

  if (rawResult.getType() == sequenceAtResult.getType())
    return rawResult;

  // Temporary code for the error in the model generated by torch.onnx.export
  // The the dim of the reuslt of SequenceAt op is different from the element
  // type of the sequence..
  // My assumption is that the exporter is confused with  squeeze and unsqueeze
  // followed by the SequenceAt. There are two cases in the model:
  // clang-format off
  // Case #1:
  //   %16 = "onnx.SequenceAt"(%14, %15) {onnx_node_name = "n0"} :
  //     (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x100xf32>
  //     %23 = "onnx.Unsqueeze"(%16, %22) {onnx_node_name = "n2"} :
  //     (tensor<1x100xf32>, tensor<i64>) -> tensor<1x1x100xf32>
  // Case#2:
  //   %67 = "onnx.SequenceAt"(%66, %15) {onnx_node_name = "n0"} :
  //   (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x1x100xf32>
  //   %71 = "onnx.Sigmoid"(%67) {onnx_node_name = "node_Sigmoid_60"} :
  //   (tensor<1x1x100xf32>) -> tensor<1x1x100xf32>
  // clang-format on
  // Thus, the compiler squeeze the tensor if needed.
  return create.squeeze(
      sequenceAtResult.getType(), rawResult, create.constantInt64(axisInt));
}

bool shouldDecomposeConvTransposeOp(Value convTransposeResult) {
  ONNXConvTransposeOp op =
      mlir::cast<ONNXConvTransposeOp>(convTransposeResult.getDefiningOp());
  return hasShapeAndRank(convTransposeResult) &&
         hasStaticSpatialDims(op.getX()) && hasStaticSpatialDims(op.getW());
}

SmallVector<int64_t> getIntVectorFromArrayAttr(ArrayAttr arrayAttr) {
  assert(mlir::dyn_cast<IntegerAttr>(arrayAttr.getValue()[0]) &&
         "Attribute must be integer");
  SmallVector<int64_t> elements;
  llvm::transform(arrayAttr.getValue().vec(), std::back_inserter(elements),
      [](auto attr) { return cast<IntegerAttr>(attr).getInt(); });
  return elements;
}

bool hasDefaultDilation(ArrayAttr dilation) {
  if (dilation == nullptr)
    return true;
  SmallVector<int64_t, 3> vDilation = getIntVectorFromArrayAttr(dilation);
  return llvm::all_of(vDilation, [](int64_t d) { return d == 1; });
}

// Check if the result of ConvTranspose is not single use, OR if single use
// not used by leakyRelu Or Relu.
bool hasNoActivationConsumer(Value convTransposeResult) {
  auto result = convTransposeResult.getDefiningOp<ONNXConvTransposeOp>().getY();
  if (result.hasOneUse()) {
    Operation *userAtDepth1FromConvt = *(result.getUsers().begin());
    // Check for ConvTranspose->Quant->Dequant->Activation chain.
    if (mlir::isa<ONNXQuantizeLinearOp>(userAtDepth1FromConvt)) {
      Operation *userAtDepth2FromConvt =
          *(userAtDepth1FromConvt->getResult(0).getUsers().begin());
      if (mlir::isa<ONNXDequantizeLinearOp>(userAtDepth2FromConvt)) {
        Operation *userAtDepth3FromConvt =
            *(userAtDepth2FromConvt->getResult(0).getUsers().begin());
        return !mlir::isa<ONNXReluOp, ONNXLeakyReluOp>(userAtDepth3FromConvt);
      }
      // If no Dequant node exists under Quant node, consider no activation.
      return true;
    }
    // If No Quantize node exists under Convt, check for activation directly.
    return !mlir::isa<ONNXReluOp, ONNXLeakyReluOp>(userAtDepth1FromConvt);
  }
  return true;
}

inline bool isTwoFourOrFive(int n) { return n == 2 || n == 4 || n == 5; }

// This decomposition currently do not support all possible convtranspose
// operations. Below are the supported usecases.
// 1) stride[2], pads [1,1], kernel 4,  where convtranspose will decompose to
// two conv operation. with each conv having kernel divided by 2. 2) stride [4],
// pads [2,2], kernel 8,  where it will decompose into 4 conv operations
//  each conv with one fourth of the convtranspose kernel size.
// 3) stride [5], pads [2,2], kernel 10 where it will decompose into 5 conv
// operations each conv with one fifth of the convtranspose kernel size.

bool ShouldDecomposeConvTransposeOp1dToPhasedConvs(Value convTransposeResult,
    ArrayAttr kernelShapeAttr, ArrayAttr padsShapeAttr,
    ArrayAttr stridesShapeAttr, ArrayAttr outputShapeAttr) {
  ONNXConvTransposeOp op =
      mlir::cast<ONNXConvTransposeOp>(convTransposeResult.getDefiningOp());
  bool areSpatialDimsStatic = hasShapeAndRank(convTransposeResult) &&
                              hasStaticSpatialDims(op.getX()) &&
                              hasStaticSpatialDims(op.getW());
  if (!areSpatialDimsStatic)
    return false;
  // kernel shape is required for decomposition, Not supporting the case where
  // kernel shape can be inferred. Not supporting the case where pad values are
  // to be inferred automatically from outputShape.
  if (!kernelShapeAttr || outputShapeAttr) {
    return false;
  }

  auto kernelShape = getIntVectorFromArrayAttr(kernelShapeAttr);
  auto padsShape = (padsShapeAttr) ? getIntVectorFromArrayAttr(padsShapeAttr)
                                   : SmallVector<int64_t>({0, 0});
  auto stridesShape = (stridesShapeAttr)
                          ? getIntVectorFromArrayAttr(stridesShapeAttr)
                          : SmallVector<int64_t>({1});

  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(convTransposeResult.getType());
  auto outputShape = outputType.getShape();
  // Checking to ensure only convtranspose with 1D spatial dims and stride 2, 4
  // or 5 are supported.
  if ((outputShape.size() != 3) || (stridesShape.size() != 1) ||
      (padsShape.size() != 2) || !isTwoFourOrFive(stridesShape[0]))
    return false;
  // number of conv phases equals to the stride.
  int numberOfPhases = stridesShape[0];
  // If numberOfPhases is not 5, the ofm spatial dim should be evenly divisible
  // by num of phases.
  if ((numberOfPhases != 5) &&
      !(outputShape[outputShape.size() - 1] % numberOfPhases == 0)) {
    return false;
  }
  // The decomposistion creates the ofm of below dim, hence checking if the
  // original convtranspose ofm dim is matching to the decomposition ofm.
  // this is specific for convtranspose with stride 5.
  auto convSpatialDim =
      std::floor(outputShape[outputShape.size() - 1] / stridesShape[0]);
  auto combinedSpatialDim = convSpatialDim * 5 + 1;
  if ((numberOfPhases == 5) &&
      (outputShape[outputShape.size() - 1] != combinedSpatialDim)) {
    return false;
  }
  auto checkPadOfValue = [](llvm::ArrayRef<int64_t> pads,
                             int checkValue) -> bool {
    return pads[0] == checkValue && llvm::all_equal(pads);
  };

  // Currently support only below scenarios.
  // 1. stride=2, pads=[1,1], kernel is 4.
  // 2. stride=4, pads=[2,2], kernel is 8.
  // 3. stride=5, pads=[2,2], kernel is 10.
  return (numberOfPhases == 2 && checkPadOfValue(padsShape, 1) &&
             kernelShape[0] == 4) ||
         (numberOfPhases == 4 && checkPadOfValue(padsShape, 2) &&
             kernelShape[0] == 8) ||
         (numberOfPhases == 5 && checkPadOfValue(padsShape, 2) &&
             kernelShape[0] == 10);
}
// This decomposition currently do not support all possible convtranspose
// operations. Below are the supported usecases.
// 1) stride[1,1] where convtranspose will decompose to one conv operation.
// 2) stride [2,2], kernel [6,6], pads [2,2,2,2] where it will decompose into 4
// conv operations each conv with [3,3] phase kernel.
// 3) stride [2,2], kernel [4,4], pads [1,1,1,1] where it will decompose into 4
// conv operations each conv with [2,2] phase kernel.
// 4) stride [2,2], kernel [3,3], pads [0,0,1,1] OR [1,1,0,0] where it will
// decompose into 4 conv operations. In this case, the original weights are
// padded at bottom right, to make it as [4,4] kernel and the four phased-conv
// operations will use [2,2] kernel.
// 5) stride [3,3], and kernel [3,3] pads [0,0,0,0] where it will decompose into
// 9 conv operations each phased conv will use [1,1] kernel
// 6) stride [2,2] and kernel [2,2] pads [0,0,0,0] where it will decompose into
// 4 conv operations each phased conv will use [1,1] kernel
bool ShouldDecomposeConvTransposeOpToPhasedConvs(Value convTransposeResult,
    ArrayAttr kernelShapeAttr, ArrayAttr padsShapeAttr,
    ArrayAttr stridesShapeAttr, ArrayAttr outputShapeAttr) {

  ONNXConvTransposeOp op =
      mlir::cast<ONNXConvTransposeOp>(convTransposeResult.getDefiningOp());
  bool areSpatialDimsStatic = hasShapeAndRank(convTransposeResult) &&
                              hasStaticSpatialDims(op.getX()) &&
                              hasStaticSpatialDims(op.getW());
  if (!areSpatialDimsStatic)
    return false;
  // kernel shape is required for decomposition, Not supporting the case where
  // kernel shape can be inferred. Not supporting the case where pad values are
  // to be inferred automatically from outputShape.
  if (!kernelShapeAttr || outputShapeAttr) {
    return false;
  }

  auto kernelShape = getIntVectorFromArrayAttr(kernelShapeAttr);
  auto padsShape = (padsShapeAttr) ? getIntVectorFromArrayAttr(padsShapeAttr)
                                   : SmallVector<int64_t>({0, 0, 0, 0});
  auto stridesShape = (stridesShapeAttr)
                          ? getIntVectorFromArrayAttr(stridesShapeAttr)
                          : SmallVector<int64_t>({1, 1});

  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(convTransposeResult.getType());
  auto outputShape = outputType.getShape();
  // Checking to ensure only convtranspose with 2D spatial dims are supported.
  if ((outputShape.size() != 4) || (stridesShape.size() != 2) ||
      (padsShape.size() != 4) || (stridesShape[0] != stridesShape[1]) ||
      (stridesShape[0] > 3))
    return false;
  bool hDivisisbleByStride =
      (outputShape[outputShape.size() - 2] % stridesShape[0] == 0);
  bool wDivisisbleByStride =
      (outputShape[outputShape.size() - 1] % stridesShape[0] == 0);
  if (!hDivisisbleByStride || !wDivisisbleByStride) {
    return false;
  }

  // one Phase Decomposition
  if (stridesShape[0] == 1) {
    return true;
  }
  // If the stride is not 1, the kernel shape should be symmetric.
  if (!llvm::all_equal(kernelShape))
    return false;
  bool fourPhaseDecomposition = (stridesShape[0] == 2);
  bool ninePhaseDecomposition = (stridesShape[0] == 3);
  if (fourPhaseDecomposition) {
    if (outputShape[0] != 1) {
      // Currently support batch=1
      return false;
    }
    if (kernelShape[0] == 6 && padsShape[0] == 2 &&
        llvm::all_equal(padsShape)) {
      // Currently support only with pads [2, 2, 2, 2]
      return true;
    }
    if (kernelShape[0] == 3) {
      // Supports [0,0,1,1] , [1,1,0,0] padding only.
      if (padsShape == SmallVector<int64_t>{0, 0, 1, 1} ||
          padsShape == SmallVector<int64_t>{1, 1, 0, 0})
        return true;
    }
    // Supports only with padding [0, 0, 0, 0]
    if (kernelShape[0] == 2 && llvm::all_equal(padsShape) &&
        padsShape[0] == 0) {
      return true;
    }
    // Supports only with padding [1,1,1,1]
    if (kernelShape[0] == 4 && llvm::all_equal(padsShape) &&
        padsShape[0] == 1) {
      return true;
    }
  }
  if (ninePhaseDecomposition) {
    // Supports only with padding [0, 0, 0, 0]
    if (kernelShape[0] == 3 && llvm::all_equal(padsShape) &&
        padsShape[0] == 0) {
      return true;
    }
  }
  return false;
}
ONNXConstantOp getONNXConstOpFromVector(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  auto int64Type = rewriter.getIntegerType(64);
  SmallVector<mlir::Attribute> elements;
  transform(values, std::back_inserter(elements),
      [&](int64_t val) { return rewriter.getI64IntegerAttr(val); });
  auto constType = RankedTensorType::get(values.size(), int64Type);
  return rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(),
      DenseElementsAttr::get(constType, llvm::ArrayRef(elements)));
}

// This decomposition is targetting the convtranspose 1D operator.
// We have another decomposition targetting convtranspose 2D operator.
// Convtranpose can be decomposed into phased convolutions.
// The phased convolutions are then merged to get the final output.
// The number of phases is determined by the strides of the convtranspose op.
// The num of phases = stride
// The phased convolutions are weights are created by slicing the weights of the
// convolution in the specified manner and output of convolutions are stiched
// together to get the final output.
// Below shows the high level view of the decomposition.
// clang-format off
//                                                                                             
// +--------+     +--------+-------+--------+-------+--------+-------+                         
// |ConvT   |     |        |       |        |       |        |       |                         
// |        +---->| Conv1  |Conv2  | Conv1  |Conv2  | Conv1  |Conv2  |                         
// |stride 2|     |        |       |        |       |        |       |                         
// +--------+     +--------+-------+--------+-------+--------+-------+                         
//                                                                                             
// +--------+      +------+------+------+------+------+------+------+------+                   
// |ConvT   |      |      |      |      |      |      |      |      |      |                   
// |        +----> |conv1 |conv2 |conv3 |conv4 |conv1 |conv2 |conv3 |conv4 |                   
// |stride4 |      +------+------+------+------+------+------+------+------+                   
// +--------+                                                                                  
//                                                                                             
// +--------+                                                                                  
// |ConvT   |   +------+------+------+------+-----+                                            
// |        +-->|      |      |      |      |     |                                            
// |Stride 5|   |conv1 |conv2 |conv3 |conv4 |conv5|                                            
// +--------+   +------+------+------+------+-----+                                            
//                                                                                             
//                                                                                             
//  ConvTranspose weights are sliced to generated phased conv weights                          
//                                                                                             
//  phased conv outputs are merged to get complete ofm                                         
//                                                                                             
//                                                -
// clang-format on
// If no activation op ( lrelu or relu) found in the matching, the alpha value
// will be passed as the null, if relu is found 0 is passed, if lrelu is found
// the alpha value is passed to this method.
Value decomposeConvT1dIntoPhasedConvs(PatternRewriter &rewriter, Location loc,
    ONNXConvTransposeOp op, Value convTransposeResult, Value input,
    Value weights, Value bias, ArrayAttr dilations, IntegerAttr group,
    ArrayAttr inputKernelShape, ArrayAttr pads, ArrayAttr strides,
    FloatAttr alpha) {

  RankedTensorType weightsType =
      mlir::cast<RankedTensorType>(weights.getType());
  assert(weightsType.hasRank() && "Weight tensor must have rank.");
  Type elementType = getElementType(op.getType());
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(convTransposeResult.getType());
  auto convTransposeOutputShape = outputType.getShape();
  auto kernelShape = getIntVectorFromArrayAttr(inputKernelShape);
  auto padsShape =
      (pads) ? getIntVectorFromArrayAttr(pads) : SmallVector<int64_t>({0, 0});
  auto stridesShape = (strides) ? getIntVectorFromArrayAttr(strides)
                                : SmallVector<int64_t>({1});

  int numPhases = stridesShape[0];
  auto getActivationAppliedToConv = [&](Value conv, Type convOutputType) {
    if (!alpha)
      return conv;
    return (alpha.getValueAsDouble() == 0)
               ? rewriter.create<ONNXReluOp>(loc, convOutputType, conv)
                     .getResult()
               : rewriter
                     .create<ONNXLeakyReluOp>(loc, convOutputType, conv, alpha)
                     .getResult();
  };

  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);

  SmallVector<mlir::Value> weightSlices;
  int step = stridesShape[0];
  int convKernelSize = kernelShape[0] / stridesShape[0];

  auto axisOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {2});
  auto stepOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {step});
  auto weightsShape = weightsType.getShape();
  auto convWeightsShapedType =
      weightsType.get({weightsShape[0], weightsShape[1], convKernelSize},
          weightsType.getElementType());
  int64_t maxIndex = stridesShape[0];
  for (int row = 0; row < maxIndex; row++) {
    int rowEnd = (convKernelSize * step) + row;
    llvm::SmallVector<int64_t> startVector({row});
    llvm::SmallVector<int64_t> endVector({rowEnd});
    auto startOnnxConstant =
        getONNXConstOpFromVector(rewriter, loc, startVector);
    auto endOnnxConstant = getONNXConstOpFromVector(rewriter, loc, endVector);
    weightSlices.push_back(
        create.onnx.slice(convWeightsShapedType, weights, startOnnxConstant,
            endOnnxConstant, axisOnnxConstant, stepOnnxConstant));
  }

  auto convKernelShapeArrayAttr = rewriter.getI64ArrayAttr({convKernelSize});

  // This is the shape of the output from each conv, which contributes to the
  // final ofm.
  SmallVector<int64_t> convOutputShape(convTransposeOutputShape);
  int64_t innermostDim = convOutputShape.size() - 1;
  convOutputShape[innermostDim] =
      (convOutputShape[innermostDim] / stridesShape[0] + 1);

  ShapedType convTransposeOutputType =
      mlir::cast<ShapedType>(op.getY().getType());
  auto convOutputType = RankedTensorType::get(
      convOutputShape, convTransposeOutputType.getElementType());

  // for all the usecases supported by this decomposition, conv pads is [1,1]
  auto padsArrayAttr = rewriter.getI64ArrayAttr({1, 1});
  auto stridesArrayAttr = rewriter.getI64ArrayAttr({1});
  stepOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1});

  // The shape of the conv output to be consumed.
  SmallVector<int64_t> convSliceOutputShape(convTransposeOutputShape);
  convSliceOutputShape[innermostDim] =
      std::floor(convTransposeOutputShape[innermostDim] / stridesShape[0]);
  auto convSliceOutputType = RankedTensorType::get(
      convSliceOutputShape, convTransposeOutputType.getElementType());

  if (numPhases == 2) {
    Value conv1 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[1],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv2 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[0],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    auto startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1});
    auto endOnnxConstant = getONNXConstOpFromVector(
        rewriter, loc, {convOutputShape[innermostDim]});

    // for conv1 garbage is in 1st value, for conv2 it is last value.
    conv1 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv1,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {0});
    endOnnxConstant = getONNXConstOpFromVector(
        rewriter, loc, {convOutputShape[innermostDim] - 1});
    conv2 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv2,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);
    // The two convOutputs are adjusted to add an extra dimension at the
    // innermost level.
    SmallVector<int64_t> outputShapePlusOneDim(convSliceOutputShape);
    outputShapePlusOneDim.push_back(1);
    auto onnxConstForReshapeAddOneDim =
        getONNXConstOpFromVector(rewriter, loc, outputShapePlusOneDim);

    auto reshapeOutputType =
        RankedTensorType::get(outputShapePlusOneDim, elementType);

    auto reshapeOutputAddOneDimConv1 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv1, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv2 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv2, onnxConstForReshapeAddOneDim);
    SmallVector<int64_t> outputShapeLevel1Concat(outputShapePlusOneDim);
    outputShapeLevel1Concat[outputShapeLevel1Concat.size() - 1] = 2;
    auto level1ConcatOutputType =
        RankedTensorType::get(outputShapeLevel1Concat, elementType);

    // Below concats result will have the innermost dim as 2.
    auto finalConcat = rewriter.create<ONNXConcatOp>(loc,
        level1ConcatOutputType,
        ValueRange{reshapeOutputAddOneDimConv2, reshapeOutputAddOneDimConv1},
        -1);
    SmallVector<int64_t> outputShapeForResult(convSliceOutputShape);
    auto dimValueAtLastIndex =
        convSliceOutputShape[convSliceOutputShape.size() - 1] * 2;
    outputShapeForResult[outputShapeForResult.size() - 1] = dimValueAtLastIndex;

    auto onnxConstForLastReshape =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForResult);

    auto finalOutputType =
        RankedTensorType::get(outputShapeForResult, elementType);
    // Result is reshaped back to match the original convtranspose output
    // dimensions
    auto finalOutput = rewriter.create<ONNXReshapeOp>(
        loc, finalOutputType, finalConcat, onnxConstForLastReshape);
    return finalOutput;
  }
  if (numPhases == 4) {
    Value conv1 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[1],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv2 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[0],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv3 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[3],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv4 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[2],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    auto startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {0});
    auto endOnnxConstant = getONNXConstOpFromVector(
        rewriter, loc, {convOutputShape[innermostDim] - 1});

    // for conv1 and conv2 garbage is at end, for conv3 and conv4 it is at
    // start.
    conv1 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv1,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    conv2 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv2,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);
    startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1});
    endOnnxConstant = getONNXConstOpFromVector(
        rewriter, loc, {convOutputShape[innermostDim]});
    conv3 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv3,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    conv4 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv4,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);
    // The four convOutputs are adjusted to add an extra dimension at the
    // innermost level.
    SmallVector<int64_t> outputShapePlusOneDim(convSliceOutputShape);
    outputShapePlusOneDim.push_back(1);
    auto onnxConstForReshapeAddOneDim =
        getONNXConstOpFromVector(rewriter, loc, outputShapePlusOneDim);

    auto reshapeOutputType =
        RankedTensorType::get(outputShapePlusOneDim, elementType);

    auto reshapeOutputAddOneDimConv1 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv1, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv2 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv2, onnxConstForReshapeAddOneDim);

    auto reshapeOutputAddOneDimConv3 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv3, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv4 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv4, onnxConstForReshapeAddOneDim);

    SmallVector<int64_t> outputShapeLevel1Concat(outputShapePlusOneDim);
    outputShapeLevel1Concat[outputShapeLevel1Concat.size() - 1] = 4;
    auto level1ConcatOutputType =
        RankedTensorType::get(outputShapeLevel1Concat, elementType);

    // Below concats result will have the innermost dim as 2.
    auto finalConcat =
        rewriter.create<ONNXConcatOp>(loc, level1ConcatOutputType,
            ValueRange{reshapeOutputAddOneDimConv1, reshapeOutputAddOneDimConv2,
                reshapeOutputAddOneDimConv3, reshapeOutputAddOneDimConv4},
            -1);
    SmallVector<int64_t> outputShapeForResult(convSliceOutputShape);
    auto dimValueAtLastIndex =
        convSliceOutputShape[convSliceOutputShape.size() - 1] * 4;
    outputShapeForResult[outputShapeForResult.size() - 1] = dimValueAtLastIndex;

    auto onnxConstForLastReshape =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForResult);

    auto finalOutputType =
        RankedTensorType::get(outputShapeForResult, elementType);
    // Result is reshaped back to match the original convtranspose output
    // dimensions
    auto finalOutput = rewriter.create<ONNXReshapeOp>(
        loc, finalOutputType, finalConcat, onnxConstForLastReshape);
    return finalOutput;
  }
  if (numPhases == 5) {
    Value conv1 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[2],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv2 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[1],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv3 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[0],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv4 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[4],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    Value conv5 = getActivationAppliedToConv(
        rewriter.create<ONNXConvOp>(loc, convOutputType, input, weightSlices[3],
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr),
        convOutputType);
    auto startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1});
    auto endOnnxConstant = getONNXConstOpFromVector(
        rewriter, loc, {convOutputShape[innermostDim]});

    conv4 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv4,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    conv5 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv5,
        startOnnxConstant, endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    // 1. conv1 output is taken as is, it do not have any garbge.
    // 2. conv2, conv3 has garbage at the end, it will be taken care at the
    // last slice operation after the concat.
    // 3. conv4 and conv5 has garbage at the start, here we slice the start
    // garbage, and pad at the end to match with the sizes of other conv
    // outputs, to accomodate the concat of all the conv outputs.

    std::array<int64_t, 6> convOutputPadValue = {0, 0, 0, 0, 0, 1};

    auto onnxPadsValueConstant =
        getONNXConstOpFromVector(rewriter, loc, convOutputPadValue);
    RankedTensorType scalarTy = RankedTensorType::get({}, elementType);
    Value onnxPaddingConstantZero = create.onnx.constant(
        DenseElementsAttr::get(scalarTy, rewriter.getZeroAttr(elementType)));

    auto onnxAxisValueConstantNone = create.onnx.none();
    SmallVector<int64_t> paddedConvOutputShapeValue = {convSliceOutputShape[0],
        convSliceOutputShape[1], convSliceOutputShape[2] + 1};
    auto paddedConvOutputShapedType =
        convOutputType.get(paddedConvOutputShapeValue, elementType);

    conv4 = rewriter.create<ONNXPadOp>(loc, paddedConvOutputShapedType, conv4,
        onnxPadsValueConstant, onnxPaddingConstantZero,
        onnxAxisValueConstantNone, rewriter.getStringAttr("constant"));

    conv5 = rewriter.create<ONNXPadOp>(loc, paddedConvOutputShapedType, conv5,
        onnxPadsValueConstant, onnxPaddingConstantZero,
        onnxAxisValueConstantNone, rewriter.getStringAttr("constant"));

    // The five convOutputs are adjusted to add an extra dimension at the
    // innermost level.
    SmallVector<int64_t> outputShapePlusOneDim(paddedConvOutputShapeValue);
    outputShapePlusOneDim.push_back(1);
    auto onnxConstForReshapeAddOneDim =
        getONNXConstOpFromVector(rewriter, loc, outputShapePlusOneDim);

    auto reshapeOutputType =
        RankedTensorType::get(outputShapePlusOneDim, elementType);

    auto reshapeOutputAddOneDimConv1 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv1, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv2 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv2, onnxConstForReshapeAddOneDim);

    auto reshapeOutputAddOneDimConv3 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv3, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv4 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv4, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv5 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv5, onnxConstForReshapeAddOneDim);

    SmallVector<int64_t> outputShapeLevel1Concat(outputShapePlusOneDim);
    outputShapeLevel1Concat[outputShapeLevel1Concat.size() - 1] = 5;
    auto level1ConcatOutputType =
        RankedTensorType::get(outputShapeLevel1Concat, elementType);

    // Below concats result will have the innermost dim as 2.
    auto convOfmConcat =
        rewriter.create<ONNXConcatOp>(loc, level1ConcatOutputType,
            ValueRange{reshapeOutputAddOneDimConv1, reshapeOutputAddOneDimConv2,
                reshapeOutputAddOneDimConv3, reshapeOutputAddOneDimConv4,
                reshapeOutputAddOneDimConv5},
            -1);
    SmallVector<int64_t> outputShapeForResult(paddedConvOutputShapeValue);
    auto dimValueAtLastIndex =
        paddedConvOutputShapeValue[paddedConvOutputShapeValue.size() - 1] * 5;
    outputShapeForResult[outputShapeForResult.size() - 1] = dimValueAtLastIndex;

    auto onnxConstForLastReshape =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForResult);

    auto outputTypeBeforeSlice =
        RankedTensorType::get(outputShapeForResult, elementType);
    // Result is reshaped back to match the original convtranspose output
    // dimensions
    auto outputBeforeSlice = rewriter.create<ONNXReshapeOp>(
        loc, outputTypeBeforeSlice, convOfmConcat, onnxConstForLastReshape);

    SmallVector<int64_t> finalSliceOutputShape(convTransposeOutputShape);
    auto finalSliceOutputType = RankedTensorType::get(
        finalSliceOutputShape, convTransposeOutputType.getElementType());

    startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {0});
    endOnnxConstant = getONNXConstOpFromVector(rewriter, loc,
        {finalSliceOutputShape[finalSliceOutputShape.size() - 1]});

    auto finalSlicedOutput = rewriter.create<ONNXSliceOp>(loc,
        finalSliceOutputType, outputBeforeSlice, startOnnxConstant,
        endOnnxConstant, axisOnnxConstant, stepOnnxConstant);

    return finalSlicedOutput;
  }

  llvm_unreachable("Unsupported convtranspose decomposition");
}

// Convtranpose can be decomposed into phased convolutions.
// The phased convolutions are then merged to get the final output.
// The number of phases is determined by the strides of the convtranspose op.
// The num of phases = stride_x * stride_y.
// The phased convolutions are weights are created by slicing the weights of the
// convolution in the specified manner and output of convolutions are stiched
// together to get the final output. If the case where original weights cannot
// be sliced into conv weights directly, they are padded to make them compatible
// with the slicing. and subsequently the extra ofm generated by the padded
// weights are removed.
// Below shows the high level view of the decomposition.
// clang-format off
/*
 * +---------------+       +-----------+ 
 * | ConvT         |       |           | 
 * |               |       |   Conv    | 
 * |stride [1,1]   +--->   |           | 
 * |               |       |           | 
 * +---------------+       +-----------+ 
 *                                       
 *                                       
 *                                       
 *                                       
 *                         +-------+-------+-------+-------+-- --------------+  
 * +----------------+      | conv1 |conv3  | conv1 |conv3  |                 |
 * |  ConvT         |      |       |       |       |       |                 |
 * |                |      +-------+-------+-------+-------+                 |
 * |stride [2,2]    +--    |conv4  |conv2  |conv4  |conv2  |                 |
 * |                |      |       |       |       |       |                 |
 * +----------------+      +-------+-------+-------+-------+-                |
 *                         | conv1 |conv3  | conv1 |conv3  |                 |
 *                         |       |       |       |       |                 |
 *                         +-------+-------+-------+-------+                 |
 * convT weights sliced    |conv4  |conv2  |conv4  |conv2  |                 |
 * to get 4 conv weights   |       |       |       |       |                 |
 *                         +-------+-------+-------+-------+                 |
 *                         |               |                                 |
 *                         |                                                 |
 *                         |                                                 |
 *                         +-------------------------------------------------+
 *                                                                            
 *                                                                            
 *                                 4 conv ofms merged                         
 *                                                                            
 *                                                                            
 * +-------------------+         +--------+-------+------+--------+-------+------+-+
 * |                   |         |conv1   |conv2  |conv7 |conv1   |conv2  |conv7 | |
 * |convT              |         |        |       |      |        |       |      | |
 * |                   |         +--------+-------+------+--------+-------+------+ |
 * |stride [3,3]       |         |conv4   |conv5  |conv6 |conv4   |conv5  |conv6 | |
 * |                   |         |        |       |      |        |       |      | |
 * |                   |   -     +--------+-------+------+--------+-------+------+ |
 * +-------------------+         |conv3   |conv8  |conv9 |conv3   |conv8  |conv9 | |
 *                               |        |       |      |        |       |      | |
 * conT weights sliced           +--------+-------+------+--------+-------+------+-+
 *                               |conv1   |conv2  |conv7 |conv1   |conv2  |conv7 | |
 * to get 9 conv weights         |        |       |      |        |       |      | |
 *                               +--------+-------+------+--------+-------+------+ |
 *                               |conv4   |conv5  |conv6 |conv4   |conv5  |conv6 | |
 *                               |        |       |      |        |       |      | |
 *                               +--------+-------+------+--------+-------+------+ |
 *                               |conv3   |conv8  |conv9 |conv3   |conv8  |conv9 | |
 *                               |        |       |      |        |       |      | |
 *                               +--------+-------+------+--------+-------+------+ |
 *                               |                                                 |
 *                               |                                                 |
 *                               |                                                 |
 *                               +-------------------------------------------------+
 *                               9 conv ofms are merged                             
 */
// clang-format on
// If no activation op ( lrelu or relu) found in the matching, the alpha value
// will be passed as the null, if relu is found 0 is passed, if lrelu is found
// the alpha value is passed to this method.
Value decomposeIntoPhasedConvs(PatternRewriter &rewriter, Location loc,
    ONNXConvTransposeOp op, Value convTransposeResult, Value input,
    Value weights, Value bias, ArrayAttr dilations, IntegerAttr group,
    ArrayAttr kernel_shape, ArrayAttr pads, ArrayAttr strides, FloatAttr alpha,
    Value convtDeQuant, Value convtQuant, Value wtsDequant) {

  auto wtsDequantOp = wtsDequant.getDefiningOp<ONNXDequantizeLinearOp>();

  auto convtDequantOp = convtDeQuant.getDefiningOp<ONNXDequantizeLinearOp>();
  auto convtQuantOp = convtQuant.getDefiningOp<ONNXQuantizeLinearOp>();
  bool addQDQBetweenConvAndActivation =
      (convtDequantOp != nullptr && convtQuantOp != nullptr);

  bool addDequantizeNodeForWts = (wtsDequantOp != nullptr);

  RankedTensorType weightsType =
      mlir::cast<RankedTensorType>(weights.getType());
  assert(weightsType.hasRank() && "Weight tensor must have rank.");
  Type elementType = getElementType(op.getType());
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(convTransposeResult.getType());
  auto convTransposeOutputShape = outputType.getShape();
  auto kernelShape = getIntVectorFromArrayAttr(kernel_shape);
  auto padsShape = (pads) ? getIntVectorFromArrayAttr(pads)
                          : SmallVector<int64_t>({0, 0, 0, 0});
  auto stridesShape = (strides) ? getIntVectorFromArrayAttr(strides)
                                : SmallVector<int64_t>({1, 1});

  int numPhases = stridesShape[0] * stridesShape[1];

  //   +---------------+
  //   |               |
  //   |ConvTranspose  |
  //   |               |
  //   +--------+------+
  //            |
  //            v
  // +-----------------------+
  // |                       |
  // |ONNXQuantizeLinearOp   |
  // |                       |
  // +--------+--------------+
  //          |
  //          v
  // +--------------------------+
  // |                          |
  // |ONNXDequantizeLinearOp    |
  // |                          |
  // +----------+---------------+
  //            |
  //            v
  // +--------------------------+
  // |                          |
  // |Activation ( RELU / LRELU)|
  // |                          |
  // +--------------------------+
  //

  // This function helps to create the Q and DQ nodes for the activation from
  // the values matched in the pattern ConvTransposeOp -> Q -> DQ -> Relu The
  // result will be Conv->Q->DQ->Activation Between the Conv and Activation the
  // same Q and DQ nodes will be added

  auto addQDQNodesForActivationIfNeeded = [&](Value conv) -> Value {
    if (!addQDQBetweenConvAndActivation)
      return conv;

    // Properties from the ONNXQuantizeLinearOp Node taking input from
    // ConvTranspose.
    auto convtQuantScale = convtQuantOp.getYScale();
    auto convtQuantZeroPoint = convtQuantOp.getYZeroPoint();
    auto convtQuantAxis = convtQuantOp.getAxis();
    auto convtQuantSaturate = convtQuantOp.getSaturate();
    auto convtQuantBlockSize = convtQuantOp.getBlockSize();
    auto convtQuantOutputDtype = convtQuantOp.getOutputDtype();
    auto convtQuantLoc = convtQuantOp->getLoc();
    Type quantElementType = getElementType(convtQuantOp.getType());

    // Properties from the ONNXDequantizeLinearOp Node taking input from Q node
    // which inturn taking input from ConvTranspose Node.
    auto convtDequantScale = convtDequantOp.getXScale();
    auto convtDequantZeroPoint = convtDequantOp.getXZeroPoint();
    auto convtDequantBlockSize = convtDequantOp.getBlockSize();
    auto convtDequantAxis = convtDequantOp.getAxis();
    auto convtDequantLoc = convtDequantOp.getLoc();
    Type dequantElementType = getElementType(convtDequantOp.getType());

    RankedTensorType convOutputType =
        mlir::cast<RankedTensorType>(conv.getType());

    auto dequantOutputType =
        RankedTensorType::get(convOutputType.getShape(), dequantElementType);

    RankedTensorType quantOutputType =
        RankedTensorType::get(convOutputType.getShape(), quantElementType);

    auto quantNode =
        rewriter.create<ONNXQuantizeLinearOp>(convtQuantLoc, quantOutputType,
            conv, convtQuantScale, convtQuantZeroPoint, convtQuantAxis,
            convtQuantBlockSize, convtQuantOutputDtype, convtQuantSaturate);

    auto dequantNode = rewriter.create<ONNXDequantizeLinearOp>(convtDequantLoc,
        dequantOutputType, quantNode, convtDequantScale, convtDequantZeroPoint,
        convtDequantAxis, convtDequantBlockSize);
    return dequantNode;
  };

  auto getActivationAppliedToConv = [&](Value conv, Type convOutputType) {
    if (!alpha)
      return conv;
    return (alpha.getValueAsDouble() == 0)
               ? rewriter.create<ONNXReluOp>(loc, convOutputType, conv)
                     .getResult()
               : rewriter
                     .create<ONNXLeakyReluOp>(loc, convOutputType, conv, alpha)
                     .getResult();
  };

  auto addDequantizeNodeIfNeeded = [&](Value constantValue) -> Value {
    if (!addDequantizeNodeForWts)
      return constantValue;

    auto wtsDqScale = wtsDequantOp.getXScale();
    auto wtsDqZeroPoint = wtsDequantOp.getXZeroPoint();
    auto wtsDqAxis = wtsDequantOp.getAxis();
    auto wtsDqBlockSize = wtsDequantOp.getBlockSize();

    RankedTensorType constantType =
        mlir::cast<RankedTensorType>(constantValue.getType());

    auto dqOuputType =
        RankedTensorType::get(constantType.getShape(), elementType);

    return rewriter.create<ONNXDequantizeLinearOp>(loc, dqOuputType,
        constantValue, wtsDqScale, wtsDqZeroPoint, wtsDqAxis, wtsDqBlockSize);
  };
  if (numPhases == 1) {
    const std::array<int64_t, 4> convPadsShape = {
        (kernelShape[0] - 1 - padsShape[0]),
        (kernelShape[1] - 1 - padsShape[1]),
        (kernelShape[0] - 1 - padsShape[2]),
        (kernelShape[1] - 1 - padsShape[3])};

    auto convPadsArrayAttr = rewriter.getI64ArrayAttr(convPadsShape);

    return getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(
            rewriter.create<ONNXConvOp>(loc, op.getY().getType(), input,
                addDequantizeNodeIfNeeded(weights), bias, mlir::StringAttr(),
                dilations, group, kernel_shape, convPadsArrayAttr, strides)),
        op.getY().getType());
  }

  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  // If the convTranspose kernel is 3x3, then the weights needs to be padded to
  // 4x4
  bool needWeightsPadding = (kernelShape[0] == 3 && stridesShape[0] == 2);
  if (needWeightsPadding) {
    std::array<int64_t, 8> weightsPadValue = {0, 0, 0, 0, 0, 0, 0, 0};

    assert((padsShape == SmallVector<int64_t>{0, 0, 1, 1}) ||
           (padsShape == SmallVector<int64_t>{1, 1, 0, 0}));
    // Supports [0,0,1,1] , [1,1,0,0] padding only.
    if (padsShape[0] == 1) {
      weightsPadValue[2] = 1;
      weightsPadValue[3] = 1;
    }
    if (padsShape[2] == 1) {
      weightsPadValue[6] = 1;
      weightsPadValue[7] = 1;
    }
    auto onnxPadsValueConstant =
        getONNXConstOpFromVector(rewriter, loc, weightsPadValue);
    auto weightsElementType = weightsType.getElementType();
    RankedTensorType scalarTy = RankedTensorType::get({}, weightsElementType);
    Value onnxPaddingConstantZero = create.onnx.constant(DenseElementsAttr::get(
        scalarTy, rewriter.getZeroAttr(weightsElementType)));

    auto onnxAxisValueConstantNone = create.onnx.none();
    auto wts_shape = weightsType.getShape();
    // Padding the orignal weights at the bottom and right with zeros.
    auto paddedWeightsShapedType = weightsType.get(
        {wts_shape[0], wts_shape[1], wts_shape[2] + 1, wts_shape[3] + 1},
        weightsType.getElementType());
    weightsType = paddedWeightsShapedType;

    weights = rewriter.create<ONNXPadOp>(loc, paddedWeightsShapedType, weights,
        onnxPadsValueConstant, onnxPaddingConstantZero,
        onnxAxisValueConstantNone, rewriter.getStringAttr("constant"));
    kernelShape = {4, 4};
  }

  SmallVector<mlir::Value> weightSlices;
  int step = stridesShape[0];
  int convKernelSize = kernelShape[0] / stridesShape[0];

  auto axisOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {2, 3});
  auto stepOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {step, step});
  auto weightsShape = weightsType.getShape();
  auto convWeightsShapedType = weightsType.get(
      {weightsShape[0], weightsShape[1], convKernelSize, convKernelSize},
      weightsType.getElementType());
  int64_t maxIndex = stridesShape[0];
  for (int column = 0; column < maxIndex; column++) {
    for (int row = 0; row < maxIndex; row++) {
      int rowEnd = (convKernelSize * step) + row;
      int columnEnd = (convKernelSize * step) + column;
      llvm::SmallVector<int64_t> startVector({row, column});
      llvm::SmallVector<int64_t> endVector({rowEnd, columnEnd});
      auto startOnnxConstant =
          getONNXConstOpFromVector(rewriter, loc, startVector);
      auto endOnnxConstant = getONNXConstOpFromVector(rewriter, loc, endVector);
      weightSlices.push_back(
          create.onnx.slice(convWeightsShapedType, weights, startOnnxConstant,
              endOnnxConstant, axisOnnxConstant, stepOnnxConstant));
    }
  }
  auto convKernelShapeArrayAttr =
      rewriter.getI64ArrayAttr({convKernelSize, convKernelSize});
  // This is the shape of the output from each conv, which contributes to the
  // final ofm.
  SmallVector<int64_t> convOutputShape(convTransposeOutputShape);
  convOutputShape[convOutputShape.size() - 1] =
      convOutputShape[convOutputShape.size() - 1] / stridesShape[0];
  convOutputShape[convOutputShape.size() - 2] =
      convOutputShape[convOutputShape.size() - 2] / stridesShape[0];
  ShapedType convTransposeOutputType =
      mlir::cast<ShapedType>(op.getY().getType());
  // In the case where weights are padded, we will get the extra output from
  // conv.
  auto convOutputType = RankedTensorType::get(
      (needWeightsPadding)
          ? SmallVector<int64_t>({convOutputShape[0], convOutputShape[1],
                convOutputShape[2] + 1, convOutputShape[3] + 1})
          : convOutputShape,
      convTransposeOutputType.getElementType());
  if (numPhases == 4) {
    auto getPadsArrayAttr = [&](int64_t kernelSize, int64_t convSequence,
                                bool weightsPadded) {
      // weights are padded for case, kernel[3,3], stride[2,2] and pads either
      // [0,0,1,1] or [1,1,0,0]
      if (weightsPadded) {
        return rewriter.getI64ArrayAttr({1, 1, 1, 1});
      }
      // for kernel [2,2], stride [2,2] and pads [0,0,0,0]
      if (kernelSize == 2)
        return rewriter.getI64ArrayAttr({0, 0, 0, 0});
      if (kernelSize == 4) {
        // for kernel [4,4], stride [2,2] and pads [1,1,1,1]
        switch (convSequence) {
        case 1:
          return rewriter.getI64ArrayAttr({0, 0, 1, 1});
        case 2:
          return rewriter.getI64ArrayAttr({1, 1, 0, 0});
        case 3:
          return rewriter.getI64ArrayAttr({0, 1, 1, 0});
        case 4:
          return rewriter.getI64ArrayAttr({1, 0, 0, 1});
        default:
          llvm_unreachable("Invalid conv sequence.");
        }
      } else {
        // for kernel [6,6], stride [2,2] and pads [2,2,2,2]
        if (kernelSize == 6) {
          return rewriter.getI64ArrayAttr({1, 1, 1, 1});
        } else {
          llvm_unreachable("Invalid conv sequence.");
        }
      }
    };
    auto stridesArrayAttr = rewriter.getI64ArrayAttr({1, 1});
    Value conv;
    if (needWeightsPadding || (kernelShape[0] == 4)) {
      Value conv1 = getActivationAppliedToConv(
          addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
              convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[3]),
              bias, mlir::StringAttr(), dilations, group,
              convKernelShapeArrayAttr,
              getPadsArrayAttr(kernelShape[0], 1, needWeightsPadding),
              stridesArrayAttr)),
          convOutputType);
      Value conv2 = getActivationAppliedToConv(
          addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
              convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[0]),
              bias, mlir::StringAttr(), dilations, group,
              convKernelShapeArrayAttr,
              getPadsArrayAttr(kernelShape[0], 2, needWeightsPadding),
              stridesArrayAttr)),
          convOutputType);
      Value conv3 = getActivationAppliedToConv(
          addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
              convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[1]),
              bias, mlir::StringAttr(), dilations, group,
              convKernelShapeArrayAttr,
              getPadsArrayAttr(kernelShape[0], 3, needWeightsPadding),
              stridesArrayAttr)),
          convOutputType);
      Value conv4 = getActivationAppliedToConv(
          addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
              convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[2]),
              bias, mlir::StringAttr(), dilations, group,
              convKernelShapeArrayAttr,
              getPadsArrayAttr(kernelShape[0], 4, needWeightsPadding),
              stridesArrayAttr)),
          convOutputType);
      // Need to remove excess the ofm  when weights are padded.

      if (needWeightsPadding) {
        auto startOnnxConstant =
            getONNXConstOpFromVector(rewriter, loc, {1, 1});
        auto endOnnxConstant = getONNXConstOpFromVector(rewriter, loc,
            {convOutputShape[convOutputShape.size() - 2] + 2,
                convOutputShape[convOutputShape.size() - 1] + 2});
        auto axisOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {2, 3});
        auto stepOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1, 1});
        auto convSliceOutputType = RankedTensorType::get(
            convOutputShape, convTransposeOutputType.getElementType());
        conv1 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv1,
            startOnnxConstant, endOnnxConstant, axisOnnxConstant,
            stepOnnxConstant);

        startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {0, 0});
        endOnnxConstant = getONNXConstOpFromVector(rewriter, loc,
            {convOutputShape[convOutputShape.size() - 2],
                convOutputShape[convOutputShape.size() - 1]});
        conv2 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv2,
            startOnnxConstant, endOnnxConstant, axisOnnxConstant,
            stepOnnxConstant);

        startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {1, 0});
        endOnnxConstant = getONNXConstOpFromVector(rewriter, loc,
            {convOutputShape[convOutputShape.size() - 2] + 2,
                convOutputShape[convOutputShape.size() - 1]});
        conv3 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv3,
            startOnnxConstant, endOnnxConstant, axisOnnxConstant,
            stepOnnxConstant);

        startOnnxConstant = getONNXConstOpFromVector(rewriter, loc, {0, 1});
        endOnnxConstant = getONNXConstOpFromVector(rewriter, loc,
            {convOutputShape[convOutputShape.size() - 2],
                convOutputShape[convOutputShape.size() - 1] + 2});
        conv4 = rewriter.create<ONNXSliceOp>(loc, convSliceOutputType, conv4,
            startOnnxConstant, endOnnxConstant, axisOnnxConstant,
            stepOnnxConstant);
      }
      // Four conv outputs are merged in channel dim
      SmallVector<int64_t> outputShapeOfConcat = {
          1, convOutputShape[1] * 4, convOutputShape[2], convOutputShape[3]};
      auto concatOutputType =
          RankedTensorType::get(outputShapeOfConcat, elementType);
      // for the case where convtranspose kernel is [4, 4] and with pads [1, 1,
      // 1, 1] The phased convs output are to be concatenated in the reverse
      // order. This is observed by looking at the phased conv outputs with
      // respect to convtranspose output.
      bool reverseConcatOrder = (needWeightsPadding || (kernelShape[0] == 4));
      // The concat output will have 4 times the channels of a single conv.
      conv = (reverseConcatOrder)
                 ? rewriter.create<ONNXConcatOp>(loc, concatOutputType,
                       ValueRange{conv2, conv4, conv3, conv1}, 1)
                 : rewriter.create<ONNXConcatOp>(loc, concatOutputType,
                       ValueRange{conv1, conv3, conv4, conv2}, 1);
    } else {
      // Combining the 4 phased weights into single weight.
      bool reverseOrder = (kernelShape[0] == 4);
      auto combinedConvWeightsShapedType =
          weightsType.get({weightsShape[0] * 4, weightsShape[1], convKernelSize,
                              convKernelSize},
              weightsType.getElementType());

      Value combinedWeights =
          (reverseOrder) ? rewriter.create<ONNXConcatOp>(loc,
                               combinedConvWeightsShapedType,
                               ValueRange{weightSlices[0], weightSlices[2],
                                   weightSlices[1], weightSlices[3]},
                               0)
                         : rewriter.create<ONNXConcatOp>(loc,
                               combinedConvWeightsShapedType,
                               ValueRange{weightSlices[3], weightSlices[1],
                                   weightSlices[2], weightSlices[0]},
                               0);

      if (!bias.getDefiningOp<ONNXNoneOp>()) {
        RankedTensorType biasType =
            mlir::cast<RankedTensorType>(bias.getType());
        auto biasShape = biasType.getShape();

        auto combinedBiasShapedType =
            biasType.get({biasShape[0] * 4}, biasType.getElementType());

        bias = rewriter.create<ONNXConcatOp>(
            loc, combinedBiasShapedType, ValueRange{bias, bias, bias, bias}, 0);
      }

      auto combinedConvOutputType = RankedTensorType::get(
          SmallVector<int64_t>({convOutputShape[0], convOutputShape[1] * 4,
              convOutputShape[2], convOutputShape[3]}),
          convTransposeOutputType.getElementType());
      conv = getActivationAppliedToConv(
          addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
              combinedConvOutputType, input,
              addDequantizeNodeIfNeeded(combinedWeights), bias,
              mlir::StringAttr(), dilations, group, convKernelShapeArrayAttr,
              getPadsArrayAttr(kernelShape[0], 1, needWeightsPadding),
              stridesArrayAttr)),
          combinedConvOutputType);
    }

    // Here we are reshaping the concatenated conv channels of 4*Conv_channels
    // into groups of 2x2 channels. This can be visualized as
    // H_chan(2) * W_Chan(2) * C_real,  then doing the transpose into
    // Conv_channels H H_chan W W_chan. The adjecent H and H_chan will be merged
    // into H, same way W and W_chan will be merged into W. This leads to
    // doubling of the H and W. Keeping the channels same.

    SmallVector<int64_t> outputShapeForDimAdjust = {
        2, 2, convOutputShape[1], convOutputShape[2], convOutputShape[3]};

    auto onnxConstForReshapeDimAdjust =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForDimAdjust);

    auto reshapeOutputForDimAdjustType =
        RankedTensorType::get(outputShapeForDimAdjust, elementType);
    auto reshapeOutputDimAdjust = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputForDimAdjustType, conv, onnxConstForReshapeDimAdjust);

    SmallVector<int64_t> transposeOuputShape = {
        convOutputShape[1], convOutputShape[2], 2, convOutputShape[3], 2};

    auto transposeOutputType =
        RankedTensorType::get(transposeOuputShape, elementType);

    auto permArrayAttr = rewriter.getI64ArrayAttr({2, 3, 0, 4, 1});

    auto transpose = rewriter.create<ONNXTransposeOp>(
        loc, transposeOutputType, reshapeOutputDimAdjust, permArrayAttr);

    SmallVector<int64_t> outputShapeForResult = {
        1, convOutputShape[1], convOutputShape[2] * 2, convOutputShape[3] * 2};

    auto onnxConstForLastReshape =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForResult);

    auto finalOutputType =
        RankedTensorType::get(outputShapeForResult, elementType);
    // Result is reshaped back to match the original convtranspose output
    // dimensions
    auto finalOutput = rewriter.create<ONNXReshapeOp>(
        loc, finalOutputType, transpose, onnxConstForLastReshape);
    return finalOutput;
  }
  if (numPhases == 9) {
    auto padsArrayAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto stridesArrayAttr = rewriter.getI64ArrayAttr({1, 1});

    auto conv1 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[8]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv2 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[5]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv3 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[6]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv4 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[7]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv5 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[4]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv6 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[1]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv7 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[2]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv8 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[3]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);
    auto conv9 = getActivationAppliedToConv(
        addQDQNodesForActivationIfNeeded(rewriter.create<ONNXConvOp>(loc,
            convOutputType, input, addDequantizeNodeIfNeeded(weightSlices[0]),
            bias, mlir::StringAttr(), dilations, group,
            convKernelShapeArrayAttr, padsArrayAttr, stridesArrayAttr)),
        convOutputType);

    // The nine convOutputs are adjusted to add an extra dimension at the
    // innermost level.
    SmallVector<int64_t> outputShapePlusOneDim(convOutputShape);
    outputShapePlusOneDim.push_back(1);
    auto onnxConstForReshapeAddOneDim =
        getONNXConstOpFromVector(rewriter, loc, outputShapePlusOneDim);

    auto reshapeOutputType =
        RankedTensorType::get(outputShapePlusOneDim, elementType);

    auto reshapeOutputAddOneDimConv1 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv1, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv2 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv2, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv3 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv3, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv4 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv4, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv5 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv5, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv6 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv6, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv7 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv7, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv8 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv8, onnxConstForReshapeAddOneDim);
    auto reshapeOutputAddOneDimConv9 = rewriter.create<ONNXReshapeOp>(
        loc, reshapeOutputType, conv9, onnxConstForReshapeAddOneDim);

    SmallVector<int64_t> outputShapeForLevel1Concat(outputShapePlusOneDim);
    outputShapeForLevel1Concat[outputShapeForLevel1Concat.size() - 1] = 3;
    auto level1ConcatOutputType =
        RankedTensorType::get(outputShapeForLevel1Concat, elementType);

    // Below concats result will have the innermost dim as 2.
    auto firstRowConcat =
        rewriter.create<ONNXConcatOp>(loc, level1ConcatOutputType,
            ValueRange{reshapeOutputAddOneDimConv1, reshapeOutputAddOneDimConv2,
                reshapeOutputAddOneDimConv7},
            -1);
    auto secondRowConcat =
        rewriter.create<ONNXConcatOp>(loc, level1ConcatOutputType,
            ValueRange{reshapeOutputAddOneDimConv4, reshapeOutputAddOneDimConv5,
                reshapeOutputAddOneDimConv6},
            -1);
    auto thirdRowConcat =
        rewriter.create<ONNXConcatOp>(loc, level1ConcatOutputType,
            ValueRange{reshapeOutputAddOneDimConv3, reshapeOutputAddOneDimConv8,
                reshapeOutputAddOneDimConv9},
            -1);

    // Reshaping to modify the two innermost levels,ensuring the second
    // innermost level is set to 1
    SmallVector<int64_t> outputShapeForDimAdjust(convOutputShape);
    auto dimValueAtLastIndex = convOutputShape[convOutputShape.size() - 1] * 3;
    outputShapeForDimAdjust[outputShapeForDimAdjust.size() - 1] = 1;
    outputShapeForDimAdjust.push_back(dimValueAtLastIndex);

    auto onnxConstForReshapeDimAdjust =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForDimAdjust);
    auto reshapeOutputForDimAdjustType =
        RankedTensorType::get(outputShapeForDimAdjust, elementType);

    auto reshapeOutputDimAdjustOfFirstConcat =
        rewriter.create<ONNXReshapeOp>(loc, reshapeOutputForDimAdjustType,
            firstRowConcat, onnxConstForReshapeDimAdjust);
    auto reshapeOutputDimAdjustOfSecondConcat =
        rewriter.create<ONNXReshapeOp>(loc, reshapeOutputForDimAdjustType,
            secondRowConcat, onnxConstForReshapeDimAdjust);
    auto reshapeOutputDimAdjustOfThirdConcat =
        rewriter.create<ONNXReshapeOp>(loc, reshapeOutputForDimAdjustType,
            thirdRowConcat, onnxConstForReshapeDimAdjust);

    SmallVector<int64_t> outputShapeForFinalConcat(outputShapeForDimAdjust);
    outputShapeForFinalConcat[outputShapeForFinalConcat.size() - 2] = 3;

    auto finalConcatOutputType =
        RankedTensorType::get(outputShapeForFinalConcat, elementType);

    // Final Concat is performed on the three reshaped outputs at the
    // second innermost level
    auto finalConcat = rewriter.create<ONNXConcatOp>(loc, finalConcatOutputType,
        ValueRange{reshapeOutputDimAdjustOfFirstConcat,
            reshapeOutputDimAdjustOfSecondConcat,
            reshapeOutputDimAdjustOfThirdConcat},
        -2);
    SmallVector<int64_t> outputShapeForResult(convOutputShape);
    dimValueAtLastIndex = convOutputShape[convOutputShape.size() - 1] * 3;
    auto dimValueAtSecondLastIndex =
        convOutputShape[convOutputShape.size() - 2] * 3;
    outputShapeForResult[outputShapeForResult.size() - 2] =
        dimValueAtSecondLastIndex;
    outputShapeForResult[outputShapeForResult.size() - 1] = dimValueAtLastIndex;

    auto onnxConstForLastReshape =
        getONNXConstOpFromVector(rewriter, loc, outputShapeForResult);

    auto finalOutputType =
        RankedTensorType::get(outputShapeForResult, elementType);
    // Result is reshaped back to match the original convtranspose output
    // dimensions
    auto finalOutput = rewriter.create<ONNXReshapeOp>(
        loc, finalOutputType, finalConcat, onnxConstForLastReshape);
    return finalOutput;
  }
  llvm_unreachable("Unsupported convtranspose decomposition");
}

// Split on the specified axis. The length of each output is one.
ValueRange emitSplitAxisOutputLength1(
    PatternRewriter &rewriter, Location loc, Value input, int64_t axis) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Create `split` to split each output in `axis` into length 1.
  // Ex. inputShape[axis] = 3, then  onnx.Constant dense<1> : tensor<3xi64>
  // TODO: Support dynamic dim for spatial dim.
  assert(!inputType.isDynamicDim(axis) &&
         "Spatial dimensions for input data tensor need to be static.");
  SmallVector<int64_t, 1> values(inputShape[axis], 1);
  Value split = create.onnx.constantInt64(ArrayRef(values));
  Type resultType = UnrankedTensorType::get(elementType);
  SmallVector<Type, 4> resultTypes(values.size(), resultType);
  ValueRange results =
      create.onnx.split(ArrayRef(resultTypes), input, split, axis);
  return results;
}

// Emit ONNXPadOp to add pads of `size` at end of the `axis`.
Value emitPadsAxisEnd(PatternRewriter &rewriter, Location loc, Value input,
    ArrayRef<int64_t> inputShape, int64_t axis, int64_t size) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  // Specify padding at the end of each axis.
  SmallVector<int64_t, 1> values((int64_t)inputShape.size() * 2, 0);
  values[inputShape.size() + axis] = size;
  Value pads = create.onnx.constantInt64(ArrayRef(values));
  Value result = create.onnx.padZero(input, pads);
  return result;
}

// Insert pads in specified axis.
Value insertPadAxis(PatternRewriter &rewriter, Location loc, Value input,
    int64_t axis, int64_t padSize) {
  // Split on the specified axis. The length of each output is one.
  ValueRange splitResults =
      emitSplitAxisOutputLength1(rewriter, loc, input, axis);
  // Add pad in split results except last one.
  Value splitLastResults = splitResults.back();
  ValueRange padInputs = splitResults.drop_back();
  SmallVector<Value, 4> padResults;
  for (Value v : padInputs) {
    ArrayRef<int64_t> vShape = mlir::cast<ShapedType>(v.getType()).getShape();
    padResults.emplace_back(
        emitPadsAxisEnd(rewriter, loc, v, vShape, axis, padSize));
  }
  padResults.emplace_back(splitLastResults);
  // Concat padded results.
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  Type elementType = getElementType(padResults[0].getType());
  Type concatType = UnrankedTensorType::get(elementType);
  Value concatResult =
      create.onnx.concat(concatType, ValueRange(padResults), axis);
  return concatResult;
}

// Insert pads between elements in input tensor in spatial dimensions.
// The padding size is strides - 1
Value insertPadsConvTransposeInput(PatternRewriter &rewriter, Location loc,
    ONNXConvTransposeOp op, Value input) {
  ONNXConvTransposeOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  SmallVector<int64_t, 2> strides = shapeHelper.strides;
  int64_t spatialOffset = 2;
  for (unsigned int i = 0; i < strides.size(); ++i) {
    input = insertPadAxis(rewriter, loc, input, /*axis*/ spatialOffset + i,
        /*padSize*/ strides[i] - 1);
  }
  return input;
}

// Insert additional padding to output of ConvOp in ConvTransposeOp.
Value insertAdditionalPadsConvTranspose(PatternRewriter &rewriter, Location loc,
    ONNXConvOp convOp, Value input, ONNXConvTransposeOp op) {
  ONNXConvOpShapeHelper convShapeHelper(convOp.getOperation(), {});
  Type elementType = getElementType(input.getType());
  (void)convShapeHelper.computeShapeAndUpdateType(elementType);
  int inputRank = convShapeHelper.getOutputDims().size();
  SmallVector<int64_t, 4> inputShape;
  for (int i = 0; i < inputRank; ++i) {
    int64_t d = convShapeHelper.getOutputDims()[i].isLiteral()
                    ? convShapeHelper.getOutputDims()[i].getLiteral()
                    : ShapedType::kDynamic;
    inputShape.emplace_back(d);
  }
  ONNXConvTransposeOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  SmallVector<int64_t, 2> padSize;
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  int64_t spatialOffset = 2;
  int64_t spatialRank = inputType.getRank() - spatialOffset;
  DimsExpr outputDims = shapeHelper.getOutputDims();
  for (int i = 0; i < spatialRank; ++i) {
    assert(outputDims[spatialOffset + i].isLiteral() &&
           "Only static spatial dims supported");
    int64_t size = outputDims[spatialOffset + i].getLiteral() -
                   inputShape[spatialOffset + i];
    assert(size >= 0 && "Invalid output_shape attribute");
    padSize.emplace_back(size);
  }
  Value paddedInput = emitPadsAxisEnd(
      rewriter, loc, input, ArrayRef(inputShape), /*axis*/ 2, padSize[0]);
  for (int i = 1; i < spatialRank; ++i) {
    ArrayRef<int64_t> paddedInputShape =
        mlir::cast<ShapedType>(paddedInput.getType()).getShape();
    paddedInput = emitPadsAxisEnd(rewriter, loc, paddedInput, paddedInputShape,
        /*axis*/ 2 + i, padSize[i]);
  }
  return paddedInput;
}
// ConvTransposeOp END

Value normalizeConstantOp(
    PatternRewriter &rewriter, Value output, Attribute attr) {
  ShapedType outputType = mlir::cast<ShapedType>(output.getType());
  Type elementType = outputType.getElementType();

  DenseElementsAttr denseAttr;
  if (ArrayAttr arrayAttr = mlir::dyn_cast<ArrayAttr>(attr)) {
    int64_t dim = arrayAttr.size();
    auto tensorType = RankedTensorType::get({dim}, elementType);
    denseAttr = DenseElementsAttr::get(tensorType, arrayAttr.getValue());
  } else {
    auto tensorType = RankedTensorType::get({}, elementType);
    if (FloatAttr floatAttr = mlir::dyn_cast<FloatAttr>(attr)) {
      denseAttr = DenseElementsAttr::get(tensorType, {floatAttr.getValue()});
    } else if (IntegerAttr intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      denseAttr = DenseElementsAttr::get(tensorType, intAttr.getSInt());
    } else if (StringAttr strAttr = mlir::dyn_cast<StringAttr>(attr)) {
      denseAttr = DenseElementsAttr::get(tensorType, {strAttr.getValue()});
    } else {
      llvm_unreachable("unexpected Attribute");
    }
  }
  onnx_mlir::OnnxBuilder createONNX(rewriter, output.getLoc());
  return createONNX.constant(denseAttr);
}

} // namespace onnx_mlir

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Dialect/ONNX/Transforms/ONNXDecompose.inc"
namespace convtranspose {
#include "src/Dialect/ONNX/Transforms/ONNXDecomposeConvTranspose.inc"
}
namespace convtranspose_phased {
#include "src/Dialect/ONNX/Transforms/ONNXDecomposeConvTransposePhased.inc"
}
namespace convtranspose_1d_phased {
#include "src/Dialect/ONNX/Transforms/ONNXDecomposeConvTranspose1dPhased.inc"
}
RankedTensorType createReducedType(
    Type outputType, int64_t axisValue, bool keepDims) {
  RankedTensorType outputShapeType =
      mlir::dyn_cast<RankedTensorType>(outputType);
  llvm::ArrayRef<int64_t> shapeVector = outputShapeType.getShape();
  int64_t rank = outputShapeType.getRank();
  if (axisValue < 0)
    axisValue += rank;
  SmallVector<int64_t, 4> reducedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != axisValue)
      reducedShape.push_back(shapeVector[i]);
    else if (keepDims)
      reducedShape.push_back(1);
  }
  Type elementType = outputShapeType.getElementType();
  RankedTensorType resultType =
      RankedTensorType::get(reducedShape, elementType);
  return resultType;
}

#ifdef ONNX_MLIR_ENABLE_STABLEHLO

struct SoftmaxPattern : public OpRewritePattern<ONNXSoftmaxOp> {
  using OpRewritePattern<ONNXSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSoftmaxOp softmaxOp, PatternRewriter &rewriter) const final {
    // Match
    Value input = softmaxOp.getInput();
    Type inputType = input.getType();
    int64_t axisValue = softmaxOp.getAxis();

    // Rewrite
    Location odsLoc = softmaxOp.getLoc();
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, odsLoc);

    IntegerAttr keepDimsAttr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), 1);
    ArrayAttr axisAttr = rewriter.getI64ArrayAttr({axisValue});
    RankedTensorType resultType =
        createReducedType(inputType, axisValue, /*keepDims=*/true);
    Value maxInput = rewriter.create<ONNXReduceMaxV13Op>(
        odsLoc, resultType, input, axisAttr, keepDimsAttr);
    Value subValue =
        rewriter.create<ONNXSubOp>(odsLoc, inputType, input, maxInput);
    Value expValue = rewriter.create<ONNXExpOp>(odsLoc, inputType, subValue);
    Value axisOp = create.onnx.constantInt64({axisValue});
    IntegerAttr noopWithEmptyAxes = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), 0);
    Value sumValue = rewriter.create<ONNXReduceSumOp>(odsLoc, resultType,
        /*input=*/expValue,
        /*axis=*/axisOp, keepDimsAttr, noopWithEmptyAxes);
    Value divValue =
        rewriter.create<ONNXDivOp>(odsLoc, inputType, expValue, sumValue);
    rewriter.replaceOp(softmaxOp, divValue);
    return success();
  }
};

void populateDecomposingONNXBeforeStablehloPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<SoftmaxPattern>(ctx);
}

#endif

// Special Op fusion for the following pattern:
//   %1 = Concat(inputs, axis)
//   %2 = Shape(%1, start, end)
//   %3 = Transpose(%1, perm)
// into a special Op
//   %2, %3 = ConcatShapeTranspose(inputs, axis, start, end, perm)
// This fusion is an experimental work for performance

// Helper function: is the ConcatOp matched to the fusion pattern?
static bool isConcatFuseMatched(
    ONNXConcatOp concatOp, ONNXShapeOp &shapeOp, ONNXTransposeOp &transposeOp) {
  shapeOp = nullptr;
  transposeOp = nullptr;
  bool failed = false;
  for (Operation *user : concatOp->getUsers()) {
    if (isa<ONNXShapeOp>(user) && !shapeOp)
      shapeOp = cast<ONNXShapeOp>(user);
    else if (isa<ONNXTransposeOp>(user) && !transposeOp)
      transposeOp = cast<ONNXTransposeOp>(user);
    else
      failed = true;
  }
  return (shapeOp && transposeOp && !failed);
}

struct ConcatFusePattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    // Match
    ONNXShapeOp shapeOp;
    ONNXTransposeOp transposeOp;
    if (!isConcatFuseMatched(concatOp, shapeOp, transposeOp))
      return failure();

    // Rewrite
    SmallVector<Type, 2> outputTypes;
    outputTypes.emplace_back(shapeOp.getResult().getType());
    outputTypes.emplace_back(transposeOp.getResult().getType());

    auto fusedV = rewriter.create<ONNXConcatShapeTransposeOp>(concatOp.getLoc(),
        outputTypes, concatOp->getOperands(), concatOp.getAxisAttr(),
        shapeOp.getEndAttr(), shapeOp.getStartAttr(),
        transposeOp.getPermAttr());
    rewriter.replaceOp(shapeOp.getOperation(), fusedV.getResults()[0]);
    rewriter.replaceOp(transposeOp.getOperation(), fusedV.getResults()[1]);
    rewriter.eraseOp(concatOp);
    return success();
  }
};

// ONNXHardSwishOp(input) can be decomposed as:
//   input * ONNXHardSigmoid input, with alpha = 1/6 and beta = 0.5.
struct DecomposeHardSwishPattern : public OpRewritePattern<ONNXHardSwishOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXHardSwishOp hardSwishOp, PatternRewriter &rewriter) const final {

    auto input = hardSwishOp.getX();
    auto hardSigmoid = rewriter.create<ONNXHardSigmoidOp>(hardSwishOp->getLoc(),
        hardSwishOp.getType(), input, rewriter.getF32FloatAttr(1.0 / 6.0),
        rewriter.getF32FloatAttr(0.5));
    rewriter.replaceOpWithNewOp<ONNXMulOp>(
        hardSwishOp, hardSwishOp.getType(), hardSigmoid, input);
    return success();
  }
};

// Decompose a pad with negative padding size to slice + pad
// Only supports static shapes
struct DecomposeSlicePadPattern : public OpRewritePattern<ONNXPadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXPadOp padOp, PatternRewriter &rewriter) const final {
    auto constantPad = padOp.getPads().getDefiningOp<ONNXConstantOp>();
    if (!constantPad) {
      return failure();
    }
    std::optional<Attribute> padValues;
    if (auto intAttrs = constantPad.getValueInts()) {
      padValues = intAttrs;
    } else if (auto attrs = constantPad.getValue()) {
      padValues = attrs;
    }
    if (!padValues) {
      return failure();
    }
    auto elementsAttr = llvm::dyn_cast<ElementsAttr>(*padValues);
    if (!elementsAttr) {
      return failure();
    }
    const auto padElements = onnx_mlir::getElementsArray<int64_t>(elementsAttr);
    const auto padElementsArray = padElements.get();
    if (llvm::none_of(padElementsArray, [](const auto v) { return v < 0; })) {
      // No slicing needed
      return failure();
    }
    if (!padOp.getAxes().getDefiningOp<ONNXNoneOp>()) {
      // This is possible to implement but makes the implementation more
      // difficult, so skip for now
      return failure();
    }
    const auto inputType = padOp.getData().getType().cast<ShapedType>();
    if (!inputType.hasStaticShape()) {
      // We need a static shape to calculate the ends for slice
      return failure();
    }
    auto sliceOp = buildSliceOp(padOp, rewriter, padElementsArray, inputType);
    auto newPadOp = buildPadOp(padOp, rewriter, padElementsArray, sliceOp);
    rewriter.replaceOp(padOp, newPadOp);
    return success();
  }

private:
  // Builds ands inserts a pad op, that is guaranteed to only pad and not
  // slice
  static Value buildPadOp(ONNXPadOp orignalPadOp, PatternRewriter &rewriter,
      ArrayRef<int64_t> padElementsArray, ONNXSliceOp sliceOp) {
    SmallVector<int64_t> pads;
    for (const auto padElem : padElementsArray) {
      pads.push_back((padElem < 0) ? 0 : padElem);
    }
    if (llvm::any_of(pads, [](const auto p) { return p > 0; })) {
      auto padsConstOp = onnx_mlir::createConstantOp(
          rewriter, orignalPadOp->getLoc(), rewriter.getI64ArrayAttr(pads));
      auto padOp = rewriter.create<ONNXPadOp>(orignalPadOp->getLoc(),
          orignalPadOp.getType(), sliceOp, padsConstOp,
          orignalPadOp.getConstantValue(), orignalPadOp.getAxes(),
          orignalPadOp.getMode());
      return padOp;
    }
    return sliceOp; // No pad needed if we only slice
  }

  // Builds and inserts a slice op, and its inputs, that handles negative
  // pads
  static ONNXSliceOp buildSliceOp(ONNXPadOp padOp, PatternRewriter &rewriter,
      ArrayRef<int64_t> padElementsArray, ShapedType inputType) {
    const auto inputShape = inputType.getShape();
    const size_t dims = padElementsArray.size() / 2;

    assert(inputShape.size() == dims);
    SmallVector<int64_t> sliceShape;
    for (size_t i = 0; i < dims; ++i) {
      auto sliceDimSize = inputShape[i];
      if (padElementsArray[i] < 0) {
        sliceDimSize += padElementsArray[i];
      }
      if (padElementsArray[i + dims] < 0) {
        sliceDimSize += padElementsArray[i + dims];
      }
      sliceShape.push_back(sliceDimSize);
    }
    auto sliceType = inputType.clone(sliceShape);

    SmallVector<int64_t> sliceStarts;
    for (size_t i = 0; i < dims; ++i) {
      if (padElementsArray[i] < 0) {
        sliceStarts.push_back(-padElementsArray[i]);
      } else {
        sliceStarts.push_back(0);
      }
    }
    auto startsConstOp = onnx_mlir::createConstantOp(
        rewriter, padOp->getLoc(), rewriter.getI64ArrayAttr(sliceStarts));

    SmallVector<int64_t> sliceEnds;
    for (size_t i = 0; i < dims; ++i) {
      const auto endIdx = inputShape[i];
      if (padElementsArray[i + dims] < 0) {
        sliceEnds.push_back(endIdx + padElementsArray[i + dims]);
      } else {
        sliceEnds.push_back(endIdx);
      }
    }
    auto endsConstOp = onnx_mlir::createConstantOp(
        rewriter, padOp->getLoc(), rewriter.getI64ArrayAttr(sliceEnds));

    auto sliceOp = rewriter.create<ONNXSliceOp>(padOp->getLoc(), sliceType,
        padOp.getData(), startsConstOp, endsConstOp,
        rewriter.create<ONNXNoneOp>(padOp->getLoc()),
        rewriter.create<ONNXNoneOp>(padOp->getLoc()));
    return sliceOp;
  }
};

namespace {
template <typename T>
class SubArrayAccessHelper {
public:
  explicit SubArrayAccessHelper(ArrayRef<T> data, size_t iterArraySize)
      : data(data), iterArraySize(iterArraySize) {
    assert((data.size() % iterArraySize) == 0);
  }

  [[nodiscard]] size_t size() const { return data.size() / iterArraySize; }

  ArrayRef<T> operator[](size_t idx) const {
    return data.slice(idx * iterArraySize, iterArraySize);
  }

private:
  ArrayRef<T> data;
  size_t iterArraySize;
};

class IndicesContiguousCounter {
public:
  explicit IndicesContiguousCounter(
      ArrayRef<int64_t> firstElem, ArrayRef<int64_t> shapeToCheck)
      : counter(firstElem), firstElem(firstElem), shapeToCheck(shapeToCheck) {}

  ArrayRef<int64_t> getCounter() const { return counter; }

  void increment() {
    // Increment from the back, carry if necessary
    for (auto [shapeToCheckDimSize, firstElemDimSize, c] :
        llvm::zip(llvm::reverse(shapeToCheck), llvm::reverse(firstElem),
            llvm::reverse(counter))) {
      if (c == (shapeToCheckDimSize + firstElemDimSize - 1)) {
        c = firstElemDimSize; // Carry and keep an eventual shift in mind
      } else {
        c++;
        break;
      }
    }
  }

private:
  SmallVector<int64_t> counter;
  ArrayRef<int64_t> firstElem;
  ArrayRef<int64_t> shapeToCheck;
};

} // namespace

// Decomposes ScatterNDs into a single Split and Concat.
// We can always split ScatterNDs by splitting the input tensor together with
// the indices and their updates belonging to that part of the input tensor,
// performing the ScatterNDs on each split, and the concatenating the result.
// Here, we handle certain ScatterNDs where after splitting them into three,
// the first and last ScatterND have empty indices (because the indices don't
// affect their parts of the input tensor), and the middle ScatterND overwrites
// the full input with sequential indices (i.e. can be replaced by a copy of its
// update).
//
// Example:
// ` %indices = onnx.Constant dense<[[[[0, 1, 0], [0, 1, 1], [0, 1, 2],
//     [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8],
//     [0, 1, 9]]]]> : tensor<1x1x10x3xi64>
//   %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} :
//     (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) ->
//     tensor<1x6x10x12xf32>`
// gets decomposed to:
// ` %0 = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
//   %1:3 = "onnx.Split"(%data, %0) {axis = 1 : si64} : (tensor<1x6x10x12xf32>,
//    tensor<3xi64>) -> (tensor<1x1x10x12xf32>, tensor<1x1x10x12xf32>,
//    tensor<1x4x10x12xf32>)
//   %2 = "onnx.Concat"(%1#0, %updates, %1#2) {axis = 1 : si64} :
//    (tensor<1x1x10x12xf32>,tensor<1x1x10x12xf32>, tensor<1x4x10x12xf32>) ->
//    tensor<1x6x10x12xf32>`
//
// ScatterND pseudo code:
//   output = np.copy(data)
//   update_indices = indices.shape[:-1]
//   for idx in np.ndindex(update_indices):
//     output[indices[idx]] = updates[idx]
//
// Inputs:
//  data (heterogeneous) - T: Tensor of rank r >= 1.
//  indices (heterogeneous) - tensor(int64): Tensor of rank q >= 1.
//  updates (heterogeneous) - T: Tensor of rank q + r - indices_shape[-1] - 1.
//
// Outputs:
//  output (heterogeneous) - T: Tensor of rank r >= 1.
//
// To ensure that this decomposition to split and concat is
// valid, the following constraints need to hold:
// - r == rank(updates)
// - The shape of data and updates differs only in one dimension 'a'
// -- 'a' is the dimension where the split and concat will happen
// - The update indices need to be contiguous
// -- The update indices are the last dim in indices
// -- We call them contiguous, if each idx in indices is indexing the element
//    in data, that is logically directly after the element indexed by the
//    previous idx
// --- logically directly after means the element that will be accessed if
//     the least significant value of an elements index is increased by one
// - The update indices need to cover/index the complete data, with the
//   exception of dimension 'a', where they need to cover only updates[a]
struct DecomposeScatterNDPattern : public OpRewritePattern<ONNXScatterNDOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXScatterNDOp scatterNDOp, PatternRewriter &rewriter) const final {
    // Check preconditions
    if (scatterNDOp.getReductionAttr().strref() != "none") {
      return rewriter.notifyMatchFailure(
          scatterNDOp, "Scatters with reduction are not supported");
    }
    const auto data = scatterNDOp.getData();
    const auto indices = scatterNDOp.getIndices();
    const auto updates = scatterNDOp.getUpdates();
    if (!onnx_mlir::hasStaticShape(data.getType()) ||
        !onnx_mlir::hasStaticShape(indices.getType()) ||
        !onnx_mlir::hasStaticShape(updates.getType())) {
      return rewriter.notifyMatchFailure(
          scatterNDOp, "All operands need to have a static shape");
    }
    const auto dataType = cast<RankedTensorType>(data.getType());
    const auto dataShape = dataType.getShape();
    const auto updatesType = cast<RankedTensorType>(updates.getType());
    const auto updateShape = updatesType.getShape();
    const auto indicesType = cast<RankedTensorType>(indices.getType());
    const auto indicesShape = indicesType.getShape();
    if (dataType.getRank() != updatesType.getRank()) {
      return rewriter.notifyMatchFailure(scatterNDOp,
          "Only the case where data and update have the same rank "
          "is supported");
    }

    const auto splitAxis = [&]() -> uint64_t {
      // Split at the dim where the update and original data have a
      // different size
      for (auto [idx, dimData, dimUpdates] :
          llvm::enumerate(dataShape, updateShape)) {
        if (dimData != dimUpdates) {
          return idx;
        }
      }
      return dataType.getRank() -
             1; // Edge case, all elements get updated, split on the last dim
    }();

    for (auto [idx, dimData, dimUpdates] :
        llvm::enumerate(dataShape, updateShape)) {
      if (idx != splitAxis && dimData != dimUpdates) {
        return rewriter.notifyMatchFailure(
            scatterNDOp, "Only a single differing dimension is supported");
      }
    }

    SmallVector<int64_t> indicesAsFlatArray;
    if (!onnx_mlir::getI64ValuesFromONNXConstantOp(
            indices, indicesAsFlatArray)) {
      return rewriter.notifyMatchFailure(
          scatterNDOp, "The indices need to be constant");
    }
    if (indicesAsFlatArray.empty()) {
      return rewriter.notifyMatchFailure(
          scatterNDOp, "Empty indices are not supported"); // Skip the edge case
                                                           // of empty indices
    }
    const auto indicesLastDimSize = indicesShape.back();
    SubArrayAccessHelper<int64_t> indicesFlatAccessor(
        indicesAsFlatArray, indicesLastDimSize);
    const auto firstIndex =
        indicesFlatAccessor[0]; // Safe, we have checked the length before
    for (auto [idx, firstIndexDim] : llvm::enumerate(firstIndex)) {
      if (idx != splitAxis && firstIndexDim != 0) {
        return rewriter.notifyMatchFailure(
            scatterNDOp, " Shifting is only supported on the split axis");
      }
      if (idx == splitAxis && firstIndexDim < 0) {
        return rewriter.notifyMatchFailure(scatterNDOp,
            "Negative values with wrap around are not yet "
            "supported"); // onnx allows negative values with
                          // wrap-around, this decomposition does
                          // not (for now)
      }
    }

    // Check that all indices are contiguous.
    // - The check for contiguity and covering works the following way:
    // -- Iterated over all idx in indices and compare the idx against the
    //    expected index, fail if it differs
    // -- The expected index is calculated the following way:
    // --- The expected index is initialized with the first index in indices and
    //     then always incremented by one.
    // --- The increment works like a manual addition, the least significant
    //     digit/subindex gets incremented by one. If a digit overflows, it
    //     gets reset to the first index and the addition carries to the next,
    //     more significant digit. The addition overflows, if the index for an
    //     axis is equal to the size of this axis in updates/indices. (By
    //     definition the shape for indices.shape().drop(-1) must match the
    //     first dimensions in updates). If the addition overflows , the
    //     overflowing digit is reset to its value in the first index. This is
    //     zero for all axes, except for 'a', where it can be a positive number
    //     if the split/concat is in the middle of the tensor
    assert(
        updateShape.drop_back(updateShape.size() - (indicesShape.size() - 1)) ==
            indicesShape.drop_back(1) &&
        "Update and indicesShape should partially match for scatterNd");
    {
      IndicesContiguousCounter counter(firstIndex, indicesShape.drop_back(1));
      for (size_t i = 0; i < indicesFlatAccessor.size(); ++i) {
        if (counter.getCounter() != indicesFlatAccessor[i]) {
          return rewriter.notifyMatchFailure(
              scatterNDOp, "Indices are not contiguous");
        }
        counter.increment();
      }
    }

    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, scatterNDOp->getLoc());
    // Strategy for the decomposition:
    // Split at the split axis, concat the update and part of the split
    // a, b = split(input)
    // a1, a2 = split(a)
    // concat(a1, update, b)
    // In onnx this split can be done in one:
    // a1, a2, b = split(input)
    const auto firstSplitPosition =
        (splitAxis < firstIndex.size()) ? firstIndex[splitAxis] : 0;
    const auto secondSplitPosition =
        updateShape[splitAxis] + firstSplitPosition;
    SmallVector<int64_t> splitTyFirstQuarter(dataShape);
    splitTyFirstQuarter[splitAxis] = firstSplitPosition;
    SmallVector<int64_t> splitTySecondQuarter(dataShape);
    splitTySecondQuarter[splitAxis] = updateShape[splitAxis];
    SmallVector<int64_t> splitTySecondHalf(dataShape);
    splitTySecondHalf[splitAxis] -= secondSplitPosition;
    Value splitSize = create.onnx.constantInt64({firstSplitPosition,
        updateShape[splitAxis], splitTySecondHalf[splitAxis]});
    const Type dataElementType = dataType.getElementType();
    ValueRange split = create.onnx.split(
        {RankedTensorType::get(splitTyFirstQuarter, dataElementType),
            RankedTensorType::get(splitTySecondQuarter, dataElementType),
            RankedTensorType::get(splitTySecondHalf, dataElementType)},
        scatterNDOp.getData(), splitSize, splitAxis);

    Value concat = create.onnx.concat(
        dataType, {split[0], scatterNDOp.getUpdates(), split[2]}, splitAxis);
    rewriter.replaceOp(scatterNDOp, concat);
    return success();
  }
};

// Decompose the custom op FusedMatMul that is produced by ONNXRuntime.
// According to FusedMatMul specification, it is the result of fusing MatMul and
// Transpose:
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
//
// To decompose FusedMatMul, we need to know ranks of inputs A and B, so that
// we can emit Transpose operations. But, in general, we have no information
// about the ranks of A and B.
//
// The rewriting here only applies to a situation in which the transposed input
// comes from another Transpose that we have rank information via looking at
// `perm` // attribute. For example, if `transA = 1`, A must be from a Transpose
// to determine the rank of A.
//
// Example of onnx.Custom:
//  ```
// "onnx.Custom"(%0, %1) {alpha = 1.250000e-01 : f32,
//                        domain_name = "com.microsoft",
//                        function_name = "FusedMatMul",
//                        transA = 0 : si64, transB = 1 : si64} :
//              (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// ```

struct CustomOpFuseMatMulPattern : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern<ONNXCustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Location loc = customOp.getLoc();

    // Match
    FloatAttr alphaAttr;
    int64_t rankA, rankB;
    if (!isCustomOpFusedMatMulMatched(customOp, alphaAttr, rankA, rankB))
      return failure();

    // Rewrite ONNXCustomOp {alpha} (A, B) into `Mul(alpha, MatMul(A, B)`
    Value A = customOp.getOperands()[0];
    Value B = customOp.getOperands()[1];

    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Type resType = customOp.getResult(0).getType();
    Type elementType = onnx_mlir::getElementType(resType);
    UnrankedTensorType unrankedType = UnrankedTensorType::get(elementType);

    Value matmulA = A;
    Value matmulB = B;
    // Transpose A if transA.
    if (rankA != -1) {
      // Prepare permutation attribute.
      SmallVector<int64_t, 4> indices;
      for (int64_t i = 0; i < rankA - 2; ++i)
        indices.emplace_back(i);
      // Permute the last two dimensions.
      indices.emplace_back(rankA - 1);
      indices.emplace_back(rankA - 2);
      ArrayAttr permAttr = rewriter.getI64ArrayAttr(llvm::ArrayRef(indices));
      matmulA = create.onnx.transpose(unrankedType, A, permAttr);
    }
    // Transpose B if transB.
    if (rankB != -1) {
      // Prepare permutation attribute.
      SmallVector<int64_t, 4> indices;
      for (int64_t i = 0; i < rankB - 2; ++i)
        indices.emplace_back(i);
      // Permute the last two dimensions.
      indices.emplace_back(rankB - 1);
      indices.emplace_back(rankB - 2);
      ArrayAttr permAttr = rewriter.getI64ArrayAttr(llvm::ArrayRef(indices));
      matmulB = create.onnx.transpose(unrankedType, B, permAttr);
    }
    // alpha
    DenseElementsAttr alphaDenseAttr =
        onnx_mlir::createDenseElementsAttrFromFloatAttr(
            rewriter, elementType, alphaAttr);
    Value alpha = create.onnx.constant(alphaDenseAttr);

    Value res = create.onnx.matmul(resType, matmulA, matmulB);
    res = create.onnx.mul(alpha, res);

    rewriter.replaceOp(customOp, res);
    return success();
  }

public:
  static bool isCustomOpFusedMatMulMatched(ONNXCustomOp customOp,
      FloatAttr &alphaAttr, int64_t &rankA, int64_t &rankB) {
    Operation *genericOp = customOp.getOperation();
    // CustomOp has two operands.
    if (customOp.getNumOperands() != 2)
      return false;
    Value A = genericOp->getOperands()[0];
    Value B = genericOp->getOperands()[1];

    // function_name is FusedMatMul.
    StringRef funcName = customOp.getFunctionName();
    if (!funcName.equals_insensitive("FusedMatMul"))
      return false;

    // domain_name exists and is "com.microsoft";
    StringAttr domAttr = genericOp->getAttrOfType<StringAttr>("domain_name");
    if (!domAttr)
      return false;
    if (!domAttr.getValue().equals_insensitive("com.microsoft"))
      return false;

    // transA and transB exist.
    IntegerAttr transA = genericOp->getAttrOfType<IntegerAttr>("transA");
    IntegerAttr transB = genericOp->getAttrOfType<IntegerAttr>("transB");
    if (!transA || !transB)
      return false;
    bool isTransA = (transA.getValue().getSExtValue() == 1);
    bool isTransB = (transB.getValue().getSExtValue() == 1);

    // If transA=true, we have to know A's rank to generate ONNXTransposeOp for
    // A. In a good condition, A is ranked then its rank is available.
    //
    // If A is unranked, we hope that A is a result of another ONNXTransposeOp
    // whose permutation is available and can be used to infer the rank of A.
    // For example,
    // %A = "onnx.Transpose"(%0) {perm = [0, 2, 1, 3]} :
    //                      (tensor<*xf32>) -> tensor<*xf32>
    // A must have rank 4 as perm has 4 indices.
    if (isTransA) {
      if (onnx_mlir::hasShapeAndRank(A)) {
        rankA = mlir::cast<ShapedType>(A.getType()).getRank();
      } else {
        if (isa<BlockArgument>(A))
          return false;
        if (auto transOp = dyn_cast<ONNXTransposeOp>(A.getDefiningOp())) {
          if (transOp.getPermAttr())
            rankA = transOp.getPermAttr().size();
          else
            return false;
        } else
          // Cannot determine the rank of A.
          return false;
      }
    } else
      rankA = -1;
    if (isTransB) {
      if (onnx_mlir::hasShapeAndRank(B)) {
        rankB = mlir::cast<ShapedType>(B.getType()).getRank();
      } else {
        if (isa<BlockArgument>(B))
          return false;
        if (auto transOp = dyn_cast<ONNXTransposeOp>(B.getDefiningOp())) {
          if (transOp.getPermAttr())
            rankB = transOp.getPermAttr().size();
          else
            return false;
        } else
          // Cannot determine the rank of B.
          return false;
      }
    } else
      rankB = -1;

    // Get alpha.
    alphaAttr = genericOp->getAttrOfType<FloatAttr>("alpha");
    if (!alphaAttr)
      return false;

    // CustomOp is in a good form to rewrite.
    return true;
  }
};

namespace {

[[nodiscard]] bool isCustomMicrosoftOp(
    ONNXCustomOp customOp, StringRef expectedName) {
  if (!customOp.getFunctionName().equals_insensitive(expectedName)) {
    return false;
  }

  const auto domAttr = customOp->getAttrOfType<StringAttr>("domain_name");
  return domAttr && domAttr.getValue().equals_insensitive("com.microsoft");
}

} // namespace

struct CustomOpMicrosoftToOnnxOps : public OpRewritePattern<ONNXCustomOp> {
  CustomOpMicrosoftToOnnxOps(MLIRContext *context,
      std::string operationNameToRewrite, PatternBenefit benefit = 1)
      : OpRewritePattern<ONNXCustomOp>(context, benefit),
        operationNameToRewrite(std::move(operationNameToRewrite)) {}

  LogicalResult matchAndRewrite(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const final {
    if (!isCustomMicrosoftOp(customOp, operationNameToRewrite)) {
      return failure();
    }

    return matchAndRewriteImpl(customOp, rewriter);
  }

  virtual LogicalResult matchAndRewriteImpl(
      ONNXCustomOp /*customOp*/, PatternRewriter & /*rewriter*/) const {
    return failure();
  }

  static LogicalResult verifyOpValidity(Operation *op) {
    assert(op);
    onnx_mlir::IgnoreDiagnostic diag(op->getContext()->getDiagEngine());
    if (auto info = op->getName().getRegisteredInfo()) {
      return info->verifyInvariants(op);
    }
    return mlir::verify(op);
  }

  static LogicalResult verifyOpsErasingOnError(
      ValueRange values, PatternRewriter &rewriter) {
    if (llvm::any_of(values, [](Value value) {
          return value && failed(verifyOpValidity(value.getDefiningOp()));
        })) {
      for (auto value : values)
        if (value) {
          rewriter.eraseOp(value.getDefiningOp());
        }
      return failure();
    }
    return success();
  }

  static SmallVector<NamedAttribute> getFilteredAttrs(
      ArrayRef<NamedAttribute> attrs,
      ArrayRef<StringRef> additionalAttrNamesToFilter = {}) {
    static const llvm::StringSet<> commonFilter{"domain_name", "function_name",
        "output_element_type", "shape_infer_pattern", "inputs_for_infer"};
    return SmallVector<NamedAttribute>{llvm::make_filter_range(
        attrs, [&additionalAttrNamesToFilter](NamedAttribute attr) {
          return !llvm::is_contained(commonFilter, attr.getName()) &&
                 !llvm::is_contained(
                     additionalAttrNamesToFilter, attr.getName());
        })};
  }

  const std::string operationNameToRewrite;
};

struct MicrosoftBiasGelu : public CustomOpMicrosoftToOnnxOps {
  MicrosoftBiasGelu(MLIRContext *context, PatternBenefit benefit = 1)
      : CustomOpMicrosoftToOnnxOps(context, "BiasGelu", benefit) {}

  LogicalResult matchAndRewriteImpl(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    assert(customOp->getNumOperands() == 2 &&
           "Expected two operands for BiasGelu");

    auto input = customOp->getOperand(0);
    auto bias = customOp->getOperand(1);
    MultiDialectBuilder<OnnxBuilder> create(rewriter, customOp->getLoc());
    Value biasedInput = create.onnx.add(input, bias);
    Value gelu = create.onnx.gelu(biasedInput,
        /*approximateAttr=*/rewriter.getStringAttr("none"));
    if (failed(verifyOpsErasingOnError({biasedInput, gelu}, rewriter))) {
      return rewriter.notifyMatchFailure(customOp, "Failed verification");
    }

    rewriter.replaceOp(customOp, gelu);
    return success();
  }
};

struct MicrosoftFusedConv : public CustomOpMicrosoftToOnnxOps {
  MicrosoftFusedConv(MLIRContext *context, PatternBenefit benefit = 1)
      : CustomOpMicrosoftToOnnxOps(context, "FusedConv", benefit) {}

  LogicalResult matchAndRewriteImpl(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    assert(customOp.getNumOperands() >= 2 && customOp.getNumOperands() <= 4 &&
           "Expected 2 to 4 operands for FusedConv");
    if (customOp.getNumOperands() > 3) {
      return rewriter.notifyMatchFailure(
          customOp, "Decomposition does not support 'Sum/Z'");
    }

    assert(customOp->hasAttrOfType<StringAttr>("activation"));
    assert(customOp->hasAttrOfType<ArrayAttr>("activation_params"));

    const SmallVector<NamedAttribute> filteredAttrs(getFilteredAttrs(
        customOp->getAttrs(), {"activation", "activation_params"}));
    SmallVector<Value> convOperands{customOp.getOperands()};
    Value noneBias;
    if (convOperands.size() < 3) {
      noneBias = rewriter.create<ONNXNoneOp>(customOp->getLoc())->getResult(0);
      convOperands.push_back(noneBias);
    }

    auto conv = rewriter.create<ONNXConvOp>(customOp->getLoc(),
        customOp->getResultTypes(), convOperands, filteredAttrs);
    Value convOpResult = conv.getResult();
    const auto activation =
        customOp->getAttrOfType<StringAttr>("activation").strref();
    auto activationParams =
        customOp->getAttrOfType<ArrayAttr>("activation_params");
    SmallVector<FloatAttr> activationParamsValues;
    for (auto attr : activationParams) {
      auto asFloatAttr = dyn_cast<FloatAttr>(attr);
      assert(asFloatAttr && asFloatAttr.getType().isF32() &&
             "All activation params "
             "must be f32");
      activationParamsValues.push_back(asFloatAttr);
    }
    Value activationFunc;
    Value castMin;
    Value castMax;
    if (activation == "Relu") {
      activationFunc = rewriter.create<ONNXReluOp>(
          customOp->getLoc(), convOpResult.getType(), convOpResult);
    } else if (activation == "Tanh") {
      activationFunc = rewriter.create<ONNXTanhOp>(
          customOp->getLoc(), convOpResult.getType(), convOpResult);
    } else if (activation == "Sigmoid") {
      activationFunc = rewriter.create<ONNXSigmoidOp>(
          customOp->getLoc(), convOpResult.getType(), convOpResult);
    } else if (activation == "LeakyRelu") {
      assert(activationParamsValues.size() == 1 &&
             "LeakyRelu must have exactly one parameter");
      activationFunc = rewriter.create<ONNXLeakyReluOp>(customOp->getLoc(),
          convOpResult.getType(), convOpResult, activationParamsValues[0]);
    } else if (activation == "Clip") {
      assert(activationParamsValues.size() == 2 &&
             "Clip must have exactly two parameters");
      MultiDialectBuilder<OnnxBuilder> create(rewriter, customOp->getLoc());
      auto scalarType = RankedTensorType::get({}, rewriter.getF32Type());
      auto minVal = create.onnx.constant(
          DenseElementsAttr::get(scalarType, activationParamsValues[0]));
      auto castToType =
          cast<ShapedType>(convOpResult.getType()).getElementType();
      castMin = create.onnx.cast(minVal, castToType);
      auto maxVal = create.onnx.constant(
          DenseElementsAttr::get(scalarType, activationParamsValues[1]));
      castMax = create.onnx.cast(maxVal, castToType);
      activationFunc = rewriter.create<ONNXClipOp>(customOp->getLoc(),
          convOpResult.getType(), convOpResult, castMin, castMax);
    } else if (activation == "HardSigmoid") {
      assert(activationParamsValues.size() == 2 &&
             "HardSigmoid must have exactly two parameters");
      activationFunc = rewriter.create<ONNXHardSigmoidOp>(customOp->getLoc(),
          convOpResult.getType(), convOpResult, activationParamsValues[0],
          activationParamsValues[1]);
    } else {
      rewriter.eraseOp(conv);
      if (noneBias) {
        rewriter.eraseOp(noneBias.getDefiningOp());
      }
      return rewriter.notifyMatchFailure(customOp,
          "Decomposition only supports Relu, Tanh, Sigmoid, LeakyRelu, Clip, "
          "and HardSigmoid activations");
    }
    if (failed(verifyOpsErasingOnError(
            {noneBias, conv, castMin, castMax, activationFunc}, rewriter))) {
      return rewriter.notifyMatchFailure(customOp, "Failed verification");
    }
    rewriter.replaceOp(customOp, activationFunc);
    return success();
  }
};

template <typename OpToCreate>
struct CustomOpMicrosoftToSingleOnnxOp : public CustomOpMicrosoftToOnnxOps {
  using CustomOpMicrosoftToOnnxOps::CustomOpMicrosoftToOnnxOps;

  LogicalResult matchAndRewriteImpl(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const final {
    if (failed(shouldBeRewritten(customOp, rewriter))) {
      return failure();
    }

    const SmallVector<NamedAttribute> filteredAttrs(
        getFilteredAttrs(customOp->getAttrs()));

    auto newOp = rewriter.create<OpToCreate>(customOp->getLoc(),
        customOp->getResultTypes(), customOp.getOperands(), filteredAttrs);

    postProcess(customOp, newOp, rewriter);

    if (failed(verifyOpsErasingOnError({newOp}, rewriter))) {
      return rewriter.notifyMatchFailure(customOp, "Failed verification");
    }
    rewriter.replaceOp(customOp, newOp);
    return success();
  }

  virtual LogicalResult shouldBeRewritten(
      ONNXCustomOp /*customOp*/, PatternRewriter & /*rewriter*/) const {
    return success();
  }

  virtual void postProcess(const ONNXCustomOp & /*customOp*/,
      OpToCreate & /*newOP*/, PatternRewriter & /*rewriter*/) const {}
};

template <typename OpToCreate>
struct CustomOpMicrosoftQDquantizeLinear
    : CustomOpMicrosoftToSingleOnnxOp<OpToCreate> {
  using CustomOpMicrosoftToSingleOnnxOp<
      OpToCreate>::CustomOpMicrosoftToSingleOnnxOp;

  LogicalResult shouldBeRewritten(
      ONNXCustomOp customOp, PatternRewriter &rewriter) const override {
    using namespace onnx_mlir;
    // Check if the input is a quantized type.
    if (customOp->getNumOperands() != 3) {
      return failure();
    }

    const auto scale = customOp->getOperand(1);
    const auto zeroPoint = customOp->getOperand(2);
    const auto isScalarOr1dTensor = [](Value v) {
      auto shapedType = dyn_cast<ShapedType>(v.getType());
      return shapedType && shapedType.hasRank() &&
             (shapedType.getRank() == 0 || (shapedType.getRank() == 1));
    };
    if (!isScalarOr1dTensor(scale) || !isScalarOr1dTensor(zeroPoint)) {
      return rewriter.notifyMatchFailure(customOp,
          "Only supports per-tensor or per-layer quantization for now");
    }
    return success();
  }

  void postProcess(const ONNXCustomOp &customOp, OpToCreate &newOP,
      PatternRewriter & /*rewriter*/) const override {
    if (customOp->hasAttr("axis")) {
      newOP.setAxisAttr(customOp->getAttrOfType<IntegerAttr>("axis"));
    }
  }
};

/// Decompose `onnx.SoftmaxCrossEntropyLoss` to the following sequence:
/// In the following we assume classes is in dim=1 of scores.
/// 1. one_hot_encoded = onnx.Castlike(onnx.OneHot(labels, dim=1), scores)
/// 2. log_softmax = onnx.Log(onnx.Softmax(scores, dim=1))
/// 3. product = onnx.Mul(log_softmax, one_hot_encoded)
///    if `weights` arg is nont `none` then we additionally perform
///    product = onnx.Mul(product, op.Unsqueeze(weights))
///    where unsqueezing makes the operation broadcastable.
/// 4. reduce_sum = onnx.ReduceSum(product, dim=1)
/// 5. loss = onnx.ReduceMean(reduce_sum) if reduciton == "mean"
///           onnx.ReduceSum(reduce_sum)  if reduction == "sum"
///           onnx.Squeeze(reduce_sum)    if reduciton == "none"
///
struct SoftmaxCrossEntropyPattern
    : public OpRewritePattern<ONNXSoftmaxCrossEntropyLossOp> {
  using OpRewritePattern<ONNXSoftmaxCrossEntropyLossOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXSoftmaxCrossEntropyLossOp sceOp,
      PatternRewriter &rewriter) const final {
    auto loc = sceOp.getLoc();
    onnx_mlir::OnnxBuilder create(rewriter, loc);
    auto scores = sceOp.getScores();
    auto labels = sceOp.getLabels();
    auto weights = sceOp.getWeights();
    auto scoresTy = cast<ShapedType>(scores.getType());
    auto labelsTy = cast<ShapedType>(labels.getType());
    SmallVector<int64_t> newLabelsShape(labelsTy.getShape());
    newLabelsShape.insert(newLabelsShape.begin() + 1, scoresTy.getShape()[1]);
    auto none = rewriter.create<ONNXNoneOp>(loc);
    auto numClasses = (scoresTy.isDynamicDim(1))
                          ? create.dim(scores, 1)
                          : create.constantInt64({scoresTy.getShape()[1]});
    auto elemTy = scoresTy.getElementType();
    // Compute one hot encoded labels and cast to `scores` element type.
    auto oneHotValsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{0, 1});
    auto oneHotVals = create.constant(oneHotValsAttr);
    auto oneHot = create.cast(
        rewriter.create<ONNXOneHotOp>(loc,
            RankedTensorType::get(newLabelsShape, labelsTy.getElementType()),
            labels, numClasses, oneHotVals, /*axis=*/1),
        /*saturate=*/
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1),
        TypeAttr::get(elemTy));
    // Compute logsoftmax of scores.
    auto softmax =
        rewriter.create<ONNXSoftmaxOp>(loc, scoresTy, scores, /*axis=*/1);
    auto logSoftmax = rewriter.create<ONNXLogOp>(loc, scoresTy, softmax);
    auto prod = rewriter.create<ONNXMulOp>(loc, logSoftmax, oneHot);
    // Multiply by `weights` if not none.
    if (auto weightTy = dyn_cast<ShapedType>(weights.getType())) {
      // Unsqueeze weight from [C] to [1 x C x 1 x ... x 1] to make it
      // broadcast-compliant.
      llvm::SmallVector<int64_t, 4> unsqueezedShape(scoresTy.getRank(), 1);
      unsqueezedShape[1] = scoresTy.getShape()[1];
      llvm::SmallVector<int64_t, 4> axesList(scoresTy.getRank() - 1, 0);
      std::iota(axesList.begin() + 1, axesList.end(), 2);
      auto axes = create.constantInt64(axesList);
      auto weightsUnsqueezed = create.unsqueeze(
          RankedTensorType::get(unsqueezedShape, elemTy), weights, axes);
      prod = rewriter.create<ONNXMulOp>(loc, prod, weightsUnsqueezed);
    }
    // Reduction across `class` (dim=1) axis.
    auto axes = create.constant(onnx_mlir::createDenseArrayAttr(
        rewriter, rewriter.getI64ArrayAttr({1})));
    auto reducedType = createReducedType(scoresTy, 1, /*keepdims=*/true);
    Value loss = rewriter.create<ONNXReduceSumOp>(loc, reducedType, prod, axes);
    // ReduceMean/ReduceSum/Squeeze if reduction = mean/sum/none respectively.
    // Set `axes=none` to indicate reducing all dims.
    auto reduction = cast<StringAttr>(sceOp.getReductionAttr()).getValue();
    if (reduction == "mean") {
      if (isa<NoneType>(weights.getType())) {
        loss = rewriter.create<ONNXReduceMeanOp>(loc,
            RankedTensorType::get({}, elemTy), loss, none,
            /*keepdims=*/0);
      } else {
        auto sumL = rewriter.create<ONNXReduceSumOp>(loc,
            RankedTensorType::get({}, elemTy), loss, none,
            /*keepdims=*/0);
        // Perform einsum(one_hot, weights) as a simple way of producing
        // W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]
        auto scatteredWeights = rewriter.create<ONNXEinsumOp>(loc,
            RankedTensorType::get(labelsTy.getShape(), elemTy),
            ValueRange{oneHot, weights}, "ij...,j->i...");
        auto sumW = rewriter.create<ONNXReduceSumOp>(loc,
            RankedTensorType::get({}, elemTy), scatteredWeights, none,
            /*keepdims=*/0);
        loss = rewriter.create<ONNXDivOp>(loc, sumL, sumW);
      }
    } else if (reduction == "sum") {
      loss = rewriter.create<ONNXReduceSumOp>(loc,
          RankedTensorType::get({}, elemTy), loss, none,
          /*keepdims=*/0);
    } else if (reduction == "none") {
      loss = rewriter.create<ONNXSqueezeOp>(loc,
          createReducedType(reducedType, 1, /*keepdims=*/false), loss, axes);
    } else {
      llvm_unreachable("unexpected reduction type");
    }
    // Negate.
    loss = rewriter.create<ONNXNegOp>(loc, loss.getType(), loss);
    // Second return value replacement depends if it is `none` or not.
    if (isa<NoneType>(sceOp.getLogProb().getType()))
      rewriter.replaceOp(sceOp, {loss, none});
    else
      rewriter.replaceOp(sceOp, {loss, logSoftmax});
    return success();
  }
};

/// Decompose `onnx.Sum` to a sequence of `onnx.Add`
struct SumToAddPattern : public OpRewritePattern<ONNXSumOp> {
  using OpRewritePattern<ONNXSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSumOp sumOp, PatternRewriter &rewriter) const final {
    SmallVector<Value> inputs(sumOp.getData_0());
    assert(inputs.size() > 0 && "expected at least one input");
    Value result = inputs[0];
    if (inputs.size() > 1) {
      inputs.erase(inputs.begin());
      for (auto input : inputs) {
        result = rewriter.create<ONNXAddOp>(sumOp.getLoc(), result, input);
      }
    }
    auto resultType = mlir::cast<ShapedType>(sumOp.getResult().getType());
    if (resultType != result.getType())
      result = rewriter.create<ONNXCastOp>(
          sumOp.getLoc(), resultType, result, 1, resultType.getElementType());
    rewriter.replaceOp(sumOp, result);
    return success();
  }
};

// =============================================================================
// Pattern for replacing CastLikeOp by CastOp.
// =============================================================================
// A pattern to turn
//   `CastLikeOp(input, saturate, targetLike)`
// into
//   `CastOp(input, saturate, targetType)`
class ReplaceCastLikeByCastPattern : public OpRewritePattern<ONNXCastLikeOp> {
public:
  using OpRewritePattern<ONNXCastLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXCastLikeOp castLikeOp, PatternRewriter &rewriter) const override {
    Location loc = castLikeOp.getLoc();

    Value input = castLikeOp.getInput();
    Value output = castLikeOp.getOutput();
    Value target = castLikeOp.getTargetType();
    IntegerAttr saturate = castLikeOp.getSaturateAttr();

    // The output type will be the same as the target_type or the second input
    Type targetType = mlir::cast<ShapedType>(target.getType()).getElementType();

    // Replace
    Value res;
    if (mlir::cast<ShapedType>(output.getType()).hasRank())
      res = onnx_mlir::OnnxBuilder(rewriter, loc)
                .cast(input, saturate, TypeAttr::get(targetType));
    else {
      Type resultType = UnrankedTensorType::get(targetType);
      res = onnx_mlir::OnnxBuilder(rewriter, loc)
                .cast(resultType, input, saturate, TypeAttr::get(targetType),
                    false);
    }
    rewriter.replaceOp(castLikeOp, res);
    return success();
  }
};

// =============================================================================
// Decompose Hardswish to simpler ONNX ops
// =============================================================================
// DecomposeHardSwishPattern replaces ONNXHardSwishOp with its equivalent
// mathematical decomposition using basic ONNX operations:
//
//    HardSwish(x) = x * max(0, min(1, (x / 6) + 0.5))
//
// This pass:
//  - Multiplies input by `1/6`
//  - Adds `0.5` to the scaled input
//  - Clamps the result between `0` and `1` using Min and Max ops
//  - Multiplies the clamped value with the original input

// AMD: Disabled, instead decompose to input * ONNXHardSigmoid
// struct DecomposeHardSwishPattern : public OpRewritePattern<ONNXHardSwishOp> {
//   using OpRewritePattern<ONNXHardSwishOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(
//       ONNXHardSwishOp hardswishOp, PatternRewriter &rewriter) const final {

//     // Get location and element type
//     Location loc = hardswishOp.getLoc();
//     onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
//         rewriter, loc);

//     Value alphaConst = create.onnx.constantFloat32(1.0f / 6.0f);
//     Value betaConst = create.onnx.constantFloat32(0.5f);
//     Value minConst = create.onnx.constantFloat32(1.0f);
//     Value maxConst = create.onnx.constantFloat32(0.0f);

//     // Multiply input by alpha
//     auto scaledInput =
//         rewriter.create<ONNXMulOp>(loc, hardswishOp.getOperand().getType(),
//             hardswishOp.getOperand(), alphaConst);

//     // Add beta to (input * alpha)
//     auto shiftedInput = rewriter.create<ONNXAddOp>(
//         loc, scaledInput.getType(), scaledInput, betaConst);

//     // Compute min(1.0, shiftedInput)
//     auto minOp = rewriter.create<ONNXMinOp>(
//         loc, shiftedInput.getType(), ValueRange({shiftedInput, minConst}));

//     // Compute max(0, min(1, shiftedInput))
//     auto maxOp = rewriter.create<ONNXMaxOp>(
//         loc, minOp.getType(), ValueRange({minOp, maxConst}));

//     // Compute final HardSwish: input * max(0, min(1, add(mul(x, alpha),
//     beta))) auto hardswishResult = rewriter.create<ONNXMulOp>(loc,
//         hardswishOp.getOperand().getType(), hardswishOp.getOperand(), maxOp);

//     // Replace the original HardSwishOp with the new computation
//     rewriter.replaceOp(hardswishOp, hardswishResult.getResult());
//     return success();
//   }
// };

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeONNXToONNXPass)

  DecomposeONNXToONNXPass(const std::string &target,
      bool enableConvTransposeDecompose = false,
      bool enableConvTransposeDecomposeToPhasedConv = false,
      bool enableConvTranspose1dDecomposeToPhasedConv = false) {
    this->target = target;
    this->enableConvTransposeDecompose = enableConvTransposeDecompose;
    this->enableConvTransposeDecomposeToPhasedConv =
        enableConvTransposeDecomposeToPhasedConv;
    this->enableConvTranspose1dDecomposeToPhasedConv =
        enableConvTranspose1dDecomposeToPhasedConv;
  }

  DecomposeONNXToONNXPass(const DecomposeONNXToONNXPass &pass)
      : mlir::PassWrapper<DecomposeONNXToONNXPass,
            OperationPass<func::FuncOp>>() {
    this->target = pass.target.getValue();
    this->enableConvTransposeDecompose =
        pass.enableConvTransposeDecompose.getValue();
    this->enableConvTransposeDecomposeToPhasedConv =
        pass.enableConvTransposeDecomposeToPhasedConv.getValue();
  }

  StringRef getArgument() const override { return "decompose-onnx"; }

  StringRef getDescription() const override {
    return "Decompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  Option<std::string> target{*this, "target",
      llvm::cl::desc("Target Dialect to decompose into"), ::llvm::cl::init("")};

  Option<bool> enableConvTransposeDecompose{*this, "enable-convtranspose",
      llvm::cl::desc("Enable decomposition of ConvTranspose"),
      ::llvm::cl::init(false)};

  Option<bool> enableConvTransposeDecomposeToPhasedConv{*this,
      "enable-convtranspose-phased",
      llvm::cl::desc("Enable decomposition of ONNX ConvTranspose operator to 4 "
                     "phased Conv"),
      ::llvm::cl::init(false)};

  Option<bool> enableConvTranspose1dDecomposeToPhasedConv{*this,
      "enable-convtranspose-1d-phased",
      llvm::cl::desc(
          "Enable decomposition of ONNX ConvTranspose 1D operator to "
          "phased Conv"),
      ::llvm::cl::init(false)};

  void runOnOperation() final;

  typedef PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>>
      BaseType;
};

void DecomposeONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  onnx_mlir::getDecomposeONNXToONNXPatterns(patterns,
      enableConvTransposeDecompose, enableConvTransposeDecomposeToPhasedConv,
      enableConvTranspose1dDecomposeToPhasedConv);
  patterns.insert<ReplaceCastLikeByCastPattern>(context);

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  if (this->target == "stablehlo") {
    populateDecomposingONNXBeforeStablehloPatterns(patterns, context);
  }
#endif

  if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getDecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns, bool enableConvTransposeDecompose,
    bool enableConvTransposeDecomposeToPhasedConv,
    bool enableConvTranspose1dDecomposeToPhasedConv) {
  MLIRContext *context = patterns.getContext();
  populateWithGenerated(patterns);
  if (enableConvTransposeDecompose)
    convtranspose::populateWithGenerated(patterns);
  if (enableConvTransposeDecomposeToPhasedConv)
    convtranspose_phased::populateWithGenerated(patterns);
  if (enableConvTranspose1dDecomposeToPhasedConv)
    convtranspose_1d_phased::populateWithGenerated(patterns);
  patterns.insert<onnx_mlir::DecomposeEinsumPattern>(context);
  patterns.insert<ConcatFusePattern>(context);
  patterns.insert<DecomposeHardSwishPattern>(context);
  // Decompose CustomOp FusedMatMul introduced by onnxruntime:
  // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
  patterns.insert<CustomOpFuseMatMulPattern>(context);
  patterns.insert<CustomOpMicrosoftQDquantizeLinear<ONNXQuantizeLinearOp>>(
      context, "QuantizeLinear");
  patterns.insert<CustomOpMicrosoftQDquantizeLinear<ONNXDequantizeLinearOp>>(
      context, "DequantizeLinear");
  patterns.insert<CustomOpMicrosoftToSingleOnnxOp<ONNXGeluOp>>(context, "Gelu");
  patterns.insert<MicrosoftBiasGelu>(context);
  patterns.insert<MicrosoftFusedConv>(context);
  patterns.insert<DecomposeSlicePadPattern>(context);
  patterns.insert<DecomposeScatterNDPattern>(context);
  patterns.insert<SoftmaxCrossEntropyPattern>(context);
  patterns.insert<SumToAddPattern>(context);

  //   for (const auto &op : onnx_mlir::decomposeOpsInONNX) {
  //     if (op == "HardSwish") {
  //       patterns.insert<DecomposeHardSwishPattern>(context);
  //     }
  //   }
  // }

  // TODO: consider whether to include SoftmaxPattern here
}

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createDecomposeONNXToONNXPass(
    const std::string &target, bool enableConvTransposeDecompose,
    bool enableConvTransposeDecomposeToPhasedConv,
    bool enableConvTranspose1dDecomposeToPhasedConv) {
  return std::make_unique<DecomposeONNXToONNXPass>(target,
      enableConvTransposeDecompose, enableConvTransposeDecomposeToPhasedConv,
      enableConvTranspose1dDecomposeToPhasedConv);
}
