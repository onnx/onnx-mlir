/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
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

#include <numeric>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/Decompose.hpp"
#include "src/Dialect/ONNX/Transforms/DecomposeEinsum.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

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
  Value position = ONNXNoneOp::create(rewriter, loc);

  for (auto input : inputs)
    seq = ONNXSequenceInsertOp::create(
        rewriter, loc, resType, seq, input, position);

  return seq;
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
//   start: %position*%split for %axis, 0 for other dimensions
//   end: (%position+1)*%split for %axis, upper bound for other dimension
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
  if (stringMode.strref() == "trilinear") {
    return rewriter.getStringAttr("linear");
  }
  // Mode is already in new format (linear, cubic, nearest) or is nearest
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

// New ConvTranspose decomposition pattern following the approach in
// conv-trans-with-group-v4.py. This pattern decomposes ConvTranspose into:
// 1. UpsampleAndPad for zero insertion and padding
// 2. Weight transformation (reshape, flip, permute)
// 3. Regular Conv operation
//
// For detailed explanation of this decomposition approach, see:
// docs/optimization-onnx-lowering/ConvTranspose.md
struct DecomposeConvTransposePattern
    : public OpRewritePattern<ONNXConvTransposeOp> {
  using OpRewritePattern<ONNXConvTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvTransposeOp convTransposeOp,
      PatternRewriter &rewriter) const final {
    Location loc = convTransposeOp.getLoc();
    Value X = convTransposeOp.getX();
    Value W = convTransposeOp.getW();
    Value B = convTransposeOp.getB();

    // Get input and weight types.
    ShapedType xType = mlir::cast<ShapedType>(X.getType());
    ShapedType wType = mlir::cast<ShapedType>(W.getType());

    if (!xType.hasRank() || !wType.hasRank())
      return failure();

    ArrayRef<int64_t> wShape = wType.getShape();

    // Check that all weight dimensions are known at compile time.
    for (int64_t dim : wShape) {
      if (ShapedType::isDynamic(dim))
        return failure();
    }

    // Get attributes.
    ONNXConvTransposeOpShapeHelper shapeHelper(
        convTransposeOp.getOperation(), {});
    if (failed(shapeHelper.computeShape()))
      return failure();

    // Check if output_shape is provided.
    std::optional<ArrayAttr> outputShapeOpt = convTransposeOp.getOutputShape();

    // When output_shape is provided, input spatial dimensions must be known at
    // compile time to compute pads.
    if (outputShapeOpt.has_value()) {
      ArrayRef<int64_t> xShape = xType.getShape();
      // Check spatial dimensions (skip batch and channel dimensions).
      for (size_t i = 2; i < xShape.size(); ++i) {
        if (ShapedType::isDynamic(xShape[i])) {
          return failure();
        }
      }
    }

    // Get attributes from shape helper and op.
    auto strides = shapeHelper.strides;
    auto dilations = shapeHelper.dilations;
    DimsExpr pads = shapeHelper.pads;
    auto outputPadding = shapeHelper.outputPadding;
    int64_t group = convTransposeOp.getGroup();
    auto kernelShape = shapeHelper.kernelShape;
    StringRef autoPad = convTransposeOp.getAutoPad();

    if (!IndexExpr::isLiteral(kernelShape))
      return failure();

    int64_t spatialRank = strides.size();

    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);

    // Compute pads based on auto_pad mode or output_shape following Python
    // reference.
    SmallVector<int64_t, 4> padTop(spatialRank);
    SmallVector<int64_t, 4> padBottom(spatialRank);

    // When output_shape is provided, compute pads from it.
    if (outputShapeOpt.has_value()) {
      // Get input shape to compute pads.
      ArrayRef<int64_t> xShape = xType.getShape();

      for (int64_t i = 0; i < spatialRank; ++i) {
        int64_t inputSize = xShape[2 + i];
        int64_t outputSize = ArrayAttrIntVal(outputShapeOpt, i);
        int64_t kH = kernelShape[i].getLiteral();
        int64_t dH = dilations[i];
        int64_t sH = strides[i];
        int64_t opH = outputPadding[i];

        // Calculate total padding needed.
        // Formula: total_pad = stride * (input - 1) + output_padding + kernel -
        // output_shape.
        // Note: Unlike SAME_UPPER/SAME_LOWER without output_shape, we do NOT
        // clamp to non-negative here because negative padding is valid and will
        // be swapped to the opposite side below.
        int64_t totalPad =
            sH * (inputSize - 1) + opH + ((kH - 1) * dH + 1) - outputSize;

        // Distribute padding based on auto_pad mode.
        if (autoPad == "SAME_UPPER" || autoPad == "NOTSET") {
          // Extra padding on bottom/right (default behavior).
          padTop[i] = totalPad / 2;
          padBottom[i] = totalPad - padTop[i];
        } else { // SAME_LOWER
          // Extra padding on top/left.
          padBottom[i] = totalPad / 2;
          padTop[i] = totalPad - padBottom[i];
        }

        // When pads are negative, swap them to opposite sides.
        // This matches ONNX Runtime behavior where negative padding adds zeros
        // on opposite edges.
        if (padTop[i] < 0) {
          int64_t temp = padTop[i];
          padTop[i] = padBottom[i];
          padBottom[i] = temp;
        }
      }
    } else if (autoPad == "VALID") {
      // No padding.
      for (int64_t i = 0; i < spatialRank; ++i) {
        padTop[i] = 0;
        padBottom[i] = 0;
      }
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // For SAME padding: output_size = input_size * stride.
      // total_pad = output_padding + ((kernel_size - 1) * dilation + 1) -
      // stride.
      for (int64_t i = 0; i < spatialRank; ++i) {
        int64_t kH = kernelShape[i].getLiteral();
        int64_t dH = dilations[i];
        int64_t sH = strides[i];
        int64_t opH = outputPadding[i];

        int64_t totalPad = opH + ((kH - 1) * dH + 1) - sH;
        totalPad = std::max(0LL, (long long)totalPad);

        if (autoPad == "SAME_UPPER") {
          // Extra padding on bottom/right.
          padTop[i] = totalPad / 2;
          padBottom[i] = totalPad - padTop[i];
        } else { // SAME_LOWER
          // Extra padding on top/left.
          padBottom[i] = totalPad / 2;
          padTop[i] = totalPad - padBottom[i];
        }
      }
    } else { // NOTSET
      // Use explicit pads parameter.
      std::optional<ArrayAttr> padOpt = convTransposeOp.getPads();

      for (int64_t i = 0; i < spatialRank; ++i) {
        if (padOpt.has_value()) {
          padTop[i] = ArrayAttrIntVal(padOpt, i);
          padBottom[i] = ArrayAttrIntVal(padOpt, i + spatialRank);
        } else {
          padTop[i] = 0;
          padBottom[i] = 0;
        }
      }
    }

    // Step 1 & 2: Upsampling and padding using UpsampleAndPad.
    // Compute padding for UpsampleAndPad following Python reference.
    SmallVector<int64_t, 4> upsamplePads;
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t kH = kernelShape[i].getLiteral();
      int64_t dH = dilations[i];

      // Compute padding as in Python reference.
      int64_t basePad = (kH - 1) * dH;
      int64_t padLeft = basePad - padTop[i];

      upsamplePads.push_back(padLeft);
    }
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t kH = kernelShape[i].getLiteral();
      int64_t dH = dilations[i];
      int64_t opH = outputPadding[i];

      int64_t basePad = (kH - 1) * dH;
      int64_t padRight = basePad - padBottom[i] + opH;

      upsamplePads.push_back(padRight);
    }

    // Create UpsampleAndPad operation.
    ArrayAttr stridesAttr = rewriter.getI64ArrayAttr(strides);
    ArrayAttr padsAttr = rewriter.getI64ArrayAttr(upsamplePads);

    Type xUpType = UnrankedTensorType::get(xType.getElementType());
    Value xUp = ONNXUpsampleAndPadOp::create(
        rewriter, loc, xUpType, X, stridesAttr, padsAttr);

    // Step 3: Weight transformation.
    // Weight shape: (Cin, Cout_per_group, kH, kW, ...)
    int64_t Cin = wShape[0];
    int64_t CoutPg = wShape[1];
    int64_t CinPg = Cin / group;

    // Reshape to (group, Cin_per_group, Cout_per_group, kH, kW, ...).
    SmallVector<int64_t, 6> reshapeShape1;
    reshapeShape1.push_back(group);
    reshapeShape1.push_back(CinPg);
    reshapeShape1.push_back(CoutPg);
    for (int64_t i = 0; i < spatialRank; ++i)
      reshapeShape1.push_back(wShape[2 + i]);

    Value reshapeShapeVal1 = create.onnx.constantInt64(reshapeShape1);
    Type reshapeType1 =
        RankedTensorType::get(reshapeShape1, wType.getElementType());
    Value wReshaped1 = create.onnx.reshape(reshapeType1, W, reshapeShapeVal1);

    // Flip all spatial dimensions using a single Slice operation with negative
    // steps. Use INT64_MAX for start and INT64_MIN for end to flip entire
    // dimension.
    SmallVector<int64_t, 4> starts;
    SmallVector<int64_t, 4> ends;
    SmallVector<int64_t, 4> axes;
    SmallVector<int64_t, 4> steps;

    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t axis = 3 + i; // After group, Cin_pg, Cout_pg.

      // Flip by slicing from end to start with step -1.
      starts.push_back(INT64_MAX);
      ends.push_back(INT64_MIN);
      axes.push_back(axis);
      steps.push_back(-1);
    }

    Value startsVal = create.onnx.constantInt64(starts);
    Value endsVal = create.onnx.constantInt64(ends);
    Value axesVal = create.onnx.constantInt64(axes);
    Value stepsVal = create.onnx.constantInt64(steps);

    Type sliceType = UnrankedTensorType::get(wType.getElementType());
    Value wFlipped = create.onnx.slice(
        sliceType, wReshaped1, startsVal, endsVal, axesVal, stepsVal);

    // Permute: swap Cin_pg and Cout_pg (indices 1 and 2).
    // From (group, Cin_pg, Cout_pg, kH, kW, ...) to (group, Cout_pg, Cin_pg,
    // kH, kW, ...).
    SmallVector<int64_t, 6> permIndices;
    permIndices.push_back(0); // group
    permIndices.push_back(2); // Cout_pg
    permIndices.push_back(1); // Cin_pg
    for (int64_t i = 0; i < spatialRank; ++i)
      permIndices.push_back(3 + i);

    ArrayAttr permAttr = rewriter.getI64ArrayAttr(permIndices);
    Type transposeType = UnrankedTensorType::get(wType.getElementType());
    Value wPermuted = create.onnx.transpose(transposeType, wFlipped, permAttr);

    // Reshape back to (group * Cout_pg, Cin_pg, kH, kW, ...).
    SmallVector<int64_t, 5> reshapeShape2;
    reshapeShape2.push_back(group * CoutPg);
    reshapeShape2.push_back(CinPg);
    for (int64_t i = 0; i < spatialRank; ++i)
      reshapeShape2.push_back(wShape[2 + i]);

    Value reshapeShapeVal2 = create.onnx.constantInt64(reshapeShape2);
    Type reshapeType2 =
        RankedTensorType::get(reshapeShape2, wType.getElementType());
    Value wConv =
        create.onnx.reshape(reshapeType2, wPermuted, reshapeShapeVal2);

    // Step 4: Create Conv operation.
    // Use stride=1, dilation from original, group from original, no padding.
    SmallVector<int64_t, 4> convStrides(spatialRank, 1);
    SmallVector<int64_t, 4> convPads(2 * spatialRank, 0);

    // Extract kernel shape from weight dimensions (spatial dimensions).
    SmallVector<int64_t, 4> convKernelShape;
    for (int64_t i = 0; i < spatialRank; ++i)
      convKernelShape.push_back(wShape[2 + i]);

    Type convResultType = convTransposeOp.getResult().getType();
    Value convResult = create.onnx.conv(convResultType, xUp, wConv, B, "VALID",
        dilations, group, convKernelShape, convPads, convStrides);
    rewriter.replaceOp(convTransposeOp, convResult);
    return success();
  }
};

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

ElementsAttr reshapeElementsAttrToRank0WithDefaultValue(
    PatternRewriter &rewriter, Value shape, Attribute val) {
  if (!val) {
    // Default is 0.0 in float32. It is not created by default in the ONNX
    // getValue() as the ONNX td does not define a default value. So explicitly
    // create a dense array of 1 zero value here.
    Type elementType = rewriter.getF32Type();
    RankedTensorType tensorType = RankedTensorType::get({1}, elementType);
    FloatAttr floatAttr = rewriter.getFloatAttr(elementType, 0.0);
    val = DenseElementsAttr::get(tensorType, floatAttr.getValue());
  }
  return OnnxElementsAttrBuilder(shape.getContext())
      .reshape(cast<ElementsAttr>(val), {});
}

} // namespace onnx_mlir

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Dialect/ONNX/Transforms/ONNXDecompose.inc"

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
    Value maxInput = ONNXReduceMaxV13Op::create(
        rewriter, odsLoc, resultType, input, axisAttr, keepDimsAttr);
    Value subValue =
        ONNXSubOp::create(rewriter, odsLoc, inputType, input, maxInput);
    Value expValue = ONNXExpOp::create(rewriter, odsLoc, inputType, subValue);
    Value axisOp = create.onnx.constantInt64({axisValue});
    IntegerAttr noopWithEmptyAxes = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), 0);
    Value sumValue = ONNXReduceSumOp::create(rewriter, odsLoc, resultType,
        /*input=*/expValue,
        /*axis=*/axisOp, keepDimsAttr, noopWithEmptyAxes);
    Value divValue =
        ONNXDivOp::create(rewriter, odsLoc, inputType, expValue, sumValue);
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

    auto fusedV = ONNXConcatShapeTransposeOp::create(rewriter,
        concatOp.getLoc(), outputTypes, concatOp->getOperands(),
        concatOp.getAxisAttr(), shapeOp.getEndAttr(), shapeOp.getStartAttr(),
        transposeOp.getPermAttr());
    rewriter.replaceOp(shapeOp.getOperation(), fusedV.getResults()[0]);
    rewriter.replaceOp(transposeOp.getOperation(), fusedV.getResults()[1]);
    rewriter.eraseOp(concatOp);
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
    auto none = ONNXNoneOp::create(rewriter, loc);
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
        ONNXOneHotOp::create(rewriter, loc,
            RankedTensorType::get(newLabelsShape, labelsTy.getElementType()),
            labels, numClasses, oneHotVals, /*axis=*/1),
        /*saturate=*/
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1),
        TypeAttr::get(elemTy));
    // Compute logsoftmax of scores.
    auto softmax =
        ONNXSoftmaxOp::create(rewriter, loc, scoresTy, scores, /*axis=*/1);
    auto logSoftmax = ONNXLogOp::create(rewriter, loc, scoresTy, softmax);
    auto prod = ONNXMulOp::create(rewriter, loc, logSoftmax, oneHot);
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
      prod = ONNXMulOp::create(rewriter, loc, prod, weightsUnsqueezed);
    }
    // Reduction across `class` (dim=1) axis.
    auto axes = create.constant(onnx_mlir::createDenseArrayAttr(
        rewriter, rewriter.getI64ArrayAttr({1})));
    auto reducedType = createReducedType(scoresTy, 1, /*keepdims=*/true);
    Value loss =
        ONNXReduceSumOp::create(rewriter, loc, reducedType, prod, axes);
    // ReduceMean/ReduceSum/Squeeze if reduction = mean/sum/none respectively.
    // Set `axes=none` to indicate reducing all dims.
    auto reduction = cast<StringAttr>(sceOp.getReductionAttr()).getValue();
    if (reduction == "mean") {
      if (isa<NoneType>(weights.getType())) {
        loss = ONNXReduceMeanOp::create(rewriter, loc,
            RankedTensorType::get({}, elemTy), loss, none,
            /*keepdims=*/0);
      } else {
        auto sumL = ONNXReduceSumOp::create(rewriter, loc,
            RankedTensorType::get({}, elemTy), loss, none,
            /*keepdims=*/0);
        // Perform einsum(one_hot, weights) as a simple way of producing
        // W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]
        auto scatteredWeights = ONNXEinsumOp::create(rewriter, loc,
            RankedTensorType::get(labelsTy.getShape(), elemTy),
            ValueRange{oneHot, weights}, "ij...,j->i...");
        auto sumW = ONNXReduceSumOp::create(rewriter, loc,
            RankedTensorType::get({}, elemTy), scatteredWeights, none,
            /*keepdims=*/0);
        loss = ONNXDivOp::create(rewriter, loc, sumL, sumW);
      }
    } else if (reduction == "sum") {
      loss = ONNXReduceSumOp::create(rewriter, loc,
          RankedTensorType::get({}, elemTy), loss, none,
          /*keepdims=*/0);
    } else if (reduction == "none") {
      loss = ONNXSqueezeOp::create(rewriter, loc,
          createReducedType(reducedType, 1, /*keepdims=*/false), loss, axes);
    } else {
      llvm_unreachable("unexpected reduction type");
    }
    // Negate.
    loss = ONNXNegOp::create(rewriter, loc, loss.getType(), loss);
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
        result = ONNXAddOp::create(rewriter, sumOp.getLoc(), result, input);
      }
    }
    auto resultType = mlir::cast<ShapedType>(sumOp.getResult().getType());
    if (resultType != result.getType())
      result = ONNXCastOp::create(rewriter, sumOp.getLoc(), resultType, result,
          1, resultType.getElementType());
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

struct DecomposeHardSwishPattern : public OpRewritePattern<ONNXHardSwishOp> {
  using OpRewritePattern<ONNXHardSwishOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXHardSwishOp hardswishOp, PatternRewriter &rewriter) const final {

    // Get location and element type
    Location loc = hardswishOp.getLoc();
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);

    Value alphaConst = create.onnx.constantFloat32(1.0f / 6.0f);
    Value betaConst = create.onnx.constantFloat32(0.5f);
    Value minConst = create.onnx.constantFloat32(1.0f);
    Value maxConst = create.onnx.constantFloat32(0.0f);

    // Multiply input by alpha
    auto scaledInput =
        ONNXMulOp::create(rewriter, loc, hardswishOp.getOperand().getType(),
            hardswishOp.getOperand(), alphaConst);

    // Add beta to (input * alpha)
    auto shiftedInput = ONNXAddOp::create(
        rewriter, loc, scaledInput.getType(), scaledInput, betaConst);

    // Compute min(1.0, shiftedInput)
    auto minOp = ONNXMinOp::create(rewriter, loc, shiftedInput.getType(),
        ValueRange({shiftedInput, minConst}));

    // Compute max(0, min(1, shiftedInput))
    auto maxOp = ONNXMaxOp::create(
        rewriter, loc, minOp.getType(), ValueRange({minOp, maxConst}));

    // Compute final HardSwish: input * max(0, min(1, add(mul(x, alpha), beta)))
    auto hardswishResult = ONNXMulOp::create(rewriter, loc,
        hardswishOp.getOperand().getType(), hardswishOp.getOperand(), maxOp);

    // Replace the original HardSwishOp with the new computation
    rewriter.replaceOp(hardswishOp, hardswishResult.getResult());
    return success();
  }
};

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeONNXToONNXPass)

  DecomposeONNXToONNXPass(const std::string &target) { this->target = target; }
  DecomposeONNXToONNXPass(const DecomposeONNXToONNXPass &pass)
      : mlir::PassWrapper<DecomposeONNXToONNXPass,
            OperationPass<func::FuncOp>>() {
    this->target = pass.target.getValue();
  }

  StringRef getArgument() const override { return "decompose-onnx"; }

  StringRef getDescription() const override {
    return "Decompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  Option<std::string> target{*this, "target",
      llvm::cl::desc("Target Dialect to decompose into"), ::llvm::cl::init("")};

  void runOnOperation() final;

  typedef PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>>
      BaseType;
};

void DecomposeONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXCastLikeOp>();
  target.addIllegalOp<ONNXClipV11Op>();
  target.addIllegalOp<ONNXClipV12Op>();
  target.addIllegalOp<ONNXClipV6Op>();
  target.addIllegalOp<ONNXConstantOfShapeOp>();
  target.addIllegalOp<ONNXDFTV17Op>();
  target.addIllegalOp<ONNXGridSampleV16Op>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXPadV11Op>();
  target.addIllegalOp<ONNXPadV13Op>();
  target.addIllegalOp<ONNXPadV18Op>();
  target.addIllegalOp<ONNXPadV2Op>();
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL1V13Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceL2V13Op>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceMaxV18Op>();
  target.addIllegalOp<ONNXReduceMinV18Op>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXResizeV10Op>();
  target.addIllegalOp<ONNXResizeV11Op>();
  target.addIllegalOp<ONNXResizeV13Op>();
  target.addIllegalOp<ONNXResizeV18Op>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXScatterOp>();
  target.addIllegalOp<ONNXSequenceConstructOp>();
  target.addIllegalOp<ONNXSoftmaxCrossEntropyLossOp>();
  target.addIllegalOp<ONNXSplitV11Op>();
  target.addIllegalOp<ONNXSplitV13Op>();
  target.addIllegalOp<ONNXSqueezeV11Op>();
  target.addIllegalOp<ONNXSumOp>();
  target.addIllegalOp<ONNXUnsqueezeV11Op>();
  target.addIllegalOp<ONNXUpsampleOp>();
  target.addIllegalOp<ONNXUpsampleV7Op>();

  if (!onnx_mlir::decomposeOpsInONNX.empty()) {
    for (const auto &op : onnx_mlir::decomposeOpsInONNX) {
      if (op == "HardSwish") {
        target.addIllegalOp<ONNXHardSwishOp>();
      }
    }
  }
  target.addDynamicallyLegalOp<ONNXEinsumOp>([](ONNXEinsumOp op) {
    return !onnx_mlir::DecomposeEinsumPattern::isDecomposable(op);
  });

  target.addDynamicallyLegalOp<ONNXConcatOp>([](ONNXConcatOp op) {
    ONNXShapeOp shapeOp;
    ONNXTransposeOp transposeOp;
    return !isConcatFuseMatched(op, shapeOp, transposeOp);
  });

  target.addDynamicallyLegalOp<ONNXSequenceAtOp>([](ONNXSequenceAtOp op) {
    return !onnx_mlir::canSequenceAtBeReplaced(op.getResult());
  });

  // Rewrite ONNXConstantOp with scalar values into the one using ElementAttrs.
  target.addDynamicallyLegalOp<ONNXConstantOp>([](ONNXConstantOp op) {
    return !(op.getValueFloatAttr() || op.getValueFloatsAttr() ||
             op.getValueIntAttr() || op.getValueIntsAttr() ||
             op.getValueStringAttr() || op.getValueStringsAttr());
  });

  // Decompose CustomOp FusedMatMul introduced by onnxruntime:
  // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
  target.addDynamicallyLegalOp<ONNXCustomOp>([](ONNXCustomOp op) {
    int64_t rankA, rankB;
    FloatAttr alpha;
    return !CustomOpFuseMatMulPattern::isCustomOpFusedMatMulMatched(
        op, alpha, rankA, rankB);
  });

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  // ONNXtoStablehlo pass has own rewriting for ConvTranspose Op using
  // stablehlo ops. To avoid conflict with it, decomposing for ConvTranspose
  // is disabled when the target is stablehlo.
  if (this->target != "stablehlo") {
#endif
    target.addDynamicallyLegalOp<ONNXConvTransposeOp>(
        [](ONNXConvTransposeOp op) {
          return !onnx_mlir::shouldDecomposeConvTransposeOp(op);
        });
#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  }
#endif

  RewritePatternSet patterns(context);
  onnx_mlir::getDecomposeONNXToONNXPatterns(patterns);
  patterns.insert<ReplaceCastLikeByCastPattern>(context);
#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  if (this->target == "stablehlo") {
    populateDecomposingONNXBeforeStablehloPatterns(patterns, context);
    target.addIllegalOp<ONNXSoftmaxOp>();
  }
#endif

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getDecomposeONNXToONNXPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  populateWithGenerated(patterns);
  patterns.insert<onnx_mlir::DecomposeEinsumPattern>(context);
  patterns.insert<ConcatFusePattern>(context);
  // Decompose CustomOp FusedMatMul introduced by onnxruntime:
  // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
  patterns.insert<CustomOpFuseMatMulPattern>(context);
  patterns.insert<SoftmaxCrossEntropyPattern>(context);
  patterns.insert<SumToAddPattern>(context);
  patterns.insert<DecomposeConvTransposePattern>(context);

  if (!onnx_mlir::decomposeOpsInONNX.empty()) {
    for (const auto &op : onnx_mlir::decomposeOpsInONNX) {
      if (op == "HardSwish") {
        patterns.insert<DecomposeHardSwishPattern>(context);
      }
    }
  }

  // TODO: consider whether to include SoftmaxPattern here
}

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createDecomposeONNXToONNXPass(
    const std::string &target) {
  return std::make_unique<DecomposeONNXToONNXPass>(target);
}
