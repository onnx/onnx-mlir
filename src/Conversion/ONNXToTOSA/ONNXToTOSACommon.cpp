/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2021 Arm Limited.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace tosa {

static int64_t multiplyDims(llvm::ArrayRef<int64_t> dims, int64_t res = 1) {
  for (auto dim : dims) {
    if (ShapedType::isDynamic(dim)) {
      return ShapedType::kDynamic;
    }
    res = res * dim;
  }
  return res;
}

static int64_t countDynamicDims(llvm::ArrayRef<int64_t> dims) {
  int64_t count = 0;
  for (auto dim : dims)
    if (ShapedType::isDynamic(dim))
      ++count;
  return count;
}

// Lowers Gather operators to a sequence of TOSA ops.
// This Code is mostly the same as TF to TOSA.
std::optional<Value> convertGatherOp(PatternRewriter &rewriter, Location loc,
    Value resultValue, Value inputValue, Value indicesValue, int32_t batchDims,
    int32_t axis) {

  TosaBuilder tosaBuilder(rewriter, loc);

  auto resultType = resultValue.getType().dyn_cast<ShapedType>();
  auto inputType = inputValue.getType().dyn_cast<RankedTensorType>();
  auto indicesType = indicesValue.getType().dyn_cast<RankedTensorType>();

  if (!resultType || !inputType || !indicesType)
    return std::nullopt;

  // batchDims indicates the number of batch dimensions in input and
  // indices axis indicates the axis at which the gather indexing is
  // applied.  axis must be >= batch_dims.  When axis is equal to
  // batch_dims, the right-most batch dimension disappears.
  //
  // N: number of batches
  // Computed as product of input.shape[0:batch_dims-1]
  //
  // W: number of indices in each batch
  // Computed as product of indices.shape[batch_dims:]
  //
  // K: range of each index
  // Computed as  input.shape[axis:axis+rank(indices)-1]
  //
  // C: number of channels for each index
  // Computed as:  LeftChannels * RightChannels:
  // product(input.shape[batch_dims:axis]) * product(input.shape[axis+1:])
  //
  // The input tensor needs to be transposed, then reshaped to move the
  // dimensions into [N, K, C] order.
  //
  // The dimensions of the input input[] tensor are grouped in the following
  // order to begin with:
  //
  //  [Batch, LeftChannels, Indices, RightChannels]
  //  |-----||------------||-------||-------------|
  //     N         C_l         K          C_r
  //
  // Where Batch (N), Indices (K) can be one or more dimensions in size,
  // while LeftChannels and RightChannels represent the group of data channels
  // (C) to the left and right (C_l, C_r) of the indices; the sum of these two
  // is one or more dimensions in size, but either one may be zero depending
  // on how axis was specified by the caller.
  //
  // The resulting tensor will look like:
  //
  //  [Batch, Indices, LeftChannels, RightChannels]
  //  |-----||-------||---------------------------|
  //     N       K                 C
  //
  // The indices tensor simply needs a reshape to flatten all of the
  // batch dimensions (N) together and flatten all of the indices (W)
  // together.
  //
  // Then do the tosa.GATHER
  //
  // output[N,W,C] = tosa.GATHER(values[N,K,C], indices[N,W])
  //
  // Finally, the resulting tensor will have shape [N, W, C], where C is a
  // flattened version of [LeftChannels, RightChannels].  We need to reshape
  // to unflatten to:
  //
  //  [N, W, LeftChannels, RightChannels]
  //
  // and finally transpose back to the output shape
  //
  //  [Batch, LeftChannels, Non-Batch-Indices, RightChannels]

  size_t inputRank = inputType.getShape().size();
  size_t indicesRank = indicesType.getShape().size();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  if (!((size_t)batchDims <= indicesRank)) {
    (void)rewriter.notifyMatchFailure(
        loc, "batch_dims must be <= indices_rank for a valid gather op");
    return std::nullopt;
  }

  if (!(axis >= batchDims)) {
    (void)rewriter.notifyMatchFailure(
        loc, "axis must be >= batch_dims for a valid gather op");
    return std::nullopt;
  }

  // onnx allows i64 indices, but tosa does not.
  if (indicesType.getElementType().isInteger(64)) {
    indicesType =
        indicesType.clone(rewriter.getI32Type()).dyn_cast<RankedTensorType>();
    indicesValue = CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, loc, indicesType, indicesValue)
                       .getResult();
  }

  // Sizes for each of these fields.
  SmallVector<int64_t> inputBatch, inputIndices, inputLeftChannels,
      inputRightChannels;

  // Dimension indices for each of these fields.
  SmallVector<int32_t> inputIdxBatch, inputIdxIndices, inputIdxLeftChannels,
      inputIdxRightChannels;

  // Read through the input tensor dimensions left-to-right and extract the
  // different fields.
  for (int i = 0; i < (int)inputRank; i++) {
    // When batch_dims == axis, the batch dimension gets replaced.
    if (i < batchDims && i < axis) {
      inputBatch.push_back(inputShape[i]);
      inputIdxBatch.push_back(i);
    } else if (i < axis) {
      inputLeftChannels.push_back(inputShape[i]);
      inputIdxLeftChannels.push_back(i);
    } else if (i < (axis + 1)) {
      inputIndices.push_back(inputShape[i]);
      inputIdxIndices.push_back(i);
    } else {
      inputRightChannels.push_back(inputShape[i]);
      inputIdxRightChannels.push_back(i);
    }
  }

  // Calculate N, K, W, C
  int64_t N = multiplyDims(inputShape.take_front(batchDims));
  int64_t W =
      multiplyDims(indicesShape.slice(batchDims, indicesRank - batchDims));
  int64_t K = inputShape[axis];

  int64_t C = multiplyDims(inputShape.slice(batchDims, axis - batchDims));
  C = multiplyDims(inputShape.slice(axis + 1, inputRank - axis - 1), C);

  /////////////////////////////////////////////
  // Build up the input transpose operator
  SmallVector<int32_t> inputTransposePerm;

  // Batch
  inputTransposePerm.append(inputIdxBatch);

  // Indices
  inputTransposePerm.append(inputIdxIndices);

  // LeftChannels
  inputTransposePerm.append(inputIdxLeftChannels);

  // RightChannels
  inputTransposePerm.append(inputIdxRightChannels);

  /////////////////////////////////////////////
  // Build up the result reshape, in prepration for transpose
  // [N, W, C] -> [ Batch, Indices, LeftChannels, RightChannels ]
  SmallVector<int64_t> resultReshapeShape;

  // Indices
  // Use llvm::transform because range is an ArrayRef
  llvm::transform(indicesShape, std::back_inserter(resultReshapeShape),
      [](int64_t indiceDim) { return indiceDim; });

  // Left channels
  resultReshapeShape.append(inputLeftChannels);

  // Right channels.  But remove the axis dimension.
  resultReshapeShape.append(inputRightChannels);

  /////////////////////////////////////////////
  // Build up the result transpose operator.
  SmallVector<int32_t> resultTransposePerm;

  // Batch dimensions
  for (int i = 0; i < batchDims; i++) {
    resultTransposePerm.push_back(i);
  }

  // LeftChannels
  for (int i = 0; i < (int)inputLeftChannels.size(); i++) {
    resultTransposePerm.push_back(i + indicesType.getShape().size());
  }

  // Indices (remainder of dimensions after batch).
  for (int i = batchDims; i < (int)(indicesType.getShape().size()); i++) {
    resultTransposePerm.push_back(i);
  }

  // RightChannels, coming from after both the Indices and LeftChannels.
  for (int i = 0; i < (int)inputRightChannels.size(); i++) {
    resultTransposePerm.push_back(
        i + indicesType.getShape().size() + inputLeftChannels.size());
  }

  SmallVector<int64_t> tosaValuesShape = {N, K, C};
  SmallVector<int64_t> tosaIndicesShape = {N, W};

  // Begin of rewrite.

  auto inputTransposeOp = tosaBuilder.transpose(inputValue, inputTransposePerm);

  if (countDynamicDims(tosaValuesShape) > 1) {
    return (void)rewriter.notifyMatchFailure(loc,
               "only one dynamic dimension allowed when reshaping indices "
               "values."),
           std::nullopt;
  }

  auto tosaValuesReshapeOp =
      tosaBuilder.reshape(inputTransposeOp, tosaValuesShape);

  if (countDynamicDims(tosaIndicesShape) > 1) {
    return (void)rewriter.notifyMatchFailure(loc,
               "only one dynamic dimension allowed when reshaping indices"),
           std::nullopt;
  }

  auto tosaIndicesReshapeOp =
      tosaBuilder.reshape(indicesValue, tosaIndicesShape);

  Value tosaGatherOp = CreateOpAndInfer<mlir::tosa::GatherOp>(rewriter, loc,
      RankedTensorType::get(llvm::SmallVector<int64_t>(3, ShapedType::kDynamic),
          resultType.getElementType()),
      tosaValuesReshapeOp, tosaIndicesReshapeOp);

  if (countDynamicDims(resultReshapeShape) > 1) {
    return (void)rewriter.notifyMatchFailure(loc,
               "only one dynamic dimension allowed when reshaping result."),
           std::nullopt;
  }

  Value tosaResultReshapeOp =
      tosaBuilder.reshape(tosaGatherOp, resultReshapeShape);

  return tosaBuilder.transpose(tosaResultReshapeOp, resultTransposePerm);
}

// Common function for lowering reduce operations to TOSA ops.
template <typename T>
std::optional<Value> convertReduceOpCommon(PatternRewriter &rewriter,
    Operation *op, RankedTensorType output_type, Value input_value,
    ElementsAttr axes_elems, bool keep_dims, Type reduce_element_type,
    bool is_quantized, double input_scale, int64_t input_zp,
    double output_scale, int64_t output_zp) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  ArrayRef<int64_t> input_shape = input_type.getShape();
  ArrayRef<int64_t> output_shape = output_type.getShape();
  auto input_rank = input_shape.size();
  Value val = input_value;

  if (axes_elems.getNumElements() == 0) {
    // No axes means return the original tensor.
    auto identity_op = CreateOpAndInfer<mlir::tosa::IdentityOp>(
        rewriter, op->getLoc(), output_type, val);
    val = identity_op.getResult();
  } else {
    // Reduce along each axis
    SmallVector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    if (is_quantized) {
      val = buildRescaleToInt32(rewriter, op, val, input_scale, input_zp);
    }

    for (int i = 0; i < axes_elems.getNumElements(); i++) {
      int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
      if (axis_val < 0)
        axis_val += input_rank;
      auto axis_attr = rewriter.getI64IntegerAttr(axis_val);

      shape_vec[axis_val] = 1;
      RankedTensorType reduce_type =
          RankedTensorType::get(shape_vec, reduce_element_type);

      auto reduce_op = CreateOpAndInfer<T>(
          rewriter, op->getLoc(), reduce_type, val, axis_attr);

      val = reduce_op.getResult();
    }

    if (is_quantized) {
      RankedTensorType output_rescale_type =
          RankedTensorType::get(shape_vec, output_type.getElementType());
      val = buildRescale(rewriter, op, output_rescale_type, val, output_scale,
          0, output_zp, false, true);
    }

    // Optionally squeeze out the reduced axes.
    if (!keep_dims) {
      auto reshape_op =
          CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
              output_type, val, rewriter.getDenseI64ArrayAttr(output_shape));
      val = reshape_op.getResult();
    }
  }

  return val;
}

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value> convertReduceMeanOp(PatternRewriter &rewriter,
    Operation *op, TosaBuilder &tosaBuilder, RankedTensorType output_type,
    Value input_value, ElementsAttr axes_elems, bool keep_dims) {
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    op->emitOpError("ConvertReduceSumOp: input/output tensor should "
                    "be all quantized or all floating-point.");
    return std::nullopt;
  }

  // Only supports float type mean() if it's non-quantized
  if (!input_is_qtype && !output_type.getElementType().isa<mlir::FloatType>()) {
    op->emitWarning(
        "Failed convertReduceMean: input unquantized type but output element "
        "not FloatType!");
    return std::nullopt;
  }

  int64_t input_rank = input_type.getRank();
  int64_t num_elems_on_reduced_axis = 1;
  for (int i = 0; i < axes_elems.getNumElements(); i++) {
    int64_t axis_val = axes_elems.getValues<mlir::IntegerAttr>()[i].getInt();
    if (axis_val < 0)
      axis_val += input_rank;
    num_elems_on_reduced_axis *= input_type.getShape()[axis_val];
  }
  double div_scale = 1.0 / static_cast<double>(num_elems_on_reduced_axis);

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  mlir::Type reduce_element_type = input_type.getElementType();

  if (input_is_qtype) {
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    // Combine 'div_scale' as part of output rescale
    output_scale = div_scale * input_qtype.getScale() / output_qtype.getScale();

    input_zp = input_qtype.getZeroPoint();
    output_zp = output_qtype.getZeroPoint();
    reduce_element_type = rewriter.getI32Type();
  }

  auto val = convertReduceOpCommon<mlir::tosa::ReduceSumOp>(rewriter, op,
      output_type, input_value, axes_elems, keep_dims, reduce_element_type,
      input_is_qtype, input_scale, input_zp, output_scale, output_zp);

  if (!val.has_value())
    return std::nullopt;

  if (!input_is_qtype) {
    Value div_const = tosaBuilder.getSplattedConst(div_scale);
    return CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, op->getLoc(), output_type, val.value(), div_const, 0)
        .getResult();
  }

  return val;
}
} // namespace tosa
} // namespace onnx_mlir
