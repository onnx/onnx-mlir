/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
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

static int64_t multiply_dims(int64_t a, int64_t b) {
  if (a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize) {
    return ShapedType::kDynamicSize;
  }
  return a * b;
}

static int64_t multiply_dims(llvm::ArrayRef<int64_t> dims, int64_t res = 1) {
  for (auto dim : dims) {
    if (ShapedType::isDynamic(dim)) {
      return ShapedType::kDynamicSize;
    }
    res = res * dim;
  }
  return res;
}

static int64_t count_dynamic_dims(llvm::ArrayRef<int64_t> dims) {
  int64_t count = 0;
  for (auto dim : dims)
    if (ShapedType::isDynamic(dim))
      ++count;
  return count;
}

// Lowers Gather operators to a sequence of TOSA ops.
llvm::Optional<Value> convertGatherOp(PatternRewriter &rewriter, Operation *op,
    Value result_value, Value params_value, Value indices_value,
    int32_t batch_dims, int32_t axis) {
  auto result_type = result_value.getType().dyn_cast<ShapedType>();
  auto params_type = params_value.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices_value.getType().dyn_cast<RankedTensorType>();

  if (!result_type || !params_type || !indices_type)
    return llvm::None;

  // batch_dims indicates the number of batch dimensions in params and
  // indices axis indicates the axis at which the gather indexing is
  // applied.  axis must be >= batch_dims.  When axis is equal to
  // batch_dims, the right-most batch dimension disappears.
  //
  // N: number of batches
  // Computed as product of params.shape[0:batch_dims-1]
  //
  // W: number of indices in each batch
  // Computed as product of indices.shape[batch_dims:]
  //
  // K: range of each index
  // Computed as  params.shape[axis:axis+rank(indices)-1]
  //
  // C: number of channels for each index
  // Computed as:  LeftChannels * RightChannels:
  // product(params.shape[batch_dims:axis]) * product(params.shape[axis+1:])
  //
  // The params tensor needs to be transposed, then reshaped to move the
  // dimensions into [N, K, C] order.
  //
  // The dimensions of the input params[] tensor are grouped in the following
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

  int params_rank = params_type.getShape().size();
  int indices_rank = indices_type.getShape().size();

  if (!(batch_dims <= indices_rank)) {
    (void)rewriter.notifyMatchFailure(
        op, "batch_dims must be <= indices_rank for a valid gather op");
    return llvm::None;
  }

  if (!(axis >= batch_dims)) {
    (void)rewriter.notifyMatchFailure(
        op, "axis must be >= batch_dims for a valid gather op");
    return llvm::None;
  }

  // onnx allows i64 indices, but tosa does not.
  if (indices_type.getElementType().isInteger(64)) {
    indices_type =
        indices_type.clone(rewriter.getI32Type()).dyn_cast<RankedTensorType>();
    indices_value = CreateOpAndInfer<mlir::tosa::CastOp>(
        rewriter, op->getLoc(), indices_type, indices_value)
                        .getResult();
  }

  // Sizes for each of these fields.
  SmallVector<int64_t> params_batch, params_indices, params_left_channels,
      params_right_channels;

  // Dimension indices for each of these fields.
  SmallVector<int64_t> params_idx_batch, params_idx_indices,
      params_idx_left_channels, params_idx_right_channels;

  // Read through the params tensor dimensions left-to-right and extract the
  // different fields.
  for (int i = 0; i < params_rank; i++) {
    // When batch_dims == axis, the batch dimension gets replaced.
    if (i < batch_dims && i < axis) {
      params_batch.push_back(params_type.getShape()[i]);
      params_idx_batch.push_back(i);
    } else if (i < axis) {
      params_left_channels.push_back(params_type.getShape()[i]);
      params_idx_left_channels.push_back(i);
    } else if (i < (axis + 1)) {
      params_indices.push_back(params_type.getShape()[i]);
      params_idx_indices.push_back(i);
    } else {
      params_right_channels.push_back(params_type.getShape()[i]);
      params_idx_right_channels.push_back(i);
    }
  }

  // Calculate N, K, W, C
  int64_t N = multiply_dims(params_type.getShape().take_front(batch_dims));
  int64_t W = multiply_dims(
      indices_type.getShape().slice(batch_dims, indices_rank - batch_dims));
  int64_t K = params_type.getShape()[axis];

  int64_t C = multiply_dims(
      params_type.getShape().slice(batch_dims, axis - batch_dims));
  C = multiply_dims(
      params_type.getShape().slice(axis + 1, params_rank - axis - 1), C);

  /////////////////////////////////////////////
  // Build up the params transpose operator
  SmallVector<int64_t> params_transpose_perm;
  SmallVector<int64_t> params_transpose_shape;

  // Batch
  for (int i = 0; i < params_batch.size(); i++) {
    params_transpose_perm.push_back(params_idx_batch[i]);
    params_transpose_shape.push_back(params_batch[i]);
  }

  // Indices
  for (int i = 0; i < params_indices.size(); i++) {
    params_transpose_perm.push_back(params_idx_indices[i]);
    params_transpose_shape.push_back(params_indices[i]);
  }

  // LeftChannels
  for (int i = 0; i < params_left_channels.size(); i++) {
    params_transpose_perm.push_back(params_idx_left_channels[i]);
    params_transpose_shape.push_back(params_left_channels[i]);
  }

  // RightChannels
  for (int i = 0; i < params_right_channels.size(); i++) {
    params_transpose_perm.push_back(params_idx_right_channels[i]);
    params_transpose_shape.push_back(params_right_channels[i]);
  }

  /////////////////////////////////////////////
  // Build up the result reshape, in prepration for transpose
  // [N, W, C] -> [ Batch, Indices, LeftChannels, RightChannels ]
  SmallVector<int64_t> result_reshape_shape;

  // Indices
  for (int i = 0; i < indices_type.getShape().size(); i++) {
    result_reshape_shape.push_back(indices_type.getShape()[i]);
  }

  // Left channels
  for (int i = 0; i < params_left_channels.size(); i++) {
    result_reshape_shape.push_back(params_left_channels[i]);
  }

  // Right channels.  But remove the axis dimension.
  for (int i = 0; i < params_right_channels.size(); i++) {
    result_reshape_shape.push_back(params_right_channels[i]);
  }

  /////////////////////////////////////////////
  // Build up the result transpose operator.
  SmallVector<int64_t> result_transpose_perm;

  // Batch dimensions
  for (int i = 0; i < batch_dims; i++) {
    result_transpose_perm.push_back(i);
  }

  // LeftChannels
  for (int i = 0; i < params_left_channels.size(); i++) {
    result_transpose_perm.push_back(i + indices_type.getShape().size());
  }

  // Indices (remainder of dimensions after batch).
  for (int i = batch_dims; i < (indices_type.getShape().size()); i++) {
    result_transpose_perm.push_back(i);
  }

  // RightChannels, coming from after both the Indices and LeftChannels.
  for (int i = 0; i < params_right_channels.size(); i++) {
    result_transpose_perm.push_back(
        i + indices_type.getShape().size() + params_left_channels.size());
  }

  SmallVector<int64_t> tosa_values_shape = {N, K, C};
  SmallVector<int64_t> tosa_indices_shape = {N, W};
  SmallVector<int64_t> tosa_gather_result_shape = {N, W, C};

  auto params_transpose_op = tosa::createTosaTransposedTensor(
      rewriter, op, params_value, params_transpose_perm);

  if (count_dynamic_dims(tosa_values_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(op,
               "multiply dynamic shapes when reshaping values down to "
               "tosa.gather"),
           llvm::None;
  }

  auto tosa_values_reshape_op = CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(tosa_values_shape, params_type.getElementType()),
      params_transpose_op, rewriter.getI64ArrayAttr(tosa_values_shape));

  if (count_dynamic_dims(tosa_indices_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(op,
               "multiply dynamic shapes when reshaping indices down to "
               "tosa.gather"),
           llvm::None;
  }

  auto tosa_indices_reshape_op = CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(tosa_indices_shape, indices_type.getElementType()),
      indices_value, rewriter.getI64ArrayAttr(tosa_indices_shape));

  auto tosa_gather_op = CreateOpAndInfer<mlir::tosa::GatherOp>(rewriter,
      op->getLoc(),
      RankedTensorType::get(
          tosa_gather_result_shape, result_type.getElementType()),
      tosa_values_reshape_op.getResult(), tosa_indices_reshape_op.getResult());

  if (count_dynamic_dims(result_reshape_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(op,
               "multiply dynamic shapes when reshaping tosa.gather result up"),
           llvm::None;
  }

  Value tosa_result_reshape_op = CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(result_reshape_shape, params_type.getElementType()),
      tosa_gather_op.getResult(),
      rewriter.getI64ArrayAttr(result_reshape_shape));

  return tosa::createTosaTransposedTensor(
      rewriter, op, tosa_result_reshape_op, result_transpose_perm);
}

// Common function for lowering reduce operations to TOSA ops.
template <typename T>
llvm::Optional<Value> convertReduceOpCommon(PatternRewriter &rewriter,
    Operation *op, RankedTensorType output_type, Value input_value,
    ElementsAttr axes_elems, bool keep_dims, Type reduce_element_type,
    bool is_quantized, double input_scale, int64_t input_zp,
    double output_scale, int64_t output_zp) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

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
              output_type, val, rewriter.getI64ArrayAttr(output_shape));
      val = reshape_op.getResult();
    }
  }

  return val;
}

// Lowers ReduceMean to a sequence of TOSA ops.
llvm::Optional<Value> convertReduceMeanOp(PatternRewriter &rewriter,
    Operation *op, TosaBuilder& tosaBuilder, RankedTensorType output_type, Value input_value,
    ElementsAttr axes_elems, bool keep_dims) {
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    op->emitOpError("ConvertReduceSumOp: input/output tensor should "
                    "be all quantized or all floating-point.");
    return llvm::None;
  }

  // Only supports float type mean() if it's non-quantized
  if (!input_is_qtype && !output_type.getElementType().isa<mlir::FloatType>()) {
    op->emitWarning(
        "Failed convertReduceMean: input unquantized type but output element "
        "not FloatType!");
    return llvm::None;
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
    return llvm::None;

  if (!input_is_qtype) {
    Value div_const = tosaBuilder.getConst(div_scale);
    return CreateOpAndInfer<mlir::tosa::MulOp>(
        rewriter, op->getLoc(), output_type, val.value(), div_const, 0)
        .getResult();
  }

  return val;
}
} // namespace tosa
} // namespace onnx_mlir
