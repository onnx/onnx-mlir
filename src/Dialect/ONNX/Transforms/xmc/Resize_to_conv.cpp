//
// Copyright (C) 2019 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <vector>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if weights are within quantization bounds
bool isWeightsWithinBound(
    std::vector<float> &weights, int ic, int &fix_point, float limit = 127.0f) {
  float min_value = 1.0f;
  float max_value = 0.0f;
  for (size_t i = 0; i < weights.size(); i += ic) {
    max_value = std::max(max_value, weights[i]);
    if (weights[i] != 0.0f) {
      min_value = std::min(min_value, weights[i]);
    }
  }
  fix_point = -std::ceil(std::log2(max_value / limit));
  bool within_bound = (1.0f / std::pow(2.0f, fix_point)) <= min_value;
  return within_bound;
}

/// Create transposed depthwise conv weights for upsampling
std::vector<float> createTransposedDepthwiseConvWeightsData(
    const std::vector<int32_t> &scale, ONNXResizeOp op_resize,
    mlir::Value op_input) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(op_input.getType());
  const int64_t ic = inputType.getShape()[1];

  bool half_pixel_centers = false;
  if (!op_resize.getCoordinateTransformationMode().empty()) {
    std::string coordMode = op_resize.getCoordinateTransformationMode().str();
    half_pixel_centers = (coordMode == "half_pixel");
  }

  std::vector<std::vector<float>> whd_upsample;
  std::vector<int32_t> kernel_sizes;

  for (auto scale_i : scale) {
    int32_t kernel_size = 2 * scale_i;
    kernel_sizes.push_back(kernel_size);

    std::vector<float> upsample(kernel_size, 0.0f);

    if (half_pixel_centers) {
      for (int i = 0; i < scale_i; i++) {
        upsample[i] =
            (static_cast<float>(i) + std::floor(scale_i / 2.0f) + 0.5f) /
                scale_i -
            0.5f;
      }
    } else {
      for (int i = 0; i < scale_i; i++) {
        upsample[i] = static_cast<float>(i) / static_cast<float>(scale_i);
      }
    }

    for (int i = 0; i < scale_i; i++) {
      upsample[scale_i + i] = 1.0f - upsample[i];
    }

    whd_upsample.push_back(std::move(upsample));
  }

  const size_t rank = inputType.getRank();

  if (rank == 4) {
    int kernel_h = kernel_sizes[0];
    int kernel_w = kernel_sizes[1];

    size_t wsize = ic * 1 * kernel_h * kernel_w;
    std::vector<float> weights(wsize, 0.0f);

    for (int64_t c = 0; c < ic; c++) {
      for (int h = 0; h < kernel_h; h++) {
        for (int w = 0; w < kernel_w; w++) {
          int offset = c * (static_cast<int64_t>(1) * kernel_h * kernel_w) +
                       static_cast<int64_t>(h) * kernel_w + w;
          float value = whd_upsample[0][h] * whd_upsample[1][w];
          weights[offset] = value;
        }
      }
    }
    return weights;

  } else if (rank == 5) {
    int kernel_d = kernel_sizes[0];
    int kernel_h = kernel_sizes[1];
    int kernel_w = kernel_sizes[2];

    size_t wsize = ic * 1 * kernel_d * kernel_h * kernel_w;
    std::vector<float> weights(wsize, 0.0f);

    for (int64_t c = 0; c < ic; c++) {
      for (int d = 0; d < kernel_d; d++) {
        for (int h = 0; h < kernel_h; h++) {
          for (int w = 0; w < kernel_w; w++) {
            int offset =
                c * (static_cast<int64_t>(1) * kernel_d * kernel_h * kernel_w) +
                static_cast<int64_t>(d) *
                    (static_cast<int64_t>(kernel_h) * kernel_w) +
                static_cast<int64_t>(h) * kernel_w + w;
            float value =
                whd_upsample[0][d] * whd_upsample[1][h] * whd_upsample[2][w];
            weights[offset] = value;
          }
        }
      }
    }
    return weights;
  }

  return std::vector<float>();
}

/// Create depthwise conv weights for downsampling
std::vector<float> createDepthwiseConvWeightsData(
    const std::vector<int32_t> &scale, ONNXResizeOp op_resize,
    mlir::Value op_input, int32_t bank_width,
    std::vector<int32_t> &weights_shape) {
  auto input_type = cast<RankedTensorType>(op_input.getType());
  const auto input_shape = input_type.getShape();

  const int ic = input_shape[1];
  const size_t rank = input_shape.size();

  bool half_pixel_centers = false;
  if (!op_resize.getCoordinateTransformationMode().empty()) {
    auto modeAttr = op_resize.getCoordinateTransformationMode().str();
    half_pixel_centers = (modeAttr == "half_pixel");
  }

  std::vector<float> first_idx(scale.size());
  std::vector<int> kernel(scale.size());

  for (size_t i = 0; i < scale.size(); ++i) {
    if (half_pixel_centers)
      first_idx[i] = 0.5f * static_cast<float>(scale[i]) - 0.5f;
    else
      first_idx[i] = 0.0f;

    kernel[i] = static_cast<int>(std::ceil(first_idx[i] + 1));

    if (rank == 5 && (ic % bank_width != 0) && i == 2) {
      kernel[2] += std::floor(first_idx[2]);
    }
  }

  if (rank == 4) {
    weights_shape = {ic, 1, kernel[0], kernel[1]};
  } else if (rank == 5) {
    weights_shape = {ic, 1, kernel[0], kernel[1], kernel[2]};
  }

  size_t wsize = 1;
  for (auto s : weights_shape) {
    wsize *= s;
  }

  std::vector<float> weights(wsize, 0.0f);

  const int kh = std::fmod(first_idx[1], 1.0f) == 0.0f ? 1 : 2;
  const int kw = std::fmod(first_idx[0], 1.0f) == 0.0f ? 1 : 2;
  float value_h = (kh == 1) ? 1.0f : 0.5f;
  float value_w = (kw == 1) ? 1.0f : 0.5f;

  if (rank == 4) {
    for (int c = 0; c < ic; ++c) {
      for (int h_iter = 0; h_iter < kh; ++h_iter) {
        for (int w_iter = 0; w_iter < kw; ++w_iter) {
          int h_idx = h_iter + std::floor(first_idx[1]);
          int w_idx = w_iter + std::floor(first_idx[0]);

          int offset =
              c * (1 * kernel[0] * kernel[1]) + h_idx * kernel[1] + w_idx;
          float value = value_h * value_w;
          weights[offset] = value;
        }
      }
    }
  } else if (rank == 5) {
    const int kd = std::fmod(first_idx[2], 1.0f) == 0.0f ? 1 : 2;
    float value_d = (kd == 1) ? 1.0f : 0.5f;

    for (int c = 0; c < ic; ++c) {
      for (int d_iter = 0; d_iter < kd; ++d_iter) {
        for (int h_iter = 0; h_iter < kh; ++h_iter) {
          for (int w_iter = 0; w_iter < kw; ++w_iter) {
            int d_idx = d_iter + std::floor(first_idx[2]);
            int h_idx = h_iter + std::floor(first_idx[1]);
            int w_idx = w_iter + std::floor(first_idx[0]);

            int offset = c * (1 * kernel[0] * kernel[1] * kernel[2]) +
                         d_idx * (kernel[1] * kernel[2]) + h_idx * kernel[2] +
                         w_idx;
            float value = value_h * value_w * value_d;
            weights[offset] = value;
          }
        }
      }
    }
  }

  return weights;
}

/// Create transposed conv2d weights (3D to 2D transformation)
std::vector<float> createTransposedConv2dWeightsData(
    const std::vector<int32_t> &scale, ONNXResizeOp op_resize,
    mlir::Value op_input, const std::vector<float> &weights) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(op_input.getType());
  auto inputShape = inputType.getShape();

  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op_resize.getResult().getType());
  auto outputShape = outputType.getShape();

  int64_t input_C = inputShape[1];
  int64_t input_D = inputShape[2];
  int64_t output_D = outputShape[2];

  bool half_pixel_centers = false;
  if (!op_resize.getCoordinateTransformationMode().empty()) {
    std::string coordMode = op_resize.getCoordinateTransformationMode().str();
    half_pixel_centers = (coordMode == "half_pixel");
  }

  int32_t scale_d = scale[0];
  int32_t scale_h = scale[1];
  int32_t scale_w = scale[2];

  int32_t kernel_d = 2 * scale_d;
  int32_t kernel_h = 2 * scale_h;
  int32_t kernel_w = 2 * scale_w;

  int32_t stride_d = scale_d;
  int32_t dilation_d = 1;

  std::vector<int32_t> pad_with(10, 0);

  if (half_pixel_centers && scale_d != 1) {
    pad_with[6] = 1;
    pad_with[7] = 1;
  } else {
    pad_with[7] = 1;
  }

  if (half_pixel_centers && scale_h != 1) {
    pad_with[2] = 1;
    pad_with[3] = 1;
  } else {
    pad_with[3] = 1;
  }

  if (half_pixel_centers && scale_w != 1) {
    pad_with[4] = 1;
    pad_with[5] = 1;
  } else {
    pad_with[5] = 1;
  }

  std::vector<int32_t> padding;
  std::vector<int32_t> scale_vec = {scale_d, scale_h, scale_w};
  for (auto scale_i : scale_vec) {
    int pad_l;
    int pad_r;
    if (half_pixel_centers && scale_i != 1) {
      pad_l =
          static_cast<int>(std::floor(scale_i / 2.0f) + (scale_i % 2 != 0)) +
          scale_i;
      pad_r = pad_l - (scale_i % 2 != 0);
    } else {
      pad_l = scale_i;
      pad_r = scale_i;
    }
    padding.push_back(pad_l);
    padding.push_back(pad_r);
  }

  int32_t conv3d_input_depth = input_D + pad_with[6] + pad_with[7];
  int32_t conv3d_output_depth = output_D;
  int32_t conv3d_output_channel = input_C;
  int32_t conv3d_input_channel = input_C;

  int32_t conv2d_input_channel = conv3d_input_depth * conv3d_input_channel;
  int32_t conv2d_output_channel = conv3d_output_depth * conv3d_output_channel;

  std::vector<int32_t> raw2d_w_shape{
      conv2d_output_channel, kernel_h, kernel_w, conv2d_input_channel};

  size_t raw_2d_wsize = static_cast<size_t>(conv2d_output_channel) * kernel_h *
                        kernel_w * conv2d_input_channel;

  std::vector<float> raw_ret(raw_2d_wsize, 0.0f);

  for (int idx_d_in = 0; idx_d_in < conv3d_input_depth; ++idx_d_in) {
    int idx_d_out_base = idx_d_in * stride_d - padding[0];

    for (int idx_c_out = 0; idx_c_out < conv3d_output_channel; ++idx_c_out) {
      int ori_addr_base = idx_c_out * kernel_d * kernel_h * kernel_w * 1;

      for (int idx_kh = 0; idx_kh < kernel_h; ++idx_kh) {
        for (int idx_kw = 0; idx_kw < kernel_w; ++idx_kw) {
          for (int idx_kd = 0; idx_kd < kernel_d; ++idx_kd) {
            int idx_d_out = idx_d_out_base + idx_kd * dilation_d;

            if (idx_d_out < 0 || idx_d_out >= conv3d_output_depth)
              continue;

            int new_addr_base =
                (idx_d_out * conv3d_output_channel + idx_c_out) * kernel_h *
                kernel_w * conv2d_input_channel;

            int ori_addr = ori_addr_base +
                           (kernel_d - 1 - idx_kd) * kernel_h * kernel_w * 1 +
                           idx_kh * kernel_w * 1 + idx_kw * 1;

            int new_addr = new_addr_base +
                           idx_kh * kernel_w * conv2d_input_channel +
                           idx_kw * conv2d_input_channel +
                           idx_d_in * conv3d_input_channel + idx_c_out;

            raw_ret[new_addr] = weights[ori_addr];
          }
        }
      }
    }
  }

  std::vector<int32_t> w_shape{
      conv2d_output_channel, kernel_h, kernel_w, static_cast<int32_t>(input_D)};

  size_t wsize = static_cast<size_t>(conv2d_output_channel) * kernel_h *
                 kernel_w * static_cast<size_t>(input_D);
  std::vector<float> ret_custom_layout(wsize, 0.0f);

  for (int idx_c_out = 0; idx_c_out < conv2d_output_channel; idx_c_out++) {
    for (int idx_kh = 0; idx_kh < kernel_h; idx_kh++) {
      for (int idx_kw = 0; idx_kw < kernel_w; idx_kw++) {
        for (int idx_d_in = 0; idx_d_in < w_shape[3]; idx_d_in++) {
          int src_addr_base =
              idx_c_out * kernel_h * kernel_w * raw2d_w_shape[3] +
              idx_kh * kernel_w * raw2d_w_shape[3] + idx_kw * raw2d_w_shape[3];

          int dst_addr = idx_c_out * kernel_h * kernel_w * w_shape[3] +
                         idx_kh * kernel_w * w_shape[3] + idx_kw * w_shape[3] +
                         idx_d_in;

          if (idx_d_in < pad_with[6]) {
            int idx_d_src1 = idx_d_in;
            int idx_d_src2 = 2 * pad_with[6] - idx_d_src1 - 1;
            ret_custom_layout[dst_addr] = raw_ret[src_addr_base + idx_d_src1] +
                                          raw_ret[src_addr_base + idx_d_src2];
          } else if (idx_d_in >= w_shape[3] - pad_with[7]) {
            int idx_d_src1 = pad_with[6] + idx_d_in;
            int idx_d_src2 = 2 * input_D + pad_with[6] - idx_d_in - 1;
            ret_custom_layout[dst_addr] = raw_ret[src_addr_base + idx_d_src1] +
                                          raw_ret[src_addr_base + idx_d_src2];
          } else {
            int idx_d_src = pad_with[6] + idx_d_in;
            ret_custom_layout[dst_addr] = raw_ret[src_addr_base + idx_d_src];
          }
        }
      }
    }
  }

  size_t onnx_wsize = conv2d_output_channel * input_D * kernel_h * kernel_w;
  std::vector<float> ret_onnx(onnx_wsize, 0.0f);

  for (int co = 0; co < conv2d_output_channel; ++co) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        for (int ci = 0; ci < input_D; ++ci) {
          size_t custom_idx = co * kernel_h * kernel_w * w_shape[3] +
                              kh * kernel_w * w_shape[3] + kw * w_shape[3] + ci;

          size_t onnx_idx = co * input_D * kernel_h * kernel_w +
                            static_cast<int64_t>(ci) * kernel_h * kernel_w +
                            static_cast<int64_t>(kh) * kernel_w + kw;

          ret_onnx[onnx_idx] = ret_custom_layout[custom_idx];
        }
      }
    }
  }

  return ret_onnx;
}

/// Replace resize with transposed depthwise conv (upsampling)
void replaceWithTransposedDepthwiseConv(PatternRewriter &rewriter, Location loc,
    Value input, Operation *resizeop, const std::vector<int32_t> &scale,
    const std::vector<float> &weights, int fix_point) {
  auto resizeOutput = resizeop->getResult(0);

  auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto outputType = mlir::dyn_cast<RankedTensorType>(resizeOutput.getType());
  if (!inputType || !outputType) {
    llvm::errs() << "[ERROR] Missing input/output types\n";
    return;
  }

  const int64_t spatialRank = inputType.getShape().size() - 2;
  if (spatialRank != 2 && spatialRank != 3) {
    llvm::errs() << "[ERROR] Unsupported spatial rank: " << spatialRank << "\n";
    return;
  }

  SmallVector<int64_t> kernelShape;
  for (auto s : scale)
    kernelShape.push_back(static_cast<int64_t>(2) * s);

  int64_t kernelD = 1;
  int64_t kernelH = 1;
  int64_t kernelW = 1;

  if (spatialRank == 2) {
    kernelH = kernelShape[0];
    kernelW = kernelShape[1];
  } else if (spatialRank == 3) {
    kernelD = kernelShape[0];
    kernelH = kernelShape[1];
    kernelW = kernelShape[2];
  }

  auto inShape = inputType.getShape();
  int64_t C_in = inShape[1];
  int64_t group = C_in;
  int64_t CoutPerGroup = 1;

  size_t totalElems;
  RankedTensorType weightType;

  if (spatialRank == 2) {
    totalElems = C_in * CoutPerGroup * kernelH * kernelW;
    weightType = RankedTensorType::get(
        {C_in, CoutPerGroup, kernelH, kernelW}, rewriter.getF32Type());
  } else {
    totalElems = C_in * CoutPerGroup * kernelD * kernelH * kernelW;
    weightType = RankedTensorType::get(
        {C_in, CoutPerGroup, kernelD, kernelH, kernelW}, rewriter.getF32Type());
  }

  assert(totalElems == weights.size() && "weights size mismatch");

  std::vector<float> weightsQ = weights;
  if (fix_point != 0) {
    const float scale_fp = std::pow(2.0f, static_cast<float>(fix_point));
    for (auto &w : weightsQ)
      w = std::round(w * scale_fp) / scale_fp;
  }

  DenseElementsAttr weightsAttr =
      DenseElementsAttr::get(weightType, ArrayRef<float>(weightsQ));
  auto valueAttr = rewriter.getNamedAttr("value", weightsAttr);

  auto weightsConst = rewriter.create<ONNXConstantOp>(loc, weightType,
      mlir::ValueRange{}, mlir::ArrayRef<mlir::NamedAttribute>{valueAttr});

  SmallVector<int64_t> strides(scale.begin(), scale.end());
  SmallVector<int64_t> dilations(scale.size(), 1);

  SmallVector<int64_t> pads;
  SmallVector<int64_t> output_padding;
  SmallVector<int64_t> kernelShapeAttr;
  auto outShape = outputType.getShape();

  if (spatialRank == 2) {
    kernelShapeAttr = {kernelH, kernelW};

    for (size_t i = 0; i < 2; ++i) {
      int64_t spatial_idx = 2 + i;
      int64_t in_size = inShape[spatial_idx];
      int64_t out_size = outShape[spatial_idx];
      int64_t kernel = kernelShapeAttr[i];
      int64_t stride = strides[i];

      int64_t pad = stride / 2;
      int64_t calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      int64_t out_pad = out_size - calc_out;

      if (out_pad < 0 || out_pad >= stride) {
        pad = (stride * (in_size - 1) + kernel - out_size) / 2;
        out_pad = out_size - (stride * (in_size - 1) + kernel - 2 * pad);
      }

      pads.push_back(pad);
      pads.push_back(pad);
      output_padding.push_back(out_pad);
    }
  } else {
    kernelShapeAttr = {kernelD, kernelH, kernelW};

    for (size_t i = 0; i < 3; ++i) {
      int64_t spatial_idx = 2 + i;
      int64_t in_size = inShape[spatial_idx];
      int64_t out_size = outShape[spatial_idx];
      int64_t kernel = kernelShapeAttr[i];
      int64_t stride = strides[i];

      int64_t pad = stride / 2;
      int64_t calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      int64_t out_pad = out_size - calc_out;

      if (out_pad < 0 || out_pad >= stride) {
        pad = (stride * (in_size - 1) + kernel - out_size) / 2;
        out_pad = out_size - (stride * (in_size - 1) + kernel - 2 * pad);
      }

      pads.push_back(pad);
      pads.push_back(pad);
      output_padding.push_back(out_pad);
    }
  }

  auto noneVal = rewriter.create<mlir::ONNXNoneOp>(loc).getResult();

  rewriter.setInsertionPointAfterValue(input);

  auto convTransposeOp = rewriter.create<ONNXConvTransposeOp>(loc, outputType,
      input, weightsConst.getResult(), noneVal,
      rewriter.getStringAttr("NOTSET"), rewriter.getI64ArrayAttr(dilations),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), group),
      rewriter.getI64ArrayAttr(kernelShapeAttr),
      rewriter.getI64ArrayAttr(output_padding), nullptr,
      rewriter.getI64ArrayAttr(pads), rewriter.getI64ArrayAttr(strides));

  mlir::Value convResult = convTransposeOp.getResult();

  mlir::Operation *oldOp = resizeOutput.getDefiningOp();
  if (!oldOp) {
    llvm::errs() << "[ERROR] resizeOutput has no defining op; cannot replace\n";
    return;
  }

  rewriter.replaceOp(oldOp, convResult);
}

/// Replace resize with transposed conv2d (5D to 4D case)
void replaceWithTransposedConv2d(PatternRewriter &rewriter, Location loc,
    Value op_input, Value op_output, const std::vector<int32_t> &scale,
    const std::vector<float> &weights, int fix_point) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(op_input.getType());
  auto outputType = mlir::dyn_cast<RankedTensorType>(op_output.getType());

  if (!inputType || !outputType) {
    llvm::errs() << "[ERROR] Invalid input/output types\n";
    return;
  }

  auto inShape5D = inputType.getShape();
  auto outShape5D = outputType.getShape();

  int64_t N = inShape5D[0];
  int64_t C = inShape5D[1];
  int64_t H_in = inShape5D[3];
  int64_t W_in = inShape5D[4];
  int64_t H_out = outShape5D[3];
  int64_t W_out = outShape5D[4];

  SmallVector<int64_t> reshaped4DInShape = {N, C, H_in, W_in};
  auto reshaped4DInType =
      RankedTensorType::get(reshaped4DInShape, inputType.getElementType());

  auto inShapeType = RankedTensorType::get({4}, rewriter.getI64Type());
  auto inShapeAttr = DenseElementsAttr::get(
      inShapeType, llvm::ArrayRef<int64_t>(reshaped4DInShape));
  auto inShapeConst = rewriter.create<ONNXConstantOp>(loc, inShapeType,
      ValueRange{}, rewriter.getNamedAttr("value", inShapeAttr));

  mlir::Value reshapedInput = rewriter
                                  .create<ONNXReshapeOp>(loc, reshaped4DInType,
                                      op_input, inShapeConst.getResult())
                                  .getReshaped();

  SmallVector<int64_t> reshaped4DOutShape = {N, C, H_out, W_out};
  auto reshaped4DOutType =
      RankedTensorType::get(reshaped4DOutShape, outputType.getElementType());

  int64_t kernel_h = static_cast<int64_t>(2) * scale[1];
  int64_t kernel_w = static_cast<int64_t>(2) * scale[2];

  size_t total_elems = weights.size();

  if ((size_t)(kernel_h * kernel_w * C) == 0) {
    llvm::errs() << "[ERROR] Invalid kernel or channel size\n";
    return;
  }

  auto Cout = (int64_t)(total_elems / (kernel_h * kernel_w * C));

  bool isDepthwise =
      (Cout == C) && (total_elems == (size_t)(C * 1 * kernel_h * kernel_w));
  int64_t group = isDepthwise ? C : 1;

  std::vector<float> weights_quantized = weights;
  if (fix_point != 0) {
    const float scale_fp = std::pow(2.0f, static_cast<float>(fix_point));
    for (auto &w : weights_quantized) {
      w = std::round(w * scale_fp) / scale_fp;
    }
  }

  std::vector<int64_t> weightShape;
  if (isDepthwise) {
    weightShape = {C, 1, kernel_h, kernel_w};
  } else {
    weightShape = {Cout, C, kernel_h, kernel_w};
  }

  RankedTensorType weightType =
      RankedTensorType::get(weightShape, rewriter.getF32Type());
  DenseElementsAttr weightsAttr = DenseElementsAttr::get(
      weightType, llvm::ArrayRef<float>(weights_quantized));

  auto weightsConst = rewriter.create<ONNXConstantOp>(loc, weightType,
      ValueRange{}, rewriter.getNamedAttr("value", weightsAttr));

  auto noneVal = rewriter.create<mlir::ONNXNoneOp>(loc).getResult();

  SmallVector<int64_t> strides = {scale[1], scale[2]};
  SmallVector<int64_t> dilations = {1, 1};
  SmallVector<int64_t> kernelShapeAttr = {kernel_h, kernel_w};

  SmallVector<int64_t> pads;
  SmallVector<int64_t> output_padding;

  // H dimension
  {
    int64_t in_size = H_in;
    int64_t out_size = H_out;
    int64_t kernel = kernel_h;
    int64_t stride = strides[0];

    int64_t pad = stride / 2;
    int64_t calc_out = stride * (in_size - 1) + kernel - 2 * pad;
    int64_t out_pad = out_size - calc_out;

    while (out_pad < 0) {
      pad--;
      calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      out_pad = out_size - calc_out;
    }
    while (out_pad >= stride) {
      pad++;
      calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      out_pad = out_size - calc_out;
    }

    if (pad < 0) {
      llvm::errs() << "[ERROR] Negative padding for H\n";
      return;
    }

    pads.push_back(pad);
    pads.push_back(pad);
    output_padding.push_back(out_pad);
  }

  // W dimension
  {
    int64_t in_size = W_in;
    int64_t out_size = W_out;
    int64_t kernel = kernel_w;
    int64_t stride = strides[1];

    int64_t pad = stride / 2;
    int64_t calc_out = stride * (in_size - 1) + kernel - 2 * pad;
    int64_t out_pad = out_size - calc_out;

    while (out_pad < 0) {
      pad--;
      calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      out_pad = out_size - calc_out;
    }
    while (out_pad >= stride) {
      pad++;
      calc_out = stride * (in_size - 1) + kernel - 2 * pad;
      out_pad = out_size - calc_out;
    }

    if (pad < 0) {
      llvm::errs() << "[ERROR] Negative padding for W\n";
      return;
    }

    pads.push_back(pad);
    pads.push_back(pad);
    output_padding.push_back(out_pad);
  }

  auto convT = rewriter.create<ONNXConvTransposeOp>(loc, reshaped4DOutType,
      reshapedInput, weightsConst.getResult(), noneVal,
      rewriter.getStringAttr("NOTSET"), rewriter.getI64ArrayAttr(dilations),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), group),
      rewriter.getI64ArrayAttr(kernelShapeAttr),
      rewriter.getI64ArrayAttr(output_padding), nullptr,
      rewriter.getI64ArrayAttr(pads), rewriter.getI64ArrayAttr(strides));

  SmallVector<int64_t> final5DShape = {N, C, 1, H_out, W_out};
  auto final5DType =
      RankedTensorType::get(final5DShape, outputType.getElementType());

  auto outShapeType = RankedTensorType::get({5}, rewriter.getI64Type());
  auto outShapeAttr = DenseElementsAttr::get(
      outShapeType, llvm::ArrayRef<int64_t>(final5DShape));
  auto outShapeConst = rewriter.create<ONNXConstantOp>(loc, outShapeType,
      ValueRange{}, rewriter.getNamedAttr("value", outShapeAttr));

  mlir::Value result = rewriter
                           .create<ONNXReshapeOp>(loc, final5DType,
                               convT.getResult(), outShapeConst.getResult())
                           .getReshaped();

  mlir::Operation *oldOp = op_output.getDefiningOp();
  if (!oldOp) {
    llvm::errs() << "[ERROR] No defining op\n";
    return;
  }

  rewriter.replaceOp(oldOp, result);
}

/// Replace resize with depthwise conv (downsampling)
void replaceWithDepthwiseConv(PatternRewriter &rewriter, Location loc,
    Value input, Value output, const std::vector<int32_t> &scale,
    const std::vector<float> &weights,
    const std::vector<int32_t> &weights_shape) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto outputType = mlir::dyn_cast<RankedTensorType>(output.getType());

  if (!inputType || !outputType) {
    llvm::errs() << "[ERROR] Missing input/output types\n";
    return;
  }

  int64_t inChannels = inputType.getShape()[1];

  std::vector<int64_t> weightsShape64(
      weights_shape.begin(), weights_shape.end());

  RankedTensorType weightsType =
      RankedTensorType::get(weightsShape64, rewriter.getF32Type());

  DenseElementsAttr weightsAttr =
      DenseElementsAttr::get(weightsType, llvm::ArrayRef<float>(weights));

  auto weightsConst = rewriter.create<ONNXConstantOp>(loc, weightsType,
      ValueRange{}, rewriter.getNamedAttr("value", weightsAttr));

  SmallVector<int64_t> kernelShape;
  for (size_t i = 2; i < weightsShape64.size(); ++i) {
    kernelShape.push_back(weightsShape64[i]);
  }

  SmallVector<int64_t> strides(scale.begin(), scale.end());
  SmallVector<int64_t> dilations(kernelShape.size(), 1);
  SmallVector<int64_t> pads(kernelShape.size() * 2, 0);

  auto noneVal = rewriter.create<mlir::ONNXNoneOp>(loc).getResult();

  auto convOp = rewriter.create<ONNXConvOp>(loc, outputType, input,
      weightsConst.getResult(), noneVal, rewriter.getStringAttr("NOTSET"),
      rewriter.getI64ArrayAttr(dilations),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), inChannels),
      rewriter.getI64ArrayAttr(kernelShape), rewriter.getI64ArrayAttr(pads),
      rewriter.getI64ArrayAttr(strides));

  mlir::Operation *oldOp = output.getDefiningOp();
  if (!oldOp) {
    llvm::errs() << "[ERROR] output has no defining op; cannot replace\n";
    return;
  }

  rewriter.replaceOp(oldOp, convOp.getResult());
}

//===----------------------------------------------------------------------===//
// Pattern: TransferResizeLinearToDwConv
//===----------------------------------------------------------------------===//

/// Pattern to transform linear Resize operations to depthwise convolutions
struct TransferResizeLinearToDwConv
    : public OpRewritePattern<mlir::ONNXResizeOp> {
  using OpRewritePattern<mlir::ONNXResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::ONNXResizeOp resizeOp, PatternRewriter &rewriter) const override {
    mlir::Location loc = resizeOp.getLoc();
    mlir::Value input = resizeOp.getX();
    mlir::Value output = resizeOp.getY();
    bool is_downsample = false;

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return failure();

    // Only handle linear/trilinear resize
    std::string mode = resizeOp.getMode().str();
    if (mode != "linear" && mode != "trilinear") {
      return rewriter.notifyMatchFailure(resizeOp, "unsupported resize mode");
    }

    auto outputType =
        mlir::dyn_cast<RankedTensorType>(resizeOp.getY().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(resizeOp, "unranked output");

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    if (inputShape.size() < 4 || inputShape.size() > 5)
      return rewriter.notifyMatchFailure(resizeOp, "only 2D or 3D supported");

    // Compute per-dimension scale factors
    std::vector<float> scale_f;
    for (size_t dim_i = 2; dim_i < inputShape.size(); ++dim_i) {
      float scale = static_cast<float>(outputShape[dim_i]) /
                    static_cast<float>(inputShape[dim_i]);
      scale_f.push_back(scale);
    }

    // Get original scales from the Resize op
    std::vector<float> scale_origin;
    bool has_scales = false;

    if (resizeOp.getScales()) {
      mlir::Value scalesValue = resizeOp.getScales();

      if (auto *scalesDefOp = scalesValue.getDefiningOp()) {
        if (auto constantOp = llvm::dyn_cast<ONNXConstantOp>(scalesDefOp)) {
          auto valueAttr = constantOp.getValue();

          if (valueAttr.has_value()) {
            auto scalesAttr =
                mlir::dyn_cast<DenseElementsAttr>(valueAttr.value());

            if (scalesAttr) {
              for (auto val : scalesAttr.getValues<float>()) {
                scale_origin.push_back(val);
              }
              has_scales = true;
            }
          }
        }
      }
    }

    // Check if sizes input is provided
    bool has_sizes = false;
    if (resizeOp.getSizes()) {
      mlir::Value sizesValue = resizeOp.getSizes();

      if (sizesValue && sizesValue.getDefiningOp() &&
          !llvm::isa<ONNXNoneOp>(sizesValue.getDefiningOp())) {
        has_sizes = true;
      }
    }

    // Validate scales match
    if (!has_sizes && has_scales && scale_origin.size() == scale_f.size()) {
      bool scales_match = true;
      const float epsilon = 1e-6f;

      for (size_t i = 0; i < scale_origin.size(); ++i) {
        if (std::abs(scale_origin[i] - scale_f[i]) > epsilon) {
          scales_match = false;
          break;
        }
      }

      if (!scales_match) {
        llvm::errs() << "[WARN] Resize op has undividable scales. Skipping.\n";
        return failure();
      }
    }

    // Detect upsample vs downsample
    if (std::all_of(scale_f.begin(), scale_f.end(),
            [](float s) { return s >= 1.0f; })) {
      // upsample
    } else if (std::all_of(scale_f.begin(), scale_f.end(),
                   [](float s) { return s <= 1.0f; })) {
      is_downsample = true;
      for (size_t i = 0; i < scale_f.size(); ++i)
        scale_f[i] = static_cast<float>(inputShape[i + 2]) /
                     static_cast<float>(outputShape[i + 2]);
    } else {
      return rewriter.notifyMatchFailure(
          resizeOp, "Resize must be purely upsample or downsample");
    }

    // Ensure integer scale
    for (auto s : scale_f) {
      if (std::fmod(s, 1.0f) != 0.0f)
        return rewriter.notifyMatchFailure(
            resizeOp, "Non-integer scaling not supported");
    }

    std::vector<int32_t> scale;
    for (auto s : scale_f)
      scale.push_back(static_cast<int>(s));

    if (!is_downsample) {
      std::vector<float> weights =
          createTransposedDepthwiseConvWeightsData(scale, resizeOp, input);
      int fixPoint = 0;

      if (inputShape.size() == 5 && inputShape[2] == 1 && scale[0] == 1) {
        auto weights2D =
            createTransposedConv2dWeightsData(scale, resizeOp, input, weights);
        int fixPoint2D;

        if (isWeightsWithinBound(weights2D, 1, fixPoint2D)) {
          replaceWithTransposedConv2d(
              rewriter, loc, input, output, scale, weights2D, fixPoint2D);
        } else {
          replaceWithTransposedDepthwiseConv(
              rewriter, loc, input, resizeOp, scale, weights, fixPoint);
        }
      } else {
        replaceWithTransposedDepthwiseConv(
            rewriter, loc, input, resizeOp, scale, weights, fixPoint);
      }
    } else {
      // Downsample case
      int output_bank_width = 1;
      std::vector<int32_t> weights_shape;
      std::vector<float> weights = createDepthwiseConvWeightsData(
          scale, resizeOp, input, output_bank_width, weights_shape);
      int fixPoint;
      if (!isWeightsWithinBound(weights, inputShape.back(), fixPoint)) {
        return failure();
      }

      replaceWithDepthwiseConv(
          rewriter, loc, input, output, scale, weights, weights_shape);
    }
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct TransferResizeLinearToDwConvPass
    : public PassWrapper<TransferResizeLinearToDwConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-resize-linear-to-dwconv";
  }
  StringRef getDescription() const override {
    return "Transfer linear Resize operations to depthwise convolutions";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransferResizeLinearToDwConv>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferResizeLinearToDwConv() {
  return std::make_unique<TransferResizeLinearToDwConvPass>();
}

} // namespace onnx_mlir
