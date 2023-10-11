// SPDX-License-Identifier: Apache-2.0

//===- ConvOpt.td - Rewriting for Convolution Optimizations -*- tablegen --===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_CONV_OPT
#define ONNX_CONV_OPT

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

// Layouts that are used in this optimization phase.
def NHWC4LayoutAttr:
  NativeCodeCall<"ONNXTensorEncodingAttr::get($_builder.getContext(), ONNXTensorEncodingAttr::DataLayout::NCHWxC, 4, 0)">;
def KCMN4C4KLayoutAttr:
  NativeCodeCall<"ONNXTensorEncodingAttr::get($_builder.getContext(), ONNXTensorEncodingAttr::DataLayout::KCNMxCyK, 4, 4)">;
def StandardLayoutAttr:
  NativeCodeCall<"$_builder.getStringAttr(::onnx_mlir::LAYOUT_STANDARD)">;

// Rewrite ranked and shaped type to have the given ONNX Tensor Encoding.
def tensorTypeWithONNXTensorEncoding:
  NativeCodeCall<"::onnx_mlir::convertTensorTypeToTensorTypeWithEncoding($0.getType(), $1)">;

// To avoid converting ops over and over again, has to have a mechanism
// to check if values are already using a custom ONNX data layout or not.
def HasNoCustomDataLayout:
    Constraint<CPred<"::onnx_mlir::hasCustomONNXTensorDataLayout($_self.getType()) == false">,
      "has no custom layout (i.e. original ONNX layout)">;

// Check if value has a type that is ranked with a shape.
def HasRankAndShape:
  Constraint<CPred<"::onnx_mlir::isRankedShapedType($_self.getType())">,
  "A value has rank and shape">;

// Generate a convolution with custom data layout pattern.
// TODO: do not do so if it is a candidate for 1x1 transformation to conv,
// unless there are less transform by switching it to custom layout.

def replaceONNXConvPatternWithLayout : Pat<
  (ONNXConvOp:$res $x, $w, $b, $auto_pad, $dilations, $group, $kernel_shape, $pad, $strides),
  (ONNXLayoutTransformOp
    (ONNXConvOp
      // Output type with layout
      // (tensorTypeWithONNXTensorEncoding $res, (NHWC4LayoutAttr)),
      // Custom layout for x and w.
      (ONNXLayoutTransformOp $x, (NHWC4LayoutAttr)),
      (ONNXLayoutTransformOp $w, (KCMN4C4KLayoutAttr)),
      // Remaining arguments unchanged.
      $b, $auto_pad, $dilations, $group, $kernel_shape, $pad, $strides,
      //(tensorTypeWithONNXTensorEncoding $res, (NHWC4LayoutAttr))
      //(returnType $res)
      (returnType (tensorTypeWithONNXTensorEncoding $res, (NHWC4LayoutAttr)))
      ),
    (StandardLayoutAttr)),
  [(HasRankAndShape: $res), (HasNoCustomDataLayout:$x), (HasNoCustomDataLayout:$w)]
  //(addBenefit 1)
>;

#endif // ONNX_CONV_OPT
