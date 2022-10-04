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
///    dag benefitsAdded = (addBenefit 0)
/// >;

def NHWC4LayoutAttr: NativeCodeCall<"$_builder.getStringAttr(::onnx_mlir::LAYOUT_NCHW4C)">;
def KCMN4C4KLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(::onnx_mlir::LAYOUT_KCMN4C4K)">;
def UndefinedLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(::onnx_mlir::LAYOUT_UNDEFINED)">;

// To avoid converting ops over and over again, has to have a mechanism
// to check if values are already using a custom ONNX data layout or not.
def HasNoCustomDataLayout :
    Constraint<CPred<"::onnx_mlir::hasCustomONNXTensorDataLayout($_self.getType()) == false">,
      "has no custom layout (i.e. original ONNX layout)">;

// Generate a convolution with custom data layout pattern.
// TODO: do not do so if it is a candidate for 1x1 transformation to conv, 
// unless there are less transform by switching it to custom layout.

def replaceONNXConvPatternWithLayout : Pat<
  (ONNXConvOp:$res $x, $w, $b, $auto_pad, $dilations, $group, $kernel_shape, $pad, $strides),
  (ONNXLayoutTransformOp
    (ONNXConvOp
       // Custom layout for x and w. 
       (ONNXLayoutTransformOp $x, (NHWC4LayoutAttr)),
       (ONNXLayoutTransformOp $w, (KCMN4C4KLayoutAttr)),
       // Remaining arguments unchanged.
       $b, $auto_pad, $dilations, $group, $kernel_shape, $pad, $strides, (returnType $res)),
    (UndefinedLayoutAttr)),
  [(HasNoCustomDataLayout:$x), (HasNoCustomDataLayout:$w)]
  //(addBenefit 1)
>;

//def replaceONNXConvPatternWithLayout : Pat<
//  (ONNXConvOp:$res $x, $w, $b, $auto_pad, $dilations, $group, $kernel_shape, $pad, $strides),
//    (ONNXConvOp
//       (ONNXLayoutTransformOp $x, (NHWC4LayoutAttr)),
//       $w,
//       $b,
//       $auto_pad, 
//       $dilations, 
//       $group, 
//       $kernel_shape, 
//       $pad, 
//       $strides),
//  [(HasNoCustomDataLayout:$x), (HasNoCustomDataLayout:$w)], 
//  (addBenefit 1)
//>;

#endif // ONNX_CONV_OPT
