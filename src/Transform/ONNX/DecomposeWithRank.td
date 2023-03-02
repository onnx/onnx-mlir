// SPDX-License-Identifier: Apache-2.0

//===- ONNXDecomposeWithRank.td - Decompose ops with rank   -*- tablegen --===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_DECOMPOSEWITHRANK
#define ONNX_DECOMPOSEWITHRANK

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

//===----------------------------------------------------------------------===//
// Decomposition for ONNXConvTransposeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ONNXConvTransposeOp is rewritten into ONNXConv op and other ONNX ops to
// convert input, weight, and pad.
//
// 1. Reverse elements for spatial axis in weight tensor
//    Emit ONNXTransposeOp and ONNXReverseSequenceOp in reverseWeightConvTranspose()
// 2. Insert pads between elements in input data tensor if `strides` is not unit strides
//    Emit ONNXSplitOp, ONNXPadOp, and ONNXConcatOp in insertPadsConvTransposeInput()
// 3. Calculate new pads attribute from kernel_shape, original pads, and dilation
//    Emit no ONNX op in getPadsConvTranspose()
// 4. Create unit strides if `strides` is not unit strides
//    Emit no ONNX op in createUnitStrides()
// 5. Use ONNXConv op using converted weight tensor, input data tensor, other
//    attributes.
// 6. Insert additional pads if needed
//    Emit ONNXPadOp in insertAdditionalPadsConvTranspose()
//===----------------------------------------------------------------------===//

def reverseWeightConvTranspose: NativeCodeCall<
  "onnx_mlir::reverseWeightTensor($_builder, $_loc, $0)">;

def getPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::getPadsConvTranspose($_builder, $_loc, $0, $1, $2)">;

def HasUnitStrides: Constraint<
  CPred<"onnx_mlir::hasUnitStrides($_self)">,
  "has unit strides"
>;

def createUnitStrides: NativeCodeCall<
  "onnx_mlir::createUnitStrides($_builder, $0)">;

def insertPadsConvTransposeInput: NativeCodeCall<
  "onnx_mlir::insertPadsConvTransposeInput($_builder, $_loc, $0, $1)">;

def insertAdditionalPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::insertAdditionalPadsConvTranspose($_builder, $_loc, $0.getDefiningOp<ONNXConvOp>(), $0, $1)">;

def ConvTransposeOpPattern1: Pattern<
   (ONNXConvTransposeOp $x, $w, $b,
      $auto_pad, $dilation, $group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (getPadsConvTranspose:$new_pads $kernel_shape, $pads, $dilation),
    (ONNXConvOp:$conv_res $x, $reversed_w, $b,
       $auto_pad, $dilation, $group, $kernel_shape, $new_pads, $strides),
    (insertAdditionalPadsConvTranspose $conv_res, $output_shape)
   ],
   [(HasUnitStrides:$strides)],
   (addBenefit 1)
>;

def ConvTransposeOpPattern2: Pattern<
   (ONNXConvTransposeOp $x, $w, $b,
      $auto_pad, $dilation, $group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (insertPadsConvTransposeInput:$new_x $x, $strides),
    (getPadsConvTranspose:$new_pads $kernel_shape, $pads, $dilation),
    (createUnitStrides:$unit_strides $strides),
    (ONNXConvOp:$conv_res $new_x, $reversed_w, $b,
       $auto_pad, $dilation, $group, $kernel_shape, $new_pads, $unit_strides),
    (insertAdditionalPadsConvTranspose
        $conv_res,
        $output_padding, $output_shape)
   ],
   [],
   (addBenefit 0)
>;

#endif // ONNX_DECOMPOSEWITHRANK
