// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_DECOMPOSE_CONVTRANSPOSE_PHASED
#define ONNX_DECOMPOSE_CONVTRANSPOSE_PHASED

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "Decompose.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def hasNoActivationConsumer: Constraint<
  CPred<"onnx_mlir::hasNoActivationConsumer($_self)">,
  "Op has no activation consumer"
>;

//===----------------------------------------------------------------------===//
// ONNXConvTransposeOp is rewritten into ONNXConv op and other ONNX ops to
// convert input, weight, and pad.
//
// 1. Reverse elements for spatial axis in weight tensor.
//    Emit ONNXTransposeOp and ONNXReverseSequenceOp in reverseWeightConvTranspose()
// 2. Insert pads between elements in input data tensor if `strides` is not unit strides.
//    Emit ONNXSplitOp, ONNXPadOp, and ONNXConcatOp in insertPadsConvTransposeInput().
// 3. Calculate new pads attribute from kernel_shape, original pads, and dilation.
//    Emit no ONNX op in getPadsConvTranspose().
// 4. Use ONNXConv op using converted weight tensor, input data tensor, other
//    attributes.
// 5. Insert additional pads if needed.
//    Emit ONNXPadOp in insertAdditionalPadsConvTranspose().
// TODO: Consider to write C++ code instead of using tablegen because some code
//       are redundant. (eg. currently shapeHelper is calld multiple times.)
//===----------------------------------------------------------------------===//

def reverseWeightConvTranspose: NativeCodeCall<
  "onnx_mlir::reverseWeightTensor($_builder, $_loc, $0)">;

def decomposeIntoPhasedConvs: NativeCodeCall<
  "onnx_mlir::decomposeIntoPhasedConvs($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>(), $0,$1, $2, $3, $4, $5, $6, $7, $8, $9)">;

def ShouldDecomposeConvTransposeOpToPhasedConvs: Constraint<
  CPred<"onnx_mlir::ShouldDecomposeConvTransposeOpToPhasedConvs($_self, $0, $1, $2, $3)">,
  "X and W have static spatial dims, few more specific conditions and ConvTransposeOp phased conv decomposition is enabled"  
>;

def hasDefaultDilation: Constraint<
  CPred<"onnx_mlir::hasDefaultDilation($0)">,
  "Check if the dilation given is default"
>;

def ConvTransposeOpAsConv2dWithLReluPattern1: Pattern<
   (ONNXLeakyReluOp (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides), $alpha),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides, $alpha)
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(ShouldDecomposeConvTransposeOpToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 2)
>;

def ConvTransposeOpAsConv2dWithReluPattern1: Pattern<
   (ONNXReluOp (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides)),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides, (GetZeroFloatAttr))
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(ShouldDecomposeConvTransposeOpToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 2)
>;

def ConvTransposeOpAsConv2dPattern1: Pattern<
   (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides,  (GetNullFloatAttr))
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(hasNoActivationConsumer:$res),(ShouldDecomposeConvTransposeOpToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 1)
>;

#endif // ONNX_DECOMPOSE_CONVTRANSPOSE_PHASED
