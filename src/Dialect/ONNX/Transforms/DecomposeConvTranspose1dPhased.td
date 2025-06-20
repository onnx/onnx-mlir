// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_DECOMPOSE_CONVTRANSPOSE_1D_PHASED
#define ONNX_DECOMPOSE_CONVTRANSPOSE_1D_PHASED

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "Decompose.td"
include "DecomposeConvTransposePhased.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// ONNXConvTransposeOp is rewritten into ONNXConv op and other ONNX ops to
// convert input, weight, and pad.
//
// 1. Reverse elements for spatial axis in weight tensor.
//    Emit ONNXTransposeOp and ONNXReverseSequenceOp in reverseWeightConvTranspose()
// 2. splits the weights into phased conv weights.
// 3. combines the conv outputs in a specific way to produce the complete ofm

// TODO: Consider to write C++ code instead of using tablegen because some code
//       are redundant. (eg. currently shapeHelper is calld multiple times.)
//===----------------------------------------------------------------------===//

def decomposeConvT1dIntoPhasedConvs: NativeCodeCall<
  "onnx_mlir::decomposeConvT1dIntoPhasedConvs($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>(), $0,$1, $2, $3, $4, $5, $6, $7, $8, $9)">;

def ShouldDecomposeConvTransposeOp1dToPhasedConvs: Constraint<
  CPred<"onnx_mlir::ShouldDecomposeConvTransposeOp1dToPhasedConvs($_self, $0, $1, $2, $3)">,
  "X and W have static spatial dims, few more specific conditions and ConvTransposeOp phased conv decomposition is enabled"  
>;

def ConvTranspose1dOpAsConv1dWithLReluPattern1: Pattern<
   (ONNXLeakyReluOp (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides), $alpha),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeConvT1dIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides, $alpha)
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(ShouldDecomposeConvTransposeOp1dToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 2)
>;

def ConvTranspose1dOpAsConv1dWithReluPattern1: Pattern<
   (ONNXReluOp (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides)),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeConvT1dIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides, (GetZeroFloatAttr))
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(ShouldDecomposeConvTransposeOp1dToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 2)
>;

def ConvTranspose1dOpAsConv1dPattern1: Pattern<
   (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (decomposeConvT1dIntoPhasedConvs:$output $res, $x, $reversed_w, $b, $dilation, $group, $kernel_shape, $pads, $strides,  (GetNullFloatAttr))
   ],
   [(HasRankAndShape:$res), (hasDefaultDilation:$res $dilation),(hasNoActivationConsumer:$res),(ShouldDecomposeConvTransposeOp1dToPhasedConvs:$res $kernel_shape,$pads,$strides,$output_shape)], [],
   (addBenefit 1)
>;
#endif // ONNX_DECOMPOSE_CONVTRANSPOSE_1D_PHASED
