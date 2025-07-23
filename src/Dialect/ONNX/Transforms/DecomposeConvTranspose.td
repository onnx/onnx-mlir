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

def getFinalOutputFromFourConvOutput: NativeCodeCall<
  "onnx_mlir::getFinalOutputFromFourConvOutput($_builder, $_loc, $0.getDefiningOp<ONNXConvOp>(),$0,$1,$2,$3)">;

def ph1WeightForConv: NativeCodeCall<
  "onnx_mlir::ph1WeightTensor($_builder, $_loc, $0)">;

def ph2WeightForConv: NativeCodeCall<
  "onnx_mlir::ph2WeightTensor($_builder, $_loc, $0)">;

def ph3WeightForConv: NativeCodeCall<
  "onnx_mlir::ph3WeightTensor($_builder, $_loc, $0)">;

def ph4WeightForConv: NativeCodeCall<
  "onnx_mlir::ph4WeightTensor($_builder, $_loc, $0)">;

def getAttrForPhaseConv: NativeCodeCall<
  "onnx_mlir::getAttrForPhaseConv($_builder, $_loc, $0)">;

def getPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::getPadsConvTranspose($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>())">;

def HasUnitStrides: Constraint<
  CPred<"onnx_mlir::hasUnitStrides($_self)">,
  "has unit strides"
>;

def hasNoActivationConsumer: Constraint<
  CPred<"onnx_mlir::hasNoActivationConsumer($_self)">,
  "Op has no activation consumer"
>;

def hasDefaultDilation: Constraint<
  CPred<"onnx_mlir::hasDefaultDilation($0)">,
  "Check if the dilation given is default"
>;

def HasStaticSpatialDims: Constraint<
  CPred<"onnx_mlir::hasStaticSpatialDims($_self)">,
  "has static spatial dims"
>;

def insertPadsConvTransposeInput: NativeCodeCall<
  "onnx_mlir::insertPadsConvTransposeInput($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>(), $1)">;

def insertAdditionalPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::insertAdditionalPadsConvTranspose($_builder, $_loc, $0.getDefiningOp<ONNXConvOp>(), $0, $1.getDefiningOp<ONNXConvTransposeOp>())">;

def ShouldDecomposeConvTransposeOp: Constraint<
  CPred<"onnx_mlir::shouldDecomposeConvTransposeOp($_self)">,
  "X and W have static spatial dims and ConvTransposeOp decomposition is enabled"
>;


def ConvTransposeOpPattern1: Pattern<
   (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (getPadsConvTranspose:$new_pads $res),
    (ONNXConvOp:$conv_res $x, $reversed_w, $b,
       (GetNullStringAttr), $dilation, $group, $kernel_shape, $new_pads, $strides),
    (insertAdditionalPadsConvTranspose $conv_res, $res)
   ],
   [(HasRankAndShape:$res), (ShouldDecomposeConvTransposeOp:$res), (HasUnitStrides:$strides)], [],
   (addBenefit 1)
>;

def ConvTransposeOpPattern2: Pattern<
   (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, IntOne:$group, $kernel_shape,
      $output_padding, $output_shape, $pads, $strides),
   [
    (reverseWeightConvTranspose:$reversed_w $w),
    (insertPadsConvTransposeInput:$new_x $res, $x),
    (getPadsConvTranspose:$new_pads $res),
    (ONNXConvOp:$conv_res $new_x, $reversed_w, $b,
       (GetNullStringAttr), $dilation, $group, $kernel_shape, $new_pads, (GetNullArrayAttr)),
    (insertAdditionalPadsConvTranspose
        $conv_res, $res)
   ],
   [(HasRankAndShape:$res), (ShouldDecomposeConvTransposeOp:$res)], [],
   (addBenefit 0)
>;

#endif // ONNX_DECOMPOSE_CONVTRANSPOSE_PHASED
