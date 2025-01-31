// SPDX-License-Identifier: Apache-2.0

//===- ONNXDecompose.td - Rewriting for decomposing ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_DECOMPOSE
#define ONNX_DECOMPOSE

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

def KeepdimsTrue
    : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, "
                     "/*isSigned=*/true), APInt(64, 1, /*isSigned=*/true))">;

def KeepdimsIsTrue
    : Constraint<CPred<"mlir::cast<IntegerAttr>($_self).getSInt() == 1">,
          "keepdims attribute is true">;

def ONNXConstantOpNormalize: NativeCodeCall<
  "onnx_mlir::normalizeConstantOp($_builder, $0, $1)">;

def AttributeIsNull : Constraint<CPred<"! ($_self)">, "Attribute is null">;

def AttributeIsNotNull : Constraint<CPred<"($_self)">, "Attribute is not null">;

def HasFloatType : Constraint<CPred<"(mlir::dyn_cast<ShapedType>(($_self).getType())"
                                    ".getElementType().isF32())">>;

def IsNoneType : Constraint<CPred<"mlir::isa<NoneType>(($_self).getType())">>;

def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetNullFloatAttr : NativeCodeCall<"FloatAttr()">;

def GetNullIntegerAttr : NativeCodeCall<"IntegerAttr()">;

def GetNullStringAttr : NativeCodeCall<"StringAttr()">;

def GetNullArrayAttr :  NativeCodeCall<"ArrayAttr()">;

// Create a unit constant that will be used as none input.
def CreateNoneValue : NativeCodeCall<"$_builder.create<ONNXNoneOp>($_loc)">;

// Create a scalar DenseElementsAttr (rank 0) from a single attribute.
// E.g return type is tensor<f32> instead of tensor<0xf32> or tensor<1xf32>
def createScalarDenseAttrRank0
    : NativeCodeCall<"::onnx_mlir::createScalarDenseAttr($_builder, $0)">;

// Create a scalar DenseElementsAttr of tensor<dtype> from an ElementsAttr.
// The input ElementsAttr must have only one element. Otherwise only the first
// element is used to create the scalar DenseElementsAttr
def ReshapeElementsAttrToRank0 : NativeCodeCall<
    "onnx_mlir::OnnxElementsAttrBuilder($0.getContext()).reshape(cast<ElementsAttr>($0), {})">;

def ReplaceSequenceAt :  NativeCodeCall<
    "onnx_mlir::replaceSequenceAt($_builder, $_loc, $0)">;

def UpgradeGridSampleV16Mode :  NativeCodeCall<
    "onnx_mlir::upgradeGridSampleV16Mode($_builder, $0)">;
    
def CanSequenceAtBeReplaced :
   Constraint<CPred<"::onnx_mlir::canSequenceAtBeReplaced($_self)">, "check whether the SequenceAt can be replaced with split">;

// Create a DenseElementsAttr from a single attribute.
def createDenseArrayAttrFromSingleAttr
    : NativeCodeCall<"::onnx_mlir::createDenseArrayAttr($_builder, $_builder.getArrayAttr($0))">;

// Create a DenseElementsAttr from an ArrayAttr.
def createDenseArrayAttr
    : NativeCodeCall<"::onnx_mlir::createDenseArrayAttr($_builder, $0)">;

def createDenseArrayAttrOrEmpty
    : NativeCodeCall<"::onnx_mlir::createDenseArrayAttrOrEmpty($_builder, $0)">;

def ScalerT : NativeCodeCall<"TypeAttr::get($_builder.getF32Type())">;

def noop_with_empty_axes
    : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, "
                     "/*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;

// Check if value is a dense constant
def IsDenseONNXConstant:
   Constraint<CPred<"::onnx_mlir::isDenseONNXConstant($_self)">, "not a dense constant">;

def createSequenceConstructOp : NativeCodeCall<
  "::onnx_mlir::createSequenceConstructOp($_builder, $0, $1)">;

def ONNXDataType : NativeCodeCall<
  "IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), "
  "::onnx_mlir::mlirTypeToOnnxType(mlir::cast<ShapedType>($0.getType().front()).getElementType()))">;

// Check if value has a type that is ranked with a shape.
def HasRankAndShape:
  Constraint<CPred<"::onnx_mlir::isRankedShapedType($_self.getType())">,
  "A value has rank and shape">;

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
def ReduceL1OpPattern
    : Pat<(ONNXReduceL1Op $oprd, $axes, $keepdims, $noop),
          (ONNXReduceSumOp(ONNXAbsOp $oprd), $axes, $keepdims, $noop)>;

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
def ReduceL2OpPattern
    : Pat<(ONNXReduceL2Op $oprd, $axes, $keepdims, $noop),
          (ONNXSqrtOp(ONNXReduceSumSquareOp $oprd, $axes, $keepdims, $noop))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
def ReduceLogSumOpPattern
    : Pat<(ONNXReduceLogSumOp $oprd, $axes, $keepdims, $noop),
          (ONNXLogOp(ONNXReduceSumOp $oprd, $axes, $keepdims, $noop))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X
//===----------------------------------------------------------------------===//
// keepdims is true
def ReduceLogSumExpOpPattern1
    : Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims, $noop),
          (ONNXAddOp(ONNXReduceLogSumOp(ONNXExpOp(ONNXSubOp $oprd,
                         (ONNXReduceMaxOp
                             : $max $oprd, $axes, (KeepdimsTrue), $noop))),
               $axes, $keepdims, $noop),
              $max),
          [(KeepdimsIsTrue
              : $keepdims)], [],
          (addBenefit 1)>;

// keepdims is false
def ReduceLogSumExpOpPattern2
    : Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims, $noop),
          (ONNXAddOp(ONNXReduceLogSumOp(ONNXExpOp(ONNXSubOp $oprd,
                         (ONNXReduceMaxOp
                             : $max $oprd, $axes, (KeepdimsTrue), $noop))),
               $axes, $keepdims, $noop),
              (ONNXSqueezeOp $max, $axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
def ReduceSumSquareOpPattern
    : Pat<(ONNXReduceSumSquareOp $oprd, $axes, $keepdims, $noop),
          (ONNXReduceSumOp (ONNXMulOp $oprd, $oprd), $axes, $keepdims, $noop)>;

//===----------------------------------------------------------------------===//
// ONNXScalerOp %X, %Offest, %Scale
// x input, a offset, b scale
//===----------------------------------------------------------------------===//
// No attribute
def ScalerNullPattern
    : Pat<(ONNXScalerOp $x, $offset, $scale), (replaceWithValue $x),
          [(HasFloatType:$x), (AttributeIsNull:$offset), (AttributeIsNull:$scale)], [],
          (addBenefit 5)>;

// No attribute, input x not float type
def ScalerNullPattern2
    : Pat<(ONNXScalerOp $x, $offset, $scale), (ONNXCastOp $x, (GetNullIntegerAttr), (ScalerT)),
          [(AttributeIsNull:$offset), (AttributeIsNull:$scale)], [],
          (addBenefit 4)>;

// No scale
def ScalerNoScalePattern
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXSubOp $x,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $offset))),
          [(HasFloatType:$x), (AttributeIsNull:$scale)], [],
          (addBenefit 3)>;

// No scale, input x not float type
def ScalerNoScalePattern2
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXSubOp(ONNXCastOp $x, (GetNullIntegerAttr), (ScalerT)),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $offset))),
          [(AttributeIsNull:$scale)], [],
          (addBenefit 2)>;

// No offset
def ScalerNoOffsetPattern
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXMulOp $x,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scale))),
          [(HasFloatType:$x), (AttributeIsNull:$offset)], [],
          (addBenefit 3)>;

// No offset, input x not float type
def ScalerNoOffsetPattern2
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXMulOp(ONNXCastOp $x, (GetNullIntegerAttr), (ScalerT)),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scale))),
          [(AttributeIsNull:$offset)], [],
          (addBenefit 2)>;

// Normal ONNXScalerOp
def ScalerPattern
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXMulOp(ONNXSubOp $x,
               (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $offset))),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scale))),
          [(HasFloatType:$x)], [],
          (addBenefit 1)>;

// Normal ONNXScalerOp, input x not float type
def ScalerPattern2
    : Pat<(ONNXScalerOp $x, $offset, $scale),
          (ONNXMulOp(ONNXSubOp(ONNXCastOp $x, (GetNullIntegerAttr), (ScalerT)),
               (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $offset))),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scale))),
          [], [], (addBenefit 0)>;

// Express LogSoftmax using Log and Softmax.
def LogSoftmaxPattern
    : Pat<(ONNXLogSoftmaxOp $x, $axis), (ONNXLogOp(ONNXSoftmaxOp $x, $axis))>;

// Express Upsample using Resize.
def UpsamplePattern : Pat<
  (ONNXUpsampleOp $x, $scales, $mode),
  (ONNXResizeOp $x, (CreateNoneValue), $scales, (CreateNoneValue),
     (GetNullIntegerAttr), (GetNullArrayAttr), (GetNullStringAttr),
     (GetNullFloatAttr), (GetNullIntegerAttr), (GetNullFloatAttr), (GetNullStringAttr),
     $mode, (GetNullFloatAttr))
>;

def UpsampleV7Pattern : Pat<
  (ONNXUpsampleV7Op $x, $mode, $scales),
  (ONNXResizeOp $x, (CreateNoneValue),
     (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scales)),
     (CreateNoneValue), (GetNullIntegerAttr), (GetNullArrayAttr), (GetNullStringAttr),
     (GetNullFloatAttr), (GetNullIntegerAttr), (GetNullFloatAttr), (GetNullStringAttr),
     $mode, (GetNullFloatAttr))
>;

// Express PadV2 using the latest Pad version.
def PadV2Pattern : Pat<
  (ONNXPadV2Op $x, $modeAttr, $padsAttr, $valueAttr),
  (ONNXPadOp $x, (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $padsAttr)),
     (ONNXConstantOpFromDenseAttr(createDenseArrayAttrFromSingleAttr $valueAttr)),
     (CreateNoneValue),
     $modeAttr)
>;

// Express PadV11 using the latest Pad version.
def PadV11Pattern : Pat<
  (ONNXPadV11Op $x, $pads, $mode, $value),
  (ONNXPadOp $x, $pads, $mode, (CreateNoneValue), $value)
>;

def PadV13Pattern : Pat<
  (ONNXPadV13Op $x, $pads, $value, $mode),
  (ONNXPadOp $x, $pads, $value, (CreateNoneValue), $mode)
>;

def PadV18Pattern : Pat<
  (ONNXPadV18Op $x, $pads, $value, $axes, $mode),
  (ONNXPadOp $x, $pads, $value, $axes, $mode)
>;

def ReduceMaxV18Pattern : Pat<
  (ONNXReduceMaxV18Op $data, $axes, $keepdims, $noop_with_empty_axes),
  (ONNXReduceMaxOp $data, $axes, $keepdims, $noop_with_empty_axes)
>;

def ReduceMinV18Pattern : Pat<
  (ONNXReduceMinV18Op $data, $axes, $keepdims, $noop_with_empty_axes),
  (ONNXReduceMinOp $data, $axes, $keepdims, $noop_with_empty_axes)
>;

def ResizeV10Pattern: Pat<
   (ONNXResizeV10Op:$res $x, $scales, $mode),
   (ONNXResizeOp $x, (CreateNoneValue),
        $scales, (CreateNoneValue),
       /*antialias*/(GetNullIntegerAttr),
       /*axes*/(GetNullArrayAttr),
       /*coordinate_transformation_mode*/(GetNullStringAttr),
       /*cubic_coeff_a*/(GetNullFloatAttr),
       /*exclude_outside*/(GetNullIntegerAttr),
       /*extrapolation_value*/(GetNullFloatAttr),
       /*keep_aspect_ratio_policy*/(GetNullStringAttr),
       /*mode*/$mode,
       /*nearest_mode*/(GetNullStringAttr))
>;

def ResizeV11Pattern: Pat<
   (ONNXResizeV11Op:$res $x, $roi, $scales, $size,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value, $mode,
                    $nearest_mode),
   (ONNXResizeOp    $x, $roi, $scales, $size,
                    /*antialias*/(GetNullIntegerAttr),
                    /*axes*/(GetNullArrayAttr),
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value,
                    /*keep_aspect_ratio_policy*/(GetNullStringAttr),
                    $mode, $nearest_mode)
>;

def ResizeV13Pattern: Pat<
   (ONNXResizeV13Op:$res $x, $roi, $scales, $size,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value, $mode,
                    $nearest_mode),
   (ONNXResizeOp    $x, $roi, $scales, $size,
                    /*antialias*/(GetNullIntegerAttr),
                    /*axes*/(GetNullArrayAttr),
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value,
                    /*keep_aspect_ratio_policy*/(GetNullStringAttr),
                    $mode, $nearest_mode)
>;

def ResizeV18Pattern: Pat<
   (ONNXResizeV18Op:$res $x, $roi, $scales, $size, $antialias, $axes,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value,
                    $keep_aspect_ratio_policy, $mode, $nearest_mode),
   (ONNXResizeOp    $x, $roi, $scales, $size, $antialias, $axes,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $extrapolation_value,
                    $keep_aspect_ratio_policy, $mode, $nearest_mode)
>;

def SequenceConstructPattern1: Pat<
    (ONNXSequenceConstructOp:$res $x1),
    (createSequenceConstructOp
        (ONNXSequenceEmptyOp (ONNXDataType $x1), (returnType $res)),
        $x1)
>;

// Express Clip V6 using Clip V11.
def ClipV6Pattern : Pat<
  (ONNXClipV6Op $x, $maxAttr, $minAttr),
  (ONNXClipV11Op $x, (ONNXConstantOpFromDenseAttr(createScalarDenseAttrRank0 $minAttr)),
     (ONNXConstantOpFromDenseAttr(createScalarDenseAttrRank0 $maxAttr)))
>;

// Express Clip V11 using Clip V12.
def ClipV11Pattern : Pat<
  (ONNXClipV11Op $x, $min, $max),
  (ONNXClipV12Op $x, $min, $max)
>;

// Express Clip V12 using Clip V13 (the lastest).
def ClipV12Pattern : Pat<
  (ONNXClipV12Op $x, $min, $max),
  (ONNXClipOp $x, $min, $max)
>;

// Rewrite GridSample 16 to GridSample 22
def GridSampleV16Pattern : Pat<
  (ONNXGridSampleV16Op $x, $grid, $align_corners, $mode, $padding_mode),
  (ONNXGridSampleOp $x, $grid, $align_corners, (UpgradeGridSampleV16Mode $mode), $padding_mode)
>;

def DFTV17Pattern : Pat<
  (ONNXDFTV17Op $x, $dft_length, $axis, $inverse, $onesided),
  (ONNXDFTOp $x, $dft_length, (ONNXConstantOpFromDenseAttr(createScalarDenseAttrRank0 $axis)), $inverse, $onesided)
>;

def SplitV11PatternNoAttr : Pat<
  (ONNXSplitV11Op $x, $axis, $split),
  (ONNXSplitV13Op $x, (CreateNoneValue), $axis),
  [(AttributeIsNull:$split)], [], (addBenefit 1)
>;

def SplitV11Pattern : Pat<
  (ONNXSplitV11Op $x, $axis, $split),
  (ONNXSplitV13Op $x, (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $split)), $axis),
  [], [], (addBenefit 0)
>;

def SplitV13Pattern : Pat<
  (ONNXSplitV13Op $x, $split, $axis),
  (ONNXSplitOp $x, $split, $axis, (GetNullIntegerAttr))
>;

def SqueezeV11PatternNoAttr : Pat<
  (ONNXSqueezeV11Op $x, $axes),
  (ONNXSqueezeOp $x, (CreateNoneValue)),
  [(AttributeIsNull:$axes)], [], (addBenefit 1)
>;

def SqueezeV11Pattern : Pat<
  (ONNXSqueezeV11Op $x, $axes),
  (ONNXSqueezeOp $x, (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes))),
  [], [], (addBenefit 0)
>;

def UnsqueezeV11Pattern : Pat<
  (ONNXUnsqueezeV11Op $x, $axes),
  (ONNXUnsqueezeOp $x, (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)))
>;

// Express Scatter (deprecated) using ScatterElements.
def ScatterPattern : Pat<
  (ONNXScatterOp $data, $indices, $updates, $axis),
  (ONNXScatterElementsOp $data, $indices, $updates, $axis, (GetNullStringAttr))
>;

//===----------------------------------------------------------------------===//
// ONNXReduceL1V13Op %X -> ONNXReduceL1Op %X =
//===----------------------------------------------------------------------===//
def ReduceL1V13OpPattern1
    : Pat<(ONNXReduceL1V13Op $oprd, $axes, $keepdims),
          (ONNXReduceL1Op $oprd,
              (CreateNoneValue), $keepdims, (noop_with_empty_axes)),
          [(AttributeIsNull:$axes)], [], (addBenefit 1)>;

def ReduceL1V13OpPattern2
    : Pat<(ONNXReduceL1V13Op $oprd, $axes, $keepdims),
          (ONNXReduceL1Op $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceL2V13Op %X -> ONNXReduceL2Op %X =
//===----------------------------------------------------------------------===//
def ReduceL2V13OpPattern1
    : Pat<(ONNXReduceL2V13Op $oprd, $axes, $keepdims),
          (ONNXReduceL2Op $oprd,
              (CreateNoneValue), $keepdims, (noop_with_empty_axes)),
          [(AttributeIsNull:$axes)], [], (addBenefit 1)>;

def ReduceL2V13OpPattern2
    : Pat<(ONNXReduceL2V13Op $oprd, $axes, $keepdims),
          (ONNXReduceL2Op $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumV13Op %X -> ONNXReduceLogSumOp %X =
//===----------------------------------------------------------------------===//
def ReduceLogSumV13OpPattern1
    : Pat<(ONNXReduceLogSumV13Op $oprd, $axes, $keepdims),
          (ONNXReduceLogSumOp $oprd,
              (CreateNoneValue), $keepdims, (noop_with_empty_axes)),
          [(AttributeIsNull:$axes)], [], (addBenefit 1)>;

def ReduceLogSumV13OpPattern2
    : Pat<(ONNXReduceLogSumV13Op $oprd, $axes, $keepdims),
          (ONNXReduceLogSumOp $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpV13Op %X -> ONNXReduceLogSumExpOp %X =
//===----------------------------------------------------------------------===//
def ReduceLogSumExpV13OpPattern1
    : Pat<(ONNXReduceLogSumExpV13Op $oprd, $axes, $keepdims),
          (ONNXReduceLogSumExpOp $oprd,
              (CreateNoneValue), $keepdims, (noop_with_empty_axes)),
          [(AttributeIsNull:$axes)], [], (addBenefit 1)>;

def ReduceLogSumExpV13OpPattern2
    : Pat<(ONNXReduceLogSumExpV13Op $oprd, $axes, $keepdims),
          (ONNXReduceLogSumExpOp $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareV13Op %X -> ONNXReduceSumSquareOp %X =
//===----------------------------------------------------------------------===//
def ReduceSumSquareV13OpPattern1
    : Pat<(ONNXReduceSumSquareV13Op $oprd, $axes, $keepdims),
          (ONNXReduceSumSquareOp $oprd,
              (CreateNoneValue), $keepdims, (noop_with_empty_axes)),
          [(AttributeIsNull:$axes)], [], (addBenefit 1)>;

def ReduceSumSquareV13OpPattern2
    : Pat<(ONNXReduceSumSquareV13Op $oprd, $axes, $keepdims),
          (ONNXReduceSumSquareOp $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//

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

def getPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::getPadsConvTranspose($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>())">;

def HasUnitStrides: Constraint<
  CPred<"onnx_mlir::hasUnitStrides($_self)">,
  "has unit strides"
>;

def ShouldDecomposeConvTransposeOp: Constraint<
  CPred<"onnx_mlir::shouldDecomposeConvTransposeOp($_self)">,
  "X and W have static spatial dims and ConvTransposeOp decomposition is enabled"
>;

def HasStaticSpatialDims: Constraint<
  CPred<"onnx_mlir::hasStaticSpatialDims($_self)">,
  "has static spatial dims"
>;

def insertPadsConvTransposeInput: NativeCodeCall<
  "onnx_mlir::insertPadsConvTransposeInput($_builder, $_loc, $0.getDefiningOp<ONNXConvTransposeOp>(), $1)">;

def insertAdditionalPadsConvTranspose: NativeCodeCall<
  "onnx_mlir::insertAdditionalPadsConvTranspose($_builder, $_loc, $0.getDefiningOp<ONNXConvOp>(), $0, $1.getDefiningOp<ONNXConvTransposeOp>())">;

def ConvTransposeOpPattern1: Pattern<
   (ONNXConvTransposeOp:$res $x, $w, $b,
      $auto_pad, $dilation, $group, $kernel_shape,
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
      $auto_pad, $dilation, $group, $kernel_shape,
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

//===----------------------------------------------------------------------===//
// Rewrite `ONNXConstantOfShapeOp {value} (%shape)` into
//         `ONNXExpandOp(ONNXConstantOp {value}, %shape)
//===----------------------------------------------------------------------===//

def ConstantOfShapePattern: Pat<
  (ONNXConstantOfShapeOp:$res $shape, $value),
  (ONNXExpandOp (ONNXConstantOpFromDenseAttr (ReshapeElementsAttrToRank0 $value)),
                $shape)
>;

//===----------------------------------------------------------------------===//
// Normalize ONNXConstantOp to use ElementAttrs
//===----------------------------------------------------------------------===//

def ConstantOpNormalizationPattern1: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $floatAttr),
   [(AttributeIsNotNull:$floatAttr)]
>;

def ConstantOpNormalizationPattern2: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $intAttr),
   [(AttributeIsNotNull:$intAttr)]
>;

def ConstantOpNormalizationPattern3: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $stringAttr),
   [(AttributeIsNotNull:$stringAttr)]
>;

def ConstantOpNormalizationPattern4: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $floatsAttr),
   [(AttributeIsNotNull:$floatsAttr)]
>;

def ConstantOpNormalizationPattern5: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $intsAttr),
   [(AttributeIsNotNull:$intsAttr)]
>;

def ConstantOpNormalizationPattern6: Pat<
   (ONNXConstantOp:$res $sparseAttr, $denseAttr, $floatAttr, $floatsAttr, $intAttr,
       $intsAttr, $stringAttr, $stringsAttr),
   (ONNXConstantOpNormalize $res, $stringsAttr),
   [(AttributeIsNotNull:$stringsAttr)]
>;

// Optimize for the pattern coming from torch.nn.LSTM exported from pytorch
//     %32 = "onnx.SplitToSequence"(%30, %27) {axis = 0 : si64, keepdims = 0 : si64, onnx_node_name = "n1"} : (tensor<1x1x100xf32>, tensor<i64>) -> !onnx.Seq<tensor<1x1x100xf32>>
//    %33 = "onnx.SequenceAt"(%32, %26) {onnx_node_name = "n0"} : (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x100xf32>
// When shape and size/axis related value are constant, this sequence of code
// can be translated into onnx.slice

def ReplaceSequenceAtPattern: Pat<
   (ONNXSequenceAtOp:$res $seq, $position),
   (ReplaceSequenceAt $res),
   [(CanSequenceAtBeReplaced:$res)]
>;

#endif // ONNX_DECOMPOSE
