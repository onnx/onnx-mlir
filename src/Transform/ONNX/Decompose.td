// SPDX-License-Identifier: Apache-2.0

//===- ONNXDecompose.td - Rewriting for decomposing ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
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
///    dag benefitsAdded = (addBenefit 0)
/// >;

def KeepdimsTrue
    : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, "
                     "/*isSigned=*/true), APInt(64, 1, /*isSigned=*/true))">;

def KeepdimsIsTrue
    : Constraint<CPred<"$_self.cast<IntegerAttr>().getSInt() == 1">,
          "keepdims attribute is true">;

def AttributeIsNull : Constraint<CPred<"! ($_self)">, "Attribute is null">;

def HasFloatType : Constraint<CPred<"(($_self).getType().dyn_cast<ShapedType>()"
                                    ".getElementType().isF32())">>;

def IsNoneType : Constraint<CPred<"(($_self).getType().isa<NoneType>())">>;

def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetNullFloatAttr :
        NativeCodeCall<"FloatAttr()">;

def GetNullIntegerAttr :
        NativeCodeCall<"IntegerAttr()">;

def GetNullStringAttr :
        NativeCodeCall<"StringAttr()">;

// Create a unit constant that will be used as none input.
def CreateUnitConstant
    : NativeCodeCall<"createUnitConstant($_builder, $_loc)">;

// Create a scalar DenseElementsAttr (rank 0) from a single attribute.
// E.g return type is tensor<f32> instead of tensor<0xf32> or tensor<1xf32>
def createScalarDenseAttrRank0
    : NativeCodeCall<"createScalarDenseAttr($_builder, $0)">;

// Create a DenseElementsAttr from a single attribute.
def createDenseArrayAttrFromSingleAttr
    : NativeCodeCall<"createDenseArrayAttr($_builder, $_builder.getArrayAttr($0))">;

// Create a DenseElementsAttr from an ArrayAttr.
def createDenseArrayAttr
    : NativeCodeCall<"createDenseArrayAttr($_builder, $0)">;

def createDenseArrayAttrOrEmpty
    : NativeCodeCall<"createDenseArrayAttrOrEmpty($_builder, $0)">;

def ScalerT : NativeCodeCall<"TypeAttr::get($_builder.getF32Type())">;

def noop_with_empty_axes
    : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, "
                     "/*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;

// Check if value is a dense constant
def IsDenseONNXConstant:
   Constraint<CPred<"isDenseONNXConstant($_self)">, "not a dense constant">;

// Create an ArrayAttr from a dense ConstantOp
def createArrayAttrFromConstantOp : NativeCodeCall<
  "createArrayAttrFromConstantOp($_builder, $0)">;

def createSequenceConstructOp : NativeCodeCall<
  "createSequenceConstructOp($_builder, $0, $1)">;

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
def ReduceL1OpPattern1
    : Pat<(ONNXReduceL1Op $oprd, $axes, $keepdims),
          (ONNXReduceSumOp(ONNXAbsOp $oprd),
              (CreateUnitConstant), $keepdims, (noop_with_empty_axes)), 
          [(AttributeIsNull:$axes)], (addBenefit 1)>;

def ReduceL1OpPattern2
    : Pat<(ONNXReduceL1Op $oprd, $axes, $keepdims),
          (ONNXReduceSumOp(ONNXAbsOp $oprd),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
def ReduceL2OpPattern
    : Pat<(ONNXReduceL2Op $oprd, $axes, $keepdims),
          (ONNXSqrtOp(ONNXReduceSumSquareOp $oprd, $axes, $keepdims))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
def ReduceLogSumOpPattern1
    : Pat<(ONNXReduceLogSumOp $oprd, $axes, $keepdims),
          (ONNXLogOp(ONNXReduceSumOp $oprd,
              (CreateUnitConstant), $keepdims, (noop_with_empty_axes))), 
          [(AttributeIsNull:$axes)], (addBenefit 1)>;

def ReduceLogSumOpPattern2
    : Pat<(ONNXReduceLogSumOp $oprd, $axes, $keepdims),
          (ONNXLogOp(ONNXReduceSumOp $oprd,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttrOrEmpty $axes)),
              $keepdims, (noop_with_empty_axes))),
          [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X
//===----------------------------------------------------------------------===//
// keepdims is true
def ReduceLogSumExpOpPattern1
    : Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
          (ONNXAddOp(ONNXReduceLogSumOp(ONNXExpOp(ONNXSubOp $oprd,
                         (ONNXReduceMaxOp
                             : $max $oprd, $axes, (KeepdimsTrue)))),
               $axes, $keepdims),
              $max),
          [(KeepdimsIsTrue
              : $keepdims)],
          (addBenefit 1)>;

// keepdims is false
def ReduceLogSumExpOpPattern2
    : Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
          (ONNXAddOp(ONNXReduceLogSumOp(ONNXExpOp(ONNXSubOp $oprd,
                         (ONNXReduceMaxOp
                             : $max $oprd, $axes, (KeepdimsTrue)))),
               $axes, $keepdims),
              (ONNXSqueezeV11Op $max, $axes)),
          [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
def ReduceSumSquareOpPattern1
    : Pat<(ONNXReduceSumSquareOp $oprd, $axes, $keepdims),
          (ONNXReduceSumOp (ONNXMulOp $oprd, $oprd),
              (CreateUnitConstant), $keepdims, (noop_with_empty_axes)), 
          [(AttributeIsNull:$axes)], (addBenefit 1)>;

def ReduceSumSquareOpPattern2
    : Pat<(ONNXReduceSumSquareOp $oprd, $axes, $keepdims),
          (ONNXReduceSumOp(ONNXMulOp $oprd, $oprd),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $axes)),
              $keepdims, (noop_with_empty_axes)),
          [], (addBenefit 0)>;

//===----------------------------------------------------------------------===//
// ONNXScalerOp %X, %Offest, %Scale
// x input, a offset, b scale
//===----------------------------------------------------------------------===//
// No attribute
def ScalerNullPattern
    : Pat<(ONNXScalerOp $x, $a, $b), (replaceWithValue $x),
          [(HasFloatType
               : $x),
            (AttributeIsNull
                : $a),
            (AttributeIsNull
                : $b)],
          (addBenefit 5)>;

// No attribute, input x not float type
def ScalerNullPattern2
    : Pat<(ONNXScalerOp $x, $a, $b), (ONNXCastOp $x, (ScalerT)),
          [(AttributeIsNull
               : $a),
            (AttributeIsNull
                : $b)],
          (addBenefit 4)>;

// No scale
def ScalerNoScalePattern
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXSubOp $x,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $a))),
          [(HasFloatType
               : $x),
            (AttributeIsNull
                : $b)],
          (addBenefit 3)>;

// No scale, input x not float type
def ScalerNoScalePattern2
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXSubOp(ONNXCastOp $x, (ScalerT)),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $a))),
          [(AttributeIsNull
              : $b)],
          (addBenefit 2)>;

// No offset
def ScalerNoOffsetPattern
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXMulOp $x,
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $b))),
          [(HasFloatType
               : $x),
            (AttributeIsNull
                : $a)],
          (addBenefit 3)>;

// No offset, input x not float type
def ScalerNoOffsetPattern2
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXMulOp(ONNXCastOp $x, (ScalerT)),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $b))),
          [(AttributeIsNull
              : $a)],
          (addBenefit 2)>;

// Normal ONNXScalerOp
def ScalerPattern
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXMulOp(ONNXSubOp $x,
               (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $a))),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $b))),
          [(HasFloatType
              : $x)],
          (addBenefit 1)>;

// Normal ONNXScalerOp, input x not float type
def ScalerPattern2
    : Pat<(ONNXScalerOp $x, $a, $b),
          (ONNXMulOp(ONNXSubOp(ONNXCastOp $x, (ScalerT)),
               (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $a))),
              (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $b))),
          [], (addBenefit 0)>;

// Express LogSoftmax using Log and Softmax.
def LogSoftmaxPattern
    : Pat<(ONNXLogSoftmaxOp $x, $axis), (ONNXLogOp(ONNXSoftmaxOp $x, $axis))>;

// Express Upsample using Resize.
def UpsamplePattern : Pat<
  (ONNXUpsampleOp $x, $scales, $mode),
  (ONNXResizeOp $x, (CreateUnitConstant), $scales, (CreateUnitConstant),
     (GetNullStringAttr), (GetNullFloatAttr), (GetNullIntegerAttr),
     (GetNullFloatAttr), $mode, (GetNullFloatAttr))
>;

def UpsampleV9Pattern : Pat<
  (ONNXUpsampleV9Op $x, $scales, $mode),
  (ONNXResizeOp $x, (CreateUnitConstant), $scales, (CreateUnitConstant),
     (GetNullStringAttr), (GetNullFloatAttr), (GetNullIntegerAttr),
     (GetNullFloatAttr), $mode, (GetNullFloatAttr))
>;

def UpsampleV7Pattern : Pat<
  (ONNXUpsampleV7Op $x, $mode, $scales),
  (ONNXResizeOp $x, (CreateUnitConstant),
     (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $scales)),
     (CreateUnitConstant), (GetNullStringAttr), (GetNullFloatAttr),
     (GetNullIntegerAttr), (GetNullFloatAttr), $mode, (GetNullFloatAttr))
>;

// Express PadV2 using the latest Pad version.
def PadV2Pattern : Pat<
  (ONNXPadV2Op $x, $modeAttr, $padsAttr, $valueAttr),
  (ONNXPadOp $x, (ONNXConstantOpFromDenseAttr(createDenseArrayAttr $padsAttr)),
     (ONNXConstantOpFromDenseAttr(createDenseArrayAttrFromSingleAttr $valueAttr)),
     $modeAttr)
>;

// Express PadV11 using the latest Pad version.
def PadV11Pattern : Pat<
  (ONNXPadV11Op $x, $pads, $mode, $value),
  (ONNXPadOp $x, $pads, $mode, $value)
>;

def ResizeV10Pattern: Pat<
   (ONNXResizeV10Op:$res $x, $scales, $mode),
   (ONNXResizeOp $x, (createNoneFloatConstant $res),
        $scales, (createNoneIntegerConstant $res),
       /*coordinate_transformation_mode*/(GetNullStringAttr),
       /*cubic_coeff_a*/(GetNullFloatAttr),
       /*exclude_outside*/(GetNullIntegerAttr),
       /*exttrapolation_value*/(GetNullFloatAttr),
       /*mode*/$mode,
       /*nearest_mode*/(GetNullStringAttr))
>;

def ResizeV11Pattern: Pat<
   (ONNXResizeV11Op:$res $x, $roi, $scales, $size,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $exttrapolation_value, $mode,
                    $nearest_mode),
   (ONNXResizeOp    $x, $roi, $scales, $size,
                    $coordinate_transformation_node, $cubic_coeff_a,
                    $exclude_outside, $exttrapolation_value, $mode,
                    $nearest_mode)
>;

def SequenceConstructPattern1: Pat<
    (ONNXSequenceConstructOp:$res $x1),
    (createSequenceConstructOp
        (ONNXSequenceEmptyOp (GetNullIntegerAttr), (returnType $res)),
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

#endif // ONNX_DECOMPOSE
