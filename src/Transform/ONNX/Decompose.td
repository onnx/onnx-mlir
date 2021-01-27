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
include "src/Dialect/ONNX/ONNXOps.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def KeepdimsTrue: NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 1, /*isSigned=*/true))">;

def KeepdimsIsTrue:
    Constraint<CPred<"$_self.cast<IntegerAttr>().getSInt() == 1">,
  "keepdims attribute is true">;

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
def ReduceL1OpPattern: Pat<(ONNXReduceL1Op $oprd, $axes, $keepdims),
                           (ONNXReduceSumOp (ONNXAbsOp $oprd), $axes, $keepdims)>;

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
def ReduceL2OpPattern: Pat<(ONNXReduceL2Op $oprd, $axes, $keepdims),
                           (ONNXSqrtOp (ONNXReduceSumSquareOp $oprd, $axes, $keepdims))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
def ReduceLogSumOpPattern: Pat<(ONNXReduceLogSumOp $oprd, $axes, $keepdims),
                           (ONNXLogOp (ONNXReduceSumOp $oprd, $axes, $keepdims))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X
//===----------------------------------------------------------------------===//
// keepdims is true
def ReduceLogSumExpOpPattern1: Pat<
  (ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
  (ONNXAddOp
    (ONNXReduceLogSumOp
      (ONNXExpOp
         (ONNXSubOp $oprd, (ONNXReduceMaxOp:$max $oprd, $axes, (KeepdimsTrue)))),
      $axes,
      $keepdims),
    $max),
  [(KeepdimsIsTrue:$keepdims)],
  (addBenefit 1)
>;

// keepdims is false 
def ReduceLogSumExpOpPattern2: Pat<
  (ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
  (ONNXAddOp
    (ONNXReduceLogSumOp
      (ONNXExpOp
         (ONNXSubOp $oprd, (ONNXReduceMaxOp:$max $oprd, $axes, (KeepdimsTrue)))),
      $axes,
      $keepdims),
    (ONNXSqueezeOp $max, $axes)),
  [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
def ReduceSumSquareOpPattern: Pat<(ONNXReduceSumSquareOp $oprd, $axes, $keepdims),
                                  (ONNXReduceSumOp (ONNXMulOp $oprd, $oprd), $axes, $keepdims)>;

//===----------------------------------------------------------------------===//
// ONNXScalerOp %X, %Offest, %Scale
// x input, a offset, b scale
//===----------------------------------------------------------------------===//
// Useful test definitions.
def AttributeIsNull :
    Constraint<CPred<"! ($_self)">,
  "Attribute is null">;

def HasFloatType : Constraint<CPred<"(($_self).getType().dyn_cast<ShapedType>().getElementType().isF32())">>;

def GetNullAttr :
	NativeCodeCall<"Attribute()">;
// TODO: Remove GetNullPtr once corresponding attributes are supported
def GetNullPtr :
  NativeCodeCall<"nullptr">;

// Create a DenseElementsAttr from an ArrayAttr.
def createDenseArrayAttr:
  NativeCodeCall<"createDenseArrayAttr($_builder, $0)">;

def ScalerT : NativeCodeCall<"TypeAttr::get($_builder.getF32Type())">;

// No attribute
def ScalerNullPattern : Pat<
	(ONNXScalerOp $x, $a, $b),
	(replaceWithValue $x),
  [(HasFloatType:$x), (AttributeIsNull:$a), (AttributeIsNull:$b)],
  (addBenefit 5)>;

// No attribute, input x not float type
def ScalerNullPattern2 : Pat<
	(ONNXScalerOp $x, $a, $b),
	(ONNXCastOp $x, (ScalerT)),
	[(AttributeIsNull:$a), (AttributeIsNull:$b)],
  (addBenefit 4)>;

// No scale
def ScalerNoScalePattern : Pat<
  (ONNXScalerOp $x, $a, $b),
    (ONNXSubOp $x,
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a),
      (GetNullPtr), (GetNullPtr), (GetNullPtr),
      (GetNullPtr), (GetNullPtr), (GetNullPtr))),
  [(HasFloatType:$x), (AttributeIsNull:$b)],
  (addBenefit 3)>;

// No scale, input x not float type
def ScalerNoScalePattern2 : Pat<
 (ONNXScalerOp $x, $a, $b),
  (ONNXSubOp
   (ONNXCastOp $x, (ScalerT)),
   (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a),
     (GetNullPtr), (GetNullPtr), (GetNullPtr),
     (GetNullPtr), (GetNullPtr), (GetNullPtr))
  ),
  [(AttributeIsNull:$b)],
  (addBenefit 2)>;

// No offset
def ScalerNoOffsetPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp $x,
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b),
      (GetNullPtr), (GetNullPtr), (GetNullPtr),
      (GetNullPtr), (GetNullPtr), (GetNullPtr))),
  [(HasFloatType:$x), (AttributeIsNull:$a)],
  (addBenefit 3)>;

// No offset, input x not float type
def ScalerNoOffsetPattern2 : Pat<
 (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp
   (ONNXCastOp $x, (ScalerT)),
   (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b),
     (GetNullPtr), (GetNullPtr), (GetNullPtr),
     (GetNullPtr), (GetNullPtr), (GetNullPtr))),
  [(AttributeIsNull:$a)],
  (addBenefit 2)>;

// Normal ONNXScalerOp
def ScalerPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp 
    (ONNXSubOp $x,
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a),
        (GetNullPtr), (GetNullPtr), (GetNullPtr),
        (GetNullPtr), (GetNullPtr), (GetNullPtr))),
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b),
      (GetNullPtr), (GetNullPtr), (GetNullPtr),
      (GetNullPtr), (GetNullPtr), (GetNullPtr))),
  [(HasFloatType:$x)],
  (addBenefit 1)>;

// Normal ONNXScalerOp, input x not float type
def ScalerPattern2 : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp 
    (ONNXSubOp (ONNXCastOp $x, (ScalerT)),
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a),
        (GetNullPtr), (GetNullPtr), (GetNullPtr),
        (GetNullPtr), (GetNullPtr), (GetNullPtr))),
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b),
      (GetNullPtr), (GetNullPtr), (GetNullPtr),
      (GetNullPtr), (GetNullPtr), (GetNullPtr))),
  [],
  (addBenefit 0)>;

// Express LogSoftmax using Log and Softmax.
def LogSoftmaxPattern : Pat<
  (ONNXLogSoftmaxOp $x, $axis),
  (ONNXLogOp (ONNXSoftmaxOp $x, $axis))
>;

#endif // ONNX_DECOMPOSE
