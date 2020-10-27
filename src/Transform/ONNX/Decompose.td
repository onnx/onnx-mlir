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
// ONNXReduceLogSumExpOp %X = ONNXReduceLogSumOp (ONNXExpOp %X)
//===----------------------------------------------------------------------===//
def ReduceLogSumExpOpPattern: Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
                                  (ONNXReduceLogSumOp (ONNXExpOp $oprd), $axes, $keepdims)>;

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

// Create a DenseElementsAttr from an ArrayAttr.
def createDenseArrayAttr:
  NativeCodeCall<"createDenseArrayAttr($_builder, $0)">;

def ScalerT : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 1, /*isSigned=*/true))">;

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
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))),
  [(HasFloatType:$x), (AttributeIsNull:$b)],
  (addBenefit 3)>;

// No scale, input x not float type
def ScalerNoScalePattern2 : Pat<
 (ONNXScalerOp $x, $a, $b),
  (ONNXSubOp
   (ONNXCastOp $x, (ScalerT)),
   (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))
  ),
  [(AttributeIsNull:$b)],
  (addBenefit 2)>;

// No offset
def ScalerNoOffsetPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp $x,
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b))),
  [(HasFloatType:$x), (AttributeIsNull:$a)],
  (addBenefit 3)>;

// No offset, input x not float type
def ScalerNoOffsetPattern2 : Pat<
 (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp
   (ONNXCastOp $x, (ScalerT)),
   (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b))),
  [(AttributeIsNull:$a)],
  (addBenefit 2)>;

// Normal ONNXScalerOp
def ScalerPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp 
    (ONNXSubOp $x,
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))),
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b))),
  [(HasFloatType:$x)],
  (addBenefit 1)>;

// Normal ONNXScalerOp, input x not float type
def ScalerPattern2 : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp 
    (ONNXSubOp (ONNXCastOp $x, (ScalerT)),
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))),
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b))),
  [],
  (addBenefit 0)>;

// Express LogSoftmax using Log and Softmax.
def LogSoftmaxPattern : Pat<
  (ONNXLogSoftmaxOp $x, $axis),
  (ONNXLogOp (ONNXSoftmaxOp $x, $axis))
>;

#endif // ONNX_DECOMPOSE
