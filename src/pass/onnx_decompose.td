//===----------------------------------------------------------------------===//
//=- onnx_decompose.td - Rewriting for decomposing ONNX Ops -*- tablegen -*===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//

#ifndef ONNX_DECOMPOSE
#define ONNX_DECOMPOSE

#ifndef OP_BASE
include "dialect/onnx/onnx.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

// Create a StringAttr from a string.
class StringAttrOfValue<string val>: NativeCodeCall<"$_builder.getStringAttr(\"" # val # "\")">;

// Create a FloatAttr from an interger value.
// It seems Table-gen does not support `float` type, so we can not pass a float value.
class FloatAttrOfValue<int val> : NativeCodeCall<"FloatAttr::get($0.getType().cast<TensorType>().getElementType(), " # val # ")">;

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in MaxPoolSingleOut.
class ArrayAttrOfZeroWithTrail<int trailCount> : NativeCodeCall<"createArrayAttrOfZeroWithTrail($_builder, $0, " # trailCount # ")">;

// Pad a ArrayAttr with trails of zeros.
// This function is used for padding attribute in MaxPoolSingleOut.
class PadArrayAttrWithZeroTrail<int trailCount> : NativeCodeCall<"padArrayAttrWithZeroTrail($_builder, $0," # trailCount # ")">;

// Check whether an ArrayAttr contains non-zero values or not.
def HasNonZeroInArrayAttr : Constraint<CPred<"hasNonZeroInArrayAttr($_self)">, "has non-zero elements">;
def HasAllZeroInArrayAttr : Constraint<CPred<"!hasNonZeroInArrayAttr($_self)">, "has non-zero elements">;

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
// Rewrite:
// %0 = onnx.MaxPoolSingleOutOp(%D : tensor<DShape>)
//     {pads = [b0, b1, ... bK, e0, e1, ..., eK]} ->
//         tensor<OutShape>
//
// as:
// %0 = onnx.PadConstantValuePadOp(%D)
//     {pads = [0, 0, b0, b1, ... bK, 0, 0, e0, e1, ..., eK]} ->
//     tensor<DPaddedShape>
// %1 = onnx.MaxPoolSingleOutNoPads(%0 : tensor<DPaddedShape>) {pads = [0, ..., 0]} ->
//     tensor<OutShape>
//===----------------------------------------------------------------------===//

def MaxPoolSingleOutOpPaddingPattern: Pat<(ONNXMaxPoolSingleOutOp:$res $x, $auto_pad, $ceil_mode, $dilation, $kernel_shape, $pads, $storage_order, $strides),
                                          (ONNXMaxPoolSingleOutNoPadsOp (ONNXPadConstantValuePadOp $x, (PadArrayAttrWithZeroTrail<2> $pads), (FloatAttrOfValue<0> $res), (StringAttrOfValue<"constant">)), $auto_pad, $ceil_mode, $dilation, $kernel_shape, (ArrayAttrOfZeroWithTrail<2> $pads), $storage_order, $strides),
                                          [(HasNonZeroInArrayAttr:$pads)]>;


def MaxPoolSingleOutOpNoPaddingPattern: Pat<(ONNXMaxPoolSingleOutOp:$res $x, $auto_pad, $ceil_mode, $dilation, $kernel_shape, $pads, $storage_order, $strides),
                                            (ONNXMaxPoolSingleOutNoPadsOp $x, $auto_pad, $ceil_mode, $dilation, $kernel_shape, $pads, $storage_order, $strides),
                                            [(HasAllZeroInArrayAttr:$pads)]>;

#endif // ONNX_DECOMPOSE
