//===---- ONNXRewrite.td - Pattern Match Rewriting for ONNX --*- tablegen -===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_REWRITE
#define ONNX_REWRITE

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

def GetNullAttr :
   NativeCodeCall<"Attribute()">;

// Create a StringAttr from a string.
class StringAttrOfValue<string val>:
  NativeCodeCall<"$_builder.getStringAttr(\"" # val # "\")">;

// Create a DenseElementsAttr from an interger value.
// It seems Table-gen does not support `float` type, so we can not pass a float value.
class FloatAttrOfValue<int val>:
  NativeCodeCall<"createDenseFloatAttrOfValue($_builder, $0, " # val # ")">;

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
def createArrayAttrOfZerosFrom:
  NativeCodeCall<"createArrayAttrOfZeros($_builder, $0)">;

// Pad a ArrayAttr with zeros.
//
// pads = [B1, B2, ... Bk, E1, E2, ..., Ek]
//
// becomes:
//
// pads = [0,... 0, B1, B2, ... Bk, 0,... 0, E1, E2, ..., Ek]
//         |_____|                  |_____|
//                 nZeros                    nZeros
//
// This function is used for padding attribute in Conv.
class insertZerosForNonPaddedDims<int extensionLength>:
  NativeCodeCall<"insertZerosForNonPaddedDims($_builder, $0,"
                                              # extensionLength # ")">;

// Check whether an ArrayAttr contains non-zero values or not.
def HasNonZeroInArrayAttr: Constraint<CPred<"hasNonZeroInArrayAttr($_self)">,
                                       "has non-zero elements">;

// Check that a StrAttr does not contain a specific value.
class IsNotStringAttrOfValue<string val>:
  Constraint<CPred<"$0.cast<StringAttr>().getValue() != \"" # val # "\"">>;

//===----------------------------------------------------------------------===//
// Rewrite:
// %0 = onnx.ConvOp(%D : tensor<DShape>, %K)
//     {pads = [b0, b1, ... bK, e0, e1, ..., eK]} ->
//         tensor<OutShape>
//
// as:
// %0 = onnx.PadConstantValuePadOp(%D)
//     {pads = [0, 0, b0, b1, ... bK, 0, 0, e0, e1, ..., eK]} ->
//     tensor<DPaddedShape>
// %1 = onnx.Conv(%0 : tensor<DPaddedShape>, %K) {pads = [0, ..., 0]} ->
//     tensor<OutShape>
//===----------------------------------------------------------------------===//

def ConvOpPaddingPattern: Pat<
  (ONNXConvOp:$res
     $x,
     $w, $b, $auto_pad, $dilation, $group, $kernel_shape,
     $pads,
     $strides),
  (ONNXConvOp

    (ONNXPadOp $x,
       (ONNXConstantOp (GetNullAttr), 
          (insertZerosForNonPaddedDims<2> $pads)),
       (ONNXConstantOp (GetNullAttr), 
          (FloatAttrOfValue<0> $res)),
       (StringAttrOfValue<"constant">)),

        
     $w, $b, $auto_pad, $dilation, $group, $kernel_shape,
     (createArrayAttrOfZerosFrom $pads),
     $strides),
  [(HasNonZeroInArrayAttr:$pads), (IsNotStringAttrOfValue<"VALID"> $auto_pad)]
>;

//===----------------------------------------------------------------------===//
// ONNXScalerOp %X, %Offest, %Scale
// x input, a offset, b scale
//===----------------------------------------------------------------------===//
// Useful test definitions.
def AttributeIsNull :
    Constraint<CPred<"! ($_self)">,
  "Attribute is null">;

def HasFloatType : Constraint<CPred<"(($_self).getType().dyn_cast<ShapedType>().getElementType().isF32())">>;
  
// Create a DenseElementsAttr from an ArrayAttr.
def createDenseArrayAttr:
  NativeCodeCall<"createDenseArrayAttr($_builder, $0)">;

def ScalerT : NativeCodeCall<"$_builder.getI64IntegerAttr(0)">;

// No attribute
def ScalerNullPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (replaceWithValue $x),
    [(HasFloatType:$x),(AttributeIsNull:$a), (AttributeIsNull:$b)]>;

// No attribute, input x not float type
def ScalerNullPattern2 : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXCastOp $x, (ScalerT)),
    [(AttributeIsNull:$a), (AttributeIsNull:$b)]>;

// No scale
def ScalerNoScalePattern : Pat<
  (ONNXScalerOp $x, $a, $b),
    (ONNXSubOp $x,
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))),
    [(AttributeIsNull:$b)]>;

// No offset
def ScalerNoOffsetPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp $x,
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b))),
    [(AttributeIsNull:$a)]>;

// Normal ONNXScalerOp
def ScalerPattern : Pat<
  (ONNXScalerOp $x, $a, $b),
  (ONNXMulOp 
    (ONNXSubOp $x,
      (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $a))),
    (ONNXConstantOp (GetNullAttr), (createDenseArrayAttr $b)))>;


#endif // ONNX_REWRITE
