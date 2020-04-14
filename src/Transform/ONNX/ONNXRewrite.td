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
   NativeCodeCall<"(Attribute) NULL">;

// Create a StringAttr from a string.
class StringAttrOfValue<string val>:
  NativeCodeCall<"$_builder.getStringAttr(\"" # val # "\")">;

// Create a DenseElementsAttr from an interger value.
// It seems Table-gen does not support `float` type, so we can not pass a float value.
class FloatAttrOfValue<int val>:
  NativeCodeCall<"createDenseFloatAttrOfValue($_builder, $0, " # val # ")">;

// Create a FloatAttr for the negative infinity.
def FloatAttrOfNegativeInfinity:
  NativeCodeCall<"createDenseFloatAttrOfValue($_builder, $0, "
                                "-std::numeric_limits<double>::infinity())">;


// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in MaxPoolSingleOut.
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
// This function is used for padding attribute in MaxPoolSingleOut.
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
// %0 = onnx.MaxPoolSingleOutOp(%D : tensor<DShape>)
//     {pads = [b0, b1, ... bK, e0, e1, ..., eK]} ->
//         tensor<OutShape>
//
// as:
// %0 = onnx.PadConstantValuePadOp(%D)
//     {pads = [0, 0, b0, b1, ... bK, 0, 0, e0, e1, ..., eK]} ->
//     tensor<DPaddedShape>
// %1 = onnx.MaxPoolSingleOut(%0 : tensor<DPaddedShape>) {pads = [0, ..., 0]} ->
//     tensor<OutShape>
//===----------------------------------------------------------------------===//

def MaxPoolSingleOutOpPaddingPattern: Pat<
  (ONNXMaxPoolSingleOutOp:$res
     $x,
     $auto_pad, $ceil_mode, $dilation, $kernel_shape,
     $pads,
     $storage_order, $strides),
  (ONNXMaxPoolSingleOutOp
       (ONNXPadOp $x,
          (ONNXConstantOp (GetNullAttr), 
             (insertZerosForNonPaddedDims<2> $pads)),
          (ONNXConstantOp (GetNullAttr), 
             (FloatAttrOfNegativeInfinity $res)),
          (StringAttrOfValue<"constant">)),
     $auto_pad, $ceil_mode, $dilation, $kernel_shape,
     (createArrayAttrOfZerosFrom $pads),
     $storage_order, $strides),
  [(HasNonZeroInArrayAttr:$pads), (IsNotStringAttrOfValue<"VALID"> $auto_pad)]
>;

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

#endif // ONNX_REWRITE
