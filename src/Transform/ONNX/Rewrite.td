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

// Create a DenseElementsAttr from a float attribute and an element type.
def createDenseElementsAttrFromFloatAttr : NativeCodeCall<
  "createDenseElementsAttrFromFloatAttr($_builder, $0.getType().cast<ShapedType>().getElementType(), $1)">;

// Create a DenseElementsAttr from the shape of the type of a value.
def createDenseElementsAttrFromShape : NativeCodeCall<
  "createDenseElementsAttrFromShape($_builder, $0)">;

// Create a DenseElementsAttr from the size of the type of a value.
def createDenseElementsAttrFromSize : NativeCodeCall<
  "createDenseElementsAttrFromSize($_builder, $0)">;

// If '$1' is not NoneType, do subtraction '$1 - $2'.
// Otherwise, take the negative of '$2'.
def subtractOrNeg: NativeCodeCall<
  "subtractOrNeg($_builder, $0.getDefiningOp()->getLoc(), $1, $2)">;

// Create an ArrayAttr of IntergerAttr(s) of values in [1, N].
def createArrayAttrOfOneToRankOf : NativeCodeCall<
  "createArrayAttrOfOneToN($_builder, $0.getType().cast<ShapedType>().getRank() - 1)">;

// Create an ArrayAttr of IntergerAttr(s) of values in [2, rank - 1].
def createArrayAttrOfTwoToRankOf : NativeCodeCall<
  "createArrayAttrOfNToM($_builder, 2, $0.getType().cast<ShapedType>().getRank() - 1)">;

def GetNullAttr :
   NativeCodeCall<"Attribute()">;
// TODO: Remove GetNullPtr once corresponding attributes are supported
def GetNullPtr :
  NativeCodeCall<"nullptr">;

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
          (insertZerosForNonPaddedDims<2> $pads),
          (GetNullPtr), (GetNullPtr), (GetNullPtr),
          (GetNullPtr), (GetNullPtr), (GetNullPtr)),
       (ONNXConstantOp (GetNullAttr), 
          (FloatAttrOfValue<0> $res),
          (GetNullPtr), (GetNullPtr), (GetNullPtr),
          (GetNullPtr), (GetNullPtr), (GetNullPtr)),
       (StringAttrOfValue<"constant">)),

        
     $w, $b, $auto_pad, $dilation, $group, $kernel_shape,
     (createArrayAttrOfZerosFrom $pads),
     $strides),
  [(HasNonZeroInArrayAttr:$pads), (IsNotStringAttrOfValue<"VALID"> $auto_pad)]
>;

//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'BatchNorm o Conv' into 'Conv'
// by deriving new 'w' and 'b' for 'Conv':
//
// We have:
//   (Conv)      z = w * x + b 
//   (BatchNorm) y = scale * (z - mean) / sqrt(var + eps) + bias
//
// which corresponds to the following computation:
//   y = w_ * x + b_
// where
//   w_ = scale * w / sqrt(var + eps)
//   b_ = B + scale * (b - mean) / sqrt(var + eps)
//
// Hence, we rewrite: 
//   onnx.BatchNormalizationTestMode(
//       onnx.Conv(x, w, b),
//       scale, B, mean, var
//   ) {eps = ...}
//
// as:
//    onnx.Conv(x, w_, b_)
//    
//    where
//      w_ = scale * w / sqrt(var + eps)
//      b_ = B + scale * (b - mean) / sqrt(var + eps)
//
//===----------------------------------------------------------------------===//

def FuseBatchNormTestModeConvPattern: Pat<
  (ONNXBatchNormalizationTestModeOp:$res
    (ONNXConvOp $x, $w, $b,
                $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
    $scale, $B, $mean, $var, $epsilon, $momentum),
  (ONNXConvOp
     $x,
     // w_
     (ONNXMulOp
        $w,
        (ONNXUnsqueezeOp
           (ONNXDivOp:$coefficientW
              $scale,
              (ONNXSqrtOp
                 (ONNXAddOp
                    $var,
                    (ONNXConstantOp
                       (GetNullAttr),
                       (createDenseElementsAttrFromFloatAttr $res, $epsilon),
                       (GetNullPtr), (GetNullPtr), (GetNullPtr),
                       (GetNullPtr), (GetNullPtr), (GetNullPtr))))),
           (createArrayAttrOfOneToRankOf $w))),
     // b_
     (ONNXAddOp
        $B,
        (ONNXMulOp
           $coefficientW,
           (subtractOrNeg $res, $b, $mean))),

     $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides)
>;

def IsStaticShapeTensor:
   Constraint<
     CPred<
       "$_self.getType().cast<::mlir::ShapedType>().hasStaticShape()">,
     "hasStaticShape">;

def ShapeToConstantPattern: Pat<
     (ONNXShapeOp $A),
     (ONNXConstantOp
        (GetNullAttr),
        (createDenseElementsAttrFromShape $A),
        (GetNullPtr), (GetNullPtr), (GetNullPtr),
        (GetNullPtr), (GetNullPtr), (GetNullPtr)),
     [(IsStaticShapeTensor:$A)]
>;

def SizeToConstantPattern: Pat<
     (ONNXSizeOp $A),
     (ONNXConstantOp
        (GetNullAttr),
        (createDenseElementsAttrFromSize $A),
        (GetNullPtr), (GetNullPtr), (GetNullPtr),
        (GetNullPtr), (GetNullPtr), (GetNullPtr)),
     [(IsStaticShapeTensor:$A)]
>;

// Rewrite GlobalAveragePool using ReduceMean.
def GlobalAveragePoolPattern: Pat<
  (ONNXGlobalAveragePoolOp $x),
  (ONNXReduceMeanOp $x, (createArrayAttrOfTwoToRankOf $x), (GetNullAttr))
>;

// Rewrite GlobalMaxPool using ReduceMax.
def GlobalMaxPoolPattern: Pat<
  (ONNXGlobalMaxPoolOp $x),
  (ONNXReduceMaxOp $x, (createArrayAttrOfTwoToRankOf $x), (GetNullAttr))
>;

#endif // ONNX_REWRITE
