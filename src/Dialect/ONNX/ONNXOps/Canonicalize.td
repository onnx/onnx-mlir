// SPDX-License-Identifier: Apache-2.0

//===---- ONNXRewrite.td - Pattern Match Rewriting for ONNX --*- tablegen -===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
// When adding a canonicalizer for a new operation, please add that operation to
// the OpsWithCanonicalizer list in utils/gen_onnx_mlir.py
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_REWRITE
#define ONNX_REWRITE

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

//===----------------------------------------------------------------------===//
// Common utility functions.
//===----------------------------------------------------------------------===//

// Create a DenseElementsAttr from a float attribute and an element type.
def createDenseElementsAttrFromFloatAttr : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromFloatAttr($_builder, mlir::cast<ShapedType>($0.getType()).getElementType(), $1)">;

// Create a DenseElementsAttr from the shape of the type of a value.
def createDenseElementsAttrFromShape : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromShape($_builder, $0)">;

// Create a DenseElementsAttr from the shape of the type of a value at the given index.
def createDenseElementsAttrFromShapeAtIndex: NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromShapeAtIndex($_builder, $0, $1)">;

def createDenseElementsAttrFromShapeResult : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromShapeOp($_builder, $0.getDefiningOp())">;

// Create a DenseElementsAttr from the size of the type of a value.
def createDenseElementsAttrFromSize : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromSize($_builder, $0)">;

// If '$1' is not NoneType, do subtraction '$1 - $2'.
// Otherwise, take the negative of '$2'.
def subtractOrNeg: NativeCodeCall<
  "onnx_mlir::subtractOrNeg($_builder, $0.getDefiningOp()->getLoc(), $1, $2)">;

// Get the rank of the given value.
def getRankOf :
	NativeCodeCall<"mlir::cast<ShapedType>($0.getType()).getRank()">;

// Create an ArrayAttr of IntergerAttr(s) of [$0].
def createDenseElementsAttrOf : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrOfNToM($_builder, $0, $0)">;

// Create an ArrayAttr of IntergerAttr(s) of values in [1, N-1].
def createDenseElementsAttrOfOneToRankOf : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrOfNToM($_builder, 1, mlir::cast<ShapedType>($0.getType()).getRank() - 1)">;

// Create an ArrayAttr of IntergerAttr(s) of values in [1, N-2].
def createDenseElementsAttrOfOneToRankOfExclusive : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrOfNToM($_builder, 1, mlir::cast<ShapedType>($0.getType()).getRank() - 2)">;

// Create an ArrayAttr of IntergerAttr(s) of values in [2, rank - 1].
def createArrayAttrOfTwoToRankOf : NativeCodeCall<
  "onnx_mlir::createArrayAttrOfNToM($_builder, 2, mlir::cast<ShapedType>($0.getType()).getRank() - 1)">;

def AttributeIsNotNull :
  Constraint<CPred<"($_self)">, "Attribute is not null">;

// Intended to check whether there is at least one not-Null the attributes
// However, the current table gen can only support max 4 parameters
// Multiple rules are used instead of one rule
def AttributesNotAllNull :
  Constraint<Neg<And<[CPred<"($0)">, CPred<" ($1) ">, CPred<" ($2) ">]>>,
  "Attributes are not null">;

def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetNullFloatAttr : NativeCodeCall<"FloatAttr()">;

def GetNullIntegerAttr : NativeCodeCall<"IntegerAttr()">;

def GetNullStringAttr : NativeCodeCall<"StringAttr()">;

def GetNullArrayAttr :  NativeCodeCall<"ArrayAttr()">;

// Check whether an ArrayAttr contains non-zero values or not.
def HasNonZeroInArrayAttr: Constraint<CPred<"hasNonZeroInArrayAttr($_self)">,
                                       "has non-zero elements">;

// Check the rank of a value is greater than a given integer.
class HasRankGT<int rank> :
  Constraint<CPred<"mlir::isa<ShapedType>($0.getType()) && "
                   "mlir::cast<ShapedType>($0.getType()).hasRank() && "
                   "mlir::cast<ShapedType>($0.getType()).getRank() > " # rank>>;

// Check the rank of a value is of a given integer.
class HasRankOf<int rank> :
  Constraint<CPred<"mlir::isa<ShapedType>($0.getType()) && "
                   "mlir::cast<ShapedType>($0.getType()).hasRank() && "
                   "mlir::cast<ShapedType>($0.getType()).getRank() == " # rank>>;

def HaveSameLastDim: Constraint<
  CPred<"onnx_mlir::hasShapeAndRank($0) && onnx_mlir::hasShapeAndRank($1) && "
        "(mlir::cast<RankedTensorType>($0.getType()).getShape()"
        "[mlir::cast<RankedTensorType>($0.getType()).getRank() - 1] == "
        "mlir::cast<RankedTensorType>($1.getType()).getShape()"
        "[mlir::cast<RankedTensorType>($1.getType()).getRank() - 1])">,
  "Two tensors have the same last dimension">;

class HaveSameDim<int dim>: Constraint<
  CPred<"onnx_mlir::hasShapeAndRank($0) && onnx_mlir::hasShapeAndRank($1) && "
        "!mlir::cast<RankedTensorType>($0.getType()).isDynamicDim(" # dim # ") && "
        "(mlir::cast<RankedTensorType>($0.getType()).getShape()[" # dim # "] =="
        "mlir::cast<RankedTensorType>($1.getType()).getShape()[" # dim # "])">,
  "Two tensors have the same specified dimension">;

def HaveSameShapedType: Constraint<
    CPred<"(isa<ShapedType>($0.getType()) &&"
          "dyn_cast<ShapedType>($0.getType()) == "
          "dyn_cast<ShapedType>($1.getType()))">,
    "has same shaped type">;

def HaveSameStaticShape: Constraint<
  CPred<"onnx_mlir::haveSameStaticShape($0, $1)">,
  "Two tensors have the same static shape">;

def IsIdentityReshape: Constraint<
  CPred<"onnx_mlir::isIdentityReshape($0, $1)">,
  "Reshape is identity operation">;

// Create a unit constant that will be used as none input.
def CreateNoneValue : NativeCodeCall<"$_builder.create<ONNXNoneOp>($_loc).getResult()">;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def HasNoneType : Constraint<CPred<"mlir::isa<NoneType>($0.getType())">>;

def NotNoneType : Constraint<CPred<"!mlir::isa<NoneType>(($0.getType()))">>;

def HasShapeAndRank : Constraint<CPred<"onnx_mlir::hasShapeAndRank($_self)">>;

def HasSameElementType : Constraint<
    CPred<"(mlir::dyn_cast<ShapedType>($0.getType()).getElementType() == "
          "mlir::cast<::mlir::TypeAttr>($1).getValue())">,
    "has same element type">;

def HaveSameElementType : Constraint<
    CPred<"(mlir::dyn_cast<ShapedType>($0.getType()).getElementType() == "
          "mlir::dyn_cast<ShapedType>($1.getType()).getElementType())">,
    "have same element types">;

def HaveSameElementTypeBitWidth: Constraint<
    CPred<"(mlir::dyn_cast<ShapedType>($0.getType()).getElementTypeBitWidth() == "
          "mlir::dyn_cast<ShapedType>($1.getType()).getElementTypeBitWidth())">,
    "has same element type bitwidth">;

def  ElementTypeIsNotUnsigned: Constraint<
    CPred<"!mlir::dyn_cast<ShapedType>($_self.getType()).getElementType().isUnsignedInteger()">,
    "element type is not unsigned int">;

def HaveSameEncodingAttr: Constraint<
    CPred<"sameEncodingAttr($0.getType(), $1.getType())">,
    "have same RankedTensorType's encoding attribute">;

def IsStaticShapeTensor:
  Constraint<
    CPred<
      "mlir::cast<::mlir::ShapedType>($_self.getType()).hasStaticShape()">,
    "hasStaticShape">;

def IsNoneValue: Constraint<
   CPred<"onnx_mlir::isNoneValue($_self)">,
          "Is the value none">;

def HasSpecifiedConstantShape: Constraint<
    CPred<"onnx_mlir::HasSpecifiedConstantShape($0, $1)">,
          "Has the specified constant shape">;

def IsFromONNXConstantOp: Constraint<
    CPred<"onnx_mlir::isDenseONNXConstant($0)">,
    "Is a value from ONNXConstantOp">;

def IsNotFromONNXConstantOp: Constraint<
    CPred<"!(llvm::dyn_cast_or_null<ONNXConstantOp>($0.getDefiningOp()))">,
    "Is a value not from ONNXConstantOp">;

def IsNegativeSplatConstant: Constraint<
  CPred<"onnx_mlir::isNegativeSplatConstant($_self)">,
  "Is a splat constant with a negative value."
>;

def AreAllDimSizes: Constraint<
  CPred<"onnx_mlir::areAllDimSizes($_self)">,
  "All values in the input ValueRange are dimension sizes."
>;

def AreTheSameAxesConstant: Constraint<
    CPred<"onnx_mlir::AreTheSameAxesConstant("
          "(onnx_mlir::hasShapeAndRank($0) ? mlir::cast<ShapedType>($0.getType()).getRank() : 0),"
          "$1, $2)">,
    "Two values are constants with the same axis values">;

def AreTheSameAxesArrayAttr: Constraint<
    CPred<"onnx_mlir::AreTheSameAxesArrayAttr("
          "(onnx_mlir::hasShapeAndRank($0) ? mlir::cast<ShapedType>($0.getType()).getRank() : 0),"
          "$1, $2)">,
    "Two axis arrays are the same">;

class AllDimsFromAxisToEndAre<int axis, int val>: Constraint<
    CPred<"llvm::all_of("
            "ArrayRef<int64_t>(mlir::cast<ShapedType>($_self.getType()).getShape().begin() + " # axis # ","
            "                  mlir::cast<ShapedType>($_self.getType()).getShape().end()),"
            "[](int64_t val) { return (val == " # val # ");})">,
    "All dimensions from axis to the end are val">;

def DimAtIndexIsConstant: Constraint<
  CPred<"onnx_mlir::hasShapeAndRank($0) &&"
        "!mlir::cast<ShapedType>($0.getType()).isDynamicDim($1.getValue().getSExtValue())">,
  "Dim at the given index is constant"
>;

class RankXMinusRankYIs<int diff>: Constraint<
    CPred<"(mlir::cast<ShapedType>($0.getType()).getRank() "
          " - mlir::cast<ShapedType>($1.getType()).getRank() == " # diff # ")">,
    "X' rank is greater than Y's rank diff units">;

def TransposeVariadicInput: NativeCodeCall<
  "onnx_mlir::transposeVariadicInput($_builder, $_loc, $0, $1)">;

def CastVariadicInput: NativeCodeCall<
  "onnx_mlir::castVariadicInput($_builder, $_loc, $0, $1, $2)">;

// Check whether two variables are equal.
def Equal: Constraint<CPred<"$0 == $1">, "are equal">;

class EqualString<string s> : Constraint<CPred<"$0 == \"" # s # "\"">>;

def AxisIsTheLastDim: Constraint<
  CPred<"($1.getValue().getSExtValue() == -1) ||"
        "(onnx_mlir::hasShapeAndRank($0) &&"
        " (mlir::cast<ShapedType>($0.getType()).getRank() == $1.getValue().getSExtValue() + 1))">,
  "Axis is the last dimension of the input"
>;

// Check a AffineMap is identity or not.
def IsIdentityAffineMap : Constraint<
  CPred<"$_self.isIdentity()">,
  "Is identity AffineMap"
>;

/// Compose two affine maps.
def GetComposedMap : NativeCodeCall<
 "mlir::AffineMapAttr::get($0.getValue().compose($1.getValue()))"
>;

class IntegerAttrIsOf<int s> : Constraint<
  CPred<"$0.getValue().getSExtValue() == " # s>,
  "IntegerAttr is of the given value"
  >;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def GemmAlpha : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmBeta : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmTransA : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;
def GemmTransB : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXAddOp
//===----------------------------------------------------------------------===//

// onnx.add(onnx.matmul(%X, %Y), %Z) = onnx.Gemm(%X, %Y, %Z)
def MulAddToGemmOptPattern : Pat<(ONNXAddOp (ONNXMatMulOp:$res $m1, $m2), $m3),
                                 (ONNXGemmOp $m1, $m2, $m3, (GemmAlpha), (GemmBeta), (GemmTransA), (GemmTransB)),
                                 [(HasOneUse $res), (HasRankOf<2> $m1), (HasRankOf<2> $m2)]>;

// onnx.add(onnx.Gemm(%X, %Y, None), %Z) = onnx.Gemm(%X, %Y, %Z) {beta=1.0}
def FuseGemmFollowedByAddition : Pat<(ONNXAddOp (ONNXGemmOp:$res $m1, $m2, $none, $alpha, $beta, $transA, $transB), $bias),
                                     (ONNXGemmOp $m1, $m2, $bias, $alpha, (GemmBeta), $transA, $transB),
                                     [(HasOneUse $res), (HasNoneType $none)]>;

//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'Add o Conv' into 'Conv' if the other input
// of Add is a constant, by adding the constant to 'b' of 'Conv':
//
// We have:
//   (Conv)      z = w * x + b
//   (Add)       y = z + constant
//
// which corresponds to the following computation:
//   y = w * x + new_b
// where
//   new_b = b + constant
//
// The shape of 'constant' must be compatible with that of 'b'
//===----------------------------------------------------------------------===//

def NormalizeAddPattern: Pat<
  (ONNXAddOp $x, $y),
  (ONNXAddOp $y, $x),
  [(IsFromONNXConstantOp $x), (IsNotFromONNXConstantOp $y)]
>;

def FuseAddConvNullBiasPattern: Pat<
  (ONNXAddOp:$res
    (ONNXConvOp
       $x, $w, $b, $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
    (ONNXConstantOp:$y $_, $denseAttr, $_, $_, $_, $_, $_, $_)),
  (ONNXConvOp
     $x, $w,
     // new_b
     (ONNXSqueezeOp
        $y,
        (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOfOneToRankOf $y))),
     // unchanged operands and attributes.
     $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
  [(HasShapeAndRank:$res),
   (HasNoneType $b),
   (AttributeIsNotNull:$denseAttr),
   (AllDimsFromAxisToEndAre<1, 1>:$y),
   (RankXMinusRankYIs<1> $res, $y)]
>;

def FuseAddConvPattern: Pat<
  (ONNXAddOp:$res
    (ONNXConvOp
       $x, $w, $b, $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
    (ONNXConstantOp:$y $_, $denseAttr, $_, $_, $_, $_, $_, $_)),
  (ONNXConvOp
     $x, $w,
     // new_b
     (ONNXAddOp
        $b,
        (ONNXSqueezeOp
           $y,
           (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOfOneToRankOf $y)))),
     // unchanged operands and attributes.
     $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
  [(HasShapeAndRank:$res),
   (NotNoneType $b),
   (AttributeIsNotNull:$denseAttr),
   (AllDimsFromAxisToEndAre<1, 1>:$y),
   (RankXMinusRankYIs<1> $res, $y)]
>;

//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'Mul o Conv' into 'Conv' if the other input
// of Mul is a constant, by multipling constant to 'w' of 'Conv':
//
// We have:
//   (Conv)      z = i * w + b
//   (Mul)       y = z x c      (where c is a constant)
//
// which corresponds to the following computation:
//   y = i * new_w + b
// where
//   new_w = w x c
//
// The shape of 'c' must be compatible with that of 'w'
//===----------------------------------------------------------------------===//

def NormalizeMulPattern: Pat<
  (ONNXMulOp $x, $y),
  (ONNXMulOp $y, $x),
  [(IsFromONNXConstantOp $x), (IsNotFromONNXConstantOp $y)]
>;

def FuseMulConvNullBiasPattern: Pat<
  (ONNXMulOp:$res
    (ONNXConvOp
       $x, $w, $b, $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
    (ONNXConstantOp:$y $_, $denseAttr, $_, $_, $_, $_, $_, $_)),
  (ONNXConvOp
     $x,
     // new_w
     (ONNXMulOp
        $w,
        (ONNXUnsqueezeOp $y,
           (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOf(getRankOf $y))))),
     // unchanged operands and attributes.
     $b, $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
  [(HasNoneType $b),
   (AttributeIsNotNull:$denseAttr),
   (IsFromONNXConstantOp $w),
   (HaveSameElementType $w, $y),       // multiplier and Conv weight must have the same element type.
   (HasRankGT<1> $w),                  // rank of $w must be at least 2.
   (RankXMinusRankYIs<1> $w, $y),      // rank($y) must be equal to rank($w)-1.
   (HaveSameDim<0> $w, $y),            // the first dimension of $w and $y must be equal.
   (AllDimsFromAxisToEndAre<1, 1>:$y)] // all dimensions of $y must be 1 except for the first one.
>;

// TODO add pattern for non-null bias with contraints:
// - bias must be have rank equal to 1 and
// - bias element data type must be the same as mul constant
// - bias dimension (0) must be equal to mul constant dim(0)
// codegen is different too (look it up in onnx-runtime)

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXIdentityOp
//===----------------------------------------------------------------------===//

// ONNX_Op (onnx.Identity (%X)) = ONNX_Op (%X)
def IdentityEliminationPattern : Pat<(ONNXIdentityOp $arg),
                                     (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXDropoutOp
//===----------------------------------------------------------------------===//

// We assume that training_mode is not true so output, mask become data, none.
// output, mask = onnx.Dropout(data) -> output, mask = data, none
def DropoutEliminationPattern : Pattern<(ONNXDropoutOp $data, $ratio, $training_mode, $seed),
  [(replaceWithValue $data), (CreateNoneValue)]>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXCastOp
//===----------------------------------------------------------------------===//

// ONNX_Op (onnx.Cast (%X, $type)) = ONNX_Op (%X)
def CastEliminationPattern : Pat<
	(ONNXCastOp $arg, $saturate, $type),
	(replaceWithValue $arg),
  [(HasSameElementType $arg, $type)]>;

// TODO: Reintroduce pattern for sound type combinations, see issue #2210.
//// onnx.Cast (onnx.Cast (%X, $type1)), $type2 = onnx.Cast (%X, $type2)
//def FuseCastCastPattern : Pat<
//  (ONNXCastOp (ONNXCastOp $arg, $_), $type),
//  (ONNXCastOp $arg, $type)>;

// Do cast on concat's inputs instead of output in order to propagate
// cast operations together, which brings concat close to reshape
// because concat is used for shape in reshape.
def SwapCastConcatPattern: Pat<
  (ONNXCastOp (ONNXConcatOp $inputs, $axis), $saturate, $to),
  (ONNXConcatOp (CastVariadicInput $inputs, $saturate, $to), $axis)
>;

// Do cast on slice's inputs instead of output in order to propagate
// cast operations together, which brings slice close to reshape.
def SwapCastSlicePattern: Pat<
  (ONNXCastOp (ONNXSliceOp $data, $starts, $ends, $axes, $steps), $saturate, $to),
  (ONNXSliceOp (ONNXCastOp $data, $saturate, $to), $starts, $ends, $axes, $steps)
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXTileOp
//===----------------------------------------------------------------------===//

def IsFromONNXConstantOpWithOnes: Constraint<
    And<[CPred<"onnx_mlir::isDenseONNXConstant($_self)">,
         CPred<"::llvm::all_of("
               "onnx_mlir::getElementAttributeFromONNXValue($_self)"
               ".getValues<int64_t>(), "
               "[](int64_t repeat) { return repeat == 1;})">
        ]>, "Value is not a ONNXConstantOp with an ElementsAttr of ones">;

def RemoveIdentityTilePattern:  Pat<
  // Tile with `repeats` of all constant 1's
  (ONNXTileOp:$result $val, $r),
  // Remove the tile.
  (replaceWithValue $val),
  // Check that we have indeed a identity tile pattern.
  [(IsFromONNXConstantOpWithOnes:$r), (HaveSameShapedType $val,$result)]>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXLayoutTransformOp
//===----------------------------------------------------------------------===//

// ONNX_Op (onnx.ONNXLayoutTransformOp (%X)) = ONNX_Op (%X) when input and
// output layouts are identical
def ONNXLayoutTransformEliminationPattern : Pat<
  (ONNXLayoutTransformOp:$res $arg, $target_layout),
  (replaceWithValue $arg),
  [(HaveSameEncodingAttr $res, $arg),
   (HaveSameElementType $res, $arg)]>;

def ONNXLayoutTransformFusionPattern : Pat<
  (ONNXLayoutTransformOp (ONNXLayoutTransformOp $arg, $layout1), $layout2),
  (ONNXLayoutTransformOp $arg, $layout2)
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXTransposeOp
//===----------------------------------------------------------------------===//

// Combine transposes.
def CreateCombinedTransposedPattern :
   NativeCodeCall<"onnx_mlir::CombinedTransposePattern($_builder, $0, $1)">;

def IsIdentityPermuteAttribute :
  Constraint<CPred<"onnx_mlir::IsIdentityPermuteVector($_self)">,
    "has identity permute vector">;

def FuseTransposePattern:  Pat<
  // Transpose of a transpose.
  (ONNXTransposeOp (ONNXTransposeOp $v, $p1), $p2),
  // Transpose with combined pattern.
  (ONNXTransposeOp $v, (CreateCombinedTransposedPattern $p1, $p2))>;

class FuseTransposeAndElementwiseUnaryOpPattern<Op elmOp>:  Pat<
  // Transpose of elmOp on a transpose.
  (ONNXTransposeOp (elmOp (ONNXTransposeOp $v, $p1)), $p2),
  // elmOp on Transpose with combined pattern.
  (ONNXTransposeOp (elmOp $v, (returnType $v)),
                   (CreateCombinedTransposedPattern $p1, $p2))>;

class FuseTransposeAndElementwiseUnaryWithOneArgOpPattern<Op elmOp>:  Pat<
  // Transpose of elmOp on a transpose.
  (ONNXTransposeOp (elmOp (ONNXTransposeOp $v, $p1), $arg1), $p2),
  // elmOp on Transpose with combined pattern.
  (ONNXTransposeOp (elmOp $v, $arg1, (returnType $v)),
                   (CreateCombinedTransposedPattern $p1, $p2))>;

class FuseTransposeAndElementwiseUnaryWithTwoArgOpPattern<Op elmOp>:  Pat<
  // Transpose of elmOp on a transpose.
  (ONNXTransposeOp (elmOp (ONNXTransposeOp $v, $p1), $arg1, $arg2), $p2),
  // elmOp on Transpose with combined pattern.
  (ONNXTransposeOp (elmOp $v, $arg1, $arg2, (returnType $v)),
                   (CreateCombinedTransposedPattern $p1, $p2))>;

def FuseTransposeAndAtanPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAtanOp>;
def FuseTransposeAndCastPattern :
    FuseTransposeAndElementwiseUnaryWithTwoArgOpPattern<ONNXCastOp>;
def FuseTransposeAndCeilPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXCeilOp>;
def FuseTransposeAndCosPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXCosOp>;
def FuseTransposeAndCoshPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXCoshOp>;
def FuseTransposeAndEluPattern :
    FuseTransposeAndElementwiseUnaryWithOneArgOpPattern<ONNXEluOp>;
def FuseTransposeAndErfPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXErfOp>;
def FuseTransposeAndAcosPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAcosOp>;
def FuseTransposeAndAcoshPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAcoshOp>;
def FuseTransposeAndAsinPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAsinOp>;
def FuseTransposeAndAsinhPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAsinhOp>;
def FuseTransposeAndAtanhPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXAtanhOp>;
def FuseTransposeAndExpPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXExpOp>;
def FuseTransposeAndFloorPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXFloorOp>;
def FuseTransposeAndHardSigmoidPattern :
    FuseTransposeAndElementwiseUnaryWithTwoArgOpPattern<ONNXHardSigmoidOp>;
def FuseTransposeAndIsNaNPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXIsNaNOp>;
def FuseTransposeAndLeakyReluPattern :
    FuseTransposeAndElementwiseUnaryWithOneArgOpPattern<ONNXLeakyReluOp>;
def FuseTransposeAndLogPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXLogOp>;
def FuseTransposeAndNegPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXNegOp>;
def FuseTransposeAndNotPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXNotOp>;
def FuseTransposeAndReciprocalPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXReciprocalOp>;
def FuseTransposeAndReluPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXReluOp>;
def FuseTransposeAndRoundPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXRoundOp>;
def FuseTransposeAndSeluPattern :
    FuseTransposeAndElementwiseUnaryWithTwoArgOpPattern<ONNXSeluOp>;
def FuseTransposeAndSigmoidPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSigmoidOp>;
def FuseTransposeAndSignPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSignOp>;
def FuseTransposeAndSinPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSinOp>;
def FuseTransposeAndSinhPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSinhOp>;
def FuseTransposeAndSoftplusPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSoftplusOp>;
def FuseTransposeAndSoftsignPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSoftsignOp>;
def FuseTransposeAndSqrtPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXSqrtOp>;
def FuseTransposeAndTanPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXTanOp>;
def FuseTransposeAndTanhPattern :
    FuseTransposeAndElementwiseUnaryOpPattern<ONNXTanhOp>;


def RemoveIdentityTransposePattern:  Pat<
  // Transpose with an identity pattern (e.g. {0, 1, 2, 3}).
  (ONNXTransposeOp $val, $p),
  // Remove the transpose.
  (replaceWithValue $val),
  // Check that we have indeed a identity transpose pattern.
  [(IsIdentityPermuteAttribute:$p)]>;

def GetIndexOfAxisInPerm: NativeCodeCall<
  "onnx_mlir::getIndexOfAxisInPerm($_builder, $0, $1)"
>;

def ProducedByTransposeOp: Constraint<
  CPred<"onnx_mlir::areProducedByTransposeOp($_self)">,
  "all values are produced by ONNXTransposeOp"
>;

// Do transpose on concat's inputs instead of output in order to propagate
// transpose operations together, which allows more chance for transpose fusion.
// Do this only when all inputs are produced by transpose operations.
def SwapTransposeConcatPattern: Pat<
  (ONNXTransposeOp:$res (ONNXConcatOp $inputs, $axis), $perm),
  (ONNXConcatOp (TransposeVariadicInput $inputs, $perm),
                (GetIndexOfAxisInPerm $perm, $axis)),
  [(ProducedByTransposeOp:$inputs)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXReshapeOp
//===----------------------------------------------------------------------===//

def FuseTwoReshapesAllowZeroPattern:  Pat<
  // Reshape of a reshape.
  (ONNXReshapeOp (ONNXReshapeOp $v, $s1, $az1), $s2, $az2),
  // Remove the first reshape op.
  (ONNXReshapeOp $v, $s2, $az2),
  [(IntegerAttrIsOf<1> $az2)]>;

def RemoveIdentityReshapePattern1:  Pat<
  // Remove an identity pattern. Input tensor already has the specified shape.
  (ONNXReshapeOp $val, $shape, $az),
  // Remove the reshape.
  (replaceWithValue $val),
  // Check if shape has no value
  [(IsNoneValue:$shape)]>;

def RemoveIdentityReshapePattern2:  Pat<
  // Remove an identity pattern. Output and input shapes are static and the same.
  (ONNXReshapeOp:$out $val, $_, $_),
  // Remove the reshape.
  (replaceWithValue $val),
  // Check that val and out have the same static shape.
  [(IsIdentityReshape $out, $val)]>;

def GetReturnTypeForMatMulOpND2D: NativeCodeCall<
  "onnx_mlir::getReturnTypeForMatMulOpND2D($0, $1)"
>;

def SwapReshapeMatMulPattern: Pattern<
 // If the input of Reshape is suitable for the next MatMul, use the input directly
 // for MatMul. In other words, swapping Reshape and MatMul.
 // This rule will put reshape ops together, then the reshape fusion rule can be applied.
 // TODO: Support dynamic dimensions.
 (ONNXMatMulOp:$res2 (ONNXReshapeOp:$res1 $A, $_, $az), $B),
 [(ONNXReshapeOp (ONNXMatMulOp $A, $B, (returnType (GetReturnTypeForMatMulOpND2D $A, $B))),
                 (ONNXConstantOpFromDenseAttr
                    (createDenseElementsAttrFromShape $res2)
                 ), $az)],
 [(HasRankGT<2> $A), (HasRankOf<2> $res1), (HasRankOf<2> $B), // A is reshaped to 2D.
  (HaveSameLastDim $A, $res1), // The last dim of A is unchanged by reshape.
  (IsStaticShapeTensor:$res2)  // $res2 has static dims in order to create ReshapeOp.
 ]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXSqueezeOp
//===----------------------------------------------------------------------===//

/// Combine squeeze and unsqueeze.
/// Squeeze {axes = [a, b, c]} (Unsqueeze {axes = [a, b, c]} (%X)) = %X
def RemoveSqueezeUnsqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXSqueezeOp (ONNXUnsqueezeOp $val, $u_axes), $s_axes),
  // Remove both squeeze and unsqueeze.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesConstant $val, $u_axes, $s_axes)]>;

/// Combine squeeze and unsqueeze.
/// Squeeze {axes = [a, b, c]} (Unsqueeze {axes = [a, b, c]} (%X)) = %X
def RemoveSqueezeV11UnsqueezeV11Pattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXSqueezeV11Op (ONNXUnsqueezeV11Op $val, $u_axes), $s_axes),
  // Remove both squeeze and unsqueeze.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesArrayAttr $val, $u_axes, $s_axes)]>;

/// Combine squeeze, cast and unsqueeze.
/// Squeeze {axes = [a, b, c]} (Cast (Unsqueeze {axes = [a, b, c]} (%X))) = Cast %X
def RemoveSqueezeCastUnsqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXSqueezeOp (ONNXCastOp (ONNXUnsqueezeOp $val, $u_axes), $saturate, $castTo), $s_axes),
  // Remove both squeeze and unsqueeze.
  (ONNXCastOp $val, $saturate, $castTo),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesConstant $val, $u_axes, $s_axes)]>;

/// Combine squeeze, cast and unsqueeze.
/// Squeeze {axes = [a, b, c]} (Cast (Unsqueeze {axes = [a, b, c]} (%X))) = Cast %X
def RemoveSqueezeV11CastUnsqueezeV11Pattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXSqueezeV11Op (ONNXCastOp (ONNXUnsqueezeV11Op $val, $u_axes), $saturate, $castTo), $s_axes),
  // Remove both squeeze and unsqueeze.
  (ONNXCastOp $val, $saturate, $castTo),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesArrayAttr $val, $u_axes, $s_axes)]>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXUnsqueezeOp
//===----------------------------------------------------------------------===//

/// Combine unsqueeze and squeeze.
/// Unsqueeze {axes = [a, b, c]} (Squeeze {axes = [a, b, c]} (%X)) = %X
def RemoveUnsqueezeSqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXUnsqueezeOp (ONNXSqueezeOp:$res $val, $s_axes), $u_axes),
  // Remove both squeeze and unsqueeze.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesConstant $res, $u_axes, $s_axes)]>;

/// Combine unsqueeze and squeeze.
/// Unsqueeze {axes = [a, b, c]} (Squeeze {axes = [a, b, c]} (%X)) = %X
def RemoveUnsqueezeV11SqueezeV11Pattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXUnsqueezeV11Op (ONNXSqueezeV11Op:$res $val, $s_axes), $u_axes),
  // Remove both squeeze and unsqueeze.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesArrayAttr $res, $u_axes, $s_axes)]>;

/// Combine unsqueeze, cast, and squeeze.
/// Unsqueeze {axes = [a, b, c]} (Cast (Squeeze {axes = [a, b, c]} (%X))) = Cast %X
def RemoveUnsqueezeCastSqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXUnsqueezeOp (ONNXCastOp (ONNXSqueezeOp:$res $val, $s_axes), $saturate, $castTo), $u_axes),
  // Remove both squeeze and unsqueeze.
  (ONNXCastOp $val, $saturate, $castTo),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesConstant $res, $u_axes, $s_axes)]>;

/// Combine unsqueeze, cast, and squeeze.
/// Unsqueeze {axes = [a, b, c]} (Cast (Squeeze {axes = [a, b, c]} (%X))) = Cast %X
def RemoveUnsqueezeV11CastSqueezeV11Pattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXUnsqueezeV11Op (ONNXCastOp (ONNXSqueezeV11Op:$res $val, $s_axes), $saturate, $castTo), $u_axes),
  // Remove both squeeze and unsqueeze.
  (ONNXCastOp $val, $saturate, $castTo),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxesArrayAttr $res, $u_axes, $s_axes)]>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXBatchNormOp
//===----------------------------------------------------------------------===//

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
//   onnx.BatchNormalizationInferenceMode(
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

def FuseBatchNormInferenceModeConvPattern: Pat<
  (ONNXBatchNormalizationInferenceModeOp:$res
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
                    (ONNXConstantOpFromDenseAttr
                       (createDenseElementsAttrFromFloatAttr $res, $epsilon))))),
           (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOfOneToRankOf $w)))),
     // b_
     (ONNXAddOp
        $B,
        (ONNXMulOp
           $coefficientW,
           (subtractOrNeg $res, $b, $mean))),

     $auto_pad, $dilation, $group, $kernel_shape, $pads, $strides),
     [], [], (addBenefit 1)
>;

//===----------------------------------------------------------------------===//
// This is to rewrite BatchNorm into 'x * a + b'
//
// We have:
//   (BatchNorm) y = scale * (x - mean) / sqrt(var + eps) + bias
//                 = x * a + b
//
// where
//   a = scale / sqrt(var + eps)
//   b = bias - mean * a
//
// In inference mode, 'scale', 'mean', 'var', 'eps', and 'bias' are expected to
// be constants. Thus, 'a' and 'b' would be constants after constant propagation.
//
//===----------------------------------------------------------------------===//

def RewriteBatchNormInferenceModeConvPattern1: Pat<
  (ONNXBatchNormalizationInferenceModeOp:$res
    $x, $scale, $bias, $mean, $var, $epsilon, $_),
  (ONNXAddOp
     // x * a
     (ONNXMulOp
        $x,
        (ONNXUnsqueezeOp
           (ONNXDivOp:$a
              $scale,
              (ONNXSqrtOp
                 (ONNXAddOp
                    $var,
                    (ONNXConstantOpFromDenseAttr
                       (createDenseElementsAttrFromFloatAttr $res, $epsilon))))),
           (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOfOneToRankOfExclusive $x)))),
     // b
     (ONNXUnsqueezeOp
       (ONNXSubOp $bias, (ONNXMulOp $mean, $a)),
       (ONNXConstantOpFromDenseAttr (createDenseElementsAttrOfOneToRankOfExclusive $x)))),
  [(HasRankGT<2> $x)], [], (addBenefit 0)
>;

// Special case of BatchNorm whose input shape is [N]. In this case, 'scale',
// 'bias', 'mean', and 'var' will have shape of [1], according to ONNXBatchNorm
// decription: https://github.com/onnx/onnx/blob/main/docs/Operators.md#inputs-12.
// Thus, we need not unsqueeze intermediate results.
def RewriteBatchNormInferenceModeConvPattern2: Pat<
  (ONNXBatchNormalizationInferenceModeOp:$res
    $x, $scale, $bias, $mean, $var, $epsilon, $_),
  (ONNXAddOp
     // x * a
     (ONNXMulOp
        $x,
        (ONNXDivOp:$a
           $scale,
           (ONNXSqrtOp
              (ONNXAddOp
                 $var,
                 (ONNXConstantOpFromDenseAttr
                    (createDenseElementsAttrFromFloatAttr $res, $epsilon)))))),
     // b
     (ONNXSubOp $bias, (ONNXMulOp $mean, $a))),
  [(HasRankOf<1> $x)], [], (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXShapeOp
//===----------------------------------------------------------------------===//

// TODO: this rules may need to be updated once ShapeOp has start/end.
def ShapeToConstantPattern: Pat<
     (ONNXShapeOp:$res $A, $end, $start),
     (ONNXConstantOp
        (GetNullAttr), (createDenseElementsAttrFromShapeResult $res),
        (GetNullFloatAttr), (GetNullArrayAttr), (GetNullIntegerAttr),
        (GetNullArrayAttr), (GetNullStringAttr), (GetNullArrayAttr)
     ),
     [(IsStaticShapeTensor:$A)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXSizeOp
//===----------------------------------------------------------------------===//

def SizeToConstantPattern: Pat<
     (ONNXSizeOp $A),
     (ONNXConstantOp
        (GetNullAttr), (createDenseElementsAttrFromSize $A),
        (GetNullFloatAttr), (GetNullArrayAttr), (GetNullIntegerAttr),
        (GetNullArrayAttr), (GetNullStringAttr), (GetNullArrayAttr)
     ),
     [(IsStaticShapeTensor:$A)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXGlobalAveragePoolOp
//===----------------------------------------------------------------------===//

// Rewrite GlobalAveragePool using ReduceMean.
def GlobalAveragePoolPattern: Pat<
  (ONNXGlobalAveragePoolOp $x),
  (ONNXReduceMeanV13Op $x, (createArrayAttrOfTwoToRankOf $x), (GetNullAttr))
>;

// Rewrite GlobalMaxPool using ReduceMax.
def GlobalMaxPoolPattern: Pat<
  (ONNXGlobalMaxPoolOp $x),
  (ONNXReduceMaxV13Op $x, (createArrayAttrOfTwoToRankOf $x), (GetNullAttr))
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXDepthToSpaceOp
//===----------------------------------------------------------------------===//

// Combine DepthToSpace and SpaceToDepth.
// DepthToSpace (SpaceToDepth (%X)) = %X
def RemoveDepthToSpaceSpaceToDepthPattern : Pat<
        (ONNXDepthToSpaceOp (ONNXSpaceToDepthOp $v, $bs1), $bs2, $mode),
        (replaceWithValue $v),
        [(Equal $bs1, $bs2), (EqualString<"CRD"> $mode)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXSpaceToDepthOp
//===----------------------------------------------------------------------===//

// Combine SpaceToDepth and DepthToSpace.
// DepthToSpace (SpaceToDepth (%X)) = %X
def RemoveSpaceToDepthDepthToSpacePattern : Pat<
        (ONNXSpaceToDepthOp (ONNXDepthToSpaceOp $v, $bs1, $mode), $bs2),
        (replaceWithValue $v),
        [(Equal $bs1, $bs2), (EqualString<"CRD"> $mode)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXLessOp
//===----------------------------------------------------------------------===//

// The 'saturate' attribute shouldn't matter but we require it to be the same
// for $a and $b to avoid applying the pattern in surprising ways.
def LessOpSameCastPattern: Pat<
   (ONNXLessOp (ONNXCastOp:$cast_a $a, $saturate1, $to1), (ONNXCastOp $b, $saturate2, $to2)),
   (ONNXLessOp $a, $b),
   [(HaveSameElementType $a, $b), (Equal $saturate1, $saturate2), (Equal $to1, $to2),
    (HaveSameElementTypeBitWidth $a, $cast_a),
    (ElementTypeIsNotUnsigned:$cast_a)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXDimOp
//===----------------------------------------------------------------------===//


def DimOpToConstantPattern: Pat<
   (ONNXDimOp $input, $axis),
   (ONNXConstantOp
      (GetNullAttr), (createDenseElementsAttrFromShapeAtIndex $input, $axis),
      (GetNullFloatAttr), (GetNullArrayAttr), (GetNullIntegerAttr),
      (GetNullArrayAttr), (GetNullStringAttr), (GetNullArrayAttr)
   ),
   [(DimAtIndexIsConstant $input, $axis)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXSoftmaxV11 to the latest version
//===----------------------------------------------------------------------===//

// When axis is the last dimension of the input tensor, the semantics of V11 is
// the same as that of V13 (the latest version).
def SoftmaxV11ToLatestPattern: Pat<
   (ONNXSoftmaxV11Op $input, $axisAttr),
   (ONNXSoftmaxOp $input, $axisAttr),
   [(AxisIsTheLastDim $input, $axisAttr)]
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXShapeTransform
//===----------------------------------------------------------------------===//

// Remove ShapeTransform with identity map.
def ShapeTransformIdentityPattern : Pat<
  (ONNXShapeTransformOp $arg, $map1),
  (replaceWithValue $arg),
  [(IsIdentityAffineMap:$map1)]
>;

// Rewrite
// ```
//   onnx.ShapeTransform {#map1} (onnx.ShapeTransform {#map2} (%X))
// ```
// into
// ```
//   onnx.ShapeTransform {#map} (%X)
// ```
// where `#map = compose(#map1, #map2)

def ShapeTransformComposePattern : Pat<
  (ONNXShapeTransformOp (ONNXShapeTransformOp $arg, $map1), $map2),
  (ONNXShapeTransformOp $arg, (GetComposedMap $map2, $map1)),
  []
>;

//===----------------------------------------------------------------------===//
// Canonicalization for ONNXWhere
//===----------------------------------------------------------------------===//

// In this pattern, the condition in onnx.Where is always false, so we can replace
// onnx.Where by its "false" value.
// Condition in this pattern is a comparision between dimension sizes and negative values.
// Since dimension sizes are always positive, the condition is evaluated to false.

// This pattern was found in xlm-roberta-base-language-detection model in HuggingFace.

def AlwaysFalseWherePattern : Pat<
 (ONNXWhereOp (ONNXEqualOp (ONNXConcatOp $dims, $_), $negative_constant), $true_val, $false_val),
 (replaceWithValue $false_val),
 [(IsNegativeSplatConstant:$negative_constant), (AreAllDimSizes:$dims)]
>;

#endif // ONNX_REWRITE
