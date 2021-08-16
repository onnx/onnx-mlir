//===--- ONNXCombine.td - Pattern Match Opt for ONNX Dialect--*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_COMBINE
#define ONNX_COMBINE

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

// Usefult code generation invokation.
def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetUnitAttr: NativeCodeCall<"$_builder.getUnitAttr()">;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;
class HasRankOf<int rank> : Constraint<CPred<"$0.getType().isa<ShapedType>() && $0.getType().cast<ShapedType>().getRank() == " # rank>>;
def HasNoneType : Constraint<CPred<"$0.getType().isa<NoneType>()">>;
def HasSameElementType : Constraint<
    CPred<"($0.getType().dyn_cast<ShapedType>().getElementType() == "
          "$1.cast<::mlir::TypeAttr>().getValue())">,
    "has same element type">;

def AreTheSameAxisArray: Constraint<
    CPred<"AreTheSameAxisArray($0.getType().cast<ShapedType>().getRank(), $1, $2)">,
    "Two axis arrays are the same">;
def Has3DShape : Constraint<
    CPred<"((($0.getType().cast<MemRefType>()).getShape().size() == "
              "($1.getType().cast<MemRefType>()).getShape().size()) &&"
          " (($0.getType().cast<MemRefType>()).getShape()[0] == "
              "($1.getType().cast<MemRefType>()).getShape()[0]) &&"
          " (($0.getType().cast<MemRefType>()).getShape()[1] == "
              "($1.getType().cast<MemRefType>()).getShape()[1]) &&"
          " (($0.getType().cast<MemRefType>()).getShape()[2] == "
              "($1.getType().cast<MemRefType>()).getShape()[2]))">>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def GemmAlpha : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmBeta : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmTransA : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;
def GemmTransB : NativeCodeCall<"IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))">;


// onnx.add(onnx.matmul(%X, %Y), %Z) = onnx.Gemm(%X, %Y, %Z)
def MulAddToGemmOptPattern : Pat<(ONNXAddOp (ONNXMatMulOp:$res $m1, $m2), $m3),
                                 (ONNXGemmOp $m1, $m2, $m3, (GemmAlpha), (GemmBeta), (GemmTransA), (GemmTransB)),
                                 [(HasOneUse $res), (HasRankOf<2> $m1), (HasRankOf<2> $m2)]>;

// onnx.add(onnx.Gemm(%X, %Y, None), %Z) = onnx.Gemm(%X, %Y, %Z)
def FuseGemmFollowedByAddition : Pat<(ONNXAddOp (ONNXGemmOp:$res $m1, $m2, $none, $alpha, $beta, $transA, $transB), $bias),
                                     (ONNXGemmOp $m1, $m2, $bias, $alpha, $beta, $transA, $transB),
                                     [(HasOneUse $res), (HasNoneType $none)]>;

// ONNX_Op (onnx.Identity (%X)) = ONNX_Op (%X)
def IdentityEliminationPattern : Pat<(ONNXIdentityOp $arg),
                                     (replaceWithValue $arg)>;

// y, mask = onnx.Dropout(x) -> y, mask = x, none
def DropoutEliminationPattern : Pattern<(ONNXDropoutOp $arg, $arg1, $arg2, $ratio),
                                        [(replaceWithValue $arg),
                                         (ONNXConstantOp (GetNullAttr), (GetUnitAttr), (GetNullAttr), (GetNullAttr), (GetNullAttr), (GetNullAttr), (GetNullAttr), (GetNullAttr))]>;

// ONNX_Op (onnx.Cast (%X, $type)) = ONNX_Op (%X)
def CastEliminationPattern : Pat<
	(ONNXCastOp $arg, $type),
	(replaceWithValue $arg),
  [(HasSameElementType $arg, $type)]>;

// Combine transposes.
def CreateCombinedTransposedPattern :
   NativeCodeCall<"CombinedTransposePattern($_builder, $0, $1)">;

def IsIdentityPermuteAttribute :
  Constraint<CPred<"IsIdentityPermuteVector($_self)">,
    "has identity permute vector">;

def FuseTransposePattern:  Pat<
  // Transpose of a transpose.
  (ONNXTransposeOp (ONNXTransposeOp $v, $p1), $p2),
  // Transpose with combined pattern.
  (ONNXTransposeOp $v, (CreateCombinedTransposedPattern $p1, $p2))>;

def RemoveIdentityTransposePattern:  Pat<
  // Transpose with an identity pattern (e.g. {0, 1, 2, 3}).
  (ONNXTransposeOp $val, $p),
  // Remove the transpose.
  (replaceWithValue $val),
  // Check that we have indeed a identity transpose pattern.
  [(IsIdentityPermuteAttribute:$p)]>;

def FuseReshapePattern:  Pat<
  // Reshape of a reshape.
  (ONNXReshapeOp (ONNXReshapeOp $v, $s1), $s2),
  // Remove the first reshape op.
  (ONNXReshapeOp $v, $s2)>;

def RemoveIdentityReshapePattern:  Pat<
  // Remove an identity pattern. Input tensor already has the specified shape.
  (ONNXReshapeOp $val, $s),
  // Remove the transpose.
  (replaceWithValue $val),
  // Check that we have indeed the same shape tensors.
  [(Has3DShape $val, $s)]>;


/// Combine squeeze and unsqueeze.
/// Squeeze {axes = [a, b, c]} (Unsqueeze {axes = [a, b, c]} (%X)) = %X
def RemoveSqueezeUnsqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXSqueezeV11Op (ONNXUnsqueezeV11Op $val, $u_axes), $s_axes),
  // Remove the transpose.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxisArray $val, $u_axes, $s_axes)]>;

/// Combine unsqueeze and squeeze.
/// Unsqueeze {axes = [a, b, c]} (Squeeze {axes = [a, b, c]} (%X)) = %X
def RemoveUnsqueezeSqueezePattern:  Pat<
  // Squeeze and the unsqueeze with the same axes.
  (ONNXUnsqueezeV11Op (ONNXSqueezeV11Op:$res $val, $s_axes), $u_axes),
  // Remove the transpose.
  (replaceWithValue $val),
  // Check that both ops use the same `axes`.
  [(AreTheSameAxisArray $res, $u_axes, $s_axes)]>;

#endif // ONNX_COMBINE
