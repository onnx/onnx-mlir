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

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;
class HasRankOf<int rank> : Constraint<CPred<"$0.getType().isa<ShapedType>() && $0.getType().cast<ShapedType>().getRank() == " # rank>>;
def HasNoneType : Constraint<CPred<"$0.getType().isa<NoneType>()">>;
def HasSameElementType : Constraint<
    CPred<"($0.getType().dyn_cast<ShapedType>().getElementType() == "
          "convertONNXTypeToMLIRType($_builder, static_cast<onnx::TensorProto_DataType>($1.cast<::mlir::IntegerAttr>().getInt())))">,
    "has same element type">;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def GemmAlpha : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmBeta : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def GemmTransA : NativeCodeCall<"$_builder.getI64IntegerAttr(0)">;
def GemmTransB : NativeCodeCall<"$_builder.getI64IntegerAttr(0)">;

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

def ConstantPadPattern : Pat<(ONNXPadConstantValueOp $m1, (ONNXConstantOp:$res $v1, $v2), $m2, $m3),
                             (ONNXPadConstantValuePadOp $m1, $v2, $m2, $m3),
                             [(HasOneUse $res)]>;


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
  (ONNXTransposeOp $v, $p),
  // Remove the transpose.
  (replaceWithValue $v),
  // Check that we have indeed a identity transpose pattern.
  [(IsIdentityPermuteAttribute:$p)]>;


#endif // ONNX_COMBINE
