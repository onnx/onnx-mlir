//=- ONNXCombine.td - Pattern Match Optimizations for ONNX -*- tablegen -*-===//
//
// Copyright 2019 The IBM Research Authors.
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
include "dialect/onnx/onnx.td"
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

// ONNX_Op (onnx.Identity (%X)) = ONNX_Op (%X)
def IdentityEliminationPattern : Pat<(ONNXIdentityOp $arg),
                                     (replaceWithValue $arg)>;

def ConstantPadPattern : Pat<(ONNXPadConstantValueOp $m1, (ONNXConstantOp:$res $v1, $v2), $m2, $m3),
                             (ONNXPadConstantValuePadOp $m1, $v2, $m2, $m3),
                             [(HasOneUse $res)]>;

#endif // ONNX_COMBINE
