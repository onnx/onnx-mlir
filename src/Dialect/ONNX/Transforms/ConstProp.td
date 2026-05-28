// SPDX-License-Identifier: Apache-2.0

//===- ONNXConstProp.td - Rewriting for Constant Propagation in ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_CONSTPROP
#define ONNX_CONSTPROP

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Instruction to add new constant operation rules.
//===----------------------------------------------------------------------===//
// Minimally, you will have added operation in the ONNXConstProp.cpp to perform
// the element-wise single value handling of the new operator that you are dealing
// with. You will need to generate a call to the method that handle the tensor
// constant prop. Here is the call for a unary and binary operation. Adapt to your
// new operator:
//
// def CreateAddOfTwoConst :
//   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;
//
// def CreateNegOfConst :
//   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;
//
// where you will have mostly to substitute your new operator as well as using
// a new def name.
//
// Then you will need to add substitution rules, see examples below.


// Useful test definitions:

def HasOneUse : Constraint<CPred<"$_self.hasOneUse()">, "op has exactly one use">;

def IsNoneType : Constraint<CPred<"isa<NoneType>(($_self).getType())">>;

def IsIntOrFloatType : Constraint<CPred<"isa<IntegerType, FloatType>(mlir::cast<ShapedType>(($_self).getType()).getElementType())">>;

def IsNotAConstant :
  Constraint<CPred<"! static_cast<bool>(getDenseOrDisposableConstLikeElements($_self))">,
  "operation is not a constant">;

def IsFromDenseConstLike:
    Constraint<CPred<"static_cast<bool>(getDenseOrDisposableConstLikeElements($_self))">,
  "Value is produced by any ConstantLike op that folds to a dense "
  "ElementsAttr (DenseElementsAttr or DisposableElementsAttr)">;

def IsFromDenseConstLikeOrNone:
  Constraint<
    CPred<"static_cast<bool>(getDenseOrDisposableConstLikeElements($_self)) || mlir::isa<NoneType>($_self.getType())">,
    "Value is none or produced by any ConstantLike op that folds to a dense ElementsAttr"
  >;

def IsVariadicOperandDenseONNXConstantOp:
    Constraint<CPred<"isVariadicOperandFromDenseONNXConstantOp($_self)">,
  "All variadic operands are produced by dense ConstantLike ops">;

def HasStaticShape: Constraint<CPred<
  "hasStaticShape($_self.getType())">,
  "A value has static shape"
>;

def IsRankedShapedType: Constraint<CPred<
  "isRankedShapedType($_self.getType())">,
  "A value has a rank"
>;

def HasIntegerElementType: Constraint<CPred<
  "isa<IntegerType>(getElementType($_self.getType()))">,
  "A value has integer element type"
>;

def IsConstOfZeros : Constraint<
  CPred<"isConstOf($_self, 0.0)">,
  "Value is an all-zeros constant tensor">;

def IsConstOfOnes : Constraint<
  CPred<"isConstOf($_self, 1.0)">,
  "Value is an all-ones constant tensor">;

def IsNotScalar: Constraint<
  CPred<"!isScalarTensor($_self)">,
  "Value is not a scalar value">;

def ValuesHaveSameType : Constraint<
  CPred<"$0.getType() == $1.getType()">,
  "Values have same type">;

def ValuesHaveSameDType : Constraint<
  CPred<"$0.getType().cast<mlir::ShapedType>().getElementType() == $1.getType().cast<mlir::ShapedType>().getElementType()">,
  "Values have same dtype">;

def TransposeConstPropTypesMatch : Constraint<
  CPred<"valuesHaveSameDTypeForTransposeOfConst($0, $1)">,
  "Transpose result dtype matches constant input after per-axis quant axis remap">;

def IsMatMulIntegerLhsZero: Constraint<
    CPred<"isMatMulIntegerLhsZero($0, $1)">,
    "MatMulInteger lhs matrix is zero for given zero point">;

def IsMatMulIntegerRhsZero: Constraint<
    CPred<"isMatMulIntegerRhsZero($0, $1)">,
    "MatMulInteger rhs matrix is zero for given zero point">;

def OpIsUnlikelyToBeFusedWithBias: Constraint<
  CPred<"!canOpLikelyBeFusedWithBias($_self)">,
  "Operation can unlikely be fused with the following bias">;

def OpIsLikelyToBeFusedWithBias: Constraint<
  CPred<"canOpLikelyBeFusedWithBias($_self)">,
  "Operation can likely be fused with the following bias">;

def SatisfiesExpansionBound: Constraint<
  CPred<"satisfiesExpansionBound($_self)">>;

class IsNotDisabled<string name>: Constraint<
  CPred<"isNotDisabled(\"" # name # "\")">>;

// Takes initial name parameter which is checked to see if disabled.
class NamedPat<string name, dag pattern, dag result, list<dag> preds = [], list<dag> supplementalPatterns = [],
  dag benefitAdded = (addBenefit 0)> :
  Pat<pattern, result, !listconcat([(IsNotDisabled<name>)], preds), supplementalPatterns, benefitAdded>;

class EqualString<string s> : Constraint<CPred<"$0 == \"" # s # "\"">>;

// Creation helpers:

def CreateMaximumValueForClip: NativeCodeCall<
  "createMaximumValueForClip($_builder, cast<mlir::ShapedType>($0.getType()), $1)"
>;

def CreateMinimumValueForClip: NativeCodeCall<
  "createMinimumValueForClip($_builder, cast<mlir::ShapedType>($0.getType()), $1)"
>;

def CreateONNXMinOp : NativeCodeCall<"$_builder.create<ONNXMinOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;
def CreateONNXMaxOp : NativeCodeCall<"$_builder.create<ONNXMaxOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;

def CreateZeroTensorOfType: NativeCodeCall<
  "ConstZeroTensor($_builder, $_loc, mlir::cast<mlir::ShapedType>($0.getType()))"
>;

def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateSubOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXSubOp>($_builder, $0, $1, $2)">;

def CreateCastOfConst :
   NativeCodeCall<"ConstPropCast($_builder, $0, $1, $2, $3)">;

def CreateBitwiseNotOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXBitwiseNotOp>($_builder, $0, $1)">;

def CreateAbsOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXAbsOp>($_builder, $0, $1)">;

def CreateCeilOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXCeilOp>($_builder, $0, $1)">;

def CreateCosOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXCosOp>($_builder, $0, $1)">;

def CreateErfOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXErfOp>($_builder, $0, $1)">;

def CreateExpOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXExpOp>($_builder, $0, $1)">;

def CreateFloorOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXFloorOp>($_builder, $0, $1)">;

def CreateLogOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXLogOp>($_builder, $0, $1)">;

def CreateNegOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;

def CreateNotOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNotOp>($_builder, $0, $1)">;

def CreateShapeOfConst :
   NativeCodeCall<"ConstPropShape($_builder, $0, $1)">;

def CreateSinOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSinOp>($_builder, $0, $1)">;

def CreateSigmoidOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSigmoidOp>($_builder, $0, $1)">;

def CreateSqrtOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSqrtOp>($_builder, $0, $1)">;

def CreateReluOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXReluOp>($_builder, $0, $1)">;

def CreateReciprocalOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXReciprocalOp>($_builder, $0, $1)">;

def CreateRoundOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXRoundOp>($_builder, $0, $1)">;

def CreateMaxOfConst :
   NativeCodeCall<"ConstPropVariadicElementwiseBinary<mlir::ONNXMaxOp>($_builder, $0, $1)">;

def CreateMinOfConst :
   NativeCodeCall<"ConstPropVariadicElementwiseBinary<mlir::ONNXMinOp>($_builder, $0, $1)">;

def CreateSumOfConst :
   NativeCodeCall<"ConstPropVariadicElementwiseBinary<mlir::ONNXSumOp>($_builder, $0, $1)">;

def CreateMulOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXMulOp>($_builder, $0, $1, $2)">;

def CreateDivOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXDivOp>($_builder, $0, $1, $2)">;

def CreateEqualOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXEqualOp>($_builder, $0, $1, $2)">;

def CreateLessOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXLessOp>($_builder, $0, $1, $2)">;

def CreateGreaterOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXGreaterOp>($_builder, $0, $1, $2)">;

def CreateLessOrEqualOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXLessOrEqualOp>($_builder, $0, $1, $2)">;

def CreateGreaterOrEqualOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXGreaterOrEqualOp>($_builder, $0, $1, $2)">;

def CreateBitwiseAndOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXBitwiseAndOp>($_builder, $0, $1, $2)">;

def CreateBitwiseOrOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXBitwiseOrOp>($_builder, $0, $1, $2)">;

def CreateAndOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAndOp>($_builder, $0, $1, $2)">;

def CreateOrOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXOrOp>($_builder, $0, $1, $2)">;

def CreateXorOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXXorOp>($_builder, $0, $1, $2)">;

def CreatePowOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXPowOp>($_builder, $0, $1, $2)">;

def CreateModOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXModOp>($_builder, $0, $1, $2)">;

def CreateWhereOfThreeConst :
   NativeCodeCall<"ConstPropWhere($_builder, $0, $1, $2, $3)">;

def CreateReduceSumConst :
    NativeCodeCall<"ConstPropReduce<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateReduceProdConst :
    NativeCodeCall<"ConstPropReduce<mlir::ONNXMulOp>($_builder, $0, $1, $2)">;

def CreateReduceMinConst :
    NativeCodeCall<"ConstPropReduce<mlir::ONNXMinOp>($_builder, $0, $1, $2)">;

def CreateReduceMaxConst :
    NativeCodeCall<"ConstPropReduce<mlir::ONNXMaxOp>($_builder, $0, $1, $2)">;

def CreateReduceMeanConst :
    NativeCodeCall<"ConstPropReduce<mlir::ONNXReduceMeanOp>($_builder, $0, $1, $2)">;

def CreateMatMulOfConsts :
    NativeCodeCall<"ConstPropMatMul($_builder, $0, $1, $2)">;

def CreateMatMulIntegerOfConsts :
    NativeCodeCall<"ConstPropMatMulInteger($_builder, $0, $1, $2, $3, $4)">;

def CreateGemmOfConsts :
    NativeCodeCall<"ConstPropGemm($_builder, $0, $1, $2, $3)">;

def CreateTransposeOfConst :
   NativeCodeCall<"ConstPropTranspose($_builder, $0, $1)">;

def CreateReverseSequenceOfConst :
   NativeCodeCall<"ConstPropReverseSequence($_builder, $0, $1, $2)">;

def CreateUnsqueezeOfConst:
   NativeCodeCall<"ConstPropUnsqueeze($_builder, $0, $1)">;

def CreateSqueezeOfConst:
   NativeCodeCall<"ConstPropSqueeze($_builder, $0, $1)">;

def CreateSliceOfConst:
   NativeCodeCall<"ConstPropSlice($_builder, $0, $1)">;

def CreatePadOfConst:
   NativeCodeCall<"ConstPropPad($_builder, $0, $1, $2)">;

def CreateConcatOfConst:
   NativeCodeCall<"ConstPropConcat($_builder, $0, $1, $2)">;

def CreateExpandOfConst:
   NativeCodeCall<"ConstPropExpand($_builder, $0, $1)">;

def CreateGatherOfConst:
   NativeCodeCall<"ConstPropGather($_builder, $0, $1, $2)">;

def CreateReshapeOfConst:
   NativeCodeCall<"ConstPropReshape($_builder, $0, $1)">;

def CreateConstantOfShapeOfConst:
   NativeCodeCall<"ConstPropConstantOfShape($_builder, $0, $1, $2)">;

def CreateRangeOfThreeConst :
   NativeCodeCall<"ConstPropRange($_builder, $0, $1, $2, $3)">;

def CreateNonZeroOfConst :
   NativeCodeCall<"ConstPropNonZero($_builder, $0, $1)">;

def CreateScatterNDOfConst :
    NativeCodeCall<"ConstPropScatterND($_builder, $0, $1, $2, $3)">;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise ADD operations.
//===----------------------------------------------------------------------===//

// Use commutativity to normalize constants in the second position of Add.
def AddConstCommutative1 : NamedPat<"AddConstCommutative1",
  // From add(c, x).
  (ONNXAddOp:$addOp $c, $x),
  // To add(x, c).
  (ONNXAddOp $x, $c, (location $addOp)),
  // To avoid infinite loop, constrain the first argument to be a dense constant
  // and the second to not be one.
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x)]>;

// Use associativity to add constants together.
def AddConstAssociative1 : NamedPat<"AddConstAssociative1",
  // From add(add(x, c1), c2).
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, $c1),
    $c2),
  // To add(x, add(c1, c2)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $c1, $c2)),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2), (IsNotAConstant:$x), (HasOneUse:$lhs)]>;

// Do not apply this when both x and c are scalar since it is cheap to do.
def AddConstAssociative2 : NamedPat<"AddConstAssociative2",
  // From add(add(x, c), y).
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, $c),
    $y),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x), (OpIsUnlikelyToBeFusedWithBias:$x), (IsNotAConstant:$y), (HasOneUse:$lhs),
   (IsNotScalar:$x), (IsNotScalar:$c)]>;

// Do not apply this when both y and c are scalar since it is cheap to do.
def AddConstAssociative3 : NamedPat<"AddConstAssociative3",
  // From add(x, add(y, c)).
  (ONNXAddOp
    $x,
    (ONNXAddOp:$rhs $y, $c)),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x), (IsNotAConstant:$y), (OpIsUnlikelyToBeFusedWithBias:$y), (HasOneUse:$rhs),
   (IsNotScalar:$y), (IsNotScalar:$c)]>;

def AddConstAssociative4NoFuse : NamedPat<"AddConstAssociative4NoFuse",
  // From add(add(x, c1), add(y, c2)), if both x and y are not likely to be fused with bias
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, $c1),
    (ONNXAddOp:$rhs $y, $c2)),
  // To add(add(x, y), c1+c2).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    (ONNXAddOp $c1, $c2)),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2),
   (IsNotAConstant:$x), (IsNotAConstant:$y), 
   (OpIsUnlikelyToBeFusedWithBias:$x), (OpIsUnlikelyToBeFusedWithBias:$y),
   (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

def AddConstAssociative4LhsFuse : NamedPat<"AddConstAssociativeLhsFuse",
  // From add(add(x, c1), add(y, c2)), if x is likely to be fused with bias
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, $c1),
    (ONNXAddOp:$rhs $y, $c2)),
  // To add(add(x, c1+c2), y).
  (ONNXAddOp
    (ONNXAddOp $x, (ONNXAddOp $c1, $c2)),
     $y
    ),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2),
   (IsNotAConstant:$x), (IsNotAConstant:$y), 
   (OpIsLikelyToBeFusedWithBias:$x),
   (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

def AddConstAssociative4RhsFuse : NamedPat<"AddConstAssociativeRhsFuse",
  // From add(add(x, c1), add(y, c2)), if y is likely to be fused with bias and x is not
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, $c1),
    (ONNXAddOp:$rhs $y, $c2)),
  // To add(x, add(y, c1+c2)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $y, (ONNXAddOp $c1, $c2))
    ),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2),
   (IsNotAConstant:$x), (IsNotAConstant:$y), 
   (OpIsUnlikelyToBeFusedWithBias:$x), (OpIsLikelyToBeFusedWithBias:$y),
   (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

// Constant Propagation for Add
def AddConstProp : NamedPat<"AddConstProp",
    // From add(c1, c2).
    (ONNXAddOp:$addOp $lhs,
                      $rhs),
    // To c1+c2
    (CreateAddOfTwoConst $addOp, $lhs, $rhs),
    // Additional constraints (dense)
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$addOp), (HasStaticShape:$addOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def AddZerosOnRhs : NamedPat<"AddZerosOnRhs",
    // From add(x, c).
    (ONNXAddOp:$result
      $x,
      $c
    ),
    // To x.
    (replaceWithValue $x),
    [(IsNotAConstant:$x), (ValuesHaveSameType $result, $x), (IsConstOfZeros:$c)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise SUB / NEG operations.
//===----------------------------------------------------------------------===//

// Constant Propagation for Sub
def SubConstProp : NamedPat<"SubConstProp",
    // From sub(c1, c2).
    (ONNXSubOp:$subOp $lhs,
                      $rhs),
    // To c1-c2
    (CreateSubOfTwoConst $subOp, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$subOp), (HasStaticShape:$subOp)]>;

// TODO: Expand $a to $result's shape instead of requiring ValuesHaveSameType.
def SubZerosOnRhs : NamedPat<"SubZerosOnRhs",
    // From sub(x, c).
    (ONNXSubOp:$result
      $x,
      $c
    ),
    // To x.
    (replaceWithValue $x),
    [(IsNotAConstant:$x), (ValuesHaveSameType $result, $x), (IsConstOfZeros:$c)]>;

// Cast of constant is simply a constant with the new type.
def CastofConst : NamedPat<"CastofConst",
    // From (c)
    (ONNXCastOp:$castOp $input, $saturate, $to),
    // To (c)
    (CreateCastOfConst $castOp, $input, $saturate, $to),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for BitwiseNot
def BitwiseNotConstProp : NamedPat<"BitwiseNotofConst",
    // From bitwise_not(c).
    (ONNXBitwiseNotOp:$bitwiseNotOp $input),
    // To ~c
    (CreateBitwiseNotOfConst $bitwiseNotOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Abs
def AbsConstProp : NamedPat<"AbsofConst",
    // From abs(c).
    (ONNXAbsOp:$ceilOp $input),
    // To new_c
    (CreateAbsOfConst $ceilOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Ceil
def CeilConstProp : NamedPat<"CeilofConst",
    // From ceil(c).
    (ONNXCeilOp:$ceilOp $input),
    // To new_c
    (CreateCeilOfConst $ceilOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Cos
def CosConstProp : NamedPat<"CosofConst",
    // From cos(c).
    (ONNXCosOp:$cosOp $input),
    // To new_c.
    (CreateCosOfConst $cosOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Erf
def ErfConstProp : NamedPat<"ErfofConst",
    // From erf(c)
    (ONNXErfOp:$erfOp $input),
    // To new_c.
    (CreateErfOfConst $erfOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Exp
def ExpConstProp : NamedPat<"ExpofConst",
    // From exp(c).
    (ONNXExpOp:$expOp $input),
    // To new_c.
    (CreateExpOfConst $expOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Floor
def FloorConstProp : NamedPat<"FloorofConst",
    // From floor(c).
    (ONNXFloorOp:$floorOp $input),
    // To new_c.
    (CreateFloorOfConst $floorOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Log
def LogConstProp : NamedPat<"LogofConst",
    // From log(c)
    (ONNXLogOp:$logOp $input),
    // To new_c.
    (CreateLogOfConst $logOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Neg of constant is simply -const
def NegofConst : NamedPat<"NegofConst",
    // From - (c)
    (ONNXNegOp:$negOp $input),
    // To (-c)
    (CreateNegOfConst $negOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Not of constant is simply !const
def NotofConst : NamedPat<"NotofConst",
    // From c
    (ONNXNotOp:$notOp $input),
    // To !c
    (CreateNotOfConst $notOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Shape
def ShapeConstProp : NamedPat<"ShapeofConst",
    (ONNXShapeOp:$shapeOp $input, $_, $_),
    (CreateShapeOfConst $shapeOp, $input),
    [(HasStaticShape:$input)]>;

// Constant Propagation for Sin
def SinConstProp : NamedPat<"SinofConst",
    // From sin(c).
    (ONNXSinOp:$sinOp $input),
    // To new_c
    (CreateSinOfConst $sinOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Sigmoid
def SigmoidConstProp : NamedPat<"SigmoidofConst",
    // From 1 / (1 + exp(-c)).
    (ONNXSigmoidOp:$sigmoidOp $input),
    // To new_c
    (CreateSigmoidOfConst $sigmoidOp, $input),
    [(IsFromDenseConstLike:$input), (HasStaticShape:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Reciprocal
def ReciprocalConstProp : NamedPat<"ReciprocalofConst",
    // From 1/c.
    (ONNXReciprocalOp:$reciprocalOp $input),
    // To new_c
    (CreateReciprocalOfConst $reciprocalOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Round
def RoundConstProp : NamedPat<"RoundofConst",
    // From round(c)
    (ONNXRoundOp:$roundOp $input),
    // To new_c.
    (CreateRoundOfConst $roundOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Change a subtraction of a constant c by an addition of -c. Helpfull to combine
// with other add optimizations.
def SubConstToNeg : NamedPat<"SubConstToNeg",
    // From x - c.
    (ONNXSubOp:$subOp $x, $input),
    // To x + (-c).
    (ONNXAddOp $x, (CreateNegOfConst $input, $input)),
    [(IsNotAConstant:$x), (IsFromDenseConstLike:$input),
     (IsIntOrFloatType:$input)]>;

// Constant Propagation for Sqrt
def SqrtofConst : NamedPat<"SqrtofConst",
    // From  onnx.Sqrt(c)
    (ONNXSqrtOp:$sqrtOp $input),
    // To sqrt(c)
    (CreateSqrtOfConst $sqrtOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Relu
def ReluofConst : NamedPat<"ReluofConst",
    // From  onnx.Relu(c)
    (ONNXReluOp:$reluOp $input),
    // To relu(c)
    (CreateReluOfConst $reluOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Reciprocal
def ReciprocalOfConst : NamedPat<"ReciprocalOfConst",
    // From  onnx.Reciprocal(c)
    (ONNXReciprocalOp:$reciprocalOp $input),
    // To reciprocal(c)
    (CreateReciprocalOfConst $reciprocalOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

// Constant Propagation for Round
def RoundofConst : NamedPat<"RoundofConst",
    // From  onnx.Round(c)
    (ONNXRoundOp:$reluOp $input),
    // To round_even(c)
    (CreateRoundOfConst $reluOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

//===----------------------------------------------------------------------===//
// Const propagation patterns for variadic elementwise operations.
//===----------------------------------------------------------------------===//

// Constant Propagation for Max
def MaxConstProp : NamedPat<"MaxConstProp",
    // From max(c1, c2).
    (ONNXMaxOp:$maxOp $operandList),
    // To c = max(c1, c2)
    (CreateMaxOfConst $maxOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (IsIntOrFloatType:$maxOp),
     (SatisfiesExpansionBound:$maxOp), (HasStaticShape:$maxOp)]>;

// Constant Propagation for Min
def MinConstProp : NamedPat<"MinConstProp",
    // From min(c1, c2).
    (ONNXMinOp:$minOp $operandList),
    // To c = min(c1, c2)
    (CreateMinOfConst $minOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (IsIntOrFloatType:$minOp),
     (SatisfiesExpansionBound:$minOp), (HasStaticShape:$minOp)]>;

// Constant Propagation for Sum
def SumConstProp : NamedPat<"SumConstProp",
    // From sum(c1, c2).
    (ONNXSumOp:$sumOp $operandList),
    // To c = sum(c1, c2)
    (CreateSumOfConst $sumOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (IsIntOrFloatType:$sumOp),
     (SatisfiesExpansionBound:$sumOp), (HasStaticShape:$sumOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise MUL operations.
// Exactly the same pattern as for the elementwise ADD operations.
//===----------------------------------------------------------------------===//

// Use commutativity to normalize constants in the second position of Mul.
def MulConstCommutative1 : NamedPat<"MulConstCommutative1",
  // From mul(c, x).
  (ONNXMulOp:$mulOp $c, $x),
  // To mul(x, c).
  (ONNXMulOp $x, $c, (location $mulOp)),
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x)]>;

// Use associativity to mul constants together.
def MulConstAssociative1 : NamedPat<"MulConstAssociative1",
  // From mul(mul(x, c1), c2).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, $c1),
    $c2),
  // To mul(x, mul(c1, c2)).
  (ONNXMulOp
    $x,
    (ONNXMulOp $c1, $c2)),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2), (IsNotAConstant:$x), (HasOneUse:$lhs)]>;

// Do not apply this when both x and c are scalar since it is cheap to do.
def MulConstAssociative2 : NamedPat<"MulConstAssociative2",
  // From mul(mul(x, c), y).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, $c),
    $y),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs),
   (IsNotScalar:$x), (IsNotScalar:$c)]>;

// Do not apply this when both y and c are scalar since it is cheap to do.
def MulConstAssociative3 : NamedPat<"MulConstAssociative3",
  // From mul(x, mul(y, c)).
  (ONNXMulOp
    $x,
    (ONNXMulOp:$rhs $y, $c)),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
  [(IsFromDenseConstLike:$c), (IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$rhs),
   (IsNotScalar:$y), (IsNotScalar:$c)]>;

def MulConstAssociative4 : NamedPat<"MulConstAssociative4",
  // From mul(mul(x, c1), mul(y, c2)).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, $c1),
    (ONNXMulOp:$rhs $y, $c2)),
  // To mul(mul(x, y), c1+c2).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    (ONNXMulOp $c1, $c2)),
  [(IsFromDenseConstLike:$c1), (IsFromDenseConstLike:$c2),
   (IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

// mul(add(x, b), a) -> add(mul(x, a), mul(a, b)).
// mul(a, b) is folded to a single constant by MulConstProp.
def MulAddConstDistributive : NamedPat<"MulAddConstDistributive",
  // From mul(add(x, b), a) where a, b are constants.
  (ONNXMulOp
    (ONNXAddOp:$addOp $x, $b),
    $a),
  // To add(mul(x, a), mul(a, b)).
  (ONNXAddOp
    (ONNXMulOp $x, $a),
    (ONNXMulOp $a, $b)),
  [(IsFromDenseConstLike:$a), (IsFromDenseConstLike:$b),
   (IsNotAConstant:$x), (HasOneUse:$addOp), (IsIntOrFloatType:$a),
  (IsIntOrFloatType:$b), (IsIntOrFloatType:$x)]>;

// Distribute multiplication into clip: mul(clip(x, l, u), a) -> clip(mul(x, a), mul(a, l), mul(a, u)).
// mul(a, l) and mul(a, u) are folded to single constants by MulConstProp in a later iteration.
def MulClipConstDistributive : NamedPat<"MulClipConstDistributive",
  // From mul(clip(x, l, u), a) where a, l, u are constants.
  (ONNXMulOp
    (ONNXClipOp:$clipOp $x,
      $l,
      $u),
    $a),
  // To clip(mul(x, a), mul(a, l), mul(a, u)).
  (ONNXClipOp
    (ONNXMulOp $x, $a),
    (ONNXMulOp $a, $l),
    (ONNXMulOp $a, $u)),
  [(IsFromDenseConstLike:$a), (IsFromDenseConstLike:$l), (IsFromDenseConstLike:$u),
   (IsNotAConstant:$x), (HasOneUse:$clipOp), (IsIntOrFloatType:$x), 
  (IsIntOrFloatType:$a), (IsIntOrFloatType:$l), (IsIntOrFloatType:$u)]>;

// Constant Propagation for Mul
def MulConstProp : NamedPat<"MulConstProp",
    // From mul(c1, c2).
    (ONNXMulOp:$mulOp $lhs,
                      $rhs),
    // To c1+c2
    (CreateMulOfTwoConst $mulOp, $lhs, $rhs),
    // Multiplication constraints
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$mulOp), (HasStaticShape:$mulOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def MulOnesOnRhs : NamedPat<"MulOnesOnRhs",
    // From mul(x, c).
    (ONNXMulOp:$result
      $x,
      $c
    ),
    // To x.
    (replaceWithValue $x),
    [
      (IsNotAConstant:$x), (IsConstOfOnes:$c),
      (ValuesHaveSameType $result, $x)
    ]>;

// Constant Propagation for Div
def DivConstProp : NamedPat<"DivConstProp",
    // From div(c1, c2).
    (ONNXDivOp:$divOp $lhs,
                      $rhs),
    // To c1/c2
    (CreateDivOfTwoConst $divOp, $lhs, $rhs),
    // Division constraints
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$divOp), (HasStaticShape:$divOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def DivOnesOnRhs : NamedPat<"DivOnesOnRhs",
    // From div(x, c).
    (ONNXDivOp:$result
      $x,
      $c
    ),
    // To x.
    (replaceWithValue $x),
    [
      (IsNotAConstant:$x), (IsConstOfOnes:$c),
      (ValuesHaveSameType $result, $x)
    ]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXBitwiseAndOp
//===----------------------------------------------------------------------===//

def BitwiseAndConstPropPattern : NamedPat<"BitwiseAndConstPropPattern",
    (ONNXBitwiseAndOp:$result
      $lhs,
      $rhs),
    (CreateBitwiseAndOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXBitwiseOrOp
//===----------------------------------------------------------------------===//

def BitwiseOrConstPropPattern : NamedPat<"BitwiseOrConstPropPattern",
    (ONNXBitwiseOrOp:$result
      $lhs,
      $rhs),
    (CreateBitwiseOrOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXAndOp
//===----------------------------------------------------------------------===//

def AndConstPropPattern : NamedPat<"AndConstPropPattern",
    (ONNXAndOp:$result
      $lhs,
      $rhs),
    (CreateAndOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXOrOp
//===----------------------------------------------------------------------===//

def OrConstPropPattern : NamedPat<"OrConstPropPattern",
    (ONNXOrOp:$result
      $lhs,
      $rhs),
    (CreateOrOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXorOp
//===----------------------------------------------------------------------===//

def XorConstPropPattern : NamedPat<"XorConstPropPattern",
    (ONNXXorOp:$result
      $lhs,
      $rhs),
    (CreateXorOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXEqualOp
//===----------------------------------------------------------------------===//

def EqualConstProp : NamedPat<"EqualConstProp",
    // From equal(c1, c2).
    (ONNXEqualOp:$result
      $lhs,
      $rhs),
    // To c1 == c2
    (CreateEqualOfTwoConst $result, $lhs, $rhs),
    // constraints
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXLessOp
//===----------------------------------------------------------------------===//

def LessConstPropPattern : NamedPat<"LessConstPropPattern",
    (ONNXLessOp:$result
      $lhs,
      $rhs),
    (CreateLessOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXGreaterOp
//===----------------------------------------------------------------------===//

def GreaterConstPropPattern : NamedPat<"GreaterConstPropPattern",
    (ONNXGreaterOp:$result
      $lhs,
      $rhs),
    (CreateGreaterOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//

def LessOrEqualConstPropPattern : NamedPat<"LessOrEqualConstPropPattern",
    (ONNXLessOrEqualOp:$result
      $lhs,
      $rhs),
    (CreateLessOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//

def GreaterOrEqualConstPropPattern : NamedPat<"GreaterOrEqualConstPropPattern",
    (ONNXGreaterOrEqualOp:$result
      $lhs,
      $rhs),
    (CreateGreaterOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$result), (HasStaticShape:$result)]>;

//===----------------------------------------------------------------------===//
//  Constant propagation for ONNXModOp
//===----------------------------------------------------------------------===//

def ModConstPropPattern : NamedPat<"ModConstPropPattern",
    (ONNXModOp:$modOp
      $A,
      $B,
      $fmod),
    (CreateModOfTwoConst  $modOp, $A, $B),
    [(IsFromDenseConstLike:$A), (IsFromDenseConstLike:$B),
    (IsIntOrFloatType:$A), (IsIntOrFloatType:$B),
    (SatisfiesExpansionBound:$modOp), (HasStaticShape:$modOp)]>;

//===----------------------------------------------------------------------===//
// Pattern for Clip.
// Converts clip into min(max(c0, c1), c2) as Min and Max
// already have their own foldings.
// If c1 or c2 isn't defined, default max to numeric_limits::max() and
// min to numeric_limits::lowest() values according to the element type.
//===----------------------------------------------------------------------===//

// Constant Propagation for Clip
def ClipConstProp : NamedPat<"ClipConstProp",
    // From clip(c0, c1, c2).
    (ONNXClipOp:$clipOp $input,
                         $min, $max),
    // To min(max(c0, c1), c2).
    (CreateONNXMinOp
      $clipOp,
      (CreateONNXMaxOp $clipOp, $input, (CreateMinimumValueForClip $input, $min)),
      (CreateMaximumValueForClip $input, $max)),
    // Clip constraints
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input),
     (IsFromDenseConstLikeOrNone:$min), (IsFromDenseConstLikeOrNone:$max),
     (SatisfiesExpansionBound:$clipOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Where.
//===----------------------------------------------------------------------===//

// Constant Propagation for Where
def WhereConstProp : NamedPat<"WhereConstProp",
    // From where(c0, c1, c2).
    (ONNXWhereOp:$whereOp $condition,
                      $X,
                      $Y),
    // To c0?c1:c2
    (CreateWhereOfThreeConst $whereOp, $condition, $X, $Y),
    // Where constraints
    [(IsFromDenseConstLike:$condition),
     (IsFromDenseConstLike:$X), (IsFromDenseConstLike:$Y),
     (IsIntOrFloatType:$X), (IsIntOrFloatType:$Y),
     (SatisfiesExpansionBound:$whereOp), (HasStaticShape:$whereOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Reduce ops.
//===----------------------------------------------------------------------===//

def ReduceSumConstProp : NamedPat<"ReduceSumConstProp",
      (ONNXReduceSumOp:$reduceSumOp
        $data,
        $axes,
        $_,
        $_),
    (CreateReduceSumConst $reduceSumOp, $data, $axes),
    [(IsFromDenseConstLike:$data), (IsIntOrFloatType:$data),
     (IsFromDenseConstLikeOrNone:$axes)]
    >;

def ReduceProdConstProp : NamedPat<"ReduceProdConstProp",
      (ONNXReduceProdOp:$reduceProdOp
        $data,
        $axes,
        $_,
        $_),
    (CreateReduceProdConst $reduceProdOp, $data, $axes),
    [(IsFromDenseConstLike:$data), (IsIntOrFloatType:$data),
     (IsFromDenseConstLikeOrNone:$axes)]
    >;

def ReduceMinConstProp : NamedPat<"ReduceMinConstProp",
      (ONNXReduceMinOp:$reduceMinOp
        $data,
        $axes,
        $_,
        $_),
    (CreateReduceMinConst $reduceMinOp, $data, $axes),
    [(IsFromDenseConstLike:$data), (IsIntOrFloatType:$data),
     (IsFromDenseConstLikeOrNone:$axes)]
    >;

def ReduceMaxConstProp : NamedPat<"ReduceMaxConstProp",
      (ONNXReduceMaxOp:$reduceMaxOp
        $data,
        $axes,
        $_,
        $_),
    (CreateReduceMaxConst $reduceMaxOp, $data, $axes),
    [(IsFromDenseConstLike:$data), (IsIntOrFloatType:$data),
     (IsFromDenseConstLikeOrNone:$axes)]
    >;

def ReduceMeanConstProp : NamedPat<"ReduceMeanConstProp",
      (ONNXReduceMeanOp:$reduceMeanOp
        $data,
        $axes,
        $_,
        $_),
    (CreateReduceMeanConst $reduceMeanOp, $data, $axes),
    [(IsFromDenseConstLike:$data), (IsIntOrFloatType:$data),
     (IsFromDenseConstLikeOrNone:$axes)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for MatMul.
//===----------------------------------------------------------------------===//

// Limited to integers because 0 * NaN is NaN, not 0.
def MatMulZerosOnLhs : NamedPat<"MatMulZerosOnLhs",
    (ONNXMatMulOp:$resOp
      $A,
      $B
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsNotAConstant:$B), (HasIntegerElementType:$resOp),
     (HasStaticShape:$resOp), (IsConstOfZeros:$A)]
    >;

// Limited to integers because NaN * 0 is NaN, not 0.
def MatMulZerosOnRhs : NamedPat<"MatMulZerosOnRhs",
    (ONNXMatMulOp:$resOp
      $A,
      $B
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsNotAConstant:$A), (HasIntegerElementType:$resOp),
     (HasStaticShape:$resOp), (IsConstOfZeros:$B)]
    >;

def MatMulOfConsts : NamedPat<"MatMulOfConsts",
    (ONNXMatMulOp:$resOp
      $A,
      $B
    ),
    (CreateMatMulOfConsts $resOp, $A, $B),
    [(IsFromDenseConstLike:$A), (IsFromDenseConstLike:$B),
     (IsIntOrFloatType:$A), (IsIntOrFloatType:$B),
     (SatisfiesExpansionBound:$resOp)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for MatMulInteger.
//===----------------------------------------------------------------------===//

def MatMulIntegerConstZeroLhs : NamedPat<"MatMulIntegerConstZeroLhs",
    (ONNXMatMulIntegerOp:$resOp
      $A,
      $B,
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseConstLike:$A), (IsFromDenseConstLikeOrNone:$a_zero_point),
     (IsMatMulIntegerLhsZero $A, $a_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerConstZeroRhs : NamedPat<"MatMulIntegerConstZeroRhs",
    (ONNXMatMulIntegerOp:$resOp
      $A,
      $B,
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseConstLike:$B), (IsFromDenseConstLikeOrNone:$b_zero_point),
     (IsMatMulIntegerRhsZero $B, $b_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerOfConsts : NamedPat<"MatMulIntegerOfConsts",
    (ONNXMatMulIntegerOp:$resOp
      $A,
      $B,
      $a_zero_point, $b_zero_point
    ),
    (CreateMatMulIntegerOfConsts $resOp, $A, $B, $a_zero_point, $b_zero_point),
    [(IsFromDenseConstLike:$A), (IsFromDenseConstLikeOrNone:$a_zero_point),
     (IsFromDenseConstLike:$B), (IsFromDenseConstLikeOrNone:$b_zero_point),
     (SatisfiesExpansionBound:$resOp)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for Gemm.
//===----------------------------------------------------------------------===//

// Attributes are not passed to CreateGemmOfConsts. It reads them from gemmOp.
def GemmConstProp : NamedPat<"GemmConstProp",
    (ONNXGemmOp:$gemmOp
      $A,
      $B,
      $C, $alpha, $beta, $transA, $transB),
    (CreateGemmOfConsts $gemmOp, $A, $B, $C),
    [(IsFromDenseConstLike:$A), (IsFromDenseConstLike:$B),
     (IsIntOrFloatType:$A), (IsIntOrFloatType:$B),
     (IsFromDenseConstLikeOrNone:$C),
     (SatisfiesExpansionBound:$gemmOp), (HasStaticShape:$gemmOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Transpose operations.
//===----------------------------------------------------------------------===//

def TransposeofConst : NamedPat<"TransposeofConst",
    // From TransposeOp(c, p)
    (ONNXTransposeOp:$resOp $input, $_),
    // To c' where c' is transposed attribute
    (CreateTransposeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input),
     (TransposeConstPropTypesMatch $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with ReverseSequence operations.
//===----------------------------------------------------------------------===//

def ReverseSequenceofConst : NamedPat<"ReverseSequenceofConst",
    // From ReverseSequenceOp(c, ba, ta)    
    (ONNXReverseSequenceOp:$resOp $input,
                      $sequence_lens, $batch_axis, $time_axis),
    (CreateReverseSequenceOfConst $resOp, $input, $sequence_lens),
    [(IsFromDenseConstLike:$input), (IsFromDenseConstLike:$sequence_lens),
     (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Unsqueeze operations.
//===----------------------------------------------------------------------===//

def UnsqueezeofConst : NamedPat<"UnsqueezeofConst",
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeOp:$resOp $input, $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (HasStaticShape:$resOp),
     (ValuesHaveSameDType $resOp, $input)]>;

def UnsqueezeV11ofConst : NamedPat<"UnsqueezeV11ofConst",
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeV11Op:$resOp $input, $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (HasStaticShape:$resOp),
     (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Squeeze operations.
//===----------------------------------------------------------------------===//

def SqueezeofConst : NamedPat<"SqueezeofConst",
    // From Squeeze (c, axis)
    (ONNXSqueezeOp:$resOp $input, $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (HasStaticShape:$resOp),
     (ValuesHaveSameDType $resOp, $input)]>;

def SqueezeV11ofConst : NamedPat<"SqueezeV11ofConst",
    // From Squeeze (c, axis)
    (ONNXSqueezeV11Op:$resOp $input, $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (HasStaticShape:$resOp),
     (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Slice operations.
//===----------------------------------------------------------------------===//

def SliceofConst : NamedPat<"SliceofConst",
    // From Slice (x, starts, ends, axes, steps)
    (ONNXSliceOp:$resOp $input,
                        $starts, $ends, $axes, $steps),
    // To c' where c' is the sliced value.
    (CreateSliceOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (IsFromDenseConstLike:$starts),
     (IsFromDenseConstLike:$ends), (IsFromDenseConstLikeOrNone:$axes),
     (IsFromDenseConstLikeOrNone:$steps), (HasStaticShape:$resOp),
     (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns for Pad.
//===----------------------------------------------------------------------===//

// TODO: Support axes. For now it's required to be none.
def PadOfConst : NamedPat<"PadOfConst",
    // From Pad (x, pads, constant_value, axes)
    (ONNXPadOp:$resOp $input,
                      $pads, $constant_value, $axes, $mode),
    // To c where c is the padded value.
    (CreatePadOfConst $resOp, $input, $constant_value),
    [(IsFromDenseConstLike:$input),
     (IsFromDenseConstLike:$pads),
     (IsFromDenseConstLikeOrNone:$constant_value),
     (IsNoneType:$axes), (IsIntOrFloatType:$input),
     (EqualString<"constant"> $mode),
     (HasStaticShape:$resOp),
     (SatisfiesExpansionBound:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Concat operations.
//===----------------------------------------------------------------------===//

def ConcatofConst : NamedPat<"ConcatofConst",
  (ONNXConcatOp:$resOp $input, $axis),
  (CreateConcatOfConst $resOp, $input, $axis),
  [(IsVariadicOperandDenseONNXConstantOp:$input), (IsRankedShapedType:$resOp),
   (IsIntOrFloatType:$resOp)]
>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Expand operations.
//===----------------------------------------------------------------------===//

def ExpandofConst : NamedPat<"ExpandofConst",
    // From Expand (x, shape)
    (ONNXExpandOp:$resOp $input,
                         $shape),
    // To c where c is the expanded value.
    (CreateExpandOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (IsFromDenseConstLike:$shape),
     (HasStaticShape:$resOp), (ValuesHaveSameDType $resOp, $input),
     (SatisfiesExpansionBound:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Gather operations.
//===----------------------------------------------------------------------===//

def GatherofConst : NamedPat<"GatherofConst",
    // From Gather (x, indices, axis)
    (ONNXGatherOp:$resOp $input,
                         $indices,
                         $_),
    // To c' where c' is the gathered value.
    (CreateGatherOfConst $resOp, $input, $indices),
    [(IsFromDenseConstLike:$input), (IsFromDenseConstLike:$indices),
     (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Reshape operations.
//===----------------------------------------------------------------------===//

def ReshapeofConst : NamedPat<"ReshapeofConst",
    // From Reshape (x, shape)
    (ONNXReshapeOp:$resOp $input,
                          $shape,
                          $_),
    // To c where c is the reshaped value.
    (CreateReshapeOfConst $resOp, $input),
    [(IsFromDenseConstLike:$input), (IsFromDenseConstLike:$shape),
     (HasStaticShape:$resOp), (ValuesHaveSameDType $resOp, $input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with ConstantOfShape operations.
//===----------------------------------------------------------------------===//

def ConstantOfShapeofConst : NamedPat<"ConstantOfShapeofConst",
    // From ConstantOfShape (shape, x)
    (ONNXConstantOfShapeOp:$resOp $shape, $value),
    // To c where c is the expanded value.
    (CreateConstantOfShapeOfConst $resOp, $shape, $value),
    [(IsFromDenseConstLike:$shape),
     (HasStaticShape:$resOp), (IsIntOrFloatType:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Range.
//===----------------------------------------------------------------------===//

// Constant Propagation for Range if shape inference has calculated the length.
def RangeConstProp : NamedPat<"RangeConstProp",
    // From range(c0, c1, c2).
    (ONNXRangeOp:$rangeOp $start,
                      $limit,
                      $delta),
    // To [c0, c0 + c2, c0 + c2 * 2, ...] assuming shape inference calculated
    // dim size max(ceil((c1 - c0) / c2), 0).
    (CreateRangeOfThreeConst $rangeOp, $start, $limit, $delta),
    [(IsFromDenseConstLike:$start), (IsFromDenseConstLike:$limit),
     (IsFromDenseConstLike:$delta), (HasStaticShape:$rangeOp),
     (IsIntOrFloatType:$start), (IsIntOrFloatType:$limit),
     (IsIntOrFloatType:$delta)]>;

//===----------------------------------------------------------------------===//
// Patterns for NonZero.
//===----------------------------------------------------------------------===//

def NonZeroOfConst : NamedPat<"NonZeroOfConst",
    (ONNXNonZeroOp:$nonZeroOp $input),
    (CreateNonZeroOfConst $nonZeroOp, $input),
    [(IsFromDenseConstLike:$input), (IsIntOrFloatType:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns for ScatterND.
//===----------------------------------------------------------------------===//

def ScatterNDOfConst : NamedPat<"ScatterNDOfConst",
    (ONNXScatterNDOp:$scatterNDOp
      $data,
      $indices,
      $updates,
      $reduction),
    (CreateScatterNDOfConst $scatterNDOp, $data, $indices, $updates),
    [(IsFromDenseConstLike:$data),
     (IsFromDenseConstLike:$indices),
     (IsFromDenseConstLike:$updates),
     (IsIntOrFloatType:$data), (IsIntOrFloatType:$updates),
     (EqualString<"none"> $reduction)]>;

//===----------------------------------------------------------------------===//
// Constant Propagation for Pow
//===----------------------------------------------------------------------===//

def PowConstProp : NamedPat<"PowConstProp",
    // From pow(c1, c2).
    (ONNXPowOp:$powOp $lhs,
                      $rhs),
    // To c1^c2
    (CreatePowOfTwoConst $powOp, $lhs, $rhs),
    // Power constraints
    [(IsFromDenseConstLike:$lhs), (IsFromDenseConstLike:$rhs),
     (IsIntOrFloatType:$lhs), (IsIntOrFloatType:$rhs),
     (SatisfiesExpansionBound:$powOp), (ValuesHaveSameDType $lhs, $rhs),
     (HasStaticShape:$powOp)]>;

#endif // ONNX_CONSTPROP
