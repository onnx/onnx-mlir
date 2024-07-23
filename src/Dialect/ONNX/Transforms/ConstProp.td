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
  Constraint<CPred<"! isa_and_nonnull<ONNXConstantOp>(($_self).getDefiningOp())">,
  "operation is not a constant">;

def IsFromDenseONNXConstantOp:
    Constraint<CPred<"isDenseONNXConstant($_self)">,
  "Value is produced by a dense ONNXConstantOp that can be linked with "
  "a buffer during constant propagation">;

def IsFromDenseONNXConstantOpOrNone:
  Constraint<
    CPred<"isDenseONNXConstant($_self) || mlir::isa<NoneType>($_self.getType())">,
    "Value is none or produced by a true dense ONNXConstantOp"
  >;

def IsVariadicOperandDenseONNXConstantOp:
    Constraint<CPred<"isVariadicOperandFromDenseONNXConstantOp($_self)">,
  "Variadic operand is produced by dense ONNXConstantOps">;

def HasStaticShape: Constraint<CPred<
  "hasStaticShape($_self.getType())">,
  "A value has static shape"
>;

def HasIntegerElementType: Constraint<CPred<
  "isa<IntegerType>(getElementType($_self.getType()))">,
  "A value has integer element type"
>;

def IsConstOfZeros : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 0.0)">,
  "Value is an all-zeros constant tensor">;

def IsConstOfOnes : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 1.0)">,
  "Value is an all-ones constant tensor">;

def ValuesHaveSameType : Constraint<
  CPred<"$0.getType() == $1.getType()">,
  "Values have same type">;

def IsMatMulIntegerLhsZero: Constraint<
    CPred<"isMatMulIntegerLhsZero($0, $1)">,
    "MatMulInteger lhs matrix is zero for given zero point">;

def IsMatMulIntegerRhsZero: Constraint<
    CPred<"isMatMulIntegerRhsZero($0, $1)">,
    "MatMulInteger rhs matrix is zero for given zero point">;

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

def CreateZeroTensorOfType: NativeCodeCall<
  "ConstZeroTensor($_builder, $_loc, mlir::cast<mlir::ShapedType>($0.getType()))"
>;

def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateSubOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXSubOp>($_builder, $0, $1, $2)">;

def CreateCastOfConst :
   NativeCodeCall<"ConstPropCast($_builder, $0, $1, $2, $3)">;

def CreateNegOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;

def CreateSqrtOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSqrtOp>($_builder, $0, $1)">;

def CreateReluOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXReluOp>($_builder, $0, $1)">;

def CreateReciprocalOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXReciprocalOp>($_builder, $0, $1)">;

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
  (ONNXAddOp (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_), $x),
  // To add(x, c).
  (ONNXAddOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;

// Use associativity to add constants together.
def AddConstAssociative1 : NamedPat<"AddConstAssociative1",
  // From add(add(x, c1), c2).
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_)),
  // To add(x, add(c1, c2)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $c1, $c2)),
  [(IsNotAConstant:$x), (HasOneUse:$lhs)]>;

def AddConstAssociative2 : NamedPat<"AddConstAssociative2",
  // From add(add(x, c), y).
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
    $y),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs)]>;

def AddConstAssociative3 : NamedPat<"AddConstAssociative3",
  // From add(x, add(y, c)).
  (ONNXAddOp
    $x,
    (ONNXAddOp:$rhs $y, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_))),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$rhs)]>;

def AddConstAssociative4 : NamedPat<"AddConstAssociative4",
  // From add(add(x, c1), add(y, c2)).
  (ONNXAddOp
    (ONNXAddOp:$lhs $x, (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXAddOp:$rhs $y, (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_))),
  // To add(add(x, y), c1+c2).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    (ONNXAddOp $c1, $c2)),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

// Constant Propagation for Add
def AddConstProp : NamedPat<"AddConstProp",
    // From add(c1, c2).
    (ONNXAddOp:$addOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1+c2
    (CreateAddOfTwoConst $addOp, $lhs, $rhs),
    // Additional constraints (dense)
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$addOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def AddZerosOnRhs : NamedPat<"AddZerosOnRhs",
    // From add(x, c).
    (ONNXAddOp:$result
      $x,
      (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)
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
    (ONNXSubOp:$subOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1-c2
    (CreateSubOfTwoConst $subOp, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$subOp)]>;

// TODO: Expand $a to $result's shape instead of requiring ValuesHaveSameType.
def SubZerosOnRhs : NamedPat<"SubZerosOnRhs",
    // From sub(x, c).
    (ONNXSubOp:$result
      $x,
      (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    // To x.
    (replaceWithValue $x),
    [(IsNotAConstant:$x), (ValuesHaveSameType $result, $x), (IsConstOfZeros:$c)]>;

// Cast of constant is simply a constant with the new type.
def CastofConst : NamedPat<"CastofConst",
    // From (c)
    (ONNXCastOp:$castOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $saturate, $to),
    // To (c)
    (CreateCastOfConst $castOp, $input, $saturate, $to),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Neg of constant is simply -const
def NegofConst : NamedPat<"NegofConst",
    // From - (c)
    (ONNXNegOp:$negOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To (-c)
    (CreateNegOfConst $negOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Change a subtraction of a constant c by an addition of -c. Helpfull to combine
// with other add optimizations.
def SubConstToNeg : NamedPat<"SubConstToNeg",
    // From x - c.
    (ONNXSubOp:$subOp $x, (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To x + (-c).
    (ONNXAddOp $x, (CreateNegOfConst $input, $input)),
    [(IsNotAConstant:$x), (IsFromDenseONNXConstantOp:$input)]>;

// Constant Propagation for Sqrt
def SqrtofConst : NamedPat<"SqrtofConst",
    // From  onnx.Sqrt(c)
    (ONNXSqrtOp:$sqrtOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To sqrt(c)
    (CreateSqrtOfConst $sqrtOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Constant Propagation for Relu
def ReluofConst : NamedPat<"ReluofConst",
    // From  onnx.Relu(c)
    (ONNXReluOp:$reluOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To relu(c)
    (CreateReluOfConst $reluOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Constant Propagation for Reciprocal
def ReciprocalOfConst : NamedPat<"ReciprocalOfConst",
    // From  onnx.Reciprocal(c)
    (ONNXReciprocalOp:$reciprocalOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To reciprocal(c)
    (CreateReciprocalOfConst $reciprocalOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

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
     (SatisfiesExpansionBound:$maxOp)]>;

// Constant Propagation for Min
def MinConstProp : NamedPat<"MinConstProp",
    // From min(c1, c2).
    (ONNXMinOp:$minOp $operandList),
    // To c = min(c1, c2)
    (CreateMinOfConst $minOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (SatisfiesExpansionBound:$minOp)]>;

// Constant Propagation for Sum
def SumConstProp : NamedPat<"SumConstProp",
    // From sum(c1, c2).
    (ONNXSumOp:$sumOp $operandList),
    // To c = sum(c1, c2)
    (CreateSumOfConst $sumOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (SatisfiesExpansionBound:$sumOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise MUL operations.
// Exactly the same pattern as for the elementwise ADD operations.
//===----------------------------------------------------------------------===//

// Use commutativity to normalize constants in the second position of Mul.
def MulConstCommutative1 : NamedPat<"MulConstCommutative1",
  // From mul(c, x).
  (ONNXMulOp (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_), $x),
  // To mul(x, c).
  (ONNXMulOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;

// Use associativity to mul constants together.
def MulConstAssociative1 : NamedPat<"MulConstAssociative1",
  // From mul(mul(x, c1), c2).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_)),
  // To mul(x, mul(c1, c2)).
  (ONNXMulOp
    $x,
    (ONNXMulOp $c1, $c2)),
  [(IsNotAConstant:$x), (HasOneUse:$lhs)]>;

def MulConstAssociative2 : NamedPat<"MulConstAssociative2",
  // From mul(mul(x, c), y).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
    $y),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs)]>;

def MulConstAssociative3 : NamedPat<"MulConstAssociative3",
  // From mul(x, mul(y, c)).
  (ONNXMulOp
    $x,
    (ONNXMulOp:$rhs $y, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_))),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$rhs)]>;

def MulConstAssociative4 : NamedPat<"MulConstAssociative4",
  // From mul(mul(x, c1), mul(y, c2)).
  (ONNXMulOp
    (ONNXMulOp:$lhs $x, (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXMulOp:$rhs $y, (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_))),
  // To mul(mul(x, y), c1+c2).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    (ONNXMulOp $c1, $c2)),
  [(IsNotAConstant:$x), (IsNotAConstant:$y), (HasOneUse:$lhs), (HasOneUse:$rhs)]>;

// Constant Propagation for Mul
def MulConstProp : NamedPat<"MulConstProp",
    // From mul(c1, c2).
    (ONNXMulOp:$mulOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1+c2
    (CreateMulOfTwoConst $mulOp, $lhs, $rhs),
    // Multiplication constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$mulOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def MulOnesOnRhs : NamedPat<"MulOnesOnRhs",
    // From mul(x, c).
    (ONNXMulOp:$result
      $x,
      (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)
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
    (ONNXDivOp:$divOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1/c2
    (CreateDivOfTwoConst $divOp, $lhs, $rhs),
    // Division constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$divOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def DivOnesOnRhs : NamedPat<"DivOnesOnRhs",
    // From div(x, c).
    (ONNXDivOp:$result
      $x,
      (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    // To x.
    (replaceWithValue $x),
    [
      (IsNotAConstant:$x), (IsConstOfOnes:$c),
      (ValuesHaveSameType $result, $x)
    ]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXEqualOp
//===----------------------------------------------------------------------===//

def EqualConstProp : NamedPat<"EqualConstProp",
    // From equal(c1, c2).
    (ONNXEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1 == c2
    (CreateEqualOfTwoConst $result, $lhs, $rhs),
    // constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (IsIntOrFloatType:$lhs), (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXLessOp
//===----------------------------------------------------------------------===//

def LessConstPropPattern : NamedPat<"LessConstPropPattern",
    (ONNXLessOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateLessOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXGreaterOp
//===----------------------------------------------------------------------===//

def GreaterConstPropPattern : NamedPat<"GreaterConstPropPattern",
    (ONNXGreaterOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateGreaterOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//

def LessOrEqualConstPropPattern : NamedPat<"LessOrEqualConstPropPattern",
    (ONNXLessOrEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateLessOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagation for ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//

def GreaterOrEqualConstPropPattern : NamedPat<"GreaterOrEqualConstPropPattern",
    (ONNXGreaterOrEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateGreaterOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
//  Constant propagation for ONNXModOp
//===----------------------------------------------------------------------===//

def ModConstPropPattern : NamedPat<"ModConstPropPattern",
    (ONNXModOp:$modOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_),
      $fmod),
    (CreateModOfTwoConst  $modOp, $A, $B),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOp:$B),
    (SatisfiesExpansionBound:$modOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Where.
//===----------------------------------------------------------------------===//

// Constant Propagation for Where
def WhereConstProp : NamedPat<"WhereConstProp",
    // From where(c0, c1, c2).
    (ONNXWhereOp:$whereOp (ONNXConstantOp:$condition $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$X $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$Y $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c0?c1:c2
    (CreateWhereOfThreeConst $whereOp, $condition, $X, $Y),
    // Where constraints
    [(IsFromDenseONNXConstantOp:$condition),
     (IsFromDenseONNXConstantOp:$X), (IsFromDenseONNXConstantOp:$Y),
     (SatisfiesExpansionBound:$whereOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Reduce ops.
//===----------------------------------------------------------------------===//

def ReduceSumConstProp : NamedPat<"ReduceSumConstProp",
      (ONNXReduceSumOp:$reduceSumOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceSumConst $reduceSumOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceProdConstProp : NamedPat<"ReduceProdConstProp",
      (ONNXReduceProdOp:$reduceProdOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceProdConst $reduceProdOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMinConstProp : NamedPat<"ReduceMinConstProp",
      (ONNXReduceMinOp:$reduceMinOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceMinConst $reduceMinOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMaxConstProp : NamedPat<"ReduceMaxConstProp",
      (ONNXReduceMaxOp:$reduceMaxOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceMaxConst $reduceMaxOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMeanConstProp : NamedPat<"ReduceMeanConstProp",
      (ONNXReduceMeanOp:$reduceMeanOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceMeanConst $reduceMeanOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for MatMul.
//===----------------------------------------------------------------------===//

// Limited to integers because 0 * NaN is NaN, not 0.
def MatMulZerosOnLhs : NamedPat<"MatMulZerosOnLhs",
    (ONNXMatMulOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
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
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsNotAConstant:$A), (HasIntegerElementType:$resOp),
     (HasStaticShape:$resOp), (IsConstOfZeros:$B)]
    >;

def MatMulOfConsts : NamedPat<"MatMulOfConsts",
    (ONNXMatMulOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (CreateMatMulOfConsts $resOp, $A, $B),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOp:$B),
     (SatisfiesExpansionBound:$resOp)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for MatMulInteger.
//===----------------------------------------------------------------------===//

def MatMulIntegerConstZeroLhs : NamedPat<"MatMulIntegerConstZeroLhs",
    (ONNXMatMulIntegerOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      $B,
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOpOrNone:$a_zero_point),
     (IsMatMulIntegerLhsZero $A, $a_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerConstZeroRhs : NamedPat<"MatMulIntegerConstZeroRhs",
    (ONNXMatMulIntegerOp:$resOp
      $A,
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_),
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseONNXConstantOp:$B), (IsFromDenseONNXConstantOpOrNone:$b_zero_point),
     (IsMatMulIntegerRhsZero $B, $b_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerOfConsts : NamedPat<"MatMulIntegerOfConsts",
    (ONNXMatMulIntegerOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_),
      $a_zero_point, $b_zero_point
    ),
    (CreateMatMulIntegerOfConsts $resOp, $A, $B, $a_zero_point, $b_zero_point),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOpOrNone:$a_zero_point),
     (IsFromDenseONNXConstantOp:$B), (IsFromDenseONNXConstantOpOrNone:$b_zero_point),
     (SatisfiesExpansionBound:$resOp)]
    >;

//===----------------------------------------------------------------------===//
// Patterns for Gemm.
//===----------------------------------------------------------------------===//

// Attributes are not passed to CreateGemmOfConsts. It reads them from gemmOp.
def GemmConstProp : NamedPat<"GemmConstProp",
    (ONNXGemmOp:$gemmOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_),
      $C, $alpha, $beta, $transA, $transB),
    (CreateGemmOfConsts $gemmOp, $A, $B, $C),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOp:$B),
     (IsFromDenseONNXConstantOpOrNone:$C),
     (SatisfiesExpansionBound:$gemmOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Transpose operations.
//===----------------------------------------------------------------------===//

def TransposeofConst : NamedPat<"TransposeofConst",
    // From TransposeOp(c, p)
    (ONNXTransposeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is transposed attribute
    (CreateTransposeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Unsqueeze operations.
//===----------------------------------------------------------------------===//

def UnsqueezeofConst : NamedPat<"UnsqueezeofConst",
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

def UnsqueezeV11ofConst : NamedPat<"UnsqueezeV11ofConst",
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeV11Op:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Squeeze operations.
//===----------------------------------------------------------------------===//

def SqueezeofConst : NamedPat<"SqueezeofConst",
    // From Squeeze (c, axis)
    (ONNXSqueezeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

def SqueezeV11ofConst : NamedPat<"SqueezeV11ofConst",
    // From Squeeze (c, axis)
    (ONNXSqueezeV11Op:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Slice operations.
//===----------------------------------------------------------------------===//

def SliceofConst : NamedPat<"SliceofConst",
    // From Slice (x, starts, ends, axes, steps)
    (ONNXSliceOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                        $starts, $ends, $axes, $steps),
    // To c' where c' is the sliced value.
    (CreateSliceOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input), (IsFromDenseONNXConstantOp:$starts),
     (IsFromDenseONNXConstantOp:$ends), (IsFromDenseONNXConstantOpOrNone:$axes),
     (IsFromDenseONNXConstantOpOrNone:$steps), (HasStaticShape:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Pad.
//===----------------------------------------------------------------------===//

// TODO: Support axes. For now it's required to be none.
def PadOfConst : NamedPat<"PadOfConst",
    // From Pad (x, pads, constant_value, axes)
    (ONNXPadOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                      $pads, $constant_value, $axes, $mode),
    // To c where c is the padded value.
    (CreatePadOfConst $resOp, $input, $constant_value),
    [(IsFromDenseONNXConstantOp:$pads),
     (IsFromDenseONNXConstantOpOrNone:$constant_value),
     (IsNoneType:$axes),
     (EqualString<"constant"> $mode)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Concat operations.
//===----------------------------------------------------------------------===//

def ConcatofConst : NamedPat<"ConcatofConst",
  (ONNXConcatOp:$resOp $input, $axis),
  (CreateConcatOfConst $resOp, $input, $axis),
  [(IsVariadicOperandDenseONNXConstantOp:$input)]
>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Expand operations.
//===----------------------------------------------------------------------===//

def ExpandofConst : NamedPat<"ExpandofConst",
    // From Expand (x, shape)
    (ONNXExpandOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                         (ONNXConstantOp:$shape $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c where c is the expanded value.
    (CreateExpandOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input), (IsFromDenseONNXConstantOp:$shape),
     (HasStaticShape:$resOp),
     (SatisfiesExpansionBound:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Gather operations.
//===----------------------------------------------------------------------===//

def GatherofConst : NamedPat<"GatherofConst",
    // From Gather (x, indices, axis)
    (ONNXGatherOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                         (ONNXConstantOp:$indices $_, $_, $_, $_, $_, $_, $_, $_),
                         $_),
    // To c' where c' is the gathered value.
    (CreateGatherOfConst $resOp, $input, $indices),
    [(IsFromDenseONNXConstantOp:$input), (IsFromDenseONNXConstantOp:$indices)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Reshape operations.
//===----------------------------------------------------------------------===//

def ReshapeofConst : NamedPat<"ReshapeofConst",
    // From Reshape (x, shape)
    (ONNXReshapeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                          (ONNXConstantOp:$shape $_, $_, $_, $_, $_, $_, $_, $_),
                          $_),
    // To c where c is the reshaped value.
    (CreateReshapeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input), (IsFromDenseONNXConstantOp:$shape),
     (HasStaticShape:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with ConstantOfShape operations.
//===----------------------------------------------------------------------===//

def ConstantOfShapeofConst : NamedPat<"ConstantOfShapeofConst",
    // From ConstantOfShape (shape, x)
    (ONNXConstantOfShapeOp:$resOp (ONNXConstantOp:$shape $_, $_, $_, $_, $_, $_, $_, $_), $value),
    // To c where c is the expanded value.
    (CreateConstantOfShapeOfConst $resOp, $shape, $value),
    [(IsFromDenseONNXConstantOp:$shape),
     (HasStaticShape:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for Range.
//===----------------------------------------------------------------------===//

// Constant Propagation for Range if shape inference has calculated the length.
def RangeConstProp : NamedPat<"RangeConstProp",
    // From range(c0, c1, c2).
    (ONNXRangeOp:$rangeOp (ONNXConstantOp:$start $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$limit $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$delta $_, $_, $_, $_, $_, $_, $_, $_)),
    // To [c0, c0 + c2, c0 + c2 * 2, ...] assuming shape inference calculated
    // dim size max(ceil((c1 - c0) / c2), 0).
    (CreateRangeOfThreeConst $rangeOp, $start, $limit, $delta),
    [(IsFromDenseONNXConstantOp:$start), (IsFromDenseONNXConstantOp:$limit),
     (IsFromDenseONNXConstantOp:$delta), (HasStaticShape:$rangeOp)]>;

//===----------------------------------------------------------------------===//
// Patterns for NonZero.
//===----------------------------------------------------------------------===//

def NonZeroOfConst : NamedPat<"NonZeroOfConst",
    (ONNXNonZeroOp:$nonZeroOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateNonZeroOfConst $nonZeroOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns for ScatterND.
//===----------------------------------------------------------------------===//

def ScatterNDOfConst : NamedPat<"ScatterNDOfConst",
    (ONNXScatterNDOp:$scatterNDOp
      (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$indices $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$updates $_, $_, $_, $_, $_, $_, $_, $_),
      $reduction),
    (CreateScatterNDOfConst $scatterNDOp, $data, $indices, $updates),
    [(IsFromDenseONNXConstantOp:$data),
     (IsFromDenseONNXConstantOp:$indices),
     (IsFromDenseONNXConstantOp:$updates),
     (EqualString<"none"> $reduction)]>;

//===----------------------------------------------------------------------===//
// Constant Propagation for Pow
//===----------------------------------------------------------------------===//

def PowConstProp : NamedPat<"PowConstProp",
    // From pow(c1, c2).
    (ONNXPowOp:$powOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1^c2
    (CreatePowOfTwoConst $powOp, $lhs, $rhs),
    // Power constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$powOp)]>;

#endif // ONNX_CONSTPROP
