// SPDX-License-Identifier: Apache-2.0

//===- ONNXConstProp.td - Rewriting for Constant Propagation in ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
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

def IsNotAConstant :
  Constraint<CPred<"! isa_and_nonnull<ONNXConstantOp>(($_self).getDefiningOp())">,
  "operation is not a constant">;

def IsFromDenseONNXConstantOp:
    Constraint<CPred<"isDenseONNXConstant($_self)">,
  "Value is produced by a dense ONNXConstantOp that can be linked with "
  "a buffer during constant propagation">;

def IsFromDenseONNXConstantOpOrNone:
  Constraint<
    CPred<"isDenseONNXConstant($_self) || ($_self.getType().isa<NoneType>())">,
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

// Creation helpers:

def CreateZeroTensorOfType: NativeCodeCall<
  "ConstZeroTensor($_builder, $_loc, $0.getType().cast<mlir::ShapedType>())"
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

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise ADD operations.
//===----------------------------------------------------------------------===//
  
// Use commutativity to normalize constants in the second position of Add.
def AddConstCommutative1 : Pat<
  // From add(c, x).
  (ONNXAddOp (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_), $x),
  // To add(x, c).
  (ONNXAddOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;
  
// Use associativity to add constants together.
def AddConstAssociative1 : Pat<
  // From add(add(x, c1), c2).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_)),
  // To add(x, add(c1, c2)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $c1, $c2)),
    [(IsNotAConstant:$x)]>;

def AddConstAssociative2 : Pat<
  // From add(add(x, c), y).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
    $y),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def AddConstAssociative3 : Pat<
  // From add(x, add(y, c)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $y,(ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_))),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def AddConstAssociative4 : Pat<
  // From add(add(x, c1), add(y, c2)).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXAddOp $y,(ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_))),
  // To add(add(x, y), c1+c2).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    (ONNXAddOp $c1, $c2)),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;
  
// Constant Propagation for Add
def AddConstProp : Pat<
    // From add(c1, c2).
    (ONNXAddOp:$addOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1+c2
    (CreateAddOfTwoConst $addOp, $lhs, $rhs),
    // Additional constraints (dense)
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$addOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def AddZerosOnRhs : Pat<
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
def SubConstProp : Pat<
    // From sub(c1, c2).
    (ONNXSubOp:$subOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1-c2
    (CreateSubOfTwoConst $subOp, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$subOp)]>;

// TODO: Expand $a to $result's shape instead of requiring ValuesHaveSameType.
def SubZerosOnRhs : Pat<
    // From sub(x, c).
    (ONNXSubOp:$result
      $x,
      (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    // To x.
    (replaceWithValue $x),
    [(IsNotAConstant:$x), (ValuesHaveSameType $result, $x), (IsConstOfZeros:$c)]>;

// Cast of constant is simply a constant with the new type.
def CastofConst :  Pat<
    // From (c)
    (ONNXCastOp:$castOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $saturate, $to),
    // To (c)
    (CreateCastOfConst $castOp, $input, $saturate, $to),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Neg of constant is simply -const
def NegofConst :  Pat<
    // From - (c)
    (ONNXNegOp:$negOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To (-c)
    (CreateNegOfConst $negOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;
  
// Change a subtraction of a constant c by an addition of -c. Helpfull to combine
// with other add optimizations.
def SubConstToNeg : Pat<
    // From x - c.
    (ONNXSubOp:$subOp $x, (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To x + (-c).
    (ONNXAddOp $x, (CreateNegOfConst $input, $input)),
    [(IsNotAConstant:$x), (IsFromDenseONNXConstantOp:$input)]>;

// Constant Propagation for Sqrt
def SqrtofConst :  Pat<
    // From  onnx.Sqrt(c)
    (ONNXSqrtOp:$sqrtOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To sqrt(c)
    (CreateSqrtOfConst $sqrtOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

// Constant Propagation for Relu
def ReluofConst :  Pat<
    // From  onnx.Relu(c)
    (ONNXReluOp:$reluOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    // To relu(c)
    (CreateReluOfConst $reluOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;
 
//===----------------------------------------------------------------------===//
// Const propagation patterns for variadic elementwise operations.
//===----------------------------------------------------------------------===//

// Constant Propagation for Max
def MaxConstProp : Pat<
    // From max(c1, c2).
    (ONNXMaxOp:$maxOp $operandList),
    // To c = max(c1, c2)
    (CreateMaxOfConst $maxOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (SatisfiesExpansionBound:$maxOp)]>;

// Constant Propagation for Min
def MinConstProp : Pat<
    // From min(c1, c2).
    (ONNXMinOp:$minOp $operandList),
    // To c = min(c1, c2)
    (CreateMinOfConst $minOp, $operandList),
    // Constraints
    [(IsVariadicOperandDenseONNXConstantOp:$operandList),
     (SatisfiesExpansionBound:$minOp)]>;

// Constant Propagation for Sum
def SumConstProp : Pat<
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
def MulConstCommutative1 : Pat<
  // From mul(c, x).
  (ONNXMulOp (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_), $x),
  // To mul(x, c).
  (ONNXMulOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;
  
// Use associativity to mul constants together.
def MulConstAssociative1 : Pat<
  // From mul(mul(x, c1), c2).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_)),
  // To mul(x, mul(c1, c2)).
  (ONNXMulOp
    $x,
    (ONNXMulOp $c1, $c2)),
    [(IsNotAConstant:$x)]>;
  
def MulConstAssociative2 : Pat<
  // From mul(mul(x, c), y).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
    $y),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def MulConstAssociative3 : Pat<
  // From mul(x, mul(y, c)).
  (ONNXMulOp
    $x,
    (ONNXMulOp $y,(ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_))),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def MulConstAssociative4 : Pat<
  // From mul(mul(x, c1), mul(y, c2)).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_)),
    (ONNXMulOp $y,(ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_))),
  // To mul(mul(x, y), c1+c2).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    (ONNXMulOp $c1, $c2)),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

// Constant Propagation for Mul
def MulConstProp : Pat<
    // From mul(c1, c2).
    (ONNXMulOp:$mulOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1+c2
    (CreateMulOfTwoConst $mulOp, $lhs, $rhs),
    // Multiplication constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$mulOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def MulOnesOnRhs : Pat<
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
def DivConstProp : Pat<
    // From div(c1, c2).
    (ONNXDivOp:$divOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1/c2
    (CreateDivOfTwoConst $divOp, $lhs, $rhs),
    // Division constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$divOp)]>;

// TODO: Expand $x to $result's shape instead of requiring ValuesHaveSameType.
def DivOnesOnRhs : Pat<
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
// Constant propagate ONNXEqualOp
//===----------------------------------------------------------------------===//

def EqualConstProp : Pat<
    // From equal(c1, c2).
    (ONNXEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1 == c2
    (CreateEqualOfTwoConst $result, $lhs, $rhs),
    // constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagate ONNXLessOp
//===----------------------------------------------------------------------===//

def LessConstPropPattern: Pat<
    (ONNXLessOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateLessOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagate ONNXGreaterOp
//===----------------------------------------------------------------------===//

def GreaterConstPropPattern: Pat<
    (ONNXGreaterOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateGreaterOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagate ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//

def LessOrEqualConstPropPattern: Pat<
    (ONNXLessOrEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateLessOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Constant propagate ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//

def GreaterOrEqualConstPropPattern: Pat<
    (ONNXGreaterOrEqualOp:$result
      (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateGreaterOrEqualOfTwoConst $result, $lhs, $rhs),
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs),
     (SatisfiesExpansionBound:$result)]>;

//===----------------------------------------------------------------------===//
// Patterns for Where.
//===----------------------------------------------------------------------===//

// Constant Propagation for Where
def WhereConstProp : Pat<
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

def ReduceSumConstProp: Pat<
      (ONNXReduceSumOp:$reduceSumOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceSumConst $reduceSumOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceProdConstProp: Pat<
      (ONNXReduceProdOp:$reduceProdOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceProdConst $reduceProdOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMinConstProp: Pat<
      (ONNXReduceMinOp:$reduceMinOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceMinConst $reduceMinOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMaxConstProp: Pat<
      (ONNXReduceMaxOp:$reduceMaxOp
        (ONNXConstantOp:$data $_, $_, $_, $_, $_, $_, $_, $_),
        $axes,
        $_,
        $_),
    (CreateReduceMaxConst $reduceMaxOp, $data, $axes),
    [(IsFromDenseONNXConstantOp:$data), (IsFromDenseONNXConstantOpOrNone:$axes)]
    >;

def ReduceMeanConstProp: Pat<
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
def MatMulZerosOnLhs : Pat<
    (ONNXMatMulOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      $B
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsNotAConstant:$B), (HasIntegerElementType:$resOp),
     (HasStaticShape:$resOp), (IsConstOfZeros:$A)]
    >;

// Limited to integers because NaN * 0 is NaN, not 0.
def MatMulZerosOnRhs : Pat<
    (ONNXMatMulOp:$resOp
      $A,
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsNotAConstant:$A), (HasIntegerElementType:$resOp),
     (HasStaticShape:$resOp), (IsConstOfZeros:$B)]
    >;

def MatMulOfConsts : Pat<
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

def MatMulIntegerConstZeroLhs : Pat<
    (ONNXMatMulIntegerOp:$resOp
      (ONNXConstantOp:$A $_, $_, $_, $_, $_, $_, $_, $_),
      $B,
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseONNXConstantOp:$A), (IsFromDenseONNXConstantOpOrNone:$a_zero_point),
     (IsMatMulIntegerLhsZero $A, $a_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerConstZeroRhs : Pat<
    (ONNXMatMulIntegerOp:$resOp
      $A,
      (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_),
      $a_zero_point, $b_zero_point
    ),
    (CreateZeroTensorOfType $resOp),
    [(IsFromDenseONNXConstantOp:$B), (IsFromDenseONNXConstantOpOrNone:$b_zero_point),
     (IsMatMulIntegerRhsZero $B, $b_zero_point), (HasStaticShape:$resOp)]
    >;

def MatMulIntegerOfConsts : Pat<
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
def GemmConstProp : Pat<
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

def TransposeofConst :  Pat<
    // From TransposeOp(c, p)
    (ONNXTransposeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is transposed attribute
    (CreateTransposeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Unsqueeze operations.
//===----------------------------------------------------------------------===//

def UnsqueezeofConst :  Pat<
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

def UnsqueezeV11ofConst :  Pat<
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeV11Op:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateUnsqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Squeeze operations.
//===----------------------------------------------------------------------===//

def SqueezeofConst :  Pat<
    // From Squeeze (c, axis)
    (ONNXSqueezeOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

def SqueezeV11ofConst :  Pat<
    // From Squeeze (c, axis)
    (ONNXSqueezeV11Op:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $_),
    // To c' where c' is the unsqueezed value.
    (CreateSqueezeOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Slice operations.
//===----------------------------------------------------------------------===//

def SliceofConst :  Pat<
    // From Slice (x, starts, ends, axes, steps)
    (ONNXSliceOp:$resOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_),
                        $starts, $ends, $axes, $steps),
    // To c' where c' is the sliced value.
    (CreateSliceOfConst $resOp, $input),
    [(IsFromDenseONNXConstantOp:$input), (IsFromDenseONNXConstantOp:$starts),
     (IsFromDenseONNXConstantOp:$ends), (IsFromDenseONNXConstantOpOrNone:$axes),
     (IsFromDenseONNXConstantOpOrNone:$steps), (HasStaticShape:$resOp)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Concat operations.
//===----------------------------------------------------------------------===//

def ConcatofConst: Pat<
  (ONNXConcatOp:$resOp $input, $axis),
  (CreateConcatOfConst $resOp, $input, $axis),
  [(IsVariadicOperandDenseONNXConstantOp:$input)]
>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Expand operations.
//===----------------------------------------------------------------------===//

def ExpandofConst :  Pat<
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

def GatherofConst :  Pat<
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

def ReshapeofConst :  Pat<
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

def ConstantOfShapeofConst :  Pat<
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
def RangeConstProp : Pat<
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

def NonZeroOfConst : Pat<
    (ONNXNonZeroOp:$nonZeroOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_)),
    (CreateNonZeroOfConst $nonZeroOp, $input),
    [(IsFromDenseONNXConstantOp:$input)]>;

#endif // ONNX_CONSTPROP
