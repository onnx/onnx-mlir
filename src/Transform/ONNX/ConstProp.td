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
//
// To construct an ONNXConstantOp in the target pattern of a TableGen record,
// use CreateDenseONNXConstantOp instead of ONNXConstantOp to handle both output
// types: RankedTensorType and MemRefType.


  
// Useful test definitions.

def IsNotAConstant :
  Constraint<CPred<"! dyn_cast_or_null<ONNXConstantOp>(($_self).getDefiningOp())">,
  "operation is not a constant">;

def AttributeIsNull :
    Constraint<CPred<"! ($_self)">,
  "Attribute is null">;

def IsFromDenseONNXConstantOp:
    Constraint<CPred<"isFromDenseONNXConstantOp($_self)">,
  "Value is produced by a dense ONNXConstantOp">;

def IsFromDenseONNXConstantOpOrNone:
  Constraint<
    CPred<"isFromDenseONNXConstantOp($_self) || ($_self.getType().isa<NoneType>())">,
    "Value is none or produced by a dense ONNXConstantOp"
  >;

def IsVariadicOperandDenseONNXConstantOp:
    Constraint<CPred<"isVariadicOperandFromDenseONNXConstantOp($_self)">,
  "Variadic operand is produced by dense ONNXConstantOps">;

def HasStaticShape: Constraint<CPred<
  "(isRankedShapedType($_self.getType()) && "
  " $_self.getType().cast<ShapedType>().hasStaticShape())">,
  "A value has static shape"
>;

// Useful code generation invocation.
def GetNullAttr : NativeCodeCall<"Attribute()">;

// Create an dense ONNXConstantOp whose output type is the 1st argument's type.
def CreateDenseONNXConstantOp:
   NativeCodeCall<"$_builder.create<ONNXConstantOp>($0.getLoc(), $0.getType(), Attribute(), $1, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr())">;

def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateSubOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXSubOp>($_builder, $0, $1, $2)">;

def CreateCastOfConst :
   NativeCodeCall<"ConstPropCast($_builder, $0, $1)">;

def CreateNegOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;

def CreateSqrtOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSqrtOp>($_builder, $0, $1)">;

def CreateReluOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXReluOp>($_builder, $0, $1)">;

def CreateMulOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXMulOp>($_builder, $0, $1, $2)">;

def CreateDivOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXDivOp>($_builder, $0, $1, $2)">;

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
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;


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
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;

// Cast of constant is simply a constant with the new type.
def CastofConst :  Pat<
    // From (c)
    (ONNXCastOp:$castOp (ONNXConstantOp:$input $_, $_, $_, $_, $_, $_, $_, $_), $to),
    // To (c)
    (CreateCastOfConst $castOp, $input),
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
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;

// Constant Propagation for Div 
def DivConstProp : Pat<
    // From div(c1, c2).
    (ONNXDivOp:$divOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // To c1/c2
    (CreateDivOfTwoConst $divOp, $lhs, $rhs),
    // Division constraints
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;


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
     (IsFromDenseONNXConstantOpOrNone:$steps)]>;

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
     (HasStaticShape:$resOp)]>;

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

#endif // ONNX_CONSTPROP
