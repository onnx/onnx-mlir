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
include "src/Dialect/ONNX/ONNXOps.td"
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


// Usefult code generation invokation.
def GetNullAttr : NativeCodeCall<"Attribute()">;

// Create an dense ONNXConstantOp whose output type is the 1st argument's type.
def CreateDenseONNXConstantOp:
   NativeCodeCall<"$_builder.create<ONNXConstantOp>($0.getLoc(), $0.getType(), Attribute(), $1)">;

def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateSubOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXSubOp>($_builder, $0, $1, $2)">;

def CreateNegOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;

def CreateSqrtOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXSqrtOp>($_builder, $0, $1)">;

def CreateMulOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXMulOp>($_builder, $0, $1, $2)">;

def CreateDivOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXDivOp>($_builder, $0, $1, $2)">;

def CreateTransposeOfConst :
   NativeCodeCall<"ConstPropTranspose($_builder, $0, $1, $2)">;

def CreateUnsqueezeOfConst:
   NativeCodeCall<"ConstPropUnsqueeze($_builder, $0, $1)">;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise ADD operations.
//===----------------------------------------------------------------------===//
  
// Use commutativity to normalize constants in the second position of Add.
def AddConstCommutative1 : Pat<
  // From add(c, x).
  (ONNXAddOp (ONNXConstantOp:$c $_, $_), $x),
  // To add(x, c).
  (ONNXAddOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;
  
// Use associativity to add constants together.
def AddConstAssociative1 : Pat<
  // From add(add(x, c1), c2).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c1 $_, $_)),
    (ONNXConstantOp:$c2 $_, $_)),
  // To add(x, add(c1, c2)).
  (ONNXAddOp
    $x,
    (ONNXAddOp $c1, $c2)),
    [(IsNotAConstant:$x)]>;

def AddConstAssociative2 : Pat<
  // From add(add(x, c), y).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c $_, $_)),
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
    (ONNXAddOp $y,(ONNXConstantOp:$c $_, $_))),
  // To add(add(x, y), c).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def AddConstAssociative4 : Pat<
  // From add(add(x, c1), add(y, c2)).
  (ONNXAddOp
    (ONNXAddOp $x,(ONNXConstantOp:$c1 $_, $_)),
    (ONNXAddOp $y,(ONNXConstantOp:$c2 $_, $_))),
  // To add(add(x, y), c1+c2).
  (ONNXAddOp
    (ONNXAddOp $x, $y),
    (ONNXAddOp $c1, $c2)),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;
  
// Constant Propagation for Add
def AddConstProp : Pat<
    // From add(c1, c2).
    (ONNXAddOp:$addOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1+c2
    (CreateDenseONNXConstantOp $addOp, (CreateAddOfTwoConst $addOp, $v1, $v2)),
    // Additional constraints (no sparse)
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;


//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise SUB / NEG operations.
//===----------------------------------------------------------------------===//

// Constant Propagation for Sub
def SubConstProp : Pat<
    // From sub(c1, c2).
    (ONNXSubOp:$subOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1-c2
    (CreateDenseONNXConstantOp $subOp, (CreateSubOfTwoConst $subOp, $v1, $v2)),
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;

// Neg of constant is simly -const
def NegofConst :  Pat<
    // From - (c)
    (ONNXNegOp:$negOp (ONNXConstantOp:$constOp $s, $v)),
    // To (-c)
    (CreateDenseONNXConstantOp $negOp, (CreateNegOfConst $constOp, $v)),
    [(AttributeIsNull:$s)]>;
  
// Change a subtraction of a constant c by an addition of -c. Helpfull to combine
// with other add optimizations.
def SubConstToNeg : Pat<
    // From x - c.
    (ONNXSubOp:$subOp $x, (ONNXConstantOp:$constOp $s, $v)),
    // To x + (-c).
    (ONNXAddOp $x, (CreateDenseONNXConstantOp $constOp, (CreateNegOfConst $constOp, $v))),
    [(IsNotAConstant:$x), (AttributeIsNull:$s)]>;

// Constant Propagation for Sqrt
def SqrtofConst :  Pat<
    // From  onnx.Sqrt(c)
    (ONNXSqrtOp:$sqrtOp (ONNXConstantOp:$constOp $s, $v)),
    // To sqrt(c)
    (CreateDenseONNXConstantOp $sqrtOp, (CreateSqrtOfConst $constOp, $v)),
    [(AttributeIsNull:$s)]>;
 
//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with elementwise MUL operations.
// Exactly the same pattern as for the elementwise ADD operations.
//===----------------------------------------------------------------------===//

// Use commutativity to normalize constants in the second position of Mul.
def MulConstCommutative1 : Pat<
  // From mul(c, x).
  (ONNXMulOp (ONNXConstantOp:$c $_, $_), $x),
  // To mul(x, c).
  (ONNXMulOp $x, $c),
  // To avoid infinite loop, constrain the first arguments to be anything but a constant.
  [(IsNotAConstant:$x)]>;
  
// Use associativity to mul constants together.
def MulConstAssociative1 : Pat<
  // From mul(mul(x, c1), c2).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c1 $_, $_)),
    (ONNXConstantOp:$c2 $_, $_)),
  // To mul(x, mul(c1, c2)).
  (ONNXMulOp
    $x,
    (ONNXMulOp $c1, $c2)),
    [(IsNotAConstant:$x)]>;
  
def MulConstAssociative2 : Pat<
  // From mul(mul(x, c), y).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c $_, $_)),
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
    (ONNXMulOp $y,(ONNXConstantOp:$c $_, $_))),
  // To mul(mul(x, y), c).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    $c),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

def MulConstAssociative4 : Pat<
  // From mul(mul(x, c1), mul(y, c2)).
  (ONNXMulOp
    (ONNXMulOp $x,(ONNXConstantOp:$c1 $_, $_)),
    (ONNXMulOp $y,(ONNXConstantOp:$c2 $_, $_))),
  // To mul(mul(x, y), c1+c2).
  (ONNXMulOp
    (ONNXMulOp $x, $y),
    (ONNXMulOp $c1, $c2)),
    [(IsNotAConstant:$x), (IsNotAConstant:$y)]>;

// Constant Propagation for Mul
def MulConstProp : Pat<
    // From mul(c1, c2).
    (ONNXMulOp:$mulOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1+c2
    (CreateDenseONNXConstantOp $mulOp, (CreateMulOfTwoConst $mulOp, $v1, $v2)),
    // Mulitional constraints (no sparse)
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;

// Constant Propagation for Div 
def DivConstProp : Pat<
    // From div(c1, c2).
    (ONNXDivOp:$mulOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1/c2
    (CreateDenseONNXConstantOp $mulOp, (CreateDivOfTwoConst $mulOp, $v1, $v2)),
    // Division constraints (no sparse)
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;


//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Transpose operations.
//===----------------------------------------------------------------------===//

// Neg of constant is simly -const
def TransposeofConst :  Pat<
    // From TransposeOp(c, p)
    (ONNXTransposeOp:$resOp (ONNXConstantOp $s, $v), $p),
    // To c' where c' is transposed attribute
    (CreateDenseONNXConstantOp $resOp, (CreateTransposeOfConst $resOp, $v, $p)),
    [(AttributeIsNull:$s)]>;

//===----------------------------------------------------------------------===//
// Patterns to enable opportunities with Unsqueeze operations.
//===----------------------------------------------------------------------===//

def UnsqueezeofConst :  Pat<
    // From Unsqueeze (c, axis)
    (ONNXUnsqueezeOp:$resOp (ONNXConstantOp $s, $v), $_),
    // To c' where c' is the unsqueezed value.
    (CreateDenseONNXConstantOp $resOp, (CreateUnsqueezeOfConst $resOp, $v)),
    [(AttributeIsNull:$s)]>;


#endif // ONNX_CONSTPROP
