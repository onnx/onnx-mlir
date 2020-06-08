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


// =============================================================================
// Instruction to add new constant operation rules. Minimally, you will have added
// operation in the ONNXConstProp.cpp to perform the element-wise single value
// handling of the new operator that you are dealing with. You will need to
// generate a call to the method that handle the tensor constant prop. Here
// is the call for a unary and binary operation. Adapt to your new operator:
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
// =============================================================================

  
// Useful test definitions.

def IsNotAConstant :
  Constraint<CPred<"! dyn_cast_or_null<ONNXConstantOp>(($_self).getDefiningOp())">,
  "operation is not a constant">;

def AttributeIsNull :
    Constraint<CPred<"! ($_self)">,
  "Attribute is null">;


// Usefult code generation invokation.
def GetNullAttr : NativeCodeCall<"Attribute()">;
  
def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;

def CreateSubOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXSubOp>($_builder, $0, $1, $2)">;

def CreateNegOfConst :
   NativeCodeCall<"ConstPropElementwiseUnary<mlir::ONNXNegOp>($_builder, $0, $1)">;

 def CreateMulOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXMulOp>($_builder, $0, $1, $2)">;

// =============================================================================
// Patterns to enable opportunities with elementwise ADD operations.
  
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
    (ONNXAddOp $c1, $c2))>;

// Constant Propagation for Add
def AddConstProp : Pat<
    // From add(c1, c2).
    (ONNXAddOp:$addOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1+c2
    (ONNXConstantOp (GetNullAttr), (CreateAddOfTwoConst $addOp, $v1, $v2)),
    // Additional constraints (no sparse)
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;


// =============================================================================
// Patterns to enable opportunities with elementwise SUB / NEG operations.

// Constant Propagation for Sub
def SubConstProp : Pat<
    // From sub(c1, c2).
    (ONNXSubOp:$subOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1-c2
    (ONNXConstantOp (GetNullAttr), (CreateSubOfTwoConst $subOp, $v1, $v2)),
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;

// Neg of constant is simly -const
def NegofConst :  Pat<
    // From - (c)
    (ONNXNegOp (ONNXConstantOp:$constOp $s, $v)),
    // To (-c)
    (ONNXConstantOp (GetNullAttr), (CreateNegOfConst $constOp, $v)),
    [(AttributeIsNull:$s)]>;
  
// Change a subtraction of a constant c by an addition of -c. Helpfull to combine
// with other add optimizations.
def SubConstToNeg : Pat<
    // From x - c.
    (ONNXSubOp:$subOp $x, (ONNXConstantOp:$constOp $s, $v)),
    // To x + (-c).
    (ONNXAddOp $x, (ONNXConstantOp (GetNullAttr), (CreateNegOfConst $constOp, $v))),
    [(IsNotAConstant:$x), (AttributeIsNull:$s)]>;


// =============================================================================
// Patterns to enable opportunities with elementwise MUL operations.
// Exactly the same pattern as for the elementwise ADD operations.
  
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
    (ONNXMulOp $c1, $c2))>;

// Constant Propagation for Mul
def MulConstProp : Pat<
    // From mul(c1, c2).
    (ONNXMulOp:$mulOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    // To c1+c2
    (ONNXConstantOp (GetNullAttr), (CreateMulOfTwoConst $mulOp, $v1, $v2)),
    // Mulitional constraints (no sparse)
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;

#endif // ONNX_CONSTPROP
