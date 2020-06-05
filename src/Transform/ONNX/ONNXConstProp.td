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

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

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
  NativeCodeCall<"ConstPropForSubOfTwoConst($_builder, $0, $1, $2)">;


// Patterns to enable opportunities with ADD operations.
  
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

// Constant Propagation for Sub
def SubConstProp : Pat<
    (ONNXSubOp:$subOp (ONNXConstantOp $s1, $v1), (ONNXConstantOp $s2, $v2)),
    (ONNXConstantOp (GetNullAttr), (CreateSubOfTwoConst $subOp, $v1, $v2)),
    [(AttributeIsNull:$s1), (AttributeIsNull:$s2)]>;



#endif // ONNX_CONSTPROP
