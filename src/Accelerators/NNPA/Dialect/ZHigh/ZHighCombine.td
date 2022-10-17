// SPDX-License-Identifier: Apache-2.0

//===------- ZHighCombine.td - Pattern Match for ZHighDialect -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_COMBINE
#define ZHIGH_COMBINE

#ifndef OP_BASE
include "ZHigh.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def SameLayout: Constraint<
  CPred<"($0 && $1 &&"
        " (::onnx_mlir::zhigh::getZTensorLayout($0.getType()) == "
        "::onnx_mlir::zhigh::getZTensorLayout($1.getType())))">,
  "Two ztensors have the same layout"
>;

def NotSameLayout: Constraint<
  CPred<"($0 && $1 &&"
        " (::onnx_mlir::zhigh::getZTensorLayout($0.getType()) != "
        "::onnx_mlir::zhigh::getZTensorLayout($1.getType())))">,
  "Two ztensors have different layouts"
>;

def IsNoneType : Constraint<CPred<"(($_self).getType().isa<NoneType>())">>;

def GetLayout : NativeCodeCall<
  "convertDataLayoutToStringAttr($_builder, "
      "::onnx_mlir::zhigh::getZTensorLayout($0.getType()))"
>;

def GetMinusBcastConst : NativeCodeCall<
  "::onnx_mlir::zhigh::getMinusBcastConst($_builder, $_loc, $0, $1)"
>;

def IsStaticShapeTensor:
  Constraint<
    CPred<
      "$0.getType().cast<::mlir::ShapedType>().hasStaticShape()">,
    "hasStaticShape">;

def IsPlusConstantFloat : Constraint<
  CPred<"($0.getValueAsDouble() > 0.0)">,
  "Is the float attribute is greater than 0"
>;

//===----------------------------------------------------------------------===//
// Transformations
//===----------------------------------------------------------------------===//

// zhigh.Stick (None) = None 
def NoneTypeStickRemovalPattern : Pat<
  (ZHighStickOp:$stick $arg, $layout1),
  (replaceWithValue $arg),
  [(IsNoneType:$arg)]
>;

// zhigh.Unstick (zhigh.Stick (%X)) = %X
def UnstickStickRemovalPattern : Pat<
  (ZHighUnstickOp (ZHighStickOp $arg, $_)),
  (replaceWithValue $arg)
>;

// zhigh.Stick (zhigh.Unstick (%X)) = %X
def StickUnstickRemovalPattern : Pat<
  (ZHighStickOp:$stick (ZHighUnstickOp:$unstick $arg), $_),
  (replaceWithValue $arg),
  [(SameLayout $arg, $stick)]
>;

def ProducedByUnstickOp: Constraint<
  CPred<"areProducedByUnstickOp($_builder, $0, $1)">,
  "all values are produced by ZHighUnstickOp with the given layout"
>;

def HaveNoPadsWhenStickified: Constraint<
  CPred<"haveNoPadsWhenStickified($0, $1, $2)">,
  "there are no pads along the given axis when stickifying values by using the given layout"
>;

def GetStickifiedInputs: NativeCodeCall<
  "getStickifiedInputs($_builder, $_loc, $0)"
>;

def GetStickifiedConcatAxis: NativeCodeCall<
  "getStickifiedConcatAxis($_builder, $0, $1)"
>;

// The pattern
//   zhigh.Stick (onnx.Concat (zhigh.Unstick (%X1), zhigh.Unstick (%X2)) { axis })
// can be replaced by
//   zhigh.Concat (%X1, %X2) { axis }
// where `zhigh.Concat` is an operation that concatenates stickified tensors.
//
// Constraints:
//   - The stickified tensors does not have pads on the concatenation axis.
//
// `zhigh.Concat` is used instead of `onnx.Concat` because `onnx.Concat` cannot 
// allocate a 4K-aligned buffer for the output stickified tensor.
//
// Note: Support 4D/NHWC layout only at this moment.
def ReplaceONNXConcatByZHighConcatPattern: Pat<
  (ZHighStickOp (ONNXConcatOp $inputs, $axis), $layout),
  (ZHighConcatOp (GetStickifiedInputs $inputs), (GetStickifiedConcatAxis $layout, $axis)),
  [(ProducedByUnstickOp $inputs, $layout),
   (HaveNoPadsWhenStickified $inputs, $layout, $axis)]
>;

// Since sigmoid is an element-wise op and does not depend on layout, we change
// the sigmoid's layout to the one from the previous unstick, which enables
// unstick/stick removal.
// TODO: generalize this rule to the other element-wise ops, which requires
// more analyses to decide whether changing layout is the best for the whole
// model or not.
def SigmoidLayoutChangingPattern : Pat<
  // From different layouts.
  (ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp:$s_x (ZHighUnstickOp $x),
                                                $_))),
  // To the same layouts.
  (ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp:$new_s_x (ZHighUnstickOp $x),
                                                         (GetLayout $x)),
                                  (returnType $new_s_x))),
  [(NotSameLayout $x, $s_x)]
>;

// The pattern
//   zhigh.Stick (onnx.LeakyRelu (zhigh.Unstick (%X)))
// can be replaced by
//   zhigh.Sub (zhigh.Relu(%X),
//              zhigh.Relu(zhigh.Mul(%X, MinusBcastConst(%alpha))))
//
// Constraints:
//   - %X should have static shape, and %alpha should be constant.
//
def ReplaceONNXLeakyReluPattern: Pat<
  (ZHighStickOp:$stickout (ONNXLeakyReluOp:$out (ZHighUnstickOp $X), $alpha),
                $layout),
  (ZHighSubOp
     (ZHighReluOp $X, (returnType $X)),
     (ZHighReluOp (ZHighMulOp $X,
                              (ZHighStickOp (GetMinusBcastConst $alpha,
                                             $out),
                                            $layout),
                              (returnType $X)),
                  (returnType $X))),
  [(IsStaticShapeTensor $X), (IsPlusConstantFloat $alpha),
   (SameLayout $X, $stickout)]
>;

#endif // ZHIGH_COMBINE
