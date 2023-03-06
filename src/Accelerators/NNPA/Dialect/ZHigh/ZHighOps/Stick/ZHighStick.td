// SPDX-License-Identifier: Apache-2.0

//===------- Stick.td - Pattern Match for ZHighStick ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef STICK_TD 
#define STICK_TD 

#ifndef OP_BASE
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// DRR patterns 
//===----------------------------------------------------------------------===//

def NoneTypeStickRemovalPattern : Pat<
  (ZHighStickOp:$stick $arg, $layout1),
  (replaceWithValue $arg),
  [(IsNoneType:$arg)]
>;

// zhigh.Stick (zhigh.Unstick (%X)) = %X
def StickUnstickSameLayoutRemovalPattern : Pat<
  (ZHighStickOp:$stick (ZHighUnstickOp:$unstick $arg), $_),
  (replaceWithValue $arg),
  [(SameLayout $arg, $stick)]
>;

// zhigh.Stick (zhigh.Unstick (%X)) = onnx.LayoutTransform(%X)
// Does not support NHWC layout because onnx.LayoutTransform requires that
// the input and output must have the same shape, but NHWC stickify/unstickify
// transposes the shape.
def StickUnstickDiffLayoutRemovalPattern : Pat<
  (ZHighStickOp:$stick (ZHighUnstickOp:$unstick $arg), $_),
  (ONNXLayoutTransformOp $arg, (GetEncodingAttr $stick)),
  [(NotSameLayout $arg, $stick), (NoOneIsOfNHWCLayout $arg, $stick)]
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

// Since zDNN does not support Sqrt(X), we calculate it by using zDNN-supported
// operations, i.e. Exp and Log.
// Formulas: `sqrt(X) = exp(log(x)/2) = exp(0.5 * log(x))`
def ReplaceONNXSqrtPattern: Pat<
  (ZHighStickOp:$stick (ONNXSqrtOp (ZHighUnstickOp $X)), $layout),
  (ZHighExpOp (ZHighMulOp (ZHighLogOp $X, (returnType $X)),
                          (ZHighStickOp (GetConstantOfType<"0.5"> $X), $layout),
                          (returnType $X))),
  [(IsStaticShapeTensor $X), (SameLayout $X, $stick)]
>;

// Calulation of `1/sqrt(X)` or reciprocal square root is often found in
// deep learning models, but zDNN does not support it. Thus, we rewrite it into
// zDNN-supported operations.
//
// In this rewriting pattern we use:
//   - `exp(log(x)/2)` for `sqrt(x)`, and 
//   - `div(1,x) for `reciprocal`
// So, `1/sqrt(X) = div(1, exp(log(x)/2)) = exp(log(x)/(-2))
//                                        = exp(-0.5 * log(x))`
//
// The pattern
//   zhigh.Stick (onnx.Reciprocal (onnx.Sqrt (zhigh.Unstick (%X))))
// can be replaced by
//   zhigh.Exp (zhigh.Mul (zhigh.Log(%X), -0.5))
//
// Constraints:
//   - `1/sqrt(X)` must be sandwiched by an unstick and stick.
//   - %X should have static shape, and %alpha should be constant.
//
def ReplaceONNXReciprocalSqrtPattern: Pat<
  (ZHighStickOp:$stick (ONNXReciprocalOp (ONNXSqrtOp (ZHighUnstickOp $X))), $layout),
  (ZHighExpOp (ZHighMulOp (ZHighLogOp $X, (returnType $X)),
                          (ZHighStickOp (GetConstantOfType<"-0.5"> $X), $layout),
                          (returnType $X))),
  [(IsStaticShapeTensor $X), (SameLayout $X, $stick)]
>;

#endif // STICK_TD
