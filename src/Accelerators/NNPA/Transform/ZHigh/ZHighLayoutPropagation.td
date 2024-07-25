// SPDX-License-Identifier: Apache-2.0

//===------- ZHighLayoutPropagation.td - Pattern Match for ZHighDialect ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_LAYOUT_PROPAGATION
#define ZHIGH_LAYOUT_PROPAGATION

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def GetLayout : NativeCodeCall<
  "convertZTensorDataLayoutToStringAttr($_builder, ::onnx_mlir::zhigh::getZTensorLayout($0.getType()))"
>;

def LayoutIs: Constraint<
  CPred<"convertZTensorDataLayoutToStringAttr($_builder, ::onnx_mlir::zhigh::getZTensorLayout($_self.getType())) == $0">,
  "Check if the given value's layout is the same as the given layout"
>;

def LayoutIsNot: Constraint<
  CPred<"convertZTensorDataLayoutToStringAttr($_builder, ::onnx_mlir::zhigh::getZTensorLayout($_self.getType())) != $0">,
  "Check if the given value's layout is different from the given layout"
>;

//===----------------------------------------------------------------------===//
// Layout propagation for unary element-wise operations
//
// Unary operations, such as Relu and Tanh, do not depend on a specific layout.
// Thus, it is possible to use the layout of the previous zTensor. In other
// words, propagate the existing layout down to the unary operation, so that
// other rules can be applied to remove unstick-stick pairs.For example,
// This sequence of operations:
//   zhigh.Conv -> Unstick (NHWC) -> Stick (4D) -> zhigh.Relu
// will become:
//   zhigh.Conv -> zhigh.Relu -> Unstick (NHWC) -> Stick (4D)
// then, canonicalization of unstick/stick will remove the unstick-stick pair.
//
//===----------------------------------------------------------------------===//

// Exp
def ExpLayoutPropagatePattern : Pat<
  (ZHighExpOp (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation)),
  (ZHighStickOp (ZHighUnstickOp (ZHighExpOp $x, (returnType $x))), $prevLayout, $saturation),
  [(LayoutIsNot:$x $prevLayout)]
>;

// Log
def LogLayoutPropagatePattern : Pat<
  (ZHighLogOp (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation)),
  (ZHighStickOp (ZHighUnstickOp (ZHighLogOp $x, (returnType $x))), $prevLayout, $saturation),
  [(LayoutIsNot:$x $prevLayout)]
>;

// Relu
def ReluLayoutPropagatePattern : Pat<
  (ZHighReluOp (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation)),
  (ZHighStickOp (ZHighUnstickOp (ZHighReluOp $x, (returnType $x))), $prevLayout, $saturation),
  [(LayoutIsNot:$x $prevLayout)]
>;

// Sigmoid
def SigmoidLayoutPropagatePattern : Pat<
  (ZHighSigmoidOp (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation)),
  (ZHighStickOp (ZHighUnstickOp (ZHighSigmoidOp $x, (returnType $x))), $prevLayout, $saturation),
  [(LayoutIsNot:$x $prevLayout)]
>;

// Tanh
def TanhLayoutPropagatePattern : Pat<
  (ZHighTanhOp (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation)),
  (ZHighStickOp (ZHighUnstickOp (ZHighTanhOp $x, (returnType $x))), $prevLayout, $saturation),
  [(LayoutIsNot:$x $prevLayout)]
>;

//===----------------------------------------------------------------------===//
// Layout propagation for binary element-wise operations
//
// Binary operations, such as Add and Div, do not depend on a specific layout.
// Thus, it is possible to use the layout of the previous zTensor. In other
// words, propagate the existing layout down to the unary operation, so that
// other rules can be applied to remove unstick-stick pairs.For example,
// This sequence of operations:
//   Unstick (NHWC) -> Stick (4D) -> zhigh.Add
//                                       ^
//   Unstick (NHWC) -> Stick (4D) -------|
//              
// will become:
//   Unstick -> zhigh.Add -> Unstick (NHWC) -> Stick (4D)
//                ^
//   Unstick -----|
//
// then, canonicalization of unstick/stick will remove the unstick-stick pairs.
//
//===----------------------------------------------------------------------===//

// Add
def AddLayoutPropagatePattern1 : Pat<
  (ZHighAddOp
     (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighAddOp
         $x,
         (ZHighStickOp (ZHighUnstickOp $y), (GetLayout $x), $saturation),
         (returnType $x))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

def AddLayoutPropagatePattern2 : Pat<
  (ZHighAddOp
     $x,
     (ZHighStickOp (ZHighUnstickOp $y), $prevLayout, $saturation)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighAddOp
         $y,
         (ZHighStickOp (ZHighUnstickOp $x), (GetLayout $y), $saturation),
         (returnType $y))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

// Sub
def SubLayoutPropagatePattern1 : Pat<
  (ZHighSubOp
     (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighSubOp
         $x,
         (ZHighStickOp (ZHighUnstickOp $y), (GetLayout $x), $saturation),
         (returnType $x))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

def SubLayoutPropagatePattern2 : Pat<
  (ZHighSubOp
     $x,
     (ZHighStickOp (ZHighUnstickOp $y), $prevLayout, $saturation)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighSubOp
         $y,
         (ZHighStickOp (ZHighUnstickOp $x), (GetLayout $y), $saturation),
         (returnType $y))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

// Mul
def MulLayoutPropagatePattern1 : Pat<
  (ZHighMulOp
     (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighMulOp
         $x,
         (ZHighStickOp (ZHighUnstickOp $y), (GetLayout $x), $saturation),
         (returnType $x))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

def MulLayoutPropagatePattern2 : Pat<
  (ZHighMulOp
     $x,
     (ZHighStickOp (ZHighUnstickOp $y), $prevLayout, $saturation)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighMulOp
         $y,
         (ZHighStickOp (ZHighUnstickOp $x), (GetLayout $y), $saturation),
         (returnType $y))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

// Div
def DivLayoutPropagatePattern1 : Pat<
  (ZHighDivOp
     (ZHighStickOp (ZHighUnstickOp $x), $prevLayout, $saturation),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighDivOp
         $x,
         (ZHighStickOp (ZHighUnstickOp $y), (GetLayout $x), $saturation),
         (returnType $x))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

def DivLayoutPropagatePattern2 : Pat<
  (ZHighDivOp
     $x,
     (ZHighStickOp (ZHighUnstickOp $y), $prevLayout, $saturation)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighDivOp
         $y,
         (ZHighStickOp (ZHighUnstickOp $x), (GetLayout $y), $saturation),
         (returnType $y))),
     $prevLayout, $saturation),
  [(LayoutIs:$y $prevLayout),   // Two inputs must have the same layout.
   (LayoutIsNot:$x $prevLayout) // The previous layout is different.
  ]
>;

#endif // ZHIGH_LAYOUT_PROPAGATION


