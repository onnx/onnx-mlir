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

// The folowing pattern was found in bertsquad and GPT models.
// ```
// %0 = "zhigh.Unstick"(%X) {layout = "2D"} : tensor<?x768xf32, #zhigh.encoding<{dataLayout = "3DS"}>> -> tensor<?x768xf32>
// %1 = "onnx.Reshape"(%0) : tensor<?x768xf32> -> tensor<?x256x12x64xf32>
// %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]}: tensor<?x256x12x64xf32> -> tensor<?x12x256x64xf32>
// %3 = "onnx.Reshape"(%2) tensor<?x12x256x64xf32> -> tensor<?x256x64xf32>
// %4 = "zhigh.Stick"(%4) {layout = "3DS"} : tensor<?x256x64xf32> -> tensor<?x256x64xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// ```
// where
//  - the 1st Reshape can be expressed by an affine map, i.e. (d0, d1) -> (d0/256, d0%256, d1/64, d1%/64)
//  - the Transpose can be expressed by an affine map, i.e. (d0, d1, d2, d3) -> (d0, d2, d1, d3), and
//  - the 2nd Reshape can be expressed by an affine map, i.e. (d0, d1, d2, d3)  -> (d0*12+d1, d2, d3)
// 
// Thus, we will replace Reshape, Transpose by onnx.ShapeTransform operations
// that can be composed automatically into a single onnx.ShapeTransform.
//
// Note: current implementation only supports tensors with static dimensions.

def ReshapeTransposeReshape2DTo3DSPattern : Pat<
  (ZHighStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (ZHighUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout3DS),
  (ZHighStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (ZHighUnstickOp $X),
          (GetTiling2DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetCollapsing4DTo3DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout3DS),
  [(TensorHas2DLayout $X), (Is3DSLayout $layout3DS),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsTiling2DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsCollapsing4DTo3D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
  ]
>;

// Reversed direction of the above pattern.
def ReshapeTransposeReshape3DSTo2DPattern : Pat<
  (ZHighStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (ZHighUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout2D),
  (ZHighStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (ZHighUnstickOp $X),
          (GetTiling3DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetCollapsing4DTo2DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout2D),
  [(TensorHas3DSLayout $X), (Is2DLayout $layout2D),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsTiling3DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsCollapsing4DTo2D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
  ]
>;

#endif // STICK_TD
