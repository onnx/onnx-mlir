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
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// DRR patterns 
//===----------------------------------------------------------------------===//

def NoneTypeStickRemovalPattern : Pat<
  (ZHighStickOp:$stick $arg, $layout1, $_),
  (replaceWithValue $arg),
  [(IsNoneType:$arg)]
>;

// zhigh.Stick (zhigh.Unstick (%X)) = %X
def StickUnstickSameLayoutRemovalPattern : Pat<
  (ZHighStickOp:$stick (ZHighUnstickOp:$unstick $arg), $_, $_),
  (replaceWithValue $arg),
  [(SameLayout $arg, $stick)]
>;

// zhigh.Stick (zhigh.Unstick (%X)) = onnx.LayoutTransform(%X)
// Does not support NHWC layout because onnx.LayoutTransform requires that
// the input and output must have the same shape, but NHWC stickify/unstickify
// transposes the shape.
def StickUnstickDiffLayoutRemovalPattern : Pat<
  (ZHighStickOp:$stick (ZHighUnstickOp:$unstick $arg), $_, $_),
  (ONNXLayoutTransformOp $arg, (GetEncodingAttr $stick)),
  [(NotSameLayout $arg, $stick), (NoOneIsOfNHWCLayout $arg, $stick),
   // Do not support 1D and 2DS because of this issue: https://github.com/onnx/onnx-mlir/issues/1940
   (NoOneIsOf1DLayout $arg, $stick), (NoOneIsOf2DSLayout $arg, $stick)]
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
                $layout, $_),
  (ZHighSubOp
     (ZHighReluOp $X, (returnType $X)),
     (ZHighReluOp (ZHighMulOp $X,
                              (ZHighStickOp (GetMinusBcastConst $alpha,
                                             $out),
                                            $layout,
                                            // Donot saturate since input orignally from NNPA
                                            (NoneIntegerAttr)), 
                              (returnType $X)),
                  (returnType $X))),
  [(IsStaticShapeTensor $X), (IsPlusConstantFloat $alpha),
   (SameLayout $X, $stickout)]
>;

// The pattern
//   zhigh.Stick (onnx.Softplus (zhigh.Unstick (%X)))
// can be replaced by
//   %minusOne = zhigh.Stick(GetConstantOfType<"-1.0">, %X)
//   %minusX = zhigh.Mul(%X, %minusOne)
//   zhigh.Add (
//       zhigh.Relu(%X),
//       zhigh.log(zhigh.Sub(zhigh.Exp(zhigh.Min(%X,%minusX)), %minusOne)))
// References:
// http://www.beam2d.net/blog/2014/03/02/softplus/ (Japanese)
// https://www-beam2d-net.translate.goog/blog/2014/03/02/softplus/?_x_tr_sch=http&_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=ja&_x_tr_pto=wapp (Translated English)
// c.f.
// -|x| is replaced by min(x, -x), since NNPA does not have the abs(x) function.
// Constraints:
//   - %X should have static shape
//
def ReplaceONNXSoftplusPattern: Pattern<
  (ZHighStickOp:$stickout (ONNXSoftplusOp:$out (ZHighUnstickOp $X)), $layout, $_),
  [
   // Get stickified constant of minus one with input shape.
   // Donot saturate since input originally from NNPA.
   (ZHighStickOp:$minusOne (GetConstantOfType<"-1.0"> $out), $layout, (NoneIntegerAttr)),
   // Get minus X with input shape.
   (ZHighMulOp:$minusX $X, $minusOne, (returnType $X)),

   // Get Softplus
   (ZHighAddOp
      (ZHighReluOp $X, (returnType $X)),
      (ZHighLogOp (ZHighSubOp (ZHighExpOp (ZHighMinOp $X, $minusX,
                                                      (returnType $X)),
                                          (returnType $X)),
                              $minusOne, (returnType $X)),
                  (returnType $X))),
  ],
  [(IsStaticShapeTensor $X), (SameLayout $X, $stickout)]
>;

// Calculation of `1/sqrt(X)` or reciprocal square root is often found in
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
  (ZHighStickOp:$stick (ONNXReciprocalOp (ONNXSqrtOp (ZHighUnstickOp:$unstick $X))), $layout, $saturation),
  (ZHighExpOp (ZHighMulOp (ZHighLogOp $X, (returnType $X)),
                          (ZHighStickOp (GetConstantOfType<"-0.5"> $unstick), $layout, $saturation),
                          (returnType $X))),
  [(IsStaticShapeTensor $X), (SameLayout $X, $stick)]
>;

// The following pattern was found in Roberta models.
// ```
//    %66 = "zhigh.Unstick"(%65) : (tensor<12x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<12x384x64xf32>
//    %67 = "onnx.Reshape"(%66, %2) {allowzero = 0 : si64} : (tensor<12x384x64xf32>, tensor<4xi64>) -> tensor<1x12x384x64xf32>
//    %68 = "onnx.Transpose"(%67) {onnx_node_name = "Transpose_94", perm = [0, 2, 1, 3]} : (tensor<1x12x384x64xf32>) -> tensor<1x384x12x64xf32> 
//    %69 = "onnx.Reshape"(%68, %9) {allowzero = 0 : si64, onnx_node_name = "Reshape_104"} : (tensor<1x384x12x64xf32>, tensor<3xi64>) -> tensor<1x384x768xf32>
//    %70 = "zhigh.Stick"(%69) {layout = "3DS"} : (tensor<1x384x768xf32>) -> tensor<1x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// ```
// When the input tensor %65 (here tensor<12x384x64xf16) with dims E3xE2xE1 and 3DS layout satisfies the following properties
// 1) E2 % 32 == 0
// 2) E1 % 64 == 0
// Then we can guarantee that input value %65 has the same value
// in the same memory location (*) than output %70.
// (*) does not say the tensor are identical, just that they store
//     the same value at the same byte offset in the memory.
//
// High level intuition for this is that the reshape / transpose / reshape
// perform some memory layout operations, which results in no change in the
// 3DS representation as that representation also perform layout changes.
// 
// Limitation: current pattern assume static sizes.

def ReshapeTransposeReshapeRoberta3DSWPattern1 : Pat<
  // Input: X -> unstick -> reshape1 -> transpose -> reshape 2 -> stick.
  (ZHighStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (ZHighUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout3DS, $saturation),
    // Output: initial X value unchanged, but transformed with the new compatible shape.
    (ZHighReshapeOp $X, (CreateShapeOp $stick), (GetLayout $stick)),
    // Conditions.
    [(TensorHas3DSLayout $X), (Is3DSLayout $layout3DS), // Input/output are 3DS.
     (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
     // Static shapes only.
     (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
     (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
     (IsShapeDimMultipleOf32<1> $X), // Second dim of input is a multiple of 32.
     (IsShapeDimMultipleOf64<2> $X), // Third dim of input is a multiple of 64.
     (Is4DTransposePermutationEqualTo0213 $perm), // Permute middle 2 dims.
     (IsLeftmostTiling3DTo4D $reshape1), // 1st reshape is tiling in the leftmost dimension
     (IsRightmostCollapsing4DTo3D $reshape2), // 2nd reshape is collapsing the last two dimensions. 
   ]
>;

// Second pattern found in roberta
//
//    %33 = "zhigh.Unstick"(%32) : (tensor<8x384x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<8x384x768xf32> // unstick
//    %44 = "onnx.Reshape"(%33, %7) {allowzero = 0 : si64, onnx_node_name = "Reshape_64"} : (tensor<8x384x768xf32>, tensor<4xi64>) -> tensor<8x384x12x64xf32> // last goes from 768 to 12, 64
//    %45 = "onnx.Transpose"(%44) {onnx_node_name = "Transpose_65", perm = [0, 2, 1, 3]} : (tensor<8x384x12x64xf32>) -> tensor<8x12x384x64xf32> // same permute
//    %50 = "onnx.Reshape"(%45, %2) {allowzero = 0 : si64} : (tensor<8x12x384x64xf32>, tensor<3xi64>) -> tensor<96x384x64xf32>  // collapse first 2 dims
//    %52 = "zhigh.Stick"(%50) {layout = "3DS"} : (tensor<96x384x64xf32>) -> tensor<96x384x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
//
// Namely unstick -> reshape (by tiling rightmost one to E1/64, 64) -> permute (2nd & 3rd) -> reshape (collapse first 2) -> stick
// Shapes goes from 8 x 384 x (12*64 = 768) to (8*12 = 96) x 384 x 64
//
// pattern works only when input E2 % 32 and E1 % 64 == 0

def ReshapeTransposeReshapeRoberta3DSWPattern2 : Pat<
  // Input: X -> unstick -> reshape1 -> transpose -> reshape 2 -> stick.
  (ZHighStickOp:$stick
    (ONNXReshapeOp:$reshape2
      (ONNXTransposeOp:$transpose
        (ONNXReshapeOp:$reshape1
          (ZHighUnstickOp:$unstick $X),
          $shape1, $_),
        $perm),
      $shape2, $_),
    $layout3DS, $saturation),
    // Output: initial X value unchanged, but transformed with the compatible shape.
    (ZHighReshapeOp $X, (CreateShapeOp $stick), (GetLayout $stick)),
    // Conditions.
    [(TensorHas3DSLayout $X), (Is3DSLayout $layout3DS), // Input/output are 3DS.
     (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
     // Static shapes only.
     (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
     (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
     (IsShapeDimMultipleOf32<1> $X), // Second dim of input is a multiple of 32.
     (IsShapeDimMultipleOf64<2> $X), // Third dim of input is a multiple of 64.
     (Is4DTransposePermutationEqualTo0213 $perm), // Permute middle 2 dims.
     (IsRightmostTiling3DTo4DBy64 $reshape1), // 1st reshape is tiling by 64 the rightmost dimension
     (IsLeftmostCollapsing4DTo3D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
   ]
>;

// The following pattern was found in bertsquad and GPT models.
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
    $layout3DS, $saturation),
  (ZHighStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (ZHighUnstickOp $X),
          (GetTiling2DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetLeftmostCollapsing4DTo3DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout3DS, $saturation),
  [(TensorHas2DLayout $X), (Is3DSLayout $layout3DS),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsTiling2DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsLeftmostCollapsing4DTo3D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
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
    $layout2D, $saturation),
  (ZHighStickOp
    (ONNXShapeTransformOp // reshape
      (ONNXShapeTransformOp // transpose
        (ONNXShapeTransformOp // reshape
          (ZHighUnstickOp $X),
          (GetLeftmostTiling3DTo4DMap $reshape1),
          (returnType (GetResultType $reshape1))),
        (GetTransposeMap $perm),
        (returnType (GetResultType $transpose))),
      (GetCollapsing4DTo2DMap $reshape2),
      (returnType (GetResultType $reshape2))),
    $layout2D, $saturation),
  [(TensorHas3DSLayout $X), (Is2DLayout $layout2D),
   (IsStaticShapeTensor $X), (IsStaticShapeTensor $unstick),
   (IsStaticShapeTensor $reshape1), (IsStaticShapeTensor $transpose),
   (IsStaticShapeTensor $reshape2),(IsStaticShapeTensor $stick),
   (IsLeftmostTiling3DTo4D $reshape1), // 1st reshape is tiling over each input dimension
   (IsCollapsing4DTo2D $reshape2), // 2nd reshape is collapsing the first two dimensions. 
  ]
>;


// Pattern in the CCFD model.
// 4DS and 3DS have exactly same data values when the second dim of 4DS is 1.
// (The second dim of 4DS indicates unidirectional or bidirectional LSTM/GRU/RNN,
// in which: 1 means unidirectional, 2 means bidirectional)
// This rewriting is true no matter dims (except the 2nd dim) are dynamic or static.
def Stick3DSSqueezeUnstick4DSPattern: Pat<
  // Input: X -> unstick (4DS) -> Squeeze (axis=1) -> stick (3DS).
  (ZHighStickOp:$stick
    (ONNXSqueezeOp (ZHighUnstickOp:$unstick $X), $axes),
    $_, $_),
  // Output: initial X value unchanged, but transformed with the new layout.
  (ZHighReshapeOp $X, (Create3DSShapeFrom4DS $X), (GetLayout $stick)),
  // Conditions.
  [(TensorHas4DSLayout $X),     // Input is 4DS.
   (TensorHas3DSLayout $stick), // Output is 3DS.
   (IsConstOf<1> $axes),        // squeeze at axis 1.
  ]
>;

#endif // STICK_TD
