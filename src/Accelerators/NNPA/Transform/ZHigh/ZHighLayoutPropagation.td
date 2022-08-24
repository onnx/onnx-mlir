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
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def _1DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_1D)">;
def _2DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_2D)">;
def _3DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_3D)">;
def _3DSLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_3DS)">;
def NHWCLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_NHWC)">;
def NCHWLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_NCHW)">;
def HWCKLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_HWCK)">;
def NoneLayoutAttr: NativeCodeCall<"StringAttr()">;

// Check values of an ArrayAttr.
def IsPermToNHWC:
  Constraint<CPred<"$_self.getValue().size() == 4 && "
                   "($_self.getValue()[0]).cast<IntegerAttr>().getInt() == 0 && "
                   "($_self.getValue()[1]).cast<IntegerAttr>().getInt() == 2 && "
                   "($_self.getValue()[2]).cast<IntegerAttr>().getInt() == 3 && "
                   "($_self.getValue()[3]).cast<IntegerAttr>().getInt() == 1">,
                 "Permutation is [0, 2, 3, 1]">;

// Check values of an ArrayAttr.
def IsPermToNCHW:
  Constraint<CPred<"$_self.getValue().size() == 4 && "
                   "($_self.getValue()[0]).cast<IntegerAttr>().getInt() == 0 && "
                   "($_self.getValue()[1]).cast<IntegerAttr>().getInt() == 3 && "
                   "($_self.getValue()[2]).cast<IntegerAttr>().getInt() == 1 && "
                   "($_self.getValue()[3]).cast<IntegerAttr>().getInt() == 2">,
                 "Permutation is [0, 3, 1, 2]">;

def GetLayout : NativeCodeCall<
  "convertDataLayoutToStringAttr($_builder, ::onnx_mlir::zhigh::getZTensorLayout($0.getType()))"
>;

def IsNHWC: Constraint<
  CPred<"getZTensorLayout($0.getType()) == ::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC">,
  "Check if a value is NHWC"
>;

def IsNotNHWC: Constraint<
  CPred<"getZTensorLayout($0.getType()) != ::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC">,
  "Check if a value is not NHWC"
>;

def NCHWtoNHWC :
  NativeCodeCall<"emitONNXTranspose($_loc, $_builder, $0, 0, 2, 3, 1)">;

def NHWCtoNCHW:
  NativeCodeCall<"emitONNXTranspose($_loc, $_builder, $0, 0, 3, 1, 2)">;

def IsNoneLayout: Constraint<CPred<" !$_self ">, "Layout attribute is null">;

def IsNCHWLayout: Constraint<
 CPred<"convertStringAttrToDataLayout($_self) ==  ::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NCHW">,
 "Layout attribute is NCHW"
>;

// If the inputs of zhigh.add transitively/originally come from zAIU tensors
// in NHWC layout, then use the original zAIU tensors for zhigh.add.
// In other words, propagate the layout NHWC down, so that other rules can be
// applied to remove stick/unstick pairs.
def AddLayoutPropagatePattern1 : Pat<
  (ZHighAddOp
     (ZHighStickOp
        (ONNXTransposeOp
           (ZHighUnstickOp $x, $unstickNone),
           $perm),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighAddOp
          $x,
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $y, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          (returnType $x)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsPermToNCHW:$perm),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def AddLayoutPropagatePattern2 : Pat<
  (ZHighAddOp
     $x,
     (ZHighStickOp
      (ONNXTransposeOp (ZHighUnstickOp $y, $unstickNone), $perm),
      $stickLayout, 
      $noneLayout)),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighAddOp
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $x, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          $y, (returnType $y)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def AddLayoutPropagatePattern3 : Pat<
  (ZHighAddOp
     (ZHighStickOp
        (ZHighUnstickOp $x, $unstickNCHW),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighAddOp
         $x,
         (ZHighStickOp
           (ZHighUnstickOp $y, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $x)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

def AddLayoutPropagatePattern4 : Pat<
  (ZHighAddOp
     $x,
     (ZHighStickOp
           (ZHighUnstickOp $y, $unstickNCHW),
        $stickLayout,
        $noneLayout)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighAddOp
         $y,
         (ZHighStickOp
           (ZHighUnstickOp $x, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $y)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

// If the inputs of zhigh.sub transitively/originally come from zAIU tensors
// in NHWC layout, then use the original zAIU tensors for zhigh.sub.
// In other words, propagate the layout NHWC down, so that other rules can be
// applied to remove stick/unstick pairs.
def SubLayoutPropagatePattern1 : Pat<
  (ZHighSubOp
     (ZHighStickOp
        (ONNXTransposeOp
           (ZHighUnstickOp $x, $unstickNone),
           $perm),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighSubOp
          $x,
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $y, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          (returnType $x)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsPermToNCHW:$perm),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def SubLayoutPropagatePattern2 : Pat<
  (ZHighSubOp
     $x,
     (ZHighStickOp
      (ONNXTransposeOp (ZHighUnstickOp $y, $unstickNone), $perm),
      $stickLayout, 
      $noneLayout)),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighSubOp
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $x, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          $y, (returnType $y)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def SubLayoutPropagatePattern3 : Pat<
  (ZHighSubOp
     (ZHighStickOp
        (ZHighUnstickOp $x, $unstickNCHW),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighSubOp
         $x,
         (ZHighStickOp
           (ZHighUnstickOp $y, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $x)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

def SubLayoutPropagatePattern4 : Pat<
  (ZHighSubOp
     $x,
     (ZHighStickOp
           (ZHighUnstickOp $y, $unstickNCHW),
        $stickLayout,
        $noneLayout)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighSubOp
         $y,
         (ZHighStickOp
           (ZHighUnstickOp $x, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $y)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

// If the inputs of zhigh.mul transitively/originally come from zAIU tensors
// in NHWC layout, then use the original zAIU tensors for zhigh.mul.
// In other words, propagate the layout NHWC down, so that other rules can be
// applied to remove stick/unstick pairs.
def MulLayoutPropagatePattern1 : Pat<
  (ZHighMulOp
     (ZHighStickOp
        (ONNXTransposeOp
           (ZHighUnstickOp $x, $unstickNone),
           $perm),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighMulOp
          $x,
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $y, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          (returnType $x)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsPermToNCHW:$perm),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def MulLayoutPropagatePattern2 : Pat<
  (ZHighMulOp
     $x,
     (ZHighStickOp
      (ONNXTransposeOp (ZHighUnstickOp $y, $unstickNone), $perm),
      $stickLayout, 
      $noneLayout)),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighMulOp
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $x, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          $y, (returnType $y)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def MulLayoutPropagatePattern3 : Pat<
  (ZHighMulOp
     (ZHighStickOp
        (ZHighUnstickOp $x, $unstickNCHW),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighMulOp
         $x,
         (ZHighStickOp
           (ZHighUnstickOp $y, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $x)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

def MulLayoutPropagatePattern4 : Pat<
  (ZHighMulOp
     $x,
     (ZHighStickOp
           (ZHighUnstickOp $y, $unstickNCHW),
        $stickLayout,
        $noneLayout)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighMulOp
         $y,
         (ZHighStickOp
           (ZHighUnstickOp $x, $unstickNCHW),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $y)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

// If the inputs of zhigh.div transitively/originally come from zAIU tensors
// in NHWC layout, then use the original zAIU tensors for zhigh.div.
// In other words, propagate the layout NHWC down, so that other rules can be
// applied to remove stick/unstick pairs.
def DivLayoutPropagatePattern1 : Pat<
  (ZHighDivOp
     (ZHighStickOp
        (ONNXTransposeOp
           (ZHighUnstickOp $x, $unstickNone),
           $perm),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighDivOp
          $x,
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $y, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          (returnType $x)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsPermToNCHW:$perm),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def DivLayoutPropagatePattern2 : Pat<
  (ZHighDivOp
     $x,
     (ZHighStickOp
      (ONNXTransposeOp (ZHighUnstickOp $y, $unstickNone), $perm),
      $stickLayout, 
      $noneLayout)),
  (ZHighStickOp
     (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighDivOp
          (ZHighStickOp
            (NCHWtoNHWC (ZHighUnstickOp $x, (NoneLayoutAttr))),
            (NHWCLayoutAttr),
            (NoneLayoutAttr)),
          $y, (returnType $y)),
        $unstickNone)),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def DivLayoutPropagatePattern3 : Pat<
  (ZHighDivOp
     (ZHighStickOp
        (ZHighUnstickOp $x, $unstickNCHW),
        $stickLayout,
        $noneLayout),
     $y),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighDivOp
         $x,
         (ZHighStickOp
           (ZHighUnstickOp $y, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $x)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $x), (IsNotNHWC $y), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

def DivLayoutPropagatePattern4 : Pat<
  (ZHighDivOp
     $x,
     (ZHighStickOp
           (ZHighUnstickOp $y, $unstickNCHW),
        $stickLayout,
        $noneLayout)),
  (ZHighStickOp
     (ZHighUnstickOp
       (ZHighDivOp
         $y,
         (ZHighStickOp
           (ZHighUnstickOp $x, (NoneLayoutAttr)),
           (NHWCLayoutAttr),
           (NCHWLayoutAttr)),
         (returnType $y)),
       $unstickNCHW),
     $stickLayout,
     $noneLayout),
  [(IsNHWC $y), (IsNotNHWC $x), (IsNCHWLayout:$unstickNCHW),
   (IsNoneLayout:$noneLayout)]
>;

// If the input of zhigh.relu transitively/originally comes from a zAIU tensor
// in NHWC layout, then use the original zAIU tensor for zhigh.relu.
// In other words, propagate the layout NHWC down, so that other rules can be
// applied to remove stick/unstick pairs.
def ReluLayoutPropagatePattern1 : Pat<
  (ZHighReluOp
    (ZHighStickOp
      (ONNXTransposeOp
        (ZHighUnstickOp $x, $unstickNone),
        $perm),
      $stickLayout,
      $noneLayout)),
  (ZHighStickOp
    (NHWCtoNCHW
      (ZHighUnstickOp
        (ZHighReluOp $x, (returnType $x)),
        $unstickNone)),
    $stickLayout,
    $noneLayout),
  [(IsNHWC $x), (IsPermToNCHW:$perm),
   (IsNoneLayout:$noneLayout), (IsNoneLayout:$unstickNone)]
>;

def ReluLayoutPropagatePattern2 : Pat<
  (ZHighReluOp
    (ZHighStickOp
      (ZHighUnstickOp $x, $unstickNCHW),
      $stickLayout,
      $noneLayout)),
  (ZHighStickOp
    (ZHighUnstickOp
      (ZHighReluOp $x, (returnType $x)),
      $unstickNCHW),
    $stickLayout,
    $noneLayout),
  [(IsNHWC $x), (IsNoneLayout:$noneLayout), (IsNCHWLayout:$unstickNCHW)]
>;

#endif // ZHIGH_LAYOUT_PROPAGATION


