// SPDX-License-Identifier: Apache-2.0

//===- ONNXToZHigh.td - Replacing ONNX Ops by ZHigh Ops -*- tablegen ------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_TO_ZHIGH
#define ONNX_TO_ZHIGH

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.td"
include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def IsEnableScalarBcastBinary: Constraint<CPred<"isEnableScalarBcastBinary()">>;

def HasOneUse : Constraint<CPred<"$_self.hasOneUse()">, "op has exactly one use">;

def IsNotNoneType : Constraint<CPred<"!mlir::isa<NoneType>(($_self).getType())">>;

class HasRankOf<int rank> : Constraint<
  CPred<"mlir::isa<ShapedType>($0.getType()) && "
        "mlir::cast<ShapedType>($0.getType()).hasRank() && "
        "mlir::cast<ShapedType>($0.getType()).getRank() == " # rank>
>;

def IsBiasNoneOr1D : Constraint<
  CPred<"mlir::isa<NoneType>($_self.getType()) || "
        "(mlir::isa<ShapedType>($_self.getType()) && "
	"  mlir::cast<ShapedType>($_self.getType()).hasRank() && "
	"  mlir::cast<ShapedType>($_self.getType()).getRank() == 1)">
>;

class VariadicSizeIs<int N> : Constraint<
  CPred<"$0.size() == " # N>, "The variadic operand's size is N"
>;

class VariadicSizeIsGT<int N> : Constraint<
  CPred<"$0.size() > " # N>, "The variadic operand's size is greater than N"
>;

// Create a unit constant that will be used as none input.
def CreateNoneValue : NativeCodeCall<"$_builder.create<ONNXNoneOp>($_loc)">;

class getNth<int n, string x> : NativeCodeCall<x # "[" # n # "]">;

class GetNthVariadicOperand<int n> : NativeCodeCall<"$0[" # n # "]">;

class getNthVariadicResults<int n, string x> : NativeCodeCall<x # ".getResults()[" # n # "]">;

def GetShape :
	NativeCodeCall<"mlir::cast<ShapedType>($0.getType()).getShape()">;

def GetDynShape:
        NativeCodeCall<"getDynShape($_loc, $_builder, $0)">;

def GetRank :
	NativeCodeCall<"mlir::cast<RankedTensorType>($0.getType()).getRank()">;

def GetSInt :
        NativeCodeCall<"mlir::cast<IntegerAttr>($0).getSInt()">;

class GetStrAttr<string s> :
        NativeCodeCall<"$_builder.getStringAttr(\"" # s # "\")">;

def GetI64Attr :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), $0)">;

class GetI64NAttr<int n> :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), " # n # ")">;

def GetI64ZeroAttr :
        NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), 0)">;

class GetI64ArrayAttr<int n> :
        NativeCodeCall<"$_builder.getI64ArrayAttr(" # n # ")">;

def GetFloatAttr : NativeCodeCall<"$_builder.getFloatAttr($0.getType(), $0.getValueAsDouble())">;

def GetUnrankedTensorTypeOf : NativeCodeCall<
  "UnrankedTensorType::get(mlir::cast<ShapedType>($0.getType()).getElementType())"
>;

class EmitOp3<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2)">;

class EmitOp4<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2, $3)">;

def IsF32ScalarConstantTensor: Constraint<
  CPred<"isF32ScalarConstantTensor($0)">,
  "The value is a constant tensor of a single f32 value"
>;

def GetScalarF32AttrFromConstant :
        NativeCodeCall<"getScalarF32AttrFromConstant($0)">;

def IsConstOfHalf : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 0.5)">,
  "Value is all halves (0.5) for a constant tensor">;

def IsConstOfOnes : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 1.0)">,
  "Value is an all-ones constant tensor">;

def IsConstOfTwos : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 2.0)">,
  "Value is all twos for a constant tensor">;

def IsConstOfThrees : Constraint<
  CPred<"isDenseONNXConstant($_self) && isConstOf($_self, 3.0)">,
  "Value is all threes for a constant tensor">;

def IsFloatType : Constraint<
  CPred<"isFloatType($_self)">,
  "Value has element type of float">;

//===----------------------------------------------------------------------===//
// ONNXLeakyReluOp %X = ZHighUnstickOp (ZHighLeakyReluOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXLeakyReluPattern : Pat<
  (ONNXLeakyReluOp $x, $alpha),
  (ZHighUnstickOp (ZHighLeakyReluOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)), (GetFloatAttr $alpha),
                               (returnType $s_x))),
  [(IsCompatibleWithNNPALevelArch15)]
>;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = ZHighUnstickOp (ZHighReluOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXReluPattern : Pat<
  (ONNXReluOp $x),
  (ZHighUnstickOp (ZHighReluOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                               (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXGeluOp %X = ZHighUnstickOp (ZHighGeluOp (ZHighStickOp %X), $approximate)
//===----------------------------------------------------------------------===//
def replaceONNXGeluPattern : Pat<
  (ONNXGeluOp:$res $x, $approximate),
  (ZHighUnstickOp (ZHighGeluOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                                 $approximate,
                                 (returnType $s_x))),
  [(IsCompatibleWithNNPALevelArch15)]
>;

//===----------------------------------------------------------------------===//
// ONNXTanhOp %X = ZHighUnstickOp (ZHighTanhOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXTanhPattern : Pat<
  (ONNXTanhOp $x),
  (ZHighUnstickOp (ZHighTanhOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                               (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXSigmoidOp %X = ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXSigmoidPattern : Pat<
  (ONNXSigmoidOp $x),
  (ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                                  (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = ZHighUnstickOp (ZHighAddOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXAddPattern : Pat<
  (ONNXAddOp $x, $y),
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (ZHighStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x))
)
>;

//===----------------------------------------------------------------------===//
//
// ONNXSumOp %X = ZHighUnstickOp (ZHighAddOp (ZHighStickOp %X_0),
//                                           (ZHighStickOp %X_1))
//
//===----------------------------------------------------------------------===//
def GetONNXSumOpWithoutFirst : NativeCodeCall<
  "$_builder.create<ONNXSumOp>("
  "  $_loc,"
  "  mlir::cast<ShapedType>($0[0].getType()),"
  "  ValueRange(OperandRange{$0.begin() + 1, $0.end()}))"
>;

// Recursive case: ONNXSumOp (N inputs) = ONNXSumOp (N-1 inputs) + ZHighAddOp.
def replaceONNXSumOpPatternRecursion : Pat<
  (ONNXSumOp $x),
  (ZHighUnstickOp
     (ZHighAddOp
        (ZHighStickOp:$s_x (GetNthVariadicOperand<0> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp (GetONNXSumOpWithoutFirst $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (returnType $s_x))),
  [(VariadicSizeIsGT<2> $x)], [], (addBenefit 1)
>;

// Singleton case: ONNXSumOp (2 inputs) to ZHighAddOp.
def replaceONNXSumOpPatternSingleton : Pat<
  (ONNXSumOp $x),
  (ZHighUnstickOp
     (ZHighAddOp
        (ZHighStickOp:$s_x (GetNthVariadicOperand<0> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp (GetNthVariadicOperand<1> $x), (NoneLayoutAttr), (GetDefaultSaturation)),
        (returnType $s_x))),
  [(VariadicSizeIs<2> $x)], [], (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = ZHighUnstickOp (ZHighSubOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXSubPattern : Pat<
  (ONNXSubOp $x, $y),
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (ZHighStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXMulOp %X = ZHighUnstickOp (ZHighMulOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXMulPattern : Pat<
  (ONNXMulOp $x, $y),
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (ZHighStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = ZHighUnstickOp (ZHighDivOp (ZHighStickOp
// %X),(ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceONNXDivPattern : Pat<
  (ONNXDivOp $x, $y),
  (ZHighUnstickOp (ZHighDivOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (ZHighStickOp $y, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x))),
  [], [],
  (addBenefit 0)
>;

def replaceONNXDivBroadcastPattern1 : Pat<
  (ONNXDivOp $x, $y),
  (ZHighUnstickOp
    (ZHighDivOp
      (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
      (ZHighStickifiedConstantOfShapeOp
        (GetDynShape $x),
        (GetScalarF32AttrFromConstant $y),
        (NoneLayoutAttr)),
      (returnType $s_x))),
  [(IsEnableScalarBcastBinary), (IsF32ScalarConstantTensor $y)], [],
  (addBenefit 1)
>;

def replaceONNXDivBroadcastPattern2 : Pat<
  (ONNXDivOp $x, $y),
  (ZHighUnstickOp
    (ZHighDivOp
      (ZHighStickifiedConstantOfShapeOp
        (GetDynShape $y),
        (GetScalarF32AttrFromConstant $x),
        (NoneLayoutAttr)),
      (ZHighStickOp:$s_y $y, (NoneLayoutAttr), (GetDefaultSaturation)),
      (returnType $s_y))),
  [(IsEnableScalarBcastBinary), (IsF32ScalarConstantTensor $x)], [],
  (addBenefit 1)
>;

//===----------------------------------------------------------------------===//
// ONNXLogOp %X = ZHighUnstickOp (ZHighLogOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXLogPattern : Pat<
  (ONNXLogOp $x),
  (ZHighUnstickOp (ZHighLogOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXExpOp %X = ZHighUnstickOp (ZHighExpOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXExpPattern : Pat<
  (ONNXExpOp $x),
  (ZHighUnstickOp (ZHighExpOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXMinOp %X = ZHighUnstickOp (ZHighMinOp (ZHighStickOp %X[0])
//                                           (ZHighStickOp %X[1]))
//===----------------------------------------------------------------------===//
def replaceONNXMinPattern : Pat<
  (ONNXMinOp $x),
  (ZHighUnstickOp (ZHighMinOp (ZHighStickOp:$s_x (getNth<0, "x">),
                                                 (NoneLayoutAttr),
                                                 (GetDefaultSaturation)),
                              (ZHighStickOp (getNth<1, "x">),
                                            (NoneLayoutAttr),
                                            (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXMaxOp %X = ZHighUnstickOp (ZHighMaxOp (ZHighStickOp %X[0])
//                                           (ZHighStickOp %X[1]))
//===----------------------------------------------------------------------===//
def replaceONNXMaxPattern : Pat<
  (ONNXMaxOp $x),
  (ZHighUnstickOp (ZHighMaxOp (ZHighStickOp:$s_x (getNth<0, "x">),
                                                 (NoneLayoutAttr),
                                                 (GetDefaultSaturation)),
                              (ZHighStickOp (getNth<1, "x">),
                                            (NoneLayoutAttr),
                                            (GetDefaultSaturation)),
                              (returnType $s_x)))
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp(1, (ONNXSqrtOp %X)) = ZHighUnstickOp
//                                 (ZHighInvSqrtOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceDiv1SqrtPattern : Pat<
  (ONNXDivOp $a, (ONNXSqrtOp $x)),
  (ZHighUnstickOp (ZHighInvSqrtOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                               (returnType $s_x))),
  [(IsCompatibleWithNNPALevelArch15),(IsConstOfOnes:$a),(IsFloatType:$a)]
>;

//===----------------------------------------------------------------------===//
// ONNXReciprocalOp(ONNXSqrtOp %X) = ZHighUnstickOp
//                                   (ZHighInvSqrtOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceReciprocalSqrtPattern : Pat<
  (ONNXReciprocalOp (ONNXSqrtOp $x)),
  (ZHighUnstickOp (ZHighInvSqrtOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                                   (returnType $s_x))),
  [(IsCompatibleWithNNPALevelArch15)]
>;

//===----------------------------------------------------------------------===//
// ONNXSqrtOp %X = ZHighUnstickOp (ZHighSqrtOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceONNXSqrtPattern : Pat<
  (ONNXSqrtOp $x),
  (ZHighUnstickOp (ZHighSqrtOp (ZHighStickOp:$s_x $x, (NoneLayoutAttr), (GetDefaultSaturation)),
                                  (returnType $s_x))),
  [(IsCompatibleWithNNPALevelArch15)]
>;

//===----------------------------------------------------------------------===//
// ONNXSoftmaxOp %X = ONNXSqueezeOp
//                        (ZHighUnstickOp
//                            (ZHighSoftmaxOp
//                                (ZHighStickOp (ONNXUnsqueezeOp %X, 0), (GetDefaultSaturation)),
//                                0))
//===----------------------------------------------------------------------===//

def replaceONNXSoftmax2DPattern : Pattern<
  (ONNXSoftmaxOp:$res $x, $axis),
  [
   // Get unstick result
   (ZHighUnstickOp:$unstick
       (ZHighSoftmaxOp
           (ZHighStickOp:$s_x
               (EmitOp3<"ONNXUnsqueezeV11Op">
                   (GetUnrankedTensorTypeOf $x),
                   $x,
                   (GetI64ArrayAttr<0>)),
               (_3DSLayoutAttr),
               (GetDefaultSaturation)),
           (GetStringAttr<"ACT_NONE">),
           (returnType $s_x))),

   (EmitOp3<"ONNXSqueezeV11Op">
       (GetTypeOf $res),
       $unstick,
       (GetI64ArrayAttr<0>))
  ],
  [(HasRankOf<2> $x)]
>;

def replaceONNXSoftmax3DPattern : Pat<
  (ONNXSoftmaxOp:$res $x, $axis),
  // Get unstick result
  (ZHighUnstickOp:$unstick
     (ZHighSoftmaxOp
        (ZHighStickOp:$s_x $x, (_3DSLayoutAttr), (GetDefaultSaturation)),
        (GetStringAttr<"ACT_NONE">),
        (returnType $s_x))),
  [(HasRankOf<3> $x)]
>;

//===----------------------------------------------------------------------===//
// ONNXLogSoftmaxOp (ONNXSoftmaxOp %X) =
//   ONNXSqueezeOp
//     (ZHighUnstickOp
//         (ZHighSoftmaxOp
//            (ZHighStickOp (ONNXUnsqueezeOp %X, 0), (GetDefaultSaturation)),
//            "ACT_LOG"),
//     0)
//===----------------------------------------------------------------------===//
def IsSoftmaxLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXSoftmaxOp>(" #
             "dyn_cast_or_null<ONNXSoftmaxOp>($0.getDefiningOp()))">,
  "Softmax is legal for zDNN"
>;
def replaceONNXLogSoftmaxPattern : Pattern<
  (ONNXLogOp:$res (ONNXSoftmaxOp:$s $x, $axis)),
  [
   // Get unstick result
   (ZHighUnstickOp:$unstick
       (ZHighSoftmaxOp
           (ZHighStickOp:$s_x
               (EmitOp3<"ONNXUnsqueezeV11Op">
                   (GetUnrankedTensorTypeOf $x),
                   $x,
                   (GetI64ArrayAttr<0>)),
               (_3DSLayoutAttr),
               (GetDefaultSaturation)),
           (GetStringAttr<"ACT_LOG">),
           (returnType $s_x))),

   (EmitOp3<"ONNXSqueezeV11Op">
       (GetTypeOf $res),
       $unstick,
       (GetI64ArrayAttr<0>))
  ],
  [(IsSoftmaxLegalForZDNN $s)]
>;

//===----------------------------------------------------------------------===//
// ONNXReduceMeanV13Op %X = (ZHighUnstickOp
//                          (ZHighMeanReduce2DOp
//                             (ZHighStickOp %X)))
//===----------------------------------------------------------------------===//
def replaceONNXReduceMeanV13Pattern : Pat<
  (ONNXReduceMeanV13Op:$res $x, $_, $_),
  (ZHighUnstickOp (ZHighMeanReduce2DOp (ZHighStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation))))
>;

//===----------------------------------------------------------------------===//
// ONNXReduceMaxOp %X =
//                 (ZHighUnstickOp
//                     (ZHighReduceMaxOp
//                         (ZHighStickOp %X)))
//===----------------------------------------------------------------------===//
def replaceONNXReduceMaxPattern : Pat<
  (ONNXReduceMaxOp:$res $data, $axes, $keepdims, $noop_with_empty_axes),
  (ZHighUnstickOp (ZHighReduceMaxOp (ZHighStickOp:$s_x $data, (NoneLayoutAttr), 
                                    (GetDefaultSaturation)))),
  [(IsCompatibleWithNNPALevelArch15)]
>;

//===----------------------------------------------------------------------===//
// ONNXReduceMinOp %X =
//                 (ZHighUnstickOp
//                     (ZHighReduceMinOp
//                         (ZHighStickOp %X)))
//===----------------------------------------------------------------------===//
def replaceONNXReduceMinPattern : Pat<
  (ONNXReduceMinOp:$res $data, $axes, $keepdims, $noop_with_empty_axes),
  (ZHighUnstickOp (ZHighReduceMinOp (ZHighStickOp:$s_x $data, (NoneLayoutAttr), 
                                    (GetDefaultSaturation)))),
  [(IsCompatibleWithNNPALevelArch15)]
>;
//===----------------------------------------------------------------------===//
// ONNXMaxPoolSingleOutOp %X =
//                 (ZHighUnstickOp
//                     (ZHighMaxPoolOp
//                         (ZHighStickOp %X),
//                         $krnel_shape,
//                         $strides,
//		           $padding_type))
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeMaxPool : NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def GetI64ArrayAttrKernelShapeMaxPool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def GetI64ArrayAttrStridesMaxPool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXMaxPoolSingleOutOp,"
  "ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>"
  "($0.getDefiningOp<ONNXMaxPoolSingleOutOp>()))">;

def replaceONNXMaxPoolSingleOutPattern : Pattern<
  (ONNXMaxPoolSingleOutOp:$res $x, $_, $_, $_, $_, $_, $_, $_),
  [
    // Get attributes using shape helper
    (GetStrAttrPaddingtypeMaxPool:$padtype $res),
    (GetI64ArrayAttrKernelShapeMaxPool:$kernel_shape $res),
    (GetI64ArrayAttrStridesMaxPool:$strides $res),

    (ZHighUnstickOp
       (ZHighMaxPool2DOp
          (ZHighStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
          $kernel_shape,
          $strides,
          $padtype))
  ]
>;

//===----------------------------------------------------------------------===//
// ONNXAveragePoolOp %X =
//                 (ZHighUnstickOp
//                     (ZHighAvgPoolOp
//                         (ZHighStickOp %X),
//                         $krnel_shape,
//                         $strides,
//		           $padding_type))
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeAveragePool : NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def GetI64ArrayAttrKernelShapeAveragePool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def GetI64ArrayAttrStridesAveragePool: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXAveragePoolOp,"
  "ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>"
  "($0.getDefiningOp<ONNXAveragePoolOp>()))">;

def replaceONNXAveragePoolPattern : Pattern<
  (ONNXAveragePoolOp:$res $x, $_, $_, $_, $_, $_, $_, $_),
  [
    // Get attributes using shape helper
    (GetStrAttrPaddingtypeAveragePool:$padtype $res),
    (GetI64ArrayAttrKernelShapeAveragePool:$kernel_shape $res),
    (GetI64ArrayAttrStridesAveragePool:$strides $res),

    (ZHighUnstickOp
       (ZHighAvgPool2DOp
          (ZHighStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
          $kernel_shape,
          $strides,
          $padtype))
  ]
>;

//===----------------------------------------------------------------------===//
// MatMul
//===----------------------------------------------------------------------===//

def GetMatMulLayoutStringAttr : NativeCodeCall<
  "$_builder.getStringAttr(($0 == 2) ? LAYOUT_2D : LAYOUT_3DS)"
>;

def GetMatMulBiasLayoutStringAttr : NativeCodeCall<
  "$_builder.getStringAttr(((($0 == 3) && ($1 == 3)) || (($0 == 2) && ($1 == 3))) ? LAYOUT_2DS : LAYOUT_1D)"
>;

//===----------------------------------------------------------------------===//
// ONNXMatMulOp %X %Y = ZHighUnstickOp
//                          (ZHighMatMulOp
//                              (ZHighStickOp %X),
//                              (ZHighStickOp %Y),
//                              CreateNoneValue)
//===----------------------------------------------------------------------===//

def replaceONNXMatMulPattern : Pat<
  (ONNXMatMulOp:$res $x, $y),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (CreateNoneValue), (GetZeroI64Attr), (GetZeroI64Attr)))
>;

//===----------------------------------------------------------------------===//
// ONNXAddOp (ONNXMatMulOp %X %Y), %B =
//   ZHighUnstickOp
//       (ZHighMatMulOp
//           (ZHighStickOp %X),
//           (ZHighStickOp %Y),
//           (ZHighStickOp %B))
//===----------------------------------------------------------------------===//

def IsMatMulLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXMatMulOp>(" #
             "dyn_cast_or_null<ONNXMatMulOp>($0.getDefiningOp()))">,
  "MatMul is legal for zDNN"
>;

// Be careful, this check is very specific to '$0' of rank 2 and '$1' of rank 1.
def HaveSameLastDimR2R1: Constraint<
  CPred<"(!mlir::cast<ShapedType>($0.getType()).isDynamicDim(1))" #
        " && (!mlir::cast<ShapedType>($1.getType()).isDynamicDim(0))" #
        " && (mlir::cast<ShapedType>($0.getType()).getShape()[1]" #
              " == mlir::cast<ShapedType>($1.getType()).getShape()[0])">,
  "Have the same last dimension"
>;

// Only 1D bias is suitable for this transformation since only then
// the semantics of bias addition is the same for both ONNX and zDNN.
def replaceONNXMatMulAddPattern1 : Pat<
  // From Add $b, (MatMul $x, $y)
  (ONNXAddOp $b, (ONNXMatMulOp:$m $x, $y)),
  // To ZHighMatMulOp
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (ZHighStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
                                                         (GetRank $y)), (GetDefaultSaturation)),
        (GetZeroI64Attr), (GetZeroI64Attr))),
  [(IsMatMulLegalForZDNN $m), (HasRankOf<2> $y), (HasRankOf<1> $b),
   (HaveSameLastDimR2R1 $y, $b)], [],
  (addBenefit 0)
>;

def replaceONNXMatMulAddPattern2 : Pat<
  // From Add (MatMul $x, $y), $b
  (ONNXAddOp (ONNXMatMulOp:$m $x, $y), $b),
  // To ZHighMatMulOp
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (ZHighStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
                                                         (GetRank $y)), (GetDefaultSaturation)),
        (GetZeroI64Attr), (GetZeroI64Attr))),
  [(IsMatMulLegalForZDNN $m), (HasRankOf<2> $y), (HasRankOf<1> $b),
   (HaveSameLastDimR2R1 $y, $b)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// Replace onnx.add and onnx.matmul with bcast1 tensors with ZHighMatMul
//===----------------------------------------------------------------------===//

def replaceONNXMatMulAddPatternBcast1A : Pat<
  // From Add $b, (MatMul $x, $y)
  (ONNXAddOp $b, (ONNXMatMulOp:$m $x, $y)),
  // To ZHighMatMulOp
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (ZHighStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
                                                         (GetRank $y)), (GetDefaultSaturation)),
        (GetZeroI64Attr), (GetZeroI64Attr))),
  [(IsCompatibleWithNNPALevelArch15),(IsMatMulLegalForZDNN $m),
   (HasRankOf<2> $x), (HasRankOf<3> $y), (HasRankOf<2> $b)], [],
  (addBenefit 0)
>;

def replaceONNXMatMulAddPatternBcast1B : Pat<
  // From Add (MatMul $x, $y), $b
  (ONNXAddOp (ONNXMatMulOp:$m $x, $y), $b),
  // To ZHighMatMulOp
  (ZHighUnstickOp
     (ZHighMatMulOp
	(ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
	(ZHighStickOp $b, (GetMatMulBiasLayoutStringAttr (GetRank $x),
							 (GetRank $y)), (GetDefaultSaturation)),
	(GetZeroI64Attr), (GetZeroI64Attr))),
  [(IsCompatibleWithNNPALevelArch15),(IsMatMulLegalForZDNN $m),
   (HasRankOf<2> $x), (HasRankOf<3> $y), (HasRankOf<2> $b)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// Replace onnx.matmul when following onnx.transpose
//    with ZHighMatMul with either A or B transposed
//===----------------------------------------------------------------------===//

// TODO: This could likely be done in a cleaner way, such as comparing array vs array...
// Make sure the transpose permutation is the default as ZHighMatMul only supports default
def IsStandardTranspose : Constraint<
  CPred<"((mlir::cast<ArrayAttr>($0).size() == 2 ) " #
  " && (mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>($0)[0]).getInt() == 1)" #
  " && (mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>($0)[1]).getInt() == 0))" #
  " || ((mlir::cast<ArrayAttr>($0).size() == 3 ) " #
  " && (mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>($0)[0]).getInt() == 0)" #
  " && (mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>($0)[1]).getInt() == 2)" #
  " && (mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>($0)[2]).getInt() == 1))">
>;

def replaceONNXTransBMatMulPattern : Pat<
  (ONNXMatMulOp:$m $x,(ONNXTransposeOp $y,$perm)),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (CreateNoneValue), (GetI64ZeroAttr), (GetI64NAttr<1>)
      )
  ),
  [(IsCompatibleWithNNPALevelArch15),(IsMatMulLegalForZDNN $m), (IsStandardTranspose $perm)]
>;

def replaceONNXTransAMatMulPattern : Pat<
  (ONNXMatMulOp:$m (ONNXTransposeOp $x,$perm), $y),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (CreateNoneValue), (GetI64NAttr<1>), (GetI64ZeroAttr)
      )
  ),
  [(IsCompatibleWithNNPALevelArch15),(IsMatMulLegalForZDNN $m), (IsStandardTranspose $perm)]
>;

def replaceONNXTransABMatMulPattern : Pat<
  (ONNXMatMulOp:$m (ONNXTransposeOp $x,$perma), (ONNXTransposeOp $y,$permb)),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $x, (GetMatMulLayoutStringAttr (GetRank $x)), (GetDefaultSaturation)),
        (ZHighStickOp $y, (GetMatMulLayoutStringAttr (GetRank $y)), (GetDefaultSaturation)),
        (CreateNoneValue), (GetI64NAttr<1>), (GetI64NAttr<1>)
      )
  ),
  [(IsCompatibleWithNNPALevelArch15),(IsMatMulLegalForZDNN $m),
   (IsStandardTranspose $perma), (IsStandardTranspose $permb)]
>;

//===----------------------------------------------------------------------===//
// GEMM
//===----------------------------------------------------------------------===//
def IsTransposed: Constraint<CPred<"(mlir::cast<IntegerAttr>($_self).getSInt() == 1)">>;

def IsAorBTransposed: Constraint<
  CPred<"(mlir::cast<IntegerAttr>($0).getSInt() == 1) ||"
        "(mlir::cast<IntegerAttr>($1).getSInt() == 1)" >
>;

def Transpose2D: NativeCodeCall<
  "emitONNXTranspose($_loc, $_builder, $0, SmallVector<int64_t, 2>({1, 0}))">;

//===----------------------------------------------------------------------===//
// NNPA does not directly support cases with alpha != 1.0 or beta != 1.0 or bias's rank == 2.
// Since alpha and beta were legalized by the pass configuration, we check rank only.
//
// If bias's rank is 1, directly lower ONNX GEMM to ZHigh MatMul.
// ONNXGemmOp %A %B %C = ZHighUnstickOp
//                         (ZHighMatMulOp
//                            (ZHighStickOp %A),
//                            (ZHighStickOp %B),
//                            (ZHighStickOp %C)),
//
// If bias's rank is 2, lower ONNX GEMM to ZHigh MatMul and ONNX Add.
// ONNX Add could be lowered to ZHigh depending on the rule for ONNX Add.
// ONNXGemmOp %A %B %C = ONNXAdd (ZHighUnstickOp
//                                  (ZHighMatMulOp
//                                     (ZHighStickOp %A),
//                                     (ZHighStickOp %B),
//                                     (NoValue)),
//                                %C)
//===----------------------------------------------------------------------===//

def normalizeONNXGemmTransAPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB),
  (ONNXGemmOp (Transpose2D $a), $b, $c, $alpha, $beta, (GetZeroI64Attr), $transB),
  [(IsTransposed:$transA)], [],
  (addBenefit 1)
>;

def normalizeONNXGemmTransBPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB),
  (ONNXGemmOp $a, (Transpose2D $b), $c, $alpha, $beta, $transA, (GetZeroI64Attr)),
  [(IsTransposed:$transB)], [],
  (addBenefit 1)
>;


def replaceONNXGemmBiasNoneOr1DPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $_, $_, $_, $_),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $a, (_2DLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp $b, (_2DLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp $c, (_1DLayoutAttr), (GetDefaultSaturation)),
        (GetZeroI64Attr), (GetZeroI64Attr))),
  [(IsBiasNoneOr1D:$c)], [],
  (addBenefit 0)
>;

def replaceONNXGemmBias2DPattern : Pat<
  (ONNXGemmOp:$res $a, $b, $c, $_, $_, $_, $_),
  (ONNXAddOp
    (ZHighUnstickOp
       (ZHighMatMulOp
          (ZHighStickOp $a, (_2DLayoutAttr), (GetDefaultSaturation)),
          (ZHighStickOp $b, (_2DLayoutAttr), (GetDefaultSaturation)),
          (CreateNoneValue),
          (GetZeroI64Attr), (GetZeroI64Attr)),
       (returnType $res)),
    $c),
  [(HasRankOf<2> $c)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// Replace onnx.Gemm when transA or transB set to ZHighMatMulOp with transpose.
//
// ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB =
//   ZHighUnstickOp
//       (ZHighMatMulOp
//           (ZHighStickOp $a, (_2DLayoutAttr)),
//           (ZHighStickOp $b, (_2DLayoutAttr)),
//           (ZHighStickOp $c, (_1DLayoutAttr)),
//           $transA, $transB))
//===----------------------------------------------------------------------===//
def replaceONNXGemmTransPattern : Pat<
  (ONNXGemmOp $a, $b, $c, $alpha, $beta, $transA, $transB),
  (ZHighUnstickOp
     (ZHighMatMulOp
        (ZHighStickOp $a, (_2DLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp $b, (_2DLayoutAttr), (GetDefaultSaturation)),
        (ZHighStickOp $c, (_1DLayoutAttr), (GetDefaultSaturation)),
        $transA, $transB)),
  [(IsAorBTransposed $transA, $transB),
   (IsCompatibleWithNNPALevelArch15),
   (IsBiasNoneOr1D:$c)], [],
  (addBenefit 1)
>;

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

def GetLSTMGRUBiasSplitShape : NativeCodeCall<
  "getLSTMGRUBiasSplitShape($_loc, $_builder, $0)"
>;

class GetLSTMGRUZDNNWeightFromONNXWeight<string opName> : NativeCodeCall<
  "getLSTMGRUZDNNWeightFromONNXWeight($_loc, $_builder, $0,"
  "!strcmp(\"" # opName # "\", \"LSTM\"))">;

def GetLSTMGRUReturnAllStepsAttr : NativeCodeCall<
  "$_builder.getIntegerAttr($_builder.getIntegerType(64, true),"
  "(isNoneValue($0) ? 1 : -1))">;

def GetLSTMGRUGetY :
        NativeCodeCall<"getLSTMGRUGetY($_loc, $_builder, $0, $1)">;

def GetLSTMGRUGetYh :
        NativeCodeCall<"getLSTMGRUGetYh($_loc, $_builder, $0, $1, $2, $3, $4)">;

def GetLSTMGRUGetYWithSequenceLens :
        NativeCodeCall<"getLSTMGRUGetYWithSequenceLens($_loc, $_builder, $0, $1, $2, $3)">;

def GetLSTMGRUGetYhWithSequenceLens :
        NativeCodeCall<"getLSTMGRUGetYhWithSequenceLens($_loc, $_builder, $0, $1, $2, $3, $4, $5)">;

def GetLSTMGRUGetYc :
        NativeCodeCall<"getLSTMGRUGetYc($_loc, $_builder, $0, $1)">;

def EmitONNXSplitOp6Results :
        NativeCodeCall<"emitONNXSplitOp($_loc, $_builder, $0, $1, $2)", 6>;

def EmitONNXSplitOp8Results :
        NativeCodeCall<"emitONNXSplitOp($_loc, $_builder, $0, $1, $2)", 8>;

//===----------------------------------------------------------------------===//
// Y, Y_h, Y_c =
//      ONNXLSTMOp $x, $w, $r, $b, $initial_h, $sequence_lens,
//                 $initial_c, $p, $activation_alpha, $activation_beta,
//                 $activations, $clip, $direction, $hidden_size,
//                 $input_forget)
//===----------------------------------------------------------------------===//

// Pattern1 for the not-none bias($b) and not-none Y($res__0) case
def replaceONNXLSTMPattern1 : Pattern<
  (ONNXLSTMOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
    $initial_c, $p, $activation_alpha, $activation_beta,
    $activations, $clip, $direction, $hidden_size,
    $input_forget, $layout),
  [
   //
   // Pre-process to prepare input_bias and hidden_bias.
   //
   (EmitONNXSplitOp8Results:$splitB $b, (GetI64NAttr<1>), (GetLSTMGRUBiasSplitShape (GetShape $r))),
   (ZHighStickForLSTMOp:$input_bias $splitB__2, $splitB__0, $splitB__3, $splitB__1),
   (ZHighStickForLSTMOp:$hidden_bias $splitB__6, $splitB__4, $splitB__7, $splitB__5),

   //
   // Main-process to lower ONNXLSTMOp into ZHighLSTMOp.
   //
   (ZHighLSTMOp:$zhighLSTM
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (ZHighStickOp $initial_c, (_3DSLayoutAttr), (GetDefaultSaturation)),       // c0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $w),                           // input weights
      $input_bias,                                                               // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $r),                           // hidden weights
      $hidden_bias,                                                              // hidden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate three return values of ONNXLSTMOp
   // from two return values of ZHighLSTMOp(=$zHighLSTM).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighLSTM__0, (returnType $res__0)),
   (GetLSTMGRUGetY $unstick_HnOutput, $res__0),
   (GetLSTMGRUGetYh $unstick_HnOutput, $res__0, $res__1, $x, $direction),
   (GetLSTMGRUGetYc $zhighLSTM__1, $res__2),
  ],
  [(IsNotNoneType:$b), (IsNotNoneType:$res__0)]
>;

// Pattern2 for the none bias($b) and not-none Y($res__0) case
def replaceONNXLSTMPattern2 : Pattern<
  (ONNXLSTMOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $initial_c, $p, $activation_alpha, $activation_beta,
     $activations, $clip, $direction, $hidden_size,
     $input_forget, $layout),
  [
   //
   // Main-process to lower ONNXLSTMOp into ZHighLSTMOp.
   //
   (ZHighLSTMOp:$zhighLSTM
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (ZHighStickOp $initial_c, (_3DSLayoutAttr), (GetDefaultSaturation)),       // c0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $w),                           // input weights
      $b,                                                                        // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $r),                           // hidden weights
      $b,                                                                        // nontype hidden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate three return values of ONNXLSTMOp
   // from two return values of ZHighLSTMOp(=$zHighLSTM).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighLSTM__0, (returnType $res__0)),
   (GetLSTMGRUGetY $unstick_HnOutput, $res__0),
   (GetLSTMGRUGetYh $unstick_HnOutput, $res__0, $res__1, $x, $direction),
   (GetLSTMGRUGetYc $zhighLSTM__1, $res__2),
  ],
  [(IsNoneType:$b), (IsNotNoneType:$res__0)]
>;

// Pattern3 for the not-none bias($b) and none Y($res__0) case
def replaceONNXLSTMPattern3 : Pattern<
  (ONNXLSTMOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
    $initial_c, $p, $activation_alpha, $activation_beta,
    $activations, $clip, $direction, $hidden_size,
    $input_forget, $layout),
  [
   //
   // Pre-process to prepare input_bias and hidden_bias.
   //
   (EmitONNXSplitOp8Results:$splitB $b, (GetI64NAttr<1>), (GetLSTMGRUBiasSplitShape (GetShape $r))),
   (ZHighStickForLSTMOp:$input_bias $splitB__2, $splitB__0, $splitB__3, $splitB__1),
   (ZHighStickForLSTMOp:$hidden_bias $splitB__6, $splitB__4, $splitB__7, $splitB__5),

   //
   // Main-process to lower ONNXLSTMOp into ZHighLSTMOp.
   //
   (ZHighLSTMOp:$zhighLSTM
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (ZHighStickOp $initial_c, (_3DSLayoutAttr), (GetDefaultSaturation)),       // c0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $w),                           // input weights
      $input_bias,                                                               // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $r),                           // hidden weights
      $hidden_bias,                                                              // hidden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate three return values of ONNXLSTMOp
   // from two return values of ZHighLSTMOp(=$zHighLSTM).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighLSTM__0),
   (GetLSTMGRUGetY $unstick_HnOutput, $res__0),
   (GetLSTMGRUGetYh $unstick_HnOutput, $res__0, $res__1, $x, $direction),
   (GetLSTMGRUGetYc $zhighLSTM__1, $res__2),
  ],
  [(IsNotNoneType:$b), (IsNoneType:$res__0)]
>;

// Pattern4 for the none bias($b) and none Y($res__0) case
def replaceONNXLSTMPattern4 : Pattern<
  (ONNXLSTMOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $initial_c, $p, $activation_alpha, $activation_beta,
     $activations, $clip, $direction, $hidden_size,
     $input_forget, $layout),
  [
   //
   // Main-process to lower ONNXLSTMOp into ZHighLSTMOp.
   //
   (ZHighLSTMOp:$zhighLSTM
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (ZHighStickOp $initial_c, (_3DSLayoutAttr), (GetDefaultSaturation)),       // c0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $w),                           // input weights
      $b,                                                                        // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"LSTM"> $r),                           // hidden weights
      $b,                                                                        // nontype hidden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate three return values of ONNXLSTMOp
   // from two return values of ZHighLSTMOp(=$zHighLSTM).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighLSTM__0),
   (GetLSTMGRUGetY $unstick_HnOutput, $res__0),
   (GetLSTMGRUGetYh $unstick_HnOutput, $res__0, $res__1, $x, $direction),
   (GetLSTMGRUGetYc $zhighLSTM__1, $res__2),
  ],
  [(IsNoneType:$b), (IsNoneType:$res__0)]
>;

//===----------------------------------------------------------------------===//
// Y, Y_h =
//      ONNXGRUOp $x, $w, $r, $b, $initial_h, $sequence_lens, $initial_h,
//                $activation_alpha, $activation_beta, $activations, $clip,
//                $direction, $hidden_size, $linear_before_reset)
//===----------------------------------------------------------------------===//

// Pattern1 for the not-none bias($b) and not-none Y($res__0) case
def replaceONNXGRUPattern1 : Pattern<
  (ONNXGRUOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $activation_alpha, $activation_beta, $activations, $clip,
     $direction, $hidden_size, $layout, $linear_before_reset),
  [
   //
   // Pre-process to prepare input_bias and hidden_bias.
   //
   (EmitONNXSplitOp6Results:$splitB $b, (GetI64NAttr<1>),
      (GetLSTMGRUBiasSplitShape (GetShape $r))),
   (ZHighStickForGRUOp:$input_bias $splitB__0, $splitB__1, $splitB__2),
   (ZHighStickForGRUOp:$hidden_bias $splitB__3, $splitB__4, $splitB__5),

   //
   // Main-process to lower ONNXGRUOp into ZHighGRUOp.
   //
   (ZHighGRUOp:$zhighGRU
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $w),                            // input weights
      $input_bias,                                                               // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $r),                            // hidden weights
      $hidden_bias,                                                              // hiden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate two return values of ONNXGRUOp
   // from two return values of ZHighGRUOp(=$zHighGRU).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighGRU__0, (returnType $res__0)),
   (GetLSTMGRUGetYWithSequenceLens $unstick_HnOutput, $res__0, $sequence_lens, $initial_h),
   (GetLSTMGRUGetYhWithSequenceLens $unstick_HnOutput, $res__0, $res__1, $x, $direction, $sequence_lens),
  ],
  [(IsNotNoneType:$b), (IsNotNoneType:$res__0)]
>;

// Pattern2 for the none bias($b) and not-none Y($res__0) case
def replaceONNXGRUPattern2 : Pattern<
  (ONNXGRUOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $activation_alpha, $activation_beta, $activations, $clip,
     $direction, $hidden_size, $layout, $linear_before_reset),
  [
   //
   // Main-process to lower ONNXGRUOp into ZHighGRUOp.
   //
   (ZHighGRUOp:$zhighGRU
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),                // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),        // h0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $w),                             // input weights
      $b,                                                                         // nontype input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $r),                             // hidden weights
      $b,                                                                         // nontype hidden bias
      $hidden_size,                                                               // hidden_size
      $direction,                                                                 // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                           // return_all_steps

   //
   // Post-process to generate two return values of ONNXGRUOp
   // from two return values of ZHighGRUOp(=$zHighGRU).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighGRU__0, (returnType $res__0)),
   (GetLSTMGRUGetYWithSequenceLens $unstick_HnOutput, $res__0, $sequence_lens, $initial_h),
   (GetLSTMGRUGetYhWithSequenceLens $unstick_HnOutput, $res__0, $res__1, $x, $direction, $sequence_lens),
  ],
  [(IsNoneType:$b), (IsNotNoneType:$res__0)]
>;

// Pattern3 for the not-none bias($b) and none Y($res__0) case
def replaceONNXGRUPattern3 : Pattern<
  (ONNXGRUOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $activation_alpha, $activation_beta, $activations, $clip,
     $direction, $hidden_size, $layout, $linear_before_reset),
  [
   //
   // Pre-process to prepare input_bias and hidden_bias.
   //
   (EmitONNXSplitOp6Results:$splitB $b, (GetI64NAttr<1>),
      (GetLSTMGRUBiasSplitShape (GetShape $r))),
   (ZHighStickForGRUOp:$input_bias $splitB__0, $splitB__1, $splitB__2),
   (ZHighStickForGRUOp:$hidden_bias $splitB__3, $splitB__4, $splitB__5),

   //
   // Main-process to lower ONNXGRUOp into ZHighGRUOp.
   //
   (ZHighGRUOp:$zhighGRU
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),               // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),       // h0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $w),                            // input weights
      $input_bias,                                                               // input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $r),                            // hidden weights
      $hidden_bias,                                                              // hiden bias
      $hidden_size,                                                              // hidden_size
      $direction,                                                                // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                          // return_all_steps

   //
   // Post-process to generate two return values of ONNXGRUOp
   // from two return values of ZHighGRUOp(=$zHighGRU).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighGRU__0),
   (GetLSTMGRUGetYWithSequenceLens $unstick_HnOutput, $res__0, $sequence_lens, $initial_h),
   (GetLSTMGRUGetYhWithSequenceLens $unstick_HnOutput, $res__0, $res__1, $x, $direction, $sequence_lens),
  ],
  [(IsNotNoneType:$b), (IsNoneType:$res__0)]
>;

// Pattern4 for the none bias($b) and none Y($res__0) case
def replaceONNXGRUPattern4 : Pattern<
  (ONNXGRUOp:$res $x, $w, $r, $b, $sequence_lens, $initial_h,
     $activation_alpha, $activation_beta, $activations, $clip,
     $direction, $hidden_size, $layout, $linear_before_reset),
  [
   //
   // Main-process to lower ONNXGRUOp into ZHighGRUOp.
   //
   (ZHighGRUOp:$zhighGRU
      (ZHighStickOp $x, (_3DSLayoutAttr), (GetDefaultSaturation)),                // input
      (ZHighStickOp $initial_h, (_3DSLayoutAttr), (GetDefaultSaturation)),        // h0
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $w),                             // input weights
      $b,                                                                         // nontype input bias
      (GetLSTMGRUZDNNWeightFromONNXWeight<"GRU"> $r),                             // hidden weights
      $b,                                                                         // nontype hidden bias
      $hidden_size,                                                               // hidden_size
      $direction,                                                                 // direction
      (GetLSTMGRUReturnAllStepsAttr $res__0, $res__1)),                           // return_all_steps

   //
   // Post-process to generate three return values of ONNXGRUOp
   // from two return values of ZHighGRUOp(=$zHighGRU).
   //
   (ZHighUnstickOp:$unstick_HnOutput $zhighGRU__0),
   (GetLSTMGRUGetYWithSequenceLens $unstick_HnOutput, $res__0, $sequence_lens, $initial_h),
   (GetLSTMGRUGetYhWithSequenceLens $unstick_HnOutput, $res__0, $res__1, $x, $direction, $sequence_lens),
  ],
  [(IsNoneType:$b), (IsNoneType:$res__0)]
>;

//===----------------------------------------------------------------------===//
// Rewrite
//
// ONNXConvOp %X, %W, %B
//
// to
//
// (ZHighUnstickOp
//    (ZHighConvOp
//      (ZHighStickOp %X, "NHWC"),
//      (ZHighStickOp (NCHWtoHWCK %W), "HWCK"),
//      (ZHighStickOp %B, "1D"),
//      kernel_shape,
//      strides,
//      GetPaddingType,
//      ACT_NONE)))
//
//===----------------------------------------------------------------------===//

def GetStrAttrPaddingtypeConv: NativeCodeCall<
  "$_builder.getStringAttr(getStrPaddingType<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def GetI64ArrayAttrKernelShapeConv: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayKernelShape<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def GetI64ArrayAttrStridesConv: NativeCodeCall<
  "$_builder.getI64ArrayAttr(getArrayStrides<ONNXConvOp, ONNXConvOpAdaptor,"
  "ONNXConvOpShapeHelper>($0.getDefiningOp<ONNXConvOp>()))">;

def replaceONNXConv2DPattern : Pattern<
  (ONNXConvOp:$res $x, $w, $b, $_, $_, $_, $_, $_, $_),
  [
   // Get attributes using shape helper
   (GetStrAttrPaddingtypeConv:$padtype $res),
   (GetI64ArrayAttrKernelShapeConv:$kernel_shape $res),
   (GetI64ArrayAttrStridesConv:$strides $res),

   (ZHighUnstickOp
    (ZHighConv2DOp
       (ZHighStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
       (ZHighStickOp (NCHWtoHWCK $w), (HWCKLayoutAttr), (GetDefaultSaturation)),
       (ZHighStickOp $b, (_1DLayoutAttr), (GetDefaultSaturation)),
       $kernel_shape,
       $strides,
       $padtype,
       (ACT_NONEAttr)))
  ]
>;

//===----------------------------------------------------------------------===//
// Rewrite
//
// ONNXReluOp (ONNXConvOp %X, %W, %B)
//
// to
//
// (ZHighUnstickOp
//    (ZHighConvOp
//      (ZHighStickOp %X, "NHWC"),
//      (ZHighStickOp (NCHWtoHWCK %W), "HWCK"),
//      (ZHighStickOp %B, "1D"),
//      kernel_shape,
//      strides,
//      GetPaddingType,
//      ACT_RELU)))
//
//===----------------------------------------------------------------------===//

def IsConv2DLegalForZDNN: Constraint<
  CPred<"isSuitableForZDNN<ONNXConvOp>(" #
             "dyn_cast_or_null<ONNXConvOp>($0.getDefiningOp()))">,
  "Conv is legal for zDNN"
>;

def replaceONNXReluConvPattern : Pattern<
  (ONNXReluOp (ONNXConvOp:$res $x, $w, $b, $_, $_, $_, $_, $_, $_)),
  [
   // Get attributes using shape helper
   (GetStrAttrPaddingtypeConv:$padtype $res),
   (GetI64ArrayAttrKernelShapeConv:$kernel_shape $res),
   (GetI64ArrayAttrStridesConv:$strides $res),

   (ZHighUnstickOp
    (ZHighConv2DOp
       (ZHighStickOp $x, (NHWCLayoutAttr), (GetDefaultSaturation)),
       (ZHighStickOp (NCHWtoHWCK $w), (HWCKLayoutAttr), (GetDefaultSaturation)),
       (ZHighStickOp $b, (_1DLayoutAttr), (GetDefaultSaturation)),
       $kernel_shape,
       $strides,
       $padtype,
       (ACT_RELUAttr)))
  ],
  [(IsConv2DLegalForZDNN $res)], [],
  (addBenefit 0)
>;

#endif // ONNX_TO_ZHIGH
