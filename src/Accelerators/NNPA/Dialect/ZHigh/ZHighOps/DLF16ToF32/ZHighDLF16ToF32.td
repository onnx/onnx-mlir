// SPDX-License-Identifier: Apache-2.0

//===------- ZHighDLF16ToF32.td - Pattern Match for ZHighUnstick ----------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef DLF16_TO_F32_TD
#define DLF16_TO_F32_TD

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

def GetTypeInDLF16: NativeCodeCall<
  "RankedTensorType::get(mlir::cast<ShapedType>($0.getType()).getShape(), $_builder.getF16Type())"
>;

//===----------------------------------------------------------------------===//
// DRR patterns
//===----------------------------------------------------------------------===//

// zhigh.DLF16ToF32 (zhigh.F32ToDLF16(%X)) = %X
def ConversionRemovalPattern : Pat<
  (ZHighDLF16ToF32Op (ZHighF32ToDLF16Op $arg, $saturation)),
  (replaceWithValue $arg)
>;

// Delay zhigh.DLF16ToF32 as long as possible, and finally it can be cancelled
// when it meets zhigh.F32ToDLF16.
def DelayDLF16ToF32ViaReshapePattern : Pat<
  (ONNXReshapeOp:$res (ZHighDLF16ToF32Op $arg), $shape, $allowZero),
  (ZHighDLF16ToF32Op (ONNXReshapeOp $arg, $shape, $allowZero,
                                    (returnType (GetTypeInDLF16 $res))))
>;

def DelayDLF16ToF32ViaTransposePattern : Pat<
  (ONNXTransposeOp:$res (ZHighDLF16ToF32Op $arg), $perm),
  (ZHighDLF16ToF32Op (ONNXTransposeOp $arg, $perm,
                                      (returnType (GetTypeInDLF16 $res))))
>;

def DelayDLF16ToF32ViaSqueezePattern : Pat<
  (ONNXSqueezeOp:$res (ZHighDLF16ToF32Op $arg), $axes),
  (ZHighDLF16ToF32Op (ONNXSqueezeOp $arg, $axes,
                                    (returnType (GetTypeInDLF16 $res))))
>;

def DelayDLF16ToF32ViaUnsqueezePattern : Pat<
  (ONNXUnsqueezeOp:$res (ZHighDLF16ToF32Op $arg), $axes),
  (ZHighDLF16ToF32Op (ONNXUnsqueezeOp $arg, $axes,
                                      (returnType (GetTypeInDLF16 $res))))
>;

// onnx.Dim (zhigh.DLF16ToF32(%X)) = onnx.Dim %X
def DimDLF16ToF32RemovalPattern : Pat<
  (ONNXDimOp (ZHighDLF16ToF32Op $arg), $axis),
  (ONNXDimOp $arg, $axis)
>;

#endif // DLF16_TO_F32_TD
