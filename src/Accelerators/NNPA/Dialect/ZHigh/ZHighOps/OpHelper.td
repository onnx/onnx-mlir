// SPDX-License-Identifier: Apache-2.0

//===------- OpHelper.td - Common Pattern Match for ZHigh -----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef OP_HELPER
#define OP_HELPER

#ifndef OP_BASE
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

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

def IsNoneType : Constraint<CPred<"mlir::isa<NoneType>(($_self).getType())">>;

// Create an ONNX Shape Op with type
def CreateShapeOp: NativeCodeCall<
  "::onnx_mlir::OnnxBuilder($_builder, $_loc).shape($0);"
>;

def Create3DShapePermuteRightmostOp: NativeCodeCall<
  "::onnx_mlir::OnnxBuilder($_builder, $_loc).shape($0, {0, 2, 1});"
>;

// Get a type for a tensor that stores the shape of another tensor.
def GetShapeTypeOf: NativeCodeCall<
  "RankedTensorType::get({mlir::cast<ShapedType>($0.getType()).getRank()}, $_builder.getIntegerType(64))"
>;

def GetLayout : NativeCodeCall<
  "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
      "::onnx_mlir::zhigh::getZTensorLayout($0.getType()))"
>;

def NoOneIsOfNHWCLayout: Constraint<
  CPred<"!::onnx_mlir::zhigh::oneIsOfLayout($0.getType(), $1.getType(), "
        "::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC)">,
  "No tensor is of NHWC layout"
>;

def NoOneIsOf1DLayout: Constraint<
  CPred<"!::onnx_mlir::zhigh::oneIsOfLayout($0.getType(), $1.getType(), "
        "::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::_1D)">,
  "No tensor is of 1D layout"
>;

def NoOneIsOf2DSLayout: Constraint<
  CPred<"!::onnx_mlir::zhigh::oneIsOfLayout($0.getType(), $1.getType(), "
        "::onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::_2DS)">,
  "No tensor is of 2DS layout"
>;

def GetEncodingAttr : NativeCodeCall<
  "::onnx_mlir::zhigh::getZTensorEncoding($0.getType())"
>;

def GetMinusBcastConst : NativeCodeCall<
  "::onnx_mlir::zhigh::getMinusBcastConst($_builder, $_loc, $0, $1)"
>;

class GetConstantOfType<string v> : NativeCodeCall<
  "::onnx_mlir::zhigh::getConstantOfType($_builder, $_loc, $0.getType(), " # v # ")"
>;

def IsStaticShapeTensor:
  Constraint<
    CPred<
      "mlir::cast<::mlir::ShapedType>($0.getType()).hasStaticShape()">,
    "is a tensor of static shape">;

def IsPlusConstantFloat : Constraint<
  CPred<"($0.getValueAsDouble() > 0.0)">,
  "Is the float attribute is greater than 0"
>;

def TensorHas2DLayout: Constraint<
  CPred<"::onnx_mlir::is2DLayout("
        "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
        "::onnx_mlir::zhigh::getZTensorLayout($0.getType())))">,
  "ZTensor has 2D layout"
>;

def TensorHas3DSLayout: Constraint<
  CPred<"::onnx_mlir::is3DSLayout("
        "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
        "::onnx_mlir::zhigh::getZTensorLayout($0.getType())))">,
  "ZTensor has 3DS layout"
>;

def TensorHas4DSLayout: Constraint<
  CPred<"::onnx_mlir::is4DSLayout("
        "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
        "::onnx_mlir::zhigh::getZTensorLayout($0.getType())))">,
  "ZTensor has 4DS layout"
>;

def TensorHasNHWCLayout: Constraint<
  CPred<"::onnx_mlir::isNHWCLayout("
        "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
        "::onnx_mlir::zhigh::getZTensorLayout($0.getType())))">,
  "ZTensor has NHWC layout"
>;

def TensorHasNoNHWCLayout: Constraint<
  CPred<"!::onnx_mlir::isNHWCLayout("
        "::onnx_mlir::zhigh::convertZTensorDataLayoutToStringAttr($_builder, "
        "::onnx_mlir::zhigh::getZTensorLayout($0.getType())))">,
  "ZTensor has no NHWC layout"
>;

def HasNNPAUse: Constraint<
  CPred<"::onnx_mlir::zhigh::hasNNPAUse($0)">,
  "The value has NNPA users (or is consumed by an NNPA op)"
>;

def Is2DLayout : Constraint<
  CPred<"::onnx_mlir::is2DLayout($0)">,
  "Is 2D layout"
>;

def Is3DSLayout : Constraint<
  CPred<"::onnx_mlir::is3DSLayout($0)">,
  "Is 3DS layout"
>;

def IsNotNHWCLayout : Constraint<
  CPred<"!::onnx_mlir::isNHWCLayout($0)">,
  "Is not NHWC layout"
>;

/// Compose two affine maps.
def GetComposedMap : NativeCodeCall<
 "mlir::AffineMapAttr::get($0.getValue().compose($1.getValue()))"
 >;

/// Check if ONNXReshapeOp is reshaping 2D to 4D by tiling each input dimension.
def IsTiling2DTo4D : Constraint<
  CPred<"::onnx_mlir::zhigh::isTiling2DTo4D($0)">,
  "Is tiling by ONNReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling leftmost input dimension.
def IsLeftmostTiling3DTo4D : Constraint<
  CPred<"::onnx_mlir::zhigh::isLeftmostTiling3DTo4D($0)">,
  "Is leftmost tiling by ONNReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling rightmost input dimension.
def IsRightmostTiling3DTo4DBy64 : Constraint<
  CPred<"::onnx_mlir::zhigh::isRightmostTiling3DTo4D($0, 64)">,
  "Is rightmost tiling of size 64 by ONNReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 4D to 2D by collapsing the first two input dimensions.
def IsCollapsing4DTo2D : Constraint<
  CPred<"::onnx_mlir::zhigh::isCollapsing4DTo2D($0)">,
  "Is collapsing by ONNXReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 4D to 3D by collapsing the first two input dimensions.
def IsLeftmostCollapsing4DTo3D : Constraint<
  CPred<"::onnx_mlir::zhigh::isLeftmostCollapsing4DTo3D($0)">,
  "Is leftmost collapsing by ONNXReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 4D to 3D by collapsing the last two input dimensions.
def IsRightmostCollapsing4DTo3D : Constraint<
  CPred<"::onnx_mlir::zhigh::isRightmostCollapsing4DTo3D($0)">,
  "Is rightmost collapsing by ONNXReshapeOp"
>;

def GetResultType : NativeCodeCall<
  "$0.getType()"
>;

def GetTiling2DTo4DMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getTiling2DTo4DMap($_builder, $0)"
>;

def GetLeftmostTiling3DTo4DMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getLeftmostTiling3DTo4DMap($_builder, $0)"
>;

def GetCollapsing4DTo2DMap: NativeCodeCall<
  "::onnx_mlir::zhigh::getCollapsing4DTo2DMap($_builder, $0)"
>;

def GetLeftmostCollapsing4DTo3DMap: NativeCodeCall<
  "::onnx_mlir::zhigh::getLeftmostCollapsing4DTo3DMap($_builder, $0)"
>;

def GetTransposeMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getTransposeMap($_builder, $0)"
>;

def Is4DTransposePermutationEqualTo0213 : Constraint<
  CPred<"::onnx_mlir::zhigh::isTransposePermutationEqualTo($0, {0, 2, 1, 3})">,
  "Is 4D Transpose with pattern (0, 2, 1, 3)"
>;

class IsShapeDimMultipleOf32<int index> : Constraint<
  CPred<"::onnx_mlir::zhigh::isShapeDimMultipleOf($0, " # index # ", 32)">,
  "The operand shape at given index is a multiple of 32"
>;

class IsShapeDimMultipleOf64<int index> : Constraint<
  CPred<"::onnx_mlir::zhigh::isShapeDimMultipleOf($0, " # index # ", 64)">,
  "The operand shape at given index is a multiple of 64"
>;

def IsIdentityAffineMap : Constraint<
  CPred<"$_self.isIdentity()">,
  "Is identity AffineMap"
>;

def GetAxisNHWC : NativeCodeCall<
  "::onnx_mlir::zhigh::getAxisNHWC($0)"
>;

def NoneIntegerAttr: NativeCodeCall<"IntegerAttr()">;

def GetDefaultSaturation : NativeCodeCall<
  "::onnx_mlir::zhigh::getDefaultSaturation($_builder)"
>;

class IsConstOf<int v>: Constraint<
  CPred<"onnx_mlir::isConstOf($0, " # v # ")">,
  "Value is a s scalar constant of v"
 >;

 // Create an array tensor to contain three dimensions of layout 3DS.
 // The tensor is created from 4DS's shape by removing the value 1 at axis 1.
 // e.g. 4DS tensor: tensor<3, 1, 4, 5>,
 // this function returns a tensor: tensor<3xi64> = [3, 4, 5]
 def Create3DSShapeFrom4DS: NativeCodeCall<
  "::onnx_mlir::OnnxBuilder($_builder, $_loc).shape($0, {0, 2, 3});"
>;

def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetZeroI64Attr: NativeCodeCall<
  "IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, /*isSigned=*/true))"
>;

def GetOneI64Attr: NativeCodeCall<
  "IntegerAttr::get($_builder.getIntegerType(64, /*isSigned=*/true), APInt(64, 1, /*isSigned=*/true))"
>;

class IsInt64NAttr<int n> : Constraint<
  CPred<"$0.getValue().getSExtValue() == " # n>,
  "The signed extended int64 attribute equal to N"
>;

def IsCompatibleWithNNPALevelArch14: Constraint<
  CPred<"isCompatibleWithNNPALevel(NNPALevel::M14)">,
  "Input level is compatible with NNPA level"
>;

def IsCompatibleWithNNPALevelArch15: Constraint<
  CPred<"isCompatibleWithNNPALevel(NNPALevel::M15)">,
  "Input level is compatible with NNPA level"
>;

#endif // OP_HELPER
