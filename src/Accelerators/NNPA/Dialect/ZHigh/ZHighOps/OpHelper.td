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

def IsNoneType : Constraint<CPred<"(($_self).getType().isa<NoneType>())">>;

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
      "$0.getType().cast<::mlir::ShapedType>().hasStaticShape()">,
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

def Is2DLayout : Constraint<
  CPred<"::onnx_mlir::is2DLayout($0)">,
  "Is 2D layout"
>;

def Is3DSLayout : Constraint<
  CPred<"::onnx_mlir::is3DSLayout($0)">,
  "Is 3DS layout"
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

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling each input dimension.
def IsTiling3DTo4D : Constraint<
  CPred<"::onnx_mlir::zhigh::isTiling3DTo4D($0)">,
  "Is tiling by ONNReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 4D to 2D by collapsing the first two input dimensions.
def IsCollapsing4DTo2D : Constraint<
  CPred<"::onnx_mlir::zhigh::isCollapsing4DTo2D($0)">,
  "Is collapsing by ONNXReshapeOp"
>;

/// Check if ONNXReshapeOp is reshaping 4D to 3D by collapsing the first two input dimensions.
def IsCollapsing4DTo3D : Constraint<
  CPred<"::onnx_mlir::zhigh::isCollapsing4DTo3D($0)">,
  "Is collapsing by ONNXReshapeOp"
>;

def GetResultType : NativeCodeCall<
  "$0.getType()"
>;

def GetTiling2DTo4DMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getTiling2DTo4DMap($_builder, $0)"
>;

def GetTiling3DTo4DMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getTiling3DTo4DMap($_builder, $0)"
>;

def GetCollapsing4DTo2DMap: NativeCodeCall<
  "::onnx_mlir::zhigh::getCollapsing4DTo2DMap($_builder, $0)"
>;

def GetCollapsing4DTo3DMap: NativeCodeCall<
  "::onnx_mlir::zhigh::getCollapsing4DTo3DMap($_builder, $0)"
>;

def GetTransposeMap : NativeCodeCall<
  "::onnx_mlir::zhigh::getTransposeMap($_builder, $0)"
>;

def IsIdentityAffineMap : Constraint<
  CPred<"$_self.isIdentity()">,
  "Is identity AffineMap"
>;

def GetAxisNHWC : NativeCodeCall<
  "::onnx_mlir::zhigh::getAxisNHWC($0)"
>;

#endif // OP_HELPER 
