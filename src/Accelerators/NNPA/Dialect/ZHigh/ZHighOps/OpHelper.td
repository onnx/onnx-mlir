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
  CPred<"!::onnx_mlir::zhigh::oneIsOfNHWCLayout($0.getType(), $1.getType())">,
  "No tensor is of NHWC layout"
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

#endif // OP_HELPER 
