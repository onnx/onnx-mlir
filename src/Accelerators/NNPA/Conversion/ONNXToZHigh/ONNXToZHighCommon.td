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

#ifndef ONNX_TO_ZHIGH_COMMON
#define ONNX_TO_ZHIGH_COMMON

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNXOps.td"
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

class GetStringAttr<string s> :
	NativeCodeCall<"$_builder.getStringAttr(\"" # s # "\")">;

def NCHWtoNHWC : NativeCodeCall<
  "emitONNXTranspose($_loc, $_builder, $0,"
  "                  SmallVector<int64_t, 4>({0, 2, 3, 1}))">;

def NCHWtoHWCK : NativeCodeCall<
  "emitONNXTranspose($_loc, $_builder, $0,"
  "                  SmallVector<int64_t, 4>({2, 3, 1, 0}))">;

def NHWCtoNCHW: NativeCodeCall<
  "emitONNXTransposeWithType($_loc, $_builder, $0, $1,"
  "                          SmallVector<int64_t, 4>({0, 3, 1, 2}))">;

def _1DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_1D)">;
def _2DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_2D)">;
def _3DLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_3D)">;
def _3DSLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_3DS)">;
def NHWCLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_NHWC)">;
def NCHWLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_NCHW)">;
def HWCKLayoutAttr: NativeCodeCall<"$_builder.getStringAttr(LAYOUT_HWCK)">;
def NoneLayoutAttr: NativeCodeCall<"StringAttr()">;

def ACT_NONEAttr: NativeCodeCall<"$_builder.getStringAttr(\"ACT_NONE\")">;
def ACT_LOGAttr: NativeCodeCall<"$_builder.getStringAttr(\"ACT_LOG\")">;
def ACT_RELUAttr: NativeCodeCall<"$_builder.getStringAttr(\"ACT_RELU\")">;

def GetTypeOf : NativeCodeCall<"$0.getType()" >;

#endif // ONNX_TO_ZHIGH_COMMON
