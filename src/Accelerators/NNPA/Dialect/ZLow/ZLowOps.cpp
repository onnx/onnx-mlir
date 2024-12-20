/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZLow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

//===----------------------------------------------------------------------===//
// ZLowDialect
//===----------------------------------------------------------------------===//

void ZLowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"
      >();
}

void ZLowAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowLogOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowExpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowTanhOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSoftmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMatMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getHnOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getCfOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getH0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getC0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getHnOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getH0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickForLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getFGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getCGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOGateMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickForGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getZGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHGateMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowUnstickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowConv2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputKernelMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowAvgPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMeanReduce2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowBatchNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}
//===----------------------------------------------------------------------===//
// ZLowOps methods
//===----------------------------------------------------------------------===//

LogicalResult ZLowQuantizedStickOp::verify() {
  ZLowQuantizedStickOp::Adaptor operandAdaptor(*this);
  Value recScale = operandAdaptor.getRecScale();
  Value offset = operandAdaptor.getOffset();
  Value output = operandAdaptor.getOut();
  auto outputType = llvm::dyn_cast<MemRefType>(output.getType());
  if (!outputType)
    return failure();

  // Verify quantized type.
  StringRef quantizedType = getQType();
  if (!(quantizedType.equals_insensitive("dlfloat16") ||
          quantizedType.equals_insensitive("int8") ||
          quantizedType.equals_insensitive("weights")))
    return emitOpError("q_type must be one of dlfloat16, int8, and weights");

  // Verify element type of the output.
  // TODO: should we have a more stricted contraint, e.g. signed integer?
  Type elementType = outputType.getElementType();
  if (quantizedType.equals_insensitive("dfloat16") && !elementType.isF16())
    return emitOpError("q_type and element type mismatched");
  if (quantizedType.equals_insensitive("int8") && !elementType.isInteger(8))
    return emitOpError("q_type and element type mismatched");
  if (quantizedType.equals_insensitive("weights") && !elementType.isInteger(8))
    return emitOpError("q_type and element type mismatched");

  // Verify recScale and offset.
  if (auto ty = llvm::dyn_cast<MemRefType>(recScale.getType())) {
    if (!ty.getElementType().isF32())
      return emitOpError("recScale must be f32");
  }
  if (auto ty = llvm::dyn_cast<MemRefType>(offset.getType())) {
    if (!ty.getElementType().isF32())
      return emitOpError("offset must be f32");
  }

  return success();
}

} // namespace zlow
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowDialect.cpp.inc"
