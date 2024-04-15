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
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowLogOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowExpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowTanhOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowSigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowSoftmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getWorkArea(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMatMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getY(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getBias(),
      SideEffects::DefaultResource::get());
}

void ZLowLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getHnOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), getCfOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getH0(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getC0(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputWeights(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputBias(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getHiddenWeights(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getHiddenBias(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getWorkArea(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getHnOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getH0(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputWeights(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputBias(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getHiddenWeights(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getHiddenBias(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getWorkArea(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowStickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
}

void ZLowStickForLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getFGate(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getIGate(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getCGate(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getOGate(),
      SideEffects::DefaultResource::get());
}

void ZLowStickForGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getZGate(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getRGate(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getHGate(),
      SideEffects::DefaultResource::get());
}

void ZLowUnstickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOut(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getX(), SideEffects::DefaultResource::get());
}

void ZLowConv2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputKernel(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInputBias(),
      SideEffects::DefaultResource::get());
}

void ZLowAvgPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowMeanReduce2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

void ZLowBatchNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getA(), SideEffects::DefaultResource::get());
  effects.emplace_back(
      MemoryEffects::Read::get(), getB(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), getShape(),
      SideEffects::DefaultResource::get());
}

} // namespace zlow
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowDialect.cpp.inc"
