/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ EntryPoint.cpp - ONNX Operations ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect EntryPoint operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void ONNXEntryPointOp::build(
    OpBuilder &builder, OperationState &state, func::FuncOp function) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
      SymbolRefAttr::get(function));
}

ONNXEntryPointOp ONNXEntryPointOp::create(
    Location location, func::FuncOp &func) {
  OperationState state(location, "onnx.EntryPoint");
  OpBuilder builder(location->getContext());
  ONNXEntryPointOp::build(builder, state, func);
  Operation *op = Operation::create(state);
  auto onnxEntryOp = llvm::cast<ONNXEntryPointOp>(op);
  return onnxEntryOp;
}
