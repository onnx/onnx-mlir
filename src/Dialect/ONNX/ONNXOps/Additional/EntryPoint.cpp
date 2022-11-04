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

void ONNXEntryPointOp::build(mlir::OpBuilder &builder,
    mlir::OperationState &state, mlir::func::FuncOp function) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
      SymbolRefAttr::get(function));
}

ONNXEntryPointOp ONNXEntryPointOp::create(
    mlir::Location location, mlir::func::FuncOp &func) {
  mlir::OperationState state(location, "onnx.EntryPoint");
  OpBuilder builder(location->getContext());
  mlir::ONNXEntryPointOp::build(builder, state, func);
  Operation *op = mlir::Operation::create(state);
  auto onnxEntryOp = llvm::cast<mlir::ONNXEntryPointOp>(op);
  return onnxEntryOp;
}
