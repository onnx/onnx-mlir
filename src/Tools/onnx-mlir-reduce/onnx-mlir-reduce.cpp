/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- onnx-mlir-reduce.cpp - Test case reduction tool -----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the ONNX MLIR reducer tool, which can be used to reduce
// a failing MLIR test case. The tool outputs the most reduced test case variant
// after executing the reduction passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

static void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::KrnlDialect>();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerDialects(registry);

  // Register the DialectReductionPatternInterface if any.

  MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
