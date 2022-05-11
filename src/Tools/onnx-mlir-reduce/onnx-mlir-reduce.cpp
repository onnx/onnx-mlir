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
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

static void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::KrnlOpsDialect>();

  // Initialize and register dialects used by accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    if (accel->isActive())
      accel->registerDialects(registry);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerDialects(registry);

  // Register the DialectReductionPatternInterface if any.

  MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
