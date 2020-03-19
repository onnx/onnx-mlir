//===---------- Passes.hpp - ONNX MLIR Passes Definition ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX MLIR.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<Pass> createDecomposeONNXToONNXPass();

std::unique_ptr<Pass> createShapeInferencePass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<Pass> createLowerToKrnlPass();

/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<Pass> createLowerKrnlPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<Pass> createKrnlLowerToLLVMPass();

}  // end namespace mlir
