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

std::unique_ptr<Pass> createConstPropONNXToONNXPass();

/// Pass for promoting constant operands to attributes.
std::unique_ptr<Pass> createAttributePromotionPass();

/// Pass for eliding the values of constant operations.
std::unique_ptr<Pass> createElideConstantValuePass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<Pass> createLowerToKrnlPass();

/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<Pass> createLowerKrnlPass();

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<Pass> createElideConstGlobalValuePass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<Pass> createKrnlLowerToLLVMPass();

} // end namespace mlir
