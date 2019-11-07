//===- passes.hpp - ONNF Passes Definition --------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNF.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;

std::unique_ptr<Pass> createShapeInferencePass();

// TODO: Add pass for lowering to kernel IR.

// TODO: Add pass for lowering to LLVM IR.

}  // end namespace mlir
