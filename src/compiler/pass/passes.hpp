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

/// Add pass for lowering to Krnl IR.
std::unique_ptr<Pass> createLowerToKrnlPass();

/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<Pass> createLowerKrnlPass();

}  // end namespace mlir
