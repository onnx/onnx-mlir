/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CUDAAccelerator.cpp ------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// CUDA accelerator implementation.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/CUDA/CUDAAccelerator.hpp"
#include "src/Accelerators/CUDA/CUDAConversionPass.hpp"
#include "src/Accelerators/CUDA/CUDAOptimizePass.hpp"
#include "src/Accelerators/CUDA/CUDARegistration.hpp"

namespace onnx_mlir {
namespace accel {

// =============================================================================
// CUDAAccelerator class.
// =============================================================================

class CUDAAccelerator : public Accelerator {
public:
  CUDAAccelerator();
  ~CUDAAccelerator() override = default;

  // Accelerator.
  Kind getKind() const override { return Kind::CUDA; }
  std::string_view getName() const override { return "cuda"; }
  void registerPasses() override;
  void initialize() override;
  void configure(llvm::StringRef config) override;
};

CUDAAccelerator::CUDAAccelerator() = default;

void CUDAAccelerator::registerPasses() {
  // Register CUDA conversion pass.
  registerPass<CUDAConversionPass>("cuda-conv", "cuda-conversion-pass");
  // Register CUDA optimize pass.
  registerPass<CUDAOptimizePass>("cuda-opt", "cuda-optimize-pass");
}

void CUDAAccelerator::initialize() {
  // Initialize CUDA registration.
  initializeCUDARegistration();
}

void CUDAAccelerator::configure(llvm::StringRef config) {
  // Configure CUDA.
  // TODO: Add configuration logic.
}

// =============================================================================
// CUDAAccelerator registration.
// =============================================================================

static Accelerator *createCUDAAccelerator() { return new CUDAAccelerator(); }

ACCEL_REGISTER("cuda", createCUDAAccelerator);

} // namespace accel
} // namespace onnx_mlir
