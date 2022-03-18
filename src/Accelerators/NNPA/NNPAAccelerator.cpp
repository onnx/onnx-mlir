/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.cpp -----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for the IBM Telum coprocessor
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Support/OMOptions.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nnpa"

extern llvm::cl::OptionCategory OMNNPAPassOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;

namespace onnx_mlir {
namespace accel {
namespace nnpa {

NNPAAccelerator::NNPAAccelerator() : Accelerator(Accelerator::Kind::NNPA) {}

NNPAAccelerator::~NNPAAccelerator() {}

Accelerator *NNPAAccelerator::getInstance() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "Found strong definition for NNPAAccelerator::getInstance()\n";);
  return &singleton;
}

void NNPAAccelerator::prepare(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget) const {
  LLVM_DEBUG(llvm::dbgs() << "preparing NNPA accelerator\n");

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<zlow::ZLowDialect>();
  addPassesNNPA(module, pm, emissionTarget, nnpaEmissionTarget, execNodesOnCpu);
}

NNPAAccelerator NNPAAccelerator::singleton;

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
