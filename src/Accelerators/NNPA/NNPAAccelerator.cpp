/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.cpp -----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for NNPA
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/NNPA/Compiler/DLCompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/DLCPasses.hpp"
#include "src/Support/OMOptions.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "DLCompiler"

extern llvm::cl::OptionCategory OMDLCPassOptions;
extern llvm::cl::opt<onnx_mlir::DLCEmissionTargetType> dlcEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;

namespace onnx_mlir {

NNPAAccelerator::NNPAAccelerator() : Accelerator() {
  LLVM_DEBUG(llvm::dbgs() << "initializing NNPA\n");

  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
  } // else
    // getAcceleratorList()->push_back(this);
};

bool NNPAAccelerator::isActive() const {
  LLVM_DEBUG(
      llvm::dbgs() << "check if NNPA is active" << acceleratorTarget << "\n");
  if (acceleratorTarget.compare("NNPA") == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Targeting NNPA accelerator\n");
    return true;
  }

  return false;
}

void NNPAAccelerator::prepareAccelerator(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget) const {
  LLVM_DEBUG(
      llvm::dbgs() << "preparing accelerator " << acceleratorTarget << "\n");

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<mlir::ZLowDialect>();
  addPassesDLC(module, pm, emissionTarget, dlcEmissionTarget, execNodesOnCpu);
}

bool NNPAAccelerator::initialized = false;
NNPAAccelerator nnpaAccelerator;

} // namespace onnx_mlir
