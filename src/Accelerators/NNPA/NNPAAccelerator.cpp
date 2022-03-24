/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.cpp -----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for the IBM Telum processor.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Support/OMOptions.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "NNPACompiler"

extern llvm::cl::OptionCategory OMNNPAPassOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;
onnx_mlir::accel::nnpa::NNPAAccelerator *pnnpa;

void createNNPA() { pnnpa = new onnx_mlir::accel::nnpa::NNPAAccelerator; }

namespace onnx_mlir {
namespace accel {
namespace nnpa {

NNPAAccelerator::NNPAAccelerator() : Accelerator() {
  LLVM_DEBUG(llvm::dbgs() << "initializing NNPA\n");

  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
    acceleratorTargets.push_back(this);
  }
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
    onnx_mlir::EmissionTargetType &emissionTarget) const {
  LLVM_DEBUG(
      llvm::dbgs() << "preparing accelerator " << acceleratorTarget << "\n");

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<zlow::ZLowDialect>();
  addPassesNNPA(module, pm, emissionTarget, nnpaEmissionTarget, execNodesOnCpu);
}

bool NNPAAccelerator::initialized = false;
// NNPAAccelerator nnpaAccelerator;

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
