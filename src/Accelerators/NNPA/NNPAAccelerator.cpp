/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- PrepareAccelerator.cpp
//-------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for NNPA
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
//#include "src/Accelerators/Accelerator.hpp"
#include "src/Support/OMOptions.hpp"
#include <iostream>
// modified from DLC main
#include "src/Compiler/DLCompilerUtils.hpp"
#include "src/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/DLCPasses.hpp"
extern llvm::cl::OptionCategory OMDLCPassOptions;
extern llvm::cl::opt<DLCEmissionTargetType> dlcEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;

namespace mlir {

NNPAAccelerator::NNPAAccelerator() {
  std::cout << "initializing NNPA" << std::endl;
  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
  } // else
    // getAcceleratorList()->push_back(this);
};

bool NNPAAccelerator::isActive() {
  std::cout << "check if NNPA is active" << acceleratorTarget << std::endl; 
  if (acceleratorTarget.compare("NNPA") == 0) {
    std::cout << "Targeting NNPA accelerator" << std::endl;
    return true;
  } else
    return false;
}

void NNPAAccelerator::prepareAccelerator(mlir::OwningModuleRef &module, mlir::MLIRContext &context, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget) {
  std::cout << "preparing accelerator " << acceleratorTarget << std::endl;

      // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::ZHighDialect>();
  context.getOrLoadDialect<mlir::ZLowDialect>();
  addPassesDLC(module, pm, emissionTarget, dlcEmissionTarget, execNodesOnCpu);

    }
bool NNPAAccelerator::initialized = false;
NNPAAccelerator nnpaAccelerator;
} // namespace mlir


