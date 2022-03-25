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

#define DEBUG_TYPE "NNPAAccelerator"

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
  if (initialized || acceleratorTarget.compare("NNPA") == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Targeting NNPA accelerator\n");
    return true;
  }

  return false;
}

void NNPAAccelerator::getOrLoadDialects(mlir::MLIRContext &context) const {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<zlow::ZLowDialect>();
}

void NNPAAccelerator::addPasses(mlir::OwningOpRef<ModuleOp> &module,
    mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType &emissionTarget) const {
  LLVM_DEBUG(llvm::dbgs() << "adding passes for accelerator "
                          << acceleratorTarget << "\n");
  addPassesNNPA(module, pm, emissionTarget, nnpaEmissionTarget, execNodesOnCpu);
}

void NNPAAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  registry.insert<zhigh::ZHighDialect>();
  registry.insert<zlow::ZLowDialect>();
}

void NNPAAccelerator::initPasses(int optLevel) const {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXToZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createRewriteONNXForZHighPass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighToZLowPass(optLevel);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowRewritePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowToLLVMPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createFoldStdAllocPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighConstPropagationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighLayoutPropagationPass();
  });
}

bool NNPAAccelerator::initialized = false;
// NNPAAccelerator nnpaAccelerator;

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
