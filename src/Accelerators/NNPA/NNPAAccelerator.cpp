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

#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <memory>

#define DEBUG_TYPE "NNPAAccelerator"

extern llvm::cl::OptionCategory OMNNPAPassOptions;

namespace onnx_mlir {
namespace accel {

void createNNPA() { NNPAAccelerator::getInstance(); }

NNPAAccelerator *NNPAAccelerator::instance = nullptr;

NNPAAccelerator *NNPAAccelerator::getInstance() {
  if (instance == nullptr)
    instance = new NNPAAccelerator();
  return instance;
}

NNPAAccelerator::NNPAAccelerator() : Accelerator(Accelerator::Kind::NNPA) {
  LLVM_DEBUG(llvm::dbgs() << "Creating an NNPA accelerator\n");
  acceleratorTargets.push_back(this);
};

NNPAAccelerator::~NNPAAccelerator() { delete instance; }

bool NNPAAccelerator::isActive() const {
  if (instance || llvm::any_of(maccel, [](Accelerator::Kind kind) {
        return kind == Accelerator::Kind::NNPA;
      })) {
    LLVM_DEBUG(llvm::dbgs() << "NNPA accelerator is active\n");
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "NNPA accelerator is not active\n");
  return false;
}

void NNPAAccelerator::getOrLoadDialects(mlir::MLIRContext &context) const {
  LLVM_DEBUG(llvm::dbgs() << "Loading dialects for NNPA accelerator\n");
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<zlow::ZLowDialect>();
}

void NNPAAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType &emissionTarget) const {
  LLVM_DEBUG(llvm::dbgs() << "Adding passes for NNPA accelerator\n");
  addPassesNNPA(module, pm, emissionTarget);
}

void NNPAAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering dialects for NNPA accelerator\n");
  registry.insert<zhigh::ZHighDialect>();
  registry.insert<zlow::ZLowDialect>();
}

void NNPAAccelerator::initPasses(int optLevel) const {
  LLVM_DEBUG(llvm::dbgs() << "Initializing passes for NNPA accelerator\n");
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

} // namespace accel
} // namespace onnx_mlir
