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
onnx_mlir::accel::nnpa::NNPAAccelerator *pnnpa = nullptr;

void createNNPA() { pnnpa = new onnx_mlir::accel::nnpa::NNPAAccelerator; }

namespace onnx_mlir {
extern llvm::cl::list<onnx_mlir::accel::Accelerator::Kind> maccel;

namespace accel {
namespace nnpa {

NNPAAccelerator::NNPAAccelerator() : Accelerator(Accelerator::Kind::NNPA) {
  LLVM_DEBUG(llvm::dbgs() << "Initializing NNPA accelerator\n");

  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
    acceleratorTargets.push_back(this);
  }
};

bool NNPAAccelerator::isActive() const {
  if (initialized || llvm ::any_of(maccel, [](Accelerator::Kind kind) {
        return kind == Accelerator::Kind::NNPA;
      })) {
    LLVM_DEBUG(llvm::dbgs() << "NNPA accelerator is active\n");
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "NNPA accelerator is not active\n");
  return false;
}

void NNPAAccelerator::getOrLoadDialects(mlir::MLIRContext &context) const {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<zhigh::ZHighDialect>();
  context.getOrLoadDialect<zlow::ZLowDialect>();
}

void NNPAAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType &emissionTarget) const {
  LLVM_DEBUG(llvm::dbgs() << "adding passes for NNPA accelerator\n");
  addPassesNNPA(module, pm, emissionTarget);
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

} // namespace nnpa
} // namespace accel
} // namespace onnx_mlir
