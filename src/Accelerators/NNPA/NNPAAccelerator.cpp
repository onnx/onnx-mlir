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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVM.hpp"
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
  addCompilerConfig(CCM_SHARED_LIB_DEPS, {"zdnn"});
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

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowRewritePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowDummyOpForMultiDerefPass();
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

mlir::MemRefType NNPAAccelerator::convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const {
  assert(tensorType.hasRank() && "expected only ranked shapes");
  if (tensorType.cast<mlir::RankedTensorType>()
          .getEncoding()
          .dyn_cast_or_null<onnx_mlir::zhigh::ZTensorEncodingAttr>()) {
    onnx_mlir::zhigh::ZMemRefType zMemRefType =
        onnx_mlir::zhigh::convertZTensorToMemRefType(tensorType);
    return zMemRefType.value;
  }
  return nullptr;
}

void NNPAAccelerator::conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const {
  target.addLegalDialect<zlow::ZLowDialect>();
}

void NNPAAccelerator::rewritePatternONNXToKrnl(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  onnx_mlir::zhigh::populateZHighToZLowConversionPattern(
      patterns, typeConverter, ctx);
}

void NNPAAccelerator::conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const {}

void NNPAAccelerator::rewritePatternKrnlToLLVM(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  onnx_mlir::zlow::populateZLowToLLVMConversionPattern(
      patterns, typeConverter, ctx);
}

} // namespace accel
} // namespace onnx_mlir
