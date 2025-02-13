/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.cpp -----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
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
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVM.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "zdnn.h"

#include <memory>

#define DEBUG_TYPE "NNPAAccelerator"

extern llvm::cl::OptionCategory OMNNPAPassOptions;

namespace onnx_mlir {
namespace accel {

Accelerator *createNNPA() { return NNPAAccelerator::getInstance(); }

NNPAAccelerator *NNPAAccelerator::instance = nullptr;

NNPAAccelerator *NNPAAccelerator::getInstance() {
  if (instance == nullptr)
    instance = new NNPAAccelerator();
  return instance;
}

NNPAAccelerator::NNPAAccelerator() : Accelerator(Accelerator::Kind::NNPA) {
  LLVM_DEBUG(llvm::dbgs() << "Creating an NNPA accelerator\n");

  // Print a warning if mcpu is not set or < z16.
  if (!isCompatibleWithNNPALevel(NNPALevel::M14))
    llvm::outs() << "\nWarning: No NNPA code is generated because:\n"
                    "  --march is not set/older than z16.\n\n";

  acceleratorTargets.push_back(this);
  // Order is important! libRuntimeNNPA depends on libzdnn
  addCompilerConfig(CCM_SHARED_LIB_DEPS, {"RuntimeNNPA", "zdnn"}, true);
};

NNPAAccelerator::~NNPAAccelerator() { delete instance; }

// Return accelerator version number based on compile NNPA version
uint64_t NNPAAccelerator::getVersionNumber() const {
  if (isCompatibleWithNNPALevel(NNPALevel::M15))
    return NNPA_ZDNN_VERSIONS[NNPALevel::M15];
  return NNPA_ZDNN_VERSIONS[NNPALevel::M14];
}

void NNPAAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
  LLVM_DEBUG(llvm::dbgs() << "Adding passes for NNPA accelerator\n");
  addPassesNNPA(module, pm, emissionTarget, outputNameNoExt);
}

void NNPAAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering dialects for NNPA accelerator\n");
  registry.insert<zhigh::ZHighDialect>();
  registry.insert<zlow::ZLowDialect>();
}

void NNPAAccelerator::registerPasses(int optLevel) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering passes for NNPA accelerator\n");
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createDevicePlacementPass(nnpaLoadDevicePlacementFile,
        nnpaSaveDevicePlacementFile, nnpaPlacementHeuristic);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXToZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createRewriteONNXForZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createZHighToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowRewritePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowStickExpansionPass();
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

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighDecomposeStickUnstickPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighRecomposeToStickUnstickPass();
  });
}

void NNPAAccelerator::configurePasses() const {
  LLVM_DEBUG(llvm::dbgs() << "Configuring passes for NNPA accelerator\n");
  configurePassesNNPA();
}

mlir::MemRefType NNPAAccelerator::convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const {
  assert(tensorType.hasRank() && "expected only ranked shapes");
  if (mlir::dyn_cast_or_null<onnx_mlir::zhigh::ZTensorEncodingAttr>(
          mlir::cast<mlir::RankedTensorType>(tensorType).getEncoding())) {
    onnx_mlir::zhigh::ZMemRefType zMemRefType =
        onnx_mlir::zhigh::convertZTensorToMemRefType(tensorType);
    return zMemRefType.value;
  }
  return nullptr;
}

int64_t NNPAAccelerator::getDefaultAllocAlignment(
    const mlir::TensorType tensorType) const {
  assert(tensorType.hasRank() && "expected only ranked shapes");
  if (mlir::dyn_cast_or_null<onnx_mlir::zhigh::ZTensorEncodingAttr>(
          mlir::cast<mlir::RankedTensorType>(tensorType).getEncoding()))
    return gAlignment;
  return -1;
}

void NNPAAccelerator::conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const {
  target.addLegalDialect<zlow::ZLowDialect>();
}

void NNPAAccelerator::rewritePatternONNXToKrnl(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  onnx_mlir::zhigh::populateZHighToZLowConversionPattern(patterns,
      typeConverter, ctx,
      /*enableSIMD*/ OptimizationLevel >= 3 && !disableSimdOption,
      enableParallel);
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
