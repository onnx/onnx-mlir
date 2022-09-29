/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes.
//
// REQUEST: to the extend possible, passes here should not sample global
// optimization parameters specified in CompilerOptions.hpp. The passes should
// use parameters that are set by these global options where these passes are
// called. The idea is to keep our code as free of "rogue" global options used
// in random places in the code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void addONNXToMLIRPasses(mlir::PassManager &pm, int transformThreshold,
    bool transformReport, bool targetCPU) {
  // This is a transition from previous static passes to full dynamic passes
  // Static passes are kept and the dynamic pass is added as IF-THEN
  // with the static iteration.
  // The reasons are
  // 1. The debug flag, --print-ir-after/befor-all, can display IR for each
  //    static pass, but the dynamic pipeline will be viewed as one. MLIR
  //    may have solution that I am not aware of yet.
  // 2. Easy to compare two approaches.
  // In future, only the dynamic pass, ONNXOpTransformPass, will be used for
  // this function.

  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  // Convolution Optimization for CPU: enable when there are no accelerators.
  if (targetCPU) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass());
    pm.addPass(onnx_mlir::createShapeInferencePass());
  }
  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());

  if (transformThreshold > 0) {
    // Dynamic iterate in ONNXOpTransformPass
    pm.addPass(onnx_mlir::createONNXOpTransformPass(
        transformThreshold, transformReport, targetCPU));
  } else {
    // Statically add extra passes
    for (int i = 0; i < repeatOnnxTransform; i++) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass());
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    bool enableInstrumentONNXSignature, std::string ONNXOpsStatFormat) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  // Verify ONNX ops before lowering to Krnl.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXPreKrnlVerifyPass());
  // Print statistics about ONNX ops if enabled.
  if (ONNXOpsStatFormat.length() > 0) {
    transform(ONNXOpsStatFormat.begin(), ONNXOpsStatFormat.end(),
        ONNXOpsStatFormat.begin(), ::toupper);
    bool printAsJSON = ONNXOpsStatFormat.compare("JSON") == 0;
    bool printAsTXT = ONNXOpsStatFormat.compare("TXT") == 0;
    if (printAsJSON || printAsTXT) {
      // TODO: we should write the output of this pass in a file but I was not
      // able to use raw_fd_ostream of a file without it crashing.
      pm.addNestedPass<func::FuncOp>(
          mlir::createPrintOpStatsPass(llvm::outs(), printAsJSON));
    } else {
      llvm::errs() << "Skip onnx-ops-stats: expected JSON or TXT format, got \""
                   << ONNXOpsStatFormat << "\"\n";
    }
  }
  // Add instrumentation for Onnx Ops
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXPass(
      instrumentONNXOps, instrumentControlBits.getBits()));
  // Print Signatures of each op at runtime if enabled. Should not run signature
  // and instrument passes at the same time.
  if (enableInstrumentONNXSignature)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentONNXSignaturePass());
  pm.addPass(onnx_mlir::createLowerToKrnlPass(optLevel, enableParallel));
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // opportunities.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createDisconnectKrnlDimFromAllocPass());
  pm.addPass(mlir::createCanonicalizerPass());
} // namespace onnx_mlir

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::krnl::createConvertKrnlToAffinePass());
}

void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, bool enableCSE, bool verifyInputTensors) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Hoist allocations out of loop nests to avoid stack overflow.
  pm.addPass(bufferization::createBufferLoopHoistingPass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained in
  // https://mlir.llvm.org/docs/BufferDeallocationInternals.
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());
  if (enableMemoryBundling) {
    pm.addNestedPass<func::FuncOp>(krnl::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<func::FuncOp>(krnl::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(krnl::createKrnlOptimizeMemoryPoolsPass());
  }

  pm.addNestedPass<func::FuncOp>(krnl::createLowerKrnlRegionPass());
  pm.addNestedPass<func::FuncOp>(krnl::createConvertSeqToMemrefPass());
  pm.addNestedPass<func::FuncOp>(mlir::createConvertSCFToCFPass());

  pm.addPass(krnl::createConvertKrnlToLLVMPass(verifyInputTensors));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

InputIRLevelType determineInputIRLevel(mlir::OwningOpRef<ModuleOp> &module) {
  Operation *moduleOp = module->getOperation();

  // Collect dialect namespaces.
  llvm::SmallDenseSet<StringRef> dialectNamespace;
  moduleOp->walk([&](mlir::Operation *op) {
    dialectNamespace.insert(op->getDialect()->getNamespace());
  });

  // If there are ONNX ops, the input level is ONNX.
  bool hasONNXOps = llvm::any_of(dialectNamespace,
      [&](StringRef ns) { return (ns == ONNXDialect::getDialectNamespace()); });
  if (hasONNXOps)
    return ONNXLevel;

  // If there are Krnl ops, the input level is MLIR.
  bool hasKrnlOps = llvm::any_of(dialectNamespace,
      [&](StringRef ns) { return (ns == KrnlDialect::getDialectNamespace()); });
  if (hasKrnlOps)
    return MLIRLevel;

  // Otherwise, set to the lowest level, LLVMLevel.
  return LLVMLevel;
}

void addPasses(mlir::OwningOpRef<ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget) {
  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm, onnxOpTransformThreshold, onnxOpTransformReport,
        /*target CPU*/ maccel.empty());

  if (emissionTarget >= EmitMLIR) {
    if (inputIRLevel <= ONNXLevel)
      addONNXToKrnlPasses(pm, OptimizationLevel, /*enableCSE*/ true,
          instrumentONNXSignature, ONNXOpStats);
    if (inputIRLevel <= MLIRLevel)
      addKrnlToAffinePasses(pm);
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm, /*enableCSE=*/true, verifyInputTensors);
}

} // namespace onnx_mlir
