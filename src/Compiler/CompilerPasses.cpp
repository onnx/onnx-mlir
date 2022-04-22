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
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void addONNXToMLIRPasses(mlir::PassManager &pm) {
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

  pm.addNestedPass<FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());

  if (onnxOpTransformThreshold > 0) {
    // Dynamic iterate in ONNXOpTransformPass
    pm.addPass(onnx_mlir::createONNXOpTransformPass(
        onnxOpTransformThreshold, onnxOpTransformReport));
  } else {
    // Statically add extra passes
    for (int i = 0; i < repeatOnnxTransform; i++) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addNestedPass<FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    }
  }

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  // Verify ONNX ops before lowering to Krnl.
  pm.addNestedPass<FuncOp>(onnx_mlir::createONNXPreKrnlVerifyPass());
  // Add instrumentation for Onnx Ops
  pm.addNestedPass<FuncOp>(onnx_mlir::createInstrumentONNXPass(
      instrumentONNXOps, instrumentControlBits.getBits()));
  pm.addPass(onnx_mlir::createLowerToKrnlPass(optLevel));
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // opportunities.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(onnx_mlir::createDisconnectKrnlDimFromAllocPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<FuncOp>(onnx_mlir::krnl::createConvertKrnlToAffinePass());
}

void addKrnlToLLVMPasses(mlir::OpPassManager &pm, bool enableCSE) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained in
  // https://mlir.llvm.org/docs/BufferDeallocationInternals.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());
  if (enableMemoryBundling) {
    pm.addNestedPass<FuncOp>(krnl::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<FuncOp>(krnl::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(krnl::createKrnlOptimizeMemoryPoolsPass());
  }

  pm.addNestedPass<FuncOp>(krnl::createLowerKrnlRegionPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertSCFToCFPass());

  pm.addNestedPass<FuncOp>(krnl::createConvertSeqToMemrefPass());
  pm.addPass(krnl::createConvertKrnlToLLVMPass());
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
  bool hasKrnlOps = llvm::any_of(dialectNamespace, [&](StringRef ns) {
    return (ns == KrnlOpsDialect::getDialectNamespace());
  });
  if (hasKrnlOps)
    return MLIRLevel;

  // Otherwise, set to the lowest level, LLVMLevel.
  return LLVMLevel;
}

void addPasses(mlir::OwningOpRef<ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget) {
  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm);

  if (emissionTarget >= EmitMLIR) {
    if (inputIRLevel <= ONNXLevel)
      addONNXToKrnlPasses(pm, OptimizationLevel);
    if (inputIRLevel <= MLIRLevel)
      addKrnlToAffinePasses(pm);
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);
}

} // namespace onnx_mlir
