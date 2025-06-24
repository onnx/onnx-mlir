/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.cpp -------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Functions for configuring and adding passes.
//
// REQUEST: To the extent possible, passes here should not sample global
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
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void configurePasses() {
  // Handle deprecated mcpu.
  if (!mcpu.empty()) {
    if (!march.empty()) {
      llvm::outs() << "\nWarning: Got values for both --march and --mcpu, "
                      "ignore --mcpu. "
                      "Please remove deprecated --mcpu in the near future.\n\n";
    } else {
      llvm::outs()
          << "\nWarning: Got deprecated --mcpu option. Please switch to "
             "--march in the near future.\n\n";
    }
  }
  // Set global vector machine support.
  VectorMachineSupport::setGlobalVectorMachineSupport(march, mcpu, "");
  configureConstPropONNXToONNXPass(onnxConstPropRoundFPToInt,
      onnxConstPropExpansionBound, onnxConstPropDisablePatterns,
      disableConstantProp);
  configureOnnxToKrnlLoweringPass(optReport == OptReport::Parallel,
      enableParallel, parallelizeOps, optReport == OptReport::Simd,
      !disableSimdOption);
}

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr) {
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

  if (!donotScrubDisposableElementsAttr)
    pm.addInstrumentation(
        std::make_unique<DisposableGarbageCollector>(pm.getContext()));

  // Decompose first. Eliminates some unsupported ops without shape inference.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  if (!disableRecomposeOption)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createRecomposeONNXToONNXPass());
  if (enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createONNXHybridTransformPass(!disableRecomposeOption));
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          enableSimdDataLayout && !disableSimdOption));
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createONNXHybridTransformPass(!disableRecomposeOption));
    }
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          enableSimdDataLayout && !disableSimdOption));
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    }
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    if (onnxOpTransformThreshold > 0) {
      // Dynamic iterate in ONNXOpTransformPass
      pm.addPass(onnx_mlir::createONNXOpTransformPass(onnxOpTransformThreshold,
          onnxOpTransformReport, targetCPU,
          enableSimdDataLayout && !disableSimdOption, enableConvOptPass,
          !disableRecomposeOption));
    } else {
      // Statically add extra passes
      for (int i = 0; i < repeatOnnxTransform; i++) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
        pm.addNestedPass<func::FuncOp>(
            onnx_mlir::createConstPropONNXToONNXPass());
      }
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());

  // One more call to ONNX shape inference/canonicalization/... to update
  // shape if possible.
  if (enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createONNXHybridTransformPass(!disableRecomposeOption));
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  }

  // Replace ONNXReturnOp with func::ReturnOp.
  pm.addPass(onnx_mlir::createStandardFuncReturnPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  // Replace every DisposableElementsAttr with DenseElementsAttr.
  if (!donotScrubDisposableElementsAttr)
    pm.addPass(createScrubDisposablePass());

  // Set onnx_node_name if it is missing. Keep this pass at the end of this
  // function and just before instrumentation.
  pm.addPass(createSetONNXNodeNamePass());

  // Add instrumentation for profiling/ signature for Onnx Ops. Keep this pass
  // at the end of this function.
  unsigned instrumentActions = instrumentControlBits;
  if (profileIR == onnx_mlir::ProfileIRs::Onnx) {
    instrumentStage = onnx_mlir::InstrumentStages::Onnx;
    instrumentOps = "onnx.*";
    // Enable the first three bits for InstrumentBeforOp, InstrumentAfterOp
    // and InstrumentReportTime. Disable the last bit for
    // InstrumentReportMemory because of its big overhead. Users can
    // optionally enable the last bit by using
    // --InstrumentReportMemory option.
    instrumentActions |= (1 << 3) - 1;
    // Also enable instrumentation of signatures.
    instrumentSignatures = "onnx.*";
  }
  // Add createInstrument (timing) second so that it will guarantee not to
  // include timing of the signature printing.
  if (hasSignatureInstrumentation(onnx_mlir::InstrumentStages::Onnx))
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXSignaturePass(
        instrumentSignatures, instrumentOnnxNode));
  if (hasInstrumentation(onnx_mlir::InstrumentStages::Onnx))
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(instrumentOps, instrumentActions));
}

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    std::string ONNXOpsStatFormat) {
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

  pm.addPass(onnx_mlir::createLowerToKrnlPass(/*enableTiling*/ optLevel >= 3,
      /*enableSIMD*/ optLevel >= 3 && !disableSimdOption, enableParallel,
      /*enableFastMath*/ optLevel >= 3 && enableFastMathOption,
      /*opsToCall*/ opsForCall));
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // opportunities.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::krnl::createConvertKrnlToAffinePass(enableParallel));
}

void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, std::string outputNameNoExt, bool enableCSE) {
  if (enableCSE)
    // Eliminate common sub-expressions before lowering to Krnl.
    // TODO: enable this by default when we make sure it works flawlessly.
    pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Early introduction of omp causes problems with bufferization, delay for
  // now. May revise this decision later.

  // After affine is lowered, KrnlRegion for affine scope can be removed.
  pm.addNestedPass<func::FuncOp>(krnl::createLowerKrnlRegionPass());

  if (enableParallel) {
    // Pass to ensure that memory allocated by parallel loops stay inside the
    // parallel region (privatization of memory). Otherwise, all threads would
    // end up sharing the same temporary data. This pass works on affine
    // parallel operations, and must be executed (in presence of OMP
    // parallelism) before bufferization. In practical terms, this pass add
    // memref.alloca_scope inside each parallel for.
    pm.addPass(onnx_mlir::createProcessScfParallelPrivatePass());
    // No canonicalize passes are allowed between that pass above and the buffer
    // management passes.
  }

  // Hoist allocations out of loop nests to avoid stack overflow.
  pm.addPass(bufferization::createBufferLoopHoistingPass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained
  // in https://mlir.llvm.org/docs/BufferDeallocationInternals.
  bufferization::BufferDeallocationPipelineOptions bufferDeallocOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(
      pm, bufferDeallocOptions);
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());

  // Late introduction of OpenMP, after bufferization.
  if (enableParallel) {
    // Cannot have canonicalization before OpenMP... have seen loop disappear.
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    //  The alloca_scope ops are somewhat fragile; canonicalize remove them when
    //  redundant, which helps reliability of the compilation of these ops.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(onnx_mlir::createProcessKrnlParallelClausePass());
  }

  // The pass below is needed for subview and collapseShape.. Unfortunately,
  // MLIR supports only collapse for scalar loaded by scalar memory at this
  // time. Uncomment if subview/collapse are used.
  // pm.addNestedPass<func::FuncOp>(krnl::createConvertSeqToMemrefPass());

  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());

  if (profileIR)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentCleanupPass());

  if (enableBoundCheck)
    pm.addPass(mlir::createGenerateRuntimeVerificationPass());
  pm.addPass(krnl::createConvertKrnlToLLVMPass(verifyInputTensors,
      /*useLRODATA=*/(modelSize == ModelSize::large),
      /*storeConstantsToFile=*/storeConstantsToFile,
      constantsToFileSingleThreshold, constantsToFileTotalThreshold,
      outputNameNoExt, enableParallel));
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
    EmissionTargetType emissionTarget, std::string outputNameNoExt) {
  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());

  if (emissionTarget >= EmitMLIR) {
    if (inputIRLevel <= ONNXLevel)
      addONNXToKrnlPasses(
          pm, OptimizationLevel, /*enableCSE*/ true, ONNXOpStats);
    if (inputIRLevel <= MLIRLevel)
      addKrnlToAffinePasses(pm);
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm, outputNameNoExt, /*enableCSE=*/true);
}

} // namespace onnx_mlir
