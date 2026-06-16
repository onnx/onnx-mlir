/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OnnxToMlirPasses.cpp ------------------------------===//
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#include "OnnxToMlirPasses.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"
using namespace mlir;
namespace onnx_mlir {

// Return a copy of the hybrid options with the sub-passes that are not
// not part of the decompose-only stage disabled.
static ONNXHybridTransformPassOptions getDecomposeOnlyOptions(
    ONNXHybridTransformPassOptions options) {
  options.shapeInference = false;
  options.canonicalization = false;
  options.constantPropagation = false;
  options.qdqConstProp = false;
  options.decomposition = true;
  options.recomposition = false;
  options.quarkQuantizedOpsLegalization = false;
  options.enableRotaryEmbeddingRecompose = false;
  return options;
}

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr, OnnxToMlirOptions opts) {
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
  configureBatchNormCanonicalization(opts.disableBatchNormDecompose);
  configureUnsafeMathCanonicalization(opts.enableUnsafeMathOptimizations);

  if (!donotScrubDisposableElementsAttr)
    pm.addInstrumentation(
        std::make_unique<DisposableGarbageCollector>(pm.getContext()));

  // Decompose first. Eliminates some unsupported ops without shape inference.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
      getDecomposeOnlyOptions(opts.hybrid)));
  if (opts.hybrid.recomposition)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createRecomposeONNXToONNXPass(
        /*target=*/"", opts.hybrid.enableRotaryEmbeddingRecompose));

  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createONNXHybridTransformPass(opts.hybrid));
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      ONNXHybridTransformPassOptions hybridOptions = opts.hybrid;
      hybridOptions.quarkQuantizedOpsLegalization = false;
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createONNXHybridTransformPass(hybridOptions));
    }
    // If quark quantized legalization is enabled, do a last const prop after it
    // so that we cover any remaining Cast -> Cast patterns that weren't covered
    // by the pass.
    if (opts.hybrid.quarkQuantizedOpsLegalization) {
      configureConstPropONNXToONNXPass(/*roundFPToInt=*/false,
          /*expansionBound=*/-1, /*disabledPatterns=*/{""},
          /*constantPropIsDisabled=*/false);
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass(opts.hybrid.qdqConstProp));
    }
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(onnx_mlir::createCanonicalizeWithResultNamesPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    }
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createLegalizeQuarkQuantizedOpsPass());
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass(opts.hybrid.qdqConstProp));
    if (opts.onnxOpTransformThreshold > 0) {
      // Dynamic iterate in ONNXOpTransformPass
      pm.addPass(onnx_mlir::createONNXOpTransformPass(
          opts.onnxOpTransformThreshold, opts.onnxOpTransformReport, targetCPU,
          opts.enableSimdDataLayout && !opts.disableSimdOption,
          opts.enableConvOptPass, opts.hybrid.recomposition));
    } else {
      // Statically add extra passes
      for (int i = 0; i < opts.repeatOnnxTransform; i++) {
        pm.addPass(onnx_mlir::createCanonicalizeWithResultNamesPass());
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
        pm.addNestedPass<func::FuncOp>(
            onnx_mlir::createConstPropONNXToONNXPass(opts.hybrid.qdqConstProp));
      }
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass(
      opts.hybrid.quarkQuantizedOpsLegalization,
      opts.hybrid.enableGAPToReduceMean));

  // Canonicalizing Q-DQ related ops
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createQDQCanonicalizePass(
      opts.enableRemoveBinary, opts.enableRemoveDqQAroundOp));

  // One more call to ONNX shape inference/canonicalization/... to update
  // shape if possible.
  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createONNXHybridTransformPass(opts.hybrid));
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(onnx_mlir::createCanonicalizeWithResultNamesPass());
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

#ifdef ONNX_MLIR_ENABLE_KRNL
  // Add instrumentation for Onnx Ops (requires Krnl dialect for
  // KrnlInstrumentOp). Keep this pass at the end of this function.
  unsigned instrumentActions = opts.instrumentControlBits;
  if (opts.profileIR == onnx_mlir::ProfileIRs::Onnx) {
    opts.instrumentStage = onnx_mlir::InstrumentStages::Onnx;
    opts.instrumentOps = "onnx.*";
    instrumentActions |= (1 << 3) - 1;
  }
  if (opts.instrumentStage == onnx_mlir::InstrumentStages::Onnx)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(opts.instrumentOps, instrumentActions));
#endif
  if (opts.instrumentSignatures != "NONE" || opts.instrumentOnnxNode != "NONE")
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXSignaturePass(
        opts.instrumentSignatures, opts.instrumentOnnxNode));
  if (opts.enableXMCPasses)
    addXmcMlirPasses(pm, opts);
}

} // namespace onnx_mlir
