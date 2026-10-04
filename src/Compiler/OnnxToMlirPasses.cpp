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
  options.quantConstFold = false;
  options.dequantConstFold = false;
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

  if (opts.enableXMCPasses)
    opts.enableGatherElementsTileCanonicalization = false;

  configureBatchNormCanonicalization(opts.disableBatchNormDecompose);
  configureUnsafeMathCanonicalization(opts.enableUnsafeMathOptimizations);
  configureReshapeCanonicalization(opts.enableReshapeCanonicalization);
  configurePositiveAxisCanonicalization(
      opts.enablePositiveAxisCanonicalization);
  configureExpandCanonicalization(opts.enableExpandCanonicalization);
  configureKeepdimsCanonicalization(opts.enableKeepdimsCanonicalization);
  configureCastDataMovementPatterns(opts.enableCastDataMovementPatterns);
  configureGatherElementsTileCanonicalization(
      opts.enableGatherElementsTileCanonicalization);
  configureQDQDataMovementCanonicalization(
      opts.enableQDQDataMovementCanonicalization);

  if (!donotScrubDisposableElementsAttr)
    pm.addInstrumentation(
        std::make_unique<DisposableGarbageCollector>(pm.getContext()));

  // Decompose first. Eliminates some unsupported ops without shape inference.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
      getDecomposeOnlyOptions(opts.hybrid)));
  if (opts.hybrid.recomposition) {
    onnx_mlir::RecomposeONNXToONNXPassOptions recomposeOpts{
        .enableDepthToSpaceDecompose = opts.hybrid.enableDepthToSpaceDecompose,
        .enableRotaryEmbeddingRecompose =
            opts.hybrid.enableRotaryEmbeddingRecompose,
        .enableReduceL2Recompositions =
            opts.hybrid.enableReduceL2Recompositions};
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createRecomposeONNXToONNXPass(recomposeOpts));
  }

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
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass(
          {.enableQDQ = opts.hybrid.qdqConstProp,
              .enableQuantConstFold = opts.hybrid.quantConstFold,
              .enableDequantConstFold = opts.hybrid.dequantConstFold}));
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
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass(
        {.enableQDQ = opts.hybrid.qdqConstProp,
            .enableQuantConstFold = opts.hybrid.quantConstFold,
            .enableDequantConstFold = opts.hybrid.dequantConstFold}));
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
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass(
            {.enableQDQ = opts.hybrid.qdqConstProp,
                .enableQuantConstFold = opts.hybrid.quantConstFold,
                .enableDequantConstFold = opts.hybrid.dequantConstFold}));
      }
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass(
      opts.hybrid.quarkQuantizedOpsLegalization,
      opts.hybrid.enableGAPToReduceMean));

  // Canonicalizing Q-DQ related ops
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createQDQCanonicalizePass(
      {.removeBinary = opts.enableRemoveBinary,
          .removeQDQAroundOps = opts.enableRemoveDqQAroundOp}));

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

  // Hoist Gather above LayerNorm with optional Q/DQ chains.
  if (opts.enableHoistGatherAboveLayerNorm)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createHoistGatherAboveLayerNormPass());

  // Replace ONNXReturnOp with func::ReturnOp.
  pm.addPass(onnx_mlir::createStandardFuncReturnPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  // Replace every DisposableElementsAttr with DenseElementsAttr.
  //
  // AIESW-46865: pass closeAfter=false (default is true) -- scrub still runs
  // exactly as before, materializing every constant to Dense in one blanket
  // pass, so there's no regression to whatever non-weight-related cost the
  // rest of this onnx-to-onnx pipeline has. The only change is that the
  // DisposablePool isn't permanently closed afterward: DMAC's own later
  // passes build zero-copy *views* of already-Dense weight data (e.g. the
  // int4/uint4 hw_transpose tag step's reshape-only relabel) by wrapping it
  // in a fresh DisposableElementsAttr -- but DisposablePool::createElementsAttr
  // silently falls back to a real copy-via-DenseElementsAttr construction
  // whenever the pool is inactive (see its "otherwise returns conversion to
  // DenseElementsAttr" comment), which is exactly what closeAfter=true does
  // right here. Measured: that fallback was duplicating every int4/uint4
  // weight touched by the hw_transpose tag step, a dominant (single largest
  // observed jump, ~12GB on one real model) and entirely avoidable
  // contributor to DMAC frontend peak memory, once scrub has already made
  // everything Dense by the time DMAC's own passes run.
  if (!donotScrubDisposableElementsAttr)
    pm.addPass(createScrubDisposablePass(/*closeAfter=*/false));

  // Set onnx_node_name if it is missing. Keep this pass at the end of this
  // function and just before instrumentation.
  pm.addPass(createSetONNXNodeNamePass());

  if (opts.enableXFEONNXOpsetVerifier)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createXFEONNXOpsetVerifierPass());

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
