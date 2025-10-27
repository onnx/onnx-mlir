#include "OnnxToMlirPasses.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
namespace onnx_mlir {

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

  if (!donotScrubDisposableElementsAttr)
    pm.addInstrumentation(
        std::make_unique<DisposableGarbageCollector>(pm.getContext()));

  // Decompose first. Eliminates some unsupported ops without shape inference.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass(
      /*target=*/"", opts.enableConvTransposeDecompose,
      opts.enableConvTransposeDecomposeToPhasedConv,
      opts.enableConvTranspose1dDecomposeToPhasedConv));
  if (!opts.disableRecomposeOption)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createRecomposeONNXToONNXPass(
        /*target=*/"", opts.enableRecomposeLayernormByTranspose));

  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
        !opts.disableRecomposeOption, opts.enableQuarkQuantizedLegalization,
        opts.enableConvTransposeDecompose,
        opts.enableConvTransposeDecomposeToPhasedConv,
        opts.enableConvTranspose1dDecomposeToPhasedConv,
        opts.enableRecomposeLayernormByTranspose));
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createONNXHybridTransformPass(!opts.disableRecomposeOption,
              /*enableQuarkQuantizedOpsLegalization=*/false,
              opts.enableConvTransposeDecompose,
              opts.enableConvTransposeDecomposeToPhasedConv,
              opts.enableConvTranspose1dDecomposeToPhasedConv,
              opts.enableRecomposeLayernormByTranspose));
    }
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    }
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createLegalizeQuarkQuantizedOpsPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    if (opts.onnxOpTransformThreshold > 0) {
      // Dynamic iterate in ONNXOpTransformPass
      pm.addPass(onnx_mlir::createONNXOpTransformPass(
          opts.onnxOpTransformThreshold, opts.onnxOpTransformReport, targetCPU,
          opts.enableSimdDataLayout && !opts.disableSimdOption,
          opts.enableConvOptPass, !opts.disableRecomposeOption));
    } else {
      // Statically add extra passes
      for (int i = 0; i < opts.repeatOnnxTransform; i++) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
        pm.addNestedPass<func::FuncOp>(
            onnx_mlir::createConstPropONNXToONNXPass());
      }
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass(
      opts.enableQuarkQuantizedLegalization));

  // Pass for removing binary ops if one of the inputs is fed by a Constant
  if (opts.enableRemoveBinary)
    pm.addPass(createFoldDQBinaryQPass());

  // Pass for removing Dq and Q around data movement in Dq->op->Q Ops chain
  if (opts.enableRemoveDqQAroundOp)
    pm.addPass(createQDQAroundOpOptONNXToONNXPass());

  // Pass for removing redundant Dq->Q Ops chain
  // Passes for removing redundant concat, slice and cast QDQ Ops
  if (opts.enableRemoveDqQOp)
    pm.addPass(createQDQOptONNXToONNXPass());

  // One more call to ONNX shape inference/canonicalization/... to update
  // shape if possible.
  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
        !opts.disableRecomposeOption, opts.enableQuarkQuantizedLegalization,
        opts.enableConvTransposeDecompose,
        opts.enableConvTransposeDecomposeToPhasedConv,
        opts.enableConvTranspose1dDecomposeToPhasedConv,
        opts.enableRecomposeLayernormByTranspose));
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

  // Add instrumentation for Onnx Ops
  // Keep this pass at the end of this function.
  unsigned instrumentActions = opts.instrumentControlBits;
  if (opts.profileIR == onnx_mlir::ProfileIRs::Onnx) {
    opts.instrumentStage = onnx_mlir::InstrumentStages::Onnx;
    opts.instrumentOps = "onnx.*";
    instrumentActions |= (1 << 3) - 1;
  }
  if (opts.instrumentStage == onnx_mlir::InstrumentStages::Onnx)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(opts.instrumentOps, instrumentActions));
  if (opts.instrumentSignatures != "NONE" || opts.instrumentOnnxNode != "NONE")
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXSignaturePass(
        opts.instrumentSignatures, opts.instrumentOnnxNode));
}

} // namespace onnx_mlir
