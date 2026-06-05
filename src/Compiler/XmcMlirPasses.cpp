// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===------------------------ XmcMlirPasses.cpp ---------------------------===//
//
// XMC-specific ONNX-to-MLIR pass pipeline additions.
//
//===----------------------------------------------------------------------===//

#include "OnnxToMlirPasses.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void addXmcMlirPasses(mlir::OpPassManager &pm, OnnxToMlirOptions opts) {
  pm.addNestedPass<func::FuncOp>(createFixNegScalePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRecomposeHardSigmoidPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDQBinaryQOptPass());
  // Replaced by createXmcRequantizePass below (runs post-quant-types).
  // pm.addNestedPass<func::FuncOp>(
  //     onnx_mlir::createOptimizeOnnxRequantizationPass());
  pm.addNestedPass<func::FuncOp>(createONNXCSEPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvertQDQToRequantizePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createQuantTypesPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferScalarConstInputDivToRequantizePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createPropagateQuantTypeThroughDataFlowPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createXmcRequantizePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceErfToGeluPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceTanhToGeluPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertInstanceNormToGroupNormPass());
  //  pm.addNestedPass<func::FuncOp>(onnx_mlir::createSplitGroupConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveDilationConv());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferResizeLinearToDwConv());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvWithBiasPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveRedundantReshapePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferReduceMeanSumToConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQReductionPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createLowerReduceToPoolPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferPoolFixToDownsampleFixPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveRedundantReluPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createStandardizeSliceOpsPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createMergeContinuousStridedSlicePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertMulToDepthwiseConv2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferDepthwiseConv2dWithChannelMultiplierPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveUselessQLinearPoolPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createOptimizeSliceReshapeTransposeBlockPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferSpaceToDepthToConv2dPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createMergeBatchnormToConvPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createEliminateReshapeAroundSlicePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createMergeSliceConcatPass());

  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferConvSliceToConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOp1dToOp2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferScaleToDwConv2dPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvertToChannelLastPass());
  if (opts.enableMatmulAddFusion)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createFuseMatMulAddToXFEMatMulBiasPass());
  if (opts.enableMatmulToConv)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createConvertMatMulToXFEConvPass());
  // Covered by createXmcRequantizePass (propagation-induced subset).
  // pm.addNestedPass<func::FuncOp>(
  //     onnx_mlir::createConvertSCastPairToRequantizePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferSoftmaxAxisToLastPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createONNXTransposeOptimizationPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveContinuousTransposeWithReshapePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOp3dToOp2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransformReshapelikeOpToReshapePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveSemanticallyUselessOpsPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransfer5dBlockTo4dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransform5DTransposeTo4DPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createCombineTransposePairPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceNDimTransposePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransfer5dStridedSliceTo4d());
  // Note: architecture specific pass.
  // pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOpShapeTo4dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createBatchReductionToReshapeReductionPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQResizePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createReplaceQuantizedTileToAddPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQEltwisePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQSigmoidPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceAdjacentOpPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemovePairsAndMoveDownReshapePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceContainedConcatPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createOptimizeSiblingConcatPass());

  pm.addNestedPass<func::FuncOp>(createCanonicalizeWithResultNamesPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createReplaceHsigmoidAndHswishPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertXFEConvToDepthwiseConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createFuseConvActivationPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createNormalizeConvActivationPass());
  // pm.addNestedPass<func::FuncOp>(
  //     onnx_mlir::createTransferReduceHdimToReduceCdimPass());

  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createAddRequantForOutputConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
}

} // namespace onnx_mlir
