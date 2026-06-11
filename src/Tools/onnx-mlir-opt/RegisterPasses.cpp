/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- RegisterPasses.cpp -------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// Beware that we cannot access CompilerOptions or other command line options
// the functions below because they are called before command line options are
// parsed, because passes must be registered first as they instruct command
// line parsing: registered passes can be expressed as command line flags.
//
// In particular the --O flag value is passed as a function argument optLevel
// so we can avoid reading the OptimizationLevel command-line option here.
//
//===----------------------------------------------------------------------===//

#include "RegisterPasses.hpp"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerPasses.hpp"

#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Compiler/OnnxToMlirPasses.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void registerOMPasses(int optLevel) {
  // All passes implemented within onnx-mlir should register within this
  // function to make themselves available as a command-line option.

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createScrubDisposablePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXOpTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDecomposeONNXToONNXPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRecomposeONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvOptONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertToChannelLastPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createMergeSliceConcatPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createCombineTransposePairPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createMergeContinuousStridedSlicePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveContinuousTransposeWithReshapePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveUselessQLinearPoolPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceHsigmoidAndHswishPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRecomposeHardSigmoidPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferSoftmaxAxisToLastPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceErfToGeluPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceTanhToGeluPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQDQSigmoidPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createMergeStridedSliceConcatConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferResizeLinearToDwConv();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerReduceToPoolPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveSemanticallyUselessOpsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferReduceMeanSumToConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferReduceHdimToReduceCdimPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQDQReductionPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferConvSliceToConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveDilationConv();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertInstanceNormToGroupNormPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createStandardizeSliceOpsPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createConvWithBiasPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createFuseMatMulAddToXFEMatMulBiasPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveRedundantReshapePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertXFEConvToDepthwiseConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createFuseConvActivationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createNormalizeConvActivationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertMulToDepthwiseConv2dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferSpaceToDepthToConv2dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQDQClipCastPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQDQEltwisePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQuantizedTileToAddPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceQDQResizePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceAdjacentOpPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceContainedConcatPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createOptimizeSiblingConcatPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemovePairsAndMoveDownReshapePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createMergeBatchnormToConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createBatchReductionToReshapeReductionPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRemoveRedundantReluPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createOptimizeOnnxRequantizationPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createDQBinaryQOptPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertSCastPairToRequantizePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddRequantForOutputConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertQDQToRequantizePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createPropagateQuantTypeThroughDataFlowPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createXmcRequantizePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSplitGroupConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertMatMulToXFEConvPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceNDimTransposePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferOpShapeTo4dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferOp1dToOp2dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferScaleToDwConv2dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferOp3dToOp2dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferPoolFixToDownsampleFixPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransferDepthwiseConv2dWithChannelMultiplierPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransformReshapelikeOpToReshapePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransform5DTransposeTo4DPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createEliminateReshapeAroundSlicePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createOptimizeSliceReshapeTransposeBlockPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransfer5dBlockTo4dPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTransfer5dStridedSliceTo4d();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXTransposeOptimizationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXHybridTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createShapeInferencePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConstPropONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createQDQCanonicalizePass();
  });

#ifdef ONNX_MLIR_ENABLE_KRNL
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createInstrumentPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentCleanupPass();
  });
#endif

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentONNXSignaturePass("NONE", "NONE");
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSetONNXNodeNamePass();
  });

#ifdef ONNX_MLIR_ENABLE_KRNL
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXPreKrnlVerifyPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToAffinePass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToKrnlPass(/*enableTiling*/ optLevel >= 3,
        /*enableSIMD, should consider disableSimdOption*/ optLevel >= 3,
        /*enableParallel*/ false,
        /*enableFastMath*/ false, /*default is still off*/
        /*opsForCall*/ "");
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createProcessScfParallelPrivatePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createProcessKrnlParallelClausePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertSeqToMemrefPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createLowerKrnlRegionPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToLLVMPass();
  });
#endif

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSimplifyShapeRelatedOpsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createStandardFuncReturnPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXDimAnalysisPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertONNXToTOSAPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLegalizeQuarkQuantizedOpsPass();
  });

  mlir::registerPass(createQuantTypesPass);
  mlir::registerPass(createONNXCSEPass);
  mlir::registerPass(createFixNegScalePass);
  mlir::registerPass(createInferTensorNames);
  mlir::registerPass(createCanonicalizeWithResultNamesPass);
  mlir::registerPass(createFoldQuantizedBinary);
  mlir::registerPass(createTransferScalarConstInputDivToRequantizePass);

  mlir::PassPipelineRegistration<>("xmc-passes", "Run all XMC xcompiler passes",
      [](mlir::OpPassManager &pm) { addXmcMlirPasses(pm); });

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToStablehloPass();
  });
#endif
}

void registerMLIRPasses() {
  registerTransformsPasses();
  affine::registerAffinePasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  bufferization::registerBufferizationPasses();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToSCFPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLowerAffinePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertSCFToCFPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createPrintOpStatsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertSCFToOpenMPPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertOpenMPToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFinalizeMemRefToLLVMConversionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
}

void registerPasses(int optLevel) {
  registerMLIRPasses();

  registerOMPasses(optLevel);

  // Register passes for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerPasses(optLevel);
}

} // namespace onnx_mlir
