/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Passes.hpp - ONNX-MLIR Passes Definition ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PASSES_H
#define ONNX_MLIR_PASSES_H

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class Pass;
} // namespace mlir

namespace onnx_mlir {

/// Pass for removing DisposableElementsAttr attributes.
std::unique_ptr<mlir::Pass> createScrubDisposablePass(bool closeAfter = true);

/// Pass for ONNX graph level optimization
std::unique_ptr<mlir::Pass> createONNXOpTransformPass();
std::unique_ptr<mlir::Pass> createONNXOpTransformPass(int threshold,
    bool report, bool targetCPU, bool enableSimdDataLayoutOpt,
    bool enableConvOptPass, bool enableRecomposeOptPass);

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass(
    const std::string &target = "", bool enableConvTransposeDecompose = false,
    bool enableConvTransposeDecomposeToPhasedConv = false,
    bool enableConvTranspose1dDecomposeToPhasedConv = false,
    bool enableInstanceNormDecompose = true,
    bool enableSplitToSliceDecompose = false);
std::unique_ptr<mlir::Pass> createRecomposeONNXToONNXPass(
    const std::string &target = "");

std::unique_ptr<mlir::Pass> createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt = false);

std::unique_ptr<mlir::Pass> createShapeInferencePass();

// To configure ConstPropONNXToONNXPass at program start.
void configureConstPropONNXToONNXPass(bool roundFPToInt, int expansionBound,
    llvm::ArrayRef<std::string> disabledPatterns, bool constantPropIsDisabled);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass();

std::unique_ptr<mlir::Pass> createQDQCanonicalizePass(
    bool removeBinary = false, bool removeQDQAroundOps = false);

std::unique_ptr<mlir::Pass> createQuantTypesPass();

/// Pass for instrument the ops in specific stage.
std::unique_ptr<mlir::Pass> createInstrumentPass();
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions);
/// Pass for instrument cleanup.
std::unique_ptr<mlir::Pass> createInstrumentCleanupPass();

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass(
    const std::string opPattern, const std::string nodePattern);

/// Pass for simplifying shape-related ONNX operations.
std::unique_ptr<mlir::Pass> createSimplifyShapeRelatedOpsPass(
    bool disableCastOpCanonicalizations = false);

/// Pass for replacing ONNXReturnOp with func::ReturnOp.
std::unique_ptr<mlir::Pass> createStandardFuncReturnPass();

/// Pass that combines multiple ONNX dialect transformations,
/// including shape inference.
std::unique_ptr<mlir::Pass> createONNXHybridTransformPass(
    bool enableRecomposition, bool enableQuarkQuantizedOpsLegalization = false,
    bool enableConvTransposeDecompose = false,
    bool enableConvTransposeDecomposeToPhasedConv = false,
    bool enableConvTranspose1dDecomposeToPhasedConv = false,
    bool enableInstanceNormDecompose = true,
    bool enableSplitToSliceDecompose = false);

/// Pass for analyzing unknown dimension in ONNX operations.
std::unique_ptr<mlir::Pass> createONNXDimAnalysisPass();

/// Pass for setting onnx_node_name attribute if absent.
std::unique_ptr<mlir::Pass> createSetONNXNodeNamePass();

/// Pass for converting ONNX operations to ChannelLast variants with transposes.
/// Supports: Conv, AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool,
/// InstanceNormalization, DepthToSpace, SpaceToDepth
std::unique_ptr<mlir::Pass> createConvertToChannelLastPass();

/// Pass for merging Slice->Concat patterns with downstream ops.
std::unique_ptr<mlir::Pass> createMergeSliceConcatPass();
std::unique_ptr<mlir::Pass> createMergeStridedSliceConcatConvPass();

/// Pass for merging continuous chained Slice operations with quantized types.
std::unique_ptr<mlir::Pass> createMergeContinuousStridedSlicePass();

std::unique_ptr<mlir::Pass> createONNXTransposeOptimizationPass();

/// Pass to combine two transpose with same input and same perm.
std::unique_ptr<mlir::Pass> createCombineTransposePairPass();

/// Pass to remove redundant Transpose-Reshape-Transpose sequences.
std::unique_ptr<mlir::Pass> createRemoveContinuousTransposeWithReshapePass();

/// Pass for transferring Resize Linear operations to depthwise convolutions.
std::unique_ptr<mlir::Pass> createTransferResizeLinearToDwConv();

/// Pass for lowering Reduce operations to Pool operations.
std::unique_ptr<mlir::Pass> createLowerReduceToPoolPass();

/// Pass for removing semantically useless operations.
std::unique_ptr<mlir::Pass> createRemoveSemanticallyUselessOpsPass();

/// Pass for removing useless pool operations (kernel_shape and strides all 1s).
std::unique_ptr<mlir::Pass> createRemoveUselessQLinearPoolPass();

/// Pass for replacing quantized HardSigmoid with XCOMPILERFusedEltwise.
std::unique_ptr<mlir::Pass> createReplaceHsigmoidAndHswishPass();

/// Pass for transferring ReduceMean/Sum operations to Conv operations.
std::unique_ptr<mlir::Pass> createTransferReduceMeanSumToConvPass();

/// Pass for transferring Conv->Slice patterns to Conv operations.
std::unique_ptr<mlir::Pass> createTransferConvSliceToConvPass();

/// Pass for removing dilation from Conv operations.
std::unique_ptr<mlir::Pass> createRemoveDilationConv();

/// Pass for converting InstanceNorm to GroupNorm.
std::unique_ptr<mlir::Pass> createConvertInstanceNormToGroupNormPass();

/// Pass for standardizing Slice operations.
std::unique_ptr<mlir::Pass> createStandardizeSliceOpsPass();

/// Pass for converting Mul operations to DepthwiseConv2d when applicable.
std::unique_ptr<mlir::Pass> createConvertMulToDepthwiseConv2dPass();

/// Pass for transferring 3D operations to 2D operations.
std::unique_ptr<mlir::Pass> createTransferOp3dToOp2dPass();

/// Pass for transferring pool-fix operations to downsample-fix operations.
std::unique_ptr<mlir::Pass> createTransferPoolFixToDownsampleFixPass();

/// Pass for splitting depthwise conv2d with channel_multiplier > 1.
std::unique_ptr<mlir::Pass>
createTransferDepthwiseConv2dWithChannelMultiplierPass();

/// Pass for transferring PoolFix operations to DownsampleFix operations.
std::unique_ptr<mlir::Pass> createTransferPoolFixToDownsampleFixPass();

/// Pass for transforming reshape-like operations to Reshape.
std::unique_ptr<mlir::Pass> createTransformReshapelikeOpToReshapePass();

/// Pass for transforming 5D Transpose to Reshape + 4D Transpose + Reshape.
std::unique_ptr<mlir::Pass> createTransform5DTransposeTo4DPass();

/// Pass for eliminating reshape operations around slice operations.
std::unique_ptr<mlir::Pass> createEliminateReshapeAroundSlicePass();

/// Pass for optimizing MHA Slice-Reshape-Transpose blocks.
std::unique_ptr<mlir::Pass> createOptimizeSliceReshapeTransposeBlockPass();

/// Pass for transferring 5D block operations to 4D equivalents.
std::unique_ptr<mlir::Pass> createTransfer5dBlockTo4dPass();

/// Pass for transferring 5D strided Slice operations to 4D.
std::unique_ptr<mlir::Pass> createTransfer5dStridedSliceTo4d();

/// Pass for transferring SpaceToDepth patterns to Conv2D.
std::unique_ptr<mlir::Pass> createTransferSpaceToDepthToConv2dPass();

/// Pass for merging nested concats and splitting duplicate inputs.
std::unique_ptr<mlir::Pass> createReplaceAdjacentOpPass();

/// Pass for optimizing contained concat patterns.
std::unique_ptr<mlir::Pass> createReplaceContainedConcatPass();

/// Pass for optimizing sibling concats by swapping and slicing.
std::unique_ptr<mlir::Pass> createOptimizeSiblingConcatPass();

/// Pass for removing paired reshapes across small qlinear chains.
std::unique_ptr<mlir::Pass> createRemovePairsAndMoveDownReshapePass();

/// Pass for merging BatchNormalization parameters into Conv.
std::unique_ptr<mlir::Pass> createMergeBatchnormToConvPass();

/// Pass for converting batch ReduceSum operations to reshape-optimized
/// ReduceSum (XMC).
std::unique_ptr<mlir::Pass> createBatchReductionToReshapeReductionPass();

/// Pass for deleting redundant Relu chains (XMC).
std::unique_ptr<mlir::Pass> createRemoveRedundantReluPass();

/// Pass for model-specific transpose decomposition (XMC).
std::unique_ptr<mlir::Pass> createReplaceNDimTransposePass();

/// Pass for transferring element-wise ops with non-4D shapes to 4D.
std::unique_ptr<mlir::Pass> createTransferOpShapeTo4dPass();

/// Pass for transferring 1D operations to 2D operations.
std::unique_ptr<mlir::Pass> createTransferOp1dToOp2dPass();

/// Pass for verifying Onnx ops before lowering to Krnl
std::unique_ptr<mlir::Pass> createONNXPreKrnlVerifyPass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<mlir::Pass> createLowerToKrnlPass();
std::unique_ptr<mlir::Pass> createLowerToKrnlPass(bool enableTiling,
    bool enableSIMD, bool enableParallel, bool enableFastMath,
    std::string opsForCall);
void configureOnnxToKrnlLoweringPass(bool reportOnParallel,
    bool parallelIsEnabled, std::string specificParallelOps, bool reportOnSimd,
    bool simdIsEnabled);
std::unique_ptr<mlir::Pass> createProcessScfParallelPrivatePass();
std::unique_ptr<mlir::Pass> createProcessKrnlParallelClausePass();

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
/// Add pass for lowering to Stablehlo IR.
std::unique_ptr<mlir::Pass> createLowerToStablehloPass();
std::unique_ptr<mlir::Pass> createLowerToStablehloPass(bool enableUnroll);
#endif

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();

/// Pass for legalizing quark-quantized models.
std::unique_ptr<mlir::Pass> createLegalizeQuarkQuantizedOpsPass();

namespace krnl {
/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass();
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass(bool parallelEnabled);

/// Pass for lowering Seq in Krnl dialect.
std::unique_ptr<mlir::Pass> createConvertSeqToMemrefPass();

/// Pass for lowering krnl.region operation.
std::unique_ptr<mlir::Pass> createLowerKrnlRegionPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel);

} // namespace krnl

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();
} // namespace onnx_mlir
#endif
