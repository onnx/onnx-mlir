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
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PASSES_H
#define ONNX_MLIR_PASSES_H

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class Pass;
} // namespace mlir

namespace onnx_mlir {

#define GEN_PASS_DECL
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
#undef GEN_PASS_DECL

[[nodiscard]] std::optional<ONNXHybridTransformPassOptions>
parseONNXHybridTransformPassOptions(const std::string &options);

/// Pass for removing DisposableElementsAttr attributes.
std::unique_ptr<mlir::Pass> createScrubDisposablePass(bool closeAfter = true);

/// Pass for ONNX graph level optimization
std::unique_ptr<mlir::Pass> createONNXOpTransformPass();
std::unique_ptr<mlir::Pass> createONNXOpTransformPass(int threshold,
    bool report, bool targetCPU, bool enableSimdDataLayoutOpt,
    bool enableConvOptPass, bool enableRecomposeOptPass);

std::unique_ptr<mlir::Pass> createRecomposeONNXToONNXPass(
    const std::string &target = "", bool enableRotaryEmbeddingRecompose = false,
    bool enableReduceL2Recompositions = false);

std::unique_ptr<mlir::Pass> createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt = false);

std::unique_ptr<mlir::Pass> createShapeInferencePass();

// To configure ConstPropONNXToONNXPass at program start.
void configureConstPropONNXToONNXPass(bool roundFPToInt, int expansionBound,
    llvm::ArrayRef<std::string> disabledPatterns, bool constantPropIsDisabled);

// To configure whether BatchNorm decomposition is disabled in canonicalization.
void configureBatchNormCanonicalization(bool disableBatchNormDecompose);

// Configure patterns that are not numerically safe.
void configureUnsafeMathCanonicalization(bool enableUnsafeMathOptimizations);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass(
    bool enableQDQ = false);

std::unique_ptr<mlir::Pass> createQDQCanonicalizePass(
    bool removeBinary = false, bool removeQDQAroundOps = false);

std::unique_ptr<mlir::Pass> createFoldQuantizedBinary();

/// Converts quantized Div / Mul by a scalar quantized constant into a
/// single XCOMPILERRequantize op. XMC analogue of xcompiler's
/// TransferScalarConstInputDivToRequantizePass.
std::unique_ptr<mlir::Pass> createTransferScalarConstInputDivToRequantizePass();

std::unique_ptr<mlir::Pass> createONNXCSEPass();

std::unique_ptr<mlir::Pass> createQuantTypesPass();

std::unique_ptr<mlir::Pass> createFixNegScalePass();

std::unique_ptr<mlir::Pass> createInferTensorNames();

std::unique_ptr<mlir::Pass> createCanonicalizeWithResultNamesPass();

#ifdef ONNX_MLIR_ENABLE_KRNL
/// Pass for instrument the ops in specific stage.
std::unique_ptr<mlir::Pass> createInstrumentPass();
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions);
/// Pass for instrument cleanup.
std::unique_ptr<mlir::Pass> createInstrumentCleanupPass();
#endif

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass(
    const std::string opPattern, const std::string nodePattern);

/// Pass for simplifying shape-related ONNX operations.
std::unique_ptr<mlir::Pass> createSimplifyShapeRelatedOpsPass(
    bool disableCastOpCanonicalizations = false,
    bool enablGAPToReduceMean = true);

/// Pass for replacing ONNXReturnOp with func::ReturnOp.
std::unique_ptr<mlir::Pass> createStandardFuncReturnPass();

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

/// Pass for fusing Add(Conv(A, X, none), constant) -> Conv(A, X, bias).
std::unique_ptr<mlir::Pass> createConvWithBiasPass();

/// Pass for fusing Add(MatMul(A, B), constant bias) -> onnx.XFEMatMulBias when
/// the bias constant has one value per MatMul output channel (last dim of B).
std::unique_ptr<mlir::Pass> createFuseMatMulAddToXFEMatMulBiasPass();

/// Pass for removing redundant reshape operations around element-wise ops.
std::unique_ptr<mlir::Pass> createRemoveRedundantReshapePass();

/// Pass for lowering Reduce operations to Pool operations.
std::unique_ptr<mlir::Pass> createLowerReduceToPoolPass();

/// Pass for removing semantically useless operations.
std::unique_ptr<mlir::Pass> createRemoveSemanticallyUselessOpsPass();

/// Pass for removing useless pool operations (kernel_shape and strides all 1s).
std::unique_ptr<mlir::Pass> createRemoveUselessQLinearPoolPass();

/// Pass for replacing quantized HardSigmoid with XCOMPILERFusedEltwise.
std::unique_ptr<mlir::Pass> createReplaceHsigmoidAndHswishPass();

std::unique_ptr<mlir::Pass> createRecomposeHardSigmoidPass();
std::unique_ptr<mlir::Pass> createTransferSoftmaxAxisToLastPass();

/// Pass for replacing quantized Erf-based GELU subgraph with
/// XCOMPILERFusedEltwise(GELU).
std::unique_ptr<mlir::Pass> createReplaceErfToGeluPass();

/// Pass for replacing quantized tanh-approximation GELU subgraph with
/// onnx.Gelu(approximate="tanh").
std::unique_ptr<mlir::Pass> createReplaceTanhToGeluPass();

/// Pass for replacing quantized Sigmoid with XCOMPILERFusedEltwise
/// QLINEARSIGMOID.
std::unique_ptr<mlir::Pass> createReplaceQDQSigmoidPass();

/// Pass for transferring ReduceMean/Sum operations to Conv operations.
std::unique_ptr<mlir::Pass> createTransferReduceMeanSumToConvPass();

/// Reshape Reduce(Sum/Mean/Max/Min) H/W-axis to the C-axis via transposes.
std::unique_ptr<mlir::Pass> createTransferReduceHdimToReduceCdimPass();

/// Reshape Reduce(Sum/Mean/Max/Min) so its input is rank-4 + keep_dims=true.
std::unique_ptr<mlir::Pass> createReplaceQDQReductionPass();

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

/// Pass for converting XFEConv to XCOMPILERDepthwiseConv when group ==
/// input_channels.
std::unique_ptr<mlir::Pass> createConvertXFEConvToDepthwiseConvPass();

/// Pass for fusing Conv + Activation patterns into conv ops with activation
/// attribute (XFEConv, XFEConvTranspose, XCOMPILERDepthwiseConv).
std::unique_ptr<mlir::Pass> createFuseConvActivationPass();

/// Pass for normalizing conv activation attributes into hardware-compatible
/// form (LEAKYRELU/PRELU/HSIGMOID) matching xcompiler behavior.
std::unique_ptr<mlir::Pass> createNormalizeConvActivationPass();

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

/// Pass for fusing onnx.Clip(quantized->f32)+Cast(f32->uint) into CLAMP (XMC).
std::unique_ptr<mlir::Pass> createReplaceQDQClipCastPass();

/// Pass for fusing quantized eltwise+activation patterns (XMC).
std::unique_ptr<mlir::Pass> createReplaceQDQEltwisePass();

/// Pass for lowering quantized onnx.Tile to XCOMPILERFusedEltwise ADD (XMC).
std::unique_ptr<mlir::Pass> createReplaceQuantizedTileToAddPass();

/// Pass for replacing quantized XFEResize with 1x1 spatial input by a
/// broadcasting onnx.Add against a splat zero_point constant (XMC).
std::unique_ptr<mlir::Pass> createReplaceQDQResizePass();

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

/// Pass for deleting redundant Relu/LeakyRelu-like ops: collapse Relu chains,
/// drop Relu/LeakyRelu on non-negative inputs, fold adjacent LeakyRelu (XMC).
std::unique_ptr<mlir::Pass> createRemoveRedundantReluLikeOpsPass();

/// Pass for optimizing requantization in ONNX operations (XMC).
std::unique_ptr<mlir::Pass> createOptimizeOnnxRequantizationPass();

/// Pass for folding DQ-Binary-Q chains into quantization parameters (XMC).
std::unique_ptr<mlir::Pass> createDQBinaryQOptPass();

/// Pass for converting back-to-back quant.scast pairs to XCOMPILERRequantize.
std::unique_ptr<mlir::Pass> createConvertSCastPairToRequantizePass();

/// Pass for inserting a placeholder XCOMPILERRequantize on
/// `producer -> quant.scast -> DequantizeLinear` output edges when the
/// quantized producer has multiple fanouts (XMC).
std::unique_ptr<mlir::Pass> createAddRequantForOutputConvPass();

/// Pass for folding equal Q(DQ(x)) and inserting XCOMPILERRequantize between
/// DQ -> Q pairs whose quantization parameters differ. Runs before
/// QuantTypesPass on the f32 boundary.
std::unique_ptr<mlir::Pass> createConvertQDQToRequantizePass();

/// Post-quant-types pass that unifies f32 <-> !quant.uniform element types
/// across pure data-flow ops (Reshape/Transpose/Squeeze/Unsqueeze/Flatten/
/// Identity/DepthToSpace/SpaceToDepth/ReverseSequence). The f32 side is
/// retyped in place to the quant type; scast ops are left untouched.
std::unique_ptr<mlir::Pass> createPropagateQuantTypeThroughDataFlowPass();

/// Post-`PropagateQuantTypeThroughDataFlow` pass. For data-flow ops with
/// mismatched operand/result quant types (Group B), inserts an
/// XCOMPILERRequantize op directly between the producer's quant value and
/// the data-flow op (Concat: per-operand) or between the data-flow op's
/// result and the consumer (single-input ops). Replicates the union of
/// `OptimizeOnnxRequantizationPass`'s op set plus the extra shape-only ops
/// Squeeze / SqueezeV11 / Unsqueeze / UnsqueezeV11 / Flatten / Identity.
std::unique_ptr<mlir::Pass> createXmcRequantizePass();

/// Pass for splitting group convolutions (XMC).
std::unique_ptr<mlir::Pass> createSplitGroupConvPass();

/// Pass for converting MatMul to XFEConv (XMC).
std::unique_ptr<mlir::Pass> createConvertMatMulToXFEConvPass();

/// Pass for model-specific transpose decomposition (XMC).
std::unique_ptr<mlir::Pass> createReplaceNDimTransposePass();

/// Pass for transferring element-wise ops with non-4D shapes to 4D.
std::unique_ptr<mlir::Pass> createTransferOpShapeTo4dPass();

/// Pass for transferring 1D operations to 2D operations.
std::unique_ptr<mlir::Pass> createTransferOp1dToOp2dPass();

/// Pass for transferring Scale operations to DepthwiseConv2D operations.
std::unique_ptr<mlir::Pass> createTransferScaleToDwConv2dPass();

#ifdef ONNX_MLIR_ENABLE_KRNL
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
#endif

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
/// Add pass for lowering to Stablehlo IR.
std::unique_ptr<mlir::Pass> createLowerToStablehloPass();
std::unique_ptr<mlir::Pass> createLowerToStablehloPass(bool enableUnroll);
#endif

#ifdef ONNX_MLIR_ENABLE_KRNL
/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();
#endif

/// Pass for legalizing quark-quantized models.
std::unique_ptr<mlir::Pass> createLegalizeQuarkQuantizedOpsPass();

#ifdef ONNX_MLIR_ENABLE_KRNL
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
#endif

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();
} // namespace onnx_mlir
#endif
