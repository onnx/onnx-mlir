/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ONNXUnsupportedOps.hpp - ONNX Operations -------------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// List of unsupported ops (alphabetically listed)
//
// Meaning ops for which shape inference is not implemented. To be fully
// implemented, some lowering to other dialects is additionally required, though
// not reflected here.
//
//===----------------------------------------------------------------------===//

#ifndef UNSUPPORTED_OPS
#error expected UNSUPPORTED_OPS macro
#endif

// Set of ONNX Ops that we currently do not support, listed alphabetically.
// Please remove from list when support is added.
UNSUPPORTED_OPS(ONNXAdagradOp)
UNSUPPORTED_OPS(ONNXAdamOp)
UNSUPPORTED_OPS(ONNXArrayFeatureExtractorOp)
UNSUPPORTED_OPS(ONNXBatchNormalizationOp)
UNSUPPORTED_OPS(ONNXCastMapOp)
UNSUPPORTED_OPS(ONNXCenterCropPadOp)
UNSUPPORTED_OPS(ONNXCol2ImOp)
UNSUPPORTED_OPS(ONNXConcatFromSequenceOp)
UNSUPPORTED_OPS(ONNXDetOp)
UNSUPPORTED_OPS(ONNXDeformConvOp)
UNSUPPORTED_OPS(ONNXDictVectorizerOp)
UNSUPPORTED_OPS(ONNXFeatureVectorizerOp)
UNSUPPORTED_OPS(ONNXGradientOp)
UNSUPPORTED_OPS(ONNXHannWindowOp)
UNSUPPORTED_OPS(ONNXImputerOp)
UNSUPPORTED_OPS(ONNXLabelEncoderOp)
UNSUPPORTED_OPS(ONNXLinearClassifierOp)
UNSUPPORTED_OPS(ONNXLinearRegressorOp)
UNSUPPORTED_OPS(ONNXLpPoolOp)
UNSUPPORTED_OPS(ONNXMaxPoolOp)
UNSUPPORTED_OPS(ONNXMaxUnpoolOp)
UNSUPPORTED_OPS(ONNXMelWeightMatrixOp)
UNSUPPORTED_OPS(ONNXMomentumOp)
UNSUPPORTED_OPS(ONNXMultinomialOp)
UNSUPPORTED_OPS(ONNXNegativeLogLikelihoodLossOp)
UNSUPPORTED_OPS(ONNXNormalizerOp)
UNSUPPORTED_OPS(ONNXRandomUniformLikeOp)
UNSUPPORTED_OPS(ONNXSequenceMapOp)
UNSUPPORTED_OPS(ONNXSVMClassifierOp)
UNSUPPORTED_OPS(ONNXSVMRegressorOp)
UNSUPPORTED_OPS(ONNXSoftmaxCrossEntropyLossOp)
UNSUPPORTED_OPS(ONNXSTFTOp)
UNSUPPORTED_OPS(ONNXStringNormalizerOp)
UNSUPPORTED_OPS(ONNXTfIdfVectorizerOp)
UNSUPPORTED_OPS(ONNXTreeEnsembleClassifierOp)
UNSUPPORTED_OPS(ONNXTreeEnsembleRegressorOp)
UNSUPPORTED_OPS(ONNXUpsampleV7Op)
UNSUPPORTED_OPS(ONNXZipMapOp)

// Set of ONNX Ops that are supported indirectly as every instances are
// converted/decomposed in one or more supported ONNX ops. Listed
// alphabetically.
#define CONVERTED_TO_SUPPORTED_OPS(_x) UNSUPPORTED_OPS(_x)
CONVERTED_TO_SUPPORTED_OPS(ONNXCastLikeOp)
CONVERTED_TO_SUPPORTED_OPS(ONNXClipV11Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXClipV12Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXClipV6Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXDFTV17Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXGridSampleV16Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXGroupNormalizationOp)
CONVERTED_TO_SUPPORTED_OPS(ONNXGroupNormalizationV18Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXPadV18Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXPadV13Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXPadV11Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXPadV2Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXResizeV10Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXResizeV11Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXResizeV13Op)
CONVERTED_TO_SUPPORTED_OPS(ONNXResizeV18Op)
