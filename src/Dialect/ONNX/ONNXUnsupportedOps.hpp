/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ONNXUnsupportedOps.hpp - ONNX Operations -------------===//
//
// Copyright 2023 The IBM Research Authors.
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

UNSUPPORTED_OPS(ONNXAdagradOp)
UNSUPPORTED_OPS(ONNXAdamOp)
UNSUPPORTED_OPS(ONNXArrayFeatureExtractorOp)
UNSUPPORTED_OPS(ONNXBatchNormalizationOp)
UNSUPPORTED_OPS(ONNXBinarizerOp)
UNSUPPORTED_OPS(ONNXBlackmanWindowOp)
UNSUPPORTED_OPS(ONNXCastMapOp)
UNSUPPORTED_OPS(ONNXCenterCropPadOp)
UNSUPPORTED_OPS(ONNXClipV11Op)
UNSUPPORTED_OPS(ONNXClipV12Op)
UNSUPPORTED_OPS(ONNXCol2ImOp)
UNSUPPORTED_OPS(ONNXConcatFromSequenceOp)
UNSUPPORTED_OPS(ONNXDetOp)
UNSUPPORTED_OPS(ONNXDictVectorizerOp)
UNSUPPORTED_OPS(ONNXFeatureVectorizerOp)
UNSUPPORTED_OPS(ONNXGradientOp)
UNSUPPORTED_OPS(ONNXGridSampleOp)
UNSUPPORTED_OPS(ONNXGroupNormalizationOp)
UNSUPPORTED_OPS(ONNXHammingWindowOp)
UNSUPPORTED_OPS(ONNXHannWindowOp)
UNSUPPORTED_OPS(ONNXImputerOp)
UNSUPPORTED_OPS(ONNXLabelEncoderOp)
UNSUPPORTED_OPS(ONNXLayerNormalizationOp)
UNSUPPORTED_OPS(ONNXLinearClassifierOp)
UNSUPPORTED_OPS(ONNXLinearRegressorOp)
UNSUPPORTED_OPS(ONNXLpPoolOp)
UNSUPPORTED_OPS(ONNXMaxPoolOp)
UNSUPPORTED_OPS(ONNXMaxUnpoolOp)
UNSUPPORTED_OPS(ONNXMelWeightMatrixOp)
UNSUPPORTED_OPS(ONNXMishOp)
UNSUPPORTED_OPS(ONNXMomentumOp)
UNSUPPORTED_OPS(ONNXMultinomialOp)
UNSUPPORTED_OPS(ONNXNegativeLogLikelihoodLossOp)
UNSUPPORTED_OPS(ONNXNormalizerOp)
UNSUPPORTED_OPS(ONNXPadV11Op)
UNSUPPORTED_OPS(ONNXPadV2Op)
UNSUPPORTED_OPS(ONNXRandomUniformLikeOp)
UNSUPPORTED_OPS(ONNXRandomUniformOp)
UNSUPPORTED_OPS(ONNXResizeV10Op)
UNSUPPORTED_OPS(ONNXResizeV11Op)
UNSUPPORTED_OPS(ONNXSequenceMapOp)
UNSUPPORTED_OPS(ONNXSVMClassifierOp)
UNSUPPORTED_OPS(ONNXSVMRegressorOp)
UNSUPPORTED_OPS(ONNXSoftmaxCrossEntropyLossOp)
UNSUPPORTED_OPS(ONNXSTFTOp)
UNSUPPORTED_OPS(ONNXStringNormalizerOp)
UNSUPPORTED_OPS(ONNXTfIdfVectorizerOp)
UNSUPPORTED_OPS(ONNXTreeEnsembleClassifierOp)
UNSUPPORTED_OPS(ONNXTreeEnsembleRegressorOp)
UNSUPPORTED_OPS(ONNXUniqueOp)
UNSUPPORTED_OPS(ONNXUpsampleV7Op)
UNSUPPORTED_OPS(ONNXZipMapOp)
