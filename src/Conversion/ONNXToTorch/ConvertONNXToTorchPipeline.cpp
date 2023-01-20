/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTorchPipeline.cpp - ONNX dialects to Torch lowering
// pipeline -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ================================================================================================
//
// This file registers the pipeline for converting ONNX to Torch Backend IR
//
//===-----------------------------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

void registerONNXFrontendToTorchBackendPasses() {
  PassPipelineRegistration<>("convert-onnx-to-torch-pipeline",
      "Pipeline converting ONNX to Torch dialect.",
      onnx_mlir::createONNXFrontendToTorchBackendPasses);
}

void createONNXFrontendToTorchBackendPasses(OpPassManager &pm) {
  pm.addPass(createLowerToTorchPass());
  pm.addPass(createFuncTorchTypeConversionPass());
  pm.addPass(createFinalizingTorchTypeConversionPass());
  pm.addPass(createEraseONNXEntryPointPass());
}

} // namespace onnx_mlir
