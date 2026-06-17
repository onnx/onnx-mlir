/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OnnxToMlirPasses.hpp ------------------------------===//
//
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_MLIR_PASSES_H
#define ONNX_MLIR_ONNX_TO_MLIR_PASSES_H

#include "src/Compiler/CompilerOptionEnums.hpp"
#include "src/Pass/Passes.hpp"
#include <string>

namespace mlir {
class ModuleOp;
class OpPassManager;
class PassManager;
} // namespace mlir

namespace onnx_mlir {

struct OnnxToMlirOptions {
  ONNXHybridTransformPassOptions hybrid;
  bool enableRemoveDqQAroundOp = false;
  bool enableRemoveBinary = false;
  bool enableFusePadIntoAvgpool = false;
  bool enableXMCPasses = false;

  bool disableBatchNormDecompose = false;
  bool enableUnsafeMathOptimizations = true;
  bool enableONNXHybridPass = true;
  bool enableConvOptPass = true;
  bool enableSimdDataLayout = false;
  bool disableSimdOption = false;

  bool enableMatmulAddFusion = true;
  bool enableMatmulToConv = true;
  bool enableRemovePairsReshape = false;

  int onnxOpTransformThreshold = 3;
  bool onnxOpTransformReport = false;
  int repeatOnnxTransform = 0;
  unsigned instrumentControlBits = 0;
  std::string instrumentOps;
  std::string instrumentSignatures = "NONE";
  std::string instrumentOnnxNode = "NONE";
  ProfileIRs profileIR = ProfileIRs::None;
  InstrumentStages instrumentStage = InstrumentStages::Onnx;
};

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr = false, OnnxToMlirOptions opts = {});

void addXmcMlirPasses(mlir::OpPassManager &pm, OnnxToMlirOptions opts = {});
} // namespace onnx_mlir

#endif
