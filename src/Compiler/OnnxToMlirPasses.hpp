#ifndef ONNX_MLIR_ONNX_TO_MLIR_PASSES_H
#define ONNX_MLIR_ONNX_TO_MLIR_PASSES_H

#include "src/Compiler/CompilerOptionEnums.hpp"
#include <string>

namespace mlir {
class ModuleOp;
class OpPassManager;
class PassManager;
} // namespace mlir

namespace onnx_mlir {

struct OnnxToMlirOptions {
  bool enableQuarkQuantizedLegalization = false;
  bool enableConvTransposeDecompose = false;
  bool enableConvTransposeDecomposeToPhasedConv = false;
  bool enableConvTranspose1dDecomposeToPhasedConv = false;
  bool enableInstanceNormDecompose = true;
  bool enableGroupNormDecompose = true;
  bool enableMatmulNBitsDecompose = false;
  bool enableGroupQueryAttentionDecompose = true;
  bool enableRemoveDqQAroundOp = false;
  bool enableRemoveBinary = false;
  bool enableFusePadIntoAvgpool = false;
  bool enableXMCPasses = false;
  bool enableSplitToSliceDecompose = false;

  bool disableBatchNormDecompose = false;
  bool disableRecomposeOption = false;
  bool enableONNXHybridPass = true;
  bool enableConvOptPass = true;
  bool enableSimdDataLayout = false;
  bool disableSimdOption = false;
  bool enablGAPToReduceMean = true;

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
