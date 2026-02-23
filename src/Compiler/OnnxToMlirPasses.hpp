#ifndef ONNX_MLIR_ONNX_TO_MLIR_PASSES_H
#define ONNX_MLIR_ONNX_TO_MLIR_PASSES_H

#include "src/Compiler/CompilerOptionEnums.hpp"
#include <string>

namespace mlir {
class PassManager;
} // namespace mlir

namespace onnx_mlir {

struct OnnxToMlirOptions {
  bool enableQuarkQuantizedLegalization = false;
  bool enableConvTransposeDecompose = false;
  bool enableConvTransposeDecomposeToPhasedConv = false;
  bool enableConvTranspose1dDecomposeToPhasedConv = false;
  bool enableInstanceNormDecompose = true;
  bool enableMatmulNBitsDecompose = false;
  bool enableRemoveDqQAroundOp = false;
  bool enableRemoveBinary = false;
  bool enableFusePadIntoAvgpool = false;
  bool enableXMCPasses = true;
  bool enableSplitToSliceDecompose = false;

  bool disableRecomposeOption = false;
  bool enableONNXHybridPass = true;
  bool enableConvOptPass = true;
  bool enableSimdDataLayout = false;
  bool disableSimdOption = false;
  int onnxOpTransformThreshold = 3;
  bool onnxOpTransformReport = false;
  int repeatOnnxTransform = 0;
  unsigned instrumentControlBits = 0;
  std::string instrumentOps;
  std::string instrumentSignatures = "NONE";
  std::string instrumentOnnxNode = "NONE";
  ProfileIRs profileIR = ProfileIRs::None;
  InstrumentStages instrumentStage = InstrumentStages::Onnx;

  // XMC debug options: add PassInstrumentation for timing, change detection,
  // and optional MLIR dump after each XMC pass. Enabled by default so both
  // onnx-mlir.exe and vaiml-lite-cli get debug info without extra config.
  bool dumpMlirAfterEachXmcPass = true;
  std::string xmcOutputDir = ".";
};

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr = false, OnnxToMlirOptions opts = {});

/// Add all XMC passes to a PassManager. When dumpMlirAfterEachXmcPass is true,
/// a PassInstrumentation is attached for timing, change detection, and MLIR
/// dumping — no separate "debug runner" is needed.
void addXmcMlirPasses(mlir::PassManager &pm, OnnxToMlirOptions opts = {});

} // namespace onnx_mlir

#endif