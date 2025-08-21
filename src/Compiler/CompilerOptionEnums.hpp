#ifndef ONNX_MLIR_COMPILER_OPTION_ENUMS_H
#define ONNX_MLIR_COMPILER_OPTION_ENUMS_H

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {

typedef enum {
  // clang-format off
  None,
  Onnx
  APPLY_TO_ACCELERATORS(ACCEL_INSTRUMENTSTAGE_ENUM)
  // clang-format on
} InstrumentStages;

using ProfileIRs = InstrumentStages;
} // namespace onnx_mlir

#endif // ONNX_MLIR_COMPILER_OPTION_ENUMS_H
