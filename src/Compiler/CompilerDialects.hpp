/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerDialects.hpp ------------------------===//

#ifndef ONNX_MLIR_COMPILER_DIALECTS_H
#define ONNX_MLIR_COMPILER_DIALECTS_H

#include "src/Accelerators/Accelerator.hpp"

#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/ArrayRef.h"

namespace onnx_mlir {

// Adds the mlir and onnx-mlir dialects needed to compile end to end.
// Initializes accelerator(s) if required.
mlir::DialectRegistry registerDialects(
    llvm::ArrayRef<accel::Accelerator::Kind> accels);

} // namespace onnx_mlir
#endif
