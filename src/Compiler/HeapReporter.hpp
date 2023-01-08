/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- HeapReporter.hpp --------------------------===//
//
// Reports heap usage before and after compiler passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringSet.h"

namespace onnx_mlir {

struct HeapReporter : public mlir::PassInstrumentation {
  HeapReporter(std::string logFilename, std::string beforePasses, std::string afterPasses);
  ~HeapReporter() override;

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  std::string logFilename;
  llvm::StringSet beforePasses;
  llvm::StringSet afterPasses;
};

} // namespace onnx_mlir
