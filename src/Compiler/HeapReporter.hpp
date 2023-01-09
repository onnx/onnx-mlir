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
  HeapReporter(std::string logFilename, llvm::StringRef beforePasses,
      llvm::StringRef afterPasses);
  ~HeapReporter() override;

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  void reportBegin(const std::string &heading);
  void reportHeap(const std::string &heading);

  std::string logFilename;
  llvm::StringSet<> beforePassesSet;
  llvm::StringSet<> afterPassesSet;
  std::string command;
};

} // namespace onnx_mlir
