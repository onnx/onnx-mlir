/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- HeapReporter.hpp --------------------------===//
//
// Reports heap usage before and after compiler passes.
//
// See the explanation about how to run it in utils/onnx-mlir-report-heap.sh
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_HEAP_REPORTER_H
#define ONNX_MLIR_HEAP_REPORTER_H

#include <string>
#include <vector>

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringSet.h"

namespace onnx_mlir {

struct HeapReporter : public mlir::PassInstrumentation {
  HeapReporter(std::string logFilename, std::vector<std::string> beforePasses,
      std::vector<std::string> afterPasses);
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
#endif
