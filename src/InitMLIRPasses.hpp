/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

void initMLIRPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToSCFPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLowerAffinePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertSCFToCFPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToLLVMPass();
  });
}
} // namespace onnx_mlir
