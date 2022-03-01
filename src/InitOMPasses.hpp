/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"

using namespace onnx_mlir;

namespace onnx_mlir {

void initOMPasses(int optLevel) {
  // All passes implemented within onnx-mlir should register within this
  // function to make themselves available as a command-line option.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXOpTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDecomposeONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createShapeInferencePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConstPropONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createElideConstantValuePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXPreKrnlVerifyPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createKrnlEnableMemoryPoolPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createKrnlBundleMemoryPoolsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createKrnlOptimizeMemoryPoolsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertKrnlToAffinePass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToKrnlPass(optLevel);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createElideConstGlobalValuePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToLLVMPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDisconnectKrnlDimFromAllocPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerKrnlShapePass();
  });
}
} // namespace onnx_mlir
