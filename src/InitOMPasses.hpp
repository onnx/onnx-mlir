/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"

#ifdef __NNPA__
#include "src/Accelerators/NNPA/Pass/DLCPasses.hpp"
#endif

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

#ifdef __NNPA__
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createONNXToZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createRewriteONNXForZHighPass();
  });

  // mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
  //   return mlir::createZHighConstPropagationPass();
  // });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZHighToZLowPass(optLevel);
  });

  // mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
  //   return mlir::createZLowRewritePass();
  // });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZLowToLLVMPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFoldStdAllocPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZHighLayoutPropagationPass();
  });
#endif
}
} // namespace onnx_mlir
