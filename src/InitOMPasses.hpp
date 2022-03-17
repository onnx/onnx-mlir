/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- InitOMPasses.hpp - Init onnx-mlir passes  ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"

#ifdef __NNPA__
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#endif

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
    return krnl::createConvertKrnlToAffinePass();
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
    return onnx_mlir::createONNXToZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createRewriteONNXForZHighPass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighToZLowPass(optLevel);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowRewritePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zlow::createZLowToLLVMPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createFoldStdAllocPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighConstPropagationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::zhigh::createZHighLayoutPropagationPass();
  });

#endif
}

} // namespace onnx_mlir
