/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- RegisterPasses.cpp -------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// Beware that we cannot access CompilerOptions or other command line options
// the functions below because they are called before command line options are
// parsed, because passes must be registered first as they instruct command
// line parsing: registered passes can be expressed as command line flags.
//
// In particular the --O flag value is passed as a function argument optLevel
// so we can avoid reading the OptimizationLevel command-line option here.
//
//===----------------------------------------------------------------------===//

#include "RegisterPasses.hpp"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerPasses.hpp"

#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

void registerOMPasses(int optLevel) {
  // All passes implemented within onnx-mlir should register within this
  // function to make themselves available as a command-line option.

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createScrubDisposablePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXOpTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDecomposeONNXToONNXPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRecomposeONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvOptONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXHybridTransformPass(/*recompose ops*/ true);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createShapeInferencePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConstPropONNXToONNXPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createInstrumentPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentONNXSignaturePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSetONNXNodeNamePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXPreKrnlVerifyPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToAffinePass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToKrnlPass(/*enableTiling*/ optLevel >= 3,
        /*enableSIMD, should consider disableSimdOption*/ optLevel >= 3,
        /*enableParallel*/ false,
        /*opsForCall*/ "");
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createProcessScfParallelPrivatePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertSeqToMemrefPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createLowerKrnlRegionPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToLLVMPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSimplifyShapeRelatedOpsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createStandardFuncReturnPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXDimAnalysisPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertONNXToTOSAPass();
  });

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToStablehloPass();
  });
#endif
}

void registerMLIRPasses() {
  registerTransformsPasses();
  affine::registerAffinePasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  bufferization::registerBufferizationPasses();

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
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createPrintOpStatsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertSCFToOpenMPPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertOpenMPToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFinalizeMemRefToLLVMConversionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
}

void registerPasses(int optLevel) {
  registerMLIRPasses();

  registerOMPasses(optLevel);

  // Register passes for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerPasses(optLevel);
}

} // namespace onnx_mlir
