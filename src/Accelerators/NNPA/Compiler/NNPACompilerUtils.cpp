/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPACompilerUtils.cpp ---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Utilities for  NNPA
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/OMNNPAOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"

#define DEBUG_TYPE "NNPACompiler"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

extern llvm::cl::OptionCategory OnnxMlirOptions;

llvm::cl::opt<NNPAEmissionTargetType> nnpaEmissionTarget(
    llvm::cl::desc("[Optional] Choose Z-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
        clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
        clEnumVal(EmitZNONE, "Do not emit Z-related target (default)")),
    llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
    llvm::cl::desc("Comma-separated list of node names in an onnx graph. The "
                   "specified nodes are forced to run on the CPU instead of "
                   "using the zDNN. The node name is an optional attribute "
                   "in onnx graph, which is `onnx_node_name` in ONNX IR"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
    llvm::cl::cat(OnnxMlirOptions)};

namespace onnx_mlir {

void addONNXToZHighPasses(
    mlir::PassManager &pm, ArrayRef<std::string> execNodesOnCpu) {
  pm.addPass(onnx_mlir::createRewriteONNXForZHighPass(execNodesOnCpu));
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addNestedPass<FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  // Add instrumentation for Onnx Ops in the same way as onnx-mlir.
  if (instrumentZHighOps == "" || instrumentZHighOps == "NONE")
    pm.addNestedPass<FuncOp>(onnx_mlir::createInstrumentONNXPass());
  pm.addPass(onnx_mlir::createONNXToZHighPass(execNodesOnCpu));
  pm.addPass(onnx_mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all zhigh ops were
  // generated.
  pm.addNestedPass<FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Layout propagation at ZHighIR.
  pm.addNestedPass<FuncOp>(
      onnx_mlir::zhigh::createZHighLayoutPropagationPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Constant propagation at ZHighIR: constant stickify.
  // Only support BE machines.
  bool isBE = llvm::support::endian::system_endianness() ==
              llvm::support::endianness::big;
  if (isBE)
    pm.addNestedPass<FuncOp>(
        onnx_mlir::zhigh::createZHighConstPropagationPass());
}

void addZHighToZLowPasses(mlir::PassManager &pm, int optLevel) {
  // Add instrumentation for ZHigh Ops
  pm.addNestedPass<FuncOp>(onnx_mlir::zhigh::createInstrumentZHighPass());
  pm.addPass(onnx_mlir::zhigh::createZHighToZLowPass(optLevel));
  pm.addNestedPass<FuncOp>(onnx_mlir::createLowerKrnlShapePass());
  pm.addNestedPass<FuncOp>(onnx_mlir::createDisconnectKrnlDimFromAllocPass());
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addAllToLLVMPasses(mlir::PassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained in
  // https://mlir.llvm.org/docs/BufferDeallocationInternals.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());
  if (enableMemoryBundling) {
    pm.addNestedPass<FuncOp>(krnl::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<FuncOp>(krnl::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(krnl::createKrnlOptimizeMemoryPoolsPass());
  }

  pm.addNestedPass<FuncOp>(mlir::createConvertSCFToCFPass());
  pm.addPass(onnx_mlir::zlow::createZLowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addPassesNNPA(mlir::OwningOpRef<ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType &emissionTarget,
    NNPAEmissionTargetType nnpaEmissionTarget,
    ArrayRef<std::string> execNodesOnCpu) {
  // TODO: Develop and use determineInputIRLevel for NNPA
  // InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  // LLVM_DEBUG(llvm::dbgs() << "Adding NNPA passes" << std::endl;);
  if (emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm);

  if (emissionTarget >= EmitMLIR) {
    // Lower zAIU-compatible ONNX ops to ZHigh dialect where possible.
    addONNXToZHighPasses(pm, execNodesOnCpu);

    if (nnpaEmissionTarget >= EmitZHighIR)
      emissionTarget = EmitMLIR;
    else {
      pm.addPass(mlir::createCanonicalizerPass());
      // Add instrumentation for remaining Onnx Ops
      if (instrumentZHighOps != "" && instrumentZHighOps != "NONE")
        pm.addNestedPass<FuncOp>(createInstrumentONNXPass());
      // Lower all ONNX and ZHigh ops.
      std::string optStr = getCompilerOption(OptionKind::CompilerOptLevel);
      OptLevel optLevel = OptLevel::O0;
      if (optStr == "-O0")
        optLevel = OptLevel::O0;
      else if (optStr == "-O1")
        optLevel = OptLevel::O1;
      else if (optStr == "-O2")
        optLevel = OptLevel::O2;
      else if (optStr == "-O3")
        optLevel = OptLevel::O3;
      addZHighToZLowPasses(pm, optLevel); // Constant folding for std.alloc.
      pm.addNestedPass<FuncOp>(onnx_mlir::createFoldStdAllocPass());

      if (nnpaEmissionTarget >= EmitZLowIR)
        emissionTarget = EmitMLIR;
      else {
        // Partially lower Krnl ops to Affine dialect.
        addKrnlToAffinePasses(pm);
      }
    }
  }

  if (emissionTarget >= EmitLLVMIR)
    // Lower the remaining Krnl and all ZLow ops to LLVM dialect.
    addAllToLLVMPasses(pm);
}

} // namespace onnx_mlir
