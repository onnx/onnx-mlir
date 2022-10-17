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
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
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
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "NNPACompilerUtils"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
extern llvm::cl::OptionCategory OnnxMlirOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::list<std::string> execNodesOnCpu;

llvm::cl::opt<NNPAEmissionTargetType> nnpaEmissionTarget(
    llvm::cl::desc("[Optional] Choose NNPA-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
        clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
        clEnumVal(EmitZNONE, "Do not emit NNPA-related target (default)")),
    llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
    llvm::cl::desc("Comma-separated list of node names in an onnx graph. The "
                   "specified nodes are forced to run on the CPU instead of "
                   "using the zDNN. The node name is an optional attribute "
                   "in onnx graph, which is `onnx_node_name` in ONNX IR"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
    llvm::cl::cat(OnnxMlirOptions)};

void addONNXToZHighPasses(
    mlir::PassManager &pm, ArrayRef<std::string> execNodesOnCpu) {
  for (unsigned i = 0; i < 3; i++) {
    // Repeat this process so that shape-related ops such as Shape, Expand,
    // Gather generated during RewriteONNXForZHigh will become constants.
    pm.addPass(onnx_mlir::createRewriteONNXForZHighPass(execNodesOnCpu));
    // Simplify shape-related ops, including ShapeOp-to-DimOp replacement,
    // constant propagation, shape inference and canonicalize.
    pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  }
  // Add instrumentation for Onnx Ops in the same way as onnx-mlir.
  if (instrumentZHighOps == "" || instrumentZHighOps == "NONE")
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXPass(
        instrumentONNXOps, instrumentControlBits.getBits()));
  pm.addPass(onnx_mlir::createONNXToZHighPass(execNodesOnCpu));
  pm.addPass(onnx_mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all zhigh ops were
  // generated.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Layout propagation at ZHighIR.
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::zhigh::createZHighLayoutPropagationPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Constant propagation at ZHighIR: constant stickify.
  // Only support BE machines.
  bool isBE = llvm::support::endian::system_endianness() ==
              llvm::support::endianness::big;
  if (isBE)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::zhigh::createZHighConstPropagationPass());
  // Remove common sub-expressions.
  pm.addPass(mlir::createCSEPass());
}

void normalizeMemRefsPasses(mlir::PassManager &pm) {
  // Introduce DummyOps for multiple dereferencing uses in a single op.
  // This is a bypass to avoid calling normalize-memrefs on a single op with
  // multiple dereferencing uses because normalize-memrefs does not support.
  pm.addPass(zlow::createZLowDummyOpForMultiDerefPass());
  // Normalize MemRefs.
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  // This is needed for removing dummy ops.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addPassesNNPA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, EmissionTargetType &emissionTarget) {
  // TODO: Develop and use determineInputIRLevel for NNPA
  // InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  // LLVM_DEBUG(llvm::dbgs() << "Adding NNPA passes" << std::endl;);
  if (emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm, onnxOpTransformReport, onnxOpTransformReport,
        /*target CPU*/ maccel.empty());

  if (emissionTarget >= EmitMLIR) {
    // Lower zAIU-compatible ONNX ops to ZHigh dialect where possible.
    addONNXToZHighPasses(pm, execNodesOnCpu);

    if (nnpaEmissionTarget >= EmitZHighIR)
      emissionTarget = EmitMLIR;
    else {
      pm.addPass(mlir::createCanonicalizerPass());
      // Add instrumentation for ZHigh Ops
      pm.addNestedPass<func::FuncOp>(zhigh::createInstrumentZHighPass());
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
      // Lower ONNX to Krnl, ZHigh to ZLow.
      addONNXToKrnlPasses(pm, optLevel, /*enableCSE*/ true,
          instrumentONNXSignature, ONNXOpStats);

      if (nnpaEmissionTarget >= EmitZLowIR)
        emissionTarget = EmitMLIR;
      else {
        // Partially lower Krnl ops to Affine dialect.
        addKrnlToAffinePasses(pm);
        // Normalize MemRefs.
        normalizeMemRefsPasses(pm);
        // Some Knrl ops, e.g. KrnlMemset, potentially exist and will be lowered
        // to Affine when its operands are normalized.
        addKrnlToAffinePasses(pm);
        // Optimizations at ZLow.
        pm.addPass(zlow::createZLowRewritePass());
        pm.addPass(mlir::createCanonicalizerPass());
        // Constant folding for std.alloc.
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createFoldStdAllocPass());
      }
    }
  }

  if (emissionTarget >= EmitLLVMIR)
    // Lower the remaining Krnl and all ZLow ops to LLVM dialect.
    addKrnlToLLVMPasses(pm, /*enableCSE=*/true, verifyInputTensors);
}

} // namespace onnx_mlir
