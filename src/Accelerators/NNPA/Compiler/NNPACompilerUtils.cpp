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

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "NNPACompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void addONNXToZHighPasses(mlir::PassManager &pm) {
  for (unsigned i = 0; i < 3; i++) {
    // Repeat this process so that shape-related ops such as Shape, Expand,
    // Gather generated during RewriteONNXForZHigh will become constants.
    pm.addPass(onnx_mlir::createRewriteONNXForZHighPass());
    // Simplify shape-related ops, including ShapeOp-to-DimOp replacement,
    // constant propagation, shape inference and canonicalize.
    pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  }

  // Profiling ZHighIR.
  unsigned instrumentActions = instrumentControlBits;
  if (profileIR == onnx_mlir::ProfileIRs::ZHigh) {
    instrumentStage = onnx_mlir::InstrumentStages::ZHigh;
    instrumentOps = "onnx.*,zhigh.*";
    // Enable the first three bits for InstrumentBeforOp, InstrumentAfterOp and
    // InstrumentReportTime.
    // Disable the last bit for InstrumentReportMemory because of its big
    // overhead. Users can optionally enable the last bit by using
    // --InstrumentReportMemory option.
    instrumentActions |= (1 << 3) - 1;
  }

  // Insert an instrumentation before lowering onnx to zhigh to get onnx level
  // profiling.
  if (instrumentStage == onnx_mlir::InstrumentStages::Onnx)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(instrumentOps, instrumentActions));

  pm.addPass(onnx_mlir::createONNXToZHighPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all zhigh ops were
  // generated.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Layout propagation at ZHighIR.
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::zhigh::createZHighLayoutPropagationPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Clip zhigh.Stick inputs if required. This is to avoid out-of-range of
  // dlfloat. Do constant propagation after clipping to remove ONNX ops used for
  // clipping such as ONNXMax if applicable.
  if (nnpaClipToDLFloatRange) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::zhigh::createZHighClipToDLFloatPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  }

  // After all optimizations, if there are still light-weight ops (e.g. add,
  // sub, ...) that are of `stick -> light-weight op -> unstick`, it's better to
  // use CPU instead of NNPA to avoid stick/unstick. CPU is efficient to handle
  // these ops, e.g vectorize the computation.
  if (nnpaEnableZHighToOnnx)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createZHighToONNXPass());

  // One more call to ONNX shape inference/canonicalization/... to update shape
  // if possible.
  if (enableONNXHybridPass) {
    // For starters only illustrating the new hybrid pass by replacing 3 passes
    // here. The plan is to replace most of the passes in addONNXToMLIRPasses.
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createONNXHybridTransformPass(!disableRecomposeOption));
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  }

  // Replace every DisposableElementsAttr with DenseElementsAttr.
  // ZHighConstPropagation currently assumes that DenseElementsAttr is used.
  pm.addPass(createScrubDisposablePass());

  // Constant propagation at ZHighIR: constant stickify.
  // Only support BE machines.
  bool isBE = llvm::support::endian::system_endianness() ==
              llvm::support::endianness::big;
  if (isBE)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::zhigh::createZHighConstPropagationPass());

  // Remove common sub-expressions.
  pm.addPass(mlir::createCSEPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  // Insert an instrumentation after lowering onnx to zhigh to get profiling
  // for onnx and zhigh ops.
  // Keep this pass at the end of this function.
  if (instrumentStage == onnx_mlir::InstrumentStages::ZHigh)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(instrumentOps, instrumentActions));
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
    mlir::PassManager &pm, EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) {
  // TODO: Develop and use determineInputIRLevel for NNPA
  // InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  // LLVM_DEBUG(llvm::dbgs() << "Adding NNPA passes" << std::endl;);
  if (emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());
    pm.addPass(onnx_mlir::createDevicePlacementPass(nnpaLoadDevicePlacementFile,
        nnpaSaveDevicePlacementFile, nnpaPlacementHeuristic));
  }

  if (emissionTarget >= EmitMLIR) {
    // Lower zAIU-compatible ONNX ops to ZHigh dialect where possible.
    addONNXToZHighPasses(pm);

    if (nnpaEmissionTarget >= EmitZHighIR)
      emissionTarget = EmitMLIR;
    else {
      pm.addPass(mlir::createCanonicalizerPass());

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
        // Optimizations at ZLow that needs affine map in MemRef.
        pm.addPass(zlow::createZLowRewritePass());
        pm.addPass(mlir::createCanonicalizerPass());
        // Normalize MemRefs.
        normalizeMemRefsPasses(pm);
        // Some Knrl ops, e.g. KrnlMemset, potentially exist and will be lowered
        // to Affine when its operands are normalized.
        addKrnlToAffinePasses(pm);
        // Optimizations at ZLow after normalizing MemRefs.
        pm.addPass(zlow::createZLowRewritePass());
        pm.addPass(mlir::createCanonicalizerPass());
        // Constant folding for std.alloc.
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createFoldStdAllocPass());
      }
      // Insert an instrumentation after lowering zhigh to zlow to get profiling
      // for zlow ops
      if (instrumentStage == onnx_mlir::InstrumentStages::ZLow)
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentPass(
            instrumentOps, instrumentControlBits));
    }
  }

  if (emissionTarget >= EmitLLVMIR)
    // Lower the remaining Krnl and all ZLow ops to LLVM dialect.
    addKrnlToLLVMPasses(pm, outputNameNoExt, /*enableCSE=*/true);
}

} // namespace onnx_mlir
