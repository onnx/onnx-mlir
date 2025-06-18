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
#include "mlir/Conversion/Passes.h"
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
#include "src/Accelerators/NNPA/Compiler/ZHighDisposableGarbageCollector.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "NNPACompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void configurePassesNNPA() {
  // z16 does not support for hardware saturation.
  // So, force its usage to compiler generated sticks.
  if (!nnpaDisableSaturation && isLessEqualNNPALevel(NNPALevel::M14))
    nnpaDisableCompilerStickUnstick = false;

  // Configure ONNXToZHighLoweringPass.
  bool isDynQuant = !nnpaQuantDynamic.empty();
  // Default/auto mode: symmetric for weighs and asymmetric for activations.
  bool isActivationSym = false;
  bool isWeightSym = true;
  std::vector<std::string> quantOpTypes;
  if (isDynQuant) {
    // Set options for activations and weights if they are given.
    // When auto mode is specified, the other specified options are ignored.
    if (!llvm::is_contained(nnpaQuantDynamic, NNPAQuantOptions::autoQuantOpt)) {
      for (unsigned i = 0; i < nnpaQuantDynamic.size(); ++i) {
        switch (nnpaQuantDynamic[i]) {
        case NNPAQuantOptions::symWeight:
          isWeightSym = true;
          break;
        case NNPAQuantOptions::asymWeight:
          isWeightSym = false;
          break;
        case NNPAQuantOptions::symActivation:
          isActivationSym = true;
          break;
        case NNPAQuantOptions::asymActivation:
          isActivationSym = false;
          break;
        default:
          llvm_unreachable("Unsupported quantization options");
          break;
        }
      }
    }
    if (!isWeightSym) {
      // TODO: Support asymmetric quantiation for weights.
      llvm::outs()
          << "Asymmetric quantization for weights is not yet supported. "
             "Turning off quantization.\n";
      isDynQuant = false;
    }
    if (nnpaQuantOpTypes.empty()) {
      quantOpTypes.emplace_back("MatMul");
    } else {
      quantOpTypes = nnpaQuantOpTypes;
    }
  }
  // Set the proper instrumentation stage before we add any passes.
  if (profileIR == onnx_mlir::ProfileIRs::ZHigh)
    instrumentStage = onnx_mlir::InstrumentStages::ZHigh;

  configureONNXToZHighLoweringPass(optReport == OptReport::NNPAUnsupportedOps,
      isDynQuant, isActivationSym, isWeightSym, quantOpTypes);
}

void addONNXToZHighPasses(mlir::PassManager &pm) {
  for (unsigned i = 0; i < 3; i++) {
    // Repeat this process so that shape-related ops such as Shape, Expand,
    // Gather generated during RewriteONNXForZHigh will become constants.
    pm.addPass(onnx_mlir::createRewriteONNXForZHighPass());
    // Simplify shape-related ops, including ShapeOp-to-DimOp replacement,
    // constant propagation, shape inference and canonicalize.
    pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  }

  // Lowering ONNX to ZHigh.
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

  // Experimental feature: Decompose stick/unstick into two phases: layout
  // transform and data conversion. Do some optimizations after decomposing.
  // Then, recompose again layout and data conversion if they are not optimized.
  if (nnpaEnableZHighDecomposeStickUnstick) {
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::zhigh::createZHighDecomposeStickUnstickPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::zhigh::createZHighRecomposeToStickUnstickPass());
  }

  // After all optimizations, if there are still light-weight ops (e.g. add,
  // sub, ...) that are of `stick -> light-weight op -> unstick`, it's better to
  // use CPU instead of NNPA to avoid stick/unstick. CPU is efficient to handle
  // these ops, e.g vectorize the computation.
  if (!nnpaDisableZHighToOnnx)
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createZHighToONNXPass());

  // Constant propagation at ZHighIR: constant stickify.
  // Only support BE machines.
  bool isBE = llvm::endianness::native == llvm::endianness::big;
  if (isBE)
    pm.addPass(onnx_mlir::zhigh::createZHighConstPropagationPass());

  // Remove common sub-expressions.
  pm.addPass(mlir::createCSEPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  // Replace every DisposableElementsAttr with DenseElementsAttr.
  pm.addPass(onnx_mlir::zhigh::createZHighScrubDisposablePass());

  // Profiling ZHighIR.
  unsigned instrumentActions = instrumentControlBits;
  if (profileIR == onnx_mlir::ProfileIRs::ZHigh) {
    assert(instrumentStage == onnx_mlir::InstrumentStages::ZHigh &&
           "expected set to this");
    instrumentOps = "onnx.*,zhigh.*";
    // Enable the first three bits for InstrumentBeforOp, InstrumentAfterOp and
    // InstrumentReportTime.
    // Disable the last bit for InstrumentReportMemory because of its big
    // overhead. Users can optionally enable the last bit by using
    // --InstrumentReportMemory option.
    instrumentActions |= (1 << 3) - 1;
    // Also enable instrumentation of signatures.
    instrumentSignatures = "onnx.*,zhigh.*";
  }

  // Insert an instrumentation after lowering onnx to zhigh to get profiling /
  // signatures for onnx and zhigh ops. Keep this pass at the end of this
  // function. Add createInstrument (timing) second so that it will guarantee
  // not to include timing of the signature printing.
  if (hasSignatureInstrumentation(onnx_mlir::InstrumentStages::ZHigh))
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXSignaturePass(
        instrumentSignatures, instrumentOnnxNode));
  if (hasInstrumentation(onnx_mlir::InstrumentStages::ZHigh))
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

  // Disable constprop rules:
  // - add(add(x, c), y) to add(add(x, y), c)
  // - add(x, add(y, c)) to add(add(x, y), c)
  // because in foundation models we have add(add(matmul(x, z), c), y), and we
  // want to keep c near matmul so that add(matmul(x, z), c) will be run on zAIU
  // as one call.
  onnxConstPropDisablePatterns.emplace_back("AddConstAssociative2");
  onnxConstPropDisablePatterns.emplace_back("AddConstAssociative3");

  // Override pass configurations.
  configurePasses();

  // LLVM_DEBUG(llvm::dbgs() << "Adding NNPA passes" << std::endl;);
  if (emissionTarget >= EmitONNXIR) {
    pm.addInstrumentation(
        std::make_unique<onnx_mlir::zhigh::ZHighDisposableGarbageCollector>(
            pm.getContext()));
    addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty(),
        /*donotScrubDisposableElementsAttr*/ true);
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
      addONNXToKrnlPasses(pm, optLevel, /*enableCSE*/ true, ONNXOpStats);

      if (nnpaEmissionTarget >= EmitZLowIR)
        emissionTarget = EmitMLIR;
      else {
        // Partially lower Krnl ops to Affine dialect.
        addKrnlToAffinePasses(pm);
        // Optimizations at ZLow that needs affine map in MemRef.
        pm.addPass(zlow::createZLowRewritePass());
        // Late generation of code for stick/unstick, needed to be after a
        // ZLowRewrite pass.
        if (!nnpaDisableCompilerStickUnstick)
          pm.addPass(zlow::createZLowStickExpansionPass(enableParallel));
        pm.addPass(mlir::createCanonicalizerPass());
        // Normalize MemRefs.
        normalizeMemRefsPasses(pm);
        // Some Krnl ops, e.g. KrnlMemset, potentially exist and will be lowered
        // to Affine when its operands are normalized.
        addKrnlToAffinePasses(pm);
        // Optimizations at ZLow after normalizing MemRefs.
        pm.addPass(zlow::createZLowRewritePass());
        pm.addPass(mlir::createCanonicalizerPass());
        // Constant folding for std.alloc.
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createFoldStdAllocPass());
      }
      // Insert an instrumentation after lowering zhigh to zlow to get
      // profiling/signatures for zlow ops
      if (hasSignatureInstrumentation(onnx_mlir::InstrumentStages::ZLow))
        // Omit printing signatures that late.
        assert(false && "Printing signature information at ZLow instrument "
                        "stage is currently unsupported");
      if (hasInstrumentation(onnx_mlir::InstrumentStages::ZLow))
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentPass(
            instrumentOps, instrumentControlBits));
    }
  }

  if (emissionTarget >= EmitLLVMIR)
    // Lower the remaining Krnl and all ZLow ops to LLVM dialect.
    addKrnlToLLVMPasses(pm, outputNameNoExt, /*enableCSE=*/true);
}

} // namespace onnx_mlir
