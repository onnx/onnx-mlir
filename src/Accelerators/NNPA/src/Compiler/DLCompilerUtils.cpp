#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <mlir/Dialect/MemRef/Transforms/Passes.h>

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Compiler/DLCompilerUtils.hpp"
#include "src/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/DLCPasses.hpp"
#include "src/Support/OMDLCOptions.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;
extern llvm::cl::OptionCategory OnnxMlirOptions;

void addONNXToZHighPasses(
    mlir::PassManager &pm, ArrayRef<std::string> execNodesOnCpu) {
  pm.addPass(mlir::createRewriteONNXForZHighPass(execNodesOnCpu));
  pm.addPass(mlir::createShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
  // Add instrumentation for Onnx Ops in the same way as onnx-mlir.
  if (instrumentZHighOps == "" || instrumentZHighOps == "NONE")
    pm.addNestedPass<FuncOp>(mlir::createInstrumentONNXPass());
  pm.addPass(mlir::createONNXToZHighPass(execNodesOnCpu));
  pm.addPass(mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all zhigh ops were
  // generated.
  pm.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Layout propagation at ZHighIR.
  pm.addNestedPass<FuncOp>(mlir::createZHighLayoutPropagationPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Constant propagation at ZHighIR: constant stickify.
  // Only support BE machines.
  bool isBE = llvm::support::endian::system_endianness() ==
              llvm::support::endianness::big;
  if (isBE)
    pm.addNestedPass<FuncOp>(mlir::createZHighConstPropagationPass());
}

void addZHighToZLowPasses(mlir::PassManager &pm, int optLevel) {
  // Add instrumentation for ZHigh Ops
  pm.addNestedPass<FuncOp>(mlir::createInstrumentZHighPass());
  pm.addPass(mlir::createZHighToZLowPass(optLevel));
  pm.addNestedPass<FuncOp>(createLowerKrnlShapePass());
  pm.addNestedPass<FuncOp>(createDisconnectKrnlDimFromAllocPass());
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
    pm.addNestedPass<FuncOp>(mlir::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlOptimizeMemoryPoolsPass());
  }

  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createZLowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addPassesDLC(mlir::OwningModuleRef &module, mlir::PassManager &pm,
    EmissionTargetType &emissionTarget, DLCEmissionTargetType dlcEmissionTarget,
    ArrayRef<std::string> execNodesOnCpu) {
  // TODO: Develop and use determineInputIRLevel for DLC
  // InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (emissionTarget >= onnx_mlir::EmitONNXIR) {
    addONNXToMLIRPasses(pm);
  }

  if (emissionTarget >= onnx_mlir::EmitMLIR) {
    // Lower zAIU-compatible ONNX ops to ZHigh dialect where possible.
    addONNXToZHighPasses(pm, execNodesOnCpu);

    if (dlcEmissionTarget >= EmitZHighIR)
      emissionTarget = onnx_mlir::EmitMLIR;
    else {
      pm.addPass(mlir::createCanonicalizerPass());
      // Add instrumentation for remaining Onnx Ops
      if (instrumentZHighOps != "" && instrumentZHighOps != "NONE")
        pm.addNestedPass<FuncOp>(mlir::createInstrumentONNXPass());
      // Lower all ONNX and ZHigh ops.
      addZHighToZLowPasses(pm, getOptLevel());
      // Constant folding for std.alloc.
      pm.addNestedPass<FuncOp>(mlir::createFoldStdAllocPass());

      if (dlcEmissionTarget >= EmitZLowIR)
        emissionTarget = onnx_mlir::EmitMLIR;
      else {
        // Partially lower Krnl ops to Affine dialect.
        addKrnlToAffinePasses(pm);
      }
    }
  }

  if (emissionTarget >= onnx_mlir::EmitLLVMIR)
    // Lower the remaining Krnl and all ZLow ops to LLVM dialect.
    addAllToLLVMPasses(pm);
}

int compileModuleDLC(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, onnx_mlir::EmissionTargetType emissionTarget,
    DLCEmissionTargetType dlcEmissionTarget,
    ArrayRef<std::string> execNodesOnCpu) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::ZHighDialect>();
  context.getOrLoadDialect<mlir::ZLowDialect>();

  setupModule(module, context, outputBaseName);

  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
  addPassesDLC(module, pm, emissionTarget, dlcEmissionTarget, execNodesOnCpu);
  mlir::applyPassManagerCLOptions(pm);
  if (mlir::failed(pm.run(*module)))
    return 4;

  emitOutput(module, context, outputBaseName, pm, emissionTarget);

  return 0;
}
