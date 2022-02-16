#pragma once

#include "mlir/Pass/PassManager.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Support/OMOptions.hpp"

enum DLCEmissionTargetType {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
};

void addMemoryPooling(mlir::PassManager &pm);

void addONNXToZHighPasses(mlir::PassManager &pm);

void addZHighToZLowPasses(mlir::PassManager &pm);

void addAllToLLVMPasses(mlir::PassManager &pm);

int compileModuleDLC(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, onnx_mlir::EmissionTargetType emissionTarget,
    DLCEmissionTargetType dlcEmissionTarget = EmitZNONE,
    mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>());
