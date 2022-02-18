/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- PrepareAccelerator.cpp
//-------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Add accelerator support for NNPA
//
//===----------------------------------------------------------------------===//

//#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Support/OMOptions.hpp"
#include <iostream>
// modified from DLC main
#include "src/Compiler/DLCompilerUtils.hpp"
#include "src/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/DLCPasses.hpp"
extern llvm::cl::OptionCategory OMDLCPassOptions;

namespace mlir {
class NNPAAccelerator : public Accelerator {
private:
  static bool initialized;

public:
//NNPAAccelerator::NNPAAccelerator() {
NNPAAccelerator() {
  std::cout << "initializing NNPA" << std::endl;
  if (!initialized) {
    initialized = true;
    // getAcceleratorList()->push_back(this);
  } // else
    // getAcceleratorList()->push_back(this);
};


//bool NNPAAccelerator::isActive() {
bool isActive() override {
  if (acceleratorTarget.compare("NNPA") == 0) {
    std::cout << "Targeting NNPA accelerator" << std::endl;
    return true;
  } else
    return false;


}

//void NNPAAccelerator::prepareAccelerator(mlir::OwningModuleRef &module, mlir::MLIRContext &context, mlir::PassManager &pm,
void prepareAccelerator(mlir::OwningModuleRef &module, mlir::MLIRContext &context, mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType emissionTarget)  override {
  std::cout << "preparing accelerator " << acceleratorTarget << std::endl;
  llvm::cl::opt<DLCEmissionTargetType> dlcEmissionTarget(
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


      // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::ZHighDialect>();
  context.getOrLoadDialect<mlir::ZLowDialect>();
  addPassesDLC(module, pm, emissionTarget, dlcEmissionTarget, execNodesOnCpu);

    }
};

bool NNPAAccelerator::initialized = false;
NNPAAccelerator nnpaAccelerator;

} // namespace mlir
