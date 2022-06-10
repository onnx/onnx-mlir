/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========-- ModelLib.cpp - Helper function for building models -==========//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for all the models that can be built.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace test {

ModelLibBuilder::ModelLibBuilder(const std::string &name)
    : sharedLibBaseName(name), ctx(), loc(UnknownLoc::get(&ctx)), builder(&ctx),
      module(ModuleOp::create(loc)), inputs(nullptr), outputs(nullptr),
      exec(nullptr) {
  registerDialects(ctx);
}

ModelLibBuilder::~ModelLibBuilder() {
  omTensorListDestroy(inputs);
  omTensorListDestroy(outputs);
  if (exec)
    delete exec;
}

bool ModelLibBuilder::compileAndLoad() {
  OwningOpRef<ModuleOp> moduleRef(module);
  if (compileModule(moduleRef, ctx, sharedLibBaseName, onnx_mlir::EmitLib) !=
      CompilerSuccess)
    return false;
  std::string libFilename =
      getTargetFilename(sharedLibBaseName, onnx_mlir::EmitLib);
  exec = new ExecutionSession(libFilename);
  return exec != nullptr;
}

bool ModelLibBuilder::compileAndLoad(
    const onnx_mlir::CompilerOptionList &list) {
  if (setCompilerOptions(list) != CompilerSuccess)
    return false;
  return compileAndLoad();
}

bool ModelLibBuilder::checkInstructionFromEnv(
    const std::string envCheckInstruction) {
  std::string instructionName = getenv(envCheckInstruction.c_str())
                                    ? getenv(envCheckInstruction.c_str())
                                    : "";
  return checkInstruction(instructionName);
}

bool ModelLibBuilder::checkInstruction(const std::string instructionName) {
  if (instructionName.empty())
    return true;
  llvm::sys::DynamicLibrary sharedLibraryHandle =
      exec->getSharedLibraryHandle();
  void *addr = sharedLibraryHandle.getAddressOfSymbol(instructionName.c_str());
  if (!addr) {
    printf("%s not found.\n", instructionName.c_str());
    return false;
  }
  return true;
}

bool ModelLibBuilder::run() {
  assert(inputs && exec && "expected successful compile and load");
  if (outputs) {
    omTensorListDestroy(outputs);
    outputs = nullptr; // Reset in case run has an exception.
  }
  try {
    outputs = exec->run(inputs);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while running: " << error.what() << std::endl;
    return false;
  }
  assert(outputs && "when no exception are issued, output should exist");
  return true;
}

void ModelLibBuilder::setRandomNumberGeneratorSeed(const std::string &envVar) {
  bool hasSeedValue = false;
  unsigned int seed = 0;
  if (const char *envVal = std::getenv(envVar.c_str())) {
    std::string seedStr(envVal);
    seed = (unsigned int)std::stoul(seedStr, nullptr);
    hasSeedValue = true;
    std::cout
        << "Model will use the random number generator seed provided by \""
        << envVar << "=" << seed << "\"\n";
  }
  seed = omDefineSeed(seed, hasSeedValue);
  if (!hasSeedValue) {
    // We used a random seed; print that seed to that we may reproduce the
    // experiment.
    std::cout << "Model can reuse the current seed by exporting \"" << envVar
              << "=" << seed << "\"\n";
  }
}

func::FuncOp ModelLibBuilder::createEmptyTestFunction(
    const llvm::SmallVectorImpl<Type> &inputsType,
    const llvm::SmallVectorImpl<Type> &outputsType) {
  assert(!inputsType.empty() && "Expecting inputsTypes to be non-empty");
  assert(!outputsType.empty() && "Expecting outputsTypes to be non-empty");

  FunctionType funcType = builder.getFunctionType(inputsType, outputsType);

  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<func::FuncOp>(loc, "main_graph", funcType, attrs);

  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  return funcOp;
}

void ModelLibBuilder::createEntryPoint(func::FuncOp &funcOp) {
  auto entryPoint = ONNXEntryPointOp::create(loc, funcOp);
  module.push_back(entryPoint);
}

ONNXConstantOp ModelLibBuilder::buildONNXConstantOp(
    const OMTensor *omt, const RankedTensorType resultType) {
  int64_t numElems = omTensorGetNumElems(omt);
  auto bufferPtr = omTensorGetDataPtr(omt);
  float *arrayPtr = reinterpret_cast<float *>(bufferPtr);
  auto array = std::vector<float>(arrayPtr, arrayPtr + numElems);
  auto denseAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(array));
  return builder.create<ONNXConstantOp>(loc, resultType, Attribute(), denseAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

bool ModelLibBuilder::areCloseFloat(
    const OMTensor *res, const OMTensor *ref) const {
  if (!res || !ref)
    return false;
  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;
  return omTensorAreTwoOmtsClose<float>(res, ref, rtol, atol);
}

} // namespace test
} // namespace onnx_mlir
