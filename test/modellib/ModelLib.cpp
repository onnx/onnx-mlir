/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========-- ModelLib.cpp - Helper function for building models -==========//
//
// Copyright 2022-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for all the models that can be built.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CommandUtils.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"
#include <sstream>

using namespace mlir;

namespace onnx_mlir {
namespace test {

ModelLibBuilder::ModelLibBuilder(const std::string &name)
    : sharedLibBaseName(name), ctx(), loc(UnknownLoc::get(&ctx)), builder(&ctx),
      module(ModuleOp::create(loc)), inputs(nullptr), outputs(nullptr),
      exec(nullptr) {
  loadDialects(ctx);
}

ModelLibBuilder::~ModelLibBuilder() {
  omTensorListDestroy(inputs);
  omTensorListDestroy(outputs);
  if (exec)
    delete exec;
}

bool ModelLibBuilder::compileAndLoad(bool debug) {

  OwningOpRef<ModuleOp> moduleRef(module);
  if (debug) {
    fprintf(stderr, "Debugging mode: compile to save a .mlir model file\n");
    // Can use EmitONNXBasic (before decode) EmitONNXIR (after decode).
    compileModule(moduleRef, ctx, sharedLibBaseName, onnx_mlir::EmitONNXBasic);
    fprintf(stderr, "Debugging mode: .mlir file saved here: \"%s\".\n",
        sharedLibBaseName.c_str());
  }
  if (compileModule(moduleRef, ctx, sharedLibBaseName, onnx_mlir::EmitLib) !=
      CompilerSuccess)
    return false;
  std::string libFilename =
      getTargetFilename(sharedLibBaseName, onnx_mlir::EmitLib);
  std::string modelTag = getCompilerOption(OptionKind::ModelTag);
  try {
    exec = new ExecutionSession(libFilename, modelTag);
  } catch (const onnx_mlir::ExecutionSessionException &error) {
    std::cerr << error.what() << std::endl;
    exec = nullptr;
  }
  return exec != nullptr;
}

bool ModelLibBuilder::compileAndLoad(
    const onnx_mlir::CompilerOptionList &list, bool debug) {
  if (setCompilerOptions(list) != CompilerSuccess)
    return false;
  return compileAndLoad(debug);
}

bool ModelLibBuilder::checkInstructionFromEnv(
    const std::string envCheckInstruction, const bool optional) {
  std::string instructionName = getenv(envCheckInstruction.c_str())
                                    ? getenv(envCheckInstruction.c_str())
                                    : "";
  return checkInstruction(instructionName, optional);
}

bool ModelLibBuilder::checkInstruction(
    const std::string instructionName, const bool optional) {
  if (instructionName.empty())
    return true;

  // Split by comma and check each instruction
  DynamicLibraryHandleType sharedLibraryHandle = exec->getSharedLibraryHandle();
  std::stringstream ss(instructionName);
  std::string instruction;
  // Iterate over each comma separated instructions, if any.
  while (std::getline(ss, instruction, ',')) {
    // Trim whitespace
    size_t start = instruction.find_first_not_of(" \t");
    size_t end = instruction.find_last_not_of(" \t");
    if (start != std::string::npos && end != std::string::npos) {
      instruction = instruction.substr(start, end - start + 1);
    }
    if (!instruction.empty()) {
#if defined(_WIN32)
      void *addr = sharedLibraryHandle.getAddressOfSymbol(instruction.c_str());
#else
      void *addr = dlsym(sharedLibraryHandle, instruction.c_str());
#endif
      if (addr) {
        std::cout << "Binary has" << (optional ? " optional" : "")
                  << " instruction \"" << instruction << "\"\n";
        return true;
      }
    }
  }
  if (optional) {
    std::cout << "Binary didn't have the optional \"" << instructionName
              << "\" instructions\n";
    return true;
  }
  return false;
}

bool ModelLibBuilder::run(bool debug) {
  assert(inputs && exec && "expected successful compile and load");
  if (outputs) {
    omTensorListDestroy(outputs);
    outputs = nullptr; // Reset in case run has an exception.
  }
  try {
    outputs = exec->runDebug(inputs, debug);
  } catch (const onnx_mlir::ExecutionSessionException &error) {
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

std::map<std::string, std::string> ModelLibBuilder::getTestConfigFromEnv(
    const std::string &envVar) {
  std::map<std::string, std::string> opts;
  if (const char *envConfigString = std::getenv(envVar.c_str())) {
    std::stringstream envString;
    envString << envConfigString;
    std::string optionString;
    while (getline(envString, optionString, ' ')) {
      size_t pos = optionString.find('=');
      if (pos == std::string::npos)
        continue;
      std::string optionNameString = optionString.substr(0, pos);
      std::string optionValString = optionString.substr(pos + 1);
      opts[optionNameString] = optionValString;
    }
  }
  return opts;
}

std::vector<float> ModelLibBuilder::getDataRangeFromEnv(
    const std::string &envVar) {
  std::vector<float> range;
  if (const char *envRangeString = std::getenv(envVar.c_str())) {
    std::string rangeString = std::string(envRangeString);
    size_t pos = rangeString.find(',');
    assert(pos != std::string::npos);
    std::string rangeLBString = rangeString.substr(0, pos);
    std::string rangeUBString = rangeString.substr(pos + 1);
    std::cout << "Input data range from env: \"" << rangeLBString << " to "
              << rangeUBString << "\"\n";
    range.emplace_back(std::stof(rangeLBString));
    range.emplace_back(std::stof(rangeUBString));
  }
  return range;
}

func::FuncOp ModelLibBuilder::createEmptyTestFunction(
    const llvm::SmallVectorImpl<Type> &inputsType,
    const llvm::SmallVectorImpl<Type> &outputsType) {
  assert(!inputsType.empty() && "Expecting inputsTypes to be non-empty");
  assert(!outputsType.empty() && "Expecting outputsTypes to be non-empty");

  FunctionType funcType = builder.getFunctionType(inputsType, outputsType);

  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      func::FuncOp::create(builder, loc, "main_graph", funcType, attrs);

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
  auto denseAttr = DenseElementsAttr::get(resultType, llvm::ArrayRef(array));
  return ONNXConstantOp::create(builder, loc, resultType, Attribute(),
      denseAttr, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(),
      StringAttr(), ArrayAttr());
}

bool ModelLibBuilder::areCloseFloat(const OMTensor *res, const OMTensor *ref,
    float defaultRtol, float defaultAtol) const {
  if (!res || !ref)
    return false;
  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : defaultRtol;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : defaultAtol;
  if (rtol != defaultRtol || atol != defaultAtol)
    printf("RTOL and ATOL from env, with %f and %f values\n", rtol, atol);
  return omTensorAreTwoOmtsClose<float>(res, ref, rtol, atol);
}

void ModelLibBuilder::printIndices(
    const std::string message, const std::vector<int64_t> &indices) const {
  if (!message.empty())
    printf("%s, ", message.c_str());
  int rank = indices.size();
  printf("rank %d, sizes (", rank);
  for (int i = 0; i < rank; ++i)
    printf("%d%s", (int)indices[i], i != rank - 1 ? ", " : "");
  printf(")\n");
}

// Recursive printing to deal with arbitrary tensor ranks.
void ModelLibBuilder::printTensor(
    const OMTensor *t, std::vector<int64_t> &indices, bool isLast) const {
  int64_t rank = omTensorGetRank(t);
  const int64_t *shape = omTensorGetShape(t);
  int64_t currSize = indices.size();
  // Utility to print tabs.
  auto printTab = [](int currSize) {
    for (int i = 0; i < currSize; ++i)
      printf("  ");
  };
  auto printNext = [](int currSize, bool isLast) {
    printf("]");
    if (!isLast)
      printf(",\n");
    else if (currSize != 0)
      printf("\n");
  };
  if (currSize < rank - 1) {
    indices.emplace_back(0);
    printTab(currSize);
    printf("[\n");
    int size = shape[currSize];
    for (int64_t i = 0; i < size; ++i) {
      indices[currSize] = i;
      printTensor(t, indices, /*is last */ i == size - 1);
    }
    indices.pop_back();
    printTab(currSize);
    printNext(currSize, isLast);
    return;
  }
  // We need to print the last dim, do it as a one liner.
  indices.emplace_back(0);
  printTab(currSize);
  printf("[");
  int size = shape[currSize];
  for (int i = 0; i < size; ++i) {
    indices[currSize] = i;
    printf(
        "%f%s", omTensorGetElem<float>(t, indices), i < size - 1 ? ", " : "");
  }
  indices.pop_back();
  printNext(currSize, isLast);
}

void ModelLibBuilder::printTensor(
    const std::string varName, const OMTensor *t, bool asNumpy) const {
  int64_t rank = omTensorGetRank(t);
  const int64_t *shape = omTensorGetShape(t);
  std::vector<int64_t> shapeVect(shape, shape + rank);
  // Print message as comment and add rank and shape.
  printf("# ");
  printIndices(varName, shapeVect);
  // Print the actual tensor values.
  if (asNumpy) {
    if (!varName.empty())
      printf("%s = ", varName.c_str());
    printf("np.array(");
  }
  std::vector<int64_t> indices;
  printTensor(t, indices, /*isLast*/ true);
  if (asNumpy)
    printf(")\n");
}

RNNModelLibBuilder::RNNModelLibBuilder(
    const std::string &sharedLibBaseName, int64_t layout)
    : ModelLibBuilder(sharedLibBaseName), layout(layout) {
  assert(0 <= layout && layout <= 1 && "layout must be 0 or 1");
}

RNNModelLibBuilder::~RNNModelLibBuilder() {}

} // namespace test
} // namespace onnx_mlir
