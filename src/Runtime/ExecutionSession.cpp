/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ExecutionSession.cpp - ExecutionSession Implementation -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "ExecutionSession.hpp"
#include "llvm/Support/ManagedStatic.h"

namespace onnx_mlir {
const std::string ExecutionSession::_queryEntryPointsName =
    "omQueryEntryPoints";
const std::string ExecutionSession::_inputSignatureName = "omInputSignature";
const std::string ExecutionSession::_outputSignatureName = "omOutputSignature";

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, bool defaultEntryPoint) {
  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getPermanentLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid()) {
    std::stringstream errStr;
    errStr << "Cannot open library: '" << sharedLibPath << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }

  if (defaultEntryPoint)
    setEntryPoint("run_main_graph");

  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_queryEntryPointsName.c_str()));
  if (!_queryEntryPointsFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << _queryEntryPointsName << "'"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }

  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_inputSignatureName.c_str()));
  if (!_inputSignatureFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << _inputSignatureName << "'"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }

  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_outputSignatureName.c_str()));
  if (!_outputSignatureFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << _outputSignatureName << "'"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }
}

const std::string *ExecutionSession::queryEntryPoints(
    int64_t *numOfEntryPoints) const {
  return (const std::string *)_queryEntryPointsFunc(numOfEntryPoints);
}

void ExecutionSession::setEntryPoint(const std::string &entryPointName) {
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << entryPointName << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }
  _entryPointName = entryPointName;
}

std::vector<OMTensorUniquePtr> ExecutionSession::run(
    std::vector<OMTensorUniquePtr> ins) {
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Must set the entry point before calling run function"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }

  std::vector<OMTensor *> omts;
  for (const auto &inOmt : ins)
    omts.emplace_back(inOmt.get());
  auto *wrappedInput = omTensorListCreate(&omts[0], (int64_t)omts.size());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<OMTensorUniquePtr> outs;

  for (int64_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    outs.emplace_back(OMTensorUniquePtr(
        omTensorListGetOmtByIndex(wrappedOutput, i), omTensorDestroy));
  }
  return outs;
}

// Run using public interface. Explicit calls are needed to free tensor & tensor
// lists.
OMTensorList *ExecutionSession::run(OMTensorList *input) {
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Must set the entry point before calling run function"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }
  return _entryPointFunc(input);
}

const std::string ExecutionSession::inputSignature() const {
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Must set the entry point before calling signature function"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }
  return _inputSignatureFunc(_entryPointName.c_str());
}

const std::string ExecutionSession::outputSignature() const {
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Must set the entry point before calling signature function"
           << std::endl;
    throw std::runtime_error(errStr.str());
  }
  return _outputSignatureFunc(_entryPointName.c_str());
}

ExecutionSession::~ExecutionSession() {
  // Call llvm_shutdown which will take care of cleaning up our shared library
  // handles
  llvm::llvm_shutdown();
}
} // namespace onnx_mlir
