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
const std::string ExecutionSession::_inputSignatureName = "omInputSignature";
const std::string ExecutionSession::_outputSignatureName = "omOutputSignature";

ExecutionSession::ExecutionSession(std::string sharedLibPath)
    : ExecutionSession::ExecutionSession(sharedLibPath, "") {}

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string entryPointName) {

  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getPermanentLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid()) {
    std::stringstream errStr;
    errStr << "Cannot open library: '" << sharedLibPath << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }

  // When entry point name is not given, use the default "run_main_graph".
  // TODO(tung): support multiple entry point functions.
  if (entryPointName.empty())
    entryPointName = "run_main_graph";

  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << entryPointName << "'" << std::endl;
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

std::vector<OMTensorUniquePtr> ExecutionSession::run(
    std::vector<OMTensorUniquePtr> ins) {

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
  return _entryPointFunc(input);
}

std::string ExecutionSession::inputSignature() { return _inputSignatureFunc(); }

std::string ExecutionSession::outputSignature() {
  return _outputSignatureFunc();
}

ExecutionSession::~ExecutionSession() {
  // Call llvm_shutdown which will take care of cleaning up our shared library
  // handles
  llvm::llvm_shutdown();
}
} // namespace onnx_mlir
