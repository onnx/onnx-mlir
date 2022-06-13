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

#include <errno.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"

#include "ExecutionSession.hpp"

namespace onnx_mlir {
const std::string ExecutionSession::_queryEntryPointsName =
    "omQueryEntryPoints";
const std::string ExecutionSession::_inputSignatureName = "omInputSignature";
const std::string ExecutionSession::_outputSignatureName = "omOutputSignature";

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, bool defaultEntryPoint) {

  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getPermanentLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid())
    throw std::runtime_error(reportLibraryOpeningError(sharedLibPath));

  if (defaultEntryPoint)
    setEntryPoint("run_main_graph");

  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_queryEntryPointsName.c_str()));
  if (!_queryEntryPointsFunc)
    throw std::runtime_error(reportSymbolLoadingError(_queryEntryPointsName));

  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_inputSignatureName.c_str()));
  if (!_inputSignatureFunc)
    throw std::runtime_error(reportSymbolLoadingError(_inputSignatureName));

  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(_outputSignatureName.c_str()));
  if (!_outputSignatureFunc)
    throw std::runtime_error(reportSymbolLoadingError(_outputSignatureName));
  errno = 0; // No errors.
}

const std::string *ExecutionSession::queryEntryPoints(
    int64_t *numOfEntryPoints) const {
  return (const std::string *)_queryEntryPointsFunc(numOfEntryPoints);
}

void ExecutionSession::setEntryPoint(const std::string &entryPointName) {
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
  if (!_entryPointFunc)
    throw std::runtime_error(reportSymbolLoadingError(entryPointName));
  _entryPointName = entryPointName;
  errno = 0; // No errors.
}

std::vector<OMTensorUniquePtr> ExecutionSession::run(
    std::vector<OMTensorUniquePtr> ins) {
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("run"));

  std::vector<OMTensor *> omts;
  for (const auto &inOmt : ins)
    omts.emplace_back(inOmt.get());
  auto *wrappedInput = omTensorListCreate(&omts[0], (int64_t)omts.size());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  if (!wrappedOutput)
    throw std::runtime_error(reportErrnoError());
  std::vector<OMTensorUniquePtr> outs;

  for (int64_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    outs.emplace_back(OMTensorUniquePtr(
        omTensorListGetOmtByIndex(wrappedOutput, i), omTensorDestroy));
  }
  errno = 0; // No errors.
  return outs;
}

// Run using public interface. Explicit calls are needed to free tensor & tensor
// lists.
OMTensorList *ExecutionSession::run(OMTensorList *input) {
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Must set the entry point before calling run function"
           << std::endl;
    errno = EINVAL;
    throw std::runtime_error(errStr.str());
  }
  OMTensorList *output = _entryPointFunc(input);
  if (!output) {
    std::stringstream errStr;
    std::string errMessageStr = std::string(strerror(errno));
    errStr << "Runtime error during inference returning with ERRNO code '"
           << errMessageStr << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }
  errno = 0; // No errors.
  return output;
}

const std::string ExecutionSession::inputSignature() const {
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("signature"));
  errno = 0; // No errors.
  return _inputSignatureFunc(_entryPointName.c_str());
}

const std::string ExecutionSession::outputSignature() const {
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("signature"));
  errno = 0; // No errors.
  return _outputSignatureFunc(_entryPointName.c_str());
}

ExecutionSession::~ExecutionSession() {
  // Call llvm_shutdown which will take care of cleaning up our shared library
  // handles
  llvm::llvm_shutdown();
}

std::string ExecutionSession::reportLibraryOpeningError(
    const std::string &libraryName) const {
  errno = EFAULT; // Bad Address.
  std::stringstream errStr;
  errStr << "Cannot open library: '" << libraryName << "'." << std::endl;
  return errStr.str();
}

std::string ExecutionSession::reportSymbolLoadingError(
    const std::string &symbolName) const {
  errno = EFAULT; // Bad Address.
  std::stringstream errStr;
  errStr << "Cannot load symbol: '" << symbolName << "'." << std::endl;
  return errStr.str();
}

std::string ExecutionSession::reportUndefinedEntryPointIn(
    const std::string &functionName) const {
  errno = EINVAL; // Invalid argument.
  std::stringstream errStr;
  errStr << "Must set an entry point (e.g. run_main_graph) before calling "
         << functionName << " function." << std::endl;
  return errStr.str();
}

std::string ExecutionSession::reportErrnoError() const {
  std::string errMessageStr = std::string(strerror(errno));
  std::stringstream errStr;
  errStr << "Runtime error during inference returning with ERRNO code '"
         << errMessageStr << "'." << std::endl;
  return errStr.str();
}

} // namespace onnx_mlir
