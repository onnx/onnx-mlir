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

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string entryPointName) {

  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getPermanentLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid()) {
    std::stringstream errStr;
    errStr << "Cannot open library: '" << sharedLibPath << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }

  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
  if (!_entryPointFunc) {
    std::stringstream errStr;
    errStr << "Cannot load symbol: '" << entryPointName << "'" << std::endl;
    throw std::runtime_error(errStr.str());
  }
}

std::vector<std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>>
ExecutionSession::run(
    std::vector<std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>> ins) {

  std::vector<OMTensor *> omts;
  for (const auto &inOmt : ins)
    omts.emplace_back(inOmt.get());
  auto *wrappedInput = omTensorListCreate(&omts[0], omts.size());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>> outs;

  for (size_t i = 0; i < (size_t)omTensorListGetSize(wrappedOutput); i++) {
    outs.emplace_back(std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
        omTensorListGetOmtByIndex(wrappedOutput, i), omTensorDestroy));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() {
  // Call llvm_shutdown which will take care of cleaning up our shared library
  // handles
  llvm::llvm_shutdown();
}
} // namespace onnx_mlir
