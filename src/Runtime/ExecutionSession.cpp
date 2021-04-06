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

namespace onnx_mlir {

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string entryPointName) {
  // Adapted from https://www.tldp.org/HOWTO/html_single/C++-dlopen/.
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY);
  if (!_sharedLibraryHandle) {
    std::stringstream errStr;
    errStr << "Cannot open library: " << dlerror() << std::endl;
    throw std::runtime_error(errStr.str());
  }

  // Reset errors.
  dlerror();
  _entryPointFunc =
      (entryPointFuncType)dlsym(_sharedLibraryHandle, entryPointName.c_str());
  auto *dlsymError = dlerror();
  if (dlsymError) {
    std::stringstream errStr;
    errStr << "Cannot load symbol '" << entryPointName << "': " << dlsymError
           << std::endl;
    dlclose(_sharedLibraryHandle);
    throw std::runtime_error(errStr.str());
  }

  _loadConstantFunc =
      (loadConstantFuncType)dlsym(_sharedLibraryHandle, "load_constants");
  dlsymError = dlerror();
  if (dlsymError) {
    std::stringstream errStr;
    errStr << "Cannot load symbol load_constants: " << dlsymError << std::endl;
    dlclose(_sharedLibraryHandle);
    throw std::runtime_error(errStr.str());
  }

  _destroyConstantFunc =
      (destroyConstantFuncType)dlsym(_sharedLibraryHandle, "destroy_constants");
  dlsymError = dlerror();
  if (dlsymError) {
    std::stringstream errStr;
    errStr << "Cannot load symbol destroy_constants: " << dlsymError
           << std::endl;
    dlclose(_sharedLibraryHandle);
    throw std::runtime_error(errStr.str());
  }

  // Load the constants of the model.
  _loadConstantFunc();
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

  for (size_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    outs.emplace_back(std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
        omTensorListGetOmtByIndex(wrappedOutput, i), omTensorDestroy));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() {
  // Clean up the loaded constants.
  _destroyConstantFunc();
  dlclose(_sharedLibraryHandle);
}
} // namespace onnx_mlir
