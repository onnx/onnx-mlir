//===------- ExecusionSession.cpp - ExecutionSession Implementation -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of ExecusionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "ExecusionSession.hpp"

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
}

std::vector<std::unique_ptr<DynMemRef>> ExecutionSession::run(
    std::vector<std::unique_ptr<DynMemRef>> ins) {
  auto *wrappedInput = createOrderedDynMemRefDict();
  for (size_t i = 0; i < ins.size(); i++)
    setDynMemRef(wrappedInput, i, ins.at(i).get());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<std::unique_ptr<DynMemRef>> outs;
  auto outputSize = getSize(wrappedOutput);

  for (size_t i = 0; i < getSize(wrappedOutput); i++) {
    outs.emplace_back(
        std::unique_ptr<DynMemRef>(getDynMemRef(wrappedOutput, i)));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() { dlclose(_sharedLibraryHandle); }
} // namespace onnx_mlir