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
}

std::vector<std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>>
ExecutionSession::run(
    std::vector<std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>> ins) {
  auto *wrappedInput = rmrListCreate();
  for (size_t i = 0; i < ins.size(); i++)
    rmrListSetRmrByIndex(wrappedInput, ins.at(i).get(), i);

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>> outs;

  for (size_t i = 0; i < rmrListGetNumRmrs(wrappedOutput); i++) {
    outs.emplace_back(std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>(
        rmrListGetRmrByIndex(wrappedOutput, i), rmrDestroy));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() { dlclose(_sharedLibraryHandle); }
} // namespace onnx_mlir
