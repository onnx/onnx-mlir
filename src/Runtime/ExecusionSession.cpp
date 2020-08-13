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
#if _WIN32
  _sharedLibraryHandle = LoadLibrary(sharedLibPath.c_str());
#else  
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY);
#endif
  if (!_sharedLibraryHandle) {
    std::stringstream errStr;
#if _WIN32
    errStr << GetLastError() << std::endl;
#else
    errStr << "Cannot open library: " << dlerror() << std::endl;
#endif
    throw std::runtime_error(errStr.str());
  }

  // Reset errors.
#if _WIN32
  GetLastError();
  _entryPointFunc =
      (entryPointFuncType)GetProcAddress((HMODULE)_sharedLibraryHandle, entryPointName.c_str());
  auto dlsymError = GetLastError();
  if (dlsymError) {
    std::stringstream errStr;
    errStr << "Cannot load symbol '" << entryPointName << "': " << dlsymError
           << std::endl;
    FreeLibrary((HMODULE)_sharedLibraryHandle);
    throw std::runtime_error(errStr.str());
  }
#else
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
#endif
}

std::vector<std::unique_ptr<RtMemRef>> ExecutionSession::run(
    std::vector<std::unique_ptr<RtMemRef>> ins) {
  auto *wrappedInput = createOrderedRtMemRefDict();
  for (size_t i = 0; i < ins.size(); i++)
    setRtMemRef(wrappedInput, i, ins.at(i).get());

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<std::unique_ptr<RtMemRef>> outs;
  auto outputSize = getSize(wrappedOutput);

  for (size_t i = 0; i < getSize(wrappedOutput); i++) {
    outs.emplace_back(std::unique_ptr<RtMemRef>(getRtMemRef(wrappedOutput, i)));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() {
#if _WIN32
  FreeLibrary((HMODULE)_sharedLibraryHandle);
#else
  dlclose(_sharedLibraryHandle);
#endif
  }
} // namespace onnx_mlir