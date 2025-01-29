/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ExecutionSession.cpp - ExecutionSession Implementation -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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

#ifndef ENABLE_PYRUNTIME_LIGHT
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#else
#include <dlfcn.h>
#endif

#include "ExecutionSession.hpp"
#include "OMTensorListHelper.hpp"

namespace onnx_mlir {
// =============================================================================
// Constructor, destructor, and init.

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint) {
  Init(sharedLibPath, tag, defaultEntryPoint);
}

void ExecutionSession::Init(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint) {
  if (isInitialized)
    throw std::runtime_error(reportInitError());

  // If there is no tag, use the model filename without extension as a tag.
  if (tag == "") {
    // ToFix: equivalent implementation of llvm utilities.
    // The would not be an urgent issue, because tag is usually "NONE"
#ifndef ENABLE_PYRUNTIME_LIGHT
    std::string fname = llvm::sys::path::filename(sharedLibPath).str();
    llvm::SmallString<256> fnameWithoutExt(fname);
    llvm::sys::path::replace_extension(fnameWithoutExt, "");
    tag = fnameWithoutExt.str().lower();
#endif
  }

  // tag = "NONE" to use functions without tag.
  std::string lowDashTag;
  // ToFix: equivalent implementation of llv::StringRef
#ifndef ENABLE_PYRUNTIME_LIGHT
  // Assume tag is always NONE
  if (!llvm::StringRef(tag).equals_insensitive("NONE"))
    lowDashTag = "_" + tag;
#endif

#if defined(_WIN32)
  // Use functions without tags on Windows since we cannot define at compile
  // time the tagged functions in the header files in
  // `include/onnx-mlir/Runtime` to make the tagged functions visible.
  lowDashTag = "";
#endif

  // Init symbols used by execution session.
#ifndef ENABLE_PYRUNTIME_LIGHT
  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid())
    throw std::runtime_error(reportLibraryOpeningError(sharedLibPath));
#else
  // Copy code from llvm/lib/Support/DynamicLibrary.cpp, especially the flags
  // ToFix: copy the lock related code too.
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!_sharedLibraryHandle)
    throw std::runtime_error(reportLibraryOpeningError(sharedLibPath));
#endif

  std::string queryEntryPointsNameWithTag = _queryEntryPointsName + lowDashTag;
#ifndef ENABLE_PYRUNTIME_LIGHT
  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          queryEntryPointsNameWithTag.c_str()));
#else
  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      dlsym(_sharedLibraryHandle, queryEntryPointsNameWithTag.c_str()));
#endif

  if (!_queryEntryPointsFunc)
    throw std::runtime_error(
        reportSymbolLoadingError(queryEntryPointsNameWithTag));

  std::string inputSignatureNameWithTag = _inputSignatureName + lowDashTag;
#ifndef ENABLE_PYRUNTIME_LIGHT
  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          inputSignatureNameWithTag.c_str()));
#else
  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      dlsym(_sharedLibraryHandle, inputSignatureNameWithTag.c_str()));
#endif
  if (!_inputSignatureFunc)
    throw std::runtime_error(
        reportSymbolLoadingError(inputSignatureNameWithTag));

  std::string outputSignatureNameWithTag = _outputSignatureName + lowDashTag;
#ifndef ENABLE_PYRUNTIME_LIGHT
  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          outputSignatureNameWithTag.c_str()));
#else
  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      dlsym(_sharedLibraryHandle, outputSignatureNameWithTag.c_str()));
#endif
  if (!_outputSignatureFunc)
    throw std::runtime_error(
        reportSymbolLoadingError(outputSignatureNameWithTag));

  // Set OM_CONSTANT_PATH for loading constants from file if required.
  std::size_t found = sharedLibPath.find_last_of("/\\");
  if (found != std::string::npos) {
    std::string basePath = sharedLibPath.substr(0, found);
#if defined(_WIN32)
    _putenv_s("OM_CONSTANT_PATH", basePath.c_str());
#else
    setenv("OM_CONSTANT_PATH", basePath.c_str(), /*overwrite=*/0);
#endif
  }

  // Successful completion of initialization.
  isInitialized = true;

  // Set default entry point if requested.
  if (defaultEntryPoint)
    setEntryPoint("run_main_graph" + lowDashTag);

  errno = 0; // No errors.
}

ExecutionSession::~ExecutionSession() {
#ifndef ENABLE_PYRUNTIME_LIGHT
  if (_sharedLibraryHandle.isValid())
    llvm::sys::DynamicLibrary::closeLibrary(_sharedLibraryHandle);
#else
  if (!_sharedLibraryHandle)
    dlclose(_sharedLibraryHandle);
#endif
}

// =============================================================================
// Setter and getter.

const std::string *ExecutionSession::queryEntryPoints(
    int64_t *numOfEntryPoints) const {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  return reinterpret_cast<const std::string *>(
      _queryEntryPointsFunc(numOfEntryPoints));
}

void ExecutionSession::setEntryPoint(const std::string &entryPointName) {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
#ifndef ENABLE_PYRUNTIME_LIGHT
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
#else
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      dlsym(_sharedLibraryHandle, entryPointName.c_str()));
#endif
  if (!_entryPointFunc)
    throw std::runtime_error(reportSymbolLoadingError(entryPointName));
  _entryPointName = entryPointName;
  errno = 0; // No errors.
}

const std::string ExecutionSession::inputSignature() const {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("signature"));
  errno = 0; // No errors.
  return _inputSignatureFunc(_entryPointName.c_str());
}

const std::string ExecutionSession::outputSignature() const {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("signature"));
  errno = 0; // No errors.
  return _outputSignatureFunc(_entryPointName.c_str());
}

// =============================================================================
// Run.

std::vector<OMTensorUniquePtr> ExecutionSession::run(
    std::vector<OMTensorUniquePtr> ins) {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("run"));

  std::vector<OMTensor *> omts;
  for (const auto &inOmt : ins)
    omts.emplace_back(inOmt.get());
  auto *wrappedInput =
      omTensorListCreate(omts.data(), static_cast<int64_t>(omts.size()));

  // Run inference.
  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  // We created a wrapper for the input list, but the input list does not really
  // own the tensor in the list, as they are coming as OMTensorUniquePtr. So we
  // need to simply deallocate the list structure without touching the
  // OMTensors.
  omTensorListDestroyShallow(wrappedInput);
  if (!wrappedOutput)
    throw std::runtime_error(reportErrnoError());

  std::vector<OMTensorUniquePtr> outs;
  for (int64_t i = 0; i < omTensorListGetSize(wrappedOutput); i++) {
    outs.emplace_back(OMTensorUniquePtr(
        omTensorListGetOmtByIndex(wrappedOutput, i), omTensorDestroy));
  }

  // We created a wrapper for the output list, but the output list does not
  // really own the tensor in the list, as they are returned in a vector of
  // OMTensorUniquePtr. So we need to simply deallocate the list structure
  // without touching the OMTensors.
  omTensorListDestroyShallow(wrappedOutput);
  errno = 0; // No errors.
  return outs;
}

// Run using public interface. Explicit calls are needed to free tensor & tensor
// lists.
OMTensorList *ExecutionSession::run(OMTensorList *input) {
  if (!isInitialized)
    throw std::runtime_error(reportInitError());
  if (!_entryPointFunc)
    throw std::runtime_error(reportUndefinedEntryPointIn("run"));

  // Run inference.
  OMTensorList *output = _entryPointFunc(input);
  if (!output)
    throw std::runtime_error(reportErrnoError());
  errno = 0; // No errors.
  return output;
}

// =============================================================================
// Error reporting

std::string ExecutionSession::reportInitError() const {
  errno = EFAULT; // Bad Address.
  std::stringstream errStr;
  errStr << "Execution session must be initialized once." << std::endl;
  return errStr.str();
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

std::string ExecutionSession::reportCompilerError(
    const std::string &errorMessage) const {
  errno = EFAULT; // Bad Address.
  std::stringstream errStr;
  errStr << "Compiler failed with error message '" << errorMessage << "'."
         << std::endl;
  return errStr.str();
}

} // namespace onnx_mlir
