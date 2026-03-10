/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ExecutionSession.cpp - ExecutionSession Implementation -------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cctype>
#include <errno.h>
#include <filesystem>
#include <string.h>
#include <strings.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#endif

#include "ExecutionSession.hpp"
#include "OMTensorListHelper.hpp"

namespace onnx_mlir {

#if !defined(_WIN32)
// =============================================================================
// Static member initialization for signal handling (POSIX only)

std::jmp_buf ExecutionSession::signalJumpBuffer;
volatile sig_atomic_t ExecutionSession::signalCaught = 0;
volatile sig_atomic_t ExecutionSession::signalNumber = 0;

// =============================================================================
// Signal handler (POSIX only)

void ExecutionSession::signalHandler(int signum) {
  signalCaught = 1;
  signalNumber = signum;
  std::longjmp(signalJumpBuffer, signum);
}
#endif

// =============================================================================
// Constructor, destructor, and init.

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint) {
  loadModel(sharedLibPath, tag, defaultEntryPoint);
}

// Errors: If dlopen returns null, it means error. Technically, if dlsym returns
// null, it is not necessarily an error (in corner cases); but for us it would
// be because we expect all our symbols to be defined to non-null.

void ExecutionSession::loadModel(
    std::string sharedLibPath, std::string tag, bool defaultEntryPoint) {
  if (isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once at most.");

  // If there is no tag, use the model filename without extension as a tag.
  if (tag == "") {
#if defined(_WIN32)
    std::string fname = llvm::sys::path::filename(sharedLibPath).str();
    llvm::SmallString<256> fnameWithoutExt(fname);
    llvm::sys::path::replace_extension(fnameWithoutExt, "");
    tag = fnameWithoutExt.lower();
#else
    std::string fnameWithoutExt = std::filesystem::path(sharedLibPath)
                                      .filename()
                                      .replace_extension("")
                                      .string();
    std::transform(fnameWithoutExt.begin(), fnameWithoutExt.end(),
        fnameWithoutExt.begin(),
        [](unsigned char c) { return std::tolower(c); });
    tag = fnameWithoutExt;
#endif
  }

  // tag = "NONE" to use functions without tag.
  std::string lowDashTag;
#if defined(_WIN32)
  // Use functions without tags on Windows since we cannot define at compile
  // time the tagged functions in the header files in
  // `include/onnx-mlir/Runtime` to make the tagged functions visible.
  lowDashTag = "";
#else
  // Save the llvm supported implementation.
  // if (!llvm::StringRef(tag).equals_insensitive("NONE"))
  // lowDashTag = "_" + tag;
  if (strcasecmp(tag.c_str(), "NONE") != 0)
    lowDashTag = "_" + tag;
#endif

  // Init symbols used by execution session.
#if defined(_WIN32)
  _sharedLibraryHandle =
      llvm::sys::DynamicLibrary::getLibrary(sharedLibPath.c_str());
  if (!_sharedLibraryHandle.isValid())
    throw ExecutionSessionException(
        "Cannot open library: '" + sharedLibPath + "'.");
#else
  // Copy code from llvm/lib/Support/DynamicLibrary.cpp, especially the flags
  // ToFix: copy the lock related code too.
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!_sharedLibraryHandle)
    throw ExecutionSessionException(
        "Cannot open library: '" + sharedLibPath + "'.");
#endif

  std::string queryEntryPointsNameWithTag = _queryEntryPointsName + lowDashTag;
#if defined(_WIN32)
  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          queryEntryPointsNameWithTag.c_str()));
#else
  _queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      dlsym(_sharedLibraryHandle, queryEntryPointsNameWithTag.c_str()));
#endif

  if (!_queryEntryPointsFunc)
    throw ExecutionSessionException(
        "Cannot load symbol: '" + queryEntryPointsNameWithTag + "'.");

  std::string inputSignatureNameWithTag = _inputSignatureName + lowDashTag;
#if defined(_WIN32)
  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          inputSignatureNameWithTag.c_str()));
#else
  _inputSignatureFunc = reinterpret_cast<signatureFuncType>(
      dlsym(_sharedLibraryHandle, inputSignatureNameWithTag.c_str()));
#endif
  if (!_inputSignatureFunc)
    throw ExecutionSessionException(
        "Cannot load symbol: '" + inputSignatureNameWithTag + "'.");

  std::string outputSignatureNameWithTag = _outputSignatureName + lowDashTag;
#if defined(_WIN32)
  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          outputSignatureNameWithTag.c_str()));
#else
  _outputSignatureFunc = reinterpret_cast<signatureFuncType>(
      dlsym(_sharedLibraryHandle, outputSignatureNameWithTag.c_str()));
#endif
  if (!_outputSignatureFunc)
    throw ExecutionSessionException(
        "Cannot load symbol: '" + outputSignatureNameWithTag + "'.");

#if defined(_WIN32)
  _printInstrumentationFunc = reinterpret_cast<printInstrumentationFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(
          _printInstrumentationName.c_str()));
#else
  _printInstrumentationFunc = reinterpret_cast<printInstrumentationFuncType>(
      dlsym(_sharedLibraryHandle, _printInstrumentationName.c_str()));
#endif
  if (!_printInstrumentationFunc)
    throw ExecutionSessionException(
        "Cannot load symbol: '" + _printInstrumentationName + "'.");

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
#if defined(_WIN32)
  if (_sharedLibraryHandle.isValid())
    llvm::sys::DynamicLibrary::closeLibrary(_sharedLibraryHandle);
#else
  if (_sharedLibraryHandle)
    dlclose(_sharedLibraryHandle);
#endif
}

// =============================================================================
// Setter and getter.

const std::string *ExecutionSession::queryEntryPoints(
    int64_t *numOfEntryPoints) const {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  return reinterpret_cast<const std::string *>(
      _queryEntryPointsFunc(numOfEntryPoints));
}

void ExecutionSession::setEntryPoint(const std::string &entryPointName) {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
#if defined(_WIN32)
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      _sharedLibraryHandle.getAddressOfSymbol(entryPointName.c_str()));
#else
  _entryPointFunc = reinterpret_cast<entryPointFuncType>(
      dlsym(_sharedLibraryHandle, entryPointName.c_str()));
#endif
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Cannot load symbol: '" + entryPointName + "'.");
  _entryPointName = entryPointName;
  errno = 0; // No errors.
}

const std::string ExecutionSession::inputSignature() const {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Must set an entry point (e.g. run_main_graph) before calling "
        "signature function.");
  errno = 0; // No errors.
  return _inputSignatureFunc(_entryPointName.c_str());
}

const std::string ExecutionSession::outputSignature() const {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Must set an entry point (e.g. run_main_graph) before calling "
        "signature function.");
  errno = 0; // No errors.
  return _outputSignatureFunc(_entryPointName.c_str());
}

void ExecutionSession::printInstrumentation() {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  errno = 0; // No errors.
  return _printInstrumentationFunc();
}

// =============================================================================
// Run.

// When executing a graph (e.g. run_main_graph, function referenced as
// _entryPointFunc), we may encounter the following runtime errors (returned by
// errno). Error is first signaled by _entryPointFunc returning null.
//
// EPERM (accelerator compatibility check).
//
// May call functions in OMExternalConstant.c, which in turn calls
// malloc/open/map who can generate errno.
//
// Operations in src/Runtime/OM*.c (which may be called by _entryPointFunc)
// generates asserts. In Posix (Linux and MacOS), abort triggers a SIGABRT that
// can be caught by a "sigaction(SIGABRT, &sa, NULL)" installed handler. They
// also have malloc that can generate errno.

std::vector<OMTensorUniquePtr> ExecutionSession::run(
    std::vector<OMTensorUniquePtr> ins) {
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Must set an entry point (e.g. run_main_graph) before calling run "
        "function.");

  std::vector<OMTensor *> omts;
  for (const auto &inOmt : ins)
    omts.emplace_back(inOmt.get());
  auto *wrappedInput =
      omTensorListCreate(omts.data(), static_cast<int64_t>(omts.size()));

  // Run inference.
  errno = 0; // Clear errno.
  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  // We created a wrapper for the input list, but the input list does not really
  // own the tensor in the list, as they are coming as OMTensorUniquePtr. So we
  // need to simply deallocate the list structure without touching the
  // OMTensors.
  omTensorListDestroyShallow(wrappedInput);
  if (!wrappedOutput)
    throw ExecutionSessionException(reportErrnoError());

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
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Must set an entry point (e.g. run_main_graph) before calling run "
        "function.");

  // Run inference.
  errno = 0; // Clear errno.
  OMTensorList *output = _entryPointFunc(input);
  if (!output)
    throw ExecutionSessionException(reportErrnoError());
  errno = 0; // No errors.
  return output;
}

// Run with signal handling to catch crashes (POSIX only).
OMTensorList *ExecutionSession::runWithSignalHandler(OMTensorList *input) {
#if defined(_WIN32)
  // Run with signal is not supported under Windows.
  return run(input);
#endif
  if (!isInitialized)
    throw ExecutionSessionException(
        "Execution session must be initialized once.");
  if (!_entryPointFunc)
    throw ExecutionSessionException(
        "Must set an entry point (e.g. run_main_graph) before calling run "
        "function.");

  // Save old signal handlers
  struct sigaction oldSigsegv, oldSigbus, oldSigfpe, oldSigill;
  struct sigaction newAction;

  // Setup new signal handler
  newAction.sa_handler = signalHandler;
  sigemptyset(&newAction.sa_mask);
  newAction.sa_flags = 0;

  // Install signal handlers for the signals we want to catch
  sigaction(SIGSEGV, &newAction, &oldSigsegv);
  sigaction(SIGBUS, &newAction, &oldSigbus);
  sigaction(SIGFPE, &newAction, &oldSigfpe);
  sigaction(SIGILL, &newAction, &oldSigill);
  sigaction(SIGABRT, &newAction, &oldSigill);

  // Reset signal state
  signalCaught = 0;
  signalNumber = 0;

  OMTensorList *output = nullptr;

  // Set up the jump point for signal handler
  int signum = setjmp(signalJumpBuffer);

  if (signum == 0) {
    // First time through - run the inference
    errno = 0; // Clear errno.
    output = _entryPointFunc(input);

#if 0
    #warning "Injecting a null ptr dereferencing for testing only.
    float *nullPtr = nullptr;
    fprintf(stderr, "about to dereference a null pointer\n");
    fprintf(stderr, "%f\n", *nullPtr);
#endif

    // Restore old signal handlers
    sigaction(SIGSEGV, &oldSigsegv, nullptr);
    sigaction(SIGBUS, &oldSigbus, nullptr);
    sigaction(SIGFPE, &oldSigfpe, nullptr);
    sigaction(SIGILL, &oldSigill, nullptr);
    sigaction(SIGABRT, &oldSigill, nullptr);
    // Throw error if nullptr output (list expected in successful executions).
    if (!output)
      throw ExecutionSessionException(reportErrnoError());
    // No errors, clear potential errno and return output list.
    errno = 0;
    return output;
  } else {
    // We got here via longjmp from signal handler
    // Restore old signal handlers
    sigaction(SIGSEGV, &oldSigsegv, nullptr);
    sigaction(SIGBUS, &oldSigbus, nullptr);
    sigaction(SIGFPE, &oldSigfpe, nullptr);
    sigaction(SIGILL, &oldSigill, nullptr);
    // Set errno to the signal number for error reporting
    errno = signum;
    throw ExecutionSessionException(reportErrnoError(/*from signal*/ true));
  }
}

// =============================================================================
// Error reporting

std::string ExecutionSession::reportErrnoError(bool fromSignal) const {
  std::string errMessageStr = std::string(strerror(errno));
  std::stringstream errStr;
  errStr << "Runtime error during inference returning with ERRNO " << errno
         << ", '" << errMessageStr
         << (fromSignal ? "', caught in a signal handler." : "'.") << std::endl;
  return errStr.str();
}

} // namespace onnx_mlir
