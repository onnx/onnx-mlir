/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ExecutionSession.hpp - ExecutionSession Declaration --------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_EXECUTION_SESSION_H
#define ONNX_MLIR_EXECUTION_SESSION_H

#include <cassert>
#include <memory>
#include <string>

#include "OnnxMlirRuntime.h"

// LLVM provides the wrapper class, llvm::sys::DynamicLibrary, for dynamic
// library. When PYRUNTIME_LIGHT is built without the LLVM, the handle type for
// dynamic library in Linux is used. DynamicLibraryHandleType is defined for
// the two cases.
#if defined(_WIN32)
#include "llvm/Support/DynamicLibrary.h"
typedef llvm::sys::DynamicLibrary DynamicLibraryHandleType;
#else
typedef void *DynamicLibraryHandleType;
#include <dlfcn.h>
#endif

// TODO: should ExecutionSession and OMCompile be in the onnx_mlir
// namespace? They should not depend at all on the onnx-mlir compiler files
// (except implicitly).

namespace onnx_mlir {

using entryPointFuncType = OMTensorList *(*)(OMTensorList *);
using queryEntryPointsFuncType = const char **(*)(int64_t *);
using signatureFuncType = const char *(*)(const char *);
using printInstrumentationFuncType = void (*)(void);
using OMTensorUniquePtr = std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>;

/* ExecutionSession
 * Class that supports executing compiled models.
 *
 * When the execution session does not work for known reasons, this class will
 * throw std::ExecutionSessionException errors.
 */

// Exception class
class ExecutionSessionException : public std::runtime_error {
public:
  explicit ExecutionSessionException(const std::string &msg)
      : std::runtime_error(msg) {}
};

class ExecutionSession {
public:
  ExecutionSession() = default;

  // Create an execution session using the model given in sharedLibPath.
  // This path must point to the actual file, local directory is not searched.
  // Throw errors on failure.
  ExecutionSession(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);
  ~ExecutionSession();

  // Initialization of library. Called by public constructor, or by subclasses.
  void loadModel(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);

  // Get a NULL-terminated array of entry point names.
  // For example {"run_addition, "run_subtraction", NULL}
  // In order to get the number of entry points, pass an integer pointer to the
  // function.
  const std::string *queryEntryPoints(int64_t *numOfEntryPoints) const;

  // Set entry point for this session.
  // Call this before running the session or querying signatures if
  // defaultEntryPoint is false or there are multiple entry points in the model.
  void setEntryPoint(const std::string &entryPointName);

  DynamicLibraryHandleType &getSharedLibraryHandle() {
    return _sharedLibraryHandle;
  };

  // Use custom deleter since forward declared OMTensor hides destructor
  std::vector<OMTensorUniquePtr> run(std::vector<OMTensorUniquePtr>);

  // Run using public interface. Explicit calls are needed to free tensor &
  // tensor lists.
  OMTensorList *run(OMTensorList *input);

  // Get input and output signature as a Json string. For example for nminst:
  // `[ { "type" : "f32" , "dims" : [1 , 1 , 28 , 28] , "name" : "image" } ]`
  const std::string inputSignature() const;
  const std::string outputSignature() const;
  void printInstrumentation();

protected:
  // Error reporting processing when throwing runtime errors.
  std::string reportErrnoError() const;

  // Track if Init was called or not.
  bool isInitialized = false;

  // Handler to the shared library file being loaded.
  DynamicLibraryHandleType _sharedLibraryHandle;

  // Tag used to compile the model. By default, it is the model filename without
  // extension.
  std::string tag;

  // Entry point function.
  std::string _entryPointName;
  entryPointFuncType _entryPointFunc = nullptr;

  // Query entry point function.
  const std::string _queryEntryPointsName = "omQueryEntryPoints";
  queryEntryPointsFuncType _queryEntryPointsFunc = nullptr;

  // Entry point for input/output signatures
  const std::string _inputSignatureName = "omInputSignature";
  const std::string _outputSignatureName = "omOutputSignature";
  signatureFuncType _inputSignatureFunc = nullptr;
  signatureFuncType _outputSignatureFunc = nullptr;

  // Entry point for printing instrumentation
  const std::string _printInstrumentationName = "omInstrumentPrint";
  printInstrumentationFuncType _printInstrumentationFunc = nullptr;
};

} // namespace onnx_mlir
#endif
