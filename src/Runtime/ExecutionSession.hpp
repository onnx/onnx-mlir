/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ExecutionSession.hpp - ExecutionSession Declaration --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include "OnnxMlirRuntime.h"
#include "llvm/Support/DynamicLibrary.h"

namespace onnx_mlir {

using entryPointFuncType = OMTensorList *(*)(OMTensorList *);
using queryEntryPointsFuncType = const char **(*)(int64_t *);
using signatureFuncType = const char *(*)(const char *);
using OMTensorUniquePtr = std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>;

/* ExecutionSession
 * Class that supports executing compiled models.
 *
 * When the execution session does not work for known reasons, this class will
 * throw std::runtime_error errors. Errno info will provide further info about
 * the specific error that was raised.
 *
 * EFAULT when it could not load the library or a needed symbol was not found.
 * EINVAL when it expected an entry point prior to executing a specific
 * function.
 * EPERM when the model executed on a machine without a compatible
 * hardware/specialized accelerator.
 */
class ExecutionSession {
public:
  // Create an execution session using the model given in sharedLibPath.
  // This path must point to the actual file, local directory is not searched.
  ExecutionSession(std::string sharedLibPath, bool defaultEntryPoint = true);

  // Get a NULL-terminated array of entry point names.
  // For example {"run_addition, "run_subtraction", NULL}
  // In order to get the number of entry points, pass an integer pointer to the
  // function.
  const std::string *queryEntryPoints(int64_t *numOfEntryPoints) const;

  // Set entry point for this session.
  // Call this before running the session or querying signatures if
  // defaultEntryPoint is false or there are multiple entry points in the model.
  void setEntryPoint(const std::string &entryPointName);

  llvm::sys::DynamicLibrary &getSharedLibraryHandle() {
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

  ~ExecutionSession();

protected:
  // Error reporting processing when throwing runtime errors. Set errno as
  // appropriate.
  std::string reportLibraryOpeningError(const std::string &libraryName) const;
  std::string reportSymbolLoadingError(const std::string &symbolName) const;
  std::string reportUndefinedEntryPointIn(
      const std::string &functionName) const;
  std::string reportErrnoError() const;

protected:
  // Handler to the shared library file being loaded.
  llvm::sys::DynamicLibrary _sharedLibraryHandle;

  // Entry point function.
  std::string _entryPointName;
  entryPointFuncType _entryPointFunc = nullptr;

  // Query entry point function.
  static const std::string _queryEntryPointsName;
  queryEntryPointsFuncType _queryEntryPointsFunc = nullptr;

  // Entry point for input/output signatures
  static const std::string _inputSignatureName;
  static const std::string _outputSignatureName;
  signatureFuncType _inputSignatureFunc = nullptr;
  signatureFuncType _outputSignatureFunc = nullptr;
};
} // namespace onnx_mlir
