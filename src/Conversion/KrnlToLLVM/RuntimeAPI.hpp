/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ RuntimeAPI.hpp - Declaration of the Runtime API ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declare the Runtime API the compiler can use.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_RUNTIMEAPI
#define ONNX_RUNTIMEAPI

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

#include <string>

using namespace mlir;

class RuntimeAPIRegistry;

/// \class RuntimeAPI
/// Represents a Runtime API callable by the compiler.
/// Instances of this class can only be created by the RuntimeAPIRegistry
/// singleton class.
class RuntimeAPI final {
  friend class RuntimeAPIRegistry;

public:
  // Enumerate the runtime functions.
  enum class API {
    CREATE_OMTENSOR_LIST,
    CREATE_OMTENSOR,
    GET_DATA,
    SET_DATA,
    GET_DATA_SHAPE,
    GET_DATA_STRIDES,
    SET_DATA_TYPE,
    GET_DATA_TYPE,
    GET_OMT_ARRAY,
    PRINT_OMTENSOR,
  };

  // Call the runtime API identified by \p apiId, return the SSA value
  // representing the call.
  static Value callApi(OpBuilder &builder, Location loc,
      const RuntimeAPIRegistry &registry, API apiId, ArrayRef<Value> params);

private:
  RuntimeAPI(
      API id, const std::string &name, Type outputTy, ArrayRef<Type> inputTys)
      : id(id), name(name), outputTy(outputTy),
        inputTys(inputTys.begin(), inputTys.end()) {}

  // Inject the declaration for this runtime API into the given module (unless a
  // declaration exists already).
  void declareAPI(ModuleOp &module, OpBuilder &builder);

  static FlatSymbolRefAttr getOrInsertExternFunc(StringRef funcName,
      ModuleOp module, mlir::Type funcType, OpBuilder &builder);

private:
  API id;
  std::string name;
  Type outputTy;
  SmallVector<Type, 4> inputTys;
  FlatSymbolRefAttr symbolRef;
};

/// \class RuntimeAPIRegistry
/// Holds the registry for the Runtime APIs the compiler can use.
/// There is a single instance of this class in the program (singleton pattern).
class RuntimeAPIRegistry final {
public:
  using ApiRegistry = std::map<RuntimeAPI::API, RuntimeAPI>;

  ~RuntimeAPIRegistry();

  static const RuntimeAPIRegistry build(ModuleOp &module, OpBuilder &builder);

  const RuntimeAPI &getAPI(RuntimeAPI::API apiId) const {
    assert((registry.find(apiId) != registry.end()) &&
           "apiId not found in registry");
    return registry.at(apiId);
  }

private:
  RuntimeAPIRegistry(ModuleOp &module, OpBuilder &builder);

  static RuntimeAPIRegistry *instance;

  ApiRegistry registry;
};

#endif // ONNX_RUNTIMEAPI
