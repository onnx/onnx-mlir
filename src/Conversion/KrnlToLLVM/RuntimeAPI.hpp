/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ RuntimeAPI.hpp - Declaration of the Runtime API ---------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declare the Runtime API the compiler can use.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RUNTIME_API_H
#define ONNX_MLIR_RUNTIME_API_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

#include <string>

class RuntimeAPIRegistry;

/// \class RuntimeAPI
/// Represents a Runtime API callable by the compiler.
/// Instances of this class can only be created by the RuntimeAPIRegistry
/// class.
class RuntimeAPI final {
  friend class RuntimeAPIRegistry;

public:
  // Enumerate the runtime functions.
  enum class API {
    CREATE_OMTENSOR_LIST,
    CREATE_OMTENSOR,
    DESTROY_OMTENSOR,
    GET_DATA,
    SET_DATA,
    GET_DATA_RANK,
    GET_DATA_SHAPE,
    GET_DATA_STRIDES,
    SET_DATA_TYPE,
    GET_DATA_TYPE,
    GET_OMT_ARRAY,
    PRINT_OMTENSOR,
    GET_OMTENSOR_LIST_SIZE,
    MMAP_BINARY_FILE,
    GET_EXTERNAL_CONSTANT_ADDR,
  };

  // Call the runtime API identified by \p apiId, return the SSA value
  // representing the call.
  static mlir::Value callApi(mlir::OpBuilder &builder, mlir::Location loc,
      const RuntimeAPIRegistry &registry, API apiId,
      llvm::ArrayRef<mlir::Value> params);

private:
  RuntimeAPI(API id, const std::string &name, mlir::Type outputTy,
      llvm::ArrayRef<mlir::Type> inputTys)
      : id(id), name(name), outputTy(outputTy),
        inputTys(inputTys.begin(), inputTys.end()) {}

  // Inject the declaration for this runtime API into the given module (unless a
  // declaration exists already).
  void declareAPI(mlir::ModuleOp &module, mlir::OpBuilder &builder);

  static mlir::FlatSymbolRefAttr getOrInsertExternFunc(llvm::StringRef funcName,
      mlir::ModuleOp module, mlir::Type funcType, mlir::OpBuilder &builder);

private:
  API id;
  std::string name;
  mlir::Type outputTy;
  llvm::SmallVector<mlir::Type, 4> inputTys;
  mlir::FlatSymbolRefAttr symbolRef;
};

/// \class RuntimeAPIRegistry
/// Holds the registry for the Runtime APIs the compiler can use.
class RuntimeAPIRegistry final {
public:
  using ApiRegistry = std::map<RuntimeAPI::API, RuntimeAPI>;

  RuntimeAPIRegistry(mlir::ModuleOp &module, mlir::OpBuilder &builder,
      const mlir::LLVMTypeConverter &typeConverter);
  ~RuntimeAPIRegistry();

  static const RuntimeAPIRegistry build(
      mlir::ModuleOp &module, mlir::OpBuilder &builder);

  const RuntimeAPI &getAPI(RuntimeAPI::API apiId) const {
    assert((registry.find(apiId) != registry.end()) &&
           "apiId not found in registry");
    return registry.at(apiId);
  }

private:
  ApiRegistry registry;
};
#endif
