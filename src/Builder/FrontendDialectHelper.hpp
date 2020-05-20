//===--------------------- FrontendDialectHelper.hpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <numeric>
#include <regex>
#include <tuple>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "onnx/onnx_pb.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#if INCLUDE_ONNX_ML == 1
#include "src/Dialect/MLONNX/MLONNXOps.hpp"
#endif

namespace onnx_mlir {

void replaceAll(
    std::string &str, const std::string &from, const std::string &to);

std::string legalize_name(std::string name);

struct OnnxMlirSymbolMapping {
  /*!
   *  Get MLIR tensor by onnx tensor name.
   *  @param name onnx tensor name.
   *  @return onnx mlir tensor corresponding to `name`.
   */
  mlir::Value GetTensorByOnnxName(const std::string &name);

  /*!
   *  Add a new mapping from onnx tensor name to MLIR symbol.
   *  @param name onnx tensor name.
   *  @param tensor MLIR Value  pointer.
   */
  void AddMapping(const std::string &name, mlir::Value tensor);

  bool ContainKey(std::string name);

private:
  /*!
   *  mapping from onnx tensor names to MLIR tensor.
   */
  std::map<std::string, mlir::Value> onnx_name2onnx_mlir_tensor;
};

struct InitializedTensorMapping {
  // Add new entry.
  void AddMapping(std::string name, onnx::TensorProto tensor);

  // Check if input is initialized. Not all inputs are, some of the inputs
  // require input from the user and are not stored inside the ONNX model
  // itself.
  bool ContainKey(std::string name);

  // Emit constant argument (initialized arguments) as a ConstantOp.
  // This method will allow operations to use the constant data contained
  // in an ONNX model as they are being compiled.
  // This method enables the emission of such constant operation on demand.
  //
  // This will allow the propagation of shape information passed in as an
  // argument to operations such as Reshape and will enable other
  // optimizations such as constant folding.
  mlir::Value EmitInitializerForInputTensor(
      mlir::Location loc, mlir::OpBuilder &builder, const std::string &name);

  // Get initialized tensor.
  onnx::TensorProto &GetInitializedTensor(std::string name) {
    assert(
        nameToInitializedTensor.find(name) != nameToInitializedTensor.end() &&
        "Tensor initializer not found");
    return nameToInitializedTensor.at(name);
  }

private:
  // Mapping from ONNX tensor name to InitializedTensor.
  std::map<std::string, onnx::TensorProto> nameToInitializedTensor;
};

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(
    mlir::OpBuilder &builder, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
