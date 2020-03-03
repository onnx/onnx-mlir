//===- frontend_dialect_transformer.hpp - MLIR Operations -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <regex>
#include <tuple>

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "src/dialect/onnx/onnx_ops.hpp"

#include "onnx/onnx_pb.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Helper methods for handling input ONNX models.
//===----------------------------------------------------------------------===//

namespace onnf {
namespace {

void replaceAll(std::string &str, const std::string &from,
                const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

std::string legalize_name(std::string name) {
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  replaceAll(name, ":", "_colon_");
  // If tensor name starts with a number, prepend n to make it a legal c++
  // identifier.
  if (name.size() > 0 && isdigit(name.at(0)))
    name.insert(0, 1, 'n');
  return name;
}

struct OnnxOnnfSymbolMapping {
  /*!
   *  Get MLIR tensor by onnx tensor name.
   *  @param name onnx tensor name.
   *  @return onnf tensor corresponding to `name`.
   */
  mlir::Value GetTensorByOnnxName(const std::string &name) {
    assert(onnx_name2onnf_tensor.find(legalize_name(name)) !=
               onnx_name2onnf_tensor.end() &&
           "Tensor not found");
    return onnx_name2onnf_tensor.at(legalize_name(name));
  }

  /*!
   *  Add a new mapping from onnx tensor name to MLIR symbol.
   *  @param name onnx tensor name.
   *  @param tensor MLIR Value  pointer.
   */
  void AddMapping(const std::string &name, mlir::Value tensor) {
    assert(onnx_name2onnf_tensor.count(legalize_name(name)) == 0 &&
           "Tensor already exists.");
    onnx_name2onnf_tensor.emplace(legalize_name(name), tensor);
  }

  bool ContainKey(std::string name) {
    return onnx_name2onnf_tensor.count(name) != 0;
  }

private:
  /*!
   *  mapping from onnx tensor names to MLIR tensor.
   */
  std::map<std::string, mlir::Value> onnx_name2onnf_tensor;
};

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<T> data(
      onnx::TensorProto initializer) {
    return google::protobuf::RepeatedField<T>();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> data(
      onnx::TensorProto initializer) {
    return initializer.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> data(
      onnx::TensorProto initializer) {
    return initializer.float_data();
  }
};

template <>
struct TransformValueToONNXData<int32_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> data(
      onnx::TensorProto initializer) {
    return initializer.int64_data();
  }
};

struct InitializedTensorMapping {
  // Add new entry.
  void AddMapping(std::string name, onnx::TensorProto tensor) {
    assert(nameToInitializedTensor.count(name) == 0 &&
           "Tensor initializer already mapped.");
    nameToInitializedTensor.emplace(name, tensor);
  }

  // Check if input is initialized. Not all inputs are, some of the inputs
  // require input from the user and are not stored inside the ONNX model
  // itself.
  bool ContainKey(std::string name) {
    return nameToInitializedTensor.count(name) != 0;
  }

  // Emit constant argument (initialized arguments) as a ConstantOp.
  // This method will allow operations to use the constant data contained
  // in an ONNX model as they are being compiled.
  // This method enables the emission of such constant operation on demand.
  //
  // This will allow the propagation of shape information passed in as an
  // argument to operations such as Reshape and will enable other
  // optimizations such as constant folding.
  mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
  	  mlir::OpBuilder &builder, std::string name) {
  	// Initializer for input.
  	onnx::TensorProto initializer = GetInitializedTensor(name);

    // Emit ConstantOp and record the mapping between the input and
    // the constant value.
    mlir::ArrayAttr constantArrayAttribute;
    mlir::Type elementType;
    int length;
    switch (initializer.data_type()) {
      case (onnx::TensorProto::FLOAT): {
        float *typeArray =
            CreateArrayAttribute<float>(initializer, &length);
        std::vector<float> arrayAttrInitializer(
        	typeArray, typeArray + length);
        llvm::ArrayRef<float> array(typeArray, length);
        constantArrayAttribute = builder.getF32ArrayAttr(array);
        elementType = builder.getF32Type();
        break;
      }
      case (onnx::TensorProto::INT32): {
        int32_t *typeArray =
            CreateArrayAttribute<int32_t>(initializer, &length);
        std::vector<int32_t> arrayAttrInitializer(
        	typeArray, typeArray + length);
        llvm::ArrayRef<int32_t> array(typeArray, length);
        constantArrayAttribute = builder.getI32ArrayAttr(array);
        elementType = builder.getIntegerType(32);
        break;
      }
      case (onnx::TensorProto::INT64): {
        int64_t *typeArray =
            CreateArrayAttribute<int64_t>(initializer, &length);
        std::vector<int64_t> arrayAttrInitializer(
        	typeArray, typeArray + length);
        llvm::ArrayRef<int64_t> array(typeArray, length);
        constantArrayAttribute = builder.getI64ArrayAttr(array);
        elementType = builder.getIntegerType(64);
        break;
      }
    }

    llvm::ArrayRef<int64_t> tensorDims(initializer.dims().data(),
        initializer.dims().size());
    mlir::Type tensorType =
        mlir::RankedTensorType::get(tensorDims, elementType);

    return builder.create<mlir::ONNXConstantTensorOp>(
        loc, tensorType, constantArrayAttribute);
  }

private:
  // Get initialized tensor.
  onnx::TensorProto& GetInitializedTensor(std::string name) {
    assert(nameToInitializedTensor.find(name) !=
               nameToInitializedTensor.end() &&
           "Tensor initializer not found");
    return nameToInitializedTensor.at(name);
  }

  // Helper method for constructing an array attribute from a model input.
  template <typename T> T* CreateArrayAttribute(onnx::TensorProto initializer,
       int *size) {
    if (initializer.raw_data().size()) {
      // copy & take care of endianness
      std::vector<char> byteInitializer;
      std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
          back_inserter(byteInitializer));
      *size = initializer.raw_data().size() / sizeof(T);
      return reinterpret_cast<T*>(&byteInitializer[0]);
    }

    // copy, no need to take care of endianness
    auto data = TransformValueToONNXData<T>::data(initializer);
    *size = data.size();
    return &data[0];
  }

  // Mapping from ONNX tensor name to InitializedTensor.
  std::map<std::string, onnx::TensorProto> nameToInitializedTensor;
};

} // namespace
} // namespace onnf

//===----------------------------------------------------------------------===//
// Import a model into one of ONNF's frontend models.
//===----------------------------------------------------------------------===//

namespace onnf {
/*!
 *  Import an ONNX model file into ONNF's ONNX Dialect.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @return MLIR::module generated for the ONNX model.
 */
void ImportFrontendModelFile(std::string model_fname,
                             mlir::MLIRContext &context,
                             mlir::OwningModuleRef &module);

/*!
 *  The list of tensors initialized by the ONNX model.
 */
InitializedTensorMapping initializedTensors;

/*!
 *  TODO: Import models into other extension dialects that cover the
 *  operations specific to other frameworks such as Tensorflow or Pytorch.
 */
}  // namespace onnf
