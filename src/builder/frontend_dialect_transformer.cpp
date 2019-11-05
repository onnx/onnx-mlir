//===- frontend_dialect_transformer.cpp - MLIR Operations -----------------===//
//
// Copyright 2019 The IBM Research Authors. 
//
// =============================================================================
//
// This file transforms the input to available MLIR dialects that can represent
// the operations of the model. Models use the ONNX dialect and any other
// extension dialects that comprise the the operations not supported or covered
// by the ONNX specification.
//
// A `frontend` placeholder dialect is used to encode operations that are not
// covered by any existing dialects.
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <regex>
#include <string>
#include <tuple>

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "frontend_dialect_transformer.hpp"
#include "src/compiler/dialect/onnx/onnx_ops.hpp"

namespace onnf {
namespace {

void replaceAll(
    std::string& str, const std::string& from, const std::string& to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
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
  mlir::Value* GetTensorByOnnxName(std::string name) {
    return onnx_name2onnf_tensor.at(legalize_name(name));
  }

  /*!
   *  Add a new mapping from onnx tensor name to MLIR symbol.
   *  @param name onnx tensor name.
   *  @param tensor MLIR Value* pointer.
   */
  void AddMapping(std::string name, mlir::Value* tensor) {
    onnx_name2onnf_tensor.emplace(legalize_name(name), tensor);
  }

  bool ContainKey(std::string name) {
    return onnx_name2onnf_tensor.count(name) != 0;
  }

 private:
  /*!
   *  mapping from onnx tensor names to MLIR tensor.
   */
  std::map<std::string, mlir::Value*> onnx_name2onnf_tensor;
};

class FrontendGenImpl {
 public:
  FrontendGenImpl(mlir::MLIRContext& context)
      : context_(context), builder_(&context) {
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  mlir::ModuleOp ImportONNXModel(onnx::ModelProto model) {
    ImportGraph(model.graph());
    return module_;
  }

 private:
  mlir::MLIRContext& context_;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  // mapping between string name and symbol
  OnnxOnnfSymbolMapping frontend_symbols_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

  // Convert type to MLIR type.
  // A complete list of types can be found in:
  // <dlc-build-folder>/third_party/onnx/onnx/onnx.pb.h
  mlir::Type TypeConvert(onnx::TensorProto_DataType intype) {
    switch (intype) {
      case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
        return builder_.getF16Type();
      case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
        return builder_.getF32Type();
      case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
        return builder_.getF64Type();
      case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
      case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
        return builder_.getIntegerType(8);
      case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
      case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
        return builder_.getIntegerType(16);
      case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
        return builder_.getIntegerType(32);
      case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
        return builder_.getIntegerType(64);
      case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
        return builder_.getI1Type();
      case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
      case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
      case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
      case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
        DLC_REQUIRE_MSG_CTX(false, "Unsupported data type encountered.");
        return nullptr;
    }
  }

  void ImportInputTensor(onnx::ValueInfoProto& input) {
    std::vector<int64_t> dims;
    auto shape_proto = input.type().tensor_type().shape();
    auto input_tensor_legalized_name = legalize_name(input.name());
    for (int i = 0; i < shape_proto.dim_size(); i++) {
      if (shape_proto.dim()[i].dim_value()) {
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        if (dim_numeric_size > 0) {
          dims.push_back(dim_numeric_size);
        } else {  // If dim_value < 0, then dim is parametric.
                  // TODO Verify the unknown dim size in MLIR
          dims.push_back(-1);
        }
      } else {
        // TODO How to represent variable length
        dims.push_back(-1);
      }
    }
    if (!frontend_symbols_.ContainKey(input_tensor_legalized_name)) {
      mlir::Type elementType =
          TypeConvert(input.type().tensor_type().elem_type());
      llvm::ArrayRef<int64_t> llvmdimsAR(dims.data(), dims.size());
      auto dataType = mlir::RankedTensorType::get(llvmdimsAR, elementType);
      mlir::OperationState result(
          UnknownLoc(), "frontend.input " + input_tensor_legalized_name);
      result.addTypes(dataType);
      auto op = builder_.createOperation(result);
      auto value = op->getResult(0);
      frontend_symbols_.AddMapping(input_tensor_legalized_name, value);
    } else {
      // TODO  Should not happen
    }
  }

  void ImportNode(onnx::NodeProto node) {
    std::vector<mlir::Value*> inputs;
    for (auto item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    // Handle ONNX Add Operation by using its representation in the
    // ONNX Dialect.
    llvm::StringRef OpName = node.op_type();
    if (OpName == "Add") {
      auto op =
          builder_.create<mlir::ONNXAddOp>(UnknownLoc(), inputs[0], inputs[1]);
      frontend_symbols_.AddMapping(
          legalize_name(node.output()[0]), op.getResult());
      return;
    }

    // Old way of doing things.
    mlir::OperationState result(UnknownLoc(), "frontend." + node.op_type());
    for (auto item : node.output()) {
      result.addTypes(mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }
    result.addOperands(inputs);
    auto op = builder_.createOperation(result);
    for (int i = 0; i < node.output().size(); i++) {
      auto r = op->getResult(i);
      frontend_symbols_.AddMapping(legalize_name(node.output()[i]), r);
    }

    // TODO more info from node: attributes
  }

  void ImportOutputTensor(onnx::ValueInfoProto& output) {
    if (frontend_symbols_.ContainKey(legalize_name(output.name()))) {
      mlir::OperationState result(
          UnknownLoc(), "frontend.output " + output.name());
      mlir::Type elementType =
          TypeConvert(output.type().tensor_type().elem_type());
      result.addTypes(mlir::UnrankedTensorType::get(elementType));
      result.addOperands(frontend_symbols_.GetTensorByOnnxName(output.name()));
      builder_.createOperation(result);
    } else {
      // TODO: Why not in the symbol table? something is wrong
    }
  }

  void ImportGraph(onnx::GraphProto graph) {
    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<mlir::Type, 4> ret_types;
    llvm::SmallVector<mlir::Type, 4> arg_types;
    auto func_type = builder_.getFunctionType(arg_types, ret_types);
    auto llvmfunction = mlir::FuncOp::create(
        UnknownLoc(), graph.name(), func_type, /* attrs = */ {});
    auto& entryBlock = *llvmfunction.addEntryBlock();
    builder_.setInsertionPointToStart(&entryBlock);
    module_.push_back(llvmfunction);

    // TODO: import the initializer
    //

    // import the input tensors
    for (auto input : graph.input()) {
      ImportInputTensor(input);
    }

    // import nodes in the graph
    auto node = graph.node();
    for (auto item : node) {
      ImportNode(item);
    }

    // import the output tensors
    for (auto output : graph.output()) {
      ImportOutputTensor(output);
    }
  }

};  // FrontendGenImpl class

}  // namespace
}  // namespace onnf

namespace onnf {

mlir::OwningModuleRef ImportFrontendModel(onnx::ModelProto model) {
  mlir::MLIRContext context;
  FrontendGenImpl myONNXGen(context);
  auto module = myONNXGen.ImportONNXModel(model);
  module.dump();

  return module;
}

mlir::OwningModuleRef ImportFrontendModelFile(std::string model_fname) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);

  return ImportFrontendModel(model);
}
}  // namespace onnf
