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
#include <map>

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

#include "src/dialect/onnx/onnx_ops.hpp"

#include "frontend_dialect_transformer.hpp"

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
  mlir::Value GetTensorByOnnxName(std::string name) {
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
  void AddMapping(std::string name, mlir::Value tensor) {
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

class FrontendGenImpl {
public:
  FrontendGenImpl(mlir::MLIRContext &context)
      : context_(context), builder_(&context) {
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  mlir::ModuleOp ImportONNXModel(onnx::ModelProto model) {
    ImportGraph(model.graph());
    return module_;
  }

private:
  mlir::MLIRContext &context_;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  // mapping between string name and symbol
  OnnxOnnfSymbolMapping frontend_symbols_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

  // Convert type to MLIR type.
  // A complete list of types can be found in:
  // <onnf-build-folder>/third_party/onnx/onnx/onnx.pb.h
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
        assert(false && "Unsupported data type encountered.");
        return nullptr;
    }
  }

  /*!
   * Import an onnx input tensor type by determining and recording its type
   * in a list of input tensor mlir types.
   * @param input onnx input tensor ValueInfoProto.
   * @param arg_types list of mlir types representing types of graph input.
   */
  void ImportInputTensorType(const onnx::ValueInfoProto &input,
                             llvm::SmallVector<mlir::Type, 4> &arg_types) {
    std::vector<int64_t> dims;
    auto shape_proto = input.type().tensor_type().shape();
    auto input_tensor_legalized_name = legalize_name(input.name());
    for (int i = 0; i < shape_proto.dim_size(); i++) {
      if (shape_proto.dim()[i].dim_value()) {
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        assert(
            dim_numeric_size != 0 && "Parsed an input tensor with a dimension size of zero");
        if (dim_numeric_size > 0) {
          dims.push_back(dim_numeric_size);
        } else { // If dim_value < 0, then dim is parametric.
                 // TODO Verify the unknown dim size in MLIR
          dims.push_back(-1);
        }
      } else {
        // TODO How to represent variable length
        dims.push_back(-1);
      }
    }

    mlir::Type elementType =
        TypeConvert(input.type().tensor_type().elem_type());
    llvm::ArrayRef<int64_t> tensor_dims(dims.data(), dims.size());
    arg_types.emplace_back(
        mlir::RankedTensorType::get(tensor_dims, elementType));
  }

  /*!
   * Import a input tensor symbol by recording a new entry in frontend_symbols_
   * recording the mapping between legalized onnx tensor name and mlir::Value
   * for further lookup in computation node importing.
   * @param input onnx input tensor ValueInfoProto.
   * @param symbol mlir input argument.
   */
  void ImportInputTensorSymbol(const onnx::ValueInfoProto &input,
                               mlir::Value symbol) {
    auto input_tensor_legalized_name = legalize_name(input.name());
    assert(
        !frontend_symbols_.ContainKey(input_tensor_legalized_name) &&
        "Found duplicate legalized input tensor names.");
    frontend_symbols_.AddMapping(input_tensor_legalized_name, symbol);
  }

  template <typename T>
  T get_attr_generic(onnx::NodeProto &node, std::string name,
                     std::function<T(onnx::AttributeProto &)> attr_getter,
                     T default_val) {
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      if (attr.name() == name) {
        return attr_getter(attr);
      }
    }
    return default_val;
  }

  template <typename T>
  T get_attr_generic(onnx::NodeProto &node, std::string name,
                     std::function<T(onnx::AttributeProto &)> attr_getter) {
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      if (attr.name() == name) {
        return attr_getter(attr);
      }
    }
    assert(false && "ONNX Node Attribute Not Found!");
  }

  auto get_attr_ints(onnx::NodeProto &node, std::string name,
                     std::vector<int> default_val) {
    std::function<std::vector<int>(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) {
          std::vector<int> ints(attr.ints_size());
          std::copy(attr.ints().begin(), attr.ints().end(), ints.begin());
          return ints;
        };
    auto r = get_attr_generic(node, name, attr_getter, default_val);
    auto dataType =
        mlir::RankedTensorType::get(r.size(), builder_.getIntegerType(32));
    auto attr_v = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(r));
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_ints(onnx::NodeProto &node, std::string name) {
    std::function<std::vector<int>(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) {
          std::vector<int> ints(attr.ints_size());
          std::copy(attr.ints().begin(), attr.ints().end(), ints.begin());
          return ints;
        };
    auto r = get_attr_generic(node, name, attr_getter);
    auto dataType =
        mlir::RankedTensorType::get(r.size(), builder_.getIntegerType(32));
    auto attr_v = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(r));
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_floats(onnx::NodeProto &node, std::string name) {
    std::function<std::vector<float>(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) {
          std::vector<float> floats(attr.floats_size());
          std::copy(attr.floats().begin(), attr.floats().end(), floats.begin());
          return floats;
        };
    auto r = get_attr_generic(node, name, attr_getter);
    auto dataType =
        mlir::RankedTensorType::get(r.size(), builder_.getF32Type());
    auto attr_v = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(r));
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_floats(onnx::NodeProto &node, std::string name,
                       std::vector<float> default_val) {
    std::function<std::vector<float>(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) {
          std::vector<float> floats(attr.floats_size());
          std::copy(attr.floats().begin(), attr.floats().end(), floats.begin());
          return floats;
        };
    auto r = get_attr_generic(node, name, attr_getter, default_val);
    auto dataType =
        mlir::RankedTensorType::get(r.size(), builder_.getF32Type());
    auto attr_v = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(r));
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_int(onnx::NodeProto &node, std::string name) {
    std::function<int(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.i(); };
    int r = get_attr_generic(node, name, attr_getter);
    auto attr_v = builder_.getI32IntegerAttr(r);
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_int(onnx::NodeProto &node, std::string name, int default_val) {
    std::function<int(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.i(); };
    int r = get_attr_generic(node, name, attr_getter, default_val);
    auto attr_v = builder_.getI32IntegerAttr(r);
    auto aname = node.op_type() + "." + name;
    auto attr_output = builder_.getNamedAttr(aname, attr_v);
    return attr_output;
  }

  auto get_attr_float(onnx::NodeProto &node, std::string name) {
    std::function<float(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.f(); };
    auto r = get_attr_generic(node, name, attr_getter);
    auto attr_v = builder_.getF32FloatAttr(r);
    auto aname = node.op_type() + "." + name;
    return builder_.getNamedAttr(aname, attr_v);
  }

  auto get_attr_float(onnx::NodeProto &node, std::string name,
                      float default_val) {
    std::function<float(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.f(); };
    auto r = get_attr_generic(node, name, attr_getter, default_val);
    auto attr_v = builder_.getF32FloatAttr(r);
    auto aname = node.op_type() + "." + name;
    return builder_.getNamedAttr(aname, attr_v);
  }

  auto get_attr_string(onnx::NodeProto &node, std::string name) {
    std::function<std::string(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.s(); };
    auto r = get_attr_generic(node, name, attr_getter);
    auto attr_v = builder_.getStringAttr(r);
    auto aname = node.op_type() + "." + name;
    return builder_.getNamedAttr(aname, attr_v);
  }

  auto get_attr_string(onnx::NodeProto &node, std::string name,
                       std::string default_val) {
    std::function<std::string(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.s(); };
    auto r = get_attr_generic(node, name, attr_getter, default_val);
    auto attr_v = builder_.getStringAttr(r);
    auto aname = node.op_type() + "." + name;
    return builder_.getNamedAttr(aname, attr_v);
  }

  /*
    auto get_attr_strings(onnx::NodeProto &node, std::string name) {
      std::function<std::vector<std::string>(onnx::AttributeProto &)>
    attr_getter =
          [](onnx::AttributeProto &attr) {
            std::vector<std::string> strings(attr.strings_size());
            std::copy(attr.strings().begin(), attr.strings().end(),
    strings.begin()); return strings;
          };
      auto r = get_attr_generic(node, name, attr_getter);
      return r;
      return builder_.getNamedAttr(aname, attr_v);
      auto dataType =
          mlir::RankedTensorType::get(r.size(), builder_.get???Type());
      auto attr_v = mlir::DenseElementsAttr::get(dataType,
    llvm::makeArrayRef(r)); auto aname = node.op_type() + "." + name; auto
    attr_output = builder_.getNamedAttr(aname, attr_v); return attr_output;
    }
  */

  auto get_default_ints(std::string default_str) {
    std::vector<int> r;
    auto start = default_str.find("{");
    while (true) {
      auto end = default_str.find(",", start + 1);
      if (end == std::string::npos) {
        end = default_str.find("}", start + 1);
        if (end != std::string::npos && end > start + 1) {
          r.push_back(std::stoi(default_str.substr(start + 1, end)));
        }
        break;
      } else {
        r.push_back(std::stoi(default_str.substr(start + 1, end)));
      }
      start = end + 1;
    }
    return r;
  }

  auto get_default_floats(std::string default_str) {
    std::vector<float> r;
    auto start = default_str.find("{");
    while (true) {
      auto end = default_str.find(",", start + 1);
      if (end == std::string::npos) {
        end = default_str.find("}", start + 1);
        if (end != std::string::npos && end > start + 1) {
          r.push_back(std::stof(default_str.substr(start + 1, end)));
        }
        break;
      } else {
        r.push_back(std::stof(default_str.substr(start + 1, end)));
      }
      start = end + 1;
    }
    return r;
  }

  auto get_default_strings(std::string default_str) {
    std::vector<std::string> r;
    auto start = default_str.find("{");
    while (true) {
      auto end = default_str.find(",", start + 1);
      if (end == std::string::npos) {
        end = default_str.find("}", start + 1);
        if (end != std::string::npos && end > start + 1) {
          r.push_back(default_str.substr(start + 1, end));
        }
        break;
      } else {
        r.push_back(default_str.substr(start + 1, end));
      }
      start = end + 1;
    }
    return r;
  }

  onnx::TensorProto get_attr_tensor(onnx::NodeProto &node, std::string name) {
    std::function<onnx::TensorProto(onnx::AttributeProto &)> attr_getter =
        [](onnx::AttributeProto &attr) { return attr.t(); };
    return get_attr_generic(node, name, attr_getter);
  }

  auto ImportNodeAttr(onnx::NodeProto node, std::string attr_name,
                      std::string type_name, std::string default_str) {
    if (default_str == "") {
      if (type_name == "int") {
        return get_attr_int(node, attr_name);
      } else if (type_name == "float") {
        return get_attr_float(node, attr_name);
      } else if (type_name == "str") {
        return get_attr_string(node, attr_name);
      } else if (type_name == "ints") {
        return get_attr_ints(node, attr_name);
      } else if (type_name == "floats") {
        return get_attr_floats(node, attr_name);
      } else {
        assert(
            false &&
            "Got an empty initializer or initializer for this "
            "datatype is not implemented. Something is wrong.");
      }
    } else {
      // with default value
      if (type_name == "int") {
        return get_attr_int(node, attr_name, std::stoi(default_str));
      } else if (type_name == "float") {
        return get_attr_float(node, attr_name, std::stof(default_str));
      } else if (type_name == "str") {
        return get_attr_string(node, attr_name, default_str);
      } else if (type_name == "ints") {
        return get_attr_ints(node, attr_name, get_default_ints(default_str));
      } else if (type_name == "floats") {
        return get_attr_floats(node, attr_name,
                               get_default_floats(default_str));
      } else {
        assert(
            false &&
            "Got an empty initializer or initializer for this "
            "datatype is not implemented. Something is wrong.");
      }
    }
  }

  void ImportNodeGeneric(onnx::NodeProto node) {
    std::vector<mlir::Value> inputs;
    for (auto item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }
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
  }

  // if c++17 is used, ImportNodeOneOut and ImportNodeMultipleOuts can be
  // combined with 'if constexpr' the issue is the type of the output is
  // different. alternative way to use variadic output for all the op

  /*!
   * Important onnx node which generates only one output
   * @param node onnx node
   * @param nIn number of expected inputs
   * @param nOut number of expected outputs
   * @param attrs  list of desription for attributes with format {name, type,
   * default}
   */
  template <typename T>
  void ImportNodeOneOut(
      onnx::NodeProto node, int nIn, int nOut,
      std::initializer_list<std::tuple<std::string, std::string, std::string>>
          attrs) {
    std::vector<mlir::Value> inputs;
    for (auto item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    std::vector<mlir::Type> outputTypes;
    for (auto item : node.output()) {
      outputTypes.push_back(
          mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }

    std::vector<mlir::NamedAttribute> attributes;
    // for (auto [attr_name, attr_type, attr_default] : attrs) {
    for (auto oneAttr : attrs) {
      std::string attr_name;
      std::string attr_type;
      std::string attr_default;
      std::tie(attr_name, attr_type, attr_default) = oneAttr;
      if (attr_type != "") {
        auto attr = ImportNodeAttr(node, attr_name, attr_type, attr_default);
        attributes.push_back(attr);
      } else {
        // TODO: the attributes need special handling
        // std::cout << "missing " << node.op_type() << " " << attr_name <<
        // std::endl;
      }
    }

    llvm::StringRef OpName = node.op_type();

    if (nIn == inputs.size() && nOut == outputTypes.size()) {
      auto op =
          builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
      frontend_symbols_.AddMapping(legalize_name(node.output()[0]),
                                   op.getResult());
    } else {
      ImportNodeGeneric(node);
    }
  }

  template <typename T>
  void ImportNodeMultipleOuts(
      onnx::NodeProto node, int nIn, int nOut,
      std::initializer_list<std::tuple<std::string, std::string, std::string>>
          attrs) {
    std::vector<mlir::Value> inputs;
    for (auto item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    std::vector<mlir::Type> outputTypes;
    for (auto item : node.output()) {
      outputTypes.push_back(
          mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }

    std::vector<mlir::NamedAttribute> attributes;
    for (auto oneAttr : attrs) {
      std::string attr_name;
      std::string attr_type;
      std::string attr_default;
      std::tie(attr_name, attr_type, attr_default) = oneAttr;
      if (attr_type != "") {
        auto attr = ImportNodeAttr(node, attr_name, attr_type, attr_default);
        attributes.push_back(attr);
      } else {
        // TODO: the attributes need special handling
        // std::cout << "missing " << node.op_type() << " " << attr_name <<
        // std::endl;
      }
    }

    llvm::StringRef OpName = node.op_type();

    if (nIn == inputs.size() && nOut == outputTypes.size()) {
      auto op =
          builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
      for (int i = 0; i < node.output().size(); i++) {
        frontend_symbols_.AddMapping(legalize_name(node.output()[i]),
                                     op.getResult(i));
      }
    } else {
      ImportNodeGeneric(node);
    }
  }

  /*!
   * Special handle for Conv operations.
   * c++ does not allow template specialization inside a class scope
   * a specialized function is used
   */
  void ImportNodeConv(
      onnx::NodeProto node, int nIn, int nOut,
      std::initializer_list<std::tuple<std::string, std::string, std::string>>
          attrs) {
    // Conv has attribute dilations, kernel_shape, pads, the default value of
    // which  is determined by the shape of first argument. However, since the
    // shape is unknown now, these attributes can be not generated auto
    // dilations_attr = get_attr_ints(node, "dilations",
    //    std::vector<int>(inputs[0]->getType().cast<RankedTensorType>.getDims()-2,
    //    1));
    // attributes.push_back(dilations_attr)
    // similar situation for pads, strides in AveragePool
    // axes of ReduceSum,  pads, strides, dilations and kernel_shape of MaxPool
    // TODO: fix this after type inference

    if (node.input().size() == 1) {
      ImportNodeOneOut<mlir::ONNXConv1Op>(node, nIn, nOut, attrs);
    } else {
      ImportNodeOneOut<mlir::ONNXConv3Op>(node, nIn, nOut, attrs);
    }
  }

  void ImportNode(onnx::NodeProto node) {
    std::vector<mlir::Value> inputs;
    for (auto item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    std::vector<mlir::Type> outputTypes;
    for (auto item : node.output()) {
      outputTypes.push_back(
          mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }

    std::vector<mlir::NamedAttribute> attributes;
    llvm::StringRef OpName = node.op_type();

    // the following code is generated by gen_doc.py
    // refer to dialect/onnx/onnx.td for details
    // when the input or output of then op does not match the specification,
    // the generic operator is used
    // one known reeason is the optional input

#include "src/builder/op_build_table.inc"
  }

  /*!
   * Import output tensor, by doing the following:
   * - Add the type of this output tensor to a list of tensor
   *   types representing return types of this graph function.
   * - Add this output tensor to the list of mlir::Value
   *   to be returned by the function representing computation graph.
   * @param output onnx output tensor ValueInfoProto.
   * @param ret_types a vector of tensor types representing graph's
   *   output tensor types.
   * @param ret_vals a vector of mlir Value  representing graph's
   *   output tensor.
   */
  void ImportOutputTensor(const onnx::ValueInfoProto &output,
                          llvm::SmallVectorImpl<mlir::Type> &ret_types,
                          llvm::SmallVectorImpl<mlir::Value> &ret_vals) {
    auto output_tensor_legalized_name = legalize_name(output.name());
    assert(
        frontend_symbols_.ContainKey(output_tensor_legalized_name) &&
        "Output tensor not found");

    auto tensor_val =
        frontend_symbols_.GetTensorByOnnxName(output_tensor_legalized_name);
    ret_types.emplace_back(tensor_val.getType());
    ret_vals.push_back(tensor_val);
  }

  void ImportGraph(const onnx::GraphProto &graph,
                   const std::string &name = "main_graph") {
    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<mlir::Type, 4> arg_types;

    // Import the input tensor types.
    for (const auto &input : graph.input()) {
      ImportInputTensorType(input, arg_types);
    }

    // TODO: import the initializer
    auto funcType = builder_.getFunctionType(arg_types, {});
    auto mainFunc =
        mlir::FuncOp::create(UnknownLoc(), name, funcType, /* attrs = */ {});
    auto entryPoint = mlir::ONNXEntryPointOp::create(
        UnknownLoc(), mainFunc, /*numInputs=*/graph.input().size(),
        /*numOutputs=*/graph.output().size());

    auto &entryBlock = *mainFunc.addEntryBlock();
    builder_.setInsertionPointToStart(&entryBlock);

    module_.push_back(mainFunc);
    module_.push_back(entryPoint);

    for (auto it : llvm::zip(graph.input(), entryBlock.getArguments())) {
      ImportInputTensorSymbol(std::get<0>(it), std::get<1>(it));
    }

    // import nodes in the graph
    auto node = graph.node();
    for (const auto &item : node) {
      ImportNode(item);
    }

    llvm::SmallVector<mlir::Type, 4> ret_types;
    llvm::SmallVector<mlir::Value, 4> ret_vals;
    // Import the output tensors
    for (const auto &output : graph.output()) {
      ImportOutputTensor(output, ret_types, ret_vals);
    }

    // Create a return operation to return all ONNX output tensors.
    builder_.create<mlir::ReturnOp>(UnknownLoc(), ret_vals);
    // Update main function signature to reflect types of newly imported
    // output tensors.
    funcType = builder_.getFunctionType(arg_types, ret_types);
    mainFunc.setType(funcType);
  }
};  // FrontendGenImpl class
}  // namespace
}  // namespace onnf

namespace onnf {

mlir::OwningModuleRef ImportFrontendModel(onnx::ModelProto model) {
  mlir::MLIRContext context;
  FrontendGenImpl myONNXGen(context);
  auto module = myONNXGen.ImportONNXModel(model);
  return module;
}

void ImportFrontendModelFile(std::string model_fname,
                             mlir::MLIRContext &context,
                             mlir::OwningModuleRef &module) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  assert(parse_success && "Onnx Model Parsing Failed.");

  FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model);
}
}  // namespace onnf
