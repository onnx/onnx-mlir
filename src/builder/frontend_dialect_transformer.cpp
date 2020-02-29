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

#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <tuple>

// Using backported variant.
// bstd = backported standard library.
#include <mpark/variant.hpp>
namespace bstd = mpark;

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

enum DataType {
  UNDEFINED = 0,  // undefined
  DOUBLE    = 1,  // double
  FLOAT     = 2,  // float
  UINT8     = 3,  // uint8_t
  INT8      = 4,  // int8_t
  UINT16    = 5,  // uint16_t
  INT16     = 6,  // int16_t
  INT32     = 7,  // int32_t
  INT64     = 8,  // int64_t
  STRING    = 9,  // string
  BOOL      = 10, // bool
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

struct InitializedTensor {
  InitializedTensor(std::vector<char> data, std::vector<int64_t> dims,
      DataType type) : _data(data), _dims(dims), _type(type) {}

  std::vector<char> getData() {
    return _data;
  }

  std::vector<int64_t> getDimensions() {
    return _dims;
  }

  DataType getType() {
    return _type;
  }

private:
  // Data which represents the value of the tensor.
  std::vector<char> _data;

  // Tensor dimension.
  std::vector<int64_t> _dims;

  // Data type.
  DataType _type;
};

struct InitializedTensorMapping {
  // Get initialized tensor.
  InitializedTensor GetInitializedTensor(const std::string &name) {
    assert(nameToInitializedTensor.find(legalize_name(name)) !=
               nameToInitializedTensor.end() &&
           "Tensor not found");
    return nameToInitializedTensor.at(legalize_name(name));
  }

  // Add new entry.
  void AddMapping(const std::string &name, InitializedTensor tensor) {
    assert(nameToInitializedTensor.count(legalize_name(name)) == 0 &&
           "Tensor already exists.");
    nameToInitializedTensor.emplace(legalize_name(name), tensor);
  }

  // Check if input is initialized. Not all inputs are, some of the inputs
  // require input from the user and are not stored inside the ONNX model
  // itself.
  bool ContainKey(std::string name) {
    return nameToInitializedTensor.count(name) != 0;
  }

private:
  // Mapping from ONNX tensor name to InitializedTensor.
  std::map<std::string, InitializedTensor> nameToInitializedTensor;
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
  mlir::Value none_;
  // mapping between string name and symbol
  OnnxOnnfSymbolMapping frontend_symbols_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

  // Convert type to MLIR type.
  // A complete list of types can be found in:
  // <onnf-build-folder>/third_party/onnx/onnx/onnx.pb.h
  mlir::Type convertONNXTypeToMLIRType(onnx::TensorProto_DataType onnxType) {
    switch (onnxType) {
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
  mlir::Type ImportInputTensorType(const onnx::ValueInfoProto &input) {
    std::vector<int64_t> dims;
    auto shape_proto = input.type().tensor_type().shape();
    auto input_tensor_legalized_name = legalize_name(input.name());
    for (int i = 0; i < shape_proto.dim_size(); i++) {
      if (shape_proto.dim()[i].dim_value()) {
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        assert(dim_numeric_size != 0 &&
               "Parsed an input tensor with a dimension size of zero");
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

    auto elementOnnxType =
        (onnx::TensorProto_DataType)input.type().tensor_type().elem_type();
    mlir::Type elementType = convertONNXTypeToMLIRType(elementOnnxType);
    llvm::ArrayRef<int64_t> tensor_dims(dims.data(), dims.size());
    // arg_types.emplace_back(
    //     mlir::RankedTensorType::get(tensor_dims, elementType));
    return mlir::RankedTensorType::get(tensor_dims, elementType);
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
    assert(!frontend_symbols_.ContainKey(input_tensor_legalized_name) &&
           "Found duplicate legalized input tensor names.");
    frontend_symbols_.AddMapping(input_tensor_legalized_name, symbol);
  }

  typedef bstd::variant<int64_t, std::vector<int64_t>, float,
                        std::vector<float>, std::string,
                        std::vector<std::string>>
      AttrValueType;

  struct ONNXAttrVisitor {
    ONNXAttrVisitor(std::string name, mlir::OpBuilder &builder)
        : _builder(builder), _name(std::move(name)) {}

    // Op builder.
    mlir::OpBuilder &_builder;

    // Name of the attribute being inspected.
    std::string _name;

    mlir::NamedAttribute operator()(int64_t const &r) {
      auto val = _builder.getI64IntegerAttr(r);
      return _builder.getNamedAttr(_name, val);
    }

    mlir::NamedAttribute operator()(std::vector<int64_t> const &ints) {
      auto val = _builder.getI64ArrayAttr(ints);
      return _builder.getNamedAttr(_name, val);
    }

    mlir::NamedAttribute operator()(float const &r) {
      auto val = _builder.getF32FloatAttr(r);
      return _builder.getNamedAttr(_name, val);
    }

    mlir::NamedAttribute operator()(std::vector<float> const &floats) {
      auto val = _builder.getF32ArrayAttr(floats);
      return _builder.getNamedAttr(_name, val);
    }

    mlir::NamedAttribute operator()(std::string const &s) {
      auto val = _builder.getStringAttr(s);
      return _builder.getNamedAttr(_name, val);
    }

    mlir::NamedAttribute operator()(std::vector<std::string> const &r) {
      assert(false && "type of attribute value is not implemented");
      auto val = _builder.getI32IntegerAttr(1);
      return _builder.getNamedAttr(_name, val);
    };
  };

  mlir::NamedAttribute convertNameValuePairToNamedAttribute(
      std::pair<std::string, AttrValueType> nameAndVal) {
    auto visitor = ONNXAttrVisitor(nameAndVal.first, builder_);
    return mpark::visit(visitor, nameAndVal.second);
  }

  static std::pair<std::string, AttrValueType>
  convertAttributeProtoToNameValuePair(onnx::AttributeProto &attr) {
    AttrValueType val;
    switch (attr.type()) {
    case onnx::AttributeProto::FLOAT:
      return std::make_pair(attr.name(), AttrValueType(attr.f()));
    case onnx::AttributeProto::INT:
      return std::make_pair(attr.name(), AttrValueType(attr.i()));
    case onnx::AttributeProto::STRING:
      return std::make_pair(attr.name(), AttrValueType(attr.s()));
    case onnx::AttributeProto::FLOATS:
      val = AttrValueType(
          std::vector<float>(attr.floats().begin(), attr.floats().end()));
      return std::make_pair(attr.name(), val);
    case onnx::AttributeProto::INTS:
      val = AttrValueType(
          std::vector<int64_t>(attr.ints().begin(), attr.ints().end()));
      return std::make_pair(attr.name(), val);
    default:
      assert(false && "datatype for attribute is not implemented");
      break;
    }
  }

  std::vector<mlir::NamedAttribute>
  ImportNodeAttributes(const onnx::NodeProto &node) {
    std::vector<mlir::NamedAttribute> attributes;
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      auto nameValPair = convertAttributeProtoToNameValuePair(attr);
      attributes.push_back(convertNameValuePairToNamedAttribute(nameValPair));
    }
    return attributes;
  }

  void ImportNodeGeneric(const onnx::NodeProto &node) {
    std::vector<mlir::Value> inputs;
    for (const auto &item : node.input()) {
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

  template <typename T>
  void buildOperation(const onnx::NodeProto &node, int expectedNumOperands = -1,
                      int expectedNumResults = -1) {
    bool variadicIn = expectedNumOperands == -1;
    bool variadicOut = expectedNumResults == -1;
    std::vector<mlir::Value> inputs;
    for (const auto &item : node.input()) {
      if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    if (!variadicIn)
      for (auto i = inputs.size(); i < expectedNumOperands; i++)
        inputs.emplace_back(none_);

    std::vector<mlir::Type> outputTypes;
    for (auto item : node.output()) {
      outputTypes.push_back(
          mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }

    auto attributes = ImportNodeAttributes(node);

    // TODO: Handle optional inputs.
    auto op = builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
    for (int i = 0; i < node.output().size(); i++) {
      frontend_symbols_.AddMapping(legalize_name(node.output()[i]),
                                   *(op.getODSResults(i).begin()));
    }
  }

  /*!
   * Special handle for Conv operations.
   * c++ does not allow template specialization inside a class scope
   * a specialized function is used
   */
  void ImportNodeConv(onnx::NodeProto node, int nIn, int nOut) {
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
    int nOps = node.input().size();

    if (nOps == 2)
      buildOperation<mlir::ONNXConvNoBiasOp>(node, nOps, nOut);
    else
      buildOperation<mlir::ONNXConvOp>(node, nOps, nOut);
  }

  /*!
   * Special handle for MaxPool operations.
   */
  void ImportNodeMaxPool(onnx::NodeProto node, int nIn, int nOut) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      buildOperation<mlir::ONNXMaxPoolSingleOutOp>(node, nIn, nOuts);
    } else {
      buildOperation<mlir::ONNXMaxPoolOp>(node, nIn, nOuts);
    }
  }

  /*!
   * Special handle for BatchNormalization operations.
   */
  void ImportNodeBatchNormalization(onnx::NodeProto node, int nIn, int nOut) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      // Test mode with one output.
      buildOperation<mlir::ONNXBatchNormalizationTestModeOp>(node, nIn, nOuts);
    } else {
      // Training mode with four trailing optional outputs. Not handled yet.
      buildOperation<mlir::ONNXBatchNormalizationOp>(node, nIn, nOuts);
    }
  }

  /*!
   * Special handle for Pad operations.
   */
  void ImportNodePad(onnx::NodeProto node, int nIn, int nOut) {
    int nOps = node.input().size();
    if (nOps == 2) {
      buildOperation<mlir::ONNXPadConstantValueOp>(node, 2, nOut);
    } else {
      buildOperation<mlir::ONNXPadOp>(node, nIn, nOut);
    }
  }

  void ImportNode(const onnx::NodeProto &node) {
    llvm::StringRef opName = node.op_type();

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
    assert(frontend_symbols_.ContainKey(output_tensor_legalized_name) &&
           "Output tensor not found");

    auto tensor_val =
        frontend_symbols_.GetTensorByOnnxName(output_tensor_legalized_name);
    ret_types.emplace_back(tensor_val.getType());
    ret_vals.push_back(tensor_val);
  }

  // template <typename T> TensorPtr ImportInitializer(
  //     OnnxDlcTensorMapping& tensor_map, onnx::TensorProto initializer) {
  //   // Convert dimension to dlc format.
  //   std::vector<Dim> dims;
  //   for (int i = 0; i < initializer.dims().size(); i++) {
  //     dims.push_back(Dim(initializer.dims()[i]));
  //   }

  //   std::vector<char> initializer_data_byte_array;
  //   auto name = initializer.name();

  //   if (initializer.raw_data().size()) {  // copy & take care of endianness
  //     std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
  //         back_inserter(initializer_data_byte_array));
  //     dlc_little_to_native<T>((T*)&initializer_data_byte_array[0],
  //         initializer_data_byte_array.size());
  //   } else {  // copy, no need to take care of endianness
  //     auto data = cpp_type_to_onnx_data_field_trait<T>::data(initializer);
  //     initializer_data_byte_array.reserve(data.size() * sizeof(T));
  //     initializer_data_byte_array = std::vector<char>(data.size() * sizeof(T));
  //     std::copy(data.begin(), data.end(),
  //         reinterpret_cast<T*>(&initializer_data_byte_array[0]));
  //   }

  //   auto tensor = Tensor::Create(
  //       legalize_name(name), cpp_type_to_dlc_type_trait<T>::dlc_type, dims);
  //   tensor->initial_data(initializer_data_byte_array);

  //   tensor_map.AddMapping(legalize_name(name), tensor);
  //   return tensor;
  // }

  template <typename T> InitializedTensor ReadInitializerValueFromModel(
      onnx::TensorProto initializer, DataType type) {
    // Convert dimension to dlc format.
    std::vector<int64_t> dims;
    for (int i = 0; i < initializer.dims().size(); i++) {
      dims.push_back((int64_t)initializer.dims()[i]);
    }

    std::vector<char> dataByteArrayInitializer;

    if (initializer.raw_data().size()) {
      // copy & take care of endianness
      std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
          back_inserter(dataByteArrayInitializer));
    } else {  // copy, no need to take care of endianness
      auto data = TransformValueToONNXData<T>::data(initializer);
      dataByteArrayInitializer.reserve(data.size() * sizeof(T));
      dataByteArrayInitializer = std::vector<char>(data.size() * sizeof(T));
      std::copy(data.begin(), data.end(),
          reinterpret_cast<T*>(&dataByteArrayInitializer[0]));
      printf(" data.size() = %d\n", data.size());
    }

    return InitializedTensor(dataByteArrayInitializer, dims, type);
  }

  template <typename T> T* CreateArrayAttribute(
      onnx::TensorProto initializer, int *size) {
    printf("CreateArrayAttribute \n");
    if (initializer.raw_data().size()) {
      printf("This tensor is stored in RAW format!\n");
      // copy & take care of endianness
      std::vector<char> byteInitializer;
      std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
          back_inserter(byteInitializer));
      *size = initializer.raw_data().size() / sizeof(T);
      return reinterpret_cast<T*>(&byteInitializer[0]);
    }

    // copy, no need to take care of endianness
    auto data = TransformValueToONNXData<T>::data(initializer);
    printf(" data.size() = %d\n", data.size());
    *size = data.size();
    printf("CreateArrayAttribute size = %d\n", *size);
    return &data[0];
  }

  void ImportGraph(const onnx::GraphProto &graph,
                   const std::string &name = "main_graph") {
    InitializedTensorMapping initializedTensors;
    printf("Print constant input data types:\n");
    for (auto initializer : graph.initializer()) {
      printf("\nArg: %s\n", initializer.name().c_str());
      auto name = initializer.name();
      switch (initializer.data_type()) {
        case (onnx::TensorProto::FLOAT): {
          printf("onnx::TensorProto::FLOAT case!\n");
          initializedTensors.AddMapping(legalize_name(name),
              ReadInitializerValueFromModel<float>(initializer, FLOAT));
          break;
        }
        case (onnx::TensorProto::INT32): {
          printf("onnx::TensorProto::INT32 case!\n");
          initializedTensors.AddMapping(legalize_name(name),
              ReadInitializerValueFromModel<int32_t>(initializer, INT32));
          break;
        }
        case (onnx::TensorProto::INT64): {
          printf("onnx::TensorProto::INT64 case!\n");
          initializedTensors.AddMapping(legalize_name(name),
              ReadInitializerValueFromModel<int64_t>(initializer, INT64));
          break;
        }
        default:
          printf("Default case!\n");
      }
    }

    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<mlir::Type, 4> arg_types;

    printf("Import graph, num inputs = %d\n", graph.input().size());
    // Import the input tensor types that are not constant.
    for (const auto &input : graph.input())
      if (!initializedTensors.ContainKey(input.name()))
        arg_types.emplace_back(ImportInputTensorType(input));

    // Create the main function.
    auto funcType = builder_.getFunctionType(arg_types, {});
    auto mainFunc =
        mlir::FuncOp::create(UnknownLoc(), name, funcType, /* attrs = */ {});

    // Emit the entry point operation which specifies the number of user
    // inputs and outputs.
    auto entryPoint = mlir::ONNXEntryPointOp::create(
        UnknownLoc(), mainFunc,
        /*numInputs=*/graph.input().size() - graph.initializer().size(),
        /*numOutputs=*/graph.output().size());

    // Get the entru block inside the main function and set the insertion point
    // to it.
    auto &entryBlock = *mainFunc.addEntryBlock();
    builder_.setInsertionPointToStart(&entryBlock);

    module_.push_back(mainFunc);
    module_.push_back(entryPoint);

    // Map graph inputs to entry block arguments.
    for (int i = 0; i < graph.input().size(); ++i)
      if (!initializedTensors.ContainKey(graph.input()[i].name()))
        ImportInputTensorSymbol(
            graph.input()[i], entryBlock.getArguments()[i]);

    // Create a NoneTyped constant to be used for optional operation inputs
    // which are not used.
    none_ = builder_.create<mlir::ConstantOp>(UnknownLoc(),
        builder_.getUnitAttr());

    // Emit constant arguments (initialized arguments) as ConstantOps.
    // For every argument which can be initialized with a tensor of constant
    // values we emit a ConstantOp with the constant value as an attribute.
    // This will allow the propagation of shapes given as an argument to
    // operations such as Reshape and enable other optimizations such as
    // constant folding.
    for (auto initializer : graph.initializer()) {
      // Emit ConstantOp and record the mapping between the input and
      // the constant value.
      std::string inputName = initializer.name();
      printf("\nARG --->  %s\n", inputName.c_str());

      mlir::ArrayAttr constantArrayAttribute;
      int length;
      printf("Dims: ");
      for (int j = 0; j < initializer.dims().size(); j++) {
        printf(" %d ", initializer.dims()[j]);
      }
      printf("\n");
      switch (initializer.data_type()) {
        case (onnx::TensorProto::FLOAT): {
          float *typeArray =
              CreateArrayAttribute<float>(initializer, &length);
          std::vector<float> arrayAttrInitializer(typeArray, typeArray + length);
          llvm::ArrayRef<float> array(typeArray, length);
          constantArrayAttribute = builder_.getF32ArrayAttr(array);
          break;
        }
        case (onnx::TensorProto::INT32): {
          int32_t *typeArray =
              CreateArrayAttribute<int32_t>(initializer, &length);
          std::vector<int32_t> arrayAttrInitializer(typeArray, typeArray + length);
          llvm::ArrayRef<int32_t> array(typeArray, length);
          constantArrayAttribute = builder_.getI32ArrayAttr(array);
          break;
        }
        case (onnx::TensorProto::INT64): {
          int64_t *typeArray =
              CreateArrayAttribute<int64_t>(initializer, &length);
          std::vector<int64_t> arrayAttrInitializer(typeArray, typeArray + length);
          llvm::ArrayRef<int64_t> array(typeArray, length);
          constantArrayAttribute = builder_.getI64ArrayAttr(array);
          break;
        }
      }

      onnx::ValueInfoProto inputTensor;
      for (const auto &input : graph.input()) {
        if (input.name() == initializer.name()) {
          inputTensor = input;
          break;
        }
      }
      auto initializedArg = builder_.create<mlir::ConstantOp>(
          UnknownLoc(), ImportInputTensorType(inputTensor),
          constantArrayAttribute);

      frontend_symbols_.AddMapping(
          legalize_name(inputName), initializedArg);
    }

    // Import nodes in the graph.
    for (const auto &item : graph.node()) {
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
}; // FrontendGenImpl class
} // namespace
} // namespace onnf

namespace onnf {

mlir::OwningModuleRef ImportFrontendModel(onnx::ModelProto model) {
  printf(" --> ImportFrontendModel\n");
  mlir::MLIRContext context;
  FrontendGenImpl myONNXGen(context);
  auto module = myONNXGen.ImportONNXModel(model);
  return module;
}

void ImportFrontendModelFile(std::string model_fname,
                             mlir::MLIRContext &context,
                             mlir::OwningModuleRef &module) {
  printf(" --> ImportFrontendModelFile\n");
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  assert(parse_success && "Onnx Model Parsing Failed.");

  FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model);
}
} // namespace onnf
