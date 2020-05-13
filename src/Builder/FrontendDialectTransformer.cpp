//===--------- FrontendDialectTransformer.cpp - MLIR Operations -----------===//
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

// Using backported variant.
// bstd = backported standard library.
#include <mpark/variant.hpp>
namespace bstd = mpark;

#include "FrontendDialectTransformer.hpp"

namespace onnx_mlir {
namespace {

/*!
 *  The list of tensors initialized by the ONNX model.
 */
InitializedTensorMapping initializedTensors;

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
  OnnxMlirSymbolMapping frontend_symbols_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

  // Convert type to MLIR type.
  // A complete list of types can be found in:
  // <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
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
    default:
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
    return mlir::RankedTensorType::get(tensor_dims, elementType);
  }

  /*!
   * Import a input tensor symbol by recording a new entry in frontend_symbols_
   * recording the mapping between legalized onnx tensor name and mlir::Value
   * for further lookup in computation node importing.
   * @param input onnx input tensor ValueInfoProto.
   * @param symbol mlir input argument.
   */
  void ImportInputTensorSymbol(
      const onnx::ValueInfoProto &input, mlir::Value symbol) {
    auto input_tensor_legalized_name = legalize_name(input.name());
    assert(!frontend_symbols_.ContainKey(input_tensor_legalized_name) &&
           "Found duplicate legalized input tensor names.");
    frontend_symbols_.AddMapping(input_tensor_legalized_name, symbol);
  }

  typedef bstd::variant<int64_t, std::vector<int64_t>, float,
      std::vector<float>, std::string, std::vector<std::string>>
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
    llvm_unreachable("Failed to convert attribute proto to name/value pair");
  }

  std::vector<mlir::NamedAttribute> ImportNodeAttributes(
      const onnx::NodeProto &node) {
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
  void buildOutputAndOperation(const onnx::NodeProto &node,
      std::vector<mlir::Value> inputs, int expectedNumOperands,
      int expectedNumResults) {
    bool variadicIn = expectedNumOperands == -1;
    bool variadicOut = expectedNumResults == -1;

    // In ONNX, there are two ways to leave an optional input or output
    // unspecified: the first, available only for trailing inputs and outputs,
    // is to simply not provide that input; the second method is to use an empty
    // string in place of an input or output name.
    //
    // Here, we import optional inputs and outputs as NoneType.

    // Trailing optional inputs.
    if (!variadicIn)
      for (auto i = inputs.size(); i < expectedNumOperands; i++)
        inputs.emplace_back(none_);

    std::vector<mlir::Type> outputTypes;
    for (auto item : node.output()) {
      // Optional outputs using empty string.
      if (item.empty())
        outputTypes.emplace_back(builder_.getNoneType());
      else
        outputTypes.push_back(
            mlir::UnrankedTensorType::get(builder_.getF32Type()));
    }
    // Trailing optional outputs.
    if (!variadicOut)
      for (int i = node.output().size(); i < expectedNumResults; ++i)
        outputTypes.emplace_back(builder_.getNoneType());

    auto attributes = ImportNodeAttributes(node);

    // TODO: Handle optional inputs.
    auto op = builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
    for (int i = 0; i < node.output().size(); i++) {
      frontend_symbols_.AddMapping(
          legalize_name(node.output()[i]), *(op.getODSResults(i).begin()));
    }
  }

  template <typename T>
  void buildOperation(const onnx::NodeProto &node, int expectedNumOperands = -1,
      int expectedNumResults = -1) {
    std::vector<mlir::Value> inputs;
    for (const auto &item : node.input())
      if (initializedTensors.ContainKey(legalize_name(item))) {
        inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, legalize_name(item)));
      } else if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }

    buildOutputAndOperation<T>(
        node, inputs, expectedNumOperands, expectedNumResults);
  }

  void ImportNodeReshape(onnx::NodeProto node, int nIn, int nOut) {
    std::vector<mlir::Value> inputs;
    std::string item;
    for (int i = 0; i < node.input().size(); ++i) {
      item = node.input()[i];
      // For the second argument, check if there exists an initializer.
      if (initializedTensors.ContainKey(legalize_name(item))) {
        inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, legalize_name(item)));
      } else if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    buildOutputAndOperation<mlir::ONNXReshapeOp>(node, inputs, nIn, nOut);
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
    // refer to Dialect/ONNX/ONNXOps.td for details
    // when the input or output of then op does not match the specification,
    // the generic operator is used
    // one known reeason is the optional input

#include "src/Builder/OpBuildTable.inc"
#if INCLUDE_ONNX_ML == 1
#include "src/Builder/MLOpBuildTable.inc"
#endif

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

  void ImportGraph(
      const onnx::GraphProto &graph, const std::string &name = "main_graph") {
    // Maintain a mapping between the parameter and its initializer.
    for (auto initializer : graph.initializer()) {
      auto name = initializer.name();
      initializedTensors.AddMapping(legalize_name(name), initializer);
    }

    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<mlir::Type, 4> arg_types;

    // Import the input tensor types that are not constant and not initialized.
    for (const auto &input : graph.input())
      if (!initializedTensors.ContainKey(legalize_name(input.name())))
        arg_types.emplace_back(ImportInputTensorType(input));

    // Create the main function.
    auto funcType = builder_.getFunctionType(arg_types, {});
    auto mainFunc =
        mlir::FuncOp::create(UnknownLoc(), name, funcType, /* attrs = */ {});

    // Emit the entry point operation which specifies the number of user
    // inputs and outputs.
    auto entryPoint = mlir::ONNXEntryPointOp::create(UnknownLoc(), mainFunc,
        /*numInputs=*/graph.input().size() - graph.initializer().size(),
        /*numOutputs=*/graph.output().size());

    // Get the entru block inside the main function and set the insertion point
    // to it.
    auto &entryBlock = *mainFunc.addEntryBlock();
    builder_.setInsertionPointToStart(&entryBlock);

    module_.push_back(mainFunc);
    module_.push_back(entryPoint);

    // Map graph inputs to entry block arguments.
    // Counter of un-initialized tensors. This counter is used to index the
    // entry block arguments.
    int entryBlockArgIdx = 0;
    for (int i = 0; i < graph.input().size(); ++i) {
      if (!initializedTensors.ContainKey(
              legalize_name(graph.input()[i].name()))) {
        ImportInputTensorSymbol(
            graph.input()[i], entryBlock.getArguments()[entryBlockArgIdx]);
        entryBlockArgIdx++;
      }
    }

    // Create a NoneTyped constant to be used for optional operation inputs
    // which are not used.
    none_ =
        builder_.create<mlir::ConstantOp>(UnknownLoc(), builder_.getUnitAttr());

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
} // namespace onnx_mlir

namespace onnx_mlir {

void ImportFrontendModelFile(std::string model_fname,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  assert(parse_success && "Onnx Model Parsing Failed.");

  FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model);
}
} // namespace onnx_mlir
