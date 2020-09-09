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

#include <type_traits>
// Using backported variant.
// bstd = backported standard library.
#include <mpark/variant.hpp>
namespace bstd = mpark;

#include "src/Interface/ResultTypeInferenceOpInterface.hpp"

#include "FrontendDialectTransformer.hpp"

namespace onnx_mlir {
namespace detail {

/*!
 *  The list of tensors initialized by the ONNX model.
 */
InitializedTensorMapping initializedTensors;

class FrontendGenImpl {
public:
  FrontendGenImpl(mlir::MLIRContext &context)
      : context_(context), builder_(&context) {
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    InitHandlerMap();
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

  typedef void (onnx_mlir::detail::FrontendGenImpl::*ImportHandlerType)(
      const onnx::NodeProto &);

  std::map<std::string, ImportHandlerType> import_handler_map_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

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
    mlir::Type elementType =
        convertONNXTypeToMLIRType(builder_, elementOnnxType);
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

  mlir::NamedAttribute convertOnnxAttributeProtoToMlirNamedAttribute(
      onnx::AttributeProto attr) {
    mlir::Attribute mlirAttr;
    switch (attr.type()) {
    case onnx::AttributeProto::FLOAT:
      mlirAttr = builder_.getF32FloatAttr(attr.f());
      break;
    case onnx::AttributeProto::INT:
      mlirAttr = builder_.getI64IntegerAttr(attr.i());
      break;
    case onnx::AttributeProto::STRING:
      mlirAttr = builder_.getStringAttr(attr.s());
      break;
    case onnx::AttributeProto::FLOATS:
      mlirAttr = builder_.getF32ArrayAttr(
          llvm::makeArrayRef(attr.floats().begin(), attr.floats().end()));
      break;
    case onnx::AttributeProto::INTS:
      mlirAttr = builder_.getI64ArrayAttr(
          llvm::makeArrayRef(attr.ints().begin(), attr.ints().end()));
      break;
    case onnx::AttributeProto::TENSOR:
      mlirAttr = onnxTensorProtoToDenseElmAttr(builder_, attr.t());
      break;
    case onnx::AttributeProto::STRINGS: {
      llvm::SmallVector<mlir::StringRef, 4> vectorStringRef;
      for (const auto &item : attr.strings()) {
        vectorStringRef.push_back(llvm::StringRef(item));
      }
      mlirAttr = builder_.getStrArrayAttr(llvm::makeArrayRef(vectorStringRef));
    } break;
    default:
      llvm_unreachable("datatype for attribute is not implemented");
      break;
    }
    return builder_.getNamedAttr(attr.name(), mlirAttr);
  }

  std::vector<mlir::NamedAttribute> ImportNodeAttributes(
      const onnx::NodeProto &node) {
    std::vector<mlir::NamedAttribute> attributes;
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      attributes.push_back(convertOnnxAttributeProtoToMlirNamedAttribute(attr));
    }

    // If the node has a name, then import it.
    if (node.has_name()) {
      attributes.push_back(builder_.getNamedAttr(
          "onnx_node_name", builder_.getStringAttr(node.name())));
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
    result.addAttributes(ImportNodeAttributes(node));
    auto op = builder_.createOperation(result);
    for (int i = 0; i < node.output().size(); i++) {
      auto r = op->getResult(i);
      frontend_symbols_.AddMapping(legalize_name(node.output()[i]), r);
    }
  }

#define MAX_TYPE 20
  // itblgen_types = ('I1', 'I8', 'I16', 'I32', 'I64', 'BF16', 'F16', 'F32',
  // 'F64', 'Complex<F32>', 'Complex<F64>' )
  mlir::Type buildTypeFromIndex(int index) {
    switch (index) {
    case 0:
      return builder_.getI1Type();
    case 1:
      return builder_.getIntegerType(8);
    case 2:
      return builder_.getIntegerType(16);
    case 3:
      return builder_.getIntegerType(32);
    case 4:
      return builder_.getIntegerType(64);
    case 5:
      return builder_.getBF16Type();
    case 6:
      return builder_.getF16Type();
    case 7:
      return builder_.getF32Type();
    case 8:
      return builder_.getF64Type();
    case 9: {
      std::vector<mlir::Type> typeTuple(2);
      typeTuple.push_back(builder_.getF32Type());
      typeTuple.push_back(builder_.getF32Type());
      return builder_.getTupleType(llvm::ArrayRef<mlir::Type>(typeTuple));
    }
    case 10: {
      std::vector<mlir::Type> typeTuple(2);
      typeTuple.push_back(builder_.getF64Type());
      typeTuple.push_back(builder_.getF64Type());
      return builder_.getTupleType(llvm::ArrayRef<mlir::Type>(typeTuple));
    }
    default:
      assert(false && "Unsupported type index encountered.");
      return nullptr;
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
      for (auto i = inputs.size(); i < expectedNumOperands; i++) {
        if (!none_)
          none_ = builder_.create<mlir::ConstantOp>(
              UnknownLoc(), builder_.getUnitAttr());
        inputs.emplace_back(none_);
      }

    std::vector<mlir::Type> outputTypes;

    // Use the type map to determine the data type of output.
    std::vector<int> outputMap = T::getTypeMap();
    for (auto i = 0; i < node.output().size(); i++) {
      // Optional outputs using empty string.
      if (node.output()[i].empty()) {
        outputTypes.emplace_back(builder_.getNoneType());
      } else {
        auto j = i;
        // Variadic output is a single ODS result.
        if (variadicOut)
          j = 0;
        if (j < outputMap.size() && outputMap[j] >= MAX_TYPE) {
          // Mapping gives a connection with an input.
          mlir::Type inputType = inputs[outputMap[j] - MAX_TYPE].getType();
          if (inputType.isa<mlir::TensorType>()) {
            auto elementType =
                inputType.cast<mlir::TensorType>().getElementType();
            auto outType = mlir::UnrankedTensorType::get(elementType);
            outputTypes.emplace_back(outType);
          } else {
            outputTypes.push_back(inputType);
          }
        } else if (j < outputMap.size() && outputMap[j] != -1) {
          // Mapping gives a direct type.
          auto elementType = buildTypeFromIndex(outputMap[j]);
          auto outType = mlir::UnrankedTensorType::get(elementType);
          outputTypes.emplace_back(outType);
        } else {
          outputTypes.emplace_back(builder_.getNoneType());
        }
      }
    }
    // Trailing optional outputs.
    if (!variadicOut)
      for (int i = node.output().size(); i < expectedNumResults; ++i)
        outputTypes.emplace_back(builder_.getNoneType());

    auto attributes = ImportNodeAttributes(node);

    // TODO: Handle optional inputs.
    auto op = builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);

    // Type inference for results.
    if (auto opWithTypeInference =
            mlir::dyn_cast<mlir::ResultTypeInferenceOpInterface>(
                op.getOperation())) {
      auto outTypes = opWithTypeInference.resultTypeInference();
      for (int i = 0; i < node.output().size(); i++) {
        if (variadicOut)
          (*(op.getODSResults(0).begin() + i)).setType(outTypes[i]);
        else
          (*op.getODSResults(i).begin()).setType(outTypes[i]);
      }
    }

    for (int i = 0; i < node.output().size(); i++) {
      if (variadicOut)
        frontend_symbols_.AddMapping(legalize_name(node.output()[i]),
            *(op.getODSResults(0).begin() + i));
      else
        frontend_symbols_.AddMapping(
            legalize_name(node.output()[i]), *(op.getODSResults(i).begin()));
    }
  }

  template <typename T>
  void buildOperation(const onnx::NodeProto &node) {
    std::vector<mlir::Value> inputs;
    int expectedNumOperands = T::getNumberOfOperands();
    int expectedNumResults = T::getNumberOfResults();
    for (const auto &item : node.input())
      if (item.empty()) {
        // Optional inputs using empty string will be imported as NoneType.
        if (!none_)
          none_ = builder_.create<mlir::ConstantOp>(
              UnknownLoc(), builder_.getUnitAttr());
        inputs.emplace_back(none_);
      } else if (initializedTensors.ContainKey(legalize_name(item))) {
        inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, legalize_name(item)));
      } else if (frontend_symbols_.ContainKey(legalize_name(item))) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }

    buildOutputAndOperation<T>(
        node, inputs, expectedNumOperands, expectedNumResults);
  }

  /*!
   * Special handle for MaxPool operations.
   */
  void ImportNodeMaxPool(const onnx::NodeProto &node) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      buildOperation<mlir::ONNXMaxPoolSingleOutOp>(node);
    } else {
      buildOperation<mlir::ONNXMaxPoolOp>(node);
    }
  }

  /*!
   * Special handle for BatchNormalization operations.
   */
  void ImportNodeBatchNormalization(const onnx::NodeProto &node) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      // Test mode with one output.
      buildOperation<mlir::ONNXBatchNormalizationTestModeOp>(node);
    } else {
      // Training mode with four trailing optional outputs. Not handled yet.
      buildOperation<mlir::ONNXBatchNormalizationOp>(node);
    }
  }

  /*!
   * Special handle for Pad operations.
   */
  void ImportNodePad(const onnx::NodeProto &node) {

    int nOps = node.input().size();
    if (nOps == 2) {
      llvm::SmallVector<int64_t, 2> dims;
      dims.push_back(1);
      llvm::SmallVector<float, 2> values;
      values.push_back(0.);
      auto elementType = builder_.getF32Type();
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = mlir::RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));

      // Use the special builder defined in ONNXOp.td.inc.
      auto constantOp = builder_.create<mlir::ONNXConstantOp>(
          UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = *(constantOp.getODSResults(0).begin());
      std::vector<mlir::Value> inputs;
      for (const auto &item : node.input())
        if (initializedTensors.ContainKey(legalize_name(item))) {
          inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
              UnknownLoc(), builder_, legalize_name(item)));
        } else if (frontend_symbols_.ContainKey(legalize_name(item))) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      inputs.push_back(constantResult);

      int nIn = mlir::ONNXPadOp::getNumberOfOperands();
      int nOut = mlir::ONNXPadOp::getNumberOfResults();
      buildOutputAndOperation<mlir::ONNXPadOp>(node, inputs, nIn, nOut);
    } else {
      buildOperation<mlir::ONNXPadOp>(node);
    }
  }

  void ImportNodeSlice(const onnx::NodeProto &node) {
    std::array<mlir::Value, 5> inVals = {
        nullptr,
    };

    for (const auto &item : llvm::enumerate(node.input())) {
      if (initializedTensors.ContainKey(legalize_name(item.value()))) {
        inVals[item.index()] = initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, legalize_name(item.value()));
      } else if (frontend_symbols_.ContainKey(legalize_name(item.value()))) {
        inVals[item.index()] =
            frontend_symbols_.GetTensorByOnnxName(item.value());
      } else {
        assert(false && "Unknown input");
      }
    }

    // Data input is imported but starts, ends, axes, and steps may come from
    // attributes, and need to be created as constant ops.
    const auto elementType = builder_.getIntegerType(64);
    const auto tensorType = mlir::RankedTensorType::get({1}, elementType);
    const auto attributes = ImportNodeAttributes(node);
    for (auto attr : attributes) {
      if (auto arrayAttr = attr.second.dyn_cast<mlir::ArrayAttr>()) {
        auto constantDenseAttribute =
            mlir::DenseElementsAttr::get(tensorType, arrayAttr.getValue());
        auto constantOp = builder_.create<mlir::ONNXConstantOp>(
            UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
        mlir::Value constantValue = constantOp.output();

        // Map from ONNX attributes to indices, which are
        // matched with ONNXSliceOp::build ordering.
        auto inputIdx = llvm::StringSwitch<int>(attr.first)
                            .Case("starts", 1)
                            .Case("ends", 2)
                            .Case("axes", 3)
                            .Case("steps", 4)
                            .Default(-1);
        if (inputIdx < 0)
          continue;
        assert(inVals[inputIdx] == nullptr &&
               "This input has already been filled in");
        inVals[inputIdx] = constantValue;
      }
    }

    assert(inVals[1] != nullptr && "Slice requires a starts attribute");
    assert(inVals[2] != nullptr && "Slice requires an ends attribute");
    const auto startsType = inVals[1].getType().dyn_cast<RankedTensorType>();
    assert(startsType != nullptr && "starts type is not a RankedTensorType");
    auto startsDim = startsType.getShape()[0];

    // If axes is not specified, default to [0, ..., ndim-1]
    if (inVals[3] == nullptr) {
      SmallVector<int64_t, 1> vals = {};
      for (size_t s = 0; s < startsDim; ++s)
        vals.emplace_back(s);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(vals));
      auto constantOp = builder_.create<mlir::ONNXConstantOp>(
          UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = constantOp.output();
      inVals[3] = constantResult;
    }

    // If steps is not specified, default to [1, ..., 1]
    if (inVals[4] == nullptr) {
      SmallVector<int64_t, 1> vals(startsDim, 1);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(vals));
      auto constantOp = builder_.create<mlir::ONNXConstantOp>(
          UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = constantOp.output();
      inVals[4] = constantResult;
    }

    int nIn = mlir::ONNXSliceOp::getNumberOfOperands();
    int nOut = mlir::ONNXSliceOp::getNumberOfResults();
    const auto in = std::vector<mlir::Value>(inVals.begin(), inVals.end());
    buildOutputAndOperation<mlir::ONNXSliceOp>(node, in, nIn, nOut);
  }

  void ImportNode(const onnx::NodeProto &node) {
    llvm::StringRef opName = node.op_type();

    // the following code is generated by gen_doc.py
    // refer to Dialect/ONNX/ONNXOps.td for details
    // when the input or output of then op does not match the specification,
    // the generic operator is used
    // one known reeason is the optional input

    auto found = import_handler_map_.find(opName.str());
    assert(found != import_handler_map_.end() && "Could not find op importer");
    (this->*(found->second))(node);
  }

  void InitHandlerMap() {
#include "src/Builder/OpBuildTable.inc"
  }

  /*!
   * Import output tensor, by doing the following:
   * - Add the t/yp this output tensor to a list of tensor
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

    // Get a list of function attributes - including names of inputs and outputs
    llvm::SmallVector<mlir::NamedAttribute, 4> funcAttrs;
    llvm::SmallVector<llvm::StringRef, 4> inputNames;
    llvm::SmallVector<llvm::StringRef, 4> outputNames;

    // Import the input tensor types that are not constant and not initialized.
    int numInputs = 0;
    for (const auto &input : graph.input()) {
      if (!initializedTensors.ContainKey(legalize_name(input.name()))) {
        inputNames.push_back(input.name());
        arg_types.emplace_back(ImportInputTensorType(input));
        // numInputs is the number of graph inputs not contained within the
        // initializer
        ++numInputs;
      }
    }

    for (const auto &output : graph.output()) {
      outputNames.push_back(output.name());
    }

    funcAttrs.emplace_back(builder_.getNamedAttr(
        "input_names", builder_.getStrArrayAttr(inputNames)));
    funcAttrs.emplace_back(builder_.getNamedAttr(
        "output_names", builder_.getStrArrayAttr(outputNames)));

    // Create the main function.
    auto funcType = builder_.getFunctionType(arg_types, {});
    auto mainFunc = mlir::FuncOp::create(UnknownLoc(), name, funcType,
        /* attrs = */ llvm::makeArrayRef(funcAttrs));

    // Emit the entry point operation which specifies the number of user
    // inputs and outputs.
    auto entryPoint = mlir::ONNXEntryPointOp::create(UnknownLoc(), mainFunc,
        /*numInputs=*/numInputs,
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
} // namespace detail
} // namespace onnx_mlir

namespace onnx_mlir {

void ImportFrontendModelFile(std::string model_fname,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  assert(parse_success && "Onnx Model Parsing Failed.");

  detail::FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model);
}
} // namespace onnx_mlir
