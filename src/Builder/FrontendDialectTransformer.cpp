/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

#include "FrontendDialectTransformer.hpp"
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Support/SuppressWarnings.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
SUPPRESS_WARNINGS_POP

#include <fstream>
#include <iostream>
#include <map>
#include <type_traits>

#define DEBUG_TYPE "frontend_dialect_transformer"

/// We consider opset < 6 is old. Users will see a warning if their model
/// contains ops of old opset.
static constexpr int32_t MINIMUM_SUPPORTED_OPSET = 6;

using namespace mlir;

namespace onnx_mlir {
namespace detail {

using ValueSymbolMapping = SymbolMapping<Value>;
using SymbolToOnnxTypeMapping = SymbolMapping<onnx::TypeProto>;

class FrontendGenImpl {
public:
  explicit FrontendGenImpl(MLIRContext &context)
      : context_(context), builder_(&context),
        force_dim_dynamic_enabled_(false) {
    module_ = ModuleOp::create(UnknownLoc::get(&context));
    InitHandlerMap();
    if (const char *envInputString = std::getenv("IMPORTER_FORCE_DYNAMIC")) {
      force_dim_dynamic_enabled_ = true;
      std::stringstream envString;
      envString << envInputString;
      std::string dynamicInput;
      while (getline(envString, dynamicInput, '|')) {
        size_t pos = dynamicInput.find(':');
        std::string inputString = dynamicInput.substr(0, pos);
        std::string dimString = dynamicInput.substr(pos + 1);

        std::stringstream dimIndices(dimString);
        std::string dimIndex;
        std::vector<int> dims;
        while (getline(dimIndices, dimIndex, ',')) {
          dims.emplace_back(stoi(dimIndex));
        }
        // Default to the all dimensions if dims are not specified.
        if (dims.empty())
          dims.emplace_back(-1);
        forced_inputs_dims.insert(std::make_pair(stoi(inputString), dims));
      }
      // Default to the all inputs and dimensions.
      if (forced_inputs_dims.empty())
        forced_inputs_dims.insert(std::make_pair(-1, std::vector<int>(1, -1)));
    }
  }

  ModuleOp ImportONNXModel(
      const onnx::ModelProto &model, ImportOptions options) {
    options_ = options;
    SetOpSetImport(model); // Determines which opsets to use.
    SetCustomShapeInfo();  // Set custom shapes for the inputs if available.
    importGraph(model.graph());
    return module_;
  }

private:
  ImportOptions options_;
  MLIRContext &context_;
  ModuleOp module_;
  OpBuilder builder_;

  std::map<std::string, std::vector<int>> op_dialect_version_map_;

  /*!
   *  The list of tensors initialized by the ONNX model.
   */
  InitializedTensorMapping initializedTensors;

  // mapping between string name and symbol
  ValueSymbolMapping frontend_symbols_;

  // Flag to change the inputs of function to unknown dimension.
  // Temporarily added to use the test cases with static shape to test.
  // The values are set by enviroment variable IMPORTER_FORCE_DYNAMIC
  // The Backus–Naur Form (BNF) for IMPORTER_FORCE_DYNAMIC is as follows.
  //
  // <ImportForceDymanicExpr> :== `'` <expr> `'`
  //                   <expr> ::= <inputString> | <inputString> `|` <expr>
  //             <inputString ::= <inputIndex> `:` <dimString>
  //              <dimString> ::= <dimIndex> | <dimIndex> `,` <dimString>
  //             <inputIndex> ::= <index>
  //               <dimIndex> ::= <index>
  //                  <index> ::= -1 | <number>
  //                 <number> ::= <digit> | <digit><number>
  //                  <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
  //
  // Value `-1` semantically represents all inputs or all dimensions, and it
  // has the highest priority. E.g. `'0: -1, 0'` means all dimensions of the
  // first input will be changed. Input and dimension indices start from 0.
  //
  // Examples:
  // 1. IMPORTER_FORCE_DYNAMIC='-1:-1'
  //    - change all dimensions in all inputs to unknown dimensions.
  // 2. IMPORTER_FORCE_DYNAMIC='-1:0'
  //    - change the first dimension in all inputs to unknown dimensions.
  // 3. IMPORTER_FORCE_DYNAMIC='1:-1'
  //    - change all dimensions in the second input to unknown dimensions.
  // 4. IMPORTER_FORCE_DYNAMIC='1:0,1'
  //    - change the first and second dimensions in the second input to unknown
  //    dimensions.
  // 5. IMPORTER_FORCE_DYNAMIC='0:1|1:0,1'
  //    - change the second dimension in the first input to unknown dimensions,
  //    and
  //    - change the first and second dimensions in the second input to unknown
  //    dimensions,

  bool force_dim_dynamic_enabled_;
  // A map from an input index to a list of dim indices those are changed to
  // dynamic. Default value corresponds to IMPORTER_FORCE_DYNAMIC='-1:-1'
  std::map<int, std::vector<int>> forced_inputs_dims;

  // Custom shape information for the graph inputs.
  std::map<int64_t, std::vector<int64_t>> inputs_shape_information;
  void SetCustomShapeInfo() {
    // Use the custom shape for the inputs if avaiable.
    if (options_.shapeInformation.empty()) {
      return;
    }

    std::stringstream shapeInfoString(options_.shapeInformation);
    std::string shapeString;
    while (getline(shapeInfoString, shapeString, ',')) {
      size_t pos = shapeString.find(':');
      std::string inputString = shapeString.substr(0, pos);
      std::string dimString = shapeString.substr(pos + 1);

      int64_t inputID = std::stoi(inputString);
      assert(inputID >= 0 && "input_id must be >= 0");

      std::stringstream dimSizes(dimString);
      std::string dimStr;
      std::vector<int64_t> dims;
      while (getline(dimSizes, dimStr, 'x')) {
        int64_t dimSize = std::stoi(dimStr);
        assert((dimSize == -1 || dimSize > 0) && "dim must be -1 or > 0");
        dims.emplace_back(dimSize);
      }
      inputs_shape_information.insert(std::make_pair(inputID, dims));
    }
  }

  using ImportHandlerType = void (onnx_mlir::detail::FrontendGenImpl::*)(
      const onnx::NodeProto &);

  std::map<std::string, ImportHandlerType> import_handler_map_;

  Location UnknownLoc() const { return UnknownLoc::get(&context_); }

  Value none() { return builder_.create<ONNXNoneOp>(UnknownLoc()).getResult(); }

  // onnx_type_map: a map from ONNX tensor name to ONNX TypeProto.
  SymbolToOnnxTypeMapping onnx_type_map;

  void AddValueInfo(const onnx::ValueInfoProto &vi, bool allowExist = false) {
    if (allowExist && onnx_type_map.ContainKey(vi.name()))
      return;
    onnx_type_map.AddMapping(vi.name(), vi.type());
  }

  // opset_map_ is the internal (map) representation of ModelProto::opset_import
  // It maps each domain (e.g., "ai.onnx") to the specific version of that opset
  // used by this model.
  std::map<std::string, int64_t> opset_map_;
  void SetOpSetImport(const onnx::ModelProto &model) {
    opset_map_.clear();
    for (auto &binding : model.opset_import()) {
      opset_map_[binding.domain()] = binding.version();
    }
  }

  void BindOnnxName(const std::string &onnx_name, Value symbol) {
    frontend_symbols_.AddMapping(onnx_name, symbol);
  }

  Value LookupOnnxName(const std::string &onnx_name) {
    return frontend_symbols_.GetTensorByOnnxName(onnx_name);
  }

  /*!
   * Import an onnx tensor type by determining and returning its type
   * @param type_proto onnx tensor TypeProto.
   */
  Type ImportTensorType(const onnx::TypeProto &type_proto) {
    assert(type_proto.value_case() == onnx::TypeProto::kTensorType &&
           "expect tensor type");
    std::vector<int64_t> dims;
    auto tensor_type = type_proto.tensor_type();
    auto elementOnnxType = (onnx::TensorProto_DataType)tensor_type.elem_type();
    Type elementType = convertONNXTypeToMLIRType(builder_, elementOnnxType);
    if (!tensor_type.has_shape()) {
      return UnrankedTensorType::get(elementType);
    }
    auto shape_proto = tensor_type.shape();
    for (int i = 0; i < shape_proto.dim_size(); i++) {
      if (shape_proto.dim()[i].dim_value()) {
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        assert(dim_numeric_size != 0 &&
               "Parsed an tensor with a dimension size of zero");
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

    llvm::ArrayRef<int64_t> tensor_dims(dims.data(), dims.size());
    return RankedTensorType::get(tensor_dims, elementType);
  }

  Type ImportSequenceType(const onnx::TypeProto &type_proto) {
    auto input_seq_type = type_proto.sequence_type();
    if (input_seq_type.has_elem_type()) {
      onnx::TypeProto elem_type = input_seq_type.elem_type();
      assert(elem_type.value_case() == onnx::TypeProto::kTensorType &&
             "expect tensor inside sequence type");
      Type mlir_elem_type = ImportTensorType(elem_type);
      if (!mlir_elem_type.isa<ShapedType>())
        llvm_unreachable("Seq type is incorrect");
      Type seq_type = mlir::SeqType::get(mlir_elem_type.cast<ShapedType>(), -1);
      return seq_type;
    }
    llvm_unreachable("unexpected type");
  }

  Type ImportType(const onnx::TypeProto &type_proto) {
    switch (type_proto.value_case()) {
    case onnx::TypeProto::kTensorType:
      return ImportTensorType(type_proto);
      break;
    case onnx::TypeProto::kSequenceType:
      return ImportSequenceType(type_proto);
      break;
    default:
      llvm_unreachable("unexpected type");
      break;
    }
  }

  llvm::Optional<Type> ConvertOnnxType(const std::string &onnx_name) {
    if (options_.useOnnxModelTypes) {
      if (onnx_type_map.ContainKey(onnx_name)) {
        return llvm::Optional<Type>(
            ImportType(onnx_type_map.GetTensorByOnnxName(onnx_name)));
      }
    }
    return llvm::Optional<Type>();
  }

  /*!
   * Import a input tensor symbol by recording a new entry in frontend_symbols_
   * recording the mapping between legalized onnx tensor name and Value
   * for further lookup in computation node importing.
   * @param input onnx input tensor ValueInfoProto.
   * @param symbol mlir input argument.
   */
  void ImportInputTensorSymbol(
      const onnx::ValueInfoProto &input, Value symbol) {
    BindOnnxName(input.name(), symbol);
  }

  NamedAttribute convertOnnxAttributeProtoToMlirNamedAttribute(
      onnx::AttributeProto attr) {
    Attribute mlirAttr;
    switch (attr.type()) {
    case onnx::AttributeProto::FLOAT:
      mlirAttr = builder_.getF32FloatAttr(attr.f());
      break;
    case onnx::AttributeProto::INT:
      mlirAttr =
          IntegerAttr::get(builder_.getIntegerType(64, /*isSigned=*/true),
              APInt(64, /*value=*/attr.i(), /*isSigned=*/true));
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
      llvm::SmallVector<StringRef, 4> vectorStringRef;
      for (const auto &item : attr.strings()) {
        vectorStringRef.push_back(llvm::StringRef(item));
      }
      mlirAttr = builder_.getStrArrayAttr(llvm::makeArrayRef(vectorStringRef));
    } break;
    case onnx::AttributeProto::GRAPH: {
      llvm_unreachable("Subgraph attribute is imported as regions.");
      break;
    }
    default:
      llvm_unreachable("datatype for attribute is not implemented");
      break;
    }
    return builder_.getNamedAttr(attr.name(), mlirAttr);
  }

  std::vector<NamedAttribute> ImportNodeAttributes(
      const onnx::NodeProto &node) {
    std::vector<NamedAttribute> attributes;
    for (int i = 0; i < node.attribute_size(); ++i) {
      const auto &attr = node.attribute(i);
      // Ignore subgraph attributes, as they will be imported as regions.
      if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH)
        continue;
      attributes.push_back(convertOnnxAttributeProtoToMlirNamedAttribute(attr));
    }

    // If the node has a name, then import it.
    if (node.has_name()) {
      attributes.push_back(builder_.getNamedAttr(
          "onnx_node_name", builder_.getStringAttr(node.name())));
    }
    return attributes;
  }

  /*!
   * An alternative graph importing procedure for importing ONNX subgraphs.
   * ONNX subgraphs, unlike the main computation graph, are imported as regions
   * nested within the associated operations (e.g., the loop body subgraph
   * associated with Loop operation).
   * @param graph sub-computation graph to import.
   * @param region region to import computation graph to.
   * @param op operations whose attributes will be updated to contain
   * input/output names.
   * @param useStdReturn if set to true, will emit standard return op as
   * terminator, otherwise, will use OnnxReturn op as terminator.
   * @return function type corresponding to the subgraph input/output signature.
   */
  FunctionType importGraph(const onnx::GraphProto &graph, Region &region,
      Operation *op, bool useStdReturn) {
    frontend_symbols_.pushScope(graph.name());
    initializedTensors.pushScope(graph.name());
    onnx_type_map.pushScope(graph.name());
    Block *entryBlock = &region.back();

    // Maintain a mapping between the parameter and its initializer.
    for (const auto &initializer : graph.initializer()) {
      const auto &initializerName = initializer.name();
      initializedTensors.AddMapping(initializerName, initializer);
    }

    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<Type, 4> argTypes;

    llvm::SmallVector<llvm::StringRef, 4> inputNames;
    llvm::SmallVector<llvm::StringRef, 4> outputNames;

    // Import the input tensor types that are not constant and not initialized.
    int numInputs = 0;
    for (const auto &input : graph.input()) {
      AddValueInfo(input);
      if (!initializedTensors.ContainKey(input.name())) {
        inputNames.push_back(input.name());
        auto argTy = ImportType(input.type());
        auto shapedTy = argTy.dyn_cast<RankedTensorType>();
        // Change the first dimension to unknown (-1) for test purpose only
        if (shapedTy && force_dim_dynamic_enabled_ &&
            ((forced_inputs_dims.find(-1) != forced_inputs_dims.end()) ||
                (forced_inputs_dims.find(numInputs) !=
                    forced_inputs_dims.end()))) {
          std::vector<int> forced_dims;
          if (forced_inputs_dims.find(-1) != forced_inputs_dims.end())
            forced_dims = forced_inputs_dims.at(-1);
          else
            forced_dims = forced_inputs_dims.at(numInputs);
          auto argShape = shapedTy.getShape();
          llvm::SmallVector<int64_t, 4> newDims;
          for (unsigned int i = 0; i < argShape.size(); i++) {
            if (llvm::is_contained(forced_dims, -1) ||
                llvm::is_contained(forced_dims, i)) {
              newDims.push_back(-1);
            } else {
              newDims.push_back(argShape[i]);
            }
          }
          argTy = RankedTensorType::get(newDims, shapedTy.getElementType());
        } else if (shapedTy && !inputs_shape_information.empty() &&
                   (inputs_shape_information.find(numInputs) !=
                       inputs_shape_information.end())) {
          // Change to the custom shape if users provide.
          std::vector<int64_t> shape = inputs_shape_information.at(numInputs);
          argTy = RankedTensorType::get(shape, shapedTy.getElementType());
        }

        argTypes.emplace_back(argTy);

        // numInputs is the number of graph inputs not contained within the
        // initializer
        ++numInputs;
      }
    }

    for (const auto &internal : graph.value_info()) {
      AddValueInfo(internal);
    }

    for (const auto &output : graph.output()) {
      // Output tensor may be in input list
      AddValueInfo(output, true);
      outputNames.push_back(output.name());
    }

    entryBlock->addArguments(argTypes,
        llvm::SmallVector<Location, 4>(argTypes.size(), UnknownLoc()));

    // Map graph inputs to entry block arguments.
    // Counter of un-initialized tensors. This counter is used to index the
    // entry block arguments.
    int entryBlockArgIdx = 0;
    for (const onnx::ValueInfoProto &inputProto : graph.input()) {
      if (!initializedTensors.ContainKey(inputProto.name())) {
        ImportInputTensorSymbol(
            inputProto, entryBlock->getArguments()[entryBlockArgIdx]);
        entryBlockArgIdx++;
      }
    }

    // Import nodes in the subgraph.
    for (const auto &item : graph.node()) {
      ImportNode(item);
    }

    llvm::SmallVector<Type, 4> retTys;
    llvm::SmallVector<Value, 4> retVals;
    // Import the output tensors
    for (const auto &output : graph.output()) {
      ImportOutputTensor(output, retTys, retVals);
    }

    if (useStdReturn)
      builder_.create<func::ReturnOp>(UnknownLoc(), retVals);
    else
      // Create a return operation to return all ONNX output tensors.
      builder_.create<ONNXReturnOp>(UnknownLoc(), retVals);

    op->setAttr("input_names", builder_.getStrArrayAttr(inputNames));
    op->setAttr("output_names", builder_.getStrArrayAttr(outputNames));

    frontend_symbols_.popScope(graph.name());
    initializedTensors.popScope(graph.name());
    onnx_type_map.popScope(graph.name());
    return builder_.getFunctionType(argTypes, retTys);
  }

  void ImportNodeGeneric(const onnx::NodeProto &node) {
    std::vector<Value> inputs;
    for (const auto &item : node.input()) {
      if (frontend_symbols_.ContainKey(item)) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }
    OperationState result(UnknownLoc(), "frontend." + node.op_type());
    for (auto item : node.output()) {
      result.addTypes(UnrankedTensorType::get(builder_.getF32Type()));
    }
    result.addOperands(inputs);
    result.addAttributes(ImportNodeAttributes(node));
    // Create corresponding regions for graph attributes.
    for (const auto &attr : node.attribute())
      // Ignore subgraph attributes, as they will be imported as regions.
      if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH)
        result.addRegion();

    auto op = builder_.createOperation(result);
    for (int i = 0; i < node.output().size(); i++) {
      auto r = op->getResult(i);
      frontend_symbols_.AddMapping(node.output()[i], r);
    }
  }

  static constexpr int MAX_TYPE = 20;

  // itblgen_types = ('I1', 'I8', 'I16', 'I32', 'I64', 'BF16', 'F16', 'F32',
  // 'F64', 'Complex<F32>', 'Complex<F64>' )
  Type buildTypeFromIndex(int index) {
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
      std::vector<Type> typeTuple(2);
      typeTuple.push_back(builder_.getF32Type());
      typeTuple.push_back(builder_.getF32Type());
      return builder_.getTupleType(llvm::ArrayRef<Type>(typeTuple));
    }
    case 10: {
      std::vector<Type> typeTuple(2);
      typeTuple.push_back(builder_.getF64Type());
      typeTuple.push_back(builder_.getF64Type());
      return builder_.getTupleType(llvm::ArrayRef<Type>(typeTuple));
    }
    default:
      assert(false && "Unsupported type index encountered.");
      return nullptr;
    }
  }

  template <typename T>
  void buildOutputAndOperation(const onnx::NodeProto &node,
      std::vector<Value> inputs, int expectedNumOperands,
      int expectedNumResults, const std::vector<NamedAttribute> &attributes,
      std::vector<Type> givenOutputTypes = std::vector<Type>()) {
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
      for (int i = (int)inputs.size(); i < expectedNumOperands; i++) {
        inputs.emplace_back(none());
      }

    std::vector<Type> outputTypes;

    // Use the type map or types in input model to determine the data type of
    // output.
    std::vector<int> outputMap = T::getTypeMap();
    for (unsigned int i = 0; i < (unsigned int)node.output().size(); i++) {
      // Optional outputs using empty string.
      if (node.output()[i].empty()) {
        outputTypes.emplace_back(builder_.getNoneType());
      } else if (auto onnxModelType = ConvertOnnxType(node.output(i))) {
        outputTypes.emplace_back(onnxModelType.getValue());
      } else {
        unsigned int j = i;
        // Variadic output is a single ODS result.
        if (variadicOut)
          j = 0;
        if (j < outputMap.size() && outputMap[j] >= MAX_TYPE) {
          // Mapping gives a connection with an input.
          Type inputType = inputs[outputMap[j] - MAX_TYPE].getType();
          if (inputType.isa<TensorType>()) {
            auto elementType = inputType.cast<TensorType>().getElementType();
            auto outType = UnrankedTensorType::get(elementType);
            outputTypes.emplace_back(outType);
          } else {
            outputTypes.push_back(inputType);
          }
        } else if (j < outputMap.size() && outputMap[j] != -1) {
          // Mapping gives a direct type.
          auto elementType = buildTypeFromIndex(outputMap[j]);
          auto outType = UnrankedTensorType::get(elementType);
          outputTypes.emplace_back(outType);
        } else if (!givenOutputTypes.empty()) {
          outputTypes.emplace_back(
              UnrankedTensorType::get(givenOutputTypes[i]));
        } else {
          outputTypes.emplace_back(builder_.getNoneType());
        }
      }
    }
    // Trailing optional outputs.
    if (!variadicOut)
      for (int i = node.output().size(); i < expectedNumResults; ++i)
        outputTypes.emplace_back(builder_.getNoneType());

    // TODO: Handle optional inputs.
    auto op = builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
    Operation *genericOp = op.getOperation();
    // Type inference for results.
    for (const auto &attr : node.attribute()) {
      if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH) {
        if (auto opWithSubgraph =
                dyn_cast<HasOnnxSubgraphOpInterface>(op.getOperation())) {
          auto regionIdx = opWithSubgraph.getSubgraphRegionIdx(attr.name());
          auto &region = op->getRegion(regionIdx);
          region.push_back(new Block);
          OpBuilder::InsertionGuard guard(builder_);
          builder_.setInsertionPointToStart(&region.back());
          auto funcType =
              importGraph(attr.g(), region, op.getOperation(), false);
          // Use type info from graph to reset type of output for current op
          for (int i = 0; i < node.output().size(); i++) {
            Type type = funcType.getResults()[i];
            genericOp->getOpResult(i).setType(type);
          }
        } else {
          llvm_unreachable("Op contains subgraph attributes but does not "
                           "implement HasOnnxSubgraphOpInterface interface.");
        }
      }
    }
    if (auto opWithTypeInference =
            dyn_cast<ResultTypeInferenceOpInterface>(genericOp)) {
      auto outTypes = opWithTypeInference.resultTypeInference();
      for (int i = 0; i < node.output().size(); i++) {
        auto result = genericOp->getOpResult(i);
        if (!options_.useOnnxModelTypes || result.getType().isa<NoneType>())
          genericOp->getOpResult(i).setType(outTypes[i]);
      }
    }

    for (const auto &output : llvm::enumerate(node.output()))
      frontend_symbols_.AddMapping(
          output.value(), genericOp->getOpResult(output.index()));
  }

  void getNodeInputs(const onnx::NodeProto &node, std::vector<Value> &inputs) {
    for (const auto &item : node.input())
      if (item.empty()) {
        inputs.emplace_back(none());
      } else {
        if (initializedTensors.ContainKey(item)) {
          inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
              UnknownLoc(), builder_, item));
        } else if (frontend_symbols_.ContainKey(item)) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      }
  }

  template <typename T>
  void buildOperation(const onnx::NodeProto &node) {
    std::vector<Value> inputs;
    int expectedNumOperands = T::getNumberOfOperands();
    int expectedNumResults = T::getNumberOfResults();
    getNodeInputs(node, inputs);
    auto attributes = ImportNodeAttributes(node);
    buildOutputAndOperation<T>(
        node, inputs, expectedNumOperands, expectedNumResults, attributes);
  }

  // The output type of CategoryMapper needs special handling
  // If the input is I64, the output is string.
  // If the input is string, the output is I64.
  void ImportCategoryMapper(const onnx::NodeProto &node) {
    std::vector<Value> inputs;
    int expectedNumOperands = ONNXCategoryMapperOp::getNumberOfOperands();
    int expectedNumResults = ONNXCategoryMapperOp::getNumberOfResults();
    getNodeInputs(node, inputs);
    auto attributes = ImportNodeAttributes(node);
    std::vector<Type> outputTypes;
    auto inputType = inputs[0].getType().cast<TensorType>();
    if (inputType.getElementType().isInteger(64)) {
      outputTypes.emplace_back(
          mlir::ONNXStringType::get(builder_.getContext()));
    } else {
      outputTypes.emplace_back(builder_.getIntegerType(64));
    }
    buildOutputAndOperation<ONNXCategoryMapperOp>(node, inputs,
        expectedNumOperands, expectedNumResults, attributes, outputTypes);
  }

  std::vector<NamedAttribute> ImportCastAttributes(
      const onnx::NodeProto &node) {
    std::vector<NamedAttribute> attributes;
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      auto mlir_type = convertONNXTypeToMLIRType(
          builder_, static_cast<onnx::TensorProto_DataType>(attr.i()));
      Attribute mlirAttr = TypeAttr::get(mlir_type);
      attributes.push_back(builder_.getNamedAttr(attr.name(), mlirAttr));
    }

    // If the node has a name, then import it.
    if (node.has_name()) {
      attributes.push_back(builder_.getNamedAttr(
          "onnx_node_name", builder_.getStringAttr(node.name())));
    }
    return attributes;
  }

  /*!
   * Special handle for Cast operations.
   */
  void ImportNodeCast(const onnx::NodeProto &node) {
    std::vector<Value> inputs;
    int expectedNumOperands = ONNXCastOp::getNumberOfOperands();
    int expectedNumResults = ONNXCastOp::getNumberOfResults();
    for (const auto &item : node.input())
      if (item.empty()) {
        // Optional inputs using empty string will be imported as NoneType.
        inputs.emplace_back(none());
      } else {
        if (initializedTensors.ContainKey(item)) {
          inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
              UnknownLoc(), builder_, item));
        } else if (frontend_symbols_.ContainKey(item)) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      }
    auto attributes = ImportCastAttributes(node);
    buildOutputAndOperation<ONNXCastOp>(
        node, inputs, expectedNumOperands, expectedNumResults, attributes);
  }

  /*!
   * Special handle for MaxPool operations.
   */
  void ImportNodeMaxPool(const onnx::NodeProto &node) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      buildOperation<ONNXMaxPoolSingleOutOp>(node);
    } else {
      buildOperation<ONNXMaxPoolOp>(node);
    }
  }

  /*!
   * Special handle for BatchNormalization operations.
   */
  void ImportNodeBatchNormalization(const onnx::NodeProto &node) {
    int nOuts = node.output().size();
    if (nOuts == 1) {
      // Inference mode with one output.
      buildOperation<ONNXBatchNormalizationInferenceModeOp>(node);
    } else {
      // Training mode with four trailing optional outputs. Not handled yet.
      buildOperation<ONNXBatchNormalizationOp>(node);
    }
  }

  /*!
   * Special handle for Dropout operations.
   */
  void ImportNodeDropout(const onnx::NodeProto &node) {
    int nOps = node.input().size();
    int nIn = ONNXDropoutOp::getNumberOfOperands();
    if (nOps == nIn) {
      // All inputs are specified
      buildOperation<ONNXDropoutOp>(node);
      return;
    }

    // Add the default value for optional input
    // Copy the provided inputs first
    std::vector<Value> inputs;
    for (const auto &item : node.input()) {
      if (initializedTensors.ContainKey(item)) {
        inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, item));
      } else {
        if (frontend_symbols_.ContainKey(item)) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      }
    }

    // If ratio is not specified, the default value is 0.5
    if (nOps < 2) {
      llvm::SmallVector<int64_t, 1> dims;
      dims.push_back(1);
      llvm::SmallVector<float, 1> values;
      values.push_back(0.5);
      auto elementType = builder_.getF32Type();
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
      auto constantOp = builder_.create<ONNXConstantOp>(
          UnknownLoc(), Attribute(), constantDenseAttribute);
      Value constantResult = *(constantOp.getODSResults(0).begin());
      inputs.push_back(constantResult);
    }

    // If training_mode is not specified, the default value is false
    if (nOps < 3) {
      llvm::SmallVector<int64_t, 1> dims;
      dims.push_back(1);
      llvm::SmallVector<bool, 1> values;
      values.push_back(false);
      auto elementType = builder_.getIntegerType(1);
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
      auto constantOp = builder_.create<ONNXConstantOp>(
          UnknownLoc(), Attribute(), constantDenseAttribute);
      Value constantResult = *(constantOp.getODSResults(0).begin());
      inputs.push_back(constantResult);
    }
    int nOut = ONNXDropoutOp::getNumberOfResults();
    auto attributes = ImportNodeAttributes(node);
    buildOutputAndOperation<ONNXDropoutOp>(node, inputs, nIn, nOut, attributes);
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
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));

      // Use the special builder defined in ONNXOp.td.inc.
      auto constantOp = builder_.create<ONNXConstantOp>(
          UnknownLoc(), Attribute(), constantDenseAttribute);
      Value constantResult = *(constantOp.getODSResults(0).begin());
      std::vector<Value> inputs;
      for (const auto &item : node.input()) {
        if (initializedTensors.ContainKey(item)) {
          inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
              UnknownLoc(), builder_, item));
        } else if (frontend_symbols_.ContainKey(item)) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      }
      inputs.push_back(constantResult);

      int nIn = ONNXPadOp::getNumberOfOperands();
      int nOut = ONNXPadOp::getNumberOfResults();
      auto attributes = ImportNodeAttributes(node);
      buildOutputAndOperation<ONNXPadOp>(node, inputs, nIn, nOut, attributes);
    } else {
      buildOperation<ONNXPadOp>(node);
    }
  }

  void ImportNodeSlice(const onnx::NodeProto &node) {
    std::array<Value, 5> inVals = {
        nullptr,
    };

    for (const auto &item : llvm::enumerate(node.input())) {
      if (initializedTensors.ContainKey(item.value())) {
        inVals[item.index()] = initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, item.value());
      } else {
        if (frontend_symbols_.ContainKey(item.value())) {
          inVals[item.index()] =
              frontend_symbols_.GetTensorByOnnxName(item.value());
        } else {
          assert(false && "Unknown input");
        }
      }
    }

    // Data input is imported but starts, ends, axes, and steps may come from
    // attributes, and need to be created as constant ops.
    const auto elementType = builder_.getIntegerType(64);
    const auto attributes = ImportNodeAttributes(node);
    for (auto attr : attributes) {
      if (auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>()) {
        const auto tensorType =
            RankedTensorType::get({(int64_t)arrayAttr.size()}, elementType);
        auto constantDenseAttribute =
            DenseElementsAttr::get(tensorType, arrayAttr.getValue());
        auto constantOp = builder_.create<ONNXConstantOp>(
            UnknownLoc(), Attribute(), constantDenseAttribute);
        Value constantValue = constantOp.output();

        // Map from ONNX attributes to indices, which are
        // matched with ONNXSliceOp::build ordering.
        auto inputIdx = llvm::StringSwitch<int>(attr.getName())
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
    inVals[3] = inVals[3] == nullptr ? none() : inVals[3];
    inVals[4] = inVals[4] == nullptr ? none() : inVals[4];

    int nIn = ONNXSliceOp::getNumberOfOperands();
    int nOut = ONNXSliceOp::getNumberOfResults();
    const auto in = std::vector<Value>(inVals.begin(), inVals.end());

    buildOutputAndOperation<ONNXSliceOp>(node, in, nIn, nOut, attributes);
  }

  /*!
   * Special handle for Softmax operation where the default axis value depends
   * on the opset version.
   */
  void ImportNodeSoftmax(const onnx::NodeProto &node) {
    // Copy the provided inputs first.
    std::vector<Value> inputs;
    for (const auto &item : node.input()) {
      if (initializedTensors.ContainKey(item)) {
        inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
            UnknownLoc(), builder_, item));
      } else if (frontend_symbols_.ContainKey(item)) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }

    int nIn = ONNXSoftmaxOp::getNumberOfOperands();
    int nOut = ONNXSoftmaxOp::getNumberOfResults();

    // If no attribute is provided, axis would depend on the opset version.
    // - With opset version < 13, default axis value is 1.
    // - With opset version 13, default axis value is -1.
    auto currentOpset = opset_map_.find(node.domain())->second;
    auto attributes = ImportNodeAttributes(node);
    bool hasAxisAttribute = false;
    for (auto &attr : attributes)
      if (attr.getName().strref().equals_insensitive("axis")) {
        hasAxisAttribute = true;
        break;
      }

    if (!hasAxisAttribute) {
      if (currentOpset < 13)
        attributes.push_back(builder_.getNamedAttr("axis",
            IntegerAttr::get(builder_.getIntegerType(64, /*isSigned=*/true),
                APInt(64, /*value=*/1, /*isSigned=*/true))));
    }

    // Store the opset version in an attribute, which is used for the lowering.
    attributes.push_back(builder_.getNamedAttr("onnx_opset",
        IntegerAttr::get(builder_.getIntegerType(64, /*isSigned=*/true),
            APInt(64, /*value=*/currentOpset, /*isSigned=*/true))));

    buildOutputAndOperation<ONNXSoftmaxOp>(node, inputs, nIn, nOut, attributes);
  }

  const onnx::OpSchema *GetOpSchema(const onnx::NodeProto &node) {
    auto &domain = node.domain();
    auto version_it = opset_map_.find(domain);
    if (version_it == opset_map_.end())
      return nullptr;
    auto version = version_it->second;
    return onnx::OpSchemaRegistry::Schema(node.op_type(), version, domain);
  }

  std::string GetImportVersionOfNode(const onnx::NodeProto &node) {
    auto schema = GetOpSchema(node);
    // Assume the top version
    if (schema == nullptr) {
      return std::string("");
    }
    auto current_opset = opset_map_.find(node.domain())->second;

    if (current_opset < MINIMUM_SUPPORTED_OPSET)
      llvm::outs() << "Warning: ONNX " << node.op_type()
                   << " in your model is using Opset " << current_opset
                   << ", which is quite old. Please consider regenerating your "
                      "model with a newer Opset.\n";
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": Importing ONNX"
                            << node.op_type() << " (" << node.name() << ")"
                            << ", Opset: " << current_opset << "\n");

    // Custom ops may not be present in op_dialect_version_map_. If no version
    // info is found, treat as unversioned (no renaming).
    auto opset_list_it = op_dialect_version_map_.find(node.op_type());
    if (opset_list_it != op_dialect_version_map_.end()) {
      auto opset_list = opset_list_it->second;
      // A new opset is added to onnx-mlir when it becomes imcompactible.
      // But the lowest opset in op_dialect_version_map_ is an exception.
      // It is the current opset when onnx-mlir project is started.
      // All opset lower than the last opset should use the last opset(version)
      if (opset_list.size() == 1)
        return std::string("");
      for (int i = opset_list.size() - 1; i > 0; i--) {
        if (current_opset < opset_list[i - 1]) {
          LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ":   - use Opset "
                                  << opset_list[i] << "\n");
          return "V" + std::to_string(opset_list[i]);
        }
      }
    }
    return std::string("");
  }

  FuncOp CreateFuncOp(
      std::string namePrefix, TypeRange operandTypes, TypeRange resultTypes) {
    auto funcType = builder_.getFunctionType(operandTypes, resultTypes);
    if (namePrefix.empty())
      namePrefix = "fn";
    std::string funcName = namePrefix;
    // make name  unique:
    for (int suffix = 1; module_.lookupSymbol(funcName); ++suffix) {
      funcName = namePrefix + "_" + std::to_string(suffix);
    }

    auto funcOp = FuncOp::create(UnknownLoc(), funcName, funcType);
    module_.insert(module_.begin(), funcOp);
    return funcOp;
  }

  void InferTypes(const onnx::FunctionProto *func,
      std::vector<onnx::TypeProto> &inputTypes) {
    // types: Used for temporary copies of Types, freed at end of function.
    std::vector<std::unique_ptr<onnx::TypeProto>> types;
    std::unordered_map<std::string, onnx::TypeProto *> typeMap;
    // Initialize types and values (if available) of function inputs:
    const auto num_inputs =
        std::min(func->input_size(), static_cast<int>(inputTypes.size()));
    for (int i = 0; i < num_inputs; ++i) {
      const std::string &input_name = func->input(i);
      typeMap[input_name] = &inputTypes[i];
    }

    for (const onnx::NodeProto &n : func->node()) {
      const auto *schema = GetOpSchema(n);
      if (!schema) {
        continue; // TODO:
      }

      onnx::NodeProto tn(n);
      onnx::shape_inference::InferenceContextImpl node_ctx(tn, typeMap, {}, {});
      schema->GetTypeAndShapeInferenceFunction()(node_ctx);

      // Update types:
      for (int i = 0; i < n.output_size(); ++i) {
        std::unique_ptr<onnx::TypeProto> p =
            std::make_unique<onnx::TypeProto>(*node_ctx.getOutputType(i));
        typeMap[n.output(i)] = p.get();
        types.push_back(std::move(p));
      }
    }

    for (auto pair : typeMap) {
      onnx_type_map.AddMapping(pair.first, *pair.second);
    }
  }

  bool TryImportFunctionCallNode(const onnx::NodeProto &node) {
    const onnx::OpSchema *schema = GetOpSchema(node);
    if (schema == nullptr)
      return false;

    // Collect input/output MLIR types, input ONNX types, and input MLIR values.
    // TODO: Optional inputs/outputs of functions not handled yet.
    onnx::TypeProto unspecifiedType;
    llvm::SmallVector<Type, 16> operandTypes;
    llvm::SmallVector<Type, 16> resultTypes;
    llvm::SmallVector<::Value, 16> operands;
    std::vector<onnx::TypeProto> operandOnnxTypes;

    for (auto &v : node.input()) {
      if (v.empty()) {
        // Missing (optional) parameter.
        operandOnnxTypes.push_back(unspecifiedType);
        auto no_value = builder_.create<ONNXNoneOp>(UnknownLoc());

        operands.push_back(no_value);
        operandTypes.push_back(builder_.getNoneType());
        continue;
      }
      // Function translation requires input (onnx) types
      if (!onnx_type_map.ContainKey(v))
        return false;
      operandOnnxTypes.push_back(onnx_type_map.GetTensorByOnnxName(v));
      auto val = LookupOnnxName(v);
      operands.push_back(val);
      operandTypes.push_back(val.getType());
    }
    for (auto &v : node.output()) {
      if (v.empty())
        continue;
      if (!onnx_type_map.ContainKey(v))
        return false;
      auto resultType = ImportType(onnx_type_map.GetTensorByOnnxName(v));
      resultTypes.push_back(resultType);
    }

    // Get ONNX function body:
    onnx::FunctionProto functionProto;
    // Try generating a context-independent function body:
    const onnx::FunctionProto *pFunctionProto = schema->GetFunction();
    if (!pFunctionProto) {
      // Try generating a context-dependent function body:
      onnx::FunctionBodyBuildContextImpl onnxFunContext(node, operandOnnxTypes);
      if (schema->HasContextDependentFunction() &&
          schema->BuildContextDependentFunction(onnxFunContext, functionProto))
        pFunctionProto = &functionProto;
      else
        return false;
    }

    // Create MLIR function:
    const std::string &func_name_prefix =
        node.name().empty() ? node.op_type() : node.name();
    auto funcOp = CreateFuncOp(func_name_prefix, operandTypes, resultTypes);
    auto *fnEntryBlock = funcOp.addEntryBlock();

    // Save caller context, while generating callee function body.
    ValueSymbolMapping callerScope(std::move(frontend_symbols_));
    frontend_symbols_.pushScope(func_name_prefix);
    SymbolToOnnxTypeMapping callerTypeMap(std::move(onnx_type_map));
    onnx_type_map.pushScope(func_name_prefix);

    auto prev_ip = builder_.saveInsertionPoint();
    builder_.setInsertionPointToStart(fnEntryBlock);

    // Generate MLIR function body

    auto formalParamValues = fnEntryBlock->getArguments();
    // Due to missing trailing optional parameters,
    // fnEntryBlock->getNumArguments() and pFunctionProto->input_size() may be
    // unequal.
    int num_formals =
        std::min(static_cast<int>(fnEntryBlock->getNumArguments()),
            pFunctionProto->input_size());
    for (int formal_num = 0; formal_num < num_formals; formal_num++) {
      const std::string &v = pFunctionProto->input(formal_num);
      BindOnnxName(v, formalParamValues[formal_num]);
    }

    // Apply ONNX type inference to FunctionProto:
    InferTypes(pFunctionProto, operandOnnxTypes);

    for (auto &fb_node : pFunctionProto->node()) {
      ImportNode(fb_node);
    }

    // Create a return operation to return all output tensors.
    llvm::SmallVector<Value, 4> ret_vals;
    for (auto &v : pFunctionProto->output()) {
      ret_vals.push_back(LookupOnnxName(v));
    }
    builder_.create<func::ReturnOp>(UnknownLoc(), ret_vals);

    // Restore caller context
    frontend_symbols_.popScope(func_name_prefix);
    frontend_symbols_ = std::move(callerScope);
    onnx_type_map.popScope(func_name_prefix);
    onnx_type_map = std::move(callerTypeMap);

    builder_.restoreInsertionPoint(prev_ip);

    // Generate call statement
    auto op = builder_.create<ONNXCallOp>(UnknownLoc(), funcOp, operands);
    int result_num = 0;
    for (auto &v : node.output()) {
      BindOnnxName(v, op.getResult(result_num++));
    }

    return true;
  }

  void ImportCustomNode(const onnx::NodeProto &node) {
    if (!TryImportFunctionCallNode(node)) {
      emitWarning(UnknownLoc(), "Could not find op importer: assuming this "
                                "represents a custom operator.");

      llvm::StringRef opName = node.op_type();
      auto funcName = opName.str();
      std::vector<Type> outputTypes;
      std::vector<Value> inputs;
      auto attributes = ImportNodeAttributes(node);
      auto mlirAttr = builder_.getStringAttr(funcName);
      auto funcAttr = builder_.getNamedAttr("function_name", mlirAttr);
      attributes.push_back(funcAttr);
      auto domainAttr = builder_.getNamedAttr(
          "domain_name", builder_.getStringAttr(node.domain()));
      attributes.push_back(domainAttr);
      int nIn = 0;
      int nOut = 0;
      getNodeInputs(node, inputs);
      nOut = node.output().size();
      buildOutputAndOperation<ONNXCustomOp>(
          node, inputs, nIn, nOut, attributes);
    }
  }

  void ImportNode(const onnx::NodeProto &node) {
    std::string versionStr = GetImportVersionOfNode(node);

    // look up handler for the opName. If not found, create a node
    // for a custom op, and issue a warning.
    auto handler =
        import_handler_map_.find(node.op_type() + versionStr.c_str());
    if (handler != import_handler_map_.end()) {
      (this->*(handler->second))(node);
    } else {
      ImportCustomNode(node);
    }
  }

  void InitHandlerMap() {
#include "src/Builder/OpBuildTable.inc"
  }

  /*!
   * Import output tensor, by doing the following:
   * - Add the t/yp this output tensor to a list of tensor
   *   types representing return types of this graph function.
   * - Add this output tensor to the list of Value
   *   to be returned by the function representing computation graph.
   * @param output onnx output tensor ValueInfoProto.
   * @param ret_types a vector of tensor types representing graph's
   *   output tensor types.
   * @param ret_vals a vector of mlir Value  representing graph's
   *   output tensor.
   */
  void ImportOutputTensor(const onnx::ValueInfoProto &output,
      llvm::SmallVectorImpl<Type> &ret_types,
      llvm::SmallVectorImpl<Value> &ret_vals) {
    Value tensor_val = frontend_symbols_.GetTensorByOnnxName(output.name());
    if (output.type().value_case() == onnx::TypeProto::kTensorType) {
      if (output.type().tensor_type().has_shape()) {
        tensor_val.setType(ImportType(output.type()));
      }
    }
    ret_types.emplace_back(tensor_val.getType());
    ret_vals.push_back(tensor_val);
  }

  // construct JSON type from the argument type
  // for example - a 3D array of f32 would produce something like
  //     {"type" : "f32" , "dims" : [4, 256, 16] , "name": "t1"}
  // data type list:
  //     "i1" / "i8" / "i16" / "i32" / "i64"
  //     "ui8" / "ui16" / "ui32" / "ui64"
  //     "f32" / "f64"
  void concatTypeString(
      Type argType, Attribute attr, llvm::raw_ostream &dstream) {
    std::string comma = std::string("");

    TypeSwitch<Type>(argType)
        .Case<mlir::SeqType>([&](mlir::SeqType seqTy) {
          auto et = seqTy.getElementType();
          dstream << "   {\"seq\" : ";
          concatTypeString(et, attr, dstream);
        })
        .Case<ShapedType>([&](ShapedType tensorTy) {
          auto et = tensorTy.getElementType();
          dstream << "   { \"type\" : ";
          et.print(dstream);
          dstream << " , \"dims\" : [";
          if (tensorTy.hasRank()) {
            int64_t rank = tensorTy.getRank();
            for (int j = 0; j < rank; j++) {
              dstream << comma << tensorTy.getDimSize(j);
              comma = std::string(" , ");
            }
          } else {
          }
          dstream << "] ";
          auto name = attr.cast<mlir::StringAttr>().getValue().str();
          dstream << ", \"name\" : \"" << name << "\"";
        })
        .Default([&](Type type) { llvm_unreachable("input is not a tensor"); });
    dstream << " }\n";
  }

  std::string getSignature(FunctionType funcType, Operation *op) {
    auto inputs = funcType.getInputs();
    auto outputs = funcType.getResults();

    auto input_names = op->getAttrOfType<mlir::ArrayAttr>("input_names");
    auto output_names = op->getAttrOfType<mlir::ArrayAttr>("output_names");

    std::string const sf32 = std::string(" f32 ");
    std::string const sf64 = std::string(" f64 ");
    std::string const si32 = std::string(" i32 ");
    std::string const si64 = std::string(" i64 ");
    std::string const si16 = std::string(" i16 ");
    std::string const si8 = std::string(" i8 ");
    std::string const si1 = std::string(" i1 ");
    std::string const sui32 = std::string(" ui32 ");
    std::string const sui64 = std::string(" ui64 ");
    std::string const sui16 = std::string(" ui16 ");
    std::string const sui8 = std::string(" ui8 ");

    std::map<std::string, std::string> typeMap = {
        {sf32, std::string(" \"f32\" ")}, {sf64, std::string(" \"f64\" ")},
        {si32, std::string(" \"i32\" ")}, {si64, std::string(" \"i64\" ")},
        {si16, std::string(" \"i16\" ")}, {si8, std::string(" \"i8\" ")},
        {si1, std::string(" \"i1\" ")}, {sui32, std::string(" \"ui32\" ")},
        {sui64, std::string(" \"ui64\" ")}, {sui16, std::string(" \"ui16\" ")},
        {sui8, std::string(" \"ui8\" ")}};
    std::string dstring;
    llvm::raw_string_ostream dstream(dstring);
    dstream << "[ ";
    std::string comma = std::string("");
    for (unsigned int i = 0; i < funcType.getNumInputs(); i++) {
      dstream << comma;
      concatTypeString(inputs[i], input_names[i], dstream);
      comma = std::string(" , ");
    }
    dstream << "\n]";
    dstream.flush();
    dstring.push_back('\0'); // null terminate the input signature string
    dstream << "@[";
    comma = std::string("");
    for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
      dstream << comma;
      concatTypeString(outputs[i], output_names[i], dstream);
      comma = std::string(" , ");
    }
    dstream << "\n]";
    dstream.flush();
    dstring.push_back('\0'); // null terminate the output signature string
    for (auto const &x : typeMap) {
      size_t start_pos = 0;
      while (
          (start_pos = dstring.find(x.first, start_pos)) != std::string::npos) {
        dstring.replace(start_pos, x.first.length(), x.second);
        start_pos += x.first.length();
      }
    }

    return dstring;
  }

  /*!
   * Import ONNX main computation graph.
   * @param graph onnx graph proto.
   * @return A function corresponding to the imported computation graph.
   */
  FuncOp importGraph(const onnx::GraphProto &graph) {
    const std::string &name = "main_graph";
    auto mainFunc = FuncOp::create(UnknownLoc(), name,
        /*type=*/builder_.getFunctionType({}, {}), /*attrs=*/{});
    module_.push_back(mainFunc);
    // Create and set insertion point to entry block.
    mainFunc.body().push_back(new Block);
    builder_.setInsertionPointToStart(&mainFunc.body().back());

    auto funcType = importGraph(graph, /*region=*/mainFunc.body(),
        /*op=*/mainFunc.getOperation(), /*useStdReturn=*/true);
    mainFunc.setType(funcType);

    std::string sig = getSignature(funcType, mainFunc.getOperation());

    // Emit entry point op describing inference function signature.
    auto entryPoint = ONNXEntryPointOp::create(UnknownLoc(), mainFunc,
        /*numInputs=*/funcType.getNumInputs(),
        /*numOutputs=*/funcType.getNumResults(),
        /*signature=*/sig);
    module_.push_back(entryPoint);

    return mainFunc;
  }
}; // class FrontendGenImpl
} // namespace detail
} // namespace onnx_mlir
namespace onnx_mlir {

void ImportFrontendModelInternal(onnx::ModelProto &model, MLIRContext &context,
    OwningOpRef<ModuleOp> &module, ImportOptions options) {
  int originVersion = CURRENT_ONNX_OPSET;
  // Get the version of the model
  // Code copied from onnx/onnx/version_coverter/convert.cc
  for (auto it = model.opset_import().begin(); it != model.opset_import().end();
       ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      originVersion = it->version();
      break;
    }
  }

  // Didnot do downward convert because support for BatchNorm is missing
  if (options.invokeOnnxVersionConverter &&
      originVersion < CURRENT_ONNX_OPSET) {
    onnx::ModelProto convertModel =
        onnx::version_conversion::ConvertVersion(model, CURRENT_ONNX_OPSET);
    if (options.useOnnxModelTypes)
      onnx::shape_inference::InferShapes(convertModel);
    ImportFrontendModel(convertModel, context, module, options);
  } else {
    if (options.useOnnxModelTypes)
      onnx::shape_inference::InferShapes(model);
    ImportFrontendModel(model, context, module, options);
  }
}

void ImportFrontendModelArray(const void *onnxBuffer, int size,
    MLIRContext &context, OwningOpRef<ModuleOp> &module,
    ImportOptions options) {
  onnx::ModelProto model;

  auto parse_success = model.ParseFromArray(onnxBuffer, size);
  assert(parse_success && "Onnx Model Parsing Failed.");
  ImportFrontendModelInternal(model, context, module, options);
}

void ImportFrontendModelFile(std::string model_fname, MLIRContext &context,
    OwningOpRef<ModuleOp> &module, std::string *errorMessage,
    ImportOptions options) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);
  // check if the input file is opened
  if (!input.is_open()) {
    *errorMessage = "Unable to open or access " + model_fname;
    return;
  }

  auto parse_success = model.ParseFromIstream(&input);
  if (!parse_success) {
    *errorMessage = "Onnx Model Parsing Failed on " + model_fname;
    return;
  }
  ImportFrontendModelInternal(model, context, module, options);
}

void ImportFrontendModel(const onnx::ModelProto &model, MLIRContext &context,
    OwningOpRef<ModuleOp> &module, ImportOptions options) {

  detail::FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model, options);
}

} // namespace onnx_mlir
