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

#include "mlir/IR/BuiltinOps.h"
#include "onnx/defs/schema.h"

#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"

#include "FrontendDialectTransformer.hpp"

namespace onnx_mlir {
namespace detail {

class FrontendGenImpl {
public:
  explicit FrontendGenImpl(mlir::MLIRContext &context)
      : context_(context), builder_(&context) {
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    InitHandlerMap();
    force_dim_dynamic_enabled_ = false;
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

  mlir::ModuleOp ImportONNXModel(
      const onnx::ModelProto &model, ImportOptions options) {
    options_ = options;
    SetOpSetImport(model); // Determines which opsets to use.
    importGraph(model.graph());
    return module_;
  }

private:
  ImportOptions options_;
  mlir::MLIRContext &context_;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;

  mlir::Value none_;
  std::map<mlir::FuncOp, mlir::Value> func2None_;

  /*!
   *  The list of tensors initialized by the ONNX model.
   */
  InitializedTensorMapping initializedTensors;

  // mapping between string name and symbol
  SymbolMapping frontend_symbols_;

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

  typedef void (onnx_mlir::detail::FrontendGenImpl::*ImportHandlerType)(
      const onnx::NodeProto &);

  std::map<std::string, ImportHandlerType> import_handler_map_;

  mlir::Location UnknownLoc() { return mlir::UnknownLoc::get(&context_); }

  mlir::Value none() {
    // Get the enclosing Func Op.
    auto block = builder_.getInsertionBlock();
    assert(block && "Builder insertion block must be set.");
    auto *op = block->getParentOp();
    mlir::FuncOp func = isa<mlir::FuncOp>(op)
                            ? dyn_cast<mlir::FuncOp>(op)
                            : op->getParentOfType<mlir::FuncOp>();

    assert(func && "Cannot find FuncOp surrounding current insertion point.");

    // Check if there's a none-typed value in the curent Func already, if so,
    // return it; if not create one.
    if (func2None_.count(func)) {
      return func2None_.at(func);
    } else {
      auto none =
          builder_
              .create<mlir::ConstantOp>(UnknownLoc(), builder_.getUnitAttr())
              .getResult();
      func2None_.emplace(func, none);
      return none;
    }
  }

  // value_info_map is a map from ONNX symbolic names to the corresponding
  // ValueInfoProto (used primarily to get the corresponding ONNX TypeProto).
  std::map<std::string, onnx::ValueInfoProto> value_info_map;

  void AddValueInfo(const onnx::ValueInfoProto &vi) {
    value_info_map[vi.name()] = vi;
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

  void BindOnnxName(const std::string &onnx_name, mlir::Value symbol) {
    frontend_symbols_.AddMapping(onnx_name, symbol);
  }

  mlir::Value LookupOnnxName(const std::string &onnx_name) {
    return frontend_symbols_.GetTensorByOnnxName(onnx_name);
  }

  /*!
   * Import an onnx tensor type by determining and returning its type
   * @param value_info onnx tensor ValueInfoProto.
   */
  mlir::Type ImportTensorType(const onnx::ValueInfoProto &value_info) {
    std::vector<int64_t> dims;
    auto tensor_type = value_info.type().tensor_type();
    auto elementOnnxType = (onnx::TensorProto_DataType)tensor_type.elem_type();
    mlir::Type elementType =
        convertONNXTypeToMLIRType(builder_, elementOnnxType);
    if (!tensor_type.has_shape()) {
      return mlir::UnrankedTensorType::get(elementType);
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
    return mlir::RankedTensorType::get(tensor_dims, elementType);
  }

  mlir::Type ConvertOnnxType(const std::string &onnx_name) {
    auto it = value_info_map.find(onnx_name);
    if (it != value_info_map.end()) {
      return ImportTensorType(it->second);
    } else {
      return builder_.getNoneType();
    }
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
    BindOnnxName(input.name(), symbol);
  }

  mlir::NamedAttribute convertOnnxAttributeProtoToMlirNamedAttribute(
      onnx::AttributeProto attr) {
    mlir::Attribute mlirAttr;
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
      llvm::SmallVector<mlir::StringRef, 4> vectorStringRef;
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

  std::vector<mlir::NamedAttribute> ImportNodeAttributes(
      const onnx::NodeProto &node) {
    std::vector<mlir::NamedAttribute> attributes;
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
  mlir::FunctionType importGraph(const onnx::GraphProto &graph,
      mlir::Region &region, mlir::Operation *op, bool useStdReturn) {
    frontend_symbols_.pushScope(graph.name());
    mlir::Block *entryBlock = &region.back();

    // Maintain a mapping between the parameter and its initializer.
    for (const auto &initializer : graph.initializer()) {
      const auto &initializerName = initializer.name();
      initializedTensors.AddMapping(initializerName, initializer);
    }

    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<mlir::Type, 4> argTypes;

    llvm::SmallVector<llvm::StringRef, 4> inputNames;
    llvm::SmallVector<llvm::StringRef, 4> outputNames;

    // Import the input tensor types that are not constant and not initialized.
    int numInputs = 0;
    for (const auto &input : graph.input()) {
      AddValueInfo(input);
      if (!initializedTensors.ContainKey(input.name())) {
        inputNames.push_back(input.name());
        auto argTy = ImportTensorType(input);
        auto shapedTy = argTy.dyn_cast<mlir::RankedTensorType>();
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
          SmallVector<int64_t, 4> newDims;
          for (auto i = 0; i < argShape.size(); i++) {
            if (llvm::is_contained(forced_dims, -1) ||
                llvm::is_contained(forced_dims, i)) {
              newDims.push_back(-1);
            } else {
              newDims.push_back(argShape[i]);
            }
          }
          argTy =
              mlir::RankedTensorType::get(newDims, shapedTy.getElementType());
        }
        argTypes.emplace_back(argTy);

        // numInputs is the number of graph inputs not contained within the
        // initializer
        ++numInputs;
      }
    }
    for (const auto &output : graph.output()) {
      AddValueInfo(output);
      outputNames.push_back(output.name());
    }

    for (const auto &internal : graph.value_info()) {
      AddValueInfo(internal);
    }

    entryBlock->addArguments(argTypes);
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

    llvm::SmallVector<mlir::Type, 4> retTys;
    llvm::SmallVector<mlir::Value, 4> retVals;
    // Import the output tensors
    for (const auto &output : graph.output()) {
      ImportOutputTensor(output, retTys, retVals);
    }

    if (useStdReturn)
      builder_.create<ReturnOp>(UnknownLoc(), retVals);
    else
      // Create a return operation to return all ONNX output tensors.
      builder_.create<ONNXReturnOp>(UnknownLoc(), retVals);

    op->setAttr("input_names", builder_.getStrArrayAttr(inputNames));
    op->setAttr("output_names", builder_.getStrArrayAttr(outputNames));

    frontend_symbols_.popScope(graph.name());
    return builder_.getFunctionType(argTypes, retTys);
  }

  void ImportNodeGeneric(const onnx::NodeProto &node) {
    std::vector<mlir::Value> inputs;
    for (const auto &item : node.input()) {
      if (frontend_symbols_.ContainKey(item)) {
        inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
      }
    }
    mlir::OperationState result(UnknownLoc(), "frontend." + node.op_type());
    for (auto item : node.output()) {
      result.addTypes(mlir::UnrankedTensorType::get(builder_.getF32Type()));
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
      int expectedNumResults,
      const std::vector<mlir::NamedAttribute> &attributes) {
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
        inputs.emplace_back(none());
      }

    std::vector<mlir::Type> outputTypes;

    // Use the type map or types in input model to determine the data type of
    // output.
    std::vector<int> outputMap = T::getTypeMap();
    for (auto i = 0; i < node.output().size(); i++) {
      // Optional outputs using empty string.
      if (node.output()[i].empty()) {
        outputTypes.emplace_back(builder_.getNoneType());
      } else if (options_.useOnnxModelTypes) {
        outputTypes.emplace_back(ConvertOnnxType(node.output(i)));
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

    // TODO: Handle optional inputs.
    auto op = builder_.create<T>(UnknownLoc(), outputTypes, inputs, attributes);
    for (const auto &attr : node.attribute()) {
      if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH) {
        if (auto opWithSubgraph =
                mlir::dyn_cast<mlir::HasOnnxSubgraphOpInterface>(
                    op.getOperation())) {
          auto regionIdx = opWithSubgraph.getSubgraphRegionIdx(attr.name());
          auto &region = op->getRegion(regionIdx);
          region.push_back(new Block);
          mlir::OpBuilder::InsertionGuard guard(builder_);
          builder_.setInsertionPointToStart(&region.back());
          importGraph(attr.g(), region, op.getOperation(), false);
        } else {
          llvm_unreachable("Op contains subgraph attributes but does not "
                           "implement HasOnnxSubgraphOpInterface interface.");
        }
      }
    }
    Operation *genericOp = op.getOperation();
    // Type inference for results.
    if (!options_.useOnnxModelTypes)
      if (auto opWithTypeInference =
              mlir::dyn_cast<mlir::ResultTypeInferenceOpInterface>(genericOp)) {
        auto outTypes = opWithTypeInference.resultTypeInference();
        for (int i = 0; i < node.output().size(); i++)
          genericOp->getOpResult(i).setType(outTypes[i]);
      }

    for (const auto &output : llvm::enumerate(node.output()))
      frontend_symbols_.AddMapping(
          output.value(), genericOp->getOpResult(output.index()));
  }

  void getNodeInputs(
      const onnx::NodeProto &node, std::vector<mlir::Value> &inputs) {
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
    std::vector<mlir::Value> inputs;
    int expectedNumOperands = T::getNumberOfOperands();
    int expectedNumResults = T::getNumberOfResults();
    getNodeInputs(node, inputs);
    auto attributes = ImportNodeAttributes(node);
    buildOutputAndOperation<T>(
        node, inputs, expectedNumOperands, expectedNumResults, attributes);
  }

  std::vector<mlir::NamedAttribute> ImportCastAttributes(
      const onnx::NodeProto &node) {
    std::vector<mlir::NamedAttribute> attributes;
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto attr = node.attribute(i);
      auto mlir_type = convertONNXTypeToMLIRType(
          builder_, static_cast<onnx::TensorProto_DataType>(attr.i()));
      mlir::Attribute mlirAttr = TypeAttr::get(mlir_type);
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
    std::vector<mlir::Value> inputs;
    int expectedNumOperands = ONNXCastOp::getNumberOfOperands();
    int expectedNumResults = ONNXCastOp::getNumberOfResults();
    for (const auto &item : node.input())
      if (item.empty()) {
        // Optional inputs using empty string will be imported as NoneType.
        if (!none_)
          none_ = builder_.create<mlir::ConstantOp>(
              UnknownLoc(), builder_.getUnitAttr());
        inputs.emplace_back(none_);
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
   * Special handle for Dropout operations.
   */
  void ImportNodeDropout(const onnx::NodeProto &node) {
    int nOps = node.input().size();
    int nIn = mlir::ONNXDropoutOp::getNumberOfOperands();
    if (nOps == nIn) {
      // All inputs are specified
      buildOperation<mlir::ONNXDropoutOp>(node);
      return;
    }

    // Add the default value for optional input
    // Copy the provided inputs first
    std::vector<mlir::Value> inputs;
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
      auto tensorType = mlir::RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
      auto constantOp = builder_.create<mlir::ONNXConstantOp>(
          UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = *(constantOp.getODSResults(0).begin());
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
      auto tensorType = mlir::RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
      auto constantOp = builder_.create<mlir::ONNXConstantOp>(
          UnknownLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = *(constantOp.getODSResults(0).begin());
      inputs.push_back(constantResult);
    }
    int nOut = mlir::ONNXDropoutOp::getNumberOfResults();
    auto attributes = ImportNodeAttributes(node);
    buildOutputAndOperation<mlir::ONNXDropoutOp>(
        node, inputs, nIn, nOut, attributes);
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
      for (const auto &item : node.input()) {
        if (initializedTensors.ContainKey(item)) {
          inputs.push_back(initializedTensors.EmitInitializerForInputTensor(
              UnknownLoc(), builder_, item));
        } else if (frontend_symbols_.ContainKey(item)) {
          inputs.push_back(frontend_symbols_.GetTensorByOnnxName(item));
        }
      }
      inputs.push_back(constantResult);

      int nIn = mlir::ONNXPadOp::getNumberOfOperands();
      int nOut = mlir::ONNXPadOp::getNumberOfResults();
      auto attributes = ImportNodeAttributes(node);
      buildOutputAndOperation<mlir::ONNXPadOp>(
          node, inputs, nIn, nOut, attributes);
    } else {
      buildOperation<mlir::ONNXPadOp>(node);
    }
  }

  void ImportNodeSlice(const onnx::NodeProto &node) {
    std::array<mlir::Value, 5> inVals = {
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
    inVals[3] = inVals[3] == nullptr ? none() : inVals[3];
    inVals[4] = inVals[4] == nullptr ? none() : inVals[4];

    int nIn = mlir::ONNXSliceOp::getNumberOfOperands();
    int nOut = mlir::ONNXSliceOp::getNumberOfResults();
    const auto in = std::vector<mlir::Value>(inVals.begin(), inVals.end());

    buildOutputAndOperation<mlir::ONNXSliceOp>(node, in, nIn, nOut, attributes);
  }

  const onnx::OpSchema *GetOpSchema(const onnx::NodeProto &node) {
    auto &domain = node.domain();
    auto version_it = opset_map_.find(domain);
    if (version_it == opset_map_.end())
      return nullptr;
    auto version = version_it->second;
    return onnx::OpSchemaRegistry::Schema(node.op_type(), version, domain);
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

    auto funcOp = mlir::FuncOp::create(UnknownLoc(), funcName, funcType);
    module_.insert(module_.begin(), funcOp);
    return funcOp;
  }

  bool TryImportFunctionCallNode(const onnx::NodeProto &node) {
    const onnx::OpSchema *schema = GetOpSchema(node);
    if (schema == nullptr)
      return false;

    // Collect input/output MLIR types, input ONNX types, and input MLIR values.
    // TODO: Optional inputs/outputs of functions not handled yet.
    onnx::TypeProto unspecifiedType;
    llvm::SmallVector<mlir::Type, 16> operandTypes;
    llvm::SmallVector<mlir::Type, 16> resultTypes;
    llvm::SmallVector<::mlir::Value, 16> operands;
    std::vector<onnx::TypeProto> operandOnnxTypes;

    for (auto &v : node.input()) {
      if (v.empty()) {
        // Use default TypeProto() to indicate missing input/type
        operandOnnxTypes.push_back(unspecifiedType);
        continue;
      }
      operandOnnxTypes.push_back(value_info_map[v].type());
      auto val = LookupOnnxName(v);
      operands.push_back(val);
      operandTypes.push_back(val.getType());
    }
    for (auto &v : node.output()) {
      if (v.empty())
        continue;
      auto resultType = ImportTensorType(value_info_map[v]);
      resultTypes.push_back(resultType);
    }

    // Get ONNX function body:
    onnx::FunctionProto functionProto;
    // Check if op is a context-independent function
    const onnx::FunctionProto *pFunctionProto = schema->GetFunction();
    if (!pFunctionProto) {
// Check if op is a context-dependent function and build function-body
#ifdef ONNX_FUNCTION_TYPE_CONTEXT
      onnx::FunctionBodyBuildContextImpl onnxFunContext(node, operandOnnxTypes);
#else
      onnx::NodeProto node_copy(node);
      onnx::FunctionBodyBuildContextImpl onnxFunContext(node_copy);
#endif
      if (schema->HasContextDependentFunction() &&
          (schema->BuildContextDependentFunction(
              onnxFunContext, functionProto)))
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
    frontend_symbols_.pushScope(func_name_prefix);
    auto prev_ip = builder_.saveInsertionPoint();
    builder_.setInsertionPointToStart(fnEntryBlock);

    // Generate MLIR function body
    int formal_num = 0;
    auto formalParamValues = fnEntryBlock->getArguments();
    for (auto &v : pFunctionProto->input()) {
      BindOnnxName(v, formalParamValues[formal_num++]);
    }

    for (auto &fb_node : pFunctionProto->node()) {
      ImportNode(fb_node);
    }

    // Create a return operation to return all output tensors.
    llvm::SmallVector<mlir::Value, 4> ret_vals;
    for (auto &v : pFunctionProto->output()) {
      ret_vals.push_back(LookupOnnxName(v));
    }
    builder_.create<mlir::ReturnOp>(UnknownLoc(), ret_vals);

    // Restore caller context
    frontend_symbols_.popScope(func_name_prefix);
    builder_.restoreInsertionPoint(prev_ip);

    // Generate call statement
    auto op = builder_.create<CallOp>(UnknownLoc(), funcOp, operands);
    int result_num = 0;
    for (auto &v : node.output()) {
      BindOnnxName(v, op.getResult(result_num++));
    }

    return true;
  }

  void ImportCustomNode(const onnx::NodeProto &node) {
    if (!TryImportFunctionCallNode(node)) {
      mlir::emitWarning(UnknownLoc(),
          "Could not find op importer: assuming this "
          "represents a custom operator.");

      llvm::StringRef opName = node.op_type();
      int nOps = node.input().size();
      auto funcName = opName.str();
      std::vector<mlir::Type> outputTypes;
      std::vector<mlir::Value> inputs;
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

      for (const auto &item : node.output())
        ++nOut;

      buildOutputAndOperation<mlir::ONNXCustomOp>(
          node, inputs, nIn, nOut, attributes);
    }
  }

  void ImportNode(const onnx::NodeProto &node) {
    llvm::StringRef opName = node.op_type();

    // look up handler for the opName. If not found, create a node
    // for a custom op, and issue a warning.
    auto handler = import_handler_map_.find(opName.str());
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
    mlir::Value tensor_val =
        frontend_symbols_.GetTensorByOnnxName(output.name());
    if (output.type().value_case() == onnx::TypeProto::kTensorType) {
      if (output.type().tensor_type().has_shape()) {
        tensor_val.setType(ImportTensorType(output));
      }
    }
    ret_types.emplace_back(tensor_val.getType());
    ret_vals.push_back(tensor_val);
  }

  /*!
   * Import ONNX main computation graph.
   * @param graph onnx graph proto.
   * @return A function corresponding to the imported computation graph.
   */
  mlir::FuncOp importGraph(const onnx::GraphProto &graph) {
    const std::string &name = "main_graph";
    auto mainFunc = mlir::FuncOp::create(UnknownLoc(), name,
        /*type=*/builder_.getFunctionType({}, {}), /*attrs=*/{});
    module_.push_back(mainFunc);
    // Create and set insertion point to entry block.
    mainFunc.body().push_back(new Block);
    builder_.setInsertionPointToStart(&mainFunc.body().back());

    auto funcType = importGraph(graph, /*region=*/mainFunc.body(),
        /*op=*/mainFunc.getOperation(), /*useStdReturn=*/true);
    mainFunc.setType(funcType);

    // Emit entry point op describing inference function signature.
    auto entryPoint = mlir::ONNXEntryPointOp::create(UnknownLoc(), mainFunc,
        /*numInputs=*/funcType.getNumInputs(),
        /*numOutputs=*/funcType.getNumResults());
    module_.push_back(entryPoint);

    return mainFunc;
  }
}; // class FrontendGenImpl
} // namespace detail
} // namespace onnx_mlir

namespace onnx_mlir {

void ImportFrontendModelFile(std::string model_fname,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module,
    ImportOptions options) {
  onnx::ModelProto model;
  std::fstream input(model_fname, std::ios::in | std::ios::binary);

  auto parse_success = model.ParseFromIstream(&input);
  assert(parse_success && "Onnx Model Parsing Failed.");

  ImportFrontendModel(model, context, module, options);
}

void ImportFrontendModel(const onnx::ModelProto &model,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module,
    ImportOptions options) {

  detail::FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model, options);
}

} // namespace onnx_mlir
