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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Builder/ImportONNXUtils.hpp"
#include "src/Builder/ModelInputShaper.hpp"
#include "src/Builder/SymbolTable.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
SUPPRESS_WARNINGS_POP

#include <google/protobuf/util/json_util.h>

#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "frontend_dialect_transformer"

using namespace mlir;

namespace onnx_mlir {

namespace {

/// We consider opset < 6 is old. Users will see a warning if their model
/// contains ops of old opset.
constexpr int32_t MINIMUM_SUPPORTED_OPSET = 6;

using OpsetImportsMap = std::unordered_map<std::string, int>;

using AttrMap = std::unordered_map<std::string, const onnx::AttributeProto *>;

using onnx::shape_inference::ModelLocalFunctionsMap;

// -------------------------------------------------------------------------- //
// Code from third_party/onnx/onnx/shape_inference/implementation.cc:

// Either ModelProto or FunctionProto.
template <class T>
OpsetImportsMap GetOpsetImportsFromProto(const T &proto) {
  OpsetImportsMap opset_imports;
  for (const auto &opset_import : proto.opset_import()) {
    opset_imports[opset_import.domain()] = opset_import.version();
  }
  return opset_imports;
}

std::string GetModelLocalFunctionsMapIdentifier(
    const std::string &domain, const std::string &func_name) {
  return domain + ":" + func_name;
}

ModelLocalFunctionsMap GetModelLocalFunctions(const onnx::ModelProto &m) {
  ModelLocalFunctionsMap model_local_functions_by_id;
  for (const auto &function_proto : m.functions()) {
    model_local_functions_by_id.insert(
        {GetModelLocalFunctionsMapIdentifier(
             function_proto.domain(), function_proto.name()),
            &function_proto});
  }
  return model_local_functions_by_id;
}

void replaceAttrRefs(onnx::GraphProto &graph, const AttrMap &attr_map);

void replaceAttrRefs(onnx::NodeProto &n, const AttrMap &attr_map) {
  auto &attributes = *n.mutable_attribute();
  for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
    auto &attr = *attr_iter;
    if (!attr.ref_attr_name().empty()) {
      // Attribute-references must be replaced by the corresponding
      // attribute-value in the call-node if the call-node contains the
      // attribute. Otherwise, this attribute must be removed.
      auto entry = attr_map.find(attr.ref_attr_name());
      if (entry != attr_map.cend()) {
        // Copy value of attribute, but retain original name:
        std::string name = attr.name();
        attr = *(entry->second);
        attr.set_name(name);
      } else {
        attr_iter = attributes.erase(attr_iter);
        continue;
      }
    }
    // Subgraphs must be recursively processed.
    if (attr.has_g()) {
      replaceAttrRefs(*attr.mutable_g(), attr_map);
    }
    for (auto &graph : *attr.mutable_graphs()) {
      replaceAttrRefs(graph, attr_map);
    }
    ++attr_iter;
  }
}

void replaceAttrRefs(onnx::GraphProto &graph, const AttrMap &attr_map) {
  for (auto &n : *graph.mutable_node()) {
    replaceAttrRefs(n, attr_map);
  }
}

// End of copied code from third_party/onnx.
// -------------------------------------------------------------------------- //

} // namespace

namespace detail {

using ValueSymbolMapping = SymbolMapping<Value>;
using SymbolToOnnxTypeMapping = SymbolMapping<onnx::TypeProto>;

class FrontendGenImpl {
public:
  explicit FrontendGenImpl(MLIRContext &context)
      : context_(context), builder_(&context) {
    module_ = ModuleOp::create(UnknownLoc::get(&context));
    InitHandlerMap();
  }

  ModuleOp ImportONNXModel(
      const onnx::ModelProto &model, ImportOptions options) {
    options_ = options;
    modelInputShaper_.setShapeInformation(options_.shapeInformation);
    opset_map_ = GetOpsetImportsFromProto(model); // Which opsets to use.
    in_model_functions_ = GetModelLocalFunctions(model);
    importGraph(model.graph());
    if (options_.verboseOutput) {
      llvm::outs()
          << "The ONNX model has " << num_of_parameters_
          << " elements in its initializers. This value would be close to and "
             "greater than the number of parameters in the model. Because "
             "there is no way to exactly count the number of parameters, this "
             "value can be used to have a rough idea of the number of "
             "parameters in the model.\n";
    }
    return module_;
  }

private:
  ImportOptions options_;
  MLIRContext &context_;
  ModuleOp module_;
  OpBuilder builder_;

  // onnxop: list of versions supported by onnx-mlir for dialect
  std::map<std::string, std::vector<int>> op_dialect_version_map_;
  // onnxop: list of versions for dialect
  std::map<std::string, std::vector<int>> op_opsets_map_;
  // onnxop: the top version in third_part/onnx
  std::map<std::string, int> op_dialect_top_version_map_;

  // mapping between string name and symbol
  ValueSymbolMapping frontend_symbols_;

  // Keep shape information set by users.
  ModelInputShaper modelInputShaper_;

  using ImportHandlerType = void (onnx_mlir::detail::FrontendGenImpl::*)(
      const onnx::NodeProto &);

  std::map<std::string, ImportHandlerType> import_handler_map_;

  // The total number of elements in all initializers. This value is a rough
  // counter of the number of parameters in a model.
  int64_t num_of_parameters_ = 0;

  // onnx_type_map: a map from ONNX tensor name to ONNX TypeProto.
  SymbolToOnnxTypeMapping onnx_type_map;

  // opset_map_ is the internal (map) representation of ModelProto::opset_import
  // It maps each domain (e.g., "ai.onnx.ml") to the specific version of that
  // opset used by this model.
  OpsetImportsMap opset_map_;

  ModelLocalFunctionsMap in_model_functions_;

  Location UnknownLoc() const { return UnknownLoc::get(&context_); }

  Location ImportLoc(const onnx::NodeProto &node) {
    if (node.has_name()) {
      // Use the the node name as Location.
      return NameLoc::get(builder_.getStringAttr(node.name()));
    } else {
      return UnknownLoc();
    }
  }

  Value createNoneValue() {
    return builder_.create<ONNXNoneOp>(UnknownLoc()).getResult();
  }

  Value createConstantValue(ElementsAttr value, Location loc) {
    OnnxBuilder createONNX(builder_, loc);
    return createONNX.constant(value);
  }

  Value createConstantValue(ElementsAttr value) {
    return createConstantValue(value, UnknownLoc());
  }

  void AddValueInfo(const onnx::ValueInfoProto &vi, bool allowExist = false) {
    if (allowExist && onnx_type_map.ContainsKey(vi.name()))
      return;
    onnx_type_map.AddMapping(vi.name(), vi.type());
  }

  void BindOnnxName(const std::string &onnx_name, Value symbol) {
    frontend_symbols_.AddMapping(onnx_name, symbol);
  }

  Value LookupOnnxName(const std::string &onnx_name) {
    const Value *valuePtr = frontend_symbols_.GetByOnnxName(onnx_name);
    return *valuePtr;
  }

  static onnx::TypeProto fromMlirToONNXType(Type mlirType) {
    onnx::TypeProto onnxType;
    if (mlir::isa<NoneType>(mlirType)) {
      // Done: Uninitialized TypeProto onnxType represents NoneType.
    } else if (auto mlirTensorType = mlir::dyn_cast<TensorType>(mlirType)) {
      onnx::TypeProto::Tensor &onnxTensorType = *onnxType.mutable_tensor_type();
      onnxTensorType.set_elem_type(
          mlirTypeToOnnxType(mlirTensorType.getElementType()));
      if (mlirTensorType.hasRank()) {
        onnx::TensorShapeProto &onnxShape = *onnxTensorType.mutable_shape();
        for (int64_t mlirDim : mlirTensorType.getShape()) {
          onnx::TensorShapeProto::Dimension &onnxDim = *onnxShape.add_dim();
          onnxDim.set_dim_value(mlirDim);
        }
      }
    } else {
      // TODO: Convert optional and sequence types, if needed.
      llvm_unreachable("type's MLIR->ONNX conversion is unsupported");
    }
    return onnxType;
  }

  Value ImportTensor(const onnx::TensorProto &tensor) {
    mlir::ElementsAttr mlirAttr =
        onnxTensorProtoToElmAttr(&context_, options_.externalDataDir, tensor);
    // Use the tensor name as Location.
    auto loc =
        NameLoc::get(builder_.getStringAttr("Initializer_" + tensor.name()));
    Value initializer = createConstantValue(mlirAttr, loc);
    num_of_parameters_ += mlirAttr.getShapedType().getNumElements();
    return initializer;
  }

  /*!
   * Import an onnx tensor type by determining and returning its type
   * @param type_proto onnx tensor TypeProto.
   * @param dim_params a comma-separated string of dimIndex:dimParam.
   */
  Type ImportTensorType(
      const onnx::TypeProto &type_proto, std::string *dim_params = nullptr) {
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
        // Dim is a constant value.
        int dim_numeric_size = shape_proto.dim()[i].dim_value();
        assert(dim_numeric_size != 0 &&
               "Parsed an tensor with a dimension size of zero");
        if (dim_numeric_size > 0) {
          dims.push_back(dim_numeric_size);
        } else {
          // If dim_value < 0, then dim is parametric.
          dims.push_back(ShapedType::kDynamic);
        }
      } else if (dim_params && shape_proto.dim()[i].has_dim_param()) {
        // Dim is unknown but assigned a string ID that can be used to check
        // equality between unknown dimensions.
        if (!dim_params->empty())
          *dim_params += ",";
        *dim_params +=
            std::to_string(i) + ":" + shape_proto.dim()[i].dim_param();
        dims.push_back(ShapedType::kDynamic);
      } else {
        dims.push_back(ShapedType::kDynamic);
      }
    }

    llvm::ArrayRef<int64_t> tensor_dims(dims.data(), dims.size());
    return RankedTensorType::get(tensor_dims, elementType);
  }

  Type ImportSequenceType(
      const onnx::TypeProto &type_proto, std::string *dim_params = nullptr) {
    auto input_seq_type = type_proto.sequence_type();
    if (input_seq_type.has_elem_type()) {
      onnx::TypeProto elem_type = input_seq_type.elem_type();
      assert(elem_type.value_case() == onnx::TypeProto::kTensorType &&
             "expect tensor inside sequence type");
      Type mlir_elem_type = ImportTensorType(elem_type, dim_params);
      if (!mlir::isa<ShapedType>(mlir_elem_type))
        llvm_unreachable("Seq type is incorrect");
      Type seq_type =
          mlir::SeqType::get(mlir::cast<ShapedType>(mlir_elem_type), -1);
      return seq_type;
    }
    llvm_unreachable("unexpected type");
  }

  OptType ImportOptionalType(const onnx::TypeProto &type_proto) {
    auto input_opt_type = type_proto.optional_type();
    if (input_opt_type.has_elem_type()) {
      onnx::TypeProto elem_type = input_opt_type.elem_type();
      Type mlir_elem_type = ImportType(elem_type);
      return mlir::OptType::get(mlir_elem_type);
    }
    llvm_unreachable("unexpected type");
  }

  Type ImportType(
      const onnx::TypeProto &type_proto, std::string *dim_params = nullptr) {
    switch (type_proto.value_case()) {
    case onnx::TypeProto::kTensorType:
      return ImportTensorType(type_proto, dim_params);
      break;
    case onnx::TypeProto::kSequenceType:
      return ImportSequenceType(type_proto, dim_params);
      break;
    case onnx::TypeProto::kOptionalType:
      return ImportOptionalType(type_proto);
      break;
    default:
      llvm_unreachable("unexpected type");
      break;
    }
  }

  std::optional<Type> ConvertOnnxType(const std::string &onnx_name) {
    if (options_.useOnnxModelTypes) {
      if (const onnx::TypeProto *onnxTypePtr =
              onnx_type_map.GetByOnnxName(onnx_name)) {
        return std::optional<Type>(ImportType(*onnxTypePtr));
      }
    }
    return std::optional<Type>();
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
          llvm::ArrayRef(attr.floats().data(), attr.floats().size()));
      break;
    case onnx::AttributeProto::INTS:
      mlirAttr = builder_.getI64ArrayAttr(
          llvm::ArrayRef(attr.ints().data(), attr.ints().size()));
      break;
    case onnx::AttributeProto::TENSOR:
      mlirAttr = onnxTensorProtoToElmAttr(
          &context_, options_.externalDataDir, attr.t());
      break;
    case onnx::AttributeProto::STRINGS: {
      llvm::SmallVector<StringRef, 4> vectorStringRef;
      for (const auto &item : attr.strings()) {
        vectorStringRef.push_back(llvm::StringRef(item));
      }
      mlirAttr = builder_.getStrArrayAttr(llvm::ArrayRef(vectorStringRef));
    } break;
    case onnx::AttributeProto::TYPE_PROTO:
      mlirAttr = TypeAttr::get(ImportType(attr.tp()));
      break;
    case onnx::AttributeProto::GRAPH:
      llvm_unreachable("Subgraph attribute is imported as regions.");
      break;
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

  // Generate a string vector from the dimParams option string
  void getInputDimParamsMapFromOption(std::string optionStr,
      std::map<int, std::string> &paramStrMap,
      std::string &paramStrForAllArgs) {
    std::stringstream paramStrStream(optionStr);
    std::string dimParamStr;
    while (std::getline(paramStrStream, dimParamStr, '|')) {
      size_t pos = dimParamStr.find(':');
      assert((pos > 0) && "invalid dimParams option string");
      int idx = stoi(dimParamStr.substr(0, pos));
      dimParamStr = dimParamStr.substr(pos + 1);
      std::replace(dimParamStr.begin(), dimParamStr.end(), '=', ':');
      if (idx < 0) // set all arguments
        paramStrForAllArgs = dimParamStr;
      else {
        paramStrMap[idx] = dimParamStr;
      }
    }
    return;
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
   * @param useReturn if set to true, will emit ONNXReturnOp as
   * terminator, otherwise, will use ONNXYieldOp as terminator.
   * @return function type corresponding to the subgraph input/output signature.
   */
  FunctionType importGraph(const onnx::GraphProto &graph, Region &region,
      Operation *op, bool useReturn) {
    frontend_symbols_.pushScope(graph.name());
    onnx_type_map.pushScope(graph.name());
    Block *entryBlock = &region.back();

    // Maintain a mapping between the parameter and its initializer.
    std::unordered_set<std::string> initializerNames;
    for (const auto &initializer : graph.initializer()) {
      BindOnnxName(initializer.name(), ImportTensor(initializer));
      initializerNames.insert(initializer.name());
    }

    // create a function for the graph
    // TODO:
    //  * get name and type for the function.
    //  * maintain a list of the defined graph
    llvm::SmallVector<Type, 4> argTypes;

    llvm::SmallVector<llvm::StringRef, 4> inputNames, outputNames;
    // Keep dim_param for each dynamic dimension of each input tensor.
    // In ONNX specification, two dynamic dimensions with the same dim_param
    // string would be the same at runtime.
    //
    // See https://github.com/onnx/onnx/blob/main/docs/IR.md for more
    // information about dim_param.
    llvm::SmallVector<std::string, 4> inputDimParams, outputDimParams;
    std::map<int, std::string> inputDimParamsFromOption;
    std::string inputDimParamsFromOptionForAllArgs;
    getInputDimParamsMapFromOption(options_.dimParams, inputDimParamsFromOption,
        inputDimParamsFromOptionForAllArgs);

    // Import the input tensor types that are not constant and not initialized.
    int inputIndex = 0;
    for (const auto &input : graph.input()) {
      AddValueInfo(input);
      if (initializerNames.count(input.name()) == 0) {
        inputNames.push_back(input.name());
        std::string dimParams = "";
        Type argTy = ImportType(input.type(), &dimParams);
        argTy = modelInputShaper_.reshape(inputIndex, argTy);
        // For each input tensor, use either all dimensions by the compiler
        // option OR all dimensions in the original onnx model. Dimensions
        // from the option and the model in a single input tensor are not
        // merged.
        if (inputDimParamsFromOption.find(inputIndex) !=
            inputDimParamsFromOption.end())
          inputDimParams.emplace_back(inputDimParamsFromOption[inputIndex]);
        else if (!inputDimParamsFromOptionForAllArgs.empty())
          inputDimParams.emplace_back(inputDimParamsFromOptionForAllArgs);
        else if (!dimParams.empty())
          inputDimParams.emplace_back(dimParams);

        argTypes.emplace_back(argTy);

        // numInputs is the number of graph inputs not contained within the
        // initializer
        ++inputIndex;
      }
    }

    // The compiler assumes the model is correct and doesn't try to do
    // exhaustive correctness checking of its own
    for (const auto &internal : graph.value_info()) {
      AddValueInfo(internal, true);
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
    for (const auto &input : graph.input()) {
      if (initializerNames.count(input.name()) == 0) {
        BindOnnxName(
            input.name(), entryBlock->getArguments()[entryBlockArgIdx]);
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
      std::string dimParams = "";
      ImportOutputTensor(output, retTys, retVals, &dimParams);
      if (!dimParams.empty())
        outputDimParams.emplace_back(dimParams);
    }

    if (useReturn)
      builder_.create<ONNXReturnOp>(UnknownLoc(), retVals);
    else
      // Create a return operation to return all ONNX output tensors.
      builder_.create<ONNXYieldOp>(UnknownLoc(), retVals);

    SmallVector<llvm::StringRef> inputDimParamsRefs, outputDimParamsRefs;
    for (uint64_t i = 0; i < inputDimParams.size(); ++i)
      inputDimParamsRefs.emplace_back(llvm::StringRef(inputDimParams[i]));
    for (uint64_t i = 0; i < outputDimParams.size(); ++i)
      outputDimParamsRefs.emplace_back(llvm::StringRef(outputDimParams[i]));
    if (!inputNames.empty())
      op->setAttr("input_names", builder_.getStrArrayAttr(inputNames));
    if (!outputNames.empty())
      op->setAttr("output_names", builder_.getStrArrayAttr(outputNames));
    if (!inputDimParamsRefs.empty())
      op->setAttr(
          "input_dim_params", builder_.getStrArrayAttr(inputDimParamsRefs));
    if (!outputDimParamsRefs.empty())
      op->setAttr(
          "output_dim_params", builder_.getStrArrayAttr(outputDimParamsRefs));

    frontend_symbols_.popScope(graph.name());
    onnx_type_map.popScope(graph.name());
    return builder_.getFunctionType(argTypes, retTys);
  }

  void ImportNodeGeneric(const onnx::NodeProto &node) {
    std::vector<Value> inputs;
    for (const auto &item : node.input()) {
      if (const Value *valuePtr = frontend_symbols_.GetByOnnxName(item)) {
        inputs.push_back(*valuePtr);
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

    auto op = builder_.create(result);
    for (int i = 0; i < node.output().size(); i++) {
      auto r = op->getResult(i);
      frontend_symbols_.AddMapping(node.output()[i], r);
    }
  }

  static constexpr int MAX_NUM_TYPES = 30;

  // clang-format off
  // Get these indices from TensorProto in
  // https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto#L481.
  // enum DataType {
  //     UNDEFINED = 0;
  //     // Basic types.
  //     FLOAT = 1;   // float
  //     UINT8 = 2;   // uint8_t
  //     INT8 = 3;    // int8_t
  //     UINT16 = 4;  // uint16_t
  //     INT16 = 5;   // int16_t
  //     INT32 = 6;   // int32_t
  //     INT64 = 7;   // int64_t
  //     STRING = 8;  // string
  //     BOOL = 9;    // bool
  //
  //     // IEEE754 half-precision floating-point format (16 bits wide).
  //     // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  //     FLOAT16 = 10;
  //
  //     DOUBLE = 11;
  //     UINT32 = 12;
  //     UINT64 = 13;
  //     COMPLEX64 = 14;     // complex with float32 real and imaginary
  //     components COMPLEX128 = 15;    // complex with float64 real and
  //     imaginary components
  //
  //     // Non-IEEE floating-point format based on IEEE754 single-precision
  //     // floating-point number truncated to 16 bits.
  //     // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  //     BFLOAT16 = 16;
  //
  //     // Non-IEEE floating-point format based on papers
  //     // FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
  //     // 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
  //     // Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
  //     // The computation usually happens inside a block quantize / dequantize
  //     // fused by the runtime.
  //     FLOAT8E4M3FN = 17;    // float 8, mostly used for coefficients, supports nan, not inf
  //     FLOAT8E4M3FNUZ = 18;  // float 8, mostly used for coefficients, supports nan, not inf, no negative zero
  //     FLOAT8E5M2 = 19;      // follows IEEE 754, supports nan, inf, mostly used for gradients
  //     FLOAT8E5M2FNUZ = 20;  // follows IEEE 754, supports nan, inf, mostly used for gradients, no negative zero
  //
  //     // Future extensions go here.
  //   }
  //
  // They must be consistent witn onnx_types in utils/gen_onnx_mlir.py
  // onnx_types = (
  //     'undefined', 'float', 'uint8', 'int8', 'uint16', 'int16', 'int32',
  //     'int64', 'string', 'bool', 'float16', 'double', 'uint32', 'uint64',
  //     'complex64', 'complex128',
  //     'bfloat16', 'float8e4m3fn', 'float8e4m3fnuz', 'float8e5m2', 'float8e5m2fnuz'
  // )
  // clang-format on
  Type buildTypeFromIndex(int index) {
    switch (index) {
    case 1:
      return builder_.getF32Type();
    case 2:
      return builder_.getIntegerType(8, /*isSigned=*/false);
    case 3:
      return builder_.getIntegerType(8);
    case 4:
      return builder_.getIntegerType(16, /*isSigned=*/false);
    case 5:
      return builder_.getIntegerType(16);
    case 6:
      return builder_.getIntegerType(32);
    case 7:
      return builder_.getIntegerType(64);
    case 8:
      return mlir::ONNXStringType::get(builder_.getContext());
    case 9:
      return builder_.getI1Type();
    case 10:
      return builder_.getF16Type();
    case 11:
      return builder_.getF64Type();
    case 12:
      return builder_.getIntegerType(32, /*isSigned=*/false);
    case 13:
      return builder_.getIntegerType(64, /*isSigned=*/false);
    case 14: {
      std::vector<Type> typeTuple(2);
      typeTuple.push_back(builder_.getF32Type());
      typeTuple.push_back(builder_.getF32Type());
      return builder_.getTupleType(llvm::ArrayRef<Type>(typeTuple));
    }
    case 15: {
      std::vector<Type> typeTuple(2);
      typeTuple.push_back(builder_.getF64Type());
      typeTuple.push_back(builder_.getF64Type());
      return builder_.getTupleType(llvm::ArrayRef<Type>(typeTuple));
    }
    case 16:
      return builder_.getBF16Type();
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
        inputs.emplace_back(createNoneValue());
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
        outputTypes.emplace_back(onnxModelType.value());
      } else {
        unsigned int j = i;
        // Variadic output is a single ODS result.
        if (variadicOut)
          j = 0;
        if (j < outputMap.size() && outputMap[j] >= MAX_NUM_TYPES) {
          // Mapping gives a connection with an input.
          Type inputType = inputs[outputMap[j] - MAX_NUM_TYPES].getType();
          if (mlir::isa<TensorType>(inputType)) {
            Type elementType =
                mlir::cast<TensorType>(inputType).getElementType();
            auto outType = UnrankedTensorType::get(elementType);
            outputTypes.emplace_back(outType);
          } else {
            outputTypes.push_back(inputType);
          }
        } else if (j < outputMap.size() && outputMap[j] != -1) {
          // Mapping gives a direct type.
          Type elementType = buildTypeFromIndex(outputMap[j]);
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
    T op = builder_.create<T>(ImportLoc(node), outputTypes, inputs, attributes);
    // Type inference for results.
    for (const auto &attr : node.attribute()) {
      if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH) {
        OperationName opName = op->getName();
        assert(opName.hasInterface<HasOnnxSubgraphOpInterface>() &&
               "Op contains subgraph attributes but does not "
               "implement HasOnnxSubgraphOpInterface interface.");
        auto opWithSubgraph =
            mlir::cast<HasOnnxSubgraphOpInterface>(op.getOperation());
        auto regionIdx = opWithSubgraph.getSubgraphRegionIdx(attr.name());
        auto &region = op->getRegion(regionIdx);
        region.push_back(new Block);
        OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(&region.back());
        importGraph(attr.g(), region, op, false);
        if (!options_.useOnnxModelTypes) {
          // Output types are propagated from region terminator to op results
          // in opWithTypeInference logic below.
          assert(opName.hasTrait<ResultTypeInferenceOpInterface::Trait>() &&
                 "Subgraph ops must implement ResultTypeInferenceOpInterface");
        }
      }
    }
    if (auto opWithTypeInference =
            mlir::dyn_cast<ResultTypeInferenceOpInterface>(op.getOperation())) {
      auto outTypes = opWithTypeInference.resultTypeInference();
      for (int i = 0; i < node.output().size(); i++) {
        OpResult result = op->getResult(i);
        if (!options_.useOnnxModelTypes || isa<NoneType>(result.getType()))
          result.setType(outTypes[i]);
      }
    }

    for (const auto &[i, output] : llvm::enumerate(node.output())) {
      // Skip the output with empty name, which is used as a placeholder
      // in multiple outputs.
      // Found in models. Not sure about the specification.
      if (output != "")
        frontend_symbols_.AddMapping(output, op->getResult(i));
    }
  }

  void getNodeInputs(const onnx::NodeProto &node, std::vector<Value> &inputs) {
    for (const auto &item : node.input()) {
      if (item.empty()) {
        inputs.emplace_back(createNoneValue());
      } else {
        if (const Value *valuePtr = frontend_symbols_.GetByOnnxName(item)) {
          inputs.push_back(*valuePtr);
        }
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
    auto inputType = mlir::cast<TensorType>(inputs[0].getType());
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
      if (attr.name() == "to") {
        auto mlir_type = convertONNXTypeToMLIRType(
            builder_, static_cast<onnx::TensorProto_DataType>(attr.i()));
        Attribute mlirAttr = TypeAttr::get(mlir_type);
        attributes.push_back(builder_.getNamedAttr(attr.name(), mlirAttr));
      } else {
        NamedAttribute na = convertOnnxAttributeProtoToMlirNamedAttribute(attr);
        attributes.push_back(na);
      }
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
        inputs.emplace_back(createNoneValue());
      } else {
        if (const Value *valuePtr = frontend_symbols_.GetByOnnxName(item)) {
          inputs.push_back(*valuePtr);
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
      if (const Value *valuePtr = frontend_symbols_.GetByOnnxName(item)) {
        inputs.push_back(*valuePtr);
      }
    }

    // If ratio is not specified, the default value is 0.5
    if (nOps < 2) {
      llvm::SmallVector<int64_t, 1> dims;
      dims.push_back(1);
      llvm::SmallVector<float, 1> values;
      values.push_back(0.5);
      Type elementType = builder_.getF32Type();
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
      Value constantResult = createConstantValue(constantDenseAttribute);
      inputs.push_back(constantResult);
    }

    // If training_mode is not specified, the default value is false
    if (nOps < 3) {
      llvm::SmallVector<int64_t, 1> dims;
      dims.push_back(1);
      llvm::SmallVector<bool, 1> values;
      values.push_back(false);
      Type elementType = builder_.getIntegerType(1);
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
      Value constantResult = createConstantValue(constantDenseAttribute);
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

      std::vector<Value> inputs;
      getNodeInputs(node, inputs);
      Type elementType =
          mlir::cast<TensorType>(inputs[0].getType()).getElementType();

      llvm::SmallVector<Attribute, 2> values(
          1, builder_.getZeroAttr(elementType));

      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));

      // Use the special builder defined in ONNXOp.td.inc.
      Value constantResult = createConstantValue(constantDenseAttribute);
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
      if (const Value *valuePtr =
              frontend_symbols_.GetByOnnxName(item.value())) {
        inVals[item.index()] = *valuePtr;
      } else {
        assert(false && "Unknown input");
      }
    }

    // Data input is imported but starts, ends, axes, and steps may come from
    // attributes, and need to be created as constant ops.
    const Type elementType = builder_.getIntegerType(64);
    const auto attributes = ImportNodeAttributes(node);
    for (auto attr : attributes) {
      if (auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue())) {
        const auto tensorType =
            RankedTensorType::get({(int64_t)arrayAttr.size()}, elementType);
        auto constantDenseAttribute =
            DenseElementsAttr::get(tensorType, arrayAttr.getValue());
        Value constantValue = createConstantValue(constantDenseAttribute);

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
    inVals[3] = inVals[3] == nullptr ? createNoneValue() : inVals[3];
    inVals[4] = inVals[4] == nullptr ? createNoneValue() : inVals[4];

    int nIn = ONNXSliceOp::getNumberOfOperands();
    int nOut = ONNXSliceOp::getNumberOfResults();
    const auto in = std::vector<Value>(inVals.begin(), inVals.end());

    buildOutputAndOperation<ONNXSliceOp>(node, in, nIn, nOut, attributes);
  }

  const onnx::OpSchema *GetOpSchema(const onnx::NodeProto &node) {
    auto &domain = node.domain();
    auto version_it = opset_map_.find(domain);
    if (version_it == opset_map_.end())
      return nullptr;
    int version = version_it->second;
    return onnx::OpSchemaRegistry::Schema(node.op_type(), version, domain);
  }

  std::string GetImportVersionOfNode(const onnx::NodeProto &node) {
    auto current_opset_it = opset_map_.find(node.domain());
    if (current_opset_it == opset_map_.end())
      return "";

    const int current_opset = current_opset_it->second;

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": Importing ONNX"
                            << node.op_type() << " (" << node.name() << ")"
                            << ", Opset: " << current_opset << "\n");

    const auto supported_opset_list_it =
        op_dialect_version_map_.find(node.op_type());
    const auto opset_list_it = op_opsets_map_.find(node.op_type());

    // Custom ops may not be present in op_dialect_version_map_. If no version
    // info is found, treat as unversioned (no renaming).
    if (supported_opset_list_it == op_dialect_version_map_.end() ||
        opset_list_it == op_opsets_map_.end())
      return "";

    // To determine the opset version for a node/op:
    // 1: Determine the latest valid opset version. This is the newest version
    // in this opset-version-map that is older or equal to the current graph
    // opset. 2:_ Select the newest version from the versions supported by
    // onnx-mlir that is equal or newer to the latest valid opset version. This
    // allows it to skip over opset versions, that have a newer backwards
    // compatible version.
    // Example:
    // Versions in onnx and supported by onnx-mlir:[3, 5].
    // Graph opset version to node version: 3 -> 3, 4 -> 3, 5 -> 5
    //
    // Versions in onnx: [7, 9, 10].
    // Version 10 is backwards compatible to version 9.
    // Version supported by onnx-mlir: [7, 10].
    // Graph opset version to node version: 7 -> 7, 8 -> 7, 9 -> 10, 10 -> 10

    // Get the newest opset version for the op that is older or equal to the
    // model opset version. Use the oldest version as fallback
    int newestValidOpsetVersion = opset_list_it->second.back();
    for (int opset : opset_list_it->second) {
      if (opset <= current_opset) {
        newestValidOpsetVersion = opset;
        break;
      }
    }

    const auto supported_opset_list = supported_opset_list_it->second;

    // A new opset is added to onnx-mlir when it becomes incompatible.
    // All opset newest than the last opset should use the last opset(version)
    if (node.domain().compare("ai.onnx.ml") != 0 &&
        newestValidOpsetVersion < supported_opset_list.back() &&
        newestValidOpsetVersion < MINIMUM_SUPPORTED_OPSET)
      llvm::outs() << "\nWarning: ONNX " << node.op_type()
                   << " in your model is using Opset "
                   << newestValidOpsetVersion
                   << ", which is quite old. Please consider regenerating your "
                      "model with a newer Opset.\n\n";

    if (newestValidOpsetVersion >= supported_opset_list.front())
      return ""; // Use the newest version

    // Iterate over all supported opsets, starting with the oldest version.
    // Select the oldest version that is the same or newer as the version the
    // model uses. Special case: The newest supported version has no version
    // suffix
    for (int opset : llvm::reverse(supported_opset_list)) {
      if (opset >= newestValidOpsetVersion &&
          opset != supported_opset_list.front()) {
        LLVM_DEBUG(
            llvm::dbgs() << DEBUG_TYPE << ":   - use Opset " << opset << "\n");
        return "V" + std::to_string(opset);
      }
    }
    return "";
  }

  // Is called with either (1) a model local function or (2) an OpSchema with
  // a function decomposition.
  // Case 1: modelLocalFunction is the model local function, schema is null.
  // Case 2: modelLocalFunction is null, schema has the function decomposition.
  void ImportFunctionCallNode(const onnx::NodeProto &node,
      const onnx::OpSchema *schema,
      const onnx::FunctionProto *modelLocalFunction) {
    assert((schema != nullptr) != (modelLocalFunction != nullptr) &&
           "pass either schema or modelLocalFunction, not both");

    // Collect the input values and their onnx types:
    std::vector<Value> inputs;
    std::vector<onnx::TypeProto> inputOnnxTypes;
    for (const auto &input_name : node.input()) {
      if (input_name.empty()) {
        inputs.emplace_back(createNoneValue());
        // Uninitialized TypeProto represents NoneType.
        inputOnnxTypes.emplace_back();
      } else {
        Value value = inputs.emplace_back(LookupOnnxName(input_name));
        if (const onnx::TypeProto *onnxType =
                onnx_type_map.GetByOnnxName(input_name)) {
          inputOnnxTypes.push_back(*onnxType);
        } else {
          inputOnnxTypes.emplace_back(fromMlirToONNXType(value.getType()));
        }
      }
    }

    // Get ONNX function:
    onnx::FunctionProto functionProto;
    if (schema) { // Function decomposition.
      if (schema->HasFunction()) {
        functionProto = *schema->GetFunction(); // Context-independent function.
      } else {
        assert(schema->HasContextDependentFunction() &&
               "must have context dependent function absent a context "
               "independent function");
        // Generate a context-dependent function body:
        onnx::FunctionBodyBuildContextImpl onnxFunContext(node, inputOnnxTypes);
        if (!schema->BuildContextDependentFunction(
                onnxFunContext, functionProto))
          llvm_unreachable("failed to generate context dependent function");
      }
    } else {
      functionProto = *modelLocalFunction; // Model local function.
    }

    assert(node.input_size() <= functionProto.input_size() &&
           "more caller inputs than function arguments");
    assert(node.output_size() <= functionProto.output_size() &&
           "more caller outputs than function results");

    // Construct a graph with the function parameters and body so that
    // we can call onnx::shape_inference::InferShapes(&graph,..) which
    // will populate grap.value_info() with more information than we can
    // get from InferShapeForFunctionNode(functionProto,..).
    onnx::GraphProto graph;

    for (int i = 0; i < functionProto.input_size(); ++i) {
      onnx::ValueInfoProto *info = graph.add_input();
      info->set_name(functionProto.input(i));
      // Set type if known, otherwise it's uninitialized which means NoneType.
      if (i < node.input_size())
        *info->mutable_type() = inputOnnxTypes[i];
    }

    for (int i = 0; i < functionProto.output_size(); ++i) {
      onnx::ValueInfoProto *info = graph.add_output();
      info->set_name(functionProto.output(i));
      // Set type if known, otherwise it's uninitialized which means NoneType.
      if (i < node.output_size()) {
        if (const onnx::TypeProto *onnxType =
                onnx_type_map.GetByOnnxName(node.output(i))) {
          *info->mutable_type() = *onnxType;
        }
      }
    }

    *graph.mutable_node() = std::move(functionProto.node());

    // Substitute caller attributes in graph nodes:
    AttrMap caller_attr_map;
    for (const onnx::AttributeProto &attr : node.attribute())
      caller_attr_map[attr.name()] = &attr;
    AttrMap attr_map;
    for (const std::string &attrName : functionProto.attribute()) {
      auto it = caller_attr_map.find(attrName);
      if (it != caller_attr_map.end())
        attr_map[attrName] = it->second;
    }
    for (const onnx::AttributeProto &attr : functionProto.attribute_proto()) {
      const std::string &attrName = attr.name();
      auto it = caller_attr_map.find(attrName);
      if (it != caller_attr_map.end())
        attr_map[attrName] = it->second;
      else
        attr_map[attrName] = &attr;
    }
    replaceAttrRefs(graph, attr_map);

    OpsetImportsMap function_opset_map =
        GetOpsetImportsFromProto(functionProto);

    // Populates graph.value_info().
    onnx::shape_inference::InferShapes(&graph, function_opset_map,
        onnx::OpSchemaRegistry::Instance(),
        /*options=*/{}, in_model_functions_);

    // Save caller context, while generating function body.
    ModelLocalFunctionsMap callerModelFunctions;
    if (schema) {
      // Function decompositions cannot access model local functions.
      callerModelFunctions = std::move(in_model_functions_);
    }
    OpsetImportsMap callerOpsetMap(std::move(opset_map_));
    ValueSymbolMapping callerScope(std::move(frontend_symbols_));
    SymbolToOnnxTypeMapping callerTypeMap(std::move(onnx_type_map));

    std::vector<Value> outputs;

    // TODO: Reuse importGraph() logic.
    {
      opset_map_ = std::move(function_opset_map);

      std::string scopeName =
          node.name() + ":" + node.op_type() + ":" + functionProto.name();
      frontend_symbols_.pushScope(scopeName);
      onnx_type_map.pushScope(scopeName);

      for (const auto &input : graph.input())
        AddValueInfo(input);
      for (const auto &internal : graph.value_info())
        AddValueInfo(internal, true);
      for (const auto &output : graph.output()) {
        // Output tensor may be in input list
        AddValueInfo(output, true);
      }

      for (int i = 0; i < functionProto.input_size(); ++i) {
        const std::string &name = functionProto.input(i);
        // Due to missing trailing optional parameters
        // node may have fewer inputs than functionProto.
        Value value = i < node.input_size() ? inputs[i] : createNoneValue();
        BindOnnxName(name, value);
      }

      for (auto &fb_node : graph.node()) {
        ImportNode(fb_node);
      }

      for (auto &name : functionProto.output()) {
        // Skip missing optional outputs: they are not mapped.
        if (const Value *valuePtr = frontend_symbols_.GetByOnnxName(name)) {
          outputs.push_back(*valuePtr);
        }
      }

      frontend_symbols_.popScope(scopeName);
      onnx_type_map.popScope(scopeName);
    }

    // Restore caller context.
    if (schema) {
      in_model_functions_ = std::move(callerModelFunctions);
    }
    opset_map_ = std::move(callerOpsetMap);
    frontend_symbols_ = std::move(callerScope);
    onnx_type_map = std::move(callerTypeMap);

    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string &name = node.output(i);
      Value value = outputs[i];
      BindOnnxName(name, value);
    }
  }

  void ImportCustomNode(const onnx::NodeProto &node) {
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
    // ToFix: The type inference may go wrong if the element type of the output
    // of CustomOp is not the same as the first input.
    buildOutputAndOperation<ONNXCustomOp>(node, inputs, nIn, nOut, attributes);
  }

  void ImportNode(const onnx::NodeProto &node) {
    std::string opName = node.op_type() + GetImportVersionOfNode(node);
    auto handler = import_handler_map_.find(opName);
    std::vector<std::string> funcs = options_.functionsToDecompose;
    if (!(std::find(funcs.begin(), funcs.end(), opName) != funcs.end())) {
      if (handler != import_handler_map_.end()) {
        // It's a regular op with a registered handler.
        (this->*(handler->second))(node);
        return;
      }
    }

    const onnx::OpSchema *schema = GetOpSchema(node);
    if (schema &&
        (schema->HasFunction() || schema->HasContextDependentFunction())) {
      // The op has a function decomposition.
      ImportFunctionCallNode(node, schema, /*modelLocalFunction=*/nullptr);
      return;
    }

    auto model_function = in_model_functions_.find(
        GetModelLocalFunctionsMapIdentifier(node.domain(), node.op_type()));
    if (model_function != in_model_functions_.end()) {
      ImportFunctionCallNode(node, /*schema=*/nullptr, model_function->second);
      return;
    }

    emitWarning(UnknownLoc(), "Could not find op importer: assuming this "
                              "represents a custom operator.");
    ImportCustomNode(node);
  }

  void InitHandlerMap() {
#include "src/Builder/OpBuildTable.inc"
  }

  /*!
   * Import output value, by doing the following:
   * - Add the type of this output tensor to a list of
   *   types representing return types of this graph function.
   * - Add this output value to the list of values
   *   to be returned by the function representing computation graph.
   * @param output onnx output ValueInfoProto.
   * @param ret_types a vector of types representing graph's output types.
   * @param ret_vals a vector of mlir Value representing graph's output.
   * @param dim_params a comma-separated string of dimIndex:dimParam.
   */
  void ImportOutputTensor(const onnx::ValueInfoProto &output,
      llvm::SmallVectorImpl<Type> &ret_types,
      llvm::SmallVectorImpl<Value> &ret_vals,
      std::string *dim_params = nullptr) {
    const Value *valPtr = frontend_symbols_.GetByOnnxName(output.name());
    Value val = *valPtr;
    if (output.type().value_case() == onnx::TypeProto::kTensorType) {
      Type outTy = ImportType(output.type(), dim_params);
      if (std::getenv("IMPORTER_FORCE_DYNAMIC"))
        outTy = UnrankedTensorType::get(
            mlir::cast<TensorType>(outTy).getElementType());
      if (output.type().tensor_type().has_shape()) {
        val.setType(outTy);
      }
      ret_types.emplace_back(val.getType());
    } else {
      ret_types.emplace_back(ImportType(output.type(), dim_params));
    }
    ret_vals.push_back(val);
  }

  // Move function attributes for argument/result names and dim_params into
  // argument/result attributes.
  void moveFuncAttrsToArgAttrs(func::FuncOp funcOp,
      ArrayRef<std::string> funcAttrNames, ArrayRef<std::string> argAttrNames,
      bool isArg) {
    assert(funcAttrNames.size() == argAttrNames.size() &&
           "The number of attributes to move mismatched");
    Operation *op = funcOp.getOperation();
    size_t numOfArgs =
        (isArg) ? funcOp.getNumArguments() : funcOp.getNumResults();

    // Only move attributes that exists.
    SmallVector<ArrayAttr, 2> funcAttrsToMove;
    SmallVector<std::string, 2> targetArgAttrNames;
    for (size_t i = 0; i < funcAttrNames.size(); ++i) {
      ArrayAttr attr = op->getAttrOfType<ArrayAttr>(funcAttrNames[i]);
      if (!attr)
        continue;
      funcAttrsToMove.emplace_back(attr);
      targetArgAttrNames.emplace_back(argAttrNames[i]);
    }

    // Move function attributes to argument/result attributes.
    for (size_t i = 0; i < numOfArgs; ++i) {
      SmallVector<NamedAttribute, 2> argAttrs;
      for (size_t k = 0; k < funcAttrsToMove.size(); ++k) {
        if (i < funcAttrsToMove[k].size()) {
          auto name = mlir::cast<StringAttr>(funcAttrsToMove[k].getValue()[i]);
          if (name) {
            NamedAttribute namedAttr =
                builder_.getNamedAttr(argAttrNames[k], name);
            argAttrs.emplace_back(namedAttr);
          }
        }
      }
      if (!argAttrs.empty()) {
        if (isArg)
          funcOp.setArgAttrs(i, argAttrs);
        else
          funcOp.setResultAttrs(i, argAttrs);
      }
    }

    // Clean up the function attributes.
    for (std::string s : funcAttrNames)
      op->removeAttr(s);
  }

  /*!
   * Import ONNX main computation graph.
   * @param graph onnx graph proto.
   * @return A function corresponding to the imported computation graph.
   */
  func::FuncOp importGraph(const onnx::GraphProto &graph) {
    const std::string &name = "main_graph";
    auto mainFunc = func::FuncOp::create(UnknownLoc(), name,
        /*type=*/builder_.getFunctionType({}, {}), /*attrs=*/{});
    module_.push_back(mainFunc);
    // Create and set insertion point to entry block.
    mainFunc.getBody().push_back(new Block);
    builder_.setInsertionPointToStart(&mainFunc.getBody().back());

    auto funcType = importGraph(graph, /*region=*/mainFunc.getBody(),
        /*op=*/mainFunc.getOperation(), /*useReturn=*/true);
    mainFunc.setType(funcType);

    // Move function attributes for argument/result names and dim_params into
    // argument/result attributes.
    moveFuncAttrsToArgAttrs(mainFunc, {"input_names", "input_dim_params"},
        {"onnx.name", "onnx.dim_params"}, /*isArg=*/true);
    moveFuncAttrsToArgAttrs(mainFunc, {"output_names", "output_dim_params"},
        {"onnx.name", "onnx.dim_params"}, /*isArg=*/false);

    // Emit entry point op describing inference function signature.
    auto entryPoint = ONNXEntryPointOp::create(UnknownLoc(), mainFunc);
    module_.push_back(entryPoint);

    return mainFunc;
  }
}; // class FrontendGenImpl

} // namespace detail

bool ImportFrontendModelInternal(onnx::ModelProto &model, MLIRContext &context,
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

  if (options.allowSorting && !IsTopologicallySorted(model.graph())) {
    if (!SortGraph(model.mutable_graph())) {
      llvm::outs() << "The graph is not topologically sortable.\n";
      return false;
    }
  }

  // Note: when options.useOnnxModelTypes is true, the onnx::shape_inference
  // cannot handle non-onnx operations (represented as CustomOp in onnx-mlir)
  // Assertion error if the model contains a such operation.
  // onnx-mlir handles the CustomOp in a different way. It assumes the common
  // pattern that the element type of the output is the same
  // as the the first input. And later shape inference will use the
  // shape-inference-pattern attribute to perform shape inference on CustomOp.
  // The type assumption in the Importer may be incorrect and cause
  // trouble.
  // ToFix: the shape-inference-pattern should be added and used in Importer.

  // Did not do downward convert because support for BatchNorm is missing
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
  return true;
}

// Return 0 on success, error otherwise.
int ImportFrontendModelArray(const void *onnxBuffer, int size,
    MLIRContext &context, OwningOpRef<ModuleOp> &module,
    std::string *errorMessage, ImportOptions options) {
  onnx::ModelProto model;

  bool parse_success = model.ParseFromArray(onnxBuffer, size);
  if (!parse_success) {
    *errorMessage = "Unable to parse onnxBuffer";
    return InvalidOnnxFormat;
  }
  ImportFrontendModelInternal(model, context, module, options);
  return CompilerSuccess;
}

namespace {
int readAndStripComments(
    StringRef fname, std::string *errorMessage, std::string &contents) {
  contents.clear();
  auto buf = openInputFile(fname, errorMessage);
  if (!buf) {
    return InvalidInputFileAccess;
  }
  // Remove // comments, which are non-standard json and onnx text
  // but appear in lit tests in test/mlir/onnx/parse.
  for (llvm::line_iterator line(*buf, /*SkipBlanks=*/false), end; line != end;
       ++line) {
    if (line->ltrim(" \t").starts_with("//"))
      continue; // omit comment lines beginning with (whitespace and) //
    if (line->contains("//")) {
      // Not stripping end-of-line comments because there's no robust way to
      // distinguish them from valid uses of // in the json itself.
      llvm::errs() << "\nWarning: possible invalid end-of-line // comment in "
                      "json input file "
                   << fname.str() << ":" << line.line_number() << "\n\n";
    }
    contents.append(*line);
  }
  return CompilerSuccess;
}
} // namespace

// Return 0 on success, error otherwise.
int ImportFrontendModelFile(StringRef model_fname, MLIRContext &context,
    OwningOpRef<ModuleOp> &module, std::string *errorMessage,
    ImportOptions options) {
  onnx::ModelProto model;
  if (model_fname.ends_with(".onnxtext")) {
    std::string text;
    int ret = readAndStripComments(model_fname, errorMessage, text);
    if (ret != CompilerSuccess)
      return ret;

    onnx::OnnxParser parser(text.c_str());
    auto status = parser.Parse(model);
    if (!status.IsOK()) {
      *errorMessage = "ONNX Text Model Parsing Failed on " + model_fname.str() +
                      " with error '" + status.ErrorMessage() + "'";
      return InvalidOnnxFormat;
    }
  } else if (model_fname.ends_with(".json")) {
    std::string json;
    int ret = readAndStripComments(model_fname, errorMessage, json);
    if (ret != CompilerSuccess)
      return ret;

    auto status = google::protobuf::util::JsonStringToMessage(json, &model);
    if (!status.ok()) {
      *errorMessage = "Json Model Parsing Failed on " + model_fname.str() +
                      " with error '" + status.ToString() + "'";
      return InvalidOnnxFormat;
    }
  } else {
    bool parse_success;
    if (model_fname.str() == "-")
      parse_success = model.ParseFromIstream(&std::cin);
    else {
      std::fstream input(model_fname.str(), std::ios::in | std::ios::binary);
      // check if the input file is opened
      if (!input.is_open()) {
        *errorMessage = "Unable to open or access " + model_fname.str();
        return InvalidInputFileAccess;
      }
      parse_success = model.ParseFromIstream(&input);
    }
    if (!parse_success) {
      *errorMessage = "Onnx Model Parsing Failed on " + model_fname.str();
      return InvalidOnnxFormat;
    }
  }

  if (!ImportFrontendModelInternal(model, context, module, options)) {
    *errorMessage = "Onnx Model Import Failed on " + model_fname.str();
    return CompilerFailure;
  }

  return CompilerSuccess;
}

void ImportFrontendModel(const onnx::ModelProto &model, MLIRContext &context,
    OwningOpRef<ModuleOp> &module, ImportOptions options) {

  detail::FrontendGenImpl myONNXGen(context);
  module = myONNXGen.ImportONNXModel(model, options);
}

} // namespace onnx_mlir
