/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

#include "mlir/Dialect/Traits.h"

//===----------------------------------------------------------------------===//
// Unsupported Operations
//===---------------------------------------------------------------------===//

// Operations for which shape inference has not been implemented yet
// If you add the implementation for one op, move it out of this section
// Also please add test case in test/mlir/onnx/onnx_shape_inference.mlir
// Followed by the implementation of lowering to Krnl and
// Enable the corresponding node test in check-onnx-backend

#define NOT_IMPLEMENTED_INFER_SHAPES(T)                                        \
  mlir::LogicalResult mlir::T::inferShapes(                                    \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return emitOpError(                                                        \
        "op is not supported at this time. Please open an issue on "           \
        "https://github.com/onnx/onnx-mlir and/or consider contributing "      \
        "code. "                                                               \
        "Error encountered in shape inference.");                              \
  }

// Listed alphabetically.
NOT_IMPLEMENTED_INFER_SHAPES(ONNXAdagradOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXAdamOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXArrayFeatureExtractorOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXBatchNormalizationOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXBinarizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXBlackmanWindowOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXCastMapOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXCenterCropPadOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXClipV11Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXClipV12Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXClipV6Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXCol2ImOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXConcatFromSequenceOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXDetOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXDictVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXFeatureVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXGradientOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXGridSampleOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXGroupNormalizationOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXHammingWindowOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXHannWindowOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXImputerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXIsInfOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXLabelEncoderOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXLayerNormalizationOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXLinearClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXLinearRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXLpPoolOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMaxPoolOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMaxUnpoolOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMelWeightMatrixOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMishOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMomentumOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXMultinomialOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXNegativeLogLikelihoodLossOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXPadV11Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXPadV2Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXRandomUniformLikeOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXRandomUniformOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXResizeV10Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXResizeV11Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXSequenceMapOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXSVMClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXSVMRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXSoftmaxCrossEntropyLossOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXSTFTOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXStringNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXTfIdfVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXTreeEnsembleClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXTreeEnsembleRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXUniqueOp)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXUpsampleV7Op)
NOT_IMPLEMENTED_INFER_SHAPES(ONNXZipMapOp)

namespace {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Get a broadcasted type for RankedTensorType and MemRefType.
// Used in generated code in ONNXOps.cpp.inc included below.
//===----------------------------------------------------------------------===//
Type getBroadcastedRankedType(
    Type type1, Type type2, Type elementType = nullptr) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2, elementType);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Construct RankedTensorType(s).
    if (!elementType)
      elementType = type1.cast<MemRefType>().getElementType();
    RankedTensorType ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    RankedTensorType ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNXConstantOp custom assembly format print and parse.
// If the op has a sparse_value attr, it just prints its SparseElementsAttr:
//
//   onnx.Constant sparse<0, 1.000000e+00> : tensor<3xf32>
//
// and, if the op has a value attr, it just prints its DenseElementsAttr:
//
//   onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>
//
// provided the ElementsAttr type matches the op result type.
//
// TODO: Ensure constant result types always match their attributes because
//       constant folding needs that.
//
// In case of type mismatch or if the op has another attr than sparse_value or
// value or has no attr, the op prints an attribute dictionary followed by the
// op result type, just like the default assembly format.
//
//   onnx.Constant {value = dense<1024> : tensor<1xi64>} : tensor<*xi64>
//
//   onnx.Constant {value_int = 1 : si64} : tensor<i64>
//
//   onnx.Constant : tensor<64xf32>
//
//===----------------------------------------------------------------------===//

void ONNXConstantOp::print(OpAsmPrinter &odsPrinter) {
  // If the result type is dynamic then it won't match the attribute type and
  // we fall back to printing as attribute dictionary at the end.
  Type resultType = getResult().getType();
  if (auto attr = value()) {
    // ONNXConstantOp value must be ElementsAttr, but not SparseElementsAttr.
    auto elements = attr->cast<ElementsAttr>();
    assert(!elements.isa<SparseElementsAttr>() &&
           "ONNXConstantOp value cannot be sparse");
    if (elements.getType() == resultType) {
      odsPrinter << ' ';
      // Print DisposableElementsAttr as a DenseElementsAttr, because
      // DisposableElementsAttr is an internal representation, so we hide it
      // in this way.
      if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
        disposable.printAsDenseElementsAttr(odsPrinter);
      else
        odsPrinter.printAttribute(elements);
      return;
    }
  }
  if (auto attr = sparse_value()) {
    // ONNXConstantOp sparse_value must be SparseElementsAttr.
    auto sparseElements = attr->cast<SparseElementsAttr>();
    if (sparseElements.getType() == resultType) {
      odsPrinter << ' ';
      odsPrinter.printAttribute(sparseElements);
      return;
    }
  }
  // Fallback if there's something funny: no value or sparse_value attribute,
  // or types mismatch.
  odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  odsPrinter << " : " << resultType;
}

ParseResult ONNXConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute attr;
  Type type;
  // First try to parse attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // If there is no attribute dictionary, the parse above succeeds parsing
  // nothing. We detect this case by the absence of result attributes.
  if (result.attributes.empty()) {
    // Try to parse a SparseElementsAttr or or other ElementsAttr.
    OptionalParseResult opt = parser.parseOptionalAttribute(attr, type);
    if (opt.has_value()) {
      if (*opt)
        return failure();
      const char *name =
          attr.isa<SparseElementsAttr>() ? "sparse_value" : "value";
      result.addAttribute(name, attr);
      result.addTypes({attr.cast<ElementsAttr>().getType()});
      return success();
    }
    // No sparse_value or value attr, so attribute dictionary really is empty.
  }
  if (parser.parseColonType(type))
    return failure();
  result.addTypes({type});
  return success();
}
