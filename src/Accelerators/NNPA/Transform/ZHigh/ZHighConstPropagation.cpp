/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighConstPropagation.cpp - ZHigh High Level Optimizer ------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Stickify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

template <typename cpptype>
bool checkSplat(DisposableElementsAttr attr) {
  ShapedType tensorType = mlir::cast<ShapedType>(attr.getType());
  int64_t numElements = tensorType.getNumElements();
  auto vals = attr.getValues<cpptype>();

  bool isSplat = true;
  for (int64_t i = 1; i < numElements; ++i) {
    if (vals[i] != vals[0]) {
      isSplat = false;
      break;
    }
  }
  return isSplat;
}

/// Get raw data from a dense attribute.
static void getRawData(ElementsAttr attr_, std::vector<char> &data) {
  ShapedType tensorType = mlir::cast<ShapedType>(attr_.getType());
  Type elemTy = tensorType.getElementType();
  int64_t numElements = tensorType.getNumElements();
  ElementsAttr attr = attr_;
  // Figure out why DenseElementsAttr can detect splat?
  // ElementsAttr attr = ElementsAttrBuilder::toDenseElementsAttr(attr_);
  if (elemTy.isInteger(1)) {
    attr = ElementsAttrBuilder::toDenseElementsAttr(attr_);
  }
  auto denseAttr = mlir::dyn_cast_or_null<DenseElementsAttr>(attr);
  auto disposalAttr = mlir::dyn_cast_or_null<DisposableElementsAttr>(attr);

  ArrayRef<char> rawData;
  bool isSplat = false;
  if (denseAttr) {
    rawData = denseAttr.getRawData();
    isSplat = denseAttr.isSplat();
  } else if (disposalAttr) {
    if (elemTy.isF32()) {
      isSplat = checkSplat<float>(disposalAttr);
    } else if (elemTy.isInteger(8)) {
      isSplat = checkSplat<int32_t>(disposalAttr);
    } else if (elemTy.isInteger(32)) {
      isSplat = checkSplat<int32_t>(disposalAttr);
    } else if (elemTy.isInteger(64)) {
      isSplat = checkSplat<int64_t>(disposalAttr);
    } else {
      DenseElementsAttr dattr = disposalAttr.toDenseElementsAttr();
      rawData = dattr.getRawData();
      isSplat = dattr.isSplat();
    }

    if (isSplat) {
      // It's a bit tricky to deal with splat DisposableElementsAttr: it looks
      // like Splat DisposableElementsAttr has more than one element.
      // This special handling would not consume much memory in case of splat.
      rawData = disposalAttr.toDenseElementsAttr().getRawData();
    } else {
      rawData = disposalAttr.getRawBytes().get();
    }
  } else
    llvm_unreachable("Unsupported ElementsAttr type.");

  // Non-splat case.
  if (!isSplat) {
    data = rawData;
  } else {
    // Splat case.
    for (int i = 0; i < numElements; i++)
      data.insert(data.end(), rawData.begin(), rawData.end());
  }

  // Clear the buffer if possible to save memory.
  // Need to check usage, perhaps, from the caller.
  // if (disposalAttr)
  //   disposalAttr.dispose();
}

/// MLIR type to zDNN type.
zdnn_data_types mlirTypeToZDNNType(Type elementType) {
  if (mlir::isa<FloatType>(elementType)) {
    FloatType floatTy = mlir::cast<FloatType>(elementType);
    if (floatTy.getWidth() == 16) {
      return FP16;
    } else if (floatTy.getWidth() == 32) {
      return FP32;
    } else
      llvm_unreachable("Unsupported data type.");
  } else
    llvm_unreachable("Unsupported data type.");
}

/// Emit a ZHighStikifiedConstant using information from a stickified ztensor.
ZHighStickifiedConstantOp emitZHighStickifiedConstant(PatternRewriter &rewriter,
    Location loc, zdnn_ztensor *ztensor, Type outputType) {

  // Create a ZHighStickifiedConstantOp.
  ZHighStickifiedConstantOp stickifiedConstant =
      rewriter.create<ZHighStickifiedConstantOp>(loc, outputType,
          /*value=*/nullptr,
          /*alignment=*/rewriter.getI64IntegerAttr(4096));

  // Use an dense resource attribute to store stickified data.
  // Attribute type: tensor<sizeInBytes x i8>
  int64_t sizeInBytes = ztensor->buffer_size;
  DenseResourceElementsAttr valueAttr = DenseUI8ResourceElementsAttr::get(
      RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
      stickifiedConstant.getOperation()
          ->getDialect()
          ->getNamespace(), // use the dialect as the blob "hint"
      HeapAsmResourceBlob::allocateAndCopyWithAlign(
          llvm::ArrayRef((char *)ztensor->buffer, sizeInBytes), alignof(char)));

  stickifiedConstant.setValueAttr(valueAttr);

  return stickifiedConstant;
}

ZHighStickifiedConstantOp createConstantForStick(PatternRewriter &rewriter,
    Value replacingValue, Value input, StringAttr layout) {
  Location loc = replacingValue.getLoc();
  ArrayRef<int64_t> shape = mlir::cast<ShapedType>(input.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  int rank = shape.size();

  // Read dense attributes.
  ElementsAttr dataAttr = getElementAttributeFromONNXValue(input);
  assert(dataAttr && "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawData;
  getRawData(dataAttr, rawData);
  // assert((rawData.size() == (uint64_t)getMemRefSizeInBytes(input)) &&
  //        "Data size mismatched");

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  zdnn_data_layouts zDNNLayout =
      convertLayoutAttrToZDNNDataLayout(rank, layout);
  // If zDNNLayout is NHWC, we stickify directly from NCHW.
  if (zDNNLayout == ZDNN_NHWC)
    zDNNLayout = ZDNN_NCHW;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, shape);
  // transformed desc.
  zdnn_status status = generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status = stickify(&ztensor, rawData.data());
  assert(status == ZDNN_OK);
  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

ZHighStickifiedConstantOp createConstantForStickForLSTM(
    PatternRewriter &rewriter, Value replacingValue, Value inputF, Value inputI,
    Value inputC, Value inputO) {
  Location loc = replacingValue.getLoc();

  ArrayRef<int64_t> fShape =
      mlir::cast<ShapedType>(inputF.getType()).getShape();
  assert((fShape.size() == 2 || fShape.size() == 3) && "Wrong tensor shape");
  Type elementType = mlir::cast<ShapedType>(inputF.getType()).getElementType();

  // Read dense attributes.
  ElementsAttr fDataAttr = getElementAttributeFromONNXValue(inputF);
  ElementsAttr iDataAttr = getElementAttributeFromONNXValue(inputI);
  ElementsAttr cDataAttr = getElementAttributeFromONNXValue(inputC);
  ElementsAttr oDataAttr = getElementAttributeFromONNXValue(inputO);
  assert((fDataAttr && iDataAttr && cDataAttr && oDataAttr) &&
         "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawFData, rawIData, rawCData, rawOData;
  getRawData(fDataAttr, rawFData);
  getRawData(iDataAttr, rawIData);
  getRawData(cDataAttr, rawCData);
  getRawData(oDataAttr, rawOData);

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  int rank = fShape.size();
  zdnn_data_layouts zDNNLayout = (rank == 2) ? ZDNN_2DS : ZDNN_3DS;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, fShape);
  // transformed desc.
  zdnn_concat_info concatInfo = RNN_TYPE_LSTM |
                                ((rank == 2) ? USAGE_BIASES : USAGE_WEIGHTS) |
                                PREV_LAYER_NONE;
  zdnn_status status = generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concatInfo, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status = stickify(&ztensor, rawFData.data(), rawIData.data(), rawCData.data(),
      rawOData.data());
  assert(status == ZDNN_OK);

  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

ZHighStickifiedConstantOp createConstantForStickForGRU(
    PatternRewriter &rewriter, Value replacingValue, Value inputZ, Value inputR,
    Value inputH) {
  Location loc = replacingValue.getLoc();

  ArrayRef<int64_t> zShape =
      mlir::cast<ShapedType>(inputZ.getType()).getShape();
  assert((zShape.size() == 2 || zShape.size() == 3) && "Wrong tensor shape");
  Type elementType = mlir::cast<ShapedType>(inputZ.getType()).getElementType();

  // Read dense attributes.
  ElementsAttr zDataAttr = getElementAttributeFromONNXValue(inputZ);
  ElementsAttr rDataAttr = getElementAttributeFromONNXValue(inputR);
  ElementsAttr hDataAttr = getElementAttributeFromONNXValue(inputH);
  assert((zDataAttr && rDataAttr && hDataAttr) && "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawZData, rawHData, rawRData, rawOData;
  getRawData(zDataAttr, rawZData);
  getRawData(rDataAttr, rawRData);
  getRawData(hDataAttr, rawHData);

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  int rank = zShape.size();
  zdnn_data_layouts zDNNLayout = (rank == 2) ? ZDNN_2DS : ZDNN_3DS;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, zShape);
  // transformed desc.
  zdnn_concat_info concatInfo = RNN_TYPE_GRU |
                                ((rank == 2) ? USAGE_BIASES : USAGE_WEIGHTS) |
                                PREV_LAYER_NONE;
  zdnn_status status = generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concatInfo, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status =
      stickify(&ztensor, rawZData.data(), rawRData.data(), rawHData.data());
  assert(status == ZDNN_OK);
  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

//===----------------------------------------------------------------------===//
// ZHigh Stick to Krnl Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighConstPropagation.inc"

struct ZHighConstPropagationPass
    //: public PassWrapper<ZHighConstPropagationPass, OperationPass<ModuleOp>> {
    : public PassWrapper<ZHighConstPropagationPass,
          OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighConstPropagationPass)

  StringRef getArgument() const override { return "constprop-zhigh"; }

  StringRef getDescription() const override {
    return "Constant propagation for ZHigh operations.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighConstPropagationPass() {
  return std::make_unique<ZHighConstPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
