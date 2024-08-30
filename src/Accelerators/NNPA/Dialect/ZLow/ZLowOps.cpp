/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZLow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Stickify.hpp"
// #include "src/Interface/ConstantOpInterface.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

//===----------------------------------------------------------------------===//
// ZLowDialect
//===----------------------------------------------------------------------===//

void ZLowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"
      >();
}

void ZLowAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowLogOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowExpOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowTanhOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowSoftmaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMatMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getYMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBiasMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getHnOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getCfOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getH0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getC0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getHnOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getH0Mutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenWeightsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHiddenBiasMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getWorkAreaMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickForLSTMOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getFGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getCGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOGateMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowStickForGRUOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getZGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRGateMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHGateMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowUnstickOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getXMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowConv2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputKernelMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputBiasMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowAvgPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMaxPool2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowMeanReduce2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

void ZLowBatchNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getShapeMutable(),
      SideEffects::DefaultResource::get());
}

/// Get raw data from a dense attribute.
static void getRawData(DenseElementsAttr denseAttr, std::vector<char> &data) {
  if (!denseAttr.isSplat()) {
    data = denseAttr.getRawData();
  } else {
    ShapedType denseShapeType = mlir::cast<ShapedType>(denseAttr.getType());
    std::vector<char> rawData = denseAttr.getRawData();
    int64_t numElements = denseShapeType.getNumElements();
    for (int i = 0; i < numElements; i++)
      data.insert(data.end(), rawData.begin(), rawData.end());
  }
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

ArrayRef<char> ZLowStickifiedConstantOp::getBuffer() {
  MLIRContext *context = getOperation()->getContext();
  PatternRewriter rewriter(context);
  ZLowStickifiedConstantOp zlowStickifiedConstantOp =
      mlir::cast<ZLowStickifiedConstantOp>(getOperation());
  StringAttr layout = onnx_mlir::zhigh::getZTensorLayoutAttr(
      rewriter, zlowStickifiedConstantOp.getResult().getType());
  ArrayRef<char> ret;
  if (zlowStickifiedConstantOp.getValueAttr()) {
    auto dataAttr = zlowStickifiedConstantOp.getValue().value();
    TypeSwitch<Attribute>(dataAttr)
        .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
          ArrayRef<int64_t> shape = denseAttr.getType().getShape();
          Type elementType = denseAttr.getType().getElementType();
          int rank = shape.size();
          // Read attributes's raw data.
          std::vector<char> attrData;
          getRawData(denseAttr, attrData);
          // Call stickify.
          zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
          // pre-transformed desc.
          zdnn_data_layouts zDNNLayout =
              convertLayoutAttrToZDNNDataLayout(rank, layout);
          // If zDNNLayout is NHWC, we stickify directly from NCHW.
          if (zDNNLayout == ZDNN_NHWC)
            zDNNLayout = ZDNN_NCHW;
          zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
          set_info_pre_transformed_desc(
              &pre_tfrmd_desc, zDNNLayout, zDNNType, shape);
          // transformed desc.
          zdnn_status status =
              generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
          assert(status == ZDNN_OK);
          // Stick data using the software stickify.
          zdnn_ztensor ztensor;
          init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
          status = allochelper_ztensor_alloc(&ztensor);
          assert(status == ZDNN_OK);
          status = stickify(&ztensor, attrData.data());
          assert(status == ZDNN_OK);
          std::vector<char>().swap(attrData);
          zlowStickifiedConstantOp.removeValueAttr();
          int64_t sizeInBytes = ztensor.buffer_size;
          char *rawData = (char *)malloc(sizeInBytes);
          memcpy(rawData, ztensor.buffer, sizeInBytes);
          ret = llvm::ArrayRef(rawData, sizeInBytes);
          allochelper_ztensor_free(&ztensor);
        })
        .Case<DenseResourceElementsAttr>(
            [&](DenseResourceElementsAttr denseResourceAttr) {
              ArrayRef<char> attrData =
                  denseResourceAttr.getRawHandle().getBlob()->getData();
              int64_t sizeInBytes = affine::getIntOrFloatMemRefSizeInBytes(
                  zlowStickifiedConstantOp.getResult().getType())
                                        .value();
              char *rawData = (char *)malloc(sizeInBytes);
              memcpy(rawData, attrData.data(), sizeInBytes);
              ret = llvm::ArrayRef(rawData, sizeInBytes);
            })
        .Default([&](Attribute attr) {
          llvm_unreachable("Unsupported data type.");
        });
  } else if (zlowStickifiedConstantOp.getZeroconstAttr()) {
    int64_t sizeInBytes = affine::getIntOrFloatMemRefSizeInBytes(
        zlowStickifiedConstantOp.getResult().getType())
                              .value();
    char *rawData = (char *)malloc(sizeInBytes);
    memset(rawData, 0, sizeInBytes);
    ret = llvm::ArrayRef(rawData, sizeInBytes);
  }
  return ret;
}

uint64_t ZLowStickifiedConstantOp::getBufferSize() {
  ZLowStickifiedConstantOp zlowStickifiedConstantOp =
      mlir::cast<ZLowStickifiedConstantOp>(getOperation());
  return affine::getIntOrFloatMemRefSizeInBytes(
      zlowStickifiedConstantOp.getResult().getType())
      .value();
}

void ZLowStickifiedConstantOp::setBuffer(ArrayRef<char> rawData) {
  MLIRContext *context = getOperation()->getContext();
  PatternRewriter rewriter(context);
  ZLowStickifiedConstantOp zlowStickifiedConstantOp =
      mlir::cast<ZLowStickifiedConstantOp>(getOperation());
  int64_t sizeInBytes = affine::getIntOrFloatMemRefSizeInBytes(
      zlowStickifiedConstantOp.getResult().getType())
                            .value();
  DenseResourceElementsAttr valueAttr = DenseUI8ResourceElementsAttr::get(
      RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
      zlowStickifiedConstantOp.getOperation()
          ->getDialect()
          ->getNamespace(), // use the dialect as the blob "hint"
      HeapAsmResourceBlob::allocateAndCopyWithAlign(rawData, alignof(char)));
  zlowStickifiedConstantOp.setValueAttr(valueAttr);
}

void ZLowStickifiedConstantOp::freeBuffer(ArrayRef<char> rawData) {
  free(const_cast<char *>(rawData.data()));
  return;
}

} // namespace zlow
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowDialect.cpp.inc"
