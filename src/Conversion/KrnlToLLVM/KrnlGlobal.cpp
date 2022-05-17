/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlGlobal.cpp - Lower KrnlGlobalOp ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlGlobalOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "onnx/onnx_pb.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlGlobalOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlGlobalOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            KrnlGlobalOp::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto krnlGlobalOp = llvm::dyn_cast<KrnlGlobalOp>(op);

    // The element type of the array.
    const auto type = op->getResult(0).getType();
    const auto memRefTy = type.cast<mlir::MemRefType>();
    const auto constantElementType =
        typeConverter->convertType(memRefTy.getElementType());
    auto globalType = constantElementType;

    // The llvm type of the global (example: [2 x [8 x float]]).
    const auto shape = (krnlGlobalOp.shape()).dyn_cast<ArrayAttr>();
    if (shape.empty())
      globalType = LLVM::LLVMArrayType::get(globalType.cast<Type>(), 1);
    else {
      for (int i = shape.size() - 1; i >= 0; i--)
        globalType = LLVM::LLVMArrayType::get(
            globalType.cast<Type>(), ArrayAttrIntVal(shape, i));
    }

    // Create the global at the entry of the module.
    assert(krnlGlobalOp.value().hasValue() &&
           "Krnl Global must always have a value");
    auto value = krnlGlobalOp.value().getValue();
    LLVM::GlobalOp global;
    TypeSwitch<Attribute>(value)
        .Case<OpaqueElementsAttr>([&](OpaqueElementsAttr attr) {
          global = lowerOpaqueConstant(krnlGlobalOp, globalType, rewriter);
        })
        .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
          global = lowerDenseConstant(krnlGlobalOp, globalType, rewriter);
        })
        .Default([&](Attribute attr) {
          llvm_unreachable("Unsupported attribute type");
        });

    // Set the global alignment based on the alignment attribute if it exists,
    // otherwise use the module datalayout info.
    krnl::setAlignment(global, krnlGlobalOp.alignmentAttr(),
        krnlGlobalOp->getParentOfType<ModuleOp>(), rewriter,
        *getTypeConverter());

    // Prepare data to be inserted into a MemRefDescriptor (a struct).
    Value globalOpAddr =
        rewriter.create<LLVM::AddressOfOp>(krnlGlobalOp.getLoc(), global);
    MemRefDescriptor memRefDescr = createMemRefDescriptor(
        globalOpAddr, memRefTy, krnlGlobalOp.getLoc(), rewriter);

    rewriter.replaceOp(op, {memRefDescr});

    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return (a.getValue()[i]).cast<IntegerAttr>().getInt();
  }

  // LLVM::GlobalOp does not support OpaqueElementsAttr.
  // Both StringAttr and OpaqueElementsAttr use StringRef for internal data
  // array. Thus, it looks safe to use StringAtrr instead of
  // OpaqueElementsAttr.
  LLVM::GlobalOp lowerOpaqueConstant(KrnlGlobalOp &krnlGlobalOp,
      Type globalType, ConversionPatternRewriter &rewriter) const {
    assert(krnlGlobalOp.value().hasValue() &&
           "Expecting KrnlGlobalOp with a valid value");
    assert(krnlGlobalOp.value().getValue().isa<OpaqueElementsAttr>() &&
           "Expecting a global with an opaque elements attribute");

    MLIRContext *context = krnlGlobalOp.getContext();
    Location loc = krnlGlobalOp.getLoc();
    ModuleOp module = krnlGlobalOp->getParentOfType<ModuleOp>();

    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    StringRef data =
        krnlGlobalOp.value().getValue().cast<OpaqueElementsAttr>().getValue();
    // Check data size.
    int64_t sizeInBytes = computeSizeInBytes(krnlGlobalOp);
    assert(((int64_t)data.size() == sizeInBytes) && "Data size mismatch.");

    StringAttr llvmStringAttr = StringAttr::get(context, data);
    auto llvmArrayI8Ty =
        LLVM::LLVMArrayType::get(IntegerType::get(context, 8), sizeInBytes);
    LLVM::GlobalOp global = rewriter.create<LLVM::GlobalOp>(loc, llvmArrayI8Ty,
        /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.name(),
        llvmStringAttr);

    LLVM_DEBUG(llvm::dbgs() << "global: " << global << "\n";);
    return global;
  }

  LLVM::GlobalOp lowerDenseConstant(KrnlGlobalOp &krnlGlobalOp, Type globalType,
      ConversionPatternRewriter &rewriter) const {
    assert(krnlGlobalOp.value().hasValue() &&
           "Expecting KrnlGlobalOp with a valid value");
    assert(krnlGlobalOp.value().getValue().isa<DenseElementsAttr>() &&
           "Expecting a global with an dense elements attribute");

    MLIRContext *context = krnlGlobalOp.getContext();
    Location loc = krnlGlobalOp.getLoc();
    ModuleOp module = krnlGlobalOp->getParentOfType<ModuleOp>();

    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    DenseElementsAttr denseAttr =
        krnlGlobalOp.value().getValue().cast<DenseElementsAttr>();

    int64_t sizeInBytes = computeSizeInBytes(krnlGlobalOp);
    LLVM::GlobalOp global;
    if ((!denseAttr.isSplat()) && (sizeInBytes > 1024)) {
      ArrayRef<char> rawData = denseAttr.getRawData();
      assert(((int64_t)rawData.size() == sizeInBytes) && "Data size mismatch.");

      StringRef data(rawData.data(), rawData.size());
      StringAttr llvmStringAttr = StringAttr::get(context, data);
      auto llvmArrayI8Ty =
          LLVM::LLVMArrayType::get(IntegerType::get(context, 8), sizeInBytes);
      global = rewriter.create<LLVM::GlobalOp>(loc, llvmArrayI8Ty,
          /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.name(),
          llvmStringAttr);
    } else {
      if (denseAttr.getElementType().isa<StringType>())
        global = lowerStringLiteral(krnlGlobalOp, globalType, rewriter);
      else
        global = rewriter.create<LLVM::GlobalOp>(loc, globalType,
            /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.name(),
            krnlGlobalOp.value().getValue());
    }

    LLVM_DEBUG(llvm::dbgs() << "global: " << global << "\n";);
    return global;
  }

  int64_t computeSizeInBytes(KrnlGlobalOp &krnlGlobalOp) const {
    // Compute total number of elements.
    const auto shape = (krnlGlobalOp.shape()).dyn_cast<ArrayAttr>();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= ArrayAttrIntVal(shape, i);

    const auto type = krnlGlobalOp.getResult().getType();
    const auto memRefTy = type.cast<mlir::MemRefType>();

    return numElements * getMemRefEltSizeInBytes(memRefTy);
  }

  // Store the given address into a MemRefDescriptor (a struct).
  MemRefDescriptor createMemRefDescriptor(Value address, MemRefType memRefType,
      Location loc, OpBuilder &builder) const {
    Type elementType = memRefType.getElementType();
    LLVMTypeConverter &typeConverter = *getTypeConverter();
    Type llvmElemType = typeConverter.convertType(elementType);

    // Prepare data to be inserted into a MemRefDescriptor (a struct).
    auto ptrType = LLVM::LLVMPointerType::get(llvmElemType);
    // Bitcast the address to the MemRefType's element type.
    Value bitCastOp = builder.create<LLVM::BitcastOp>(loc, ptrType, address);
    // Create llvm MemRef from original MemRef and fill the data pointers.
    return MemRefDescriptor::fromStaticShape(
        builder, loc, typeConverter, memRefType, bitCastOp);
  }

  // Generate a global string for each krnlGlobalOp string value, and store
  // the address of the global strings into an array. Return the array address.
  LLVM::GlobalOp lowerStringLiteral(
      KrnlGlobalOp &krnlGlobalOp, Type globalType, OpBuilder &builder) const {
    assert(krnlGlobalOp.value().getValue().isa<DenseElementsAttr>() &&
           "Expecting a dense value");

    Location loc = krnlGlobalOp.getLoc();
    ModuleOp module = krnlGlobalOp->getParentOfType<ModuleOp>();
    DenseElementsAttr denseAttr =
        krnlGlobalOp.value().getValue().cast<DenseElementsAttr>();

    Type i8Type = IntegerType::get(builder.getContext(), 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);

    // Generate LLVM GlobalOps for each string in the KrnlGlobalOp dense
    // attribute.
    SmallVector<LLVM::GlobalOp> globalOps;
    for (StringRef str : denseAttr.getValues<StringRef>()) {
      LLVM::GlobalOp globalOp = krnl::getOrCreateGlobalString(
          str, loc, builder, module, getTypeConverter());
      globalOps.push_back(globalOp);
    }

    // Generate an LLVM GlobalOps with an initializer region containing one
    // block.
    auto arrayType = LLVM::LLVMArrayType::get(i8PtrType, globalOps.size());
    auto global = builder.create<LLVM::GlobalOp>(loc, arrayType,
        /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.name(),
        Attribute());
    Region &region = global.getInitializerRegion();
    Block *block = builder.createBlock(&region);

    // Initialize an array with the addresses of the global strings.
    builder.setInsertionPoint(block, block->begin());
    Value array = builder.create<LLVM::UndefOp>(loc, arrayType);

    int32_t index = 0;
    Value lastValue = array;
    for (const LLVM::GlobalOp &globalOp : globalOps) {
      Value strAddr = krnl::getPtrToGlobalString(globalOp, loc, builder);
      lastValue = builder.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
          strAddr, builder.getArrayAttr({builder.getIndexAttr(index++)}));
    }

    builder.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({lastValue}));
    return global;
  }
};

void populateLoweringKrnlGlobalOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlGlobalOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
