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

#include <fstream>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Support/KrnlSupport.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

/// This variable is initizalied inside ConvertKrnlToLLVMPass.
extern std::string EXTERNAL_CONSTANT_PREFIX;

class KrnlGlobalOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlGlobalOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            KrnlGlobalOp::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto krnlGlobalOp = llvm::dyn_cast<KrnlGlobalOp>(op);
    Location loc = krnlGlobalOp.getLoc();
    MLIRContext *context = krnlGlobalOp.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Basic type.
    Type llvmI8Ty = IntegerType::get(context, 8);
    Type llvmI8PtrTy = getPointerType(context, llvmI8Ty);

    // The element type of the array.
    const Type type = op->getResult(0).getType();
    const MemRefType memRefTy = mlir::cast<mlir::MemRefType>(type);
    const Type constantElementType =
        typeConverter->convertType(memRefTy.getElementType());
    Type globalType = constantElementType;

    // The llvm type of the global (example: [2 x [8 x float]]).
    const auto shape = mlir::dyn_cast<ArrayAttr>(krnlGlobalOp.getShape());
    if (shape.empty())
      globalType = LLVM::LLVMArrayType::get(mlir::cast<Type>(globalType), 1);
    else {
      for (int i = shape.size() - 1; i >= 0; i--)
        globalType = LLVM::LLVMArrayType::get(
            mlir::cast<Type>(globalType), ArrayAttrIntVal(shape, i));
    }

    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    // Pointer to the raw data of the global.
    Value dataPtr;

    if (krnlGlobalOp.getValue().has_value()) {
      auto value = krnlGlobalOp.getValue().value();
      TypeSwitch<Attribute>(value)
          .Case<DenseResourceElementsAttr>([&](DenseResourceElementsAttr attr) {
            global =
                lowerDenseResourceConstant(krnlGlobalOp, globalType, rewriter);
          })
          .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
            global = lowerDenseConstant(krnlGlobalOp, globalType, rewriter);
          })
          .Default([&](Attribute attr) {
            llvm_unreachable("Unsupported attribute type");
          });
      dataPtr = create.llvm.addressOf(global);
    } else {
      // Data are stored on files.
      global = lowerGlobalOpWithExternalFiles(krnlGlobalOp, rewriter);
      dataPtr = create.llvm.load(llvmI8PtrTy, create.llvm.addressOf(global));
    }

    // Set the global alignment based on the alignment attribute if it exists,
    // otherwise use the module datalayout info.
    krnl::setAlignment(global, krnlGlobalOp.getAlignmentAttr(),
        krnlGlobalOp->getParentOfType<ModuleOp>(), rewriter,
        *getTypeConverter());

    // Prepare data to be inserted into a MemRefDescriptor (a struct).
    MemRefDescriptor memRefDescr =
        createMemRefDescriptor(dataPtr, memRefTy, loc, rewriter);

    rewriter.replaceOp(op, {memRefDescr});

    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return mlir::cast<IntegerAttr>(a.getValue()[i]).getInt();
  }

  LLVM::GlobalOp lowerDenseResourceConstant(KrnlGlobalOp &krnlGlobalOp,
      Type globalType, ConversionPatternRewriter &rewriter) const {
    assert(krnlGlobalOp.getValue().has_value() &&
           "Expecting KrnlGlobalOp with a valid value");
    assert(
        mlir::isa<DenseResourceElementsAttr>(krnlGlobalOp.getValue().value()) &&
        "Expecting a global with an dense resource elements attribute");

    MLIRContext *context = krnlGlobalOp.getContext();
    Location loc = krnlGlobalOp.getLoc();
    ModuleOp module = krnlGlobalOp->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    auto blob =
        mlir::cast<DenseResourceElementsAttr>(krnlGlobalOp.getValue().value())
            .getRawHandle()
            .getBlob();
    assert(blob && "Expecting dense resource with a valid blob");
    ArrayRef<char> rawData = blob->getData();

    // Check data size.
    uint64_t sizeInBytes = computeSizeInBytes(krnlGlobalOp);
    assert(((uint64_t)rawData.size() == sizeInBytes) && "Data size mismatch.");

    StringRef data(rawData.data(), rawData.size());
    StringAttr llvmStringAttr = StringAttr::get(context, data);
    auto llvmArrayI8Ty =
        LLVM::LLVMArrayType::get(IntegerType::get(context, 8), sizeInBytes);
    LLVM::GlobalOp global = create.llvm.globalOp(llvmArrayI8Ty,
        /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.getName(),
        llvmStringAttr);

    LLVM_DEBUG(llvm::dbgs() << "global: " << global << "\n";);
    return global;
  }

  LLVM::GlobalOp lowerDenseConstant(KrnlGlobalOp &krnlGlobalOp, Type globalType,
      ConversionPatternRewriter &rewriter) const {
    assert(krnlGlobalOp.getValue().has_value() &&
           "Expecting KrnlGlobalOp with a valid value");
    assert(mlir::isa<DenseElementsAttr>(krnlGlobalOp.getValue().value()) &&
           "Expecting a global with an dense elements attribute");

    Location loc = krnlGlobalOp.getLoc();
    ModuleOp module = krnlGlobalOp->getParentOfType<ModuleOp>();
    MLIRContext *context = krnlGlobalOp.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Type llvmI8Ty = IntegerType::get(context, 8);

    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    DenseElementsAttr denseAttr =
        mlir::cast<DenseElementsAttr>(krnlGlobalOp.getValue().value());

    uint64_t sizeInBytes = computeSizeInBytes(krnlGlobalOp);
    LLVM::GlobalOp global;
    if (!(mlir::isa<StringType>(denseAttr.getElementType())) &&
        !(denseAttr.getElementType().isInteger(1)) && (!denseAttr.isSplat()) &&
        (sizeInBytes > 1024)) {

      ArrayRef<char> rawData = denseAttr.getRawData();
      assert(
          ((uint64_t)rawData.size() == sizeInBytes) && "Data size mismatch.");

      auto llvmArrayI8Ty = LLVM::LLVMArrayType::get(llvmI8Ty, sizeInBytes);
      StringRef data(rawData.data(), rawData.size());
      StringAttr llvmStringAttr = StringAttr::get(context, data);
      global = create.llvm.globalOp(llvmArrayI8Ty,
          /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.getName(),
          llvmStringAttr);
    } else {
      if (mlir::isa<StringType>(denseAttr.getElementType()))
        global = lowerStringLiteral(krnlGlobalOp, globalType, rewriter);
      else
        global = create.llvm.globalOp(globalType,
            /*isConstant=*/true, LLVM::Linkage::Internal,
            krnlGlobalOp.getName(), krnlGlobalOp.getValue().value());
    }

    LLVM_DEBUG(llvm::dbgs() << "global: " << global << "\n";);
    return global;
  }

  LLVM::GlobalOp lowerGlobalOpWithExternalFiles(
      KrnlGlobalOp &krnlGlobalOp, ConversionPatternRewriter &rewriter) const {
    Location loc = krnlGlobalOp.getLoc();
    MLIRContext *context = krnlGlobalOp.getContext();
    ModuleOp module = krnlGlobalOp.getOperation()->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Type llvmI8Ty = IntegerType::get(context, 8);
    Type llvmI8PtrTy = getPointerType(context, llvmI8Ty);
    Type llvmI64Ty = IntegerType::get(context, 64);

    auto offset = krnlGlobalOp.getOffset();
    assert(offset.has_value() && "Missing offset value in KrnlGlobalOp");

    // Data is store in `constants.bin` at offset.
    std::string constantName = krnlGlobalOp.getName().str();

    // Emit globals at the begining of the module.
    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    // Create an uninitialized global. Data will be loaded at runtime.
    LLVM::GlobalOp global = create.llvm.globalOp(llvmI8PtrTy,
        /*isConstant=*/false, LLVM::Linkage::Internal,
        EXTERNAL_CONSTANT_PREFIX + "data_" + constantName, nullptr);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      Region &region = global.getInitializerRegion();
      Block *block = rewriter.createBlock(&region);
      // Initialize an array with the addresses of the global op.
      rewriter.setInsertionPoint(block, block->begin());
      create.llvm._return(create.llvm.null(llvmI8PtrTy));
    }

    // Create a global to store offset.
    create.llvm.globalOp(llvmI64Ty,
        /*isConstant=*/true, LLVM::Linkage::Internal,
        EXTERNAL_CONSTANT_PREFIX + "offset_" + constantName,
        rewriter.getI64IntegerAttr(offset.value()));

    return global;
  }

  uint64_t computeSizeInBytes(KrnlGlobalOp &krnlGlobalOp) const {
    // Compute total number of elements.
    const auto shape = mlir::dyn_cast<ArrayAttr>(krnlGlobalOp.getShape());
    uint64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= ArrayAttrIntVal(shape, i);

    const auto type = krnlGlobalOp.getResult().getType();
    const auto memRefTy = mlir::cast<mlir::MemRefType>(type);

    // Special handling for bool.
    if (memRefTy.getElementType().isInteger(1))
      return llvm::divideCeil(numElements, 8);

    return numElements * getMemRefEltSizeInBytes(memRefTy);
  }

  // Store the given address into a MemRefDescriptor (a struct).
  MemRefDescriptor createMemRefDescriptor(Value address, MemRefType memRefType,
      Location loc, OpBuilder &builder) const {
    Type elementType = memRefType.getElementType();
    const LLVMTypeConverter &typeConverter = *getTypeConverter();
    Type llvmElemType = typeConverter.convertType(elementType);
    MLIRContext *context = builder.getContext();
    MultiDialectBuilder<LLVMBuilder> create(builder, loc);

    // Prepare data to be inserted into a MemRefDescriptor (a struct).
    auto ptrType = getPointerType(context, llvmElemType);
    // Bitcast the address to the MemRefType's element type.
    Value bitCastOp = create.llvm.bitcast(ptrType, address);
    // Create llvm MemRef from original MemRef and fill the data pointers.
    return MemRefDescriptor::fromStaticShape(
        builder, loc, typeConverter, memRefType, bitCastOp);
  }

  // Generate a global string for each krnlGlobalOp string value, and store
  // the address of the global strings into an array. Return the array address.
  LLVM::GlobalOp lowerStringLiteral(
      KrnlGlobalOp &krnlGlobalOp, Type globalType, OpBuilder &builder) const {
    assert(mlir::isa<DenseElementsAttr>(krnlGlobalOp.getValue().value()) &&
           "Expecting a dense value");

    Location loc = krnlGlobalOp.getLoc();
    MultiDialectBuilder<LLVMBuilder> create(builder, loc);

    DenseElementsAttr denseAttr =
        mlir::cast<DenseElementsAttr>(krnlGlobalOp.getValue().value());

    Type i8PtrType = getI8PointerType(builder.getContext());

    auto strs = denseAttr.getValues<StringRef>();
    // Collect total size of the strs.
    size_t totalSize = 0;
    for (StringRef str : strs) {
      // Add 1 for the null terminator.
      totalSize += str.size() + 1;
    }

    // Concatenate all strings into one.
    std::vector<char> concatStr(totalSize);
    size_t offset = 0;
    std::vector<size_t> offsets;
    for (StringRef str : strs) {
      offsets.emplace_back(offset);
      std::copy(str.begin(), str.end(), concatStr.begin() + offset);
      concatStr[offset + str.size()] = '\0';
      offset += str.size() + 1;
    }

    // Create a global for the concatenated string.
    StringRef data(concatStr.data(), concatStr.size());
    StringAttr llvmStringAttr = StringAttr::get(builder.getContext(), data);
    auto i8Type = IntegerType::get(builder.getContext(), 8);
    auto llvmArrayI8Ty = LLVM::LLVMArrayType::get(i8Type, totalSize);
    LLVM::GlobalOp globalStr = create.llvm.globalOp(llvmArrayI8Ty,
        /*isConstant=*/true, LLVM::Linkage::Internal,
        "om.strArray." + krnlGlobalOp.getName().str(), llvmStringAttr);

    // Generate an LLVM GlobalOps with an initializer region containing one
    // block.
    auto arrayType = LLVM::LLVMArrayType::get(i8PtrType, offsets.size());
    auto global = create.llvm.globalOp(arrayType,
        /*isConstant=*/true, LLVM::Linkage::Internal, krnlGlobalOp.getName(),
        Attribute());
    Region &region = global.getInitializerRegion();
    Block *block = builder.createBlock(&region);

    // Initialize an array with the addresses of the global strings.
    builder.setInsertionPoint(block, block->begin());
    Value array = builder.create<LLVM::UndefOp>(loc, arrayType);

    int32_t index = 0;
    Value lastValue = array;
    Value baseAddr = create.llvm.addressOf(globalStr);
    // Cast globalStr to i8Ptr.
    baseAddr = create.llvm.bitcast(i8PtrType, baseAddr);
    for (size_t offset : offsets) {
      // Get each str with gep base, offset.
      Value gepOp = create.llvm.getElemPtr(
          i8PtrType, i8Type, baseAddr, {(int32_t)offset});
      lastValue =
          create.llvm.insertValue(arrayType, lastValue, gepOp, {index++});
    }

    create.llvm._return(lastValue);
    return global;
  }
};

void populateLoweringKrnlGlobalOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlGlobalOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
