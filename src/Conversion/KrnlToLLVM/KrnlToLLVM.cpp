/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ LowerToLLVM.cpp - Lowering from KRNL+Affine+Std to LLVM -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "onnx/onnx_pb.h"
#include "llvm/ADT/Sequence.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVM.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static onnx::TensorProto::DataType llvmTypeToOnnxType(mlir::Type elemType) {
  if (elemType.isa<Float32Type>())
    return onnx::TensorProto::FLOAT;
  if (elemType.isUnsignedInteger(8))
    return onnx::TensorProto::UINT8;
  if (elemType.isSignedInteger(8))
    return onnx::TensorProto::INT8;
  if (elemType.isUnsignedInteger(16))
    return onnx::TensorProto::UINT16;
  if (elemType.isSignedInteger(16))
    return onnx::TensorProto::INT16;
  if (elemType.isSignedInteger(32))
    return onnx::TensorProto::INT32;
  if (elemType.isSignedInteger(64))
    return onnx::TensorProto::INT64;
  // TODO, wait for Tong's input about how string is represented in MLIR.
  if (elemType.isa<Float16Type>())
    return onnx::TensorProto::FLOAT16;
  if (elemType.isa<Float64Type>())
    return onnx::TensorProto::DOUBLE;
  if (elemType.isUnsignedInteger(32))
    return onnx::TensorProto::UINT32;
  if (elemType.isUnsignedInteger(64))
    return onnx::TensorProto::INT64;
  // LLVM Dialect does not have signed/unsigned int, only signless int
  if (auto llvmIntType = elemType.dyn_cast<IntegerType>()) {
    if (llvmIntType.getWidth() == 1)
      return onnx::TensorProto::BOOL;
    if (llvmIntType.getWidth() == 8)
      return onnx::TensorProto::INT8;
    if (llvmIntType.getWidth() == 16)
      return onnx::TensorProto::INT16;
    if (llvmIntType.getWidth() == 32)
      return onnx::TensorProto::INT32;
    if (llvmIntType.getWidth() == 64)
      return onnx::TensorProto::INT64;
  }
  // Complex types don't seem to exist in LLVM Dialect.
  elemType.dump();
  llvm_unreachable("Unexpected LLVM type, cannot be converted to ONNX type.");
}

static FlatSymbolRefAttr getOrInsertExternFunc(StringRef funcName,
    ModuleOp module, mlir::Type funcType, PatternRewriter &rewriter) {
  auto *context = module.getContext();
  if (auto sym = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    assert(sym.getType() == funcType && "wrong symbol type");
    return SymbolRefAttr::get(context, funcName);
  }

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  return SymbolRefAttr::get(context, funcName);
}

static size_t getRankFromMemRefType(LLVM::LLVMStructType memRefTy) {
  // Usually a MemRef is a 5-element struct, where the 4th and 5th elements in
  // this struct are arrays whose size is the rank of the tensor. In the event
  // that the corresponding tensor of this MemRef is a scalar, the 4th and 5th
  // elements will have 0-length, which in turn causes the MemRef struct to
  // degenerate into a 3-element struct. For more information, refer to
  // https://github.com/llvm/llvm-project/blob/master/mlir/docs/ConversionToLLVMDialect.md#memref-types.
  auto numElems = memRefTy.getBody().size();
  assert((numElems == 3 || numElems == 5) &&
         "Expect MemRef type to contain either 3 or 5 elements.");

  if (numElems == 3)
    return 0; // MemRef refers to a scalar.
  else
    return memRefTy.getBody()[3].cast<LLVM::LLVMArrayType>().getNumElements();
}

/// Return a symbol reference to the memcpy function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertMemcpy(
    PatternRewriter &rewriter, ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.memcpy.p0i8.p0i8.i64"))
    return SymbolRefAttr::get(context, "llvm.memcpy.p0i8.p0i8.i64");
  // Create a function declaration for memcpy, the signature is:
  //   * `void (i8*, i8* , i64, i1)`
  auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmI1Ty = IntegerType::get(context, 1);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy,
      ArrayRef<mlir::Type>({llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty, llvmI1Ty}),
      false);

  // Insert the memcpy function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "llvm.memcpy.p0i8.p0i8.i64", llvmFnType);
  return SymbolRefAttr::get(context, "llvm.memcpy.p0i8.p0i8.i64");
}

static FlatSymbolRefAttr getOrInsertMalloc(
    PatternRewriter &rewriter, ModuleOp module) {
  // Insert the malloc/aligned_alloc declaration if it is not already present.
  auto allocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
  auto ctx = rewriter.getContext();
  LLVMTypeConverter converter(ctx);
  if (!allocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    SmallVector<Type, 2> callArgTypes = {converter.getIndexType()};
    // aligned_alloc(size_t alignment, size_t size)
    auto voidPtrType = LLVM::LLVMPointerType::get(
        IntegerType::get(&converter.getContext(), 8));
    allocFunc =
        rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), "malloc",
            LLVM::LLVMFunctionType::get(voidPtrType, callArgTypes,
                /*isVarArg=*/false));
  }
  return SymbolRefAttr::get(ctx, "malloc");
}

static FlatSymbolRefAttr getOrInsertDealloc(
    PatternRewriter &rewriter, ModuleOp module) {
  // Insert the dealloc declaration if it is not already present.
  auto deallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("free");
  auto ctx = rewriter.getContext();
  LLVMTypeConverter converter(ctx);
  if (!deallocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto voidPtrType = LLVM::LLVMPointerType::get(
        IntegerType::get(&converter.getContext(), 8));
    SmallVector<Type, 2> callArgTypes = {voidPtrType};
    auto llvmVoidTy = LLVM::LLVMVoidType::get(&converter.getContext());
    deallocFunc =
        rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), "free",
            LLVM::LLVMFunctionType::get(llvmVoidTy, callArgTypes,
                /*isVarArg=*/false));
  }
  return SymbolRefAttr::get(ctx, "free");
}

// This function emits a declaration of the form:
//
// declare float <mathFuncName>(float)
//
static FlatSymbolRefAttr getOrInsertUnaryMathFunction(PatternRewriter &rewriter,
    ModuleOp module, std::string mathFuncName, mlir::Type llvmType) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(mathFuncName))
    return SymbolRefAttr::get(context, mathFuncName);

  // Create function declaration.
  // auto llvmF32Ty = FloatType::get(context);
  auto llvmFnType =
      LLVM::LLVMFunctionType::get(llvmType, ArrayRef<mlir::Type>({llvmType}));

  // Insert the unary math function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), mathFuncName, llvmFnType);
  return SymbolRefAttr::get(context, mathFuncName);
}

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlGetRefOpLowering
//===----------------------------------------------------------------------===//

class KrnlGetRefOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlGetRefOpLowering(
      MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(
            KrnlGetRefOp::getOperationName(), context, lowering_) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlGetRefOp getRefOp = llvm::dyn_cast<KrnlGetRefOp>(op);
    auto *context = op->getContext();
    auto loc = op->getLoc();

    KrnlGetRefOpAdaptor operandAdaptor(operands);

    // This is the type of the krnl.getref output. This type is used
    // for the type of the internal MemRef.
    auto type = op->getResult(0).getType();
    auto memRefTy = type.cast<mlir::MemRefType>();

    auto llvmMemRefType = typeConverter->convertType(type).cast<Type>();
    auto outputElementType =
        typeConverter->convertType(memRefTy.getElementType());

    // This is the start of the memory pool containing the output MemRef.
    Type memPoolType = operandAdaptor.mempool()
                           .getType()
                           .cast<LLVM::LLVMStructType>()
                           .getBody()[1];
    Value alignedMemPoolBase = rewriter.create<LLVM::ExtractValueOp>(loc,
        memPoolType, operandAdaptor.mempool(), rewriter.getI64ArrayAttr(1));

    // Get pointer using the offset.
    auto offset = operandAdaptor.offset();
    auto llvmMemPoolType = typeConverter->convertType(memPoolType).cast<Type>();
    auto outputMemPoolTypePtrAlloc = rewriter.create<LLVM::GEPOp>(
        loc, llvmMemPoolType, alignedMemPoolBase, ArrayRef<Value>({offset}));

    // Bitcast to output MemRef type i.e. from i8* to the element type
    // of the output MemRef.
    auto llvmOutputElementType = outputElementType.cast<Type>();
    Value outputTypedPtrAlloc = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(llvmOutputElementType),
        outputMemPoolTypePtrAlloc);

    // Handle the static case.
    if (hasAllConstantDimensions(memRefTy)) {
      // Create llvm MemRef from original MemRef and fill the data pointers.
      auto llvmMemRef = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memRefTy, outputTypedPtrAlloc);

      rewriter.replaceOp(op, {llvmMemRef});
      return success();
    }

    // Handle the dynamic case.

    // Compute strides and offset based on MemRef type.
    int64_t alignmentOffset;
    SmallVector<int64_t, 4> strides;
    auto successStrides =
        getStridesAndOffset(memRefTy, strides, alignmentOffset);
    (void)successStrides;
    assert(succeeded(successStrides) && "unexpected non-strided memref");

    // Create the memRef descriptor.
    auto structType = typeConverter->convertType(memRefTy);
    auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

    // Allocated pointer, used for malloc/free.
    memRefDescriptor.setAllocatedPtr(rewriter, loc, outputTypedPtrAlloc);

    // Actual aligned pointer to payload.
    // TODO: support aligned MemRefs.
    memRefDescriptor.setAlignedPtr(rewriter, loc, outputTypedPtrAlloc);

    // Offset in aligned pointer.
    // TODO: support non-zero here in the aligned case.
    memRefDescriptor.setOffset(
        rewriter, loc, createIndexConstant(rewriter, loc, 0));

    if (memRefTy.getRank() != 0) {
      // Prepare sizes.
      SmallVector<Value, 4> dynamicSizes = getRefOp.getDynamicSizes();
      SmallVector<Value, 4> sizes;
      sizes.reserve(memRefTy.getRank());
      unsigned i = 0;
      for (int64_t s : memRefTy.getShape())
        sizes.push_back(s == ShapedType::kDynamicSize
                            ? dynamicSizes[i++]
                            : createIndexConstant(rewriter, loc, s));

      // Store all sizes in the descriptor. Only dynamic sizes are passed in as
      // operands to AllocOp.
      Value runningStride = nullptr;
      auto nStrides = strides.size();
      SmallVector<Value, 4> strideValues(nStrides, nullptr);
      for (unsigned i = 0; i < nStrides; ++i) {
        int64_t index = nStrides - 1 - i;
        if (strides[index] == MemRefType::getDynamicStrideOrOffset())
          // Identity layout map is enforced in the match function, so we
          // compute:
          //   `runningStride *= sizes[index + 1]`
          runningStride = runningStride ? rewriter.create<LLVM::MulOp>(loc,
                                              runningStride, sizes[index + 1])
                                        : createIndexConstant(rewriter, loc, 1);
        else
          runningStride = createIndexConstant(rewriter, loc, strides[index]);
        strideValues[index] = runningStride;
      }
      // Fill size and stride descriptors in memref.
      for (auto indexedSize : llvm::enumerate(sizes)) {
        int64_t index = indexedSize.index();
        memRefDescriptor.setSize(rewriter, loc, index, indexedSize.value());
        memRefDescriptor.setStride(rewriter, loc, index, strideValues[index]);
      }
    }

    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return (a.getValue()[i]).cast<IntegerAttr>().getInt();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlGlobalOpLowering
//===----------------------------------------------------------------------===//

class KrnlGlobalOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlGlobalOpLowering(
      MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(
            KrnlGlobalOp::getOperationName(), context, lowering_) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    auto loc = op->getLoc();

    auto krnlGlobalOp = llvm::dyn_cast<KrnlGlobalOp>(op);

    // Get module.
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Global name.
    auto name = krnlGlobalOp.name();

    // Compute total number of elements.
    auto shape = (krnlGlobalOp.shape()).dyn_cast<ArrayAttr>();
    int64_t numElements = 1;
    for (int i = 0; i < shape.size(); ++i)
      numElements *= ArrayAttrIntVal(shape, i);

    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    auto type = op->getResult(0).getType();
    auto memRefTy = type.cast<mlir::MemRefType>();
    auto llvmMemRefType = typeConverter->convertType(type).cast<Type>();

    // The element type of the array.
    auto constantElementType =
        typeConverter->convertType(memRefTy.getElementType());
    auto globalType = constantElementType;

    // The llvm type of the global (example: [2 x [8 x float]])
    if (shape.empty()) {
      globalType = LLVM::LLVMArrayType::get(globalType.cast<Type>(), 1);
    } else {
      for (int i = shape.size() - 1; i >= 0; i--)
        globalType = LLVM::LLVMArrayType::get(
            globalType.cast<Type>(), ArrayAttrIntVal(shape, i));
    }
    auto llvmGlobalType = globalType.cast<Type>();

    if (!krnlGlobalOp.value().hasValue())
      llvm_unreachable("Krnl Global must always have a value");

    int64_t sizeInBytes = numElements * getMemRefEltSizeInBytes(memRefTy);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto llvmArrayI8Ty =
          LLVM::LLVMArrayType::get(IntegerType::get(context, 8), sizeInBytes);
      if (krnlGlobalOp.value().getValue().isa<OpaqueElementsAttr>()) {
        // LLVM::GlobalOp does not support OpaqueElementsAttr.
        // Both StringAttr and OpaqueElementsAttr use StringRef for internal
        // data array. Thus, it looks safe to use StringAtrr instead of
        // OpaqueElementsAttr.
        StringRef data = krnlGlobalOp.value()
                             .getValue()
                             .cast<OpaqueElementsAttr>()
                             .getValue();
        // Check data size.
        assert((data.size() == sizeInBytes) && "Data size mismatch.");

        StringAttr llvmStringAttr = StringAttr::get(context, data);
        global = rewriter.create<LLVM::GlobalOp>(loc, llvmArrayI8Ty,
            /*isConstant=*/true, LLVM::Linkage::Internal, name, llvmStringAttr);
      } else if (krnlGlobalOp.value().getValue().isa<DenseElementsAttr>()) {
        DenseElementsAttr denseAttr =
            krnlGlobalOp.value().getValue().cast<DenseElementsAttr>();
        if (!denseAttr.isSplat()) {
          std::vector<char> rawData = denseAttr.getRawData();
          // Check data size.
          assert((rawData.size() == sizeInBytes) && "Data size mismatch.");

          StringRef data = StringRef((char *)rawData.data(), rawData.size());
          StringAttr llvmStringAttr = StringAttr::get(context, data);
          global = rewriter.create<LLVM::GlobalOp>(loc, llvmArrayI8Ty,
              /*isConstant=*/true, LLVM::Linkage::Internal, name,
              llvmStringAttr);
        } else {
          global = rewriter.create<LLVM::GlobalOp>(loc, llvmGlobalType,
              /*isConstant=*/true, LLVM::Linkage::Internal, name,
              krnlGlobalOp.value().getValue());
        }
      } else
        llvm_unreachable("Unsupported attribute type");
    }

    // Prepare data to be inserted into MemRef.
    // If global needs alignment, we need to allocate a new buffer that is
    // aligned. This is a region of local memory and needs to be emitted as an
    // alloca.
    // - Otherwise, no preparation is needed, directly use the global.

    int alignment = 0;
    if (krnlGlobalOp.alignmentAttr())
      alignment = krnlGlobalOp.alignmentAttr().getValue().getSExtValue();
    Value typedGlobal;

    // Get the address of the global.
    Value globalValue = rewriter.create<LLVM::AddressOfOp>(loc, global);
    auto llvmConstantElementType = constantElementType.cast<Type>();

    if (alignment != 0) {
      // Some frequently used types.
      auto llvmI8PtrTy =
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI64Ty = IntegerType::get(context, 64);
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(1));
      Value alloc = rewriter.create<LLVM::AllocaOp>(loc,
          LLVM::LLVMPointerType::get(llvmGlobalType), one,
          /*alignment=*/alignment);

      // Copy constant value into the local alloca:
      //  - Bitcast alloc to i8*
      Value int8PtrAlloc =
          rewriter.create<LLVM::BitcastOp>(loc, llvmI8PtrTy, alloc);
      //  - Bitcast global to i8*
      Value i8PtrGlobal =
          rewriter.create<LLVM::BitcastOp>(loc, llvmI8PtrTy, globalValue);
      //  - Set size.
      Value totalElementsSize = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(sizeInBytes));
      Value int64Size =
          rewriter.create<LLVM::SExtOp>(loc, llvmI64Ty, totalElementsSize);
      //  - Set volatile.
      Value isVolatile =
          rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(context, 1),
              rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
      //  - Copy constant data into the alloca.
      auto memcpyRef = getOrInsertMemcpy(rewriter, module);
      rewriter.create<CallOp>(loc, memcpyRef, ArrayRef<Type>({}),
          ArrayRef<Value>({int8PtrAlloc, i8PtrGlobal, int64Size, isVolatile}));
      //  - Use the allocated buffer.
      typedGlobal = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(llvmConstantElementType), alloc);
    } else {
      // Bitcast the global to the MemRefType's element type.
      typedGlobal = rewriter.create<LLVM::BitcastOp>(loc,
          LLVM::LLVMPointerType::get(llvmConstantElementType), globalValue);
    }

    // Create llvm MemRef from original MemRef and fill the data pointers.
    auto llvmMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memRefTy, typedGlobal);

    rewriter.replaceOp(op, {llvmMemRef});
    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return (a.getValue()[i]).cast<IntegerAttr>().getInt();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlMemcpyOpLowering
//===----------------------------------------------------------------------===//

class KrnlMemcpyOpLowering : public ConversionPattern {
public:
  explicit KrnlMemcpyOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlMemcpyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlMemcpyOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule);

    // First operand.
    Type dstType = operandAdaptor.dest()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedDstMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, dstType, operandAdaptor.dest(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrDstMemory = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        alignedDstMemory);

    // Second operand.
    Type srcType = operandAdaptor.src()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedSrcMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, srcType, operandAdaptor.src(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrSrcMemory = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        alignedSrcMemory);

    // Size.
    Value int64Size = rewriter.create<LLVM::SExtOp>(
        loc, IntegerType::get(context, 64), operandAdaptor.size());

    // Is volatile (set to false).
    Value isVolatile =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(context, 1),
            rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));

    // Memcpy call
    rewriter.create<CallOp>(loc, memcpyRef, ArrayRef<Type>({}),
        ArrayRef<Value>({alignedInt8PtrDstMemory, alignedInt8PtrSrcMemory,
            int64Size, isVolatile}));

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlUnaryMathOpLowering
//===----------------------------------------------------------------------===//

template <typename Op>
struct MathFunctionName {
  static std::string functionName() { return "none"; };
};

template <>
struct MathFunctionName<KrnlErfOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "erff";
    if (type.isF64())
      return "erf";
    llvm_unreachable("Currently unsupported type for erf");
  }
};

template <>
struct MathFunctionName<KrnlAcosOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "acosf";
    if (type.isF64())
      return "acos";
    llvm_unreachable("Unsupported type for acos");
  }
};

template <>
struct MathFunctionName<KrnlAcoshOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "acoshf";
    if (type.isF64())
      return "acosh";
    llvm_unreachable("Unsupported type for acosh");
  }
};

template <>
struct MathFunctionName<KrnlAsinOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "asinf";
    if (type.isF64())
      return "asin";
    llvm_unreachable("Unsupported type for asin");
  }
};

template <>
struct MathFunctionName<KrnlAsinhOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "asinhf";
    if (type.isF64())
      return "asinh";
    llvm_unreachable("Unsupported type for asinh");
  }
};

template <>
struct MathFunctionName<KrnlAtanOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "atanf";
    if (type.isF64())
      return "atan";
    llvm_unreachable("Unsupported type for atan");
  }
};

template <>
struct MathFunctionName<KrnlTanOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "tanf";
    if (type.isF64())
      return "tan";
    llvm_unreachable("Unsupported type for tan");
  }
};

template <>
struct MathFunctionName<KrnlAtanhOp> {
  static std::string functionName(mlir::Type type) {
    if (type.isF32())
      return "atanhf";
    if (type.isF64())
      return "atanh";
    llvm_unreachable("Unsupported type for atanh");
  }
};

template <typename KrnlScalarMathOp>
class KrnlUnaryMathOpLowering : public ConversionPattern {
public:
  explicit KrnlUnaryMathOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlScalarMathOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    auto loc = op->getLoc();

    // get the LLVM type for the function args and result
    mlir::Type inType = op->getOperand(0).getType();
    mlir::Type llvmType;
    if (inType.isF32())
      llvmType = FloatType::getF32(context);
    else if (inType.isF64())
      llvmType = FloatType::getF64(context);

    // Insert and/or get reference to elementary math function declaration.
    assert(
        inType.isIntOrFloat() && "Type for math function must be int or float");
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto mathFunctionRef = getOrInsertUnaryMathFunction(rewriter, parentModule,
        MathFunctionName<KrnlScalarMathOp>().functionName(inType), llvmType);

    // Emit function call.
    auto funcCall = rewriter.create<CallOp>(
        loc, mathFunctionRef, llvmType, ArrayRef<Value>({operands[0]}));
    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlEntryPointOp
//===----------------------------------------------------------------------===//

class KrnlEntryPointOpLowering : public OpRewritePattern<KrnlEntryPointOp> {
public:
  using OpRewritePattern<KrnlEntryPointOp>::OpRewritePattern;

  enum class API {
    CREATE_OMTENSOR_LIST,
    CREATE_OMTENSOR,
    GET_DATA,
    SET_DATA,
    GET_DATA_SHAPE,
    GET_DATA_STRIDES,
    SET_DATA_TYPE,
    GET_DATA_TYPE,
    GET_OMT_ARRAY,
  };

  struct ApiSpec {
    API id;
    std::string name;
    FlatSymbolRefAttr symbolRef;
    Type outputTy;
    SmallVector<Type, 4> inputTys;

    ApiSpec(
        API id, const std::string &name, Type outputTy, ArrayRef<Type> inputTys)
        : id(id), name(name), outputTy(outputTy),
          inputTys(inputTys.begin(), inputTys.end()) {}

    Type funcTy() {
      return LLVM::LLVMFunctionType::get(outputTy, inputTys,
          /*isVarArg=*/false);
    }
  };

  static void genSignatureFunction(PatternRewriter &rewriter,
      MLIRContext *context, std::string funcName, LLVM::GlobalOp sigvar,
      Location loc) {
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    llvm::SmallVector<Type, 1> outputsType{opaquePtrTy};

    auto funcType = rewriter.getFunctionType(llvm::None, outputsType);
    llvm::SmallVector<NamedAttribute, 1> attrs;
    auto funcOp = rewriter.create<FuncOp>(
        UnknownLoc::get(context), funcName, funcType, attrs);

    auto entryBlock = funcOp.addEntryBlock();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    auto sigAddr = rewriter.create<LLVM::AddressOfOp>(loc, sigvar);
    auto sigVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, opaquePtrTy, sigAddr);
    llvm::SmallVector<Value, 1> results = {sigVoidPtr};
    rewriter.create<ReturnOp>(UnknownLoc::get(context), results);
  }

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    auto apiRegistry = RegisterAllApis(module, rewriter);
    auto loc = op.getLoc();
    auto numOutputs = op->getAttrOfType<IntegerAttr>(
                            KrnlEntryPointOp::getNumOutputsAttrName())
                          .getInt();
    auto sigAttr =
        op->getAttrOfType<StringAttr>(KrnlEntryPointOp::getSignatureAttrName());
    auto signature = sigAttr.getValue();

    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);

    // create global to hold signature
    auto splitSig = signature.split('@');
    llvm::StringRef inSig = splitSig.first;
    llvm::StringRef outSig = splitSig.second;
    mlir::StringAttr inSigAttr = mlir::StringAttr::get(context, inSig);
    mlir::StringAttr outSigAttr = mlir::StringAttr::get(context, outSig);

    auto inSigArrayType =
        LLVM::LLVMArrayType::get(IntegerType::get(context, 8), inSig.size());
    auto insig = rewriter.create<LLVM::GlobalOp>(loc, inSigArrayType,
        /*isConstant=*/true, LLVM::Linkage::External, "_in_signature",
        inSigAttr);

    auto outSigArrayType =
        LLVM::LLVMArrayType::get(IntegerType::get(context, 8), outSig.size());
    auto outsig = rewriter.create<LLVM::GlobalOp>(loc, outSigArrayType,
        /*isConstant=*/true, LLVM::Linkage::External, "_out_signature",
        outSigAttr);
    genSignatureFunction(rewriter, context, "omInputSignature", insig, loc);
    genSignatureFunction(rewriter, context, "omOutputSignature", outsig, loc);

    // Rewrite Krnl Entry Point Operation to an LLVM function with a dynamic
    // signature. The signature is dynamic because it remains the same no matter
    // what the model input/output schema look like. Such dynamic signature
    // takes a opaque ptr as input, representing a ptr to a data structure
    // containing a set of dynamic memrefs wrapped in a vector; similarly the
    // output is also a opaque ptr to a data structure with output memrefs
    // wrapped within it.
    auto staticEntryPointFuncName =
        op->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference();
    auto dynEntryPointName = "run_" + staticEntryPointFuncName;
    assert(module.lookupSymbol(dynEntryPointName.str()) == nullptr &&
           "dynamic entry point name is not unique");
    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVM::LLVMFunctionType::get(opaquePtrTy, {opaquePtrTy}, false);
    auto dynamicEntryPointFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc, dynEntryPointName.str(), dynEntryPointFuncTy);
    auto &entryPointEntryBlock =
        createEntryBlock(dynEntryPointFuncTy, dynamicEntryPointFunc);
    rewriter.setInsertionPointToStart(&entryPointEntryBlock);

    // Based on the static entry point type signature, unpack dynamic memory
    // refs to corresponding static memory refs.
    auto wrappedStaticEntryPointFuncName =
        "_mlir_ciface_" + staticEntryPointFuncName.lower();
    auto *staticEntryPointFunc =
        module.lookupSymbol(wrappedStaticEntryPointFuncName);
    assert(staticEntryPointFunc &&
           isa<LLVM::LLVMFuncOp>(staticEntryPointFunc) &&
           "entry point func must exist and be an llvm func op");
    auto staticEntryPointTy = dyn_cast<LLVM::LLVMFuncOp>(staticEntryPointFunc)
                                  .getType()
                                  .dyn_cast<LLVM::LLVMFunctionType>();

    // Retrieve dynamic mem refs from wrapped input, and convert every one of
    // them to static mem refs.
    SmallVector<Value, 4> staticInputs;
    auto wrappedInput = entryPointEntryBlock.getArgument(0);

    auto omTensorPtrArr =
        callApi(rewriter, loc, apiRegistry, API::GET_OMT_ARRAY, {wrappedInput});
    for (size_t i = 0; i < staticEntryPointTy.getNumParams(); i++) {
      // Call API function to retrieve the i-th dynamic memref.
      auto idxVal = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(i));

      auto omTensorPtrAddrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
      auto omTensorPtrAddr = rewriter
                                 .create<LLVM::GEPOp>(loc, omTensorPtrAddrTy,
                                     omTensorPtrArr, ArrayRef<Value>({idxVal}))
                                 .getResult();
      auto omTensorPtr =
          rewriter.create<LLVM::LoadOp>(loc, opaquePtrTy, omTensorPtrAddr)
              .getResult();

      // Create a (static) memref type corresponding to the i-th memref input to
      // the inference function on stack, and load it to memRef.
      auto memRefPtrTy = staticEntryPointTy.getParamType(i);

      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(1));
      Value ptrToMemRef = rewriter.create<LLVM::AllocaOp>(loc, memRefPtrTy, one,
          /*alignment=*/0);

      // Fill in the memref underlying ptrToMemRef with information extracted
      // from omTensorPtr.
      fillPtrToMemRefWithOMTensor(
          omTensorPtr, ptrToMemRef, rewriter, loc, apiRegistry, module);

      // ptrToMemRef will be an input to main computation graph function.
      staticInputs.emplace_back(ptrToMemRef);
    }

    // Call static entry point with the memref ptrs created, and get output.
    auto outMemRefs =
        rewriter
            .create<LLVM::CallOp>(loc, staticEntryPointTy.getReturnType(),
                rewriter.getSymbolRefAttr(wrappedStaticEntryPointFuncName),
                staticInputs)
            .getResult(0);
    auto outMemRefsType = outMemRefs.getType().dyn_cast<LLVM::LLVMStructType>();

    std::vector<mlir::Value> outMemRefList;
    if (numOutputs == 1) {
      // If only one output tensor exists, the tensor's corresponding memref
      // descriptor will be returned as is.
      outMemRefList.emplace_back(outMemRefs);
    } else {
      // Otherwise, if multiple tensors are to be returned, the returned value
      // is a struct. Multiple tensors' memref descriptors are packed into the
      // same struct. So we unpack them iteratively to outMemRefList.
      for (int i = 0; i < numOutputs; i++) {
        auto position = rewriter.getArrayAttr({rewriter.getI64IntegerAttr(i)});
        auto type = outMemRefsType.getBody()[i];
        auto extractOp = rewriter.create<LLVM::ExtractValueOp>(loc,
            /*res=*/type,
            /*type=*/outMemRefs,
            /*position=*/position);
        outMemRefList.emplace_back(extractOp.getResult());
      }
    }

    auto numOutput = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI64IntegerAttr(outMemRefList.size()));

    auto mallocSym = getOrInsertMalloc(rewriter, module);
    // TODO(tjingrant): get pointer size from data layout.
    size_t kPtrSize = 8;
    auto outputOmtPtrsArraySizeInByte = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(outMemRefList.size() * kPtrSize));
    auto outOmtPtrsArr =
        rewriter
            .create<LLVM::CallOp>(loc,
                LLVM::LLVMPointerType::get(
                    IntegerType::get(module.getContext(), 8)),
                mallocSym, ArrayRef<Value>(outputOmtPtrsArraySizeInByte))
            .getResult(0);
    outOmtPtrsArr = rewriter
                        .create<LLVM::BitcastOp>(loc,
                            LLVM::LLVMPointerType::get(
                                LLVM::LLVMPointerType::get(
                                    IntegerType::get(module.getContext(), 8)),
                                0),
                            outOmtPtrsArr)
                        .getResult();

    for (decltype(numOutputs) i = 0; i < outMemRefList.size(); i++) {
      // Get the i-th memref returned, convert to a dynamic memref and store it
      // in the wrappedOutput.

      auto memRef = outMemRefList.at(i);
      auto outMemRefTy = memRef.getType().dyn_cast<LLVM::LLVMStructType>();
      auto outMemRefRank = getRankFromMemRefType(outMemRefTy);
      auto outMemRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(outMemRefRank));
      auto outOMTensor = callApi(
          rewriter, loc, apiRegistry, API::CREATE_OMTENSOR, {outMemRefRankVal});
      fillOMTensorWithMemRef(
          memRef, outOMTensor, rewriter, loc, apiRegistry, module);

      auto idxVal = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(i));

      auto omTensorPtrAddrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
      auto omTensorPtrAddr = rewriter
                                 .create<LLVM::GEPOp>(loc, omTensorPtrAddrTy,
                                     outOmtPtrsArr, ArrayRef<Value>{idxVal})
                                 .getResult();

      rewriter.create<LLVM::StoreOp>(loc, outOMTensor, omTensorPtrAddr);
    }

    // Create wrapped output.
    auto one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));
    auto wrappedOutput = callApi(rewriter, loc, apiRegistry,
        API::CREATE_OMTENSOR_LIST, {outOmtPtrsArr, numOutput, one});

    // Return wrapped output.
    rewriter.create<LLVM::ReturnOp>(
        loc, SmallVector<Value, 1>(1, wrappedOutput));
    return success();
  }

private:
  using ApiRegistry = std::map<API, ApiSpec>;

  ApiRegistry RegisterAllApis(
      ModuleOp &module, PatternRewriter &rewriter) const {
    auto *context = module.getContext();

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);

    // Declare API type as an enum value, its string name and an LLVM Type
    // specifying its signature.
    // clang-format off
    std::vector<ApiSpec> apiSpecs = {
        ApiSpec(API::CREATE_OMTENSOR_LIST, "omTensorListCreateWithOwnership", opaquePtrTy, {opaquePtrPtrTy, int32Ty, int32Ty}),
        ApiSpec(API::CREATE_OMTENSOR, "omTensorCreateEmptyDeprecated", opaquePtrTy, {int32Ty}),
        ApiSpec(API::GET_DATA, "omTensorGetDataPtr", opaquePtrTy, {opaquePtrTy}),
        ApiSpec(API::SET_DATA, "omTensorSetDataPtr", voidTy, {opaquePtrTy, int32Ty, opaquePtrTy, opaquePtrTy}),
        ApiSpec(API::GET_DATA_SHAPE, "omTensorGetShape", int64PtrTy, {opaquePtrTy}),
        ApiSpec(API::GET_DATA_STRIDES, "omTensorGetStrides", int64PtrTy, {opaquePtrTy}),
        ApiSpec(API::GET_DATA_TYPE, "omTensorGetDataType", int32Ty, {opaquePtrTy}),
        ApiSpec(API::SET_DATA_TYPE, "omTensorSetDataType", voidTy, {opaquePtrTy, int32Ty}),
        ApiSpec(API::GET_OMT_ARRAY, "omTensorListGetOmtArray", opaquePtrPtrTy, {opaquePtrTy}),
    };
    // clang-format on

    // Declare APIs in the current module and build an API registry mapping api
    // identities to a symbol reference to the API function.
    ApiRegistry registry;
    for (auto &apiSpec : apiSpecs) {
      apiSpec.symbolRef = getOrInsertExternFunc(
          apiSpec.name, module, apiSpec.funcTy(), rewriter);
      registry.emplace(apiSpec.id, apiSpec);
    }

    return registry;
  }

  // Call a registered API, return the return SSA values if only one result is
  // returned, otherwise return nullptr.
  Value callApi(PatternRewriter &rewriter, Location loc, ApiRegistry registry,
      API apiId, ArrayRef<Value> params) const {
    // To be used as parameters in LLVM::CallOp, voidTy must be converted
    // to empty list to avoid emission of an SSA value with voidTy. However,
    // we still keep using LLVM voidTy (as opposed to empty list) when recording
    // API function signatures in API registry because when declaring API
    // functions in LLVM IR, the correct way to indicate an output type for
    // "void" is still LLVM voidTy. Relevant discussion thread:
    // https://github.com/onnx/onnx-mlir/issues/255.
    SmallVector<Type, 1> outputTys;
    auto outputTy = registry.at(apiId).outputTy;
    if (!outputTy.isa<LLVM::LLVMVoidType>())
      outputTys.emplace_back(outputTy);
    auto returnVals =
        rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>(outputTys),
            registry.at(apiId).symbolRef, ArrayRef<Value>(params));
    if (returnVals.getNumResults() == 1)
      return returnVals.getResult(0);
    return nullptr;
  }

  // Helper function to insert an entry block to LLVM function.
  // (TODO): upstream this to MLIR.
  Block &createEntryBlock(
      Type &dynEntryPoint, LLVM::LLVMFuncOp &dynamicEntryPointFunc) const {
    // Add entry block:
    auto *entryPointEntryBlock = new Block();
    auto dynEntryPointFuncType = dynEntryPoint.cast<LLVM::LLVMFunctionType>();
    dynamicEntryPointFunc.push_back(entryPointEntryBlock);
    llvm::SmallVector<Type, 4> argTypes;
    for (size_t i = 0; i < dynEntryPointFuncType.getNumParams(); i++)
      argTypes.emplace_back(dynEntryPointFuncType.getParamType(i));
    entryPointEntryBlock->addArguments(argTypes);
    return *entryPointEntryBlock;
  }

  void fillPtrToMemRefWithOMTensor(Value &rtMemRef, Value &ptrToMemRef,
      PatternRewriter &rewriter, const Location &loc,
      const std::map<API, ApiSpec> &apiRegistry, ModuleOp &module) const {
    auto *context = module.getContext();
    auto memRefPtrTy = ptrToMemRef.getType().dyn_cast<LLVM::LLVMPointerType>();
    auto memRefTy = memRefPtrTy.getElementType();
    auto int64Ty = IntegerType::get(context, 64);

    Value memRef = rewriter.create<LLVM::UndefOp>(loc, memRefTy);

    // Set dataPtr and alignedDataPtr;
    auto dataPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA, {rtMemRef});
    dataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, memRefTy.cast<LLVM::LLVMStructType>().getBody()[0], dataPtr);
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0)}));
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI32IntegerAttr(1)}));

    // Use zero offset now.
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(0));
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef, zero,
        rewriter.getArrayAttr({rewriter.getI32IntegerAttr(2)}));

    // Get rank, sizes array ptr and strides array ptr.
    auto rank = getRankFromMemRefType(memRefTy.cast<LLVM::LLVMStructType>());
    auto sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_SHAPE, {rtMemRef});
    auto stridesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_STRIDES, {rtMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Insert size of the dimension.
      auto dimSizePtr = rewriter.create<LLVM::GEPOp>(loc,
          LLVM::LLVMPointerType::get(int64Ty), sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimSize = rewriter.create<LLVM::LoadOp>(
          loc, LLVM::LLVMPointerType::get(int64Ty), dimSizePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimSize,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));

      // Insert stride of the dimension.
      auto dimStridePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimStride = rewriter.create<LLVM::LoadOp>(
          loc, LLVM::LLVMPointerType::get(int64Ty), dimStridePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimStride,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
    }

    rewriter.create<LLVM::StoreOp>(loc, memRef, ptrToMemRef);
  }

  void fillOMTensorWithMemRef(Value &outMemRef, Value &outOMTensor,
      PatternRewriter &rewriter, const Location &loc,
      const std::map<API, ApiSpec> &apiRegistry, ModuleOp &module) const {
    auto *context = module.getContext();
    auto outMemRefTy = outMemRef.getType().dyn_cast<LLVM::LLVMStructType>();
    auto int64Ty = IntegerType::get(context, 64);
    auto int32Ty = IntegerType::get(context, 32);

    // Set ownership to true, i.e., free after OMTensor is destroyed.
    Value owning = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));

    // Extract the allocated pointer.
    Value outMemRefAllocatedPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, outMemRefTy.getBody()[0],
            outMemRef, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(0)}));
    outMemRefAllocatedPtr = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        outMemRefAllocatedPtr);

    // Extract the aligned pointer.
    Value outMemRefAlignedPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, outMemRefTy.getBody()[1],
            outMemRef, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(1)}));
    outMemRefAlignedPtr = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        outMemRefAlignedPtr);

    // Set ownership, allocated and aligned pointer.
    callApi(rewriter, loc, apiRegistry, API::SET_DATA,
        {outOMTensor, owning, outMemRefAllocatedPtr, outMemRefAlignedPtr});

    auto elemTy =
        outMemRefTy.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();
    auto onnxTy = llvmTypeToOnnxType(elemTy);
    auto onnxTyVal = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(onnxTy));
    callApi(rewriter, loc, apiRegistry, API::SET_DATA_TYPE,
        {outOMTensor, onnxTyVal});

    auto rank = getRankFromMemRefType(outMemRefTy);
    auto sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_SHAPE, {outOMTensor});
    auto stridesArrayPtr = callApi(
        rewriter, loc, apiRegistry, API::GET_DATA_STRIDES, {outOMTensor});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Transfer size of dimension from memref to dynamic memref.
      auto dimSize = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
          outMemRef,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));
      auto dimSizePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimSize, dimSizePtr);

      // Transfer stride of dimension from memref to dynamic memref.
      auto dimStride = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
          outMemRef,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
      auto dimStridePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimStride, dimStridePtr);
    }
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlVectorTypeCastOpLowering
//===----------------------------------------------------------------------===//

// struct KrnlVectorTypeCastOpLowering
//    : public ConvertOpToLLVMPattern<KrnlVectorTypeCastOp> {
//  using ConvertOpToLLVMPattern<KrnlVectorTypeCastOp>::ConvertOpToLLVMPattern;
class KrnlVectorTypeCastOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlVectorTypeCastOpLowering(
      MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(
            KrnlVectorTypeCastOp::getOperationName(), context, lowering_) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto krnlVectorTypeCastOp = cast<KrnlVectorTypeCastOp>(op);
    MemRefType sourceType =
        krnlVectorTypeCastOp.getOperand().getType().cast<MemRefType>();
    MemRefType targetType = krnlVectorTypeCastOp.getType();
    if (!isSupportedMemRefType(targetType) ||
        !isSupportedMemRefType(sourceType))
      return failure();

    KrnlVectorTypeCastOp::Adaptor transformed(operands);
    MemRefDescriptor srcMemRefDesc(transformed.source());

    Type targetStructType =
        typeConverter->convertType(krnlVectorTypeCastOp.getType());
    if (!targetStructType)
      return failure();
    Location loc = op->getLoc();
    // Get memRefDescriptor, the new memref descriptor.
    MemRefDescriptor memRefDescriptor =
        MemRefDescriptor::undef(rewriter, loc, targetStructType);
    auto targetElementPtrType = memRefDescriptor.getElementPtrType();

    // Set the new memref to the same buffer as the source memref.
    Value srcBuffer = srcMemRefDesc.allocatedPtr(rewriter, loc);
    Value targetBuffer = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementPtrType, ArrayRef<Value>(srcBuffer));
    memRefDescriptor.setAllocatedPtr(rewriter, loc, targetBuffer);

    // Set the new memref alignment to the same value as source memref.
    Value srcBufferAligned = srcMemRefDesc.alignedPtr(rewriter, loc);
    Value targetBufAligned = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementPtrType, ArrayRef<Value>(srcBufferAligned));
    memRefDescriptor.setAlignedPtr(rewriter, loc, targetBufAligned);

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(targetType, strides, offset)))
      return failure();

    // Unhandled dynamic offset.
    if (offset == MemRefType::getDynamicStrideOrOffset())
      return failure();

    memRefDescriptor.setOffset(
        rewriter, loc, createIndexConstant(rewriter, loc, offset));

    // Get the sizes of the memref: all but the last one are copied from the
    // source memref. If the dimension size was static, the target memref would
    // have the same size.
    SmallVector<Value, 4> sizes;
    sizes.reserve(targetType.getRank());
    for (unsigned pos = 0, e = targetType.getRank() - 1; pos < e; ++pos) {
      int64_t dimSize = targetType.getDimSize(pos);
      if (dimSize == MemRefType::kDynamicSize)
        sizes.push_back(srcMemRefDesc.size(rewriter, loc, pos));
      else
        sizes.push_back(createIndexConstant(rewriter, loc, dimSize));
    }

    if (targetType.getShape().back() != MemRefType::kDynamicSize) {
      // The op is already verified to have the right size for the last
      // dimension.
      sizes.push_back(
          createIndexConstant(rewriter, loc, targetType.getShape().back()));
    } else {
      // We need to divide the dynamic size on the source by the vector width.
      // There is the implicit expectation that the last dimension of the
      // original memory is a multiple of the vector length.
      Value vecWidth = createIndexConstant(rewriter, loc,
          targetType.getElementType().cast<ShapedType>().getNumElements());
      sizes.push_back(rewriter.create<LLVM::UDivOp>(loc,
          srcMemRefDesc.size(rewriter, loc, sourceType.getRank() - 1),
          vecWidth));
    }

    assert(!sizes.empty() && "target memref rank can't be zero");

    // Compute the total number of memref elements.
    Value cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          loc, getIndexType(), ArrayRef<Value>{cumulativeSize, sizes[i]});

    // Calculate the strides.
    Value runningStride = nullptr;
    // Iterate strides in reverse order, compute runningStride and strideValues.
    unsigned nStrides = strides.size();
    SmallVector<Value, 4> strideValues(nStrides, nullptr);
    for (auto indexedStride : llvm::enumerate(llvm::reverse(strides))) {
      int64_t index = nStrides - 1 - indexedStride.index();
      if (strides[index] == MemRefType::getDynamicStrideOrOffset())
        // Identity layout map is enforced in the match function, so we compute:
        //   `runningStride *= sizes[index + 1]`.
        runningStride = runningStride ? rewriter.create<LLVM::MulOp>(loc,
                                            runningStride, sizes[index + 1])
                                      : createIndexConstant(rewriter, loc, 1);
      else
        runningStride = createIndexConstant(rewriter, loc, strides[index]);
      strideValues[index] = runningStride;
    }

    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(sizes)) {
      int64_t index = indexedSize.index();
      memRefDescriptor.setSize(rewriter, loc, index, indexedSize.value());
      memRefDescriptor.setStride(rewriter, loc, index, strideValues[index]);
    }

    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

  // Check if the MemRefType `type` is supported by the lowering. We currently
  // only support memrefs with identity maps.
  bool isSupportedMemRefType(MemRefType type) const {
    if (!typeConverter->convertType(type.getElementType()))
      return false;
    return type.getAffineMaps().empty() ||
           llvm::all_of(type.getAffineMaps(),
               [](AffineMap map) { return map.isIdentity(); });
  }
};

} // end namespace

void mlir::populateAffineAndKrnlToLLVMConversion(
    OwningRewritePatternList &patterns, MLIRContext *ctx,
    LLVMTypeConverter &typeConverter) {
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateStdExpandOpsPatterns(patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.insert<KrnlGlobalOpLowering, KrnlVectorTypeCastOpLowering>(
      ctx, typeConverter);
  patterns.insert<KrnlGetRefOpLowering>(ctx, typeConverter);
  patterns.insert<KrnlMemcpyOpLowering, KrnlEntryPointOpLowering>(ctx);

  // Math library functions.
  patterns.insert<KrnlUnaryMathOpLowering<KrnlErfOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcosOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcoshOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinhOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanhOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlTanOp>>(ctx);
}

//===----------------------------------------------------------------------===//
// KRNL + Standard + Affine dialects lowering to LLVM.
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
} // end anonymous namespace

void ConvertKrnlToLLVMPass::runOnOperation() {

  {
    OwningRewritePatternList patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorSlicesLoweringPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(patterns);
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addIllegalOp<LLVM::DialectCastOp>();

  // Lower the MemRef types to a representation in LLVM.
  LowerToLLVMOptions options(&getContext());
  options.emitCWrappers = true;
  LLVMTypeConverter typeConverter(&getContext(), options);

  // We have a combination of `krnl`, `affine`, and `std` operations. We
  // lower in stages until all the code is in the LLVM dialect.
  OwningRewritePatternList patterns(&getContext());
  populateAffineAndKrnlToLLVMConversion(patterns, &getContext(), typeConverter);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<mlir::Pass> mlir::createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}
