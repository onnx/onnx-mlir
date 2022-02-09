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

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#include "onnx/onnx_pb.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVM.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
namespace {

// Convert an MLIR  type to the correspoding ONNX type.
static onnx::TensorProto::DataType mlirTypeToOnnxType(mlir::Type elemType) {
  onnx::TensorProto::DataType onnxType = onnx::TensorProto::UNDEFINED;

  TypeSwitch<mlir::Type>(elemType)
      .Case<mlir::BFloat16Type>(
          [&](mlir::BFloat16Type) { onnxType = onnx::TensorProto::BFLOAT16; })
      .Case<mlir::ComplexType>([&](mlir::ComplexType type) {
        if (type.getElementType().isa<mlir::Float32Type>())
          onnxType = onnx::TensorProto::COMPLEX64;
        else if (type.getElementType().isa<mlir::Float64Type>())
          onnxType = onnx::TensorProto::COMPLEX128;
      })
      .Case<mlir::Float16Type>(
          [&](mlir::Float16Type) { onnxType = onnx::TensorProto::FLOAT16; })
      .Case<mlir::Float32Type>(
          [&](mlir::Float32Type) { onnxType = onnx::TensorProto::FLOAT; })
      .Case<mlir::Float64Type>(
          [&](mlir::Float64Type) { onnxType = onnx::TensorProto::DOUBLE; })
      .Case<mlir::IntegerType>([&](mlir::IntegerType type) {
        switch (type.getWidth()) {
        case 1:
          // only a signless type can be a bool.
          onnxType = (type.isSigned() || type.isUnsigned())
                         ? onnx::TensorProto::UNDEFINED
                         : onnx::TensorProto::BOOL;
          break;
        case 8:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT8
                                       : onnx::TensorProto::INT8;
          break;
        case 16:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT16
                                       : onnx::TensorProto::INT16;
          break;
        case 32:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT32
                                       : onnx::TensorProto::INT32;
          break;
        case 64:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT64
                                       : onnx::TensorProto::INT64;
          break;
        }
      })
      .Case<mlir::StringType>(
          [&](mlir::StringType) { onnxType = onnx::TensorProto::STRING; });

  if (onnxType == onnx::TensorProto::UNDEFINED) {
    elemType.dump();
    llvm_unreachable("MLIR type cannot be converted to ONNX type");
  }

  return onnxType;
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

static int64_t getRankFromMemRefType(LLVM::LLVMStructType memRefTy) {
  // Usually a MemRef is a 5-element struct, where the 4th and 5th elements in
  // this struct are arrays whose size is the rank of the tensor. In the event
  // that the corresponding tensor of this MemRef is a scalar, the 4th and 5th
  // elements will have 0-length, which in turn causes the MemRef struct to
  // degenerate into a 3-element struct. For more information, refer to
  // https://github.com/llvm/llvm-project/blob/main/mlir/docs/ConversionToLLVMDialect.md#memref-types.
  auto numElems = memRefTy.getBody().size();
  assert((numElems == 3 || numElems == 5) &&
         "Expect MemRef type to contain either 3 or 5 elements.");

  if (numElems == 3)
    return 0; // MemRef refers to a scalar.
  else
    return memRefTy.getBody()[3].cast<LLVM::LLVMArrayType>().getNumElements();
}

// Create a function declaration for OMInstrumentPoint, the signature is:
//   `void (i64, i64)`
static FlatSymbolRefAttr getOrInsertInstrument(
    PatternRewriter &rewriter, ModuleOp module) {
  auto *context = module.getContext();
  std::string funcName("OMInstrumentPoint");
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return SymbolRefAttr::get(context, funcName);
  auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      llvmVoidTy, ArrayRef<mlir::Type>({llvmI64Ty, llvmI64Ty}), false);

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);
  return SymbolRefAttr::get(context, funcName);
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

static Optional<FlatSymbolRefAttr> getFunctionDeclaration(
    ModuleOp module, const char *funcName) {
  assert(funcName && "Missing function name");
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return SymbolRefAttr::get(module.getContext(), funcName);

  return None;
}

static FlatSymbolRefAttr getOrInsertRandomNormal(
    PatternRewriter &rewriter, ModuleOp module, Type inType) {
  MLIRContext *context = module.getContext();
  StringRef functionName = inType.isF64() ? "get_random_normal_value_f64"
                                          : "get_random_normal_value_f32";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(functionName.str()))
    return SymbolRefAttr::get(context, functionName.str());

  // Signature of the input is:
  //  "krnl.random_normal"(%0, %c60, %cst, %cst_0, %cst_1)
  // with types:
  //  (memref<3x4x5xf32>, index, f32, f32, f32)
  // or
  //  (memref<3x4x5xf64>, index, f64, f64, f64)
  auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
  auto llvmOptionsTy = FloatType::getF32(context);
  auto llvmOutputTy = LLVM::LLVMPointerType::get(llvmOptionsTy);
  if (inType.isF64()) {
    llvmOptionsTy = FloatType::getF64(context);
    llvmOutputTy = LLVM::LLVMPointerType::get(llvmOptionsTy);
  }
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy,
      ArrayRef<mlir::Type>({llvmOutputTy, llvmI64Ty, llvmOptionsTy,
          llvmOptionsTy, llvmOptionsTy}),
      false);

  // Insert the random normal function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), functionName.str(), llvmFnType);
  return SymbolRefAttr::get(context, functionName.str());
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

ATTRIBUTE(unused)
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
    auto loc = op->getLoc();

    KrnlGetRefOpAdaptor operandAdaptor(operands);

    // This is the type of the krnl.getref output. This type is used
    // for the type of the internal MemRef.
    auto type = op->getResult(0).getType();
    auto memRefTy = type.cast<mlir::MemRefType>();

    // auto llvmMemRefType = typeConverter->convertType(type).cast<Type>();
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
      SmallVector<Value, 4> sizes;
      sizes.reserve(memRefTy.getRank());
      unsigned i = 0;
      for (int64_t s : memRefTy.getShape())
        sizes.push_back(s == ShapedType::kDynamicSize
                            ? operands[2 + i++]
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
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlGlobalOpLowering
//===----------------------------------------------------------------------===//

class KrnlGlobalOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlGlobalOpLowering(
      MLIRContext *context, LLVMTypeConverter &llvmTypeConverter)
      : ConvertToLLVMPattern(
            KrnlGlobalOp::getOperationName(), context, llvmTypeConverter) {}

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
    setAlignment(global, krnlGlobalOp.alignmentAttr(),
        krnlGlobalOp->getParentOfType<ModuleOp>(), rewriter);

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

    //  LLVM_DEBUG(llvm::dbgs() << "global: " << global << "\n";);
    return global;
  }

  // If the operation has a valid alignment attribute use it, otherwise
  // attempt to set the alignment based on the module datalayout (if it
  // exists).
  void setAlignment(LLVM::GlobalOp &global, IntegerAttr alignmentAttr,
      ModuleOp module, OpBuilder &builder) const {
    if (alignmentAttr && alignmentAttr.getValue().getSExtValue() != 0)
      global.setAlignmentAttr(alignmentAttr);
    else if (module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
      // TODO: use MLIR data layout when it becomes available.
      llvm::LLVMContext llvmContext;
      int32_t align = LLVM::TypeToLLVMIRTranslator(llvmContext)
                          .getPreferredAlignment(global.getType(),
                              getTypeConverter()->getDataLayout());
      align = std::max(align, MinGlobalAlign);
      global.setAlignmentAttr(builder.getI64IntegerAttr(align));
    } else
      global.setAlignmentAttr(builder.getI64IntegerAttr(MinGlobalAlign));
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

    int64_t numStrings = denseAttr.getValues<StringRef>().size();
    if (numStrings == 1) {
      StringRef str = *denseAttr.getValues<StringRef>().begin();
      LLVM::GlobalOp global =
          getOrCreateGlobalString(str, loc, builder, module);

      //      return builder.create<LLVM::GlobalOp>(loc, globalType,
      //        /*isConstant=*/true, LLVM::Linkage::Internal,
      //        krnlGlobalOp.name(),
      //      StringAttr::get(builder.getContext(), str));
      return global;
    }

    // Generate LLVM GlobalOps for each string in the KrnlGlobalOp dense
    // attribute.
    SmallVector<LLVM::GlobalOp> globalOps;
    for (StringRef str : denseAttr.getValues<StringRef>()) {
      LLVM::GlobalOp globalOp =
          getOrCreateGlobalString(str, loc, builder, module);
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
      LLVM::GEPOp strAddr = getPtrToGlobalString(globalOp, loc, builder);
      lastValue = builder.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
          strAddr, builder.getArrayAttr({builder.getIndexAttr(index++)}));
    }

    builder.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({lastValue}));
    return global;
  }

  // Return the GlobalOp for the given string, creating one if not found.
  LLVM::GlobalOp getOrCreateGlobalString(
      StringRef str, Location loc, OpBuilder &builder, ModuleOp module) const {
    LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(str);
    if (!global) {
      // Create the global at the entry of the module.
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      Type i8Type = IntegerType::get(builder.getContext(), 8);
      Type type = LLVM::LLVMArrayType::get(i8Type, str.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
          LLVM::Linkage::Internal, str, builder.getStringAttr(str));

      setAlignment(global, nullptr, module, builder);
    }
    return global;
  }

  // Return a pointer to the first character in a global string.
  LLVM::GEPOp getPtrToGlobalString(
      const LLVM::GlobalOp &global, Location loc, OpBuilder &builder) const {
    Type i8Type = IntegerType::get(builder.getContext(), 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    Type llvmIndexType =
        getTypeConverter()->convertType(builder.getIndexType());

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value zero = builder.create<LLVM::ConstantOp>(
        loc, llvmIndexType, builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, i8PtrType, globalPtr, ArrayRef<Value>({zero, zero}));
  }

  const int32_t MinGlobalAlign = 16;
};

class KrnlInstrumentOpLowering : public ConversionPattern {
public:
  explicit KrnlInstrumentOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlInstrumentOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlInstrumentOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    KrnlInstrumentOp instrumentOp = llvm::dyn_cast<KrnlInstrumentOp>(op);

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    // auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    // auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context,
    // 8));
    // auto llvmI64Ty = IntegerType::get(context, 64); auto llvmFnType =
    // LLVM::LLVMFunctionType::get(
    //    llvmVoidTy, ArrayRef<mlir::Type>({llvmI64Ty, llvmI64Ty}), false);

    auto instrumentRef = getOrInsertInstrument(rewriter, parentModule);

    Value nodeName =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(context, 64),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64), instrumentOp.opID()));
    Value tag =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(context, 64),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64), instrumentOp.tag()));
    // StringRef txt = instrumentOp->op_name();
    // Value nodeName = rewriter.create<LLVM::ConstantOp>(loc, llvmI8PtrTy,
    // instrumentOp->op_name());

    rewriter.create<CallOp>(loc, instrumentRef, ArrayRef<Type>({}),
        ArrayRef<Value>({nodeName, tag}));

    rewriter.eraseOp(op);
    return success();
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
// KRNL to LLVM: KrnlStrlenOpLowering
//===----------------------------------------------------------------------===//

class KrnlStrlenOpLowering : public ConversionPattern {
public:
  explicit KrnlStrlenOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlStrlenOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    KrnlStrlenOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Get a symbol reference to the strlen function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto strlenRef = getOrInsertStrlen(rewriter, parentModule);

    // Operand.
    Type strType = operandAdaptor.str()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value extractedStrPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, strType, operandAdaptor.str(), rewriter.getI64ArrayAttr(1));

    // Strlen call.
    // TODO: should return a size_t
    Type retType = IntegerType::get(context, 64);
    auto funcCall = rewriter.create<CallOp>(
        loc, strlenRef, retType, ArrayRef<Value>({extractedStrPtr}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  /// Return a symbol reference to the strlen function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertStrlen(
      PatternRewriter &rewriter, ModuleOp module) {
    constexpr const char *funcName = "strlen";
    Optional<FlatSymbolRefAttr> optFuncDecl =
        getFunctionDeclaration(module, funcName);
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'strlen' function signature: `size_t (i8*)`
    // TODO: need to create size_t not i64.
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    Type fnType = LLVM::LLVMFunctionType::get(
        rewriter.getI64Type(), ArrayRef<Type>({i8PtrType}), false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);

    return SymbolRefAttr::get(ctx, funcName);
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlStrncmpOpLowering
//===----------------------------------------------------------------------===//

class KrnlStrncmpOpLowering : public ConversionPattern {
public:
  explicit KrnlStrncmpOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlStrncmpOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlStrncmpOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Get a symbol reference to the strncmp function, inserting it if
    // necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto StrncmpRef = getOrInsertStrncmp(rewriter, parentModule);

    // Operands.
    Type strType = operandAdaptor.str1()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value extractedStrPtr1 = rewriter.create<LLVM::ExtractValueOp>(
        loc, strType, operandAdaptor.str1(), rewriter.getI64ArrayAttr(1));
    Value extractedStrPtr2 = rewriter.create<LLVM::ExtractValueOp>(
        loc, strType, operandAdaptor.str2(), rewriter.getI64ArrayAttr(1));
    Value length = operandAdaptor.len();

    // Strncmp call.
    MLIRContext *ctx = op->getContext();
    Type i32Type = IntegerType::get(ctx, 32);
    auto funcCall = rewriter.create<CallOp>(loc, StrncmpRef, i32Type,
        ArrayRef<Value>({extractedStrPtr1, extractedStrPtr2, length}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  /// Return a symbol reference to the strncmp function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertStrncmp(
      PatternRewriter &rewriter, ModuleOp module) {
    constexpr const char *funcName = "strncmp";
    Optional<FlatSymbolRefAttr> optFuncDecl =
        getFunctionDeclaration(module, funcName);
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'strncmp' function signature: `i32 (i8*, i8*, i64)`
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
    Type fnType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(),
        ArrayRef<Type>({i8PtrTy, i8PtrTy, rewriter.getI64Type()}), false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);

    return SymbolRefAttr::get(ctx, funcName);
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
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

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
  ArrayRef<bool> constantOutputs;

  KrnlEntryPointOpLowering(MLIRContext *ctx, ArrayRef<bool> constantOutputs)
      : OpRewritePattern<KrnlEntryPointOp>(ctx),
        constantOutputs(constantOutputs) {}

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
            .getLeafReference()
            .getValue();

    // When there is only a single entry point function in a model, use
    // "run_main_graph" as the default name.
    // TODO(tung): support multiple entry point functions.
    std::string entryPointName = "run_main_graph";
    assert(module.lookupSymbol(entryPointName) == nullptr &&
           "Only support a single entry point function.");

    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVM::LLVMFunctionType::get(opaquePtrTy, {opaquePtrTy}, false);
    auto dynamicEntryPointFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc, entryPointName, dynEntryPointFuncTy);
    auto &entryPointEntryBlock =
        createEntryBlock(dynEntryPointFuncTy, dynamicEntryPointFunc, loc);
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

    Value omTensorPtrArr =
        callApi(rewriter, loc, apiRegistry, API::GET_OMT_ARRAY, {wrappedInput});
    auto one = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(1));

    // Create a memref type for the return argument of the iface call
    Type memRefOutPtrTy = staticEntryPointTy.getParamType(0);
    Value ptrToOutMemRef =
        rewriter.create<LLVM::AllocaOp>(loc, memRefOutPtrTy, one,
            /*alignment=*/0);
    staticInputs.emplace_back(ptrToOutMemRef);

    // Start with param 1 because 0 is the return value
    for (size_t i = 1; i < staticEntryPointTy.getNumParams(); i++) {
      // Call API function to retrieve the i-th dynamic memref.
      auto idxVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i - 1));

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
    rewriter.create<LLVM::CallOp>(
        loc, ArrayRef<Type>({}), wrappedStaticEntryPointFuncName, staticInputs);
    auto outMemRefs = rewriter.create<LLVM::LoadOp>(loc, ptrToOutMemRef);
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
        loc, int64Ty, rewriter.getI64IntegerAttr(outMemRefList.size()));

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

    for (unsigned int i = 0; i < outMemRefList.size(); i++) {
      // Get the i-th memref returned, convert to a dynamic memref and store it
      // in the wrappedOutput.

      auto memRef = outMemRefList.at(i);
      auto outMemRefTy = memRef.getType().dyn_cast<LLVM::LLVMStructType>();
      auto outMemRefRank = getRankFromMemRefType(outMemRefTy);
      auto outMemRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(outMemRefRank));
      Value outOMTensor = callApi(
          rewriter, loc, apiRegistry, API::CREATE_OMTENSOR, {outMemRefRankVal});
      // If output is a constant tensor, OMTensor does not own it.
      auto outOwning = constantOutputs[i] ? 0 : 1;
      LLVM_DEBUG(llvm::dbgs() << "Output OMTensor " << i
                              << " with owning = " << outOwning << "\n");
      fillOMTensorWithMemRef(
          memRef, outOMTensor, outOwning, rewriter, loc, apiRegistry, module);

      auto idxVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      auto omTensorPtrAddrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
      auto omTensorPtrAddr = rewriter
                                 .create<LLVM::GEPOp>(loc, omTensorPtrAddrTy,
                                     outOmtPtrsArr, ArrayRef<Value>{idxVal})
                                 .getResult();

      rewriter.create<LLVM::StoreOp>(loc, outOMTensor, omTensorPtrAddr);
    }

    // Create wrapped output.
    Value wrappedOutput = callApi(rewriter, loc, apiRegistry,
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
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);

    // Declare API type as an enum value, its string name and an LLVM Type
    // specifying its signature.
    // clang-format off
    std::vector<ApiSpec> apiSpecs = {
        ApiSpec(API::CREATE_OMTENSOR_LIST, "omTensorListCreateWithOwnership", opaquePtrTy, {opaquePtrPtrTy, int64Ty, int64Ty}),
        ApiSpec(API::CREATE_OMTENSOR, "omTensorCreateUntyped", opaquePtrTy, {int64Ty}),
        ApiSpec(API::GET_DATA, "omTensorGetDataPtr", opaquePtrTy, {opaquePtrTy}),
        ApiSpec(API::SET_DATA, "omTensorSetDataPtr", voidTy, {opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy}),
        ApiSpec(API::GET_DATA_SHAPE, "omTensorGetShape", int64PtrTy, {opaquePtrTy}),
        ApiSpec(API::GET_DATA_STRIDES, "omTensorGetStrides", int64PtrTy, {opaquePtrTy}),
        ApiSpec(API::GET_DATA_TYPE, "omTensorGetDataType", int64Ty, {opaquePtrTy}),
        ApiSpec(API::SET_DATA_TYPE, "omTensorSetDataType", voidTy, {opaquePtrTy, int64Ty}),
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
  Block &createEntryBlock(Type &dynEntryPoint,
      LLVM::LLVMFuncOp &dynamicEntryPointFunc, Location &loc) const {
    // Add entry block:
    auto *entryPointEntryBlock = new Block();
    auto dynEntryPointFuncType = dynEntryPoint.cast<LLVM::LLVMFunctionType>();
    dynamicEntryPointFunc.push_back(entryPointEntryBlock);
    llvm::SmallVector<Type, 4> argTypes;
    for (size_t i = 0; i < dynEntryPointFuncType.getNumParams(); i++)
      argTypes.emplace_back(dynEntryPointFuncType.getParamType(i));
    auto argLocs = llvm::SmallVector<Location, 4>(
        dynEntryPointFuncType.getNumParams(), loc);
    entryPointEntryBlock->addArguments(argTypes, argLocs);
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
    Value dataPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA, {rtMemRef});
    dataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, memRefTy.cast<LLVM::LLVMStructType>().getBody()[0], dataPtr);
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(0)}));
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(1)}));

    // Use zero offset now.
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(0));
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef, zero,
        rewriter.getArrayAttr({rewriter.getI64IntegerAttr(2)}));

    // Get rank, sizes array ptr and strides array ptr.
    auto rank = getRankFromMemRefType(memRefTy.cast<LLVM::LLVMStructType>());
    Value sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_SHAPE, {rtMemRef});
    Value stridesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_STRIDES, {rtMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Insert size of the dimension.
      auto dimSizePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimSize = rewriter.create<LLVM::LoadOp>(loc, int64Ty, dimSizePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimSize,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));

      // Insert stride of the dimension.
      auto dimStridePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimStride =
          rewriter.create<LLVM::LoadOp>(loc, int64Ty, dimStridePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimStride,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
    }

    rewriter.create<LLVM::StoreOp>(loc, memRef, ptrToMemRef);
  }

  void fillOMTensorWithMemRef(Value &outMemRef, Value &outOMTensor,
      int64_t outOwning, PatternRewriter &rewriter, const Location &loc,
      const std::map<API, ApiSpec> &apiRegistry, ModuleOp &module) const {
    auto *context = module.getContext();
    auto outMemRefTy = outMemRef.getType().dyn_cast<LLVM::LLVMStructType>();
    auto int64Ty = IntegerType::get(context, 64);

    // Set ownership, i.e., free after OMTensor is destroyed.
    Value owning = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(outOwning));

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

    Type elemTy =
        outMemRefTy.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

    if (auto structType = elemTy.dyn_cast_or_null<LLVM::LLVMStructType>()) {
      elemTy = structType.getBody()[0]
                   .cast<LLVM::LLVMPointerType>()
                   .getElementType();
    }

    onnx::TensorProto::DataType onnxTy = mlirTypeToOnnxType(elemTy);
    auto onnxTyVal = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(onnxTy));
    callApi(rewriter, loc, apiRegistry, API::SET_DATA_TYPE,
        {outOMTensor, onnxTyVal});

    int64_t rank = getRankFromMemRefType(outMemRefTy);
    Value sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA_SHAPE, {outOMTensor});
    Value stridesArrayPtr = callApi(
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
      if (ShapedType::isDynamic(dimSize))
        sizes.push_back(srcMemRefDesc.size(rewriter, loc, pos));
      else
        sizes.push_back(createIndexConstant(rewriter, loc, dimSize));
    }

    if (!ShapedType::isDynamic(targetType.getShape().back())) {
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
    return type.getLayout().isIdentity();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlRandomNormalOpLowering
//===----------------------------------------------------------------------===//

class KrnlRandomNormalOpLowering : public ConversionPattern {
public:
  explicit KrnlRandomNormalOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlRandomNormalOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    KrnlRandomNormalOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    mlir::Type inType = op->getOperand(2).getType();

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto randomNormalFuncRef =
        getOrInsertRandomNormal(rewriter, parentModule, inType);

    // First operand.
    Type outputType = operandAdaptor.output()
                          .getType()
                          .cast<LLVM::LLVMStructType>()
                          .getBody()[1];
    Value alignedOutput = rewriter.create<LLVM::ExtractValueOp>(
        loc, outputType, operandAdaptor.output(), rewriter.getI64ArrayAttr(1));

    // Memcpy call
    rewriter.create<CallOp>(loc, randomNormalFuncRef, ArrayRef<Type>({}),
        ArrayRef<Value>({alignedOutput, operandAdaptor.numberOfValues(),
            operandAdaptor.mean(), operandAdaptor.scale(),
            operandAdaptor.seed()}));

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlFindIndexOpLowering
//===----------------------------------------------------------------------===//

class KrnlFindIndexOpLowering : public ConversionPattern {
public:
  explicit KrnlFindIndexOpLowering(MLIRContext *context)
      : ConversionPattern(KrnlFindIndexOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto findIndexOp = cast<KrnlFindIndexOp>(op);
    MLIRContext *context = findIndexOp.getContext();
    Location loc = findIndexOp.getLoc();
    KrnlFindIndexOpAdaptor operandAdaptor(operands);

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    ModuleOp parentModule = findIndexOp->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr findIndexRef = getOrInsertFindIndex(
        rewriter, parentModule, findIndexOp.input().getType());

    // Select the value to pass to as the first argument based on the operator
    // input type.
    Value firstOperand;
    TypeSwitch<Type>(findIndexOp.input().getType())
        .Case<IntegerType>([&](IntegerType type) {
          assert(type.getWidth() == 64 && "expecting an i64 type");
          firstOperand = operandAdaptor.input();
        })
        .Case<StringType>([&](StringType type) {
          Type ptrType = operandAdaptor.input()
                             .getType()
                             .cast<LLVM::LLVMStructType>()
                             .getBody()[1];
          firstOperand = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType,
              operandAdaptor.input(), rewriter.getI64ArrayAttr(1));
        })
        .Default([](Type) { llvm_unreachable("unexpected inputType"); });

    Type GType =
        operandAdaptor.G().getType().cast<LLVM::LLVMStructType>().getBody()[1];
    Type VType =
        operandAdaptor.V().getType().cast<LLVM::LLVMStructType>().getBody()[1];

    // Remaining operands.
    Value extractedGPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, GType, operandAdaptor.G(), rewriter.getI64ArrayAttr(1));
    Value extractedVPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, VType, operandAdaptor.V(), rewriter.getI64ArrayAttr(1));
    Value length = operandAdaptor.len();

    // Generate the call to the runtime function.
    Type retType = IntegerType::get(context, 64);
    auto funcCall = rewriter.create<CallOp>(loc, findIndexRef, retType,
        ArrayRef<Value>({firstOperand, extractedGPtr, extractedVPtr, length}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  /// Return a symbol reference to the appropriate 'find_index_*' runtime
  /// function, inserting it into the module if necessary.
  static FlatSymbolRefAttr getOrInsertFindIndex(
      PatternRewriter &rewriter, ModuleOp module, Type inputType) {
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i32Type = IntegerType::get(ctx, 32);
    Type i64Type = IntegerType::get(ctx, 64);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    Type i32PtrType = LLVM::LLVMPointerType::get(i32Type);

    // Select the runtime function to use based on the input type.
    std::string funcName = "find_index_";
    Type firstArgType;
    TypeSwitch<Type>(inputType)
        .Case<IntegerType>([&](IntegerType type) {
          assert(type.getWidth() == 64 && "expecting an i64 type");
          funcName += "i64";
          firstArgType = i64Type;
        })
        .Case<StringType>([&](StringType type) {
          funcName += "str";
          firstArgType = i8PtrType;
        })
        .Default([](Type) { llvm_unreachable("unexpected type"); });

    Optional<FlatSymbolRefAttr> optFuncDecl =
        getFunctionDeclaration(module, funcName.c_str());
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'find_index_*' signature: `i64 ([i8*|i64], i32*, i32*, i32)`
    Type fnType = LLVM::LLVMFunctionType::get(i64Type,
        ArrayRef<Type>({firstArgType, i32PtrType, i32PtrType, i32Type}), false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);

    return SymbolRefAttr::get(ctx, funcName);
  }
};

} // end namespace

void mlir::populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter,
    ArrayRef<bool> constantOutputs) {
  // TODO: look at what is done in
  // mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp in function
  // LowerVectorToLLVMPass::runOnOperation() and see what we should do about it.
  // They run it in two steps, and add additional lowerings.

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  // Removed in upgrade of LLVM:
  // vector::populateVectorSlicesLoweringPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns);

  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateStdExpandOpsPatterns(patterns);
  // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
  // performance.
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithmeticExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.insert<KrnlGlobalOpLowering, KrnlVectorTypeCastOpLowering>(
      ctx, typeConverter);
  patterns.insert<KrnlGetRefOpLowering>(ctx, typeConverter);
  patterns.insert<KrnlEntryPointOpLowering>(ctx, constantOutputs);

  patterns.insert<KrnlInstrumentOpLowering>(ctx);

  patterns.insert<KrnlRandomNormalOpLowering>(ctx);
  patterns.insert<KrnlFindIndexOpLowering>(ctx);

  // Math library functions.
  patterns.insert<KrnlUnaryMathOpLowering<KrnlErfOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcosOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAcoshOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAsinhOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlAtanhOp>>(ctx);
  patterns.insert<KrnlUnaryMathOpLowering<KrnlTanOp>>(ctx);

  // C library functions.
  patterns.insert<KrnlMemcpyOpLowering>(ctx);
  patterns.insert<KrnlStrlenOpLowering>(ctx);
  patterns.insert<KrnlStrncmpOpLowering>(ctx);
}

void mlir::checkConstantOutputs(
    ModuleOp &module, SmallVectorImpl<bool> &constantOutputs) {
  Operation *entryPointOp;
  auto walkResult = module->walk([&](mlir::Operation *op) -> WalkResult {
    if (llvm::dyn_cast<KrnlEntryPointOp>(op)) {
      entryPointOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Do nothing if there is no EntryPoint.
  if (!walkResult.wasInterrupted())
    return;

  // Get entry function name.
  StringRef entryPointFuncName =
      entryPointOp
          ->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
          .getLeafReference()
          .getValue();

  // Get entry function op.
  Operation *entryFunc;
  module->walk([&](FuncOp op) -> WalkResult {
    if (SymbolRefAttr::get(op).getValue() == entryPointFuncName) {
      entryFunc = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(entryFunc && "Entry function not found");

  // Get ReturnOp of the entry function op.
  Operation *returnOp;
  entryFunc->walk([&](Operation *op) -> WalkResult {
    if (llvm::dyn_cast<ReturnOp>(op)) {
      returnOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Check, for each output, if it was transitively produced by a constant or
  // not.
  for (Value v : returnOp->getOperands()) {
    bool isConstant = false;
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      // Block argument, not a constant.
      isConstant = false;
    else {
      // If output is just a view, trace back to find which op was producing the
      // source memref.
      while (auto viewOp = llvm::dyn_cast<ViewLikeOpInterface>(definingOp)) {
        Value source = viewOp.getViewSource();
        definingOp = source.getDefiningOp();
        // Block argument, stop.
        if (!definingOp)
          break;
      }
      if (!definingOp)
        // Block argument, not a constant.
        isConstant = false;
      else if (llvm::dyn_cast<KrnlGlobalOp>(definingOp))
        // A constant defined by KrnlGlobalOp.
        isConstant = true;
    }
    constantOutputs.emplace_back(isConstant);
    LLVM_DEBUG(llvm::dbgs()
               << "Is entry function output constant? " << isConstant << "\n");
  }
}

//===----------------------------------------------------------------------===//
// KRNL + Standard + Vector + Affine dialects lowering to LLVM.
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(
      &getContext(), dataLayoutAnalysis.getAtOrAbove(module));
  options.emitCWrappers = true;

  // Determine, for each output, whether it is a constant or not.
  SmallVector<bool, 4> constantOutputs;
  checkConstantOutputs(module, constantOutputs);

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Convert types to legal types for the LLVM dialect.
  LLVMTypeConverter typeConverter(&getContext(), options);

  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    Type elementType = type.getElementType();
    if (!elementType.isa<StringType>())
      return llvm::None;

    elementType = elementType.cast<StringType>().getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });

  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(&getContext());

  populateAffineAndKrnlToLLVMConversion(
      patterns, &getContext(), typeConverter, constantOutputs);

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
