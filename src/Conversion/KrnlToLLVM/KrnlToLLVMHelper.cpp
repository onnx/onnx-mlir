/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToLLVMHelper.cpp ------------------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Implements utility functions for the Krnl to LLVM dialect conversion.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

static const int32_t MinGlobalAlign = 16;

int64_t getRankFromMemRefType(LLVM::LLVMStructType memRefTy) {
  // Usually a MemRef is a 5-element struct, where the 4th and 5th elements in
  // this struct are arrays whose size is the rank of the tensor. In the event
  // that the corresponding tensor of this MemRef is a scalar, the 4th and 5th
  // elements will have 0-length, which in turn causes the MemRef struct to
  // degenerate into a 3-element struct. For more information, refer to
  // https://github.com/llvm/llvm-project/blob/main/mlir/docs/ConversionToLLVMDialect.md#memref-types.
  size_t numElems = memRefTy.getBody().size();
  assert((numElems == 3 || numElems == 5) &&
         "Expect MemRef type to contain either 3 or 5 elements.");

  return (numElems == 3) ? 0 // MemRef refers to a scalar.
                         : memRefTy.getBody()[3]
                               .cast<LLVM::LLVMArrayType>()
                               .getNumElements();
}

// Convert an MLIR type to the correspoding ONNX type.
onnx::TensorProto::DataType mlirTypeToOnnxType(Type elemType) {
  onnx::TensorProto::DataType onnxType = onnx::TensorProto::UNDEFINED;

  TypeSwitch<Type>(elemType)
      .Case<BFloat16Type>(
          [&](BFloat16Type) { onnxType = onnx::TensorProto::BFLOAT16; })
      .Case<ComplexType>([&](ComplexType type) {
        if (type.getElementType().isa<Float32Type>())
          onnxType = onnx::TensorProto::COMPLEX64;
        else if (type.getElementType().isa<Float64Type>())
          onnxType = onnx::TensorProto::COMPLEX128;
      })
      .Case<Float16Type>(
          [&](Float16Type) { onnxType = onnx::TensorProto::FLOAT16; })
      .Case<Float32Type>(
          [&](Float32Type) { onnxType = onnx::TensorProto::FLOAT; })
      .Case<Float64Type>(
          [&](Float64Type) { onnxType = onnx::TensorProto::DOUBLE; })
      .Case<IntegerType>([&](IntegerType type) {
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
      .Case<LLVM::LLVMStructType>(
          [&](LLVM::LLVMStructType) { onnxType = onnx::TensorProto::STRING; });

  if (onnxType == onnx::TensorProto::UNDEFINED) {
    elemType.dump();
    llvm_unreachable("MLIR type cannot be converted to ONNX type");
  }

  return onnxType;
}

void fillOMTensorWithMemRef(Value &outMemRef, Value &outOMTensor,
    int64_t outOwning, PatternRewriter &rewriter, const Location &loc,
    const RuntimeAPIRegistry &apiRegistry, ModuleOp &module) {
  MLIRContext *context = module.getContext();
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
  RuntimeAPI::callApi(rewriter, loc, apiRegistry, RuntimeAPI::API::SET_DATA,
      {outOMTensor, owning, outMemRefAllocatedPtr, outMemRefAlignedPtr});

  Type elemTy =
      outMemRefTy.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

  onnx::TensorProto::DataType onnxTy = krnl::mlirTypeToOnnxType(elemTy);
  auto onnxTyVal = rewriter.create<LLVM::ConstantOp>(
      loc, int64Ty, rewriter.getI64IntegerAttr(onnxTy));
  RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::SET_DATA_TYPE, {outOMTensor, onnxTyVal});

  int64_t rank = krnl::getRankFromMemRefType(outMemRefTy);
  Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_SHAPE, {outOMTensor});
  Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_STRIDES, {outOMTensor});

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

LLVM::GlobalOp getOrCreateGlobalString(StringRef str, Location loc,
    OpBuilder &builder, ModuleOp module, LLVMTypeConverter *typeConverter) {
  assert(typeConverter && "Expecting a valid LLVM type converter");
  LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    // Create the global at the entry of the module.
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto i8Type = IntegerType::get(builder.getContext(), 8);
    auto type = LLVM::LLVMArrayType::get(i8Type, str.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
        LLVM::Linkage::Internal, str, builder.getStringAttr(str));

    krnl::setAlignment(global, nullptr, module, builder, *typeConverter);
  }

  return global;
}

// Return a pointer to the first character in a global string.
Value getPtrToGlobalString(
    const LLVM::GlobalOp &global, Location loc, OpBuilder &builder) {
  Type i8Type = IntegerType::get(builder.getContext(), 8);
  Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
  Type i64Type = IntegerType::get(builder.getContext(), 64);
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value zero =
      builder.create<LLVM::ConstantOp>(loc, i64Type, builder.getIndexAttr(0));

  return builder.create<LLVM::GEPOp>(
      loc, i8PtrType, globalPtr, ArrayRef<Value>({zero, zero}));
}

void setAlignment(LLVM::GlobalOp &global, IntegerAttr alignmentAttr,
    ModuleOp module, OpBuilder &builder, LLVMTypeConverter &typeConverter) {
  if (alignmentAttr && alignmentAttr.getValue().getSExtValue() != 0)
    global.setAlignmentAttr(alignmentAttr);
  else if (module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
    // TODO: use MLIR data layout when it becomes available.
    llvm::LLVMContext llvmContext;
    int32_t align = LLVM::TypeToLLVMIRTranslator(llvmContext)
                        .getPreferredAlignment(
                            global.getType(), typeConverter.getDataLayout());
    align = std::max(align, MinGlobalAlign);
    global.setAlignmentAttr(builder.getI64IntegerAttr(align));
  } else
    global.setAlignmentAttr(builder.getI64IntegerAttr(MinGlobalAlign));
}

Optional<FlatSymbolRefAttr> getFunctionDeclaration(
    ModuleOp module, StringRef funcName) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return SymbolRefAttr::get(module.getContext(), funcName);
  else
    return None;
}

/// Return a symbol reference to the strncmp function, inserting it into the
/// module if necessary.
FlatSymbolRefAttr getOrInsertStrncmp(OpBuilder &builder, ModuleOp module) {
  constexpr const char *funcName = "strncmp";
  Optional<FlatSymbolRefAttr> optFuncDecl =
      krnl::getFunctionDeclaration(module, funcName);
  if (optFuncDecl.hasValue())
    return optFuncDecl.getValue();

  // Create 'strncmp' function signature: `i32 (i8*, i8*, i64)`
  MLIRContext *ctx = module.getContext();
  Type i8Type = IntegerType::get(ctx, 8);
  Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
  Type fnType = LLVM::LLVMFunctionType::get(builder.getI32Type(),
      ArrayRef<Type>({i8PtrTy, i8PtrTy, builder.getI64Type()}), false);

  // Insert the function declaration the module.
  PatternRewriter::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);

  return SymbolRefAttr::get(ctx, funcName);
}

std::string a2e_s(std::string a_s) {
  // ASCII to EBCDIC IBM-1047, generated by using __a2e_s() on z/OS.
  std::string a2e[128] = {"0x00", "0x01", "0x02", "0x03", "0x37", "0x2d",
      "0x2e", "0x2f", "0x16", "0x05", "0x15", "0x0b", "0x0c", "0x0d", "0x0e",
      "0x0f", "0x10", "0x11", "0x12", "0x13", "0x3c", "0x3d", "0x32", "0x26",
      "0x18", "0x19", "0x3f", "0x27", "0x1c", "0x1d", "0x1e", "0x1f", "0x40",
      "0x5a", "0x7f", "0x7b", "0x5b", "0x6c", "0x50", "0x7d", "0x4d", "0x5d",
      "0x5c", "0x4e", "0x6b", "0x60", "0x4b", "0x61", "0xf0", "0xf1", "0xf2",
      "0xf3", "0xf4", "0xf5", "0xf6", "0xf7", "0xf8", "0xf9", "0x7a", "0x5e",
      "0x4c", "0x7e", "0x6e", "0x6f", "0x7c", "0xc1", "0xc2", "0xc3", "0xc4",
      "0xc5", "0xc6", "0xc7", "0xc8", "0xc9", "0xd1", "0xd2", "0xd3", "0xd4",
      "0xd5", "0xd6", "0xd7", "0xd8", "0xd9", "0xe2", "0xe3", "0xe4", "0xe5",
      "0xe6", "0xe7", "0xe8", "0xe9", "0xad", "0xe0", "0xbd", "0x5f", "0x6d",
      "0x79", "0x81", "0x82", "0x83", "0x84", "0x85", "0x86", "0x87", "0x88",
      "0x89", "0x91", "0x92", "0x93", "0x94", "0x95", "0x96", "0x97", "0x98",
      "0x99", "0xa2", "0xa3", "0xa4", "0xa5", "0xa6", "0xa7", "0xa8", "0xa9",
      "0xc0", "0x4f", "0xd0", "0xa1", "0x07"};
  std::string r;
  const char *s = a_s.c_str();
  while (*s) {
    std::string hex = a2e[(int)*s];
    unsigned char dec = (unsigned char)std::stoul(hex.c_str(), nullptr, 16);
    r.push_back(dec);
    s++;
  }
  r.push_back('\0');
  return r;
}

} // namespace krnl
} // namespace onnx_mlir
