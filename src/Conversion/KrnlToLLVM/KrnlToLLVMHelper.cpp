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

namespace onnx_mlir {
namespace krnl {

static constexpr int32_t MinGlobalAlign = 16;

// clang-format off
// ASCII to EBCDIC IBM-1047 table.
static constexpr unsigned char a2e[256] = {
  0x00, 0x01, 0x02, 0x03, 0x37, 0x2D, 0x2E, 0x2F,
  0x16, 0x05, 0x15, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  0x10, 0x11, 0x12, 0x13, 0x3C, 0x3D, 0x32, 0x26,
  0x18, 0x19, 0x3F, 0x27, 0x1C, 0x1D, 0x1E, 0x1F,
  0x40, 0x5A, 0x7F, 0x7B, 0x5B, 0x6C, 0x50, 0x7D,
  0x4D, 0x5D, 0x5C, 0x4E, 0x6B, 0x60, 0x4B, 0x61,
  0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
  0xF8, 0xF9, 0x7A, 0x5E, 0x4C, 0x7E, 0x6E, 0x6F,
  0x7C, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
  0xC8, 0xC9, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
  0xD7, 0xD8, 0xD9, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
  0xE7, 0xE8, 0xE9, 0xAD, 0xE0, 0xBD, 0x5F, 0x6D,
  0x79, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
  0x97, 0x98, 0x99, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6,
  0xA7, 0xA8, 0xA9, 0xC0, 0x4F, 0xD0, 0xA1, 0x07,
  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x06, 0x17,
  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x09, 0x0A, 0x1B,
  0x30, 0x31, 0x1A, 0x33, 0x34, 0x35, 0x36, 0x08,
  0x38, 0x39, 0x3A, 0x3B, 0x04, 0x14, 0x3E, 0xFF,
  0x41, 0xAA, 0x4A, 0xB1, 0x9F, 0xB2, 0x6A, 0xB5,
  0xBB, 0xB4, 0x9A, 0x8A, 0xB0, 0xCA, 0xAF, 0xBC,
  0x90, 0x8F, 0xEA, 0xFA, 0xBE, 0xA0, 0xB6, 0xB3,
  0x9D, 0xDA, 0x9B, 0x8B, 0xB7, 0xB8, 0xB9, 0xAB,
  0x64, 0x65, 0x62, 0x66, 0x63, 0x67, 0x9E, 0x68,
  0x74, 0x71, 0x72, 0x73, 0x78, 0x75, 0x76, 0x77,
  0xAC, 0x69, 0xED, 0xEE, 0xEB, 0xEF, 0xEC, 0xBF,
  0x80, 0xFD, 0xFE, 0xFB, 0xFC, 0xBA, 0xAE, 0x59,
  0x44, 0x45, 0x42, 0x46, 0x43, 0x47, 0x9C, 0x48,
  0x54, 0x51, 0x52, 0x53, 0x58, 0x55, 0x56, 0x57,
  0x8C, 0x49, 0xCD, 0xCE, 0xCB, 0xCF, 0xCC, 0xE1,
  0x70, 0xDD, 0xDE, 0xDB, 0xDC, 0x8D, 0x8E, 0xDF
};
// EBCDIC IBM-1047 to ASCII table.
static constexpr unsigned char e2a[256] = {
  0x00, 0x01, 0x02, 0x03, 0x9C, 0x09, 0x86, 0x7F,
  0x97, 0x8D, 0x8E, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  0x10, 0x11, 0x12, 0x13, 0x9D, 0x0A, 0x08, 0x87,
  0x18, 0x19, 0x92, 0x8F, 0x1C, 0x1D, 0x1E, 0x1F,
  0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x17, 0x1B,
  0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x05, 0x06, 0x07,
  0x90, 0x91, 0x16, 0x93, 0x94, 0x95, 0x96, 0x04,
  0x98, 0x99, 0x9A, 0x9B, 0x14, 0x15, 0x9E, 0x1A,
  0x20, 0xA0, 0xE2, 0xE4, 0xE0, 0xE1, 0xE3, 0xE5,
  0xE7, 0xF1, 0xA2, 0x2E, 0x3C, 0x28, 0x2B, 0x7C,
  0x26, 0xE9, 0xEA, 0xEB, 0xE8, 0xED, 0xEE, 0xEF,
  0xEC, 0xDF, 0x21, 0x24, 0x2A, 0x29, 0x3B, 0x5E,
  0x2D, 0x2F, 0xC2, 0xC4, 0xC0, 0xC1, 0xC3, 0xC5,
  0xC7, 0xD1, 0xA6, 0x2C, 0x25, 0x5F, 0x3E, 0x3F,
  0xF8, 0xC9, 0xCA, 0xCB, 0xC8, 0xCD, 0xCE, 0xCF,
  0xCC, 0x60, 0x3A, 0x23, 0x40, 0x27, 0x3D, 0x22,
  0xD8, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
  0x68, 0x69, 0xAB, 0xBB, 0xF0, 0xFD, 0xFE, 0xB1,
  0xB0, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,
  0x71, 0x72, 0xAA, 0xBA, 0xE6, 0xB8, 0xC6, 0xA4,
  0xB5, 0x7E, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
  0x79, 0x7A, 0xA1, 0xBF, 0xD0, 0x5B, 0xDE, 0xAE,
  0xAC, 0xA3, 0xA5, 0xB7, 0xA9, 0xA7, 0xB6, 0xBC,
  0xBD, 0xBE, 0xDD, 0xA8, 0xAF, 0x5D, 0xB4, 0xD7,
  0x7B, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
  0x48, 0x49, 0xAD, 0xF4, 0xF6, 0xF2, 0xF3, 0xF5,
  0x7D, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,
  0x51, 0x52, 0xB9, 0xFB, 0xFC, 0xF9, 0xFA, 0xFF,
  0x5C, 0xF7, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
  0x59, 0x5A, 0xB2, 0xD4, 0xD6, 0xD2, 0xD3, 0xD5,
  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
  0x38, 0x39, 0xB3, 0xDB, 0xDC, 0xD9, 0xDA, 0x9F
};
// clang-format on

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
  MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

  // Set ownership, i.e., free after OMTensor is destroyed.
  Value owning = create.llvm.constant(int64Ty, (int64_t)outOwning);

  // Extract the allocated pointer.
  Value outMemRefAllocatedPtr =
      create.llvm.extractValue(outMemRefTy.getBody()[0], outMemRef, {0});
  outMemRefAllocatedPtr = create.llvm.bitcastI8Ptr(outMemRefAllocatedPtr);

  // Extract the aligned pointer.
  Value outMemRefAlignedPtr =
      create.llvm.extractValue(outMemRefTy.getBody()[1], outMemRef, {1});
  outMemRefAlignedPtr = create.llvm.bitcastI8Ptr(outMemRefAlignedPtr);

  // Set ownership, allocated and aligned pointer.
  RuntimeAPI::callApi(rewriter, loc, apiRegistry, RuntimeAPI::API::SET_DATA,
      {outOMTensor, owning, outMemRefAllocatedPtr, outMemRefAlignedPtr});

  Type elemTy =
      outMemRefTy.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

  onnx::TensorProto::DataType onnxTy = krnl::mlirTypeToOnnxType(elemTy);
  Value onnxTyVal = create.llvm.constant(int64Ty, (int64_t)onnxTy);
  RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::SET_DATA_TYPE, {outOMTensor, onnxTyVal});

  int64_t rank = krnl::getRankFromMemRefType(outMemRefTy);
  Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_SHAPE, {outOMTensor});
  Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_STRIDES, {outOMTensor});

  for (decltype(rank) i = 0; i < rank; i++) {
    Value dimIdx = create.llvm.constant(int64Ty, (int64_t)i);
    // Transfer size of dimension from memref to dynamic memref.
    Value dimSize = create.llvm.extractValue(int64Ty, outMemRef, {3, i});
    Value dimSizePtr =
        create.llvm.getElemPtr(LLVM::LLVMPointerType::get(int64Ty),
            sizesArrayPtr, ArrayRef<Value>({dimIdx}));
    create.llvm.store(dimSize, dimSizePtr);

    // Transfer stride of dimension from memref to dynamic memref.
    Value dimStride = create.llvm.extractValue(int64Ty, outMemRef, {4, i});
    Value dimStridePtr =
        create.llvm.getElemPtr(LLVM::LLVMPointerType::get(int64Ty),
            stridesArrayPtr, ArrayRef<Value>({dimIdx}));
    create.llvm.store(dimStride, dimStridePtr);
  }
}

LLVM::GlobalOp getOrCreateGlobalString(StringRef str, Location loc,
    OpBuilder &builder, ModuleOp module, LLVMTypeConverter *typeConverter) {
  MultiDialectBuilder<LLVMBuilder> create(builder, loc);
  assert(typeConverter && "Expecting a valid LLVM type converter");
  LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    // Create the global at the entry of the module.
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());

    Type i8Type = IntegerType::get(builder.getContext(), 8);
    Type type = LLVM::LLVMArrayType::get(i8Type, str.size());
    global = create.llvm.globalOp(type, /*isConstant=*/true,
        LLVM::Linkage::Internal, str, builder.getStringAttr(str));

    krnl::setAlignment(global, nullptr, module, builder, *typeConverter);
  }

  return global;
}

// Return a pointer to the first character in a global string.
Value getPtrToGlobalString(
    const LLVM::GlobalOp &global, Location loc, OpBuilder &builder) {
  MultiDialectBuilder<LLVMBuilder> create(builder, loc);
  Type i8Type = IntegerType::get(builder.getContext(), 8);
  Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
  Type i64Type = IntegerType::get(builder.getContext(), 64);
  Value globalPtr = create.llvm.addressOf(global);
  Value zero = create.llvm.constant(i64Type, (int64_t)0);
  return create.llvm.getElemPtr(i8PtrType, globalPtr, {zero, zero});
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

/// Return a symbol reference to the strncmp function, inserting it into the
/// module if necessary.
FlatSymbolRefAttr getOrInsertStrncmp(OpBuilder &builder, ModuleOp module) {
  MultiDialectBuilder<LLVMBuilder> create(builder, module.getLoc());
  MLIRContext *ctx = module.getContext();
  Type i8Type = IntegerType::get(ctx, 8);
  Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
  // Create 'strncmp' function signature: `i32 (i8*, i8*, i64)`
  return create.llvm.getOrInsertSymbolRef(module, StringRef("strncmp"),
      builder.getI32Type(), {i8PtrTy, i8PtrTy, builder.getI64Type()});
}

std::string a2e_s(std::string a_s) {
  std::string r(a_s);
  for (unsigned int i = 0; i < r.size(); i++)
    r[i] = a2e[(int)r[i]];
  return r;
}

std::string e2a_s(std::string e_s) {
  std::string r(e_s);
  for (unsigned int i = 0; i < r.size(); i++)
    r[i] = e2a[(int)r[i]];
  return r;
}

void emitErrNo(ModuleOp module, OpBuilder &builder, Location loc, int errCode) {
  Type int32Ty = builder.getI32Type();
  Type int32PtrTy = LLVM::LLVMPointerType::get(int32Ty);
  LLVMBuilder createLLVM(builder, loc);
  LLVMBuilder createLLVMModuleLoc(builder, module.getLoc());
  // Create '__errno_location' function signature: `i32 *()`
  FlatSymbolRefAttr errnoSymbolRef = createLLVMModuleLoc.getOrInsertSymbolRef(
      module, StringRef("__errno_location"), int32PtrTy, {});
  Value errNoPos =
      createLLVM.call(int32PtrTy, errnoSymbolRef, ArrayRef<Value>({}));
  Value errNoVal = createLLVM.constant(int32Ty, (int64_t)errCode);
  createLLVM.store(errNoVal, errNoPos);
}

} // namespace krnl
} // namespace onnx_mlir
