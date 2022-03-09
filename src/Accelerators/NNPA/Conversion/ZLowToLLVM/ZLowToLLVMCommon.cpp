/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVMCommon.hpp - Lowering from ZLow to LLVM ---------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods used in lowering ZLow to LLVM
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVMCommon.hpp"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "third_party/zdnn-lib/zdnn/zdnn.h"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

ApiRegistry RegisterAllApis(MLIRContext *context) {
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto int32Ty = IntegerType::get(context, 32);
  auto int64Ty = IntegerType::get(context, 64);

  // Declare API type as an enum value, its string name and an LLVM Type
  // specifying its signature.
  //
  // Note: Though zDNN APIs use int32 for their integer parameters, we have to
  // pass them as int64 to avoid segfault when compiling with -O{1,2,3} on
  // s390x. This is an issue about use of MLIR on s390x, where int32 is carried
  // inside a 64-bit register and the higher 4 bytes are not cleared correctly.
  // More info can be found here:
  // https://github.com/onnx/onnx-mlir/pull/567#issuecomment-841061475
  //
  // clang-format off
  std::vector<ApiSpec> apiSpecs = {
    // Tensor functions
    ApiSpec(API::ZDNN_INIT_PRE_TRANSFORMED_DESC, "zdnn_init_pre_transformed_desc", voidTy, {int64Ty, int64Ty, opaquePtrTy}, true),
    ApiSpec(API::ZDNN_GENERATE_TRANSFORMED_DESC, "zdnn_generate_transformed_desc", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_GENERATE_TRANSFORMED_DESC_CONCATENATED, "zdnn_generate_transformed_desc_concatenated", int32Ty, {opaquePtrTy, int64Ty, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_GETSIZE_ZTENSOR, "zdnn_getsize_ztensor", int64Ty, {opaquePtrTy}, false),
    ApiSpec(API::ZDNN_TRANSFORM_ZTENSOR, "zdnn_transform_ztensor", int32Ty, {opaquePtrTy}, true),
    ApiSpec(API::ZDNN_TRANSFORM_ORIGTENSOR, "zdnn_transform_origtensor", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    // Elementwise operations
    ApiSpec(API::ZDNN_ADD, "zdnn_add", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_SUB, "zdnn_sub", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_MUL, "zdnn_mul", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_DIV, "zdnn_div", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_MIN, "zdnn_min", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_MAX, "zdnn_max", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_LOG, "zdnn_log", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_EXP, "zdnn_exp", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    // Activation operations
    ApiSpec(API::ZDNN_RELU, "zdnn_relu", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_TANH, "zdnn_tanh", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_SIGMOID, "zdnn_sigmoid", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_SOFTMAX, "zdnn_softmax", int32Ty, {opaquePtrTy, opaquePtrTy, int64Ty, opaquePtrTy}, false),
    // RNN operations
    ApiSpec(API::ZDNN_LSTM, "zdnn_lstm", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_GRU, "zdnn_gru", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy}, false),
    // Other operations
    ApiSpec(API::ZDNN_MATMUL_OP, "zdnn_matmul_op", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, int64Ty, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_MATMUL_BCAST_OP, "zdnn_matmul_bcast_op", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, int64Ty, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_CONV2D, "zdnn_conv2d", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, int64Ty, int64Ty, int64Ty, int64Ty, opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API::ZDNN_AVGPOOL2D, "zdnn_avgpool2d", int32Ty, {opaquePtrTy, int64Ty, int64Ty, int64Ty, int64Ty, int64Ty, opaquePtrTy}, false),
    ApiSpec(API:: ZDNN_MAXPOOL2D, "zdnn_maxpool2d", int32Ty, {opaquePtrTy, int64Ty, int64Ty, int64Ty, int64Ty, int64Ty, opaquePtrTy}, false),
    ApiSpec(API:: ZDNN_MEANREDUCE2D, "zdnn_meanreduce2d", int32Ty, {opaquePtrTy, opaquePtrTy}, false),
    ApiSpec(API:: ZDNN_BATCHNORM, "zdnn_batchnorm", int32Ty, {opaquePtrTy, opaquePtrTy, opaquePtrTy, opaquePtrTy}, false),
  };
  // clang-format on

  // Declare APIs in the current module and build an API registry mapping api.
  ApiRegistry registry;
  for (auto &apiSpec : apiSpecs) {
    registry.emplace(apiSpec.id, apiSpec);
  }

  return registry;
}

ZTensorHelper::ZTensorHelper(PatternRewriter &rewriter, Location loc,
    ModuleOp module, ApiRegistry apiRegistry)
    : rewriter(rewriter), loc(loc), module(module), apiRegistry(apiRegistry) {}

// Get a pre-transformed descriptor.
Value ZTensorHelper::getPreTransformedDescPtr(zdnn_data_types zDNNDataType,
    zdnn_data_layouts zDNNDataLayout, ArrayRef<Value> dims) {
  auto *context = module.getContext();
  size_t rank = dims.size();

  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmZTensorDescStructTy = getZTensorDescStructTy(context);
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(1));

  Value preTransformedDescPtr = rewriter.create<LLVM::AllocaOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorDescStructTy), one,
      /*alignment=*/0);

  // Prepare operands for calling the function that initializes the zTensor
  // descriptor.
  SmallVector<Value, 4> operands;
  // 1. Data layout.
  auto dataLayout = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI64Ty, rewriter.getI64IntegerAttr(zDNNDataLayout));
  operands.emplace_back(dataLayout);
  // 2. Data type.
  auto dataType = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI64Ty, rewriter.getI64IntegerAttr(zDNNDataType));
  operands.emplace_back(dataType);
  // 3. Tensor descriptor.
  operands.emplace_back(
      toOpaquePtr(rewriter, loc, module, preTransformedDescPtr));
  // 4. Dimensions.
  // In zDNN, the order of dims is a logical layout of outermost to
  // innermost.
  for (size_t i = 0; i < rank; ++i) {
    operands.emplace_back(dims[i]);
  }

  // Call a zDNN function to initialize the descriptor.
  callApi(rewriter, loc, module, apiRegistry,
      API::ZDNN_INIT_PRE_TRANSFORMED_DESC, operands);

  return preTransformedDescPtr;
}

// Get a transformed descriptor.
Value ZTensorHelper::getTransformedDescPtr(
    Value preTransformedDescPtr, bool isConcat, zdnn_concat_info concatInfo) {
  auto *context = module.getContext();

  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmZTensorDescStructTy = getZTensorDescStructTy(context);
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(1));

  Value transformedDescPtr = rewriter.create<LLVM::AllocaOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorDescStructTy), one,
      /*alignment=*/0);

  if (isConcat) {
    auto concatLayout = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Ty, rewriter.getI64IntegerAttr(concatInfo));
    callApi(rewriter, loc, module, apiRegistry,
        API::ZDNN_GENERATE_TRANSFORMED_DESC_CONCATENATED,
        {toOpaquePtr(rewriter, loc, module, preTransformedDescPtr),
            concatLayout,
            toOpaquePtr(rewriter, loc, module, transformedDescPtr)});
  } else
    callApi(rewriter, loc, module, apiRegistry,
        API::ZDNN_GENERATE_TRANSFORMED_DESC,
        {toOpaquePtr(rewriter, loc, module, preTransformedDescPtr),
            toOpaquePtr(rewriter, loc, module, transformedDescPtr)});
  return transformedDescPtr;
}

// Get the pointer to memref.
Value ZTensorHelper::getAlignedI8Ptr(Value memRef) {
  auto *context = module.getContext();
  MemRefDescriptor descriptor(memRef);
  Value alignedPtr = descriptor.alignedPtr(rewriter, loc);
  auto llvmI8Ty = IntegerType::get(context, 8);
  Value alignedI8Ptr = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(llvmI8Ty), alignedPtr);
  return alignedI8Ptr;
}

// Get buffer size from a transformed descriptor.
Value ZTensorHelper::getBufferSize(Value transformedDescPtr) {
  Value opaqueZTensorDesc =
      toOpaquePtr(rewriter, loc, module, transformedDescPtr);
  Value bufferSize = callApi(rewriter, loc, module, apiRegistry,
      API::ZDNN_GETSIZE_ZTENSOR, {opaqueZTensorDesc});
  return bufferSize;
}

/// Create a zTensor.
/// TODO: support concatInfo.
ZTensor ZTensorHelper::getZTensor(Value bufferPtr, zdnn_data_types dataType,
    zdnn_data_layouts layout, ArrayRef<Value> originalDims, bool isTransformed,
    bool isConcat, zdnn_concat_info concatInfo) {
  auto *context = module.getContext();
  ZTensor zTensor;

  // LLVM types for zTensor and zTensor descriptor.
  auto llvmZTensorStructTy = getZTensorStructTy(context);
  // Some frequently used constants.
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(1));

  // Create a pre transformed descriptor.
  Value preTransformedDescPtr =
      getPreTransformedDescPtr(dataType, layout, originalDims);
  // Create a transformed descriptor.
  Value transformedDescPtr =
      getTransformedDescPtr(preTransformedDescPtr, isConcat, concatInfo);
  // Create the input zTensor.
  Value alloc = rewriter.create<LLVM::AllocaOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorStructTy), one,
      /*alignment=*/0);
  // Buffer size.
  Value bufferSize = getBufferSize(transformedDescPtr);
  // clang-format off
  fillInZTensor(rewriter, loc, module, alloc,
                /*preTransformedDescPtr=*/preTransformedDescPtr,
                /*transformedDescPtr=*/transformedDescPtr,
                /*isTransformed=*/isTransformed,
                /*bufferSize=*/bufferSize,
                /*alignedBuffer=*/bufferPtr);
  // clang-format on

  zTensor.val = alloc;
  zTensor.preTransformedDescPtr = preTransformedDescPtr;
  zTensor.transformedDescPtr = transformedDescPtr;
  zTensor.isTransformed = isTransformed;
  zTensor.bufferSize = bufferSize;
  zTensor.bufferPtr = bufferPtr;
  return zTensor;
}

/// Create a zTensor from existing descriptors.
ZTensor ZTensorHelper::getZTensor(Value preTransformedDescPtr,
    Value transformedDescPtr, Value bufferSize, Value bufferPtr,
    bool isTransformed) {
  auto *context = module.getContext();
  ZTensor zTensor;

  auto llvmZTensorStructTy = getZTensorStructTy(context);
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(1));

  Value alloc = rewriter.create<LLVM::AllocaOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorStructTy), one,
      /*alignment=*/0);
  // clang-format off
  fillInZTensor(rewriter, loc, module, alloc,
                /*preTransformedDescPtr=*/preTransformedDescPtr,
                /*transformedDescPtr=*/transformedDescPtr,
                /*isTransformed=*/isTransformed,
                /*bufferSize=*/bufferSize,
                /*alignedBuffer=*/bufferPtr);
  // clang-format on

  zTensor.val = alloc;
  zTensor.preTransformedDescPtr = preTransformedDescPtr;
  zTensor.transformedDescPtr = transformedDescPtr;
  zTensor.isTransformed = isTransformed;
  zTensor.bufferSize = bufferSize;
  zTensor.bufferPtr = bufferPtr;
  return zTensor;
}

/// Search for a function reference. Insert a new function reference if not
/// found.
FlatSymbolRefAttr getOrInsertExternFuncRef(PatternRewriter &rewriter,
    ModuleOp module, StringRef funcName, Type funcType) {
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

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
Value callApi(PatternRewriter &rewriter, Location loc, ModuleOp module,
    ApiRegistry registry, API apiId, ArrayRef<Value> params) {
  // To be used as parameters in LLVM::CallOp, voidTy must be converted
  // to empty list to avoid emission of an SSA value with voidTy. However,
  // we still keep using LLVM voidTy (as opposed to empty list) when recording
  // API function signatures in API registry because when declaring API
  // functions in LLVM IR, the correct way to indicate an output type for
  // "void" is still LLVM voidTy. Relevant discussion thread:
  // https://github.com/onnx/onnx-mlir/issues/255.
  SmallVector<Type, 1> outputTys;
  auto apiSpec = registry.at(apiId);
  auto symbolRef = getOrInsertExternFuncRef(
      rewriter, module, apiSpec.name, apiSpec.funcTy());
  auto outputTy = apiSpec.outputTy;
  if (!outputTy.isa<LLVM::LLVMVoidType>())
    outputTys.emplace_back(outputTy);
  auto returnVals = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>(outputTys), symbolRef, ArrayRef<Value>(params));
  if (returnVals.getNumResults() == 1)
    return returnVals.getResult(0);
  return nullptr;
}

size_t getRankFromMemRefType(LLVM::LLVMStructType memRefTy) {
  // Usually a MemRef is a 5-element struct, where the 4th and 5th elements in
  // this struct are arrays whose size is the rank of the tensor. In the event
  // that the corresponding tensor of this MemRef is a scalar, the 4th and 5th
  // elements will have 0-length, which in turn causes the MemRef struct to
  // degenerate into a 3-element struct. For more information, refer to
  // https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/MemRef.md.
  auto numElems = memRefTy.getBody().size();
  assert((numElems == 3 || numElems == 5) &&
         "Expect MemRef type to contain either 3 or 5 elements.");

  if (numElems == 3)
    return 0; // MemRef refers to a scalar.
  else
    return memRefTy.getBody()[3].cast<LLVM::LLVMArrayType>().getNumElements();
}

/// Get a vector of 'size' dimensions from a 1D DenseElementsAttr.
/// zDNN uses i32 for dimensions.
std::vector<Value> getDimsFromDenseElementsAttr(PatternRewriter &rewriter,
    Location loc, ModuleOp module, DenseElementsAttr valueAttr, unsigned size) {
  assert(valueAttr.getNumElements() == size);
  auto *context = module.getContext();
  std::vector<Value> dims;
  auto valueIt = valueAttr.getValues<IntegerAttr>().begin();
  for (unsigned int i = 0; i < size; ++i) {
    int64_t dim = (*valueIt++).cast<IntegerAttr>().getInt();
    Value dimVal = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(context, 64), rewriter.getI64IntegerAttr(dim));
    dims.emplace_back(dimVal);
  }
  return dims;
}

/// Get a vector of 'size' dimensions from a 1D MemRef of shape.
std::vector<Value> getDimsFromShapeMemRefBySize(PatternRewriter &rewriter,
    Location loc, ModuleOp module, Value shapeMemRef, unsigned size) {
  auto *context = module.getContext();

  // If shapeMemRef is produced by a constant op, then create values from the
  // constant values.
  // The constant op is expected to be a MemRef constructed from a LLVMGlobalOp.
  Operation *op = shapeMemRef.getDefiningOp();
  Value globalMemRef = nullptr;
  // A MemRef which is not scalar must have 5 fields.
  for (int i = 0; i < 5; ++i) {
    LLVM::InsertValueOp insertValueOp =
        llvm::dyn_cast_or_null<LLVM::InsertValueOp>(op);
    if (insertValueOp) {
      op = insertValueOp.getContainer().getDefiningOp();
      if (i == 4) {
        // Check UndefOp which defines a MemRef.
        LLVM::UndefOp undefOp = llvm::dyn_cast_or_null<LLVM::UndefOp>(op);
        if (undefOp) {
          globalMemRef = insertValueOp.getValue();
        }
      }
    }
  }
  // Check if the MemRef is constructed from a LLVMGlobaOp.
  // According to KrnlToLLVM in onnx-mlir, a LLVMGlobalOp is accessed by:
  // 1. Getting the address of the global, using AddressOfOp, and
  // 2. Casting the address pointer to element type, using BitCastOp.
  // Thus, we check these operations.
  if (globalMemRef) {
    op = globalMemRef.getDefiningOp();
    LLVM::BitcastOp bitcastOp = llvm::dyn_cast_or_null<LLVM::BitcastOp>(op);
    if (bitcastOp) {
      LLVM::AddressOfOp addressOfOp = llvm::dyn_cast_or_null<LLVM::AddressOfOp>(
          bitcastOp.getArg().getDefiningOp());
      if (addressOfOp) {
        LLVM::GlobalOp globalOp = addressOfOp.getGlobal();
        DenseElementsAttr valueAttr =
            globalOp.getValue().getValue().dyn_cast<DenseElementsAttr>();
        if (valueAttr)
          return getDimsFromDenseElementsAttr(
              rewriter, loc, module, valueAttr, size);
      }
    }
  }

  // If shapeMemRef is not produced by a constant op, read values from MemRef.
  std::vector<Value> dims;
  MemRefDescriptor inputMRD(shapeMemRef);
  auto alignedPtr = inputMRD.alignedPtr(rewriter, loc);
  auto int64Ty = IntegerType::get(context, 64);
  for (int64_t i = 0; i < size; ++i) {
    auto index = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(i));
    Value alignedGep = rewriter.create<LLVM::GEPOp>(loc,
        LLVM::LLVMPointerType::get(int64Ty), alignedPtr, ValueRange({index}));
    Value dimI64 = rewriter.create<LLVM::LoadOp>(loc, alignedGep);
    dims.emplace_back(dimI64);
  }
  return dims;
}

/// Get a vector of dimensions from a 1D MemRef of shape, using layout.
std::vector<Value> getDimsFromShapeMemRef(PatternRewriter &rewriter,
    Location loc, ModuleOp module, Value shapeMemRef, unsigned layout) {
  int dimsCount = -1;
  if (layout == ZDNN_1D)
    dimsCount = 1;
  else if (layout == ZDNN_2D || layout == ZDNN_2DS)
    dimsCount = 2;
  else if (layout == ZDNN_3D || layout == ZDNN_3DS || layout == ZDNN_ZRH ||
           layout == ZDNN_BIDIR_ZRH)
    dimsCount = 3;
  else if (layout == ZDNN_4D || layout == ZDNN_4DS || layout == ZDNN_NHWC ||
           layout == ZDNN_NCHW || layout == ZDNN_HWCK || layout == ZDNN_FICO ||
           layout == ZDNN_BIDIR_FICO)
    dimsCount = 4;
  else
    llvm_unreachable("Unsupported data layout");

  return getDimsFromShapeMemRefBySize(
      rewriter, loc, module, shapeMemRef, dimsCount);
}

/// Get dimensions from a MemRef value.
void getDimsFromMemRef(PatternRewriter &rewriter, Location loc, ModuleOp module,
    Value memRef, SmallVectorImpl<Value> &dims) {
  MemRefDescriptor memRefDesc(memRef);
  size_t rank =
      getRankFromMemRefType(memRef.getType().cast<LLVM::LLVMStructType>());
  for (size_t i = 0; i < rank; i++) {
    auto dimI64 = memRefDesc.size(rewriter, loc, i);
    dims.emplace_back(dimI64);
  }
}

/// Type conversion from LLVMType to zDNNType.
/// TODO: fill in the complete list of the zDNN types.
zdnn_data_types llvmTypeToZDNNType(Type elemType) {
  if (elemType.isa<Float16Type>())
    return FP16;
  else if (elemType.isa<Float32Type>())
    return FP32;
  else
    llvm_unreachable("Unexpected LLVM type, cannot be converted to zDNN type.");
}

/// Function to create a zTensor descriptor struct type.
Type getZTensorDescStructTy(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);

  SmallVector<Type, 4> zTensorDescTypeElements;
  // data layout
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // data format
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // data type
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // dim4: number of elements in outermost dimension
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // dim3: ... outer dimension
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // dim2: ... inner dimension
  zTensorDescTypeElements.emplace_back(llvmI32Ty);
  // dim1: number of elements in innermost dimension
  zTensorDescTypeElements.emplace_back(llvmI32Ty);

  auto zTensorDescStructTy = LLVM::LLVMStructType::getLiteral(context,
      /*elements=*/zTensorDescTypeElements,
      /*isPacked=*/false);
  return zTensorDescStructTy;
}

/// Function to return the size (in bytes) of a zTensor descriptor struct.
size_t getZTensorDescStructSizeInBytes(MLIRContext *context, Type descTy) {
  // A zTensor descriptor struct consists of seven i32 values.
  return (7 * IntegerType::get(context, 32).getWidth() / 8);
}

/// Function to create a zTensor struct type.
Type getZTensorStructTy(MLIRContext *context) {
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmI1Ty = IntegerType::get(context, 1);
  auto llvmI8Ty = IntegerType::get(context, 8);
  auto llvmArrayI8Ty = LLVM::LLVMArrayType::get(llvmI8Ty, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
  auto llvmZTensorDescStructTy = getZTensorDescStructTy(context);

  SmallVector<Type, 4> zTensorTypeElements;
  // A pointer to pre-transformed descriptor struct type
  zTensorTypeElements.emplace_back(
      LLVM::LLVMPointerType::get(llvmZTensorDescStructTy));
  // A pointer to transformed descriptor struct type
  zTensorTypeElements.emplace_back(
      LLVM::LLVMPointerType::get(llvmZTensorDescStructTy));
  // zTensor size in bytes
  zTensorTypeElements.emplace_back(llvmI64Ty);
  // pointer to the zTensor in memory
  zTensorTypeElements.emplace_back(llvmI8PtrTy);
  // indicator if data in buffer has been transformed
  zTensorTypeElements.emplace_back(llvmI1Ty);
  // reserved[32], not currently used, exploiter should not touch
  zTensorTypeElements.emplace_back(llvmArrayI8Ty);

  auto zTensorStructTy = LLVM::LLVMStructType::getLiteral(context,
      /*elements=*/zTensorTypeElements,
      /*isPacked=*/false);
  return zTensorStructTy;
}

/// Function to cast an LLVM pointer to an opaque LLVM pointer.
Value toOpaquePtr(
    PatternRewriter &rewriter, Location loc, ModuleOp module, Value ptr) {
  auto llvmI8Ty = IntegerType::get(module.getContext(), 8);
  auto llvmOpaqueTy = LLVM::LLVMPointerType::get(llvmI8Ty);
  Value opaquePtr = rewriter.create<LLVM::BitcastOp>(loc, llvmOpaqueTy, ptr);
  return opaquePtr;
}

void fillInZTensor(PatternRewriter &rewriter, Location loc, ModuleOp module,
    Value zTensor, Value preTransformedDescPtr, Value transformedDescPtr,
    bool isTransformed, Value bufferSize, Value alignedBuffer) {
  auto *context = module.getContext();

  auto llvmI1Ty = IntegerType::get(context, 1);
  auto llvmI8Ty = IntegerType::get(context, 8);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(llvmI8Ty);
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI64Ty = IntegerType::get(context, 64);
  auto llvmZTensorDescTy =
      LLVM::LLVMPointerType::get(getZTensorDescStructTy(context));

  // Got runtime error if using i64 as index to access zTensor. It looks
  // like an error in MLIR. So use i32 here, which does not affect the
  // correctness of the generated program.
  auto zero = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32Ty, rewriter.getI32IntegerAttr(0));
  auto one = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32Ty, rewriter.getI32IntegerAttr(1));
  auto two = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32Ty, rewriter.getI32IntegerAttr(2));
  auto three = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32Ty, rewriter.getI32IntegerAttr(3));
  auto four = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32Ty, rewriter.getI32IntegerAttr(4));

  // 1. Set pre-transformed descriptor.
  auto zTensorPreTransformedDescPtr = rewriter.create<LLVM::GEPOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorDescTy), zTensor,
      ArrayRef<Value>({zero, zero}));
  rewriter.create<LLVM::StoreOp>(
      loc, preTransformedDescPtr, zTensorPreTransformedDescPtr);

  // 2. Set transformed descriptor.
  auto zTensorTransformedDescPtr = rewriter.create<LLVM::GEPOp>(loc,
      LLVM::LLVMPointerType::get(llvmZTensorDescTy), zTensor,
      ArrayRef<Value>({zero, one}));
  rewriter.create<LLVM::StoreOp>(
      loc, transformedDescPtr, zTensorTransformedDescPtr);

  // 3. Set buffer_size.
  auto bufferSizePtr =
      rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(llvmI64Ty),
          zTensor, ArrayRef<Value>({zero, two}));
  rewriter.create<LLVM::StoreOp>(loc, bufferSize, bufferSizePtr);

  // 4. Set buffer. Buffer was allocated in advance by the stickified memref.
  // So get the pointer from the stickified memref and set it to the zTensor.
  auto bufferPtr =
      rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(llvmI8PtrTy),
          zTensor, ArrayRef<Value>({zero, three}));
  rewriter.create<LLVM::StoreOp>(loc, alignedBuffer, bufferPtr);

  // 5. Set is_transformed.
  auto isTransformedVal = rewriter.create<LLVM::ConstantOp>(loc, llvmI1Ty,
      rewriter.getIntegerAttr(
          rewriter.getIntegerType(1), (isTransformed) ? 1 : 0));
  auto isTransformedDescPtr =
      rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(llvmI1Ty),
          zTensor, ArrayRef<Value>({zero, four}));
  rewriter.create<LLVM::StoreOp>(loc, isTransformedVal, isTransformedDescPtr);

  // 6. Set reserved (not currently used), not touch
}

} // namespace zlow
} // namespace onnx_mlir
