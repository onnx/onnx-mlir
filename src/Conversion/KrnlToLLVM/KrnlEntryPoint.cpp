/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlEntryPoint.cpp - Lower KrnlEntryPointOp -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlEntryPointOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlEntryPointOpLowering : public OpRewritePattern<KrnlEntryPointOp> {
public:
  using OpRewritePattern<KrnlEntryPointOp>::OpRewritePattern;
  ArrayRef<bool> outputOMTensorOwnerships;
  bool singleEntryPoint;

  KrnlEntryPointOpLowering(TypeConverter typeConverter, MLIRContext *ctx,
      ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint)
      : OpRewritePattern<KrnlEntryPointOp>(ctx),
        outputOMTensorOwnerships(outputOMTensorOwnerships),
        singleEntryPoint(singleEntryPoint) {}

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry::build(module, rewriter);
    auto loc = op.getLoc();
    auto numOutputs = op->getAttrOfType<IntegerAttr>(
                            KrnlEntryPointOp::getNumOutputsAttrName())
                          .getInt();

    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto int64Ty = IntegerType::get(context, 64);

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
    // DEFAULT_DYN_ENTRY_POINT.
    std::string dynEntryPointName = "run_" + staticEntryPointFuncName.str();
    if (singleEntryPoint)
      dynEntryPointName = DEFAULT_DYN_ENTRY_POINT;
    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVM::LLVMFunctionType::get(opaquePtrTy, {opaquePtrTy}, false);
    auto dynamicEntryPointFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc, dynEntryPointName, dynEntryPointFuncTy);
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

    Value omTensorPtrArr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_OMT_ARRAY, {wrappedInput});
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
      auto outMemRefRank = krnl::getRankFromMemRefType(outMemRefTy);
      auto outMemRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(outMemRefRank));
      Value outOMTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {outMemRefRankVal});
      // If output is a constant tensor or a block argument, OMTensor does not
      // own it.
      bool outOwning = outputOMTensorOwnerships[i];
      LLVM_DEBUG(llvm::dbgs() << "Output OMTensor " << i
                              << " with owning = " << outOwning << "\n");
      krnl::fillOMTensorWithMemRef(
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
    Value wrappedOutput = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR_LIST, {outOmtPtrsArr, numOutput, one});

    // Return wrapped output.
    rewriter.create<LLVM::ReturnOp>(
        loc, SmallVector<Value, 1>(1, wrappedOutput));
    return success();
  }

private:
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
      const RuntimeAPIRegistry &apiRegistry, ModuleOp &module) const {
    auto *context = module.getContext();
    auto memRefPtrTy = ptrToMemRef.getType().dyn_cast<LLVM::LLVMPointerType>();
    auto memRefTy = memRefPtrTy.getElementType();
    auto int64Ty = IntegerType::get(context, 64);

    Value memRef = rewriter.create<LLVM::UndefOp>(loc, memRefTy);

    // Set dataPtr and alignedDataPtr;
    Value dataPtr = RuntimeAPI::callApi(
        rewriter, loc, apiRegistry, RuntimeAPI::API::GET_DATA, {rtMemRef});
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
    auto rank =
        krnl::getRankFromMemRefType(memRefTy.cast<LLVM::LLVMStructType>());
    Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_SHAPE, {rtMemRef});
    Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_STRIDES, {rtMemRef});

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

private:
  FlatSymbolRefAttr getOrInsertMalloc(
      PatternRewriter &rewriter, ModuleOp module) const {
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
};

void populateLoweringKrnlEntryPointOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint) {
  patterns.insert<KrnlEntryPointOpLowering>(
      typeConverter, ctx, outputOMTensorOwnerships, singleEntryPoint);
}

} // namespace krnl
} // namespace onnx_mlir
