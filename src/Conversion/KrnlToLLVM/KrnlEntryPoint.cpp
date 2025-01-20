/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlEntryPoint.cpp - Lower KrnlEntryPointOp -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlEntryPointOp operator.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

extern uint64_t KRNL_ENTRY_POINT_ID;

class KrnlEntryPointOpLowering : public OpRewritePattern<KrnlEntryPointOp> {
public:
  using OpRewritePattern<KrnlEntryPointOp>::OpRewritePattern;
  ArrayRef<bool> outputOMTensorOwnerships;
  bool singleEntryPoint;
  SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps;
  SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps;
  SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps;
  std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes;
  std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes;
  bool verifyInputTensors;

  KrnlEntryPointOpLowering(LLVMTypeConverter &typeConverter, MLIRContext *ctx,
      ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
      SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
      std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
      std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes,
      bool verifyInputTensors)
      : OpRewritePattern<KrnlEntryPointOp>(ctx),
        outputOMTensorOwnerships(outputOMTensorOwnerships),
        singleEntryPoint(singleEntryPoint), entryGlobalOps(entryGlobalOps),
        inSigGlobalOps(inSigGlobalOps), outSigGlobalOps(outSigGlobalOps),
        inputMemRefTypes(inputMemRefTypes),
        outputMemRefTypes(outputMemRefTypes),
        verifyInputTensors(verifyInputTensors), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, typeConverter);

    // Common information.
    StringAttr sigAttr =
        op->getAttrOfType<StringAttr>(KrnlEntryPointOp::getSignatureAttrName());
    auto numOutputs = op->getAttrOfType<IntegerAttr>(
                            KrnlEntryPointOp::getNumOutputsAttrName())
                          .getInt();

    // Common types.
    Type int8Ty = IntegerType::get(context, 8);
    Type opaquePtrTy = getPointerType(context, int8Ty);
    Type int64Ty = IntegerType::get(context, 64);
    Type omTensorPtrAddrTy = getPointerType(context, opaquePtrTy);

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

    // Record entry point name, input and output signatures in order to emit
    // signature-related functions later.
    recordEntryPointSignatures(module, dynEntryPointName, op, entryGlobalOps,
        inSigGlobalOps, outSigGlobalOps);
    // Record the postfixed entry point name if available.
    if (singleEntryPoint) {
      std::string dynEntryPointPostfixName =
          LLVMBuilder::SymbolPostfix(module, dynEntryPointName);
      if (dynEntryPointPostfixName != dynEntryPointName)
        recordEntryPointSignatures(module, dynEntryPointPostfixName, op,
            entryGlobalOps, inSigGlobalOps, outSigGlobalOps);
    }

    // If `useOpaquePointers=true` in LowerToLLVMOptions, all memref arguments
    // are converted to opaque types, e.g. `!llvm.ptr`, so we lost the
    // struct information of memref arguments, e.g. element type. To set data
    // type for OMTensor correctly, we have to obtain element types from
    // original MemRefTypes.
    SmallVector<MemRefType, 4> origInputMemRefTypes =
        inputMemRefTypes[staticEntryPointFuncName.str()];
    SmallVector<MemRefType, 4> origOutputMemRefTypes =
        outputMemRefTypes[staticEntryPointFuncName.str()];

    // Start lowering the op.
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVM::LLVMFunctionType::get(opaquePtrTy, {opaquePtrTy}, false);
    LLVM::LLVMFuncOp dynamicEntryPointFunc;
    dynamicEntryPointFunc = create.llvm.func(dynEntryPointName,
        dynEntryPointFuncTy, /*createUniqueFunc=*/singleEntryPoint);
    auto &entryPointEntryBlock =
        createEntryBlock(dynEntryPointFuncTy, dynamicEntryPointFunc, loc);
    rewriter.setInsertionPointToStart(&entryPointEntryBlock);
    // User's OMTensor inputs.
    auto omTensorInputs = entryPointEntryBlock.getArgument(0);

    // 1. Emit code to initialize accelerators by calling OMInitCompatibleAccelX
    // where X is the accelerator name.
    // OMInitCompatibleAccelX's signature is `i64 (i64)`.
    if (Attribute maccelAttr =
            module->getAttrOfType<::mlir::Attribute>("onnx-mlir.accels")) {
      assert(mlir::isa<ArrayAttr>(maccelAttr) &&
             "onnx-mlir.accels must be ArrayAttr");
      ArrayAttr accels = mlir::cast<ArrayAttr>(maccelAttr);
      Value zeroI64 = create.llvm.constant(int64Ty, static_cast<int64_t>(0));

      for (uint64_t i = 0; i < accels.size(); ++i) {
        assert(
            mlir::isa<StringAttr>(accels[i]) && "Attribute must be StringAttr");
        StringRef accelStr =
            mlir::cast<StringAttr>(accels.getValue()[i]).getValue();
        std::pair<StringRef, StringRef> NameAndVersion = accelStr.split('-');
        uint64_t versionNumberInHex =
            std::stoul(NameAndVersion.second.str(), nullptr, 16);
        FlatSymbolRefAttr funcRef = getOrInsertOMInitCompatibleAccel(
            rewriter, module, NameAndVersion.first);

        // Emit code for `if (OMInitCompatibleAccelX() == 0) then return NULL`.
        create.llvm.ifThenElse(/*cond=*/
            [&](const LLVMBuilder &createLLVM) {
              // Call OMInitCompatibleAccelX.
              Value versionNumberVal = createLLVM.constant(
                  int64Ty, static_cast<int64_t>(versionNumberInHex));
              Value isCompatible = createLLVM.call(
                  int64Ty, funcRef, ArrayRef<Value>({versionNumberVal}));
              // Condition: if (OMInitCompatibleAccelX() == 0)
              return createLLVM.icmp(
                  LLVM::ICmpPredicate::eq, isCompatible, zeroI64);
            }, /*then=*/
            [&](const LLVMBuilder &createLLVM) {
              // return NULL.
              createLLVM._return(createLLVM.null(getI8PointerType(context)));
            });
      }
    }

    // 2. Emit code to verify every tensor in the wrapped input, e.g. verifying
    // shape and data type.
    if (verifyInputTensors) {
      llvm::StringRef inSigJSON;
      std::tie(inSigJSON, std::ignore) = sigAttr.getValue().split('@');
      emitVerificationCodeForInputTensors(
          module, rewriter, loc, apiRegistry, omTensorInputs, inSigJSON);
    }

    // 3. Emit code to prepare MemRefs from OMTensor inputs and call
    // `_mlir_ciface` prefixed function of the entry point.

    // Based on the static entry point type signature, unpack dynamic memory
    // refs to corresponding static memory refs.
    // Note that, in the static entry point type signature, output type is a
    // struct but input types are unpacked into a single list of scalar types.
    auto *staticEntryPointFunc =
        module.lookupSymbol(staticEntryPointFuncName.lower());
    auto staticEntryPointFuncTy = mlir::cast<LLVM::LLVMFunctionType>(
        mlir::cast<LLVM::LLVMFuncOp>(staticEntryPointFunc).getFunctionType());
    LLVM_DEBUG(llvm::dbgs() << "Static entry point function type: "
                            << staticEntryPointFuncTy << "\n");
    // Static entry point is wrapped with prefix `_mlir_ciface` automatically by
    // MLIR when being converted to LLVM, where memref arguments are converted
    // to pointer-to-struct.
    auto wrappedStaticEntryPointFuncName =
        "_mlir_ciface_" + staticEntryPointFuncName.lower();
    auto *wrappedStaticEntryPointFunc =
        module.lookupSymbol(wrappedStaticEntryPointFuncName);
    assert(wrappedStaticEntryPointFunc &&
           isa<LLVM::LLVMFuncOp>(wrappedStaticEntryPointFunc) &&
           "entry point func must exist and be an llvm func op");
    auto wrappedStaticEntryPointOp =
        mlir::cast<LLVM::LLVMFuncOp>(wrappedStaticEntryPointFunc);
    auto wrappedStaticEntryPointTy = mlir::cast<LLVM::LLVMFunctionType>(
        wrappedStaticEntryPointOp.getFunctionType());

    Value omTensorPtrArr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_OMT_ARRAY, {omTensorInputs});
    Value one = create.llvm.constant(int64Ty, static_cast<int64_t>(1));

    // Prepare MemRefs as inputs for the wrapped static entry point function.
    // MemRefs are filled with information from user' OMTensor inputs.
    SmallVector<Value, 4> staticInputs;

    // Create a memref type for the return argument of the iface call.
    // The struct information of outputs will be obtained from the static
    // entry point instead of the wrapped static entry point.
    Type memRefOutTy = staticEntryPointFuncTy.getReturnTypes()[0];
    Type memRefOutPtrTy = getPointerType(context, memRefOutTy);
    Value ptrToOutMemRef = // alloca ok as there is only one entry point.
        create.llvm._alloca(memRefOutPtrTy, memRefOutTy, one, /*alignment=*/0);
    staticInputs.emplace_back(ptrToOutMemRef);

    // Start with param 1 because 0 is the return value.
    for (size_t i = 1; i < wrappedStaticEntryPointTy.getNumParams(); i++) {
      // Call API function to retrieve the i-th dynamic memref.
      Value omTensorPtrAddr =
          create.llvm.getElemPtr(omTensorPtrAddrTy, opaquePtrTy, omTensorPtrArr,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i) - 1});
      Value omTensorPtr = create.llvm.load(opaquePtrTy, omTensorPtrAddr);

      // Create a (static) memref type corresponding to the i-th memref input to
      // the inference function on stack, and load it to memRef.
      // Original input is shifted by 1 in the iface func.
      Type memRefInTy = typeConverter.convertType(origInputMemRefTypes[i - 1]);
      Type memRefInPtrTy = getPointerType(context, memRefInTy);
      Value ptrToMemRef = // alloca ok as there is only one entry point.
          create.llvm._alloca(memRefInPtrTy, memRefInTy, one, /*alignment=*/0);

      // Fill in the memref underlying ptrToMemRef with information extracted
      // from omTensorPtr.
      fillPtrToMemRefWithOMTensor(omTensorPtr, ptrToMemRef, memRefInTy,
          rewriter, loc, apiRegistry, module);

      // ptrToMemRef will be an input to main computation graph function.
      staticInputs.emplace_back(ptrToMemRef);
    }

    // Call the wrapped static entry point with the memref ptrs created, and get
    // output.
    create.llvm.call({}, wrappedStaticEntryPointFuncName, staticInputs);
    Value outMemRefs = create.llvm.load(memRefOutTy, ptrToOutMemRef);
    auto outMemRefsType =
        mlir::dyn_cast<LLVM::LLVMStructType>(outMemRefs.getType());

    std::vector<Value> outMemRefList;
    if (numOutputs == 1) {
      // If only one output tensor exists, the tensor's corresponding memref
      // descriptor will be returned as is.
      outMemRefList.emplace_back(outMemRefs);
    } else {
      // Otherwise, if multiple tensors are to be returned, the returned value
      // is a struct. Multiple tensors' memref descriptors are packed into the
      // same struct. So we unpack them iteratively to outMemRefList.
      for (int i = 0; i < numOutputs; i++) {
        Type type = outMemRefsType.getBody()[i];
        Value extractOp = create.llvm.extractValue(type, outMemRefs, {i});
        outMemRefList.emplace_back(extractOp);
      }
    }

    Value numOutput = create.llvm.constant(
        int64Ty, static_cast<int64_t>(outMemRefList.size()));
    // Assume that OMTensor pointer size is 8.
    // Alloca ok as its only for 1 small data structure per parameters.
    Value outOmtPtrsArr = create.llvm._alloca(
        omTensorPtrAddrTy, opaquePtrTy, numOutput, /*alignment=*/0);

    for (unsigned int i = 0; i < outMemRefList.size(); i++) {
      // Get the i-th memref returned, convert to a dynamic memref and store it
      // in the wrappedOutput.
      Value memRef = outMemRefList.at(i);
      auto outMemRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(memRef.getType());
      int64_t outMemRefRank = krnl::getRankFromMemRefType(outMemRefTy);
      Value outMemRefRankVal =
          create.llvm.constant(int64Ty, static_cast<int64_t>(outMemRefRank));
      Value outOMTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {outMemRefRankVal});
      // If output is a constant tensor or a block argument, OMTensor does not
      // own it.
      bool outOwning = outputOMTensorOwnerships[i];
      LLVM_DEBUG(llvm::dbgs() << "Output OMTensor " << i
                              << " with owning = " << outOwning << "\n");
      krnl::fillOMTensorWithMemRef(memRef,
          origOutputMemRefTypes[i].getElementType(), outOMTensor, outOwning,
          rewriter, loc, apiRegistry, module);

      Value omTensorPtrAddr =
          create.llvm.getElemPtr(omTensorPtrAddrTy, opaquePtrTy, outOmtPtrsArr,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
      create.llvm.store(outOMTensor, omTensorPtrAddr);
    }

    // Create OMTensor outputs.
    Value omTensorOutputs = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR_LIST, {outOmtPtrsArr, numOutput});

    // Return wrapped output.
    create.llvm._return(omTensorOutputs);
    return success();
  }

private:
  // Helper function to insert an entry block to LLVM function.
  // (TODO): upstream this to MLIR.
  Block &createEntryBlock(Type &dynEntryPoint,
      LLVM::LLVMFuncOp &dynamicEntryPointFunc, Location &loc) const {
    // Add entry block:
    auto *entryPointEntryBlock = new Block();
    auto dynEntryPointFuncType =
        mlir::cast<LLVM::LLVMFunctionType>(dynEntryPoint);
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
      Type memRefTy, PatternRewriter &rewriter, const Location &loc,
      const RuntimeAPIRegistry &apiRegistry, ModuleOp &module) const {
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    MLIRContext *context = module.getContext();
    auto int64Ty = IntegerType::get(context, 64);

    Value memRef = rewriter.create<LLVM::UndefOp>(loc, memRefTy);

    // Set dataPtr and alignedDataPtr;
    Value dataPtr = RuntimeAPI::callApi(
        rewriter, loc, apiRegistry, RuntimeAPI::API::GET_DATA, {rtMemRef});
    dataPtr = create.llvm.bitcast(
        mlir::cast<LLVM::LLVMStructType>(memRefTy).getBody()[0], dataPtr);
    memRef = create.llvm.insertValue(memRefTy, memRef, dataPtr, {0});
    memRef = create.llvm.insertValue(memRefTy, memRef, dataPtr, {1});

    // Use zero offset now.
    Value zero = create.llvm.constant(int64Ty, static_cast<int64_t>(0));
    memRef = create.llvm.insertValue(memRefTy, memRef, zero, {2});

    // Get rank, sizes array ptr and strides array ptr.
    auto rank =
        krnl::getRankFromMemRefType(mlir::cast<LLVM::LLVMStructType>(memRefTy));
    Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_SHAPE, {rtMemRef});
    Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_STRIDES, {rtMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      // Insert size of the dimension.
      Value dimSizePtr =
          create.llvm.getElemPtr(getPointerType(context, int64Ty), int64Ty,
              sizesArrayPtr, ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
      Value dimSize = create.llvm.load(int64Ty, dimSizePtr);
      memRef = create.llvm.insertValue(memRefTy, memRef, dimSize, {3, i});

      // Insert stride of the dimension.
      auto dimStridePtr =
          create.llvm.getElemPtr(getPointerType(context, int64Ty), int64Ty,
              stridesArrayPtr, ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
      auto dimStride = create.llvm.load(int64Ty, dimStridePtr);
      memRef = create.llvm.insertValue(memRefTy, memRef, dimStride, {4, i});
    }

    create.llvm.store(memRef, ptrToMemRef);
  }

  FlatSymbolRefAttr getOrInsertOMInitAccel(
      PatternRewriter &rewriter, ModuleOp module, StringRef accelName) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    std::string funcName = "OMInitAccel" + accelName.str();
    // OMInitAccelX's signature is `void ()`.
    MLIRContext *ctx = rewriter.getContext();
    return create.llvm.getOrInsertSymbolRef(
        module, StringRef(funcName), LLVM::LLVMVoidType::get(ctx), {});
  }

  FlatSymbolRefAttr getOrInsertOMInitCompatibleAccel(
      PatternRewriter &rewriter, ModuleOp module, StringRef accelName) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    std::string funcName = "OMInitCompatibleAccel" + accelName.str();
    // OMInitCompatibleAccelX's signature is `i64 (i64)`.
    return create.llvm.getOrInsertSymbolRef(module, StringRef(funcName),
        rewriter.getI64Type(), {rewriter.getI64Type()});
  }

  void emitVerificationCodeForInputTensors(ModuleOp &module,
      PatternRewriter &rewriter, Location loc,
      const RuntimeAPIRegistry &apiRegistry, Value omTensorInputs,
      StringRef inSigJSON) const {
    MLIRContext *context = rewriter.getContext();
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    Type int64Ty = rewriter.getI64Type();
    Type opaquePtrTy = getPointerType(context, rewriter.getI8Type());

    auto JSONInput = llvm::json::parse(inSigJSON.data());
    assert(JSONInput && "failed to parse json");
    auto JSONArray = JSONInput->getAsArray();
    assert(JSONArray && "failed to parse json as array");
    int64_t inputNum = JSONArray->size();

    // Verify the number of inputs.
    equalOrFailed(module, rewriter, loc,
        create.llvm.constant(int64Ty, static_cast<int64_t>(inputNum)),
        RuntimeAPI::callApi(rewriter, loc, apiRegistry,
            RuntimeAPI::API::GET_OMTENSOR_LIST_SIZE, {omTensorInputs}),
        "Wrong number of input tensors: expect " + std::to_string(inputNum) +
            ", but got ");

    // Get a pointer to the list of input omTensors.
    Value omTensorPtrArr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_OMT_ARRAY, {omTensorInputs});
    for (int64_t i = 0; i < inputNum; ++i) {
      // Call API function to retrieve the i-th omTensor.
      Value omTensorPtrAddr = create.llvm.getElemPtr(
          getPointerType(context, opaquePtrTy), opaquePtrTy, omTensorPtrArr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
      Value omTensorPtr = create.llvm.load(opaquePtrTy, omTensorPtrAddr);

      // Verify data type.
      auto JSONItem = (*JSONArray)[i].getAsObject();
      auto JSONItemType = JSONItem->getString("type");
      assert(JSONItemType && "failed to get type");
      Type elemTy = parseType(JSONItemType.value(), rewriter.getContext());
      std::string elemTyStr;
      llvm::raw_string_ostream dstream(elemTyStr);
      dstream << elemTy;
      dstream.flush();
      int64_t dtype = krnl::mlirTypeToOnnxType(elemTy);
      equalOrFailed(module, rewriter, loc, create.llvm.constant(int64Ty, dtype),
          RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::GET_DATA_TYPE, {omTensorPtr}),
          "Wrong data type for the input " + std::to_string(i) + ": expect " +
              elemTyStr,
          false);

      // Verify data rank.
      auto JSONDimArray = JSONItem->getArray("dims");
      int64_t rank = JSONDimArray->size();
      equalOrFailed(module, rewriter, loc,
          create.llvm.constant(int64Ty, static_cast<int64_t>(rank)),
          RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::GET_DATA_RANK, {omTensorPtr}),
          "Wrong rank for the input " + std::to_string(i) + ": expect " +
              std::to_string(rank) + ", but got ");

      // Verify dimensions.
      Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::GET_DATA_SHAPE, {omTensorPtr});
      for (int d = 0; d < rank; ++d) {
        // Get actual dimension size.
        Value actualDim = create.llvm.load(
            int64Ty, create.llvm.getElemPtr(getPointerType(context, int64Ty),
                         int64Ty, sizesArrayPtr,
                         ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(d)}));
        // Get reference dimension size.
        auto JSONDimValue = (*JSONDimArray)[d].getAsInteger();
        assert(JSONDimValue && "failed to get value");
        int64_t dim = JSONDimValue.value();
        // Verify.
        if (ShapedType::isDynamic(dim) || dim == -1) {
          // In case that the reference dimension size is unknown, verify that
          // the actual dimension size is a non-negative value.
          create.llvm.ifThenElse(/*cond=*/
              [&](const LLVMBuilder &createLLVM) {
                Value zero =
                    createLLVM.constant(int64Ty, static_cast<int64_t>(d));
                return createLLVM.icmp(
                    LLVM::ICmpPredicate::slt, actualDim, zero);
              }, /*then=*/
              [&](const LLVMBuilder &createLLVM) {
                MultiDialectBuilder<LLVMBuilder, KrnlBuilder> create(
                    createLLVM);
                // Print an error message.
                std::string msg = "Wrong size for the dimension " +
                                  std::to_string(d) + " of the input " +
                                  std::to_string(i) +
                                  ": expect a non-negative value\n";
                StringRef errorMsg(msg);
                create.krnl.printf(errorMsg);
                // Set errno.
                krnl::emitErrNo(module, rewriter, loc, EINVAL);
                // Return NULL.
                create.llvm._return(
                    create.llvm.null(getI8PointerType(context)));
              });
        } else {
          Value referenceDim =
              create.llvm.constant(int64Ty, static_cast<int64_t>(dim));
          equalOrFailed(module, rewriter, loc, referenceDim, actualDim,
              "Wrong size for the dimension " + std::to_string(d) +
                  " of the input " + std::to_string(i) + ": expect " +
                  std::to_string(dim) + ", but got ");
        }
      }
    }
  }

  void recordEntryPointSignatures(ModuleOp &module,
      std::string currentEntryPointName, KrnlEntryPointOp entryOp,
      SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps) const {
    Operation *op = entryOp.getOperation();
    MLIRContext *context = module.getContext();
    Location loc = module.getLoc();
    OpBuilder b(context);
    MultiDialectBuilder<LLVMBuilder> create(b, loc);

    // Common information.
    Type i8Type = IntegerType::get(context, 8);

    // A helper function to emit a global constant operation storing a string.
    auto emitGlobalOp = [&context, &i8Type, &create](
                            std::string name, std::string value) {
      mlir::StringAttr valueAttr = mlir::StringAttr::get(context, value);
      Type valueArrayType = LLVM::LLVMArrayType::get(i8Type, value.size());
      LLVM::GlobalOp globalOp = create.llvm.globalOp(valueArrayType,
          /*isConstant=*/true, LLVM::Linkage::External, name, valueAttr);
      return globalOp;
    };

    bool zOS = isZOS(module);
    // NULL terminated entry point name.
    std::string terminatedEntryPointName = currentEntryPointName + '\0';
    terminatedEntryPointName = (zOS) ? krnl::a2e_s(terminatedEntryPointName)
                                     : terminatedEntryPointName;

    // Input/output signature strings.
    StringAttr sigAttr =
        op->getAttrOfType<StringAttr>(KrnlEntryPointOp::getSignatureAttrName());
    llvm::StringRef signature = sigAttr.getValue();
    auto splitSig = signature.split('@');
    std::string inSignature =
        (zOS) ? krnl::a2e_s(splitSig.first.str()) : splitSig.first.str();
    std::string outSignature =
        (zOS) ? krnl::a2e_s(splitSig.second.str()) : splitSig.second.str();

    // For each entry point name, emit three global constants to store the entry
    // point name and input/output signatures. For the i-th entry point, these
    // constants are named as follows:
    // - Entry point name: `_entry_point_i`.
    // - Input signature: `_entry_point_i_in_sig`.
    // - Output signature: `_entry_point_i_out_sig`.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    // Global constants for entry point names.
    std::string entryVarName =
        "_entry_point_" + std::to_string(KRNL_ENTRY_POINT_ID);
    KRNL_ENTRY_POINT_ID++;
    LLVM::GlobalOp entryGlobalOp =
        emitGlobalOp(entryVarName, terminatedEntryPointName);
    entryGlobalOps.emplace_back(entryGlobalOp);

    // Global constants for input signatures.
    std::string inSigVarName = entryVarName + "_in_sig";
    LLVM::GlobalOp inSigGlobalOp = emitGlobalOp(inSigVarName, inSignature);
    inSigGlobalOps.emplace_back(inSigGlobalOp);

    // Global constants for output signatures.
    std::string outSigVarName = entryVarName + "_out_sig";
    LLVM::GlobalOp outSigGlobalOp = emitGlobalOp(outSigVarName, outSignature);
    outSigGlobalOps.emplace_back(outSigGlobalOp);
  }

protected:
  LLVMTypeConverter &typeConverter;
};

void populateLoweringKrnlEntryPointOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors) {
  patterns.insert<KrnlEntryPointOpLowering>(typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
      verifyInputTensors);
}

} // namespace krnl
} // namespace onnx_mlir
