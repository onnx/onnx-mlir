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
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static FlatSymbolRefAttr getOrInsertExternFunc(StringRef funcName,
    ModuleOp module, mlir::LLVM::LLVMType funcType, PatternRewriter &rewriter) {
  auto *context = module.getContext();
  if (auto sym = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    assert(sym.getType() == funcType && "wrong symbol type");
    return SymbolRefAttr::get(funcName, context);
  }

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  return SymbolRefAttr::get(funcName, context);
}

static size_t getRankFromMemRefType(LLVM::LLVMType memRefTy) {
  // Usually a MemRef is a 5-element struct, where the 4th and 5th elements in
  // this struct are arrays whose size is the rank of the tensor. In the event
  // that the corresponding tensor of this MemRef is a scalar, the 4th and 5th
  // elements will have 0-length, which in turn causes the MemRef struct to
  // degenerate into a 3-element struct. For more information, refer to
  // https://github.com/llvm/llvm-project/blob/master/mlir/docs/ConversionToLLVMDialect.md#memref-types.
  auto numElems = memRefTy.getStructNumElements();
  assert((numElems == 3 || numElems == 5) &&
         "Expect MemRef type to contain either 3 or 5 elements.");

  if (numElems == 3)
    return 0; // MemRef refers to a scalar.
  else
    return memRefTy.getStructElementType(3).getArrayNumElements();
}

/// Return a symbol reference to the memcpy function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertMemcpy(PatternRewriter &rewriter,
    ModuleOp module, LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.memcpy.p0i8.p0i8.i64"))
    return SymbolRefAttr::get("llvm.memcpy.p0i8.p0i8.i64", context);
  // Create a function declaration for memcpy, the signature is:
  //   * `void (i8*, i8* , i64, i1)`
  auto llvmVoidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto llvmI1Ty = LLVM::LLVMType::getInt1Ty(llvmDialect);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmVoidTy,
      ArrayRef<mlir::LLVM::LLVMType>(
          {llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty, llvmI1Ty}),
      false);

  // Insert the memcpy function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "llvm.memcpy.p0i8.p0i8.i64", llvmFnType);
  return SymbolRefAttr::get("llvm.memcpy.p0i8.p0i8.i64", context);
}

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
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

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
    auto llvmMemRefType =
        typeConverter.convertType(type).cast<LLVM::LLVMType>();

    // The element type of the array.
    auto constantElementType =
        typeConverter.convertType(memRefTy.getElementType());
    auto globalType = constantElementType;
    for (int i = shape.size() - 1; i >= 0; i--)
      globalType = LLVM::LLVMType::getArrayTy(
          globalType.cast<LLVM::LLVMType>(), ArrayAttrIntVal(shape, i));
    // The llvm type of the global (example: [2 x [8 x float]])
    auto llvmGlobalType = globalType.cast<LLVM::LLVMType>();

    //    {
    //      OpBuilder::InsertionGuard insertGuard(rewriter);
    //      rewriter.setInsertionPointToStart(module.getBody());
    //
    //      assert(krnlGlobalOp.value().hasValue() &&
    //             "Krnl Global must always have a value");
    //      global = rewriter.create<LLVM::GlobalOp>(loc, llvmGlobalType,
    //          /*isConstant=*/true, LLVM::Linkage::Internal, name,
    //          krnlGlobalOp.value().getValue());
    //    }

    // Some frequently used types.
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);

    // Allocate the memory where the constants will be used from.
    // This is a region of local memory and needs to be emitted as an alloca.
    auto one = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Ty, rewriter.getI64IntegerAttr(1));
    //    auto alloc = rewriter.create<LLVM::AllocaOp>(
    //        loc, llvmGlobalType.getPointerTo(), one, /*alignment=*/0);
    //
    //    // Copy constant value into the local alloca:
    //    //  - Bitcast alloc to i8*
    //    Value int8PtrAlloc =
    //        rewriter.create<LLVM::BitcastOp>(loc, llvmI8PtrTy, alloc);
    //    auto getSegmentDataRef = getOrInsertExternFunc("getSegmentData",
    //    module,
    //        LLVM::LLVMType::getFunctionTy(llvmI8PtrTy, {},
    //        /*isVarArg=*/false), rewriter);
    //    Value alloc = rewriter
    //                      .create<CallOp>(loc, getSegmentDataRef, llvmI8PtrTy,
    //                          ArrayRef<Value>({}))
    //                      .getResult(0);

    auto base = module.lookupSymbol<LLVM::GlobalOp>("packedConst");
    assert(base);

    Value constPackBasePtrAddr = rewriter.create<LLVM::AddressOfOp>(loc, base);
    Value constPackBasePtr = rewriter.create<LLVM::LoadOp>(loc, base.getType(), constPackBasePtrAddr);
    auto offset = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            krnlGlobalOp.offsetAttr().getValue().getSExtValue()));
    Value alloc = rewriter.create<LLVM::GEPOp>(
        loc, llvmI8PtrTy, constPackBasePtr, ValueRange({offset}));

    //
    //    auto offset = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
    //    rewriter.getI64IntegerAttr(krnlGlobalOp.offsetAttr().getValue().getSExtValue()));
    //    Value alloc = rewriter.create<LLVM::GEPOp>(loc, llvmI8PtrTy, base,
    //    offset);

    //    //  - Bitcast global to i8*
    //    Value globalValue = rewriter.create<LLVM::AddressOfOp>(loc, global);
    //    Value i8PtrGlobal =
    //        rewriter.create<LLVM::BitcastOp>(loc, llvmI8PtrTy, globalValue);
    //    //  - Set size.
    //    Value memRefElementSize = rewriter.create<LLVM::ConstantOp>(loc,
    //    llvmI64Ty,
    //        rewriter.getI64IntegerAttr(getMemRefEltSizeInBytes(memRefTy)));
    //    Value numElementsValue = rewriter.create<LLVM::ConstantOp>(
    //        loc, llvmI64Ty, rewriter.getI64IntegerAttr(numElements));
    //    Value totalElementsSize =
    //        rewriter.create<LLVM::MulOp>(loc, memRefElementSize,
    //        numElementsValue);
    //    Value int64Size =
    //        rewriter.create<LLVM::SExtOp>(loc, llvmI64Ty, totalElementsSize);
    //    //  - Set volatile.
    //    Value isVolatile = rewriter.create<LLVM::ConstantOp>(loc,
    //        LLVM::LLVMType::getInt1Ty(llvmDialect),
    //        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    //    //  - Copy constant data into the alloca.
    //    auto memcpyRef = getOrInsertMemcpy(rewriter, module, llvmDialect);
    //    rewriter.create<CallOp>(loc, memcpyRef,
    //        LLVM::LLVMType::getVoidTy(llvmDialect),
    //        ArrayRef<Value>({int8PtrAlloc, i8PtrGlobal, int64Size,
    //        isVolatile}));

    // Prepare data to be inserted into MemRef.
    auto llvmConstantElementType = constantElementType.cast<LLVM::LLVMType>();
    Value typedAlloc = rewriter.create<LLVM::BitcastOp>(
        loc, llvmConstantElementType.getPointerTo(), alloc);

    // Create llvm MemRef from original MemRef and fill the data pointers.
    auto llvmMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, typeConverter, memRefTy, typedAlloc);

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
    KrnlMemcpyOpOperandAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule, llvmDialect);

    // First operand.
    Type dstType = operandAdaptor.dest()
                       .getType()
                       .cast<LLVM::LLVMType>()
                       .getStructElementType(1);
    Value alignedDstMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, dstType, operandAdaptor.dest(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrDstMemory = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), alignedDstMemory);

    // Second operand.
    Type srcType = operandAdaptor.src()
                       .getType()
                       .cast<LLVM::LLVMType>()
                       .getStructElementType(1);
    Value alignedSrcMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, srcType, operandAdaptor.src(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrSrcMemory = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), alignedSrcMemory);

    // Size.
    Value int64Size = rewriter.create<LLVM::SExtOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect), operandAdaptor.size());

    // Is volatile (set to false).
    Value isVolatile = rewriter.create<LLVM::ConstantOp>(loc,
        LLVM::LLVMType::getInt1Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));

    // Memcpy call
    rewriter.create<CallOp>(loc, memcpyRef,
        LLVM::LLVMType::getVoidTy(llvmDialect),
        ArrayRef<Value>({alignedInt8PtrDstMemory, alignedInt8PtrSrcMemory,
            int64Size, isVolatile}));

    rewriter.eraseOp(op);
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
    CREATE_ORDERED_DYN_MEM_REF_DICT,
    CREATE_DYN_MEM_REF,
    GET_DYN_MEM_REF,
    SET_DYN_MEM_REF,
    GET_DATA,
    SET_DATA,
    GET_SIZES,
    GET_STRIDES,
  };

  struct ApiSpec {
    API id;
    std::string name;
    FlatSymbolRefAttr symbolRef;
    LLVM::LLVMType outputTy;
    SmallVector<LLVM::LLVMType, 4> inputTys;

    ApiSpec(API id, const std::string &name, LLVM::LLVMType outputTy,
        ArrayRef<LLVM::LLVMType> inputTys)
        : id(id), name(name), outputTy(outputTy),
          inputTys(inputTys.begin(), inputTys.end()) {}

    LLVM::LLVMType funcTy() {
      return LLVM::LLVMType::getFunctionTy(outputTy, inputTys,
          /*isVarArg=*/false);
    }
  };

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {

    auto *llvmDialect =
        op.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");
    auto module = op.getParentOfType<ModuleOp>();
    auto apiRegistry = RegisterAllApis(module, rewriter, llvmDialect);
    auto loc = op.getLoc();
    auto numOutputs =
        op.getAttrOfType<IntegerAttr>(KrnlEntryPointOp::getNumOutputsAttrName())
            .getInt();

    using LLVMType = LLVM::LLVMType;
    auto opaquePtrTy = LLVMType::getInt8PtrTy(llvmDialect);
    auto int32Ty = LLVMType::getInt32Ty(llvmDialect);

    // Rewrite Krnl Entry Point Operation to an LLVM function with a dynamic
    // signature. The signature is dynamic because it remains the same no matter
    // what the model input/output schema look like. Such dynamic signature
    // takes a opaque ptr as input, representing a ptr to a data structure
    // containing a set of dynamic memrefs wrapped in a vector; similarly the
    // output is also a opaque ptr to a data structure with output memrefs
    // wrapped within it.
    auto staticEntryPointFuncName =
        op.getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference();
    auto dynEntryPointName = "_dyn_entry_point_" + staticEntryPointFuncName;
    assert(module.lookupSymbol(dynEntryPointName.str()) == nullptr &&
           "dynamic entry point name is not unique");
    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVMType::getFunctionTy(opaquePtrTy, {opaquePtrTy}, false);
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
                                  .dyn_cast<LLVMType>();

    // Retrieve dynamic mem refs from wrapped input, and convert every one of
    // them to static mem refs.
    SmallVector<Value, 4> staticInputs;
    auto wrappedInput = entryPointEntryBlock.getArgument(0);
    for (size_t i = 0; i < staticEntryPointTy.getFunctionNumParams(); i++) {
      // Call API function to retrieve the i-th dynamic memref.
      auto idxVal = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(i));
      auto dynMemRef = callApi(rewriter, loc, apiRegistry, API::GET_DYN_MEM_REF,
          {wrappedInput, idxVal});

      // Create a (static) memref type corresponding to the i-th memref input to
      // the inference function on stack, and load it to memRef.
      auto memRefPtrTy = staticEntryPointTy.getFunctionParamType(i);
      auto memRefTy = memRefPtrTy.getPointerElementTy();
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(1));
      Value ptrToMemRef = rewriter.create<LLVM::AllocaOp>(loc, memRefPtrTy, one,
          /*alignment=*/0);

      // Fill in the memref underlying ptrToMemRef with information extracted
      // from dynMemRef.
      fillPtrToMemRefWithDynMemRef(
          dynMemRef, ptrToMemRef, rewriter, loc, apiRegistry, llvmDialect);

      // ptrToMemRef will be an input to main computation graph function.
      staticInputs.emplace_back(ptrToMemRef);
    }

    // Call static entry point with the memref ptrs created, and get output.
    auto outMemRefs =
        rewriter
            .create<LLVM::CallOp>(loc,
                staticEntryPointTy.getFunctionResultType(),
                rewriter.getSymbolRefAttr(wrappedStaticEntryPointFuncName),
                staticInputs)
            .getResult(0);
    auto outMemRefsType = outMemRefs.getType().dyn_cast<LLVMType>();

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
        auto type = outMemRefsType.getStructElementType(i);
        auto extractOp = rewriter.create<LLVM::ExtractValueOp>(loc,
            /*res=*/type,
            /*type=*/outMemRefs,
            /*position=*/position);
        outMemRefList.emplace_back(extractOp.getResult());
      }
    }

    // Create wrapped output.
    auto wrappedOutput = callApi(
        rewriter, loc, apiRegistry, API::CREATE_ORDERED_DYN_MEM_REF_DICT, {});

    for (decltype(numOutputs) i = 0; i < outMemRefList.size(); i++) {
      // Get the i-th memref returned, convert to a dynamic memref and store it
      // in the wrappedOutput.
      auto memRef = outMemRefList.at(i);
      auto outMemRefTy = memRef.getType().dyn_cast<LLVMType>();
      auto outMemRefRank = getRankFromMemRefType(outMemRefTy);
      auto outMemRefRankVal = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(outMemRefRank));
      auto outDynMemRef = callApi(rewriter, loc, apiRegistry,
          API::CREATE_DYN_MEM_REF, {outMemRefRankVal});
      fillDynMemRefWithMemRef(
          memRef, outDynMemRef, rewriter, loc, apiRegistry, llvmDialect);
      auto idx = rewriter.create<LLVM::ConstantOp>(
          loc, int32Ty, rewriter.getI32IntegerAttr(i));
      callApi(rewriter, loc, apiRegistry, API::SET_DYN_MEM_REF,
          {wrappedOutput, idx, outDynMemRef});
    }
    // Return wrapped output.
    rewriter.create<LLVM::ReturnOp>(
        loc, SmallVector<Value, 1>({wrappedOutput}));
    return success();
  }

private:
  using ApiRegistry = std::map<API, ApiSpec>;

  ApiRegistry RegisterAllApis(ModuleOp &module, PatternRewriter &rewriter,
      LLVM::LLVMDialect *llvmDialect) const {
    using LLVMType = LLVM::LLVMType;
    auto voidTy = LLVMType::getVoidTy(llvmDialect);
    auto opaquePtrTy = LLVMType::getInt8PtrTy(llvmDialect);
    auto int32Ty = LLVMType::getInt32Ty(llvmDialect);
    auto int64Ty = LLVMType::getInt64Ty(llvmDialect);
    auto int64PtrTy = int64Ty.getPointerTo();

    // Declare API type as an enum value, its string name and an LLVM Type
    // specifying its signature.
    // clang-format off
    std::vector<ApiSpec> apiSpecs = {
        ApiSpec(API::CREATE_ORDERED_DYN_MEM_REF_DICT, "createOrderedDynMemRefDict", opaquePtrTy, {}),
        ApiSpec(API::CREATE_DYN_MEM_REF, "createDynMemRef", opaquePtrTy, {int32Ty}),
        ApiSpec(API::GET_DATA, "getData", opaquePtrTy, {opaquePtrTy}),
        ApiSpec(API::SET_DATA, "setData", voidTy, {opaquePtrTy, opaquePtrTy}),
        ApiSpec(API::GET_DYN_MEM_REF, "getDynMemRef", opaquePtrTy, {opaquePtrTy, int32Ty}),
        ApiSpec(API::SET_DYN_MEM_REF, "setDynMemRef", voidTy, {opaquePtrTy, int32Ty, opaquePtrTy}),
        ApiSpec(API::GET_SIZES, "getSizes", int64PtrTy, {opaquePtrTy}),
        ApiSpec(API::GET_STRIDES, "getStrides", int64PtrTy, {opaquePtrTy})
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
    auto returnVals =
        rewriter.create<LLVM::CallOp>(loc, registry.at(apiId).outputTy,
            registry.at(apiId).symbolRef, ArrayRef<Value>(params));
    if (returnVals.getNumResults() == 1)
      return returnVals.getResult(0);
    return nullptr;
  }

  // Helper function to insert an entry block to LLVM function.
  // (TODO): upstream this to MLIR.
  Block &createEntryBlock(LLVM::LLVMType &dynEntryPointFuncType,
      LLVM::LLVMFuncOp &dynamicEntryPointFunc) const {
    // Add entry block:
    auto *entryPointEntryBlock = new Block();
    dynamicEntryPointFunc.push_back(entryPointEntryBlock);
    llvm::SmallVector<Type, 4> argTypes;
    for (size_t i = 0; i < dynEntryPointFuncType.getFunctionNumParams(); i++)
      argTypes.emplace_back(dynEntryPointFuncType.getFunctionParamType(i));
    entryPointEntryBlock->addArguments(argTypes);
    return *entryPointEntryBlock;
  }

  void fillPtrToMemRefWithDynMemRef(Value &dynMemRef, Value &ptrToMemRef,
      PatternRewriter &rewriter, const Location &loc,
      const std::map<API, ApiSpec> &apiRegistry,
      LLVM::LLVMDialect *llvmDialect) const {
    auto memRefPtrTy = ptrToMemRef.getType().dyn_cast<LLVM::LLVMType>();
    auto memRefTy = memRefPtrTy.getPointerElementTy();
    auto int64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);

    Value memRef = rewriter.create<LLVM::LoadOp>(loc, memRefPtrTy, ptrToMemRef);

    // Set dataPtr and alignedDataPtr;
    auto dataPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_DATA, {dynMemRef});
    dataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, memRefTy.getStructElementType(0), dataPtr);
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
    auto rank = getRankFromMemRefType(memRefTy);
    auto sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_SIZES, {dynMemRef});
    auto stridesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_STRIDES, {dynMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Insert size of the dimension.
      auto dimSizePtr = rewriter.create<LLVM::GEPOp>(loc,
          int64Ty.getPointerTo(), sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimSize = rewriter.create<LLVM::LoadOp>(
          loc, int64Ty.getPointerTo(), dimSizePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimSize,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));

      // Insert stride of the dimension.
      auto dimStridePtr = rewriter.create<LLVM::GEPOp>(loc,
          int64Ty.getPointerTo(), sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimStride = rewriter.create<LLVM::LoadOp>(
          loc, int64Ty.getPointerTo(), dimStridePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimStride,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
    }

    rewriter.create<LLVM::StoreOp>(loc, memRef, ptrToMemRef);
  }

  void fillDynMemRefWithMemRef(Value &outMemRef, Value &outDynMemRef,
      PatternRewriter &rewriter, const Location &loc,
      const std::map<API, ApiSpec> &apiRegistry,
      LLVM::LLVMDialect *llvmDialect) const {
    auto outMemRefTy = outMemRef.getType().dyn_cast<LLVM::LLVMType>();
    auto int64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);

    // Extract the data pointer, and record it in dynamic mem ref created.
    Value outMemRefDataPtr = rewriter.create<LLVM::ExtractValueOp>(loc,
        outMemRefTy.getStructElementType(0), outMemRef,
        rewriter.getArrayAttr({rewriter.getI64IntegerAttr(0)}));
    outMemRefDataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), outMemRefDataPtr);
    callApi(rewriter, loc, apiRegistry, API::SET_DATA,
        {outDynMemRef, outMemRefDataPtr});

    auto rank = getRankFromMemRefType(outMemRefTy);
    auto sizesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_SIZES, {outDynMemRef});
    auto stridesArrayPtr =
        callApi(rewriter, loc, apiRegistry, API::GET_STRIDES, {outDynMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
          loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Transfer size of dimension from memref to dynamic memref.
      auto dimSize = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
          outMemRef,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));
      auto dimSizePtr = rewriter.create<LLVM::GEPOp>(loc,
          int64Ty.getPointerTo(), sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimSize, dimSizePtr);

      // Transfer stride of dimension from memref to dynamic memref.
      auto dimStride = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
          outMemRef,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
      auto dimStridePtr = rewriter.create<LLVM::GEPOp>(loc,
          int64Ty.getPointerTo(), stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimStride, dimStridePtr);
    }
  }
};

//===----------------------------------------------------------------------===//
// KRNL to LLVM: KrnlPackedConstOpLowering
//===----------------------------------------------------------------------===//

class KrnlPackedConstOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlPackedConstOpLowering(
      MLIRContext *context, LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(
            KrnlPackedConstantOp::getOperationName(), context, lowering_) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    auto packedConstOp = llvm::dyn_cast<KrnlPackedConstantOp>(op);
    LLVM::GlobalOp globalBase;
    // Some frequently used types.
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      globalBase = rewriter.create<LLVM::GlobalOp>(loc, llvmI8PtrTy,
          /*isConstant=*/false, LLVM::Linkage::Internal, "packedConst",
          nullptr);
    }

    auto mainFunc = module.lookupSymbol<FuncOp>("main_graph");
    assert(mainFunc);

    rewriter.setInsertionPoint(
        &mainFunc.getBody().front(), mainFunc.getBody().front().begin());

    //  - Initialize the global constant base.
    Value basePtrAddr = rewriter.create<LLVM::AddressOfOp>(loc, globalBase);
    auto getSegmentDataRef = getOrInsertExternFunc("getSegmentData", module,
        LLVM::LLVMType::getFunctionTy(
            llvmI8PtrTy, {llvmI64Ty}, /*isVarArg=*/false),
        rewriter);
    auto constPackSize = rewriter.create<LLVM::ConstantOp>(loc,
        LLVM::LLVMType::getInt64Ty(llvmDialect),
        packedConstOp.sizeInBytesAttr());
    Value alloc = rewriter
                      .create<CallOp>(loc, getSegmentDataRef, llvmI8PtrTy,
                          ArrayRef<Value>({constPackSize}))
                      .getResult(0);
    rewriter.create<LLVM::StoreOp>(loc, alloc, basePtrAddr);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      const auto &fileNameAttr = packedConstOp.file_nameAttr();
      auto type =
          LLVM::LLVMType::getArrayTy(LLVM::LLVMType::getInt8Ty(llvmDialect),
              fileNameAttr.getValue().size());
      rewriter.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
          LLVM::Linkage::Internal, "constPackFileName", fileNameAttr);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return (a.getValue()[i]).cast<IntegerAttr>().getInt();
  }
};
} // end namespace

//===----------------------------------------------------------------------===//
// KRNL + Stadard + Affine dialects lowering to LLVM.
//===----------------------------------------------------------------------===//

namespace {
struct KrnlToLLVMLoweringPass
    : public PassWrapper<KrnlToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
} // end anonymous namespace

void KrnlToLLVMLoweringPass::runOnOperation() {
  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // Lower the MemRef types to a representation in LLVM.
  LLVMTypeConverter typeConverter(&getContext());

  // We have a combination of `krnl`, `affine`, and `std` operations. We
  // lower in stages until all the code is in the LLVM dialect.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns,
      /*emitCWrapperS=*/true,
      /*useAlignedAlloc=*/false);
  patterns.insert<KrnlGlobalOpLowering, KrnlPackedConstOpLowering>(
      &getContext(), typeConverter);

  // Lower from the `krnl` dialect i.e. the Reshape operation.
  patterns.insert<KrnlMemcpyOpLowering, KrnlEntryPointOpLowering>(
      &getContext());

  ConstantOp::getCanonicalizationPatterns(patterns, &getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(
          getOperation(), target, patterns, &typeConverter))) {
    signalPassFailure();
  }
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<mlir::Pass> mlir::createKrnlLowerToLLVMPass() {
  return std::make_unique<KrnlToLLVMLoweringPass>();
}

static PassRegistration<KrnlToLLVMLoweringPass> pass(
    "lower-all-llvm", "Lower the Krnl Affine and Std dialects to LLVM.");
