/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.cpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of Krnl operations to a combination of
// other dialects (affine, std, LLVM).
//
//===----------------------------------------------------------------------===//

#include <fstream>

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"

#include "onnx/onnx_pb.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

using namespace mlir;

#define DEBUG_TYPE "krnl_to_llvm"

namespace onnx_mlir {
namespace krnl {

bool LLVM_USE_OPAQUE_POINTER = true;
std::string EXTERNAL_CONSTANT_PREFIX = "om_external_constant_";

uint64_t KRNL_ENTRY_POINT_ID = 0;

// Return true if the value owns the storge. A value defined by memref.alloc
// owns the storage. A value defined by constant, krnl.Global, does not own
// the storage (in the sense that the storage can not be freed)
// The result determines whether the returned tensor owns the storage
// It is assumed that bufferization dealloc pass already added bufferization
// clone when necessary.
// Currently, the ViewLikeOp and arith.select are traced back. Any other
// cases? A general solution is suggested in issue#2033
// If this function returns a false positive, seg fault may occur when the
// storage is freed.
// If this function returns false negative, memory leak may occur.
static bool shouldOwn(Value v) {
  bool result = true;
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp)
    // Block argument, do not own this since it is an input that can be owned
    // by an input OMTensor.
    result = false;
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
      // Block argument, do not own this since it is an input that can be
      // owned by an input OMTensor.
      result = false;
    else if (llvm::dyn_cast<KrnlGlobalOp>(definingOp))
      // Do not own a constant that is defined by KrnlGlobalOp.
      result = false;
    else if (auto selectOp = llvm::dyn_cast<arith::SelectOp>(definingOp)) {
      // Temporary fix: the value come from select. Should further track
      // the false and true inputs of arith.select. But leave it to PR
      // which will focus on this problem.
      result = shouldOwn(selectOp.getTrueValue()) &&
               shouldOwn(selectOp.getFalseValue());
    }
  }
  return result;
}

void determineOwnershipForOutputOMTensors(
    ModuleOp &module, SmallVectorImpl<bool> &outputOMTensorOwnerships) {
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
  module->walk([&](func::FuncOp op) -> WalkResult {
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
    if (llvm::dyn_cast<func::ReturnOp>(op)) {
      returnOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Check, for each output, if it was transitively produced by a constant or
  // a block argument.
  for (Value v : returnOp->getOperands()) {
    bool shouldOwnResult = shouldOwn(v);
    outputOMTensorOwnerships.emplace_back(shouldOwnResult);
    LLVM_DEBUG(llvm::dbgs()
               << "Should the OMTensor own the entry function output? "
               << shouldOwnResult << "\n");
  }
}

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
    std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes,
    bool verifyInputTensors) {
  // TODO: look at what is done in
  // mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp in function
  // LowerVectorToLLVMPass::runOnOperation() and see what we should do about it.
  // They run it in two steps, and add additional lowerings.

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorTransformsOptions());
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransformsOptions());

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  memref::populateExpandOpsPatterns(patterns);
  // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
  // performance.
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, inputMemRefTypes, outputMemRefTypes, verifyInputTensors);
}

bool hasSingleEntryPoint(ModuleOp &module) {
  uint64_t i = 0;
  module->walk([&](KrnlEntryPointOp entryOp) -> WalkResult {
    if (++i >= 2)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return (i == 1);
}

/// Keep original MemRefTypes for inputs and outputs. These information will be
/// used for constructing OMTensors for inputs and outputs. We have to record
/// this information at this point before they are disappeared during the
/// lowering to LLVM. For example, unsigned types do not exist at LLVM level,
/// typed pointers becomes opaque if opaque point is enabled.
void recordInputOutputMemRefTypes(ModuleOp &module,
    std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
    std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes) {
  module->walk([&](KrnlEntryPointOp entryOp) -> WalkResult {
    StringRef entryPointFuncName =
        entryOp.getOperation()
            ->getAttrOfType<SymbolRefAttr>(
                KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference()
            .getValue();
    auto *entryPointFunc = module.lookupSymbol(entryPointFuncName);
    assert(entryPointFunc && isa<func::FuncOp>(entryPointFunc) &&
           "entry point func must exist and be an llvm func op");
    auto entryPointTy = dyn_cast<func::FuncOp>(entryPointFunc)
                            .getFunctionType()
                            .dyn_cast<FunctionType>();
    SmallVector<MemRefType, 4> inputTypes, outputTypes;
    for (Type ty : entryPointTy.getInputs())
      inputTypes.emplace_back(dyn_cast<MemRefType>(ty));
    for (Type ty : entryPointTy.getResults())
      outputTypes.emplace_back(dyn_cast<MemRefType>(ty));
    inputMemRefTypes.emplace(
        std::make_pair(entryPointFuncName.str(), inputTypes));
    outputMemRefTypes.emplace(
        std::make_pair(entryPointFuncName.str(), outputTypes));
    return WalkResult::advance();
  });
  return;
}

/// This function emits three functions: omQueryEntryPoints, omInputSignature
/// and omOutputSignature.
/// - omQueryEntryPoints has type of `**i8 (*i64)` to query an array of entry
/// point names.
/// - omInputSignature and omOutputSignature have type of type `*i8 (*i8)` to
/// return input and output signatures of the given entry point.
void genSignatureFunction(ModuleOp &module,
    const SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    const SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    const SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps) {
  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  OpBuilder b(context);
  MultiDialectBuilder<LLVMBuilder> create(b, loc);

  // Common information.
  Type i8Type = IntegerType::get(context, 8);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type i64PtrTy = getPointerType(context, i64Type);
  Type i8PtrTy = getPointerType(context, i8Type);
  Type i8PtrPtrTy = getPointerType(context, i8PtrTy);

  uint64_t numOfEntryPoints = entryGlobalOps.size();

  // Emit a global constant to store an array of pointers pointing to each entry
  // point constants. The array ends with NULL.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  auto arrayType = LLVM::LLVMArrayType::get(i8PtrTy, entryGlobalOps.size() + 1);
  LLVM::GlobalOp entryArrayOp = create.llvm.globalOp(arrayType,
      /*isConstant=*/true, LLVM::Linkage::Internal, "_entry_point_arrays",
      Attribute());
  { // Fill the initializer with pointers to entry point constants.
    Region &region = entryArrayOp.getInitializerRegion();
    Block *block = b.createBlock(&region);

    // Initialize an array with the addresses of the global strings.
    b.setInsertionPointToStart(block);
    Value array = b.create<LLVM::UndefOp>(loc, arrayType);

    uint32_t index = 0;
    Value lastValue = array;
    for (const LLVM::GlobalOp &globalOp : entryGlobalOps) {
      Value strAddr = krnl::getPtrToGlobalString(globalOp, loc, b);
      lastValue =
          create.llvm.insertValue(arrayType, lastValue, strAddr, {index++});
    }

    // The last element of the array is NULL.
    Value nullPtr = create.llvm.null(getI8PointerType(context));
    lastValue =
        create.llvm.insertValue(arrayType, lastValue, nullPtr, {index++});
    create.llvm._return(lastValue);
  }

  // Emit a function, omQueryEntryPoints, of type `**i8 (*i64)` to query an
  // array of entry point names.
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // Emit the function type.
    Type llvmFnType =
        LLVM::LLVMFunctionType::get(i8PtrPtrTy, {i64PtrTy}, false);
    LLVM::LLVMFuncOp funcOp =
        create.llvm.func("omQueryEntryPoints", llvmFnType);
    // Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);
    Value numOfEntryPoints = entryBlock->getArgument(0);
    // If the argument is not NULL, update its value to return the number of
    // entry points.
    create.llvm.ifThenElse(/*cond=*/
        [&](LLVMBuilder &createLLVM) {
          Value nullPtr = createLLVM.null(i64PtrTy);
          return createLLVM.icmp(
              LLVM::ICmpPredicate::ne, numOfEntryPoints, nullPtr);
        }, /*then=*/
        [&](LLVMBuilder &createLLVM) {
          Value numOfEntryPointsPtr = createLLVM.getElemPtr(
              i64PtrTy, i64Type, numOfEntryPoints, ArrayRef<LLVM::GEPArg>{0});
          Value noep =
              createLLVM.constant(i64Type, (int64_t)entryGlobalOps.size());
          createLLVM.store(noep, numOfEntryPointsPtr);
        });
    // Emit code to return the entry point array.
    Value entryAddr = create.llvm.addressOf(entryArrayOp);
    Value entryI8Ptr = create.llvm.bitcast(i8PtrPtrTy, entryAddr);
    create.llvm._return(entryI8Ptr);
  }

  // Emit two signature functions, omInputSignature and omOutputSignature, of
  // type `*i8 (*i8)` at the end of the module.
  SmallVector<std::string, 2> funcNames = {
      "omInputSignature", "omOutputSignature"};
  for (uint64_t i = 0; i < funcNames.size(); ++i) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // 1. Emit the function type.
    Type llvmFnType = LLVM::LLVMFunctionType::get(i8PtrTy, {i8PtrTy}, false);
    LLVM::LLVMFuncOp funcOp = create.llvm.func(funcNames[i], llvmFnType);

    // 2. Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);

    Value zeroI32 = create.llvm.constant(i32Type, (int64_t)0);

    // 2.1 The name of the entry point that we want to return its signature.
    Value input = entryBlock->getArgument(0);

    // 2.2 Emit code to find the signature of the given entry point.
    // Iterate over the list of the entry points and check string equality.
    for (uint64_t j = 0; j < numOfEntryPoints; ++j) {
      LLVM::GlobalOp globalEntryPoint = entryGlobalOps[j];
      LLVM::GlobalOp globalSignature =
          (i == 0) ? inSigGlobalOps[j] : outSigGlobalOps[j];
      assert(globalEntryPoint.getValueAttr().isa<StringAttr>() &&
             "Entry point value is not StringAttr");
      StringAttr entryPointValueAttr =
          globalEntryPoint.getValueAttr().cast<StringAttr>();

      // Return the signature if found.
      create.llvm.ifThenElse(/*cond=*/
          [&](LLVMBuilder &createLLVM) {
            // Read an entry point name.
            Value entryI8Ptr =
                krnl::getPtrToGlobalString(globalEntryPoint, loc, b);
            // Compare it with the user's entry point name.
            FlatSymbolRefAttr StrncmpRef = krnl::getOrInsertStrncmp(b, module);
            Value length = createLLVM.constant(
                i64Type, (int64_t)entryPointValueAttr.getValue().size());
            Value strncmpResult = createLLVM.call(i32Type, StrncmpRef,
                ArrayRef<Value>({input, entryI8Ptr, length}));
            // Found if strncmp returns `0`.
            return createLLVM.icmp(
                LLVM::ICmpPredicate::eq, strncmpResult, zeroI32);
          }, /*then=*/
          [&](LLVMBuilder &createLLVM) {
            Value sigAddr = createLLVM.addressOf(globalSignature);
            Value sigI8Ptr = createLLVM.bitcast(i8PtrTy, sigAddr);
            createLLVM._return(sigI8Ptr);
          });
    }

    // Return NULL if not found.
    create.llvm._return(create.llvm.null(getI8PointerType(context)));
  }
}

/// Extract then pack constant arrays and store to a file.
/// Return true if there are constants that are OK to store on files.
/// A single constant's size must be greater than singleThreshold.
/// The total size of contants must be greater than totalThreshold.
bool extractConstantsToFile(ModuleOp &module, std::string filepath,
    uint64_t singleThreshold, uint64_t totalThreshold) {
  Location loc = module.getLoc();
  MLIRContext *context = module.getContext();
  OpBuilder b(module.getContext());
  MultiDialectBuilder<LLVMBuilder> create(b, loc);

  SmallVector<KrnlGlobalOp> globalOfInterest;
  // Check constants with thresholds.
  // Do not count constants whose size is <= singleThreshold.
  uint64_t totalSize = 0;
  module.walk([&](KrnlGlobalOp op) {
    // Ignore constants that are return values.
    bool isReturnedValue = false;
    for (Operation *user : op.getResult().getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        isReturnedValue = true;
        break;
      }
    }
    if (isReturnedValue)
      return WalkResult::advance();

    assert(op.getValue().has_value() && "Krnl Global must always have a value");
    auto value = op.getValue().value();
    ArrayRef<char> rawData;
    // Only handle DenseElementsAttr and DenseResourceElementsAttr.
    TypeSwitch<Attribute>(value)
        .Case<DenseResourceElementsAttr>([&](DenseResourceElementsAttr attr) {
          auto blob =
              value.cast<DenseResourceElementsAttr>().getRawHandle().getBlob();
          assert(blob && "Expecting dense resource with a valid blob");
          rawData = blob->getData();
          if (attr.isSplat() || rawData.size() <= singleThreshold)
            return;
          globalOfInterest.emplace_back(op);
        })
        .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
          DenseElementsAttr denseAttr =
              value.dyn_cast_or_null<DenseElementsAttr>();
          rawData = denseAttr.getRawData();
          if (attr.isSplat() || rawData.size() <= singleThreshold)
            return;
          globalOfInterest.emplace_back(op);
        })
        .Default([&](Attribute attr) { return; });
    totalSize += rawData.size();
    return WalkResult::advance();
  });
  // Do not use file if the total size of satisfied constants is <=
  // totalThreshold.
  if (totalSize <= totalThreshold)
    return false;

  // Pack all constants into a single buffer in order to save to file.
  std::vector<char> packedConst;
  for (KrnlGlobalOp op : globalOfInterest) {
    assert(op.getValue().has_value() && "Krnl Global must always have a value");
    auto value = op.getValue().value();
    TypeSwitch<Attribute>(value)
        .Case<DenseResourceElementsAttr>([&](DenseResourceElementsAttr attr) {
          auto blob =
              value.cast<DenseResourceElementsAttr>().getRawHandle().getBlob();
          assert(blob && "Expecting dense resource with a valid blob");
          ArrayRef<char> rawData = blob->getData();
          op.setOffsetAttr(b.getI64IntegerAttr(packedConst.size()));
          op.removeValueAttr();
          packedConst.insert(packedConst.end(), rawData.begin(), rawData.end());
        })
        .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
          DenseElementsAttr denseAttr =
              value.dyn_cast_or_null<DenseElementsAttr>();
          ArrayRef<char> rawData = denseAttr.getRawData();
          op.setOffsetAttr(b.getI64IntegerAttr(packedConst.size()));
          op.removeValueAttr();
          packedConst.insert(packedConst.end(), rawData.begin(), rawData.end());
        })
        .Default([&](Attribute attr) { return; });
  }

  // No constant statisfying thresholds, do not store constants to file.
  if (packedConst.empty())
    return false;

  // Save to file.
  std::ofstream outfile(filepath, std::ofstream::binary);
  outfile.write(packedConst.data(), packedConst.size());

  // Create a global op to store the filename in the IR.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  Type llvmI8Ty = IntegerType::get(context, 8);
  std::string fname = llvm::sys::path::filename(filepath).str() + '\0';
  mlir::StringAttr valueAttr = mlir::StringAttr::get(context, fname);
  Type valueArrayType = LLVM::LLVMArrayType::get(llvmI8Ty, fname.size());
  create.llvm.globalOp(valueArrayType,
      /*isConstant=*/true, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "filename", valueAttr);
  // Create a global to store isLE.
  bool isLE = llvm::support::endian::system_endianness() ==
              llvm::support::endianness::little;
  create.llvm.globalOp(llvmI8Ty,
      /*isConstant=*/true, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "isLE", b.getI8IntegerAttr(isLE));

  return true;
}

/// Emit a function "omLoadConstantsFromFile" in the IR to load constants from
/// external files into global operations. Aligned buffers are allocated. Make
/// sure to free these buffers at the end of the program by calling
/// "freeBuffersForConstants".
/// By default, user program (C/C++/Java/Python) would call
/// "omLoadConstantsFromFile".
/// If calledByEntryPoint, this function will be called by entry points.
void loadConstantsFromFile(ModuleOp &module,
    const RuntimeAPIRegistry &apiRegistry,
    const SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &dataGlobalOps,
    bool calledByEntryPoint = false) {
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  OpBuilder b(ctx);
  MultiDialectBuilder<LLVMBuilder> create(b, loc);

  Type llvmI8Ty = IntegerType::get(ctx, 8);
  Type llvmI64Ty = IntegerType::get(ctx, 64);
  Type llvmI8PtrTy = getPointerType(ctx, llvmI8Ty);
  Type llvmVoidTy = LLVM::LLVMVoidType::get(ctx);

  // The following function will be emitted inside the IR to load constants from
  // file.
  std::string loadAllConstantsFuncName = "omLoadConstantsFromFile";
  Type llvmFnType =
      LLVM::LLVMFunctionType::get(llvmVoidTy, {llvmI8PtrTy}, false);

  // By default, user program (C/C++/Java/Python) would call this function.
  // If calledByEntryPoint, this function will be called by entry points.
  LLVM::LLVMFuncOp funcOp;
  if (calledByEntryPoint) {
    Operation *firstEntryPointOp =
        getFirstEntryOpInBlock(module, entryGlobalOps);
    assert(firstEntryPointOp && "No entry function exists");
    b.setInsertionPoint(firstEntryPointOp);
    funcOp = create.llvm.func(loadAllConstantsFuncName, llvmFnType);
    // Call loadAllConstantsFuncName in each entry point function.
    for (auto entryGlobalOp : entryGlobalOps) {
      std::string entryName =
          entryGlobalOp.getValue().value().cast<StringAttr>().getValue().str();
      // Erase the null symbol.
      entryName.erase(
          std::find(entryName.begin(), entryName.end(), '\0'), entryName.end());
      auto entryFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(entryName);
      assert(entryFunc && "Entry function not found");
      b.setInsertionPoint(
          &entryFunc.getBody().front(), entryFunc.getBody().front().begin());
      FlatSymbolRefAttr loadAllConstantsRef = create.llvm.getOrInsertSymbolRef(
          module, loadAllConstantsFuncName, llvmVoidTy, {},
          /*isVarArg=*/false);
      create.llvm.call({}, loadAllConstantsRef, {});
    }
  } else {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    funcOp = create.llvm.func(loadAllConstantsFuncName, llvmFnType);
  }

  // Emit the body of the function.
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entryBlock);

  // Open the constant file in binary mode.
  Value fnameI8Ptr = entryBlock->getArgument(0);
  // Get the global op for isLE.
  std::string isleSymbol = EXTERNAL_CONSTANT_PREFIX + "isLE";
  auto isleGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(isleSymbol);
  assert(isleGlobalOp && "Could not find the global op for data isle");
  int64_t isle = isleGlobalOp.getValue()
                     .value()
                     .cast<IntegerAttr>()
                     .getValue()
                     .getSExtValue();
  Value isleVal = create.llvm.constant(llvmI64Ty, isle);
  Value filePtr = RuntimeAPI::callApi(b, loc, apiRegistry,
      RuntimeAPI::API::OPEN_BINARY_FILE, {fnameI8Ptr, isleVal});

  // Read data from file to global constants.
  module->walk([&](LLVM::GlobalOp dataGlobalOp) -> WalkResult {
    // Get the global op for data.
    StringRef dataSymbol = dataGlobalOp.getSymName();
    std::string prefixData = EXTERNAL_CONSTANT_PREFIX + "data";
    if (!dataSymbol.startswith(prefixData))
      return WalkResult::advance();
    std::string constantName = dataSymbol.drop_front(prefixData.size()).str();

    // Get the global op for data size.
    std::string sizeSymbol = EXTERNAL_CONSTANT_PREFIX + "size" + constantName;
    auto sizeGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(sizeSymbol);
    assert(sizeGlobalOp && "Could not find the global op for data size");
    int64_t dataSize = sizeGlobalOp.getValue()
                           .value()
                           .cast<IntegerAttr>()
                           .getValue()
                           .getSExtValue();

    // Get offset.
    std::string offsetSymbol =
        EXTERNAL_CONSTANT_PREFIX + "offset" + constantName;
    auto offsetGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(offsetSymbol);
    assert(offsetGlobalOp && "Could not find the global op for offset");
    int64_t offset = offsetGlobalOp.getValue()
                         .value()
                         .cast<IntegerAttr>()
                         .getValue()
                         .getSExtValue();

    // Get alignment.
    int64_t alignment = -1;
    if (dataGlobalOp.getAlignment().has_value())
      alignment = dataGlobalOp.getAlignment().value();

    // Read data from the external file.
    Value dataGlobalAddr = create.llvm.addressOf(dataGlobalOp);
    Value dataPtr = create.llvm.bitcast(llvmI8PtrTy, dataGlobalAddr);
    Value offsetVal = create.llvm.constant(llvmI64Ty, offset);
    Value sizeVal = create.llvm.constant(llvmI64Ty, dataSize);
    Value alignmentVal = create.llvm.constant(llvmI64Ty, alignment);
    RuntimeAPI::callApi(b, loc, apiRegistry,
        RuntimeAPI::API::LOAD_EXTERNAL_CONSTANT,
        {dataPtr, filePtr, offsetVal, sizeVal, alignmentVal});
    // Keep trace of global addresses in order to free their buffers.
    dataGlobalOps.emplace_back(dataGlobalOp);

    return WalkResult::advance();
  });

  // Close the file.
  RuntimeAPI::callApi(
      b, loc, apiRegistry, RuntimeAPI::API::CLOSE_FILE, {filePtr});

  create.llvm._return();
}

/// Emit a function "omFreeBuffersForConstants" in the IR to free aligned
/// buffers allocated for constants from files.
/// By default, user program (C/C++/Java/Python) would call this function.
/// If calledByEntryPoint, this function will be called by entry points.
void freeBuffersForConstants(ModuleOp &module,
    const RuntimeAPIRegistry &apiRegistry,
    const SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &dataGlobalOps,
    bool calledByEntryPoint = false) {
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  OpBuilder b(ctx);
  MultiDialectBuilder<LLVMBuilder> create(b, loc);

  Type llvmI8Ty = IntegerType::get(ctx, 8);
  Type llvmI8PtrTy = getPointerType(ctx, llvmI8Ty);
  Type llvmI64Ty = IntegerType::get(ctx, 64);
  Type llvmVoidTy = LLVM::LLVMVoidType::get(ctx);

  // The following function will be emitted inside the IR to free buffers for
  // external parameters.
  std::string freeBuffersForConstantsFuncName = "omFreeBuffersForConstants";
  Type llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, {}, false);

  // By default, user program (C/C++/Java/Python) would call this function.
  // If calledByEntryPoint, this function will be called by entry points.
  LLVM::LLVMFuncOp funcOp;
  if (calledByEntryPoint) {
    Operation *firstEntryPointOp =
        getFirstEntryOpInBlock(module, entryGlobalOps);
    assert(firstEntryPointOp && "No entry function exists");
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(firstEntryPointOp);
    funcOp = create.llvm.func(freeBuffersForConstantsFuncName, llvmFnType);
    // Call omFreeBuffersForConstants at the end of each entry point function.
    for (auto entryGlobalOp : entryGlobalOps) {
      std::string entryName =
          entryGlobalOp.getValue().value().cast<StringAttr>().getValue().str();
      // Erase the null symbol.
      entryName.erase(
          std::find(entryName.begin(), entryName.end(), '\0'), entryName.end());
      auto entryFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(entryName);
      assert(entryFunc && "Entry function not found");

      Operation *terminator = entryFunc.getRegion().back().getTerminator();
      b.setInsertionPoint(terminator);
      FlatSymbolRefAttr freeAllConstantsRef = create.llvm.getOrInsertSymbolRef(
          module, freeBuffersForConstantsFuncName, llvmVoidTy, {},
          /*isVarArg=*/false);
      create.llvm.call({}, freeAllConstantsRef, {});
    }
  } else {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    funcOp = create.llvm.func(freeBuffersForConstantsFuncName, llvmFnType);
  }

  // Emit the body of the function omFreeBuffersForConstants.
  Block *entryBlock = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard bodyGuard(b);
  b.setInsertionPointToStart(entryBlock);
  for (LLVM::GlobalOp global : dataGlobalOps) {
    Value addr = create.llvm.addressOf(global);
    Value ptr = create.llvm.load(llvmI8PtrTy, addr);
    // Get alignment.
    int64_t alignment = -1;
    if (global.getAlignment().has_value())
      alignment = global.getAlignment().value();
    Value alignVal = create.llvm.constant(llvmI64Ty, alignment);
    RuntimeAPI::callApi(
        b, loc, apiRegistry, RuntimeAPI::API::FREE_ALIGNED, {ptr, alignVal});
  }
  create.llvm._return();
}

//===----------------------------------------------------------------------===//
// Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlToLLVMPass)

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ConvertKrnlToLLVMPass() = default;
  ConvertKrnlToLLVMPass(const ConvertKrnlToLLVMPass &pass)
      : PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>>() {}
  ConvertKrnlToLLVMPass(bool verifyInputTensors, bool useOpaquePointers,
      bool useLRODATA, bool storeConstantsToFile,
      uint64_t constantsToFileSingleThreshold,
      uint64_t constantsToFileTotalThreshold, std::string outputNameNoExt) {
    this->verifyInputTensors = verifyInputTensors;
    this->useOpaquePointers = useOpaquePointers;
    // Exclusive options. no option or only one option can be True.
    this->useLRODATA = useLRODATA;
    this->storeConstantsToFile = storeConstantsToFile;
    this->constantsToFileSingleThreshold = constantsToFileSingleThreshold;
    this->constantsToFileTotalThreshold = constantsToFileTotalThreshold;
    this->outputNameNoExt = outputNameNoExt;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  Option<bool> useOpaquePointers{*this, "use-opaque-pointers",
      llvm::cl::desc("Whether to use opaque pointers instead of typed pointers "
                     "when lowering to LLVM. Default: true"),
      llvm::cl::init(true)};

  Option<bool> verifyInputTensors{*this, "verify-input-tensors",
      llvm::cl::desc(
          "Verify input tensors whenever the entry point function is called.\n"
          "Data type and shape are verified. Enable this may introduce "
          "overhead in inferencing."),
      llvm::cl::init(false)};

  Option<bool> useLRODATA{*this, "use-lrodata-section",
      llvm::cl::desc("Put global constants into the large read-only data "
                     "section. This is for linking large object files"),
      llvm::cl::init(false)};

  Option<bool> storeConstantsToFile{*this, "store-constants-to-file",
      llvm::cl::desc("Put global constants to a file."), llvm::cl::init(false)};

  Option<float> constantsToFileTotalThreshold{*this,
      "constants-to-file-total-threshold",
      llvm::cl::desc(
          "Put global constants to a file if the total size in "
          "bytes of constants is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Only count contants whose size is greater than "
          "constants-to-file-single-threshold. Value is in GB."),
      llvm::cl::init(2.0)};

  Option<float> constantsToFileSingleThreshold{*this,
      "constants-to-file-single-threshold",
      llvm::cl::desc(
          "Put global constants to a file if a single constant's size in "
          "bytes is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Total sizes in bytes of satisfied constants must be greater than "
          "constants-to-file-total-threshold. Value is in KB."),
      llvm::cl::init(1.0)};

private:
  std::string outputNameNoExt = "./model";
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));

  // MLIR/LLVM is moving to using opaque pointers instead of typed pointers.
  // Remove this once MLIR/LLVM completely uses opaque pointers.
  options.useOpaquePointers = useOpaquePointers; // for LLVMTypeConverter.
  LLVM_USE_OPAQUE_POINTER = useOpaquePointers; // for onnx-mlir util functions.

  KRNL_ENTRY_POINT_ID = 0;

  // Global Op for entry point names and their input/output JSON signatures,
  // those will generated when lowering KrnlEntryPoint.
  // This info is used to generate global signature functions.
  SmallVector<LLVM::GlobalOp, 1> entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps;

  // Keep original MemRefTypes for inputs and outputs. These information will be
  // used for constructing OMTensors for inputs and outputs.
  // We have to record this information at this point before they are
  // disappeared during the lowering to LLVM. For example, unsigned types do
  // not exist at LLVM level, typed pointers becomes opaque if opaque point is
  // enabled.
  std::map<std::string, SmallVector<MemRefType, 4>> inputMemRefTypes;
  std::map<std::string, SmallVector<MemRefType, 4>> outputMemRefTypes;
  recordInputOutputMemRefTypes(module, inputMemRefTypes, outputMemRefTypes);

  // Determine whether the module has a single entry point or not.
  bool singleEntryPoint = hasSingleEntryPoint(module);

  // Determine whether an output OMTensor should own the underlying buffer or
  // not.
  SmallVector<bool, 4> outputOMTensorOwnerships;
  determineOwnershipForOutputOMTensors(module, outputOMTensorOwnerships);

  // If storeConstantsToFile, copy constants from GlobalOp and write to a single
  // file.
  // A single constant's size must be greater than singleThreshold.
  // The total size of contants must be greater than totalThreshold.
  std::string fname = outputNameNoExt + ".constants.bin";
  if (storeConstantsToFile) {
    storeConstantsToFile = extractConstantsToFile(module, fname,
        (uint64_t)constantsToFileSingleThreshold * 1024,
        (uint64_t)constantsToFileTotalThreshold * 1024 * 1024 * 1024);
  }

  // Request C wrapper emission via attribute.
  for (auto func : module.getOps<func::FuncOp>()) {
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
        UnitAttr::get(&getContext()));
  }

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Conversion target for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->conversionTargetKrnlToLLVM(target);

  // Convert types to legal types for the LLVM dialect.
  LLVMTypeConverter typeConverter(ctx, options);
  customizeTypeConverter(typeConverter);

  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(ctx);

  populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
      verifyInputTensors);

  // Rewrite patterns for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->rewritePatternKrnlToLLVM(patterns, typeConverter, ctx);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // Generate signature functions.
  if (entryGlobalOps.size() >= 1)
    genSignatureFunction(
        module, entryGlobalOps, inSigGlobalOps, outSigGlobalOps);

  // If globals are stored on external files. Emit helper functions to load
  // constants from files, and to free the allocated buffers for constants.
  if (storeConstantsToFile) {
    // Register runtime function calls, e.g. omXXX functions.
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry(module, builder, typeConverter);

    SmallVector<LLVM::GlobalOp> dataGlobalOps;
    // Emit a function, omLoadConstantsFromFile, that loads contants from files
    // to memory.
    loadConstantsFromFile(module, apiRegistry, entryGlobalOps, dataGlobalOps);
    // Emit a function, omFreeBuffersForConstants, that frees alocated buffers
    // in memory.
    freeBuffersForConstants(module, apiRegistry, entryGlobalOps, dataGlobalOps);
  }

  // Annotate global constants with `.lrodata` section if required.
  // Make sure this is always called at the end of this pass.
  if (useLRODATA) {
    module->walk([&](LLVM::GlobalOp gop) -> WalkResult {
      // Put all global constants into `.lrodata` instead of `.rodata` because
      // AI workloads often have a large amount of constants, especially large
      // language models.
      gop.getOperation()->setAttr("section", StringAttr::get(ctx, ".lrodata"));
      return WalkResult::advance();
    });
  }
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}
std::unique_ptr<Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useOpaquePointers, bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt) {
  return std::make_unique<ConvertKrnlToLLVMPass>(verifyInputTensors,
      useOpaquePointers, useLRODATA, storeConstantsToFile,
      constantsToFileSingleThreshold, constantsToFileTotalThreshold,
      outputNameNoExt);
}

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, SmallVector<MemRefType, 4>> &inputMemRefTypes,
    std::map<std::string, SmallVector<MemRefType, 4>> &outputMemRefTypes,
    bool verifyInputTensors) {
  krnl::populateLoweringKrnlEntryPointOpPattern(typeConverter, patterns, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
      verifyInputTensors);
  krnl::populateLoweringKrnlCallOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlFindIndexOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGlobalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGetRefOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlInstrumentOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMemcpyOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlPrintOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlPrintTensorOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlVectorTypeCastOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlRandomNormalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrlenOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlUnaryMathOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrncmpOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
