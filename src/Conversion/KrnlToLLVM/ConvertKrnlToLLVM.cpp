/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.cpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
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
    bool verifyInputTensors, bool enableParallel) {
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
  // Enable OpenMP-to-LLVM pass when enable parallelism
  if (enableParallel) {
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  }
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

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

void PostfixEntrypointNames(ModuleOp &module) {
  module->walk([&](KrnlEntryPointOp entryOp) -> WalkResult {
    Operation *op = entryOp.getOperation();
    std::string entryPointFuncName =
        op->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference()
            .getValue()
            .str();
    func::FuncOp entryPointFunc =
        dyn_cast<func::FuncOp>(module.lookupSymbol(entryPointFuncName));
    assert(entryPointFunc && "entry point func must exist");
    // Update the function name.
    entryPointFunc.setSymName(
        StringRef(LLVMBuilder::SymbolPostfix(module, entryPointFuncName)));
    // Reflect the new function name in the entry point.
    op->setAttr(KrnlEntryPointOp::getEntryPointFuncAttrName(),
        FlatSymbolRefAttr::get(entryPointFunc));
    return WalkResult::advance();
  });
  return;
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
    auto entryPointTy = mlir::dyn_cast<FunctionType>(
        dyn_cast<func::FuncOp>(entryPointFunc).getFunctionType());
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
    LLVM::LLVMFuncOp funcOp = create.llvm.func(
        "omQueryEntryPoints", llvmFnType, /*createUniqueFunc=*/true);
    // Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock(b);
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
    LLVM::LLVMFuncOp funcOp =
        create.llvm.func(funcNames[i], llvmFnType, /*createUniqueFunc=*/true);

    // 2. Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock(b);
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
      assert(mlir::isa<StringAttr>(globalEntryPoint.getValueAttr()) &&
             "Entry point value is not StringAttr");
      StringAttr entryPointValueAttr =
          mlir::cast<StringAttr>(globalEntryPoint.getValueAttr());

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

  Type llvmI8Ty = IntegerType::get(context, 8);
  Type llvmI8PtrTy = getPointerType(context, llvmI8Ty);
  Type llvmI64Ty = IntegerType::get(context, 64);

  // Check constants with thresholds.
  // Do not count constants whose size is <= singleThreshold.
  uint64_t totalSize = 0;
  SmallVector<KrnlGlobalOp> globalOfInterest;
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

    // Ignore constants of bool.
    // For an unknown reason, enabling constants of bool caused segfault in the
    // IBM granite.20B model (The model with KV cache) at 1265 input tokens.
    // See issue https://github.com/onnx/onnx-mlir/issues/2713.
    if (llvm::cast<MemRefType>(op->getResult(0).getType())
            .getElementType()
            .isInteger(1))
      return WalkResult::advance();

    // Get raw data from DenseElementsAttr or DenseResourceElementsAttr.
    ArrayRef<char> rawData = getRawData(op);
    if (rawData.empty())
      return WalkResult::advance();

    auto valueAttr = mlir::cast<ElementsAttr>(op.getValue().value());
    if (valueAttr.isSplat() || rawData.size() <= singleThreshold)
      return WalkResult::advance();

    globalOfInterest.emplace_back(op);
    totalSize += rawData.size();
    return WalkResult::advance();
  });
  // Do not use file if the total size of satisfied constants is <=
  // totalThreshold.
  if (totalSize <= totalThreshold)
    return false;

  // Sort constants in the non-descending order of alignment values.
  // Non-alignment is the smallest value (-1), the others are positive.
  llvm::sort(globalOfInterest, [&](KrnlGlobalOp left, KrnlGlobalOp right) {
    int64_t leftAlign = -1;
    int64_t rightAlign = -1;
    if (left.getAlignment().has_value())
      leftAlign = left.getAlignment().value();
    if (right.getAlignment().has_value())
      rightAlign = right.getAlignment().value();
    return (leftAlign < rightAlign);
  });

  // Pack all constants into a single buffer in order to save to file.
  // Constants with the highest alignment will be packed first in the file.
  // The file will be mmaped later at runtime and aligned at the page boundary,
  // So every constants must be correctly aligned in the packed constant. Pads
  // are added if necessary.
  std::vector<char> packedConst;
  for (int64_t i = globalOfInterest.size() - 1; i >= 0; --i) {
    KrnlGlobalOp op = globalOfInterest[i];
    ArrayRef<char> rawData = getRawData(op);

    // Get alignment.
    int64_t alignment = -1;
    if (op.getAlignment().has_value())
      alignment = op.getAlignment().value();

    // Padding if necessary.
    if ((alignment > 0) && (packedConst.size() % alignment != 0)) {
      uint64_t padSize =
          ((uint64_t)(packedConst.size() / alignment) + 1) * alignment -
          packedConst.size();
      SmallVector<char> pads(padSize, (char)0);
      packedConst.insert(packedConst.end(), pads.begin(), pads.end());
    }

    op.setOffsetAttr(b.getI64IntegerAttr(packedConst.size()));
    op.removeValueAttr();
    packedConst.insert(packedConst.end(), rawData.begin(), rawData.end());
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
  std::string fname = llvm::sys::path::filename(filepath).str() + '\0';
  mlir::StringAttr valueAttr = mlir::StringAttr::get(context, fname);
  create.llvm.globalOp(LLVM::LLVMArrayType::get(llvmI8Ty, fname.size()),
      /*isConstant=*/true, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "filename", valueAttr);
  // Create a global to store filesize.
  create.llvm.globalOp(llvmI64Ty,
      /*isConstant=*/true, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "filesize",
      b.getI64IntegerAttr(packedConst.size()));
  // Create a global to store isLE.
  bool isLE = llvm::endianness::native == llvm::endianness::little;
  create.llvm.globalOp(llvmI8Ty,
      /*isConstant=*/true, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "isLE", b.getI8IntegerAttr(isLE));
  // Create an uninitialized global into which we will load/mmap constants from
  // the file at runtime.
  LLVM::GlobalOp packedConstOp = create.llvm.globalOp(llvmI8PtrTy,
      /*isConstant=*/false, LLVM::Linkage::Internal,
      EXTERNAL_CONSTANT_PREFIX + "packedConst", nullptr);
  {
    OpBuilder::InsertionGuard insertGuard(b);
    Region &region = packedConstOp.getInitializerRegion();
    Block *block = b.createBlock(&region);
    // Initialize an array with the addresses of the global op.
    b.setInsertionPoint(block, block->begin());
    create.llvm._return(create.llvm.null(llvmI8PtrTy));
  }

  return true;
}

/// Emit a function "omLoadConstantsFromFile" in the IR to load constants from
/// external files into global operations.
void loadConstantsFromFile(ModuleOp &module,
    const RuntimeAPIRegistry &apiRegistry,
    const SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    bool calledByEntryPoint = true) {
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
  Type llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, {}, false);

  // If calledByEntryPoint, this function will be called by entry points.
  // Otherwise, user program (C/C++/Java/Python) would call this function.
  LLVM::LLVMFuncOp funcOp;
  if (calledByEntryPoint) {
    Operation *firstEntryPointOp =
        getFirstEntryOpInBlock(module, entryGlobalOps);
    assert(firstEntryPointOp && "No entry function exists");
    b.setInsertionPoint(firstEntryPointOp);
    funcOp = create.llvm.func(
        loadAllConstantsFuncName, llvmFnType, /*createUniqueFunc=*/true);
    // Call loadAllConstantsFuncName in each entry point function.
    bool zOS = isZOS(module);
    for (auto entryGlobalOp : entryGlobalOps) {
      std::string entryName =
          mlir::cast<StringAttr>(entryGlobalOp.getValue().value())
              .getValue()
              .str();
      // Entry point name is encoded in EBCDIC on z/OS.
      entryName = (zOS) ? krnl::e2a_s(entryName) : entryName;
      // Erase the null symbol.
      entryName.erase(
          std::find(entryName.begin(), entryName.end(), '\0'), entryName.end());
      auto entryFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(entryName);
      assert(entryFunc && "Entry function not found");
      b.setInsertionPoint(
          &entryFunc.getBody().front(), entryFunc.getBody().front().begin());
      FlatSymbolRefAttr loadAllConstantsRef = create.llvm.getOrInsertSymbolRef(
          module, LLVMBuilder::SymbolPostfix(module, loadAllConstantsFuncName),
          llvmVoidTy, {},
          /*isVarArg=*/false);
      create.llvm.call({}, loadAllConstantsRef, {});
    }
  } else {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    funcOp = create.llvm.func(
        loadAllConstantsFuncName, llvmFnType, /*createUniqueFunc=*/true);
  }

  // Emit the body of the function.
  Block *entryBlock = funcOp.addEntryBlock(b);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(entryBlock);

  // Get the constant file name.
  std::string fnameSymbol =
      LLVMBuilder::SymbolPostfix(module, EXTERNAL_CONSTANT_PREFIX + "filename");
  auto fnameGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(fnameSymbol);
  assert(fnameGlobalOp && "Could not find the global op for filename");
  Value fnameI8Ptr = krnl::getPtrToGlobalString(fnameGlobalOp, loc, b);
  // Get the file size.
  std::string fsizeSymbol =
      LLVMBuilder::SymbolPostfix(module, EXTERNAL_CONSTANT_PREFIX + "filesize");
  auto fsizeGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(fsizeSymbol);
  assert(fsizeGlobalOp && "Could not find the global op for filesize");
  int64_t dataSize = mlir::cast<IntegerAttr>(fsizeGlobalOp.getValue().value())
                         .getValue()
                         .getSExtValue();
  // Get the global op for isLE.
  std::string isleSymbol =
      LLVMBuilder::SymbolPostfix(module, EXTERNAL_CONSTANT_PREFIX + "isLE");
  auto isleGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(isleSymbol);
  assert(isleGlobalOp && "Could not find the global op for data isle");
  int64_t isle = mlir::cast<IntegerAttr>(isleGlobalOp.getValue().value())
                     .getValue()
                     .getSExtValue();
  // Get the packedConst global.
  std::string packedSymbol = LLVMBuilder::SymbolPostfix(
      module, EXTERNAL_CONSTANT_PREFIX + "packedConst");
  auto packedGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(packedSymbol);
  Value packedGlobalAddr = create.llvm.addressOf(packedGlobalOp);
  Value packedGlobalPtr = create.llvm.bitcast(llvmI8PtrTy, packedGlobalAddr);
  // Call a function to mmap the binary file to memory.
  Value isleVal = create.llvm.constant(llvmI64Ty, isle);
  Value sizeVal = create.llvm.constant(llvmI64Ty, dataSize);
  RuntimeAPI::callApi(b, loc, apiRegistry, RuntimeAPI::API::MMAP_BINARY_FILE,
      {packedGlobalPtr, fnameI8Ptr, sizeVal, isleVal});

  // Now set pointers for constants in the IR
  module->walk([&](LLVM::GlobalOp dataGlobalOp) -> WalkResult {
    // Get the global op for data.
    StringRef dataSymbol = dataGlobalOp.getSymName();
    std::string prefixData = EXTERNAL_CONSTANT_PREFIX + "data";
    if (!dataSymbol.starts_with(prefixData))
      return WalkResult::advance();
    std::string constantName = dataSymbol.drop_front(prefixData.size()).str();

    // Get offset.
    std::string offsetSymbol =
        EXTERNAL_CONSTANT_PREFIX + "offset" + constantName;
    auto offsetGlobalOp = module.lookupSymbol<LLVM::GlobalOp>(offsetSymbol);
    assert(offsetGlobalOp && "Could not find the global op for offset");
    int64_t offset = mlir::cast<IntegerAttr>(offsetGlobalOp.getValue().value())
                         .getValue()
                         .getSExtValue();

    // Set the data pointer pointing to the packedConst.
    Value dataGlobalAddr = create.llvm.addressOf(dataGlobalOp);
    Value dataPtr = create.llvm.bitcast(llvmI8PtrTy, dataGlobalAddr);
    Value offsetVal = create.llvm.constant(llvmI64Ty, offset);
    RuntimeAPI::callApi(b, loc, apiRegistry,
        RuntimeAPI::API::GET_EXTERNAL_CONSTANT_ADDR,
        {dataPtr, packedGlobalPtr, offsetVal});

    return WalkResult::advance();
  });

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
  ConvertKrnlToLLVMPass(bool verifyInputTensors, bool useLRODATA,
      bool storeConstantsToFile, uint64_t constantsToFileSingleThreshold,
      uint64_t constantsToFileTotalThreshold, std::string outputNameNoExt,
      bool enableParallel) {
    this->verifyInputTensors = verifyInputTensors;
    // Exclusive options. no option or only one option can be True.
    this->useLRODATA = useLRODATA;
    this->storeConstantsToFile = storeConstantsToFile;
    // store-constants-to-file has not yet been supported on Windows.
#ifdef _WIN32
    this->storeConstantsToFile = false;
#endif
    this->constantsToFileSingleThreshold = constantsToFileSingleThreshold;
    this->constantsToFileTotalThreshold = constantsToFileTotalThreshold;
    this->outputNameNoExt = outputNameNoExt;
    this->enableParallel = enableParallel;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
  }

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

  Option<bool> enableParallel{*this, "enable-parallel",
      llvm::cl::desc("Enable parallelization"), llvm::cl::init(false)};

private:
  std::string outputNameNoExt = "./model";
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));

  // Append a unique string to each entry point function.
  // The string is getting from the module's attribute
  // `onnx-mlir.symbol-postfix`.
  PostfixEntrypointNames(module);

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

  // Set legality for OMP constructs.
  configureOpenMPToLLVMConversionLegality(target, typeConverter);

  // Currently, only minimum required OpenMP Ops are marked as legal, in the
  // future integration of OpenMP, probably more OpenMP Ops are required to be
  // marked as legal. Please refer the Conversion/OpenMPToLLVM/OpenMPtoLLVM.cpp
  // in MLIR repo to see see how to legalize them.
  target.addLegalOp<omp::TerminatorOp, omp::YieldOp>();
  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(ctx);

  populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
      verifyInputTensors, enableParallel);

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
  // constants from files.
  if (storeConstantsToFile) {
    // Register runtime function calls, e.g. omXXX functions.
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry(module, builder, typeConverter);

    // Emit a function, omLoadConstantsFromFile, that loads contants from files
    // to memory.
    loadConstantsFromFile(module, apiRegistry, entryGlobalOps);
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
    bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel) {
  return std::make_unique<ConvertKrnlToLLVMPass>(verifyInputTensors, useLRODATA,
      storeConstantsToFile, constantsToFileSingleThreshold,
      constantsToFileTotalThreshold, outputNameNoExt, enableParallel);
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
  krnl::populateLoweringKrnlNoneOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
