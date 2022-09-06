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

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
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
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

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

uint64_t KRNL_ENTRY_POINT_ID = 0;

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
    bool shouldOwn = true;
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      // Block argument, do not own this since it is an input that can be owned
      // by an input OMTensor.
      shouldOwn = false;
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
        shouldOwn = false;
      else if (llvm::dyn_cast<KrnlGlobalOp>(definingOp))
        // Do not own a constant that is defined by KrnlGlobalOp.
        shouldOwn = false;
    }
    outputOMTensorOwnerships.emplace_back(shouldOwn);
    LLVM_DEBUG(llvm::dbgs()
               << "Should the OMTensor own the entry function output? "
               << shouldOwn << "\n");
  }
}

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors) {
  // TODO: look at what is done in
  // mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp in function
  // LowerVectorToLLVMPass::runOnOperation() and see what we should do about it.
  // They run it in two steps, and add additional lowerings.

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns);

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
  arith::populateArithmeticExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, verifyInputTensors);
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
  Type i64PtrTy = LLVM::LLVMPointerType::get(i64Type);
  Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
  Type i8PtrPtrTy = LLVM::LLVMPointerType::get(i8PtrTy);

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
      Value address = create.llvm.addressOf(globalOp);
      Value zeroI64 = create.llvm.constant(i64Type, (int64_t)0);
      Value strAddr =
          create.llvm.getElemPtr(i8PtrTy, address, {zeroI64, zeroI64});
      lastValue =
          create.llvm.insertValue(arrayType, lastValue, strAddr, {index++});
    }

    // The last element of the array is NULL.
    Value nullPtr = create.llvm.nullI8Ptr();
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
          Value zero = createLLVM.constant(i64Type, (int64_t)0);
          Value numOfEntryPointsPtr =
              createLLVM.getElemPtr(i64PtrTy, numOfEntryPoints, {zero});
          Value noep =
              createLLVM.constant(i64Type, (int64_t)entryGlobalOps.size());
          createLLVM.store(noep, numOfEntryPointsPtr);
        });
    // Emit code to return the entry point array.
    Value entryAddr = create.llvm.addressOf(entryArrayOp);
    Value entryI8Ptr = create.llvm.bitcastI8PtrPtr(entryAddr);
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
            Value address = createLLVM.addressOf(globalEntryPoint);
            Value zeroI64 = createLLVM.constant(i64Type, (int64_t)0);
            Value entryI8Ptr =
                createLLVM.getElemPtr(i8PtrTy, address, {zeroI64, zeroI64});
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
            Value sigI8Ptr = createLLVM.bitcastI8Ptr(sigAddr);
            createLLVM._return(sigI8Ptr);
          });
    }

    // Return NULL if not found.
    create.llvm._return(create.llvm.nullI8Ptr());
  }
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
  ConvertKrnlToLLVMPass(bool verifyInputTensors) {
    this->verifyInputTensors = verifyInputTensors;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  Option<bool> verifyInputTensors{*this, "verify-input-tensors",
      llvm::cl::desc(
          "Verify input tensors whenever the entry point function is called.\n"
          "Data type and shape are verified. Enable this may introduce "
          "overhead in inferencing."),
      llvm::cl::init(false)};
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));
  KRNL_ENTRY_POINT_ID = 0;

  // Record entry point names and their input/output signatures.
  // This info is used to generate global signature functions.
  SmallVector<LLVM::GlobalOp, 1> entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps;

  // Determine the module has a single entry point or not.
  bool singleEntryPoint = hasSingleEntryPoint(module);

  // Request C wrapper emission via attribute.
  for (auto func : module.getOps<func::FuncOp>()) {
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
        UnitAttr::get(&getContext()));
  }

  // Determine whether an output OMTensor should own the underlying buffer or
  // not.
  SmallVector<bool, 4> outputOMTensorOwnerships;
  determineOwnershipForOutputOMTensors(module, outputOMTensorOwnerships);

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

#if 0
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
#endif

  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(ctx);

  populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, verifyInputTensors);

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
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}
std::unique_ptr<Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors) {
  return std::make_unique<ConvertKrnlToLLVMPass>(verifyInputTensors);
}

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors) {
  krnl::populateLoweringKrnlEntryPointOpPattern(typeConverter, patterns, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, verifyInputTensors);
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
