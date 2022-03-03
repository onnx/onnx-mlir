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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#include "onnx/onnx_pb.h"

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

void checkConstantOutputs(
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

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint) {
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
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
  krnl::populateKrnlToLLVMConversion(
      typeConverter, patterns, ctx, constantOutputs, singleEntryPoint);
}

void recordEntryPointSignatures(ModuleOp &module,
    SmallVectorImpl<std::string> &entryPointNames,
    SmallVectorImpl<std::string> &inSignatures,
    SmallVectorImpl<std::string> &outSignatures) {

  bool zOS = false;
  if (Attribute mtripleAttr =
          module->getAttrOfType<::mlir::Attribute>("llvm.target_triple"))
    zOS = llvm::Triple(mtripleAttr.cast<StringAttr>().getValue()).isOSzOS();

  module->walk([&](KrnlEntryPointOp entryOp) -> WalkResult {
    Operation *op = entryOp.getOperation();
    // Entry point name.
    llvm::StringRef entryPointName =
        op->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference()
            .getValue();
    std::string terminatedEntryPointName = "run_" + entryPointName.str();
    terminatedEntryPointName.push_back('\0'); // null terminate the string.
    if (zOS)
      entryPointNames.emplace_back(krnl::a2e_s(terminatedEntryPointName));
    else
      entryPointNames.emplace_back(terminatedEntryPointName);

    // Input/output signatures.
    StringAttr sigAttr =
        op->getAttrOfType<StringAttr>(KrnlEntryPointOp::getSignatureAttrName());
    llvm::StringRef signature = sigAttr.getValue();
    auto splitSig = signature.split('@');
    llvm::StringRef inSig = splitSig.first;
    llvm::StringRef outSig = splitSig.second;
    if (zOS) {
      inSignatures.emplace_back(krnl::a2e_s(inSig.str()));
      outSignatures.emplace_back(krnl::a2e_s(outSig.str()));
    } else {
      inSignatures.emplace_back(inSig.str());
      outSignatures.emplace_back(outSig.str());
    }

    return WalkResult::advance();
  });

  // When there is only a single entry point function in a model, use
  // DEFAULT_DYN_ENTRY_POINT.
  if (entryPointNames.size() == 1) {
    std::string defaultEntryPoint = DEFAULT_DYN_ENTRY_POINT;
    defaultEntryPoint.push_back('\0'); // null terminate the string.
    if (zOS)
      defaultEntryPoint = krnl::a2e_s(defaultEntryPoint);
    entryPointNames[0] = defaultEntryPoint;
  }
}

/// This function emits three functions: omQueryEntryPoints, omInputSignature
/// and omOutputSignature.
/// - omQueryEntryPoints has type of `**i8 ()` to query an array of entry point
/// names.
/// - omInputSignature and omOutputSignature have type of type `*i8 (*i8)` to
/// return input and output signatures of the given entry point.
void genSignatureFunction(ModuleOp module,
    const ArrayRef<std::string> entryPointNames,
    const ArrayRef<std::string> inSignatures,
    const ArrayRef<std::string> outSignatures) {
  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  OpBuilder b(context);

  // Common information.
  Type i8Type = IntegerType::get(context, 8);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
  Type i8PtrPtrTy = LLVM::LLVMPointerType::get(i8PtrTy);
  IntegerAttr zeroI32Attr = b.getI32IntegerAttr(0);
  IntegerAttr zeroI64Attr = b.getI64IntegerAttr(0);
  IntegerAttr oneI64Attr = b.getI64IntegerAttr(1);

  uint64_t numOfEntryPoints = entryPointNames.size();

  // A helper function to emit a global constant operation storing a string.
  auto emitGlobalOp = [&context, &b, &loc, &i8Type](
                          std::string name, std::string value) {
    mlir::StringAttr valueAttr = mlir::StringAttr::get(context, value);
    Type valueArrayType = LLVM::LLVMArrayType::get(i8Type, value.size());
    LLVM::GlobalOp globalOp = b.create<LLVM::GlobalOp>(loc, valueArrayType,
        /*isConstant=*/true, LLVM::Linkage::External, name, valueAttr);
    return globalOp;
  };

  // A helper function to get a pointer to the first element in an array.
  auto getGlobalOpGEP = [&loc, &b, &i8PtrTy, &i64Type, &zeroI64Attr](
                            LLVM::GlobalOp op) {
    Value zeroI64 = b.create<LLVM::ConstantOp>(loc, i64Type, zeroI64Attr);
    Value address = b.create<LLVM::AddressOfOp>(loc, op);
    LLVM::GEPOp gepOp = b.create<LLVM::GEPOp>(
        loc, i8PtrTy, address, ArrayRef<Value>({zeroI64, zeroI64}));
    return gepOp;
  };

  // For each entry point name, emit three global constants to store the entry
  // point name and input/output signatures. For the i-th entry point, these
  // constants are named as follows:
  // - Entry point name: `_entry_point_i`.
  // - Input signature: `_entry_point_i_in_sig`.
  // - Output signature: `_entry_point_i_out_sig`.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  SmallVector<LLVM::GlobalOp, 2> entryOps, inSigOps, outSigOps;
  for (uint64_t i = 0; i < numOfEntryPoints; ++i) {
    // Global constants for entry point names.
    std::string entryVarName = "_entry_point_" + std::to_string(i);
    LLVM::GlobalOp entryOp = emitGlobalOp(entryVarName, entryPointNames[i]);
    entryOps.emplace_back(entryOp);

    // Global constants for input signatures.
    std::string inSigVarName = entryVarName + "_in_sig";
    LLVM::GlobalOp inSigOp = emitGlobalOp(inSigVarName, inSignatures[i]);
    inSigOps.emplace_back(inSigOp);

    // Global constants for output signatures.
    std::string outSigVarName = entryVarName + "_out_sig";
    LLVM::GlobalOp outSigOp = emitGlobalOp(outSigVarName, outSignatures[i]);
    outSigOps.emplace_back(outSigOp);
  }

  // Emit a global constant to store an array of pointers pointing to each entry
  // point constants. The array ends with NULL.
  auto arrayType = LLVM::LLVMArrayType::get(i8PtrTy, entryOps.size() + 1);
  auto entryArrayOp = b.create<LLVM::GlobalOp>(loc, arrayType,
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
    for (const LLVM::GlobalOp &globalOp : entryOps) {
      LLVM::GEPOp strAddr = getGlobalOpGEP(globalOp);
      lastValue = b.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
          strAddr, b.getArrayAttr({b.getIndexAttr(index++)}));
    }

    // The last element of the array is NULL.
    Value nullPtr = b.create<LLVM::NullOp>(loc, i8PtrTy);
    lastValue = b.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
        nullPtr, b.getArrayAttr({b.getIndexAttr(index++)}));
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({lastValue}));
  }

  // Emit a function, omQueryEntryPoints, of type `**8 ()` to query an array of
  // entry point names.
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // Emit the function type.
    Type llvmFnType = LLVM::LLVMFunctionType::get(i8PtrPtrTy, {}, false);
    LLVM::LLVMFuncOp funcOp =
        b.create<LLVM::LLVMFuncOp>(loc, "omQueryEntryPoints", llvmFnType);
    // Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);
    Value entryAddr = b.create<LLVM::AddressOfOp>(loc, entryArrayOp);
    Value entryI8Ptr = b.create<LLVM::BitcastOp>(loc, i8PtrPtrTy, entryAddr);
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({entryI8Ptr}));
  }

  // Emit two signature functions, omInputSignature and omOutputSignature, of
  // type `*i8 (*i8)` at the end of the module.
  SmallVector<std::string, 2> funcNames = {
      "omInputSignature", "omOutputSignature"};
  SmallVector<SmallVector<LLVM::GlobalOp, 2>, 2> sigOps = {inSigOps, outSigOps};
  for (uint64_t i = 0; i < funcNames.size(); ++i) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // 1. Emit the function type.
    Type llvmFnType = LLVM::LLVMFunctionType::get(i8PtrTy, {i8PtrTy}, false);
    LLVM::LLVMFuncOp funcOp =
        b.create<LLVM::LLVMFuncOp>(loc, funcNames[i], llvmFnType);

    // 2. Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);

    Value zeroI32 = b.create<LLVM::ConstantOp>(loc, i32Type, zeroI32Attr);
    Value oneI64 = b.create<LLVM::ConstantOp>(loc, i64Type, oneI64Attr);

    // 2.1 A buffer to keep a pointer pointing to the return signature string.
    Value ptrToReturnSig = b.create<LLVM::AllocaOp>(loc, i8PtrPtrTy, oneI64,
        /*alignment=*/0);

    // 2.2 The name of the entry point that we want to return its signature.
    Value input = entryBlock->getArgument(0);

    // 2.3 Emit code to find the signature of the given entry point.
    // Iterate over the list of the entry points and check string equality.

    // Split the current block into condition, true, false, and end blocks.
    // - If the user's entry point name is found, go to the true block, then the
    // end block.
    // - Otherwise, recursively split the false block.
    Block *condBlock, *trueBlock, *falseBlock, *endBlock;
    condBlock = b.getInsertionBlock();
    trueBlock = condBlock->splitBlock(b.getInsertionPoint());
    falseBlock = b.createBlock(
        trueBlock->getParent(), std::next(Region::iterator(trueBlock)));
    endBlock = b.createBlock(
        falseBlock->getParent(), std::next(Region::iterator(falseBlock)));

    // Emit code for the end block.
    b.setInsertionPointToStart(endBlock);
    Value res = b.create<LLVM::LoadOp>(loc, i8PtrTy, ptrToReturnSig);
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({res}));

    // Emit code for the condition, true and false blocks.
    for (uint64_t j = 0; j < numOfEntryPoints; ++j) {
      LLVM::GlobalOp globalEntryPoint = entryOps[j];
      LLVM::GlobalOp globalSignature = sigOps[i][j];
      std::string entryPointName = entryPointNames[j];
      // Emit code for the condition block.
      b.setInsertionPointToEnd(condBlock);
      // Read an entry point name.
      Value entryI8Ptr = getGlobalOpGEP(globalEntryPoint).getResult();
      // Compare it with the user's entry point name.
      FlatSymbolRefAttr StrncmpRef = krnl::getOrInsertStrncmp(b, module);
      Value length = b.create<LLVM::ConstantOp>(
          loc, i64Type, b.getI64IntegerAttr(entryPointName.size()));
      Value strncmpResult = b.create<LLVM::CallOp>(loc, i32Type, StrncmpRef,
                                 ArrayRef<Value>({input, entryI8Ptr, length}))
                                .getResult(0);
      // Equal if strncmp returns `0`.
      Value found = b.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::eq, strncmpResult, zeroI32);
      llvm::SmallVector<Value, 1> results = {entryI8Ptr};
      // Branch the block into the true and false blocks.
      b.create<LLVM::CondBrOp>(
          loc, found, trueBlock, ValueRange(), falseBlock, ValueRange());

      // Emit code for the true block.
      b.setInsertionPointToStart(trueBlock);
      Value sigAddr = b.create<LLVM::AddressOfOp>(loc, globalSignature);
      Value sigI8Ptr = b.create<LLVM::BitcastOp>(loc, i8PtrTy, sigAddr);
      b.create<LLVM::StoreOp>(loc, sigI8Ptr, ptrToReturnSig);
      b.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

      // Emit code for the false block.
      b.setInsertionPointToStart(falseBlock);
      if (j == numOfEntryPoints - 1)
        b.create<LLVM::BrOp>(loc, ValueRange(), endBlock);
      else {
        // Recursively do with the other entry point names.
        condBlock = b.getInsertionBlock();
        trueBlock = condBlock->splitBlock(b.getInsertionPoint());
        falseBlock = b.createBlock(
            trueBlock->getParent(), std::next(Region::iterator(trueBlock)));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));
  options.emitCWrappers = true;

  // Determine, for each output, whether it is a constant or not.
  SmallVector<bool, 4> constantOutputs;
  checkConstantOutputs(module, constantOutputs);

  // Record entry point names and their input/output signatures.
  // This info is used to generate global signature functions.
  SmallVector<std::string, 1> entryPointNames, inSignatures, outSignatures;
  recordEntryPointSignatures(
      module, entryPointNames, inSignatures, outSignatures);

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

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
      constantOutputs,
      /*singleEntryPoint=*/entryPointNames.size() == 1);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // Generate signature functions.
  if (entryPointNames.size() >= 1)
    genSignatureFunction(module, entryPointNames, inSignatures, outSignatures);
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint) {
  krnl::populateLoweringKrnlEntryPointOpPattern(
      typeConverter, patterns, ctx, constantOutputs, singleEntryPoint);
  krnl::populateLoweringKrnlFindIndexOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGlobalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGetRefOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlInstrumentOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMemcpyOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlVectorTypeCastOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlRandomNormalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrlenOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlUnaryMathOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrncmpOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
