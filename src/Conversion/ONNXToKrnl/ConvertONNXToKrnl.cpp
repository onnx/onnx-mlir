/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToKrnl.cpp - ONNX dialects to Krnl lowering -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// EntryPoint Op lowering to Krnl Entry Point.
//===----------------------------------------------------------------------===//

class ONNXEntryPointLowering : public OpRewritePattern<ONNXEntryPointOp> {
public:
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  // A type mapping used to generate a signature in JSON.
  static std::map<std::string, std::string> typeMap;

  LogicalResult matchAndRewrite(
      ONNXEntryPointOp op, PatternRewriter &rewriter) const override {
    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();

    SymbolRefAttr funcRefAttr = op->getAttrOfType<SymbolRefAttr>(
        ONNXEntryPointOp::getEntryPointFuncAttrName());
    StringRef entryPointName = funcRefAttr.getLeafReference().getValue();
    Operation *entryPointOp = module.lookupSymbol(entryPointName);
    func::FuncOp entryPointFunc = cast<func::FuncOp>(entryPointOp);

    IntegerAttr numInputsAttr =
        rewriter.getI32IntegerAttr(entryPointFunc.getArgumentTypes().size());
    IntegerAttr numOutputsAttr =
        rewriter.getI32IntegerAttr(entryPointFunc.getResultTypes().size());
    std::string sig =
        getSignature(entryPointFunc.getFunctionType(), entryPointOp);
    StringAttr sigAtrr = rewriter.getStringAttr(sig);

    rewriter.replaceOpWithNewOp<KrnlEntryPointOp>(
        op, funcRefAttr, numInputsAttr, numOutputsAttr, sigAtrr);
    return success();
  }

private:
  // Construct JSON type from the argument type.
  // for example - a 3D array of f32 would produce something like
  //     {"type" : "f32" , "dims" : [4, 256, 16] , "name": "t1"}
  // data type list:
  //     "i1" / "i8" / "i16" / "i32" / "i64"
  //     "ui8" / "ui16" / "ui32" / "ui64"
  //     "f32" / "f64"
  void concatTypeString(
      Type argType, Attribute attr, llvm::raw_ostream &dstream) const {
    std::string comma = std::string("");

    TypeSwitch<Type>(argType)
        .Case<mlir::SeqType>([&](mlir::SeqType seqTy) {
          auto et = seqTy.getElementType();
          dstream << "   {\"seq\" : ";
          concatTypeString(et, attr, dstream);
        })
        .Case<ShapedType>([&](ShapedType tensorTy) {
          auto et = tensorTy.getElementType();
          dstream << "   { \"type\" : ";
          et.print(dstream);
          dstream << " , \"dims\" : [";
          if (tensorTy.hasRank()) {
            int64_t rank = tensorTy.getRank();
            for (int j = 0; j < rank; j++) {
              dstream << comma << tensorTy.getDimSize(j);
              comma = std::string(" , ");
            }
          } else {
          }
          dstream << "] ";
          auto name = attr.cast<mlir::StringAttr>().getValue().str();
          dstream << ", \"name\" : \"" << name << "\"";
        })
        .Default([&](Type type) { llvm_unreachable("input is not a tensor"); });
    dstream << " }\n";
  }

  std::string getSignature(FunctionType funcType, Operation *op) const {
    OpBuilder b(op);
    auto inputs = funcType.getInputs();
    auto outputs = funcType.getResults();

    ArrayAttr inputNames = op->getAttrOfType<ArrayAttr>("input_names");
    if (!inputNames) {
      SmallVector<StringRef, 4> names;
      for (uint64_t i = 0; i < inputs.size(); ++i)
        names.emplace_back(StringRef("input_" + std::to_string(i)));
      inputNames = b.getStrArrayAttr(names);
    }
    ArrayAttr outputNames = op->getAttrOfType<ArrayAttr>("output_names");
    if (!outputNames) {
      SmallVector<StringRef, 4> names;
      for (uint64_t i = 0; i < outputs.size(); ++i)
        names.emplace_back(StringRef("output_" + std::to_string(i)));
      outputNames = b.getStrArrayAttr(names);
    }

    std::string dstring;
    llvm::raw_string_ostream dstream(dstring);
    dstream << "[ ";
    std::string comma = std::string("");
    for (unsigned int i = 0; i < funcType.getNumInputs(); i++) {
      dstream << comma;
      concatTypeString(inputs[i], inputNames[i], dstream);
      comma = std::string(" , ");
    }
    dstream << "\n]";
    dstream.flush();
    dstring.push_back('\0'); // null terminate the input signature string
    dstream << "@[";
    comma = std::string("");
    for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
      dstream << comma;
      concatTypeString(outputs[i], outputNames[i], dstream);
      comma = std::string(" , ");
    }
    dstream << "\n]";
    dstream.flush();
    dstring.push_back('\0'); // null terminate the output signature string
    for (auto const &x : typeMap) {
      size_t start_pos = 0;
      while (
          (start_pos = dstring.find(x.first, start_pos)) != std::string::npos) {
        dstring.replace(start_pos, x.first.length(), x.second);
        start_pos += x.first.length();
      }
    }

    return dstring;
  }
};

std::map<std::string, std::string> ONNXEntryPointLowering::typeMap = {
    {std::string(" f32 "), std::string(" \"f32\" ")},
    {std::string(" f64 "), std::string(" \"f64\" ")},
    {std::string(" i32 "), std::string(" \"i32\" ")},
    {std::string(" i64 "), std::string(" \"i64\" ")},
    {std::string(" i16 "), std::string(" \"i16\" ")},
    {std::string(" i8 "), std::string(" \"i8\" ")},
    {std::string(" i1 "), std::string(" \"i1\" ")},
    {std::string(" ui32 "), std::string(" \"ui32\" ")},
    {std::string(" ui64 "), std::string(" \"ui64\" ")},
    {std::string(" ui16 "), std::string(" \"ui16\" ")},
    {std::string(" ui8 "), std::string(" \"ui8\" ")}};

void populateONNXToKrnlConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableTiling) {
  // Type conversion for function signatures.
  // Call MLIR FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Frontend operation lowering.
  // ControlFlow
  populateLoweringONNXLoopOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXScanOpPattern(patterns, typeConverter, ctx);
  // Math
  populateLoweringONNXClipOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXCumSumOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXElementwiseOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXGemmOpPattern(patterns, typeConverter, ctx, enableTiling);
  populateLoweringONNXHardmaxOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXReductionOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSoftmaxOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXTopKOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXMatMulOpPattern(
      patterns, typeConverter, ctx, enableTiling);
  populateLoweringONNXRandomNormalOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXRandomNormalLikeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXLRNOpPattern(patterns, typeConverter, ctx);
  // ML
  populateLoweringONNXCategoryMapperOpPattern(patterns, typeConverter, ctx);
  // ObjectDetection
  populateLoweringONNXNonMaxSuppressionOpPattern(patterns, typeConverter, ctx);
  // Tensor
  populateLoweringONNXArgMaxOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXPadOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXUnsqueezeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXUnsqueezeV11OpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXTransposeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXGatherOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXGatherElementsOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXGatherNDOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXIdentityOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXConstantOfShapeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXConstantOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXConcatOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXDepthToSpaceOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXScatterElementsOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXScatterNDOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSpaceToDepthOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXShapeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSliceOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSqueezeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSqueezeV11OpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSplitOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSplitV11OpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSizeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXTileOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXFlattenOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXRangeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXResizeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXNonZeroOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXReverseSequenceOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXExpandOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXOneHotOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXCompressOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXPrintSignaturePattern(patterns, typeConverter, ctx);
  // Neural network
  populateLoweringONNXConvOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXNormalizationOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXPoolingOpPattern(patterns, typeConverter, ctx);
  // Recurrent neural network
  populateLoweringONNXGRUOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXLSTMOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXRNNOpPattern(patterns, typeConverter, ctx);
  // Sequence
  populateLoweringONNXSequenceAtOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSequenceEmptyOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSequenceEraseOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSequenceInsertOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXSequenceLengthOpPattern(patterns, typeConverter, ctx);
  // Entry point
  patterns.insert<ONNXEntryPointLowering>(ctx);
}

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
struct FrontendToKrnlLoweringPass
    : public PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FrontendToKrnlLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-krnl"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to Krnl dialect.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FrontendToKrnlLoweringPass() = default;
  FrontendToKrnlLoweringPass(const FrontendToKrnlLoweringPass &pass)
      : PassWrapper<FrontendToKrnlLoweringPass, OperationPass<ModuleOp>>() {}
  FrontendToKrnlLoweringPass(bool emitDealloc, bool enableTiling) {
    // Below, need explicit assignment to enable implicit conversion of bool to
    // Option<bool>.
    this->emitDealloc = emitDealloc;
    this->enableTiling = enableTiling;
  }
  FrontendToKrnlLoweringPass(int optLevel)
      : FrontendToKrnlLoweringPass(
            /*emitDealloc=*/false, /*enableTiling=*/optLevel >= 3) {}

  void runOnOperation() final;

public:
  // Some ops (RNN ops for example) are lowered to other ONNX ops such as
  // ONNXMatMulOp, ONNXSplitOp, ONNXTransposeOp, etc. These ONNX ops are then
  // lowered into krnl ops in this pass.
  //
  // To write LIT tests for operations that are lowered to other ONNX
  // operations, we do not need to check the final generated krnl code (which is
  // lengthy). It is more convenient to check the intermediate generated code
  // including ONNX ops. We trust the lowering of the other ONNX ops.
  //
  // This flag is used in LIT tests to stop the lowering of the other ONNX ops.
  // Usage: onnx-mlir-opt --convert-onnx-to-krnl='emit-intermediate-ir'
  Option<bool> emitIntermediateIR{*this, "emit-intermediate-ir",
      llvm::cl::desc(
          "Emit intermediate IR rather than lowering to the krnl dialect."),
      llvm::cl::init(false)};
  Option<bool> emitDealloc{*this, "emit-dealloc",
      llvm::cl::desc("Emit dealloc for allocated memrefs or not."),
      llvm::cl::init(false)};
  Option<bool> enableTiling{*this, "enable-tiling",
      llvm::cl::desc("Enable loop tiling and unrolling optimizations"),
      llvm::cl::init(false)};
};

void FrontendToKrnlLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Set up whether emitting dealloc for allocated memrefs or not.
  ONNXToKrnl_gEmitDealloc = emitDealloc;

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<KrnlDialect, AffineDialect, arith::ArithmeticDialect,
      func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
      memref::MemRefDialect, shape::ShapeDialect, scf::SCFDialect>();
  // Needed to support unsigned int computations. To be removed if we use a
  // scheme that does not rely on the UnrealizedConversionCastOp.
  target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
  // Make ONNXNoneOp legal so that other ONNX ops can use it during the
  // lowering. ONNXNoneOp will be dangling and removed by calling
  // canonicalization after the lowering.
  target.addLegalOp<::mlir::ONNXNoneOp>();

  // Use krnl.load/store instead of std.load/store and affine.load/store.
  // krnl.load/store will be lowered to std.load/store and affine.load/store by
  // `convert-krnl-to-affine` pass.
  target.addIllegalOp<mlir::memref::LoadOp>();
  target.addIllegalOp<mlir::AffineLoadOp>();
  target.addIllegalOp<mlir::memref::StoreOp>();
  target.addIllegalOp<mlir::AffineStoreOp>();

  // If `emitDealloc` is turned off, make sure we don't have buffer deallocation
  // at this level. Will use MLIR buffer-deallocation for this purpose instead.
  if (!emitDealloc)
    target.addIllegalOp<mlir::memref::DeallocOp>();

  // TODO: enable this once more ops are supported.
  // We also define the ONNX dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<mlir::ONNXOpsDialect>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  if (emitIntermediateIR) {
    // Only used for writing LIT tests for ONNX operations that are lowered to
    // other ONNX operations. The following operations are prevented from being
    // lowered further. See the comment in the declaration of
    // 'emitIntermediateIR' for more details.
    target.addLegalOp<ONNXMatMulOp>();
    target.addLegalOp<ONNXReshapeOp>();
    target.addLegalOp<ONNXSplitV11Op>();
    target.addLegalOp<ONNXSqueezeV11Op>();
    target.addLegalOp<ONNXTransposeOp>();
  }

  // Conversion target for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->conversionTargetONNXToKrnl(target);

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  RewritePatternSet patterns(&getContext());

  // Convert types to legal types for the Krnl dialect.
  KrnlTypeConverter krnlTypeConverter;
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
        [](Type type) { return type.isa<TensorType>(); });
  });

  // Define patterns.
  populateONNXToKrnlConversionPattern(
      patterns, krnlTypeConverter, &getContext(), enableTiling);

  // Rewrite patterns for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->rewritePatternONNXToKrnl(patterns, krnlTypeConverter, &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

std::unique_ptr<Pass> createLowerToKrnlPass(int optLevel) {
  return std::make_unique<FrontendToKrnlLoweringPass>(optLevel);
}

std::unique_ptr<Pass> createLowerToKrnlPass(
    bool emitDealloc, bool enableTiling) {
  return std::make_unique<FrontendToKrnlLoweringPass>(
      emitDealloc, enableTiling);
}

} // namespace onnx_mlir
