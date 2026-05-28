
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlInstrument.cpp - Lower KrnlInstrumentOp -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlInstrumentOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

// Module-attribute key under which we cache a single DICompileUnit
// for all instrument calls. Keeps the DWARF compile-unit count at one
// per module rather than one per instrument call.
static constexpr llvm::StringLiteral kInstrumentCUAttrName =
    "onnx-mlir.instrument.cu";

// Build (or look up) a DICompileUnit attribute hosted on the parent
// module. Synthetic file/producer fields — they're never opened, just
// referenced by the DISubprograms we attach to each instrument call.
//
// The DIFile must have BOTH a normal-looking filename AND a non-empty
// directory. macOS ld21 silently skips writing an `N_OSO` debug-map
// entry for any object whose CU has either an angle-bracket
// "synthetic" filename (e.g. "<onnx-mlir-instrument>") or an empty
// `DW_AT_comp_dir` — and without N_OSO, dsymutil drops the object's
// DWARF entirely from the final `.dSYM`. The path is never opened, so
// "/" works; only its presence and shape matter.
static LLVM::DICompileUnitAttr getOrCreateInstrumentCU(ModuleOp module) {
  if (auto cached = module->getAttrOfType<LLVM::DICompileUnitAttr>(
          kInstrumentCUAttrName))
    return cached;

  MLIRContext *ctx = module.getContext();
  auto fileAttr = LLVM::DIFileAttr::get(
      ctx, /*name=*/"onnx-mlir-instrument.mlir", /*directory=*/"/");
  auto producerAttr = StringAttr::get(ctx, "onnx-mlir");
  auto cu = LLVM::DICompileUnitAttr::get(ctx,
      DistinctAttr::create(UnitAttr::get(ctx)),
      /*sourceLanguage=*/llvm::dwarf::DW_LANG_C99, fileAttr, producerAttr,
      /*isOptimized=*/true, LLVM::DIEmissionKind::Full,
      LLVM::DINameTableKind::Default,
      /*splitDebugFilename=*/StringAttr{});
  module->setAttr(kInstrumentCUAttrName, cu);
  return cu;
}

// Synthetic anchor for the inner of our FusedLoc<DISubprogramAttr>
// wrappers. The MLIR→LLVM debug translator returns nullptr for any
// inner loc that can't be turned into a `DILocation` — `UnknownLoc`
// directly, but also `NameLoc` / `FusedLoc` whose chains bottom out
// in `UnknownLoc`. When that fails, the parent `CallSiteLoc`
// translation falls back to the caller's loc, dropping the inlined
// `__omip:` scope from the `!dbg` and from the dSYM. ONNX imports
// without preserved source locations (most production `.onnx`) hit
// this path. The simplest robust shape is to always anchor with a
// concrete `FileLineColLoc`; the line/col are unused — only the
// scope carried by the surrounding FusedLoc matters for DWARF.
static Location syntheticAnchorLoc(MLIRContext *ctx) {
  return FileLineColLoc::get(
      StringAttr::get(ctx, "onnx-mlir-instrument.mlir"), 0, 0);
}

// Build a fresh DISubprogramAttr for one OMInstrumentPoint call site.
// `kind` is "begin" / "end" so begin and end calls get distinct DIEs
// even though they share opName + nodeName.
static LLVM::DISubprogramAttr buildInstrumentSubprogram(MLIRContext *ctx,
    LLVM::DICompileUnitAttr cuAttr, LLVM::DIFileAttr fileAttr,
    StringRef opName, StringRef nodeName) {
  std::string label = ("__omip:" + opName + ":" + nodeName).str();
  auto nameAttr = StringAttr::get(ctx, label);
  auto srTypeAttr = LLVM::DISubroutineTypeAttr::get(
      ctx, /*callingConvention=*/0, /*types=*/{});
  return LLVM::DISubprogramAttr::get(ctx,
      /*id=*/DistinctAttr::create(UnitAttr::get(ctx)),
      /*compileUnit=*/cuAttr, /*scope=*/fileAttr,
      /*name=*/nameAttr, /*linkageName=*/nameAttr, fileAttr,
      /*line=*/0, /*scopeLine=*/0,
      LLVM::DISubprogramFlags::Definition |
          LLVM::DISubprogramFlags::Optimized,
      srTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});
}

// Lazily attach a function-level DISubprogramAttr to `funcOp` and
// return its location (which after attachment is a FusedLocWith
// carrying the DISubprogram as metadata — the `!dbg` LLVM uses for
// the function definition). Without this anchor, the inlined-
// subroutine DIEs we attach later to call sites are orphans (no
// parent function PC range to anchor against) and LLVM's DwarfDebug
// pass silently drops them.
//
// We cache by inspecting funcOp.getLoc(): if it's already a
// FusedLocWith<DISubprogramAttr> we don't re-attach.
static Location getOrAttachFuncDISubprogram(
    LLVM::LLVMFuncOp funcOp, LLVM::DICompileUnitAttr cuAttr) {
  MLIRContext *ctx = funcOp.getContext();
  if (isa<FusedLocWith<LLVM::DISubprogramAttr>>(funcOp.getLoc()))
    return funcOp.getLoc();
  auto fileAttr = cuAttr.getFile();
  auto srTypeAttr = LLVM::DISubroutineTypeAttr::get(
      ctx, /*callingConvention=*/0, /*types=*/{});
  auto funcSP = LLVM::DISubprogramAttr::get(ctx,
      /*id=*/DistinctAttr::create(UnitAttr::get(ctx)),
      /*compileUnit=*/cuAttr, /*scope=*/fileAttr,
      /*name=*/funcOp.getSymNameAttr(),
      /*linkageName=*/funcOp.getSymNameAttr(), fileAttr,
      /*line=*/0, /*scopeLine=*/0,
      LLVM::DISubprogramFlags::Definition |
          LLVM::DISubprogramFlags::Optimized,
      srTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});
  Location funcLoc = FusedLocWith<LLVM::DISubprogramAttr>::get(
      {syntheticAnchorLoc(ctx)}, funcSP, ctx);
  funcOp->setLoc(funcLoc);
  return funcLoc;
}

// Build the call instruction's location. We emit a CallSiteLoc whose
// callee is anchored in a synthetic `__omip:<opName>:<nodeName>`
// DISubprogram and whose caller is anchored in the enclosing
// function's own DISubprogram. mlir-translate turns that pair into a
// DILocation with `inlinedAt` pointing at the function-level
// DILocation — which is exactly the shape LLVM's DwarfDebug pass
// recognizes and emits as a DW_TAG_inlined_subroutine DIE with the
// `__omip:` name. Any PC inside the call (and, more usefully, any
// later sampled PC bracketed between consecutive begin/end inline
// subroutines) is then resolvable via `addr2line --inlines` to the
// originating ONNX op.
static Location buildInstrumentMarkerLoc(MLIRContext *ctx,
    LLVM::LLVMFuncOp funcOp, ModuleOp module, Location originalLoc,
    StringRef opName, StringRef nodeName) {
  auto cuAttr = getOrCreateInstrumentCU(module);
  auto fileAttr = cuAttr.getFile();

  // Function-level anchor (created on demand, cached on the FuncOp).
  Location funcLoc = getOrAttachFuncDISubprogram(funcOp, cuAttr);

  // Inline scope for THIS call site.
  auto inlineSP = buildInstrumentSubprogram(
      ctx, cuAttr, fileAttr, opName, nodeName);
  (void)originalLoc; // intentionally not embedded; see syntheticAnchorLoc
  auto inlineLoc = FusedLocWith<LLVM::DISubprogramAttr>::get(
      {syntheticAnchorLoc(ctx)}, inlineSP, ctx);

  // CallSiteLoc(callee=inlineLoc, caller=funcLoc) → MLIR translates
  // this into a DILocation with `inlinedAt` chain pointing at funcLoc.
  return CallSiteLoc::get(/*callee=*/inlineLoc, /*caller=*/funcLoc);
}

class KrnlInstrumentOpLowering : public ConversionPattern {
public:
  explicit KrnlInstrumentOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlInstrumentOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlInstrumentOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    KrnlInstrumentOp instrumentOp = llvm::dyn_cast<KrnlInstrumentOp>(op);

    StringRef opNameStr = instrumentOp.getOpName();

    StringRef nodeName;
    if (instrumentOp.getNodeName().has_value()) {
      // If we can get it from the instrument op direct, do so
      nodeName = instrumentOp.getNodeName().value();
    } else {
      // Otherwise, backup by creating it from the op.
      std::string nodeNameStr = getNodeNameInPresenceOfOpt(op);
      nodeName = rewriter.getStringAttr(nodeNameStr).strref();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "Instrumentation_nodeName: " << nodeName << "\n");

    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *typeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto instrumentRef = getOrInsertInstrument(rewriter, parentModule);

    LLVM::GlobalOp globalOpNameStr = krnl::getOrCreateGlobalString(
        opNameStr, loc, rewriter, parentModule, typeConverter);
    Value opNamePtr =
        krnl::getPtrToGlobalString(globalOpNameStr, loc, rewriter);
    // Encode the tag with the length of the op and node name strings
    uint64_t opNameLen = opNameStr.size();
    uint64_t nodeNameLen = nodeName.size();
    uint64_t tagWithLen = instrumentOp.getTag();
    SET_INSTRUMENT_OP_NAME_LEN(tagWithLen, opNameLen);
    SET_INSTRUMENT_NODE_NAME_LEN(tagWithLen, nodeNameLen);
    Value tag = create.llvm.constant(
        IntegerType::get(context, 64), static_cast<int64_t>(tagWithLen));
    LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
        nodeName, loc, rewriter, parentModule, typeConverter);
    Value nodeNamePtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
    // Build the FusedLoc up front so we can stamp it onto the call op
    // at creation. Carries a synthetic DISubprogram named
    // `__omip:<opName>:<nodeName>`; after mlir-translate and LLVM
    // codegen this becomes a DW_TAG_subprogram DIE with explicit
    // PC-range coverage, so external tooling (addr2line --inlines,
    // llvm-dwarfdump --lookup, profile-model.py) can map any sampled
    // PC back to the originating ONNX op without reading .rodata
    // strings or doing per-arch register-dataflow recovery. The
    // synthetic name is chosen at lowering time, so it always reflects
    // the post-conversion op identity (e.g. `zhigh.MatMul` even for
    // ops that were `onnx.MatMul` upstream of the ZHigh conversion
    // pass).
    // The enclosing function is what anchors the synthetic
    // DISubprogram chain. KrnlInstrumentOp is always inside an
    // LLVM::LLVMFuncOp at this stage of lowering.
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(parentFunc && "krnl.runtime_instrument outside an LLVM func");
    Location markerLoc = buildInstrumentMarkerLoc(
        context, parentFunc, parentModule, loc, opNameStr, nodeName);
    // Bypass `create.llvm.call` here because that helper returns a
    // `Value` (null for void calls) and we need the op handle to set
    // the location. Direct CallOp::create gives us both.
    LLVM::CallOp::create(rewriter, markerLoc, /*resultTypes=*/TypeRange{},
        instrumentRef, ValueRange{opNamePtr, tag, nodeNamePtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Create a function declaration for OMInstrumentPoint, the signature is:
  //   `void (ptr, i64, ptr)`
  FlatSymbolRefAttr getOrInsertInstrument(
      PatternRewriter &rewriter, ModuleOp module) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type opaquePtrTy = getI8PointerType(context);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("OMInstrumentPoint"), llvmVoidTy,
        {opaquePtrTy, llvmI64Ty, opaquePtrTy});
  }
};

void populateLoweringKrnlInstrumentOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlInstrumentOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
